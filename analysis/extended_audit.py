from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import reanalysis as ra

PROJECT = Path(__file__).resolve().parents[1]
SUB = PROJECT
DATA = SUB / "data"
OUT = PROJECT / "reproduced_results"
N_BOOT = 5000
SEED = 20260925

def weighted_nominal_kappa(y_true, y_pred, weights):
    y = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    w = np.asarray(weights, dtype=float)
    if len(y) == 0 or w.sum() <= 0:
        return np.nan
    w = w / w.sum()
    observed = float(np.sum(w * (y == pred)))
    expected = 0.0
    for label in ra.LABELS:
        expected += float(np.sum(w * (y == label)) * np.sum(w * (pred == label)))
    return 1.0 if expected >= 1.0 and observed >= 1.0 else (observed - expected) / (1.0 - expected)

def load_maps(human, ids, probs):
    category = human.groupby("image_id").category.first().reindex(ids)
    category_pred = {}
    global_pred = {}
    for fold in range(1, 6):
        d = DATA / "adapter_folds" / f"fold_{fold:02d}"
        train = [json.loads(line) for line in (d / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        test = [json.loads(line) for line in (d / "test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        by_category = defaultdict(list)
        for row in train:
            by_category[row["category"]].append(row["image_id"])
        cat_q = {cat: probs.loc[im_ids].mean(axis=0).to_numpy() for cat, im_ids in by_category.items()}
        global_q = probs.loc[[r["image_id"] for r in train]].mean(axis=0).to_numpy()
        for row in test:
            i = row["image_id"]
            category_pred[i] = ra.LABELS[int(np.argmax(cat_q[row["category"]]))]
            global_pred[i] = ra.LABELS[int(np.argmax(global_q))]
    maps = {
        "Category-only soft prior (train only, OOF)": category_pred,
        "Global soft prior (train only, OOF)": global_pred,
    }
    vlm = pd.read_csv(DATA / "vlm_predictions_full1600.csv")
    vlm = vlm[vlm.image_id.isin(ids)]
    for name, group in vlm.groupby("model", sort=False):
        maps[name] = dict(zip(group.image_id, group.design_intent))
    adapter = pd.read_csv(DATA / "adapter_oof_predictions_200.csv")
    maps["Human-aligned LoRA adapter (submitted OOF; exploratory)"] = dict(zip(adapter.image_id, adapter.pred_design_intent))
    no_cat = pd.read_csv(PROJECT / "experiments" / "qwen25vl7b_no_target_category" / "outputs" / "qwen25_vl_7b_no_target_category_200.csv")
    maps["Qwen2.5-VL-7B (category omitted)"] = dict(zip(no_cat.image_id, no_cat.design_intent))
    return maps, category

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    human = pd.read_csv(DATA / "human_responses_30x200.csv")
    probs, winners, ties = ra.response_distribution(human)
    ids = sorted(probs.index)
    categories = human.groupby("image_id").category.first().reindex(ids)
    confidence = human.groupby("image_id").confidence.mean().reindex(ids) / 5.0
    maps, categories = load_maps(human, ids, probs)
    tie_arr = ties.reindex(ids).to_numpy(dtype=bool)
    conf_arr = confidence.reindex(ids).to_numpy(dtype=float)
    unique_idx = np.flatnonzero(~tie_arr)
    true = np.asarray([
        ra.LABELS[int(np.flatnonzero(winners.loc[i].to_numpy(dtype=bool))[0])]
        for i in ids
    ], dtype=object)
    rng = np.random.default_rng(SEED)
    boot = ra.bootstrap_indices(categories, rng)

    # Case-weighted nominal Cohen kappa: image confidence is a case weight;
    # this is not the standard quadratic/linear label-distance weighted kappa.
    kappa_rows = []
    for name, pred_map in maps.items():
        pred = np.asarray([pred_map[i] for i in ids], dtype=object)
        y_u, p_u, w_u = true[unique_idx], pred[unique_idx], conf_arr[unique_idx]
        point = weighted_nominal_kappa(y_u, p_u, w_u)
        boot_values = []
        for sample in boot:
            valid = sample[~tie_arr[sample]]
            boot_values.append(weighted_nominal_kappa(true[valid], pred[valid], conf_arr[valid]))
        kappa_rows.append({
            "predictor": name, "n_unique_plurality_images": int(len(unique_idx)),
            "case_weight": "mean human confidence / 5",
            "confidence_weighted_nominal_kappa": point,
            "ci95_low": float(np.quantile(boot_values, .025)),
            "ci95_high": float(np.quantile(boot_values, .975)),
        })
    pd.DataFrame(kappa_rows).to_csv(OUT / "confidence_weighted_kappa.csv", index=False)

    # Descriptive response-share differences for the four originally designed contrasts.
    shares = pd.read_csv(OUT / "human_distribution_by_category.csv").set_index("category")
    contrast_pairs = [
        ("Historical vs contemporary automobiles", "early_automobile", "sports_car"),
        ("Manual vs powered cleaning", "broom", "vacuum_cleaner"),
        ("Lighting vs heating appliances", "desk_lamp", "electric_kettle"),
        ("Seated vs wearable products", "lounge_chair", "sneakers"),
    ]
    contrast_rows = []
    for label, first, second in contrast_pairs:
        a, b = shares.loc[first], shares.loc[second]
        row = {"contrast": label, "first_category": first, "second_category": second,
               "n_first": int(a["n_images"]), "n_second": int(b["n_images"])}
        for cls in ra.LABELS:
            row[f"{cls}_first_minus_second_pp"] = 100.0 * (float(a[cls]) - float(b[cls]))
        contrast_rows.append(row)
    pd.DataFrame(contrast_rows).to_csv(OUT / "four_category_contrasts.csv", index=False)

    # Paired, within-category performance against the train-only category prior.
    baseline = maps["Category-only soft prior (train only, OOF)"]
    base_credit = {i: ra.plurality_credit(baseline[i], winners.loc[i]) for i in ids}
    category_rows = []
    top_models = ["GPT-5.5", "InternVL3 (38B, 2025)", "Qwen2.5-VL (7B, 2025)",
                  "Qwen2.5-VL-7B (category omitted)"]
    for name in top_models:
        pred = maps[name]
        for cat in sorted(categories.unique()):
            cat_ids = [i for i in ids if categories.loc[i] == cat]
            model_credit = np.asarray([ra.plurality_credit(pred[i], winners.loc[i]) for i in cat_ids])
            base_c = np.asarray([base_credit[i] for i in cat_ids])
            model_support = np.asarray([probs.loc[i, pred[i]] for i in cat_ids])
            base_support = np.asarray([probs.loc[i, baseline[i]] for i in cat_ids])
            delta_acc = base_c - model_credit
            delta_hass = base_support - model_support
            rng_cat = np.random.default_rng(SEED + sum(map(ord, cat + name)))
            samples = rng_cat.integers(0, len(cat_ids), size=(N_BOOT, len(cat_ids)))
            ci_acc = np.quantile(delta_acc[samples].mean(axis=1), [.025, .975])
            ci_hass = np.quantile(delta_hass[samples].mean(axis=1), [.025, .975])
            category_rows.append({
                "model": name, "category": cat, "n_images": len(cat_ids),
                "model_accuracy": float(model_credit.mean()),
                "category_prior_accuracy": float(base_c.mean()),
                "category_prior_minus_model_accuracy": float(delta_acc.mean()),
                "accuracy_ci95_low": float(ci_acc[0]), "accuracy_ci95_high": float(ci_acc[1]),
                "model_HASS": float(model_support.mean()),
                "category_prior_HASS": float(base_support.mean()),
                "category_prior_minus_model_HASS": float(delta_hass.mean()),
                "HASS_ci95_low": float(ci_hass[0]), "HASS_ci95_high": float(ci_hass[1]),
            })
    pd.DataFrame(category_rows).to_csv(OUT / "within_category_top_models.csv", index=False)

    print("Case-weighted Cohen kappa:")
    print(pd.DataFrame(kappa_rows).round(4).to_string(index=False))
    print("\nFour descriptive response-share contrasts (percentage points):")
    print(pd.DataFrame(contrast_rows).round(1).to_string(index=False))
    print("\nWithin-category comparison:")
    print(pd.DataFrame(category_rows).round(3).to_string(index=False))

if __name__ == "__main__":
    main()