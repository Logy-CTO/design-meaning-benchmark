from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["functionality", "aesthetics", "usability", "symbolism", "unclear"]
PROJECT = Path(__file__).resolve().parents[1]
SUBMISSION = PROJECT
DATA = SUBMISSION / "data"
EXPECTED = SUBMISSION / "expected_results"
OUT = PROJECT / "reproduced_results"
SEED = 20260925
N_BOOT = 5000


def cohen_kappa_score(y_true, y_pred, labels):
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    observed = float(np.mean(yt == yp))
    expected = sum(float(np.mean(yt == label)) * float(np.mean(yp == label)) for label in labels)
    return 1.0 if expected >= 1.0 and observed >= 1.0 else (observed - expected) / (1.0 - expected)

def response_distribution(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    counts = rows.pivot_table(index="image_id", columns="design_intent", values="rater",
                              aggfunc="count", fill_value=0).reindex(columns=LABELS, fill_value=0)
    probs = counts.div(counts.sum(axis=1), axis=0)
    winners = counts.eq(counts.max(axis=1), axis=0)
    tie = winners.sum(axis=1) > 1
    return probs, winners, tie


def soft_jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = (p + q) / 2
    def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.zeros_like(a, dtype=float)
        mask = a > 0
        out[mask] = a[mask] * np.log2(a[mask] / b[mask])
        return out.sum(axis=1)
    return (kl(p, m) + kl(q, m)) / 2


def plurality_credit(pred: str, winners: pd.Series) -> float:
    selected = winners.index[winners].tolist()
    return 1.0 / len(selected) if pred in selected else 0.0


def score_predictions(name: str, predictions: dict[str, str], probs: pd.DataFrame,
                      winner_mask: pd.DataFrame, ties: pd.Series,
                      confidence: pd.Series | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    ids = list(probs.index)
    pred = np.asarray([predictions[i] for i in ids], dtype=object)
    credits = np.asarray([plurality_credit(pred[j], winner_mask.loc[i]) for j, i in enumerate(ids)])
    hass = np.asarray([probs.loc[i, pred[j]] for j, i in enumerate(ids)])
    untied = [i for i in ids if not bool(ties.loc[i])]
    kappa = cohen_kappa_score(
        [winner_mask.columns[np.flatnonzero(winner_mask.loc[i].to_numpy())[0]] for i in untied],
        [predictions[i] for i in untied],
        labels=LABELS,
    )
    weighted_hass = np.nan
    if confidence is not None:
        weights = confidence.reindex(ids).to_numpy(dtype=float)
        weighted_hass = float(np.average(hass, weights=weights))
    row = {
        "predictor": name,
        "n_images": len(ids),
        "plurality_accuracy_fractional_ties": float(credits.mean()),
        "plurality_accuracy_unique_only": float(np.mean([predictions[i] == winner_mask.columns[np.flatnonzero(winner_mask.loc[i].to_numpy())[0]] for i in untied])),
        "n_unique_plurality": len(untied),
        "cohen_kappa_unique_plurality_only": float(kappa),
        "HASS": float(hass.mean()),
        "confidence_weighted_HASS": weighted_hass,
    }
    return row, credits, hass


def bootstrap_indices(category: pd.Series, rng: np.random.Generator) -> np.ndarray:
    by_category = [np.flatnonzero(category.to_numpy() == c) for c in sorted(category.unique())]
    return np.concatenate(
        [rng.choice(ix, size=(N_BOOT, len(ix)), replace=True) for ix in by_category],
        axis=1,
    )


def confidence_interval(values: np.ndarray, boot: np.ndarray) -> tuple[float, float]:
    vals = values[boot].mean(axis=1)
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def proper_scores(p: np.ndarray, q: np.ndarray) -> dict[str, float]:
    q_safe = np.clip(q, 1e-12, 1)
    return {
        "brier_sum": float(((p - q) ** 2).sum(axis=1).mean()),
        "soft_cross_entropy_nats": float(-(p * np.log(q_safe)).sum(axis=1).mean()),
        "soft_JSD_bits": float(soft_jsd(p, q).mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    human = pd.read_csv(DATA / "human_responses_30x200.csv")
    probs, winner_mask, ties = response_distribution(human)
    ids = sorted(probs.index)
    categories = human.groupby("image_id").category.first().reindex(ids)
    mean_conf = human.groupby("image_id").confidence.mean().reindex(ids) / 5.0

    # OOF fold-fitted priors: no held-out image labels are used to fit either baseline.
    fold_records: dict[int, dict[str, list[dict]]] = {}
    for fold in range(1, 6):
        fold_records[fold] = {}
        for split in ("train", "val", "test"):
            path = DATA / "adapter_folds" / f"fold_{fold:02d}" / f"{split}.jsonl"
            fold_records[fold][split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    category_priors = {
        "train only": ({}, {}),
        "train+val": ({}, {}),
    }
    global_priors = {
        "train only": ({}, {}),
        "train+val": ({}, {}),
    }
    for fold, splits in fold_records.items():
        for fit_name, fit_rows in (
            ("train only", splits["train"]),
            ("train+val", splits["train"] + splits["val"]),
        ):
            fit_ids = [row["image_id"] for row in fit_rows]
            per_category: dict[str, list[str]] = defaultdict(list)
            for row in fit_rows:
                per_category[row["category"]].append(row["image_id"])
            category_q = {cat: probs.loc[items].mean(axis=0).to_numpy() for cat, items in per_category.items()}
            global_q = probs.loc[fit_ids].mean(axis=0).to_numpy()
            cat_pred, cat_prob = category_priors[fit_name]
            glob_pred, glob_prob = global_priors[fit_name]
            for row in splits["test"]:
                image_id = row["image_id"]
                q = category_q[row["category"]]
                cat_prob[image_id] = q
                cat_pred[image_id] = LABELS[int(np.argmax(q))]
                glob_prob[image_id] = global_q
                glob_pred[image_id] = LABELS[int(np.argmax(global_q))]

    # OOF adapter probabilities from the submitted data supplement.
    adapter = pd.read_csv(DATA / "adapter_oof_predictions_200.csv")
    adapter_pred = dict(zip(adapter.image_id, adapter.pred_design_intent))
    adapter_cols = [f"pred_p_{label}" for label in LABELS]
    adapter_probabilities = {
        row.image_id: row[adapter_cols].to_numpy(dtype=float)
        for _, row in adapter.iterrows()
    }

    # Prompted VLMs, matched to the 200 human-rated image IDs.
    vlm = pd.read_csv(DATA / "vlm_predictions_full1600.csv")
    vlm = vlm[vlm.image_id.isin(ids)]

    prediction_maps = {
        "Category-only soft prior (train only, OOF)": category_priors["train only"][0],
        "Category-only soft prior (train+val, OOF)": category_priors["train+val"][0],
        "Global soft prior (train only, OOF)": global_priors["train only"][0],
        "Global soft prior (train+val, OOF)": global_priors["train+val"][0],
        "Human-aligned LoRA adapter (submitted OOF)": adapter_pred,
    }
    for model, group in vlm.groupby("model", sort=False):
        prediction_maps[model] = dict(zip(group.image_id, group.design_intent))

    # Main scores and category-stratified, paired image bootstrap intervals.
    summary_rows = []
    item_credit: dict[str, np.ndarray] = {}
    item_hass: dict[str, np.ndarray] = {}
    for name, pred_map in prediction_maps.items():
        row, credits, hass = score_predictions(name, pred_map, probs, winner_mask, ties, mean_conf)
        item_credit[name] = credits
        item_hass[name] = hass
        summary_rows.append(row)

    rng = np.random.default_rng(SEED)
    boot = bootstrap_indices(categories, rng)
    pos = {image_id: j for j, image_id in enumerate(ids)}
    summary = pd.DataFrame(summary_rows)
    for metric, vectors in (
        ("plurality_accuracy_fractional_ties", item_credit),
        ("HASS", item_hass),
    ):
        lo, hi = [], []
        for name in summary.predictor:
            a, b = confidence_interval(vectors[name], boot)
            lo.append(a)
            hi.append(b)
        summary[f"{metric}_ci95_low"] = lo
        summary[f"{metric}_ci95_high"] = hi
    summary.to_csv(OUT / "model_and_baseline_metrics.csv", index=False)

    # Paired differences against the category-only baseline.
    contrast_rows = []
    base_name = "Category-only soft prior (train only, OOF)"
    for name in prediction_maps:
        if name == base_name:
            continue
        for metric, vectors in (
            ("fractional_plurality_accuracy", item_credit),
            ("HASS", item_hass),
        ):
            diff = vectors[base_name] - vectors[name]
            point = float(diff.mean())
            ci = np.quantile(diff[boot].mean(axis=1), [.025, .975])
            contrast_rows.append({
                "difference": f"{base_name} - {name}",
                "metric": metric,
                "point_estimate": point,
                "ci95_low": float(ci[0]),
                "ci95_high": float(ci[1]),
            })
    pd.DataFrame(contrast_rows).to_csv(OUT / "paired_category_baseline_differences.csv", index=False)

    # Proper scoring of the available soft adapter output versus a no-image category prior.
    ordered_p = probs.loc[ids].to_numpy()
    q_category = np.asarray([category_priors["train only"][1][i] for i in ids])
    q_adapter = np.asarray([adapter_probabilities[i] for i in ids])
    soft_rows = []
    for name, q in (("Category-only soft prior (OOF)", q_category),
                    ("Human-aligned LoRA adapter (submitted OOF)", q_adapter)):
        scores = proper_scores(ordered_p, q)
        soft_rows.append({"predictor": name, "n_images": len(ids), **scores})
    # Paired bootstrap differences: adapter minus category prior; negative is better.
    m = (ordered_p + q_category) / 2
    c_brier = ((q_category - ordered_p) ** 2).sum(axis=1)
    a_brier = ((q_adapter - ordered_p) ** 2).sum(axis=1)
    c_ce = -(ordered_p * np.log(np.clip(q_category, 1e-12, 1))).sum(axis=1)
    a_ce = -(ordered_p * np.log(np.clip(q_adapter, 1e-12, 1))).sum(axis=1)
    c_jsd = soft_jsd(ordered_p, q_category)
    a_jsd = soft_jsd(ordered_p, q_adapter)
    for label, delta in (
        ("Brier sum (adapter - category)", a_brier - c_brier),
        ("soft cross-entropy nats (adapter - category)", a_ce - c_ce),
        ("soft JSD bits (adapter - category)", a_jsd - c_jsd),
    ):
        ci = np.quantile(delta[boot].mean(axis=1), [.025, .975])
        soft_rows.append({
            "predictor": label,
            "n_images": len(ids),
            "paired_difference": float(delta.mean()),
            "ci95_low": float(ci[0]),
            "ci95_high": float(ci[1]),
        })
    pd.DataFrame(soft_rows).to_csv(OUT / "soft_distribution_scores.csv", index=False)

    # Category label shares and selected-subset design.
    label_rows = []
    for category in sorted(categories.unique()):
        category_ids = categories[categories == category].index
        row = {"category": category, "n_images": len(category_ids)}
        row.update({label: float(probs.loc[category_ids, label].mean()) for label in LABELS})
        row["human_plurality_label"] = LABELS[int(np.argmax([row[label] for label in LABELS]))]
        label_rows.append(row)
    pd.DataFrame(label_rows).to_csv(OUT / "human_distribution_by_category.csv", index=False)
    manifest = pd.read_csv(DATA / "survey_manifest_9b_200.csv")
    sample = manifest.groupby("product").sample_index.agg(["min", "max", "nunique", "count"]).reset_index()
    sample.to_csv(OUT / "human_subset_sample_indices.csv", index=False)

    current_agreement = pd.read_csv(OUT / "human_agreement_bootstrap.csv")
    current_agreement = current_agreement.set_index("measure")["estimate"].to_dict()
    normalized_entropy = -(probs.replace(0, np.nan) * np.log2(probs.replace(0, np.nan))).sum(axis=1) / np.log2(5)
    diagnostics = {
        "human_raters": int(human.rater.nunique()),
        "human_images": int(human.image_id.nunique()),
        "human_response_rows": int(len(human)),
        "human_plurality_ties": int(ties.sum()),
        "mean_normalized_entropy_bits_over_log2_5": float(normalized_entropy.mean()),
        "mean_plurality_support": float(probs.max(axis=1).mean()),
        "agreement_current_standard": current_agreement,
    }
    (OUT / "human_data_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print("Analysis inputs:", DATA)
    print("Output directory:", OUT)
    print("\nModel and baseline metrics")
    print(summary.round(4).to_string(index=False))
    print("\nPaired category-prior differences")
    print(pd.DataFrame(contrast_rows).round(4).to_string(index=False))
    print("\nSoft distribution scores")
    print(pd.DataFrame(soft_rows).round(4).to_string(index=False))
    print("\nDiagnostics")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()



