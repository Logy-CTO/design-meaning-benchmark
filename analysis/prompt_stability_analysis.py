from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import reanalysis as ra

PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENT = PROJECT / "experiments" / "qwen_prompt_stability"
DATA = ra.DATA
OUT = ra.OUT
LABELS = ra.LABELS
N_BOOT = 5000
SEED = 20260925
CONDITIONS = ["canonical_greedy", "label_first_greedy"] + [
    f"sample_seed_{seed}" for seed in [11, 23, 37, 47, 59]
]


def parse_label(raw: str) -> str | None:
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    value = None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            value = obj.get("design_intent")
    except json.JSONDecodeError:
        match = re.search(r'"design_intent"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
        if match:
            value = match.group(1)
    if value is None:
        return None
    label = str(value).strip().lower()
    return label if label in LABELS else None


def ci_mean(values: np.ndarray, categories: pd.Series, rng: np.random.Generator) -> tuple[float, float]:
    boot = ra.bootstrap_indices(categories, rng)
    return ra.confidence_interval(np.asarray(values, dtype=float), boot)


def weighted_mean_ci(values: np.ndarray, weights: np.ndarray, categories: pd.Series,
                     rng: np.random.Generator) -> tuple[float, float]:
    boot = ra.bootstrap_indices(categories, rng)
    draws = np.asarray([np.average(values[ix], weights=weights[ix]) for ix in boot])
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def category_soft_prior(probs: pd.DataFrame, ids: list[str]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    folds_dir = DATA / "adapter_folds"
    for fold in range(1, 6):
        fold_dir = folds_dir / f"fold_{fold:02d}"
        train = [json.loads(line) for line in (fold_dir / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        test = [json.loads(line) for line in (fold_dir / "test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        by_category: dict[str, list[str]] = {}
        for record in train:
            by_category.setdefault(record["category"], []).append(record["image_id"])
        q_by_category = {
            category: probs.loc[image_ids, LABELS].mean(axis=0).to_numpy(dtype=float)
            for category, image_ids in by_category.items()
        }
        for record in test:
            result[record["image_id"]] = q_by_category[record["category"]]
    missing = set(ids) - set(result)
    if missing:
        raise RuntimeError(f"Category prior missing {len(missing)} images")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((EXPERIMENT / "outputs" / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise RuntimeError(f"Run status is {metadata.get('status')!r}, not completed")
    if metadata.get("model_revision") != "cc594898137f460bfe9f0759e9844b3ce807cfb5":
        raise RuntimeError("Unexpected model revision")

    manifest = pd.read_csv(EXPERIMENT / "data" / "master_image_manifest.csv")
    expected_hashes = pd.read_csv(EXPERIMENT / "data" / "image_sha256.csv").set_index("image_file")["sha256"].to_dict()
    image_root = PROJECT / "images"
    local_hashes: dict[str, str] = {}
    for row in manifest.to_dict("records"):
        path = image_root / row["image_file"]
        local_hashes[row["image_id"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        if local_hashes[row["image_id"]] != expected_hashes[row["image_file"]]:
            raise RuntimeError(f"Image hash differs from submitted manifest for {row['image_id']}")
        if local_hashes[row["image_id"]] != metadata["image_sha256"].get(row["image_id"]):
            raise RuntimeError(f"Run-time image hash differs for {row['image_id']}")
    prompt_hashes = {}
    for filename in ["canonical_fields_first.txt", "label_first.txt"]:
        prompt_hashes[filename] = hashlib.sha256((EXPERIMENT / "prompts" / filename).read_bytes()).hexdigest()
    if prompt_hashes != metadata["prompt_sha256"]:
        raise RuntimeError("Prompt hashes do not match run metadata")

    records = [json.loads(line) for line in (EXPERIMENT / "outputs" / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_keys = {(condition, image_id) for condition in CONDITIONS for image_id in manifest.image_id}
    observed_keys = [(record["condition"], record["image_id"]) for record in records]
    if len(observed_keys) != len(set(observed_keys)):
        raise RuntimeError("Duplicate condition/image outputs found")
    if set(observed_keys) != expected_keys:
        raise RuntimeError(f"Output key set differs: expected {len(expected_keys)}, observed {len(observed_keys)}")
    if any(record.get("model_revision") != metadata["model_revision"] for record in records):
        raise RuntimeError("A prediction row has a different model revision")

    parsed = []
    label_maps: dict[str, dict[str, str]] = {condition: {} for condition in CONDITIONS}
    invalid_rows = []
    for record in records:
        label = parse_label(record["raw_response"])
        parsed.append({"condition": record["condition"], "image_id": record["image_id"], "design_intent": label})
        if label is None:
            invalid_rows.append({"condition": record["condition"], "image_id": record["image_id"], "raw_response": record["raw_response"]})
        else:
            label_maps[record["condition"]][record["image_id"]] = label
    pd.DataFrame(parsed).to_csv(EXPERIMENT / "outputs" / "parsed_predictions.csv", index=False)
    (EXPERIMENT / "outputs" / "invalid_predictions.json").write_text(json.dumps(invalid_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    human = pd.read_csv(DATA / "human_responses_30x200.csv")
    probs, winners, ties = ra.response_distribution(human)
    ids = sorted(probs.index)
    categories = human.groupby("image_id").category.first().reindex(ids)
    confidence = human.groupby("image_id").confidence.mean().reindex(ids) / 5.0
    prior_q_map = category_soft_prior(probs, ids)
    rng = np.random.default_rng(SEED)

    # Each condition is scored on its valid rows. Any invalid JSON remains explicitly invalid;
    # it is never recoded as the semantic label "unclear".
    metric_rows = []
    for condition in CONDITIONS:
        valid_ids = [image_id for image_id in ids if image_id in label_maps[condition]]
        pred_map = {image_id: label_maps[condition][image_id] for image_id in valid_ids}
        p_sub = probs.loc[valid_ids]
        w_sub = winners.loc[valid_ids]
        t_sub = ties.loc[valid_ids]
        c_sub = confidence.loc[valid_ids]
        cat_sub = categories.loc[valid_ids]
        row, credits, hass = ra.score_predictions(condition, pred_map, p_sub, w_sub, t_sub, c_sub)
        acc_ci = ci_mean(credits, cat_sub, rng)
        hass_ci = ci_mean(hass, cat_sub, rng)
        weighted_ci = weighted_mean_ci(hass, c_sub.to_numpy(dtype=float), cat_sub, rng)
        metric_rows.append({
            **row, "n_valid": len(valid_ids), "n_invalid": 200 - len(valid_ids),
            "accuracy_ci95_low": acc_ci[0], "accuracy_ci95_high": acc_ci[1],
            "HASS_ci95_low": hass_ci[0], "HASS_ci95_high": hass_ci[1],
            "confidence_weighted_HASS_ci95_low": weighted_ci[0],
            "confidence_weighted_HASS_ci95_high": weighted_ci[1],
        })
    pd.DataFrame(metric_rows).to_csv(EXPERIMENT / "outputs" / "condition_metrics.csv", index=False)

    # Paired prompt-order and per-seed comparisons against canonical greedy.
    pair_rows = []
    transition_rows = []
    canonical = label_maps["canonical_greedy"]
    for condition in CONDITIONS[1:]:
        common = sorted(set(canonical) & set(label_maps[condition]))
        cat_common = categories.loc[common]
        boot = ra.bootstrap_indices(cat_common, rng)
        delta_acc, delta_hass, changed = [], [], 0
        for image_id in common:
            pred_a, pred_b = canonical[image_id], label_maps[condition][image_id]
            ca = ra.plurality_credit(pred_a, winners.loc[image_id])
            cb = ra.plurality_credit(pred_b, winners.loc[image_id])
            ha = float(probs.loc[image_id, pred_a])
            hb = float(probs.loc[image_id, pred_b])
            delta_acc.append(cb - ca)
            delta_hass.append(hb - ha)
            changed += int(pred_a != pred_b)
            transition_rows.append({"comparison": f"{condition}_minus_canonical", "from_label": pred_a,
                                    "to_label": pred_b, "count": 1})
        delta_acc = np.asarray(delta_acc, dtype=float)
        delta_hass = np.asarray(delta_hass, dtype=float)
        pair_rows.append({
            "comparison": f"{condition} - canonical_greedy", "n_paired": len(common),
            "n_labels_changed": changed, "fraction_labels_changed": changed / len(common) if common else np.nan,
            "delta_accuracy": float(delta_acc.mean()) if len(common) else np.nan,
            "delta_accuracy_ci95_low": float(np.quantile(delta_acc[boot].mean(axis=1), .025)) if len(common) else np.nan,
            "delta_accuracy_ci95_high": float(np.quantile(delta_acc[boot].mean(axis=1), .975)) if len(common) else np.nan,
            "delta_HASS": float(delta_hass.mean()) if len(common) else np.nan,
            "delta_HASS_ci95_low": float(np.quantile(delta_hass[boot].mean(axis=1), .025)) if len(common) else np.nan,
            "delta_HASS_ci95_high": float(np.quantile(delta_hass[boot].mean(axis=1), .975)) if len(common) else np.nan,
        })

    archived = pd.read_csv(DATA / "vlm_predictions_full1600.csv")
    archived = archived[(archived.model == "Qwen2.5-VL (7B, 2025)") & archived.image_id.isin(ids)]
    archived_map = dict(zip(archived.image_id, archived.design_intent))
    common = sorted(set(canonical) & set(archived_map))
    cat_common = categories.loc[common]
    boot = ra.bootstrap_indices(cat_common, rng)
    delta_acc, delta_hass, changed = [], [], 0
    for image_id in common:
        current, prior = canonical[image_id], archived_map[image_id]
        delta_acc.append(ra.plurality_credit(current, winners.loc[image_id]) - ra.plurality_credit(prior, winners.loc[image_id]))
        delta_hass.append(float(probs.loc[image_id, current]) - float(probs.loc[image_id, prior]))
        changed += int(current != prior)
    delta_acc, delta_hass = np.asarray(delta_acc, dtype=float), np.asarray(delta_hass, dtype=float)
    pair_rows.append({
        "comparison": "canonical_greedy - archived_category_supplied_Qwen7B", "n_paired": len(common),
        "n_labels_changed": changed, "fraction_labels_changed": changed / len(common) if common else np.nan,
        "delta_accuracy": float(delta_acc.mean()) if len(common) else np.nan,
        "delta_accuracy_ci95_low": float(np.quantile(delta_acc[boot].mean(axis=1), .025)) if len(common) else np.nan,
        "delta_accuracy_ci95_high": float(np.quantile(delta_acc[boot].mean(axis=1), .975)) if len(common) else np.nan,
        "delta_HASS": float(delta_hass.mean()) if len(common) else np.nan,
        "delta_HASS_ci95_low": float(np.quantile(delta_hass[boot].mean(axis=1), .025)) if len(common) else np.nan,
        "delta_HASS_ci95_high": float(np.quantile(delta_hass[boot].mean(axis=1), .975)) if len(common) else np.nan,
    })
    pd.DataFrame(pair_rows).to_csv(EXPERIMENT / "outputs" / "paired_condition_differences.csv", index=False)
    trans = pd.DataFrame(transition_rows)
    if not trans.empty:
        trans.groupby(["comparison", "from_label", "to_label"], as_index=False)["count"].sum().to_csv(
            EXPERIMENT / "outputs" / "label_transitions.csv", index=False)

    # Five stochastic draws estimate an empirical per-image label distribution.
    sample_conditions = CONDITIONS[2:]
    sample_valid_ids = sorted(set.intersection(*(set(label_maps[c]) for c in sample_conditions)))
    sample_category = categories.loc[sample_valid_ids]
    sample_matrix = np.asarray([[label_maps[c][image_id] for c in sample_conditions] for image_id in sample_valid_ids], dtype=object)
    stability_rows = []
    for ix, image_id in enumerate(sample_valid_ids):
        counts = np.asarray([np.sum(sample_matrix[ix] == label) for label in LABELS], dtype=float)
        freq = counts / len(sample_conditions)
        positive = freq[freq > 0]
        entropy = -float(np.sum(positive * np.log2(positive))) / np.log2(len(LABELS))
        stability_rows.append({
            "image_id": image_id, "category": categories.loc[image_id],
            "modal_fraction": float(counts.max() / len(sample_conditions)),
            "all_five_same": bool(counts.max() == len(sample_conditions)),
            "distinct_labels": int(np.sum(counts > 0)), "normalized_label_entropy": entropy,
            "canonical_label": canonical.get(image_id),
            "greedy_matches_stochastic_mode": bool(canonical.get(image_id) in LABELS and counts[LABELS.index(canonical[image_id])] == counts.max()),
            **{f"count_{label}": int(counts[j]) for j, label in enumerate(LABELS)},
        })
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(EXPERIMENT / "outputs" / "per_image_stability.csv", index=False)

    # Pairwise agreement over the ten stochastic seed pairs; bootstrap by image within category.
    seed_pair_agreement = []
    for left in range(len(sample_conditions)):
        for right in range(left + 1, len(sample_conditions)):
            seed_pair_agreement.append(sample_matrix[:, left] == sample_matrix[:, right])
    pair_agreement_by_image = np.mean(np.asarray(seed_pair_agreement, dtype=float), axis=0)
    boot = ra.bootstrap_indices(sample_category, rng)
    paired_agreement = {
        "pairwise_seed_agreement": float(pair_agreement_by_image.mean()),
        "pairwise_seed_agreement_ci95_low": float(np.quantile(pair_agreement_by_image[boot].mean(axis=1), .025)),
        "pairwise_seed_agreement_ci95_high": float(np.quantile(pair_agreement_by_image[boot].mean(axis=1), .975)),
        "mean_modal_fraction": float(stability.modal_fraction.mean()),
        "mean_modal_fraction_ci95_low": float(np.quantile(stability.modal_fraction.to_numpy()[boot].mean(axis=1), .025)),
        "mean_modal_fraction_ci95_high": float(np.quantile(stability.modal_fraction.to_numpy()[boot].mean(axis=1), .975)),
        "all_five_same_fraction": float(stability.all_five_same.mean()),
        "all_five_same_fraction_ci95_low": float(np.quantile(stability.all_five_same.to_numpy(dtype=float)[boot].mean(axis=1), .025)),
        "all_five_same_fraction_ci95_high": float(np.quantile(stability.all_five_same.to_numpy(dtype=float)[boot].mean(axis=1), .975)),
        "mean_normalized_entropy": float(stability.normalized_label_entropy.mean()),
        "mean_normalized_entropy_ci95_low": float(np.quantile(stability.normalized_label_entropy.to_numpy(dtype=float)[boot].mean(axis=1), .025)),
        "mean_normalized_entropy_ci95_high": float(np.quantile(stability.normalized_label_entropy.to_numpy(dtype=float)[boot].mean(axis=1), .975)),
        "canonical_matches_stochastic_mode_fraction": float(stability.greedy_matches_stochastic_mode.mean()),
        "canonical_matches_stochastic_mode_fraction_ci95_low": float(np.quantile(stability.greedy_matches_stochastic_mode.to_numpy(dtype=float)[boot].mean(axis=1), .025)),
        "canonical_matches_stochastic_mode_fraction_ci95_high": float(np.quantile(stability.greedy_matches_stochastic_mode.to_numpy(dtype=float)[boot].mean(axis=1), .975)),
        "n_images": len(sample_valid_ids), "n_stochastic_draws_per_image": len(sample_conditions),
    }
    (EXPERIMENT / "outputs" / "stability_summary.json").write_text(json.dumps(paired_agreement, indent=2), encoding="utf-8")

    # Jeffreys-smoothed Monte Carlo label frequencies vs. human soft labels and the fold-fitted prior.
    alpha = 0.5
    sample_q = np.asarray([
        (np.asarray([np.sum(sample_matrix[i] == label) for label in LABELS], dtype=float) + alpha)
        / (len(sample_conditions) + alpha * len(LABELS))
        for i in range(len(sample_valid_ids))
    ])
    p_human = probs.loc[sample_valid_ids, LABELS].to_numpy(dtype=float)
    q_category = np.asarray([prior_q_map[image_id] for image_id in sample_valid_ids], dtype=float)
    soft_sample = ra.proper_scores(p_human, sample_q)
    soft_category = ra.proper_scores(p_human, q_category)
    per_image_qwen = {
        metric: np.asarray([ra.proper_scores(p_human[i:i+1], sample_q[i:i+1])[metric]
                            for i in range(len(sample_valid_ids))])
        for metric in ["brier_sum", "soft_cross_entropy_nats", "soft_JSD_bits"]
    }
    per_image_category = {
        metric: np.asarray([ra.proper_scores(p_human[i:i+1], q_category[i:i+1])[metric]
                            for i in range(len(sample_valid_ids))])
        for metric in ["brier_sum", "soft_cross_entropy_nats", "soft_JSD_bits"]
    }
    soft_rows = []
    for metric in ["brier_sum", "soft_cross_entropy_nats", "soft_JSD_bits"]:
        delta = per_image_qwen[metric] - per_image_category[metric]
        metric_boot = ra.bootstrap_indices(sample_category, rng)
        soft_rows.append({
            "metric": metric, "Qwen_five_draw_smoothed": soft_sample[metric],
            "category_only_soft_prior": soft_category[metric],
            "delta_Qwen_minus_category": float(delta.mean()),
            "delta_ci95_low": float(np.quantile(delta[metric_boot].mean(axis=1), .025)),
            "delta_ci95_high": float(np.quantile(delta[metric_boot].mean(axis=1), .975)),
            "smoothing_alpha_per_class": alpha, "n_draws_per_image": len(sample_conditions),
        })
    pd.DataFrame(soft_rows).to_csv(EXPERIMENT / "outputs" / "soft_distribution_scores.csv", index=False)

    summary = {
        "model_revision": metadata["model_revision"], "precision": metadata["precision"],
        "n_expected_rows": len(expected_keys), "n_observed_rows": len(records),
        "n_invalid_outputs": len(invalid_rows), "image_hashes_verified": len(local_hashes),
        "prompt_hashes_verified": len(prompt_hashes), "conditions": CONDITIONS,
        "stochastic_seed_rule": metadata["sampling_seed_rule"],
        "scope": "one model, one fixed 200-image subset, one prompt-order contrast, five seeded stochastic draws",
    }
    (EXPERIMENT / "outputs" / "analysis_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Condition metrics:\n", pd.DataFrame(metric_rows).round(4).to_string(index=False))
    print("\nPaired comparisons:\n", pd.DataFrame(pair_rows).round(4).to_string(index=False))
    print("\nStability summary:\n", json.dumps(paired_agreement, indent=2))
    print("\nSoft scores:\n", pd.DataFrame(soft_rows).round(4).to_string(index=False))
    print("\nValidation:\n", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
