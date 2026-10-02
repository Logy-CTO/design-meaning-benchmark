from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


LABELS = ["functionality", "aesthetics", "usability", "symbolism", "unclear"]
ARMS = ["soft_target", "majority_target", "soft_ce_only", "soft_entropy", "category_prior"]
METRICS = ["tie_aware_accuracy", "hass", "brier_sum", "cross_entropy_nats", "jsd_bits"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsd_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    p_term = np.where(p > 0, p * np.log2(np.maximum(p, 1e-15) / np.maximum(m, 1e-15)), 0.0)
    q_term = np.where(q > 0, q * np.log2(np.maximum(q, 1e-15) / np.maximum(m, 1e-15)), 0.0)
    return 0.5 * p_term.sum(axis=1) + 0.5 * q_term.sum(axis=1)


def score(p: np.ndarray, q: np.ndarray) -> dict[str, float]:
    predicted = q.argmax(axis=1)
    maxima = p == p.max(axis=1, keepdims=True)
    tie_count = maxima.sum(axis=1)
    correct_ties = maxima[np.arange(len(p)), predicted]
    return {
        "tie_aware_accuracy": float(np.mean(correct_ties / tie_count)),
        "hass": float(np.mean(p[np.arange(len(p)), predicted])),
        "brier_sum": float(np.mean(np.square(q - p).sum(axis=1))),
        "cross_entropy_nats": float(np.mean(-(p * np.log(np.maximum(q, 1e-12))).sum(axis=1))),
        "jsd_bits": float(np.mean(jsd_bits(p, q))),
    }


def stratified_indices(categories: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = []
    for category in sorted(set(categories.tolist())):
        idx = np.flatnonzero(categories == category)
        result.extend(rng.choice(idx, size=len(idx), replace=True).tolist())
    return np.asarray(result, dtype=int)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_root", required=True)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=8419)
    args = parser.parse_args()
    fold_root, run_root, image_root, out = map(Path, (args.fold_root, args.run_root, args.image_root, args.output_dir))
    rng = np.random.default_rng(args.seed)

    fold_records = []
    test_counts: dict[str, int] = {}
    prediction_sets: dict[str, dict[str, dict]] = {arm: {} for arm in ARMS}
    image_hashes = {}
    for fold in range(1, 6):
        fold_dir = fold_root / f"fold_{fold:02d}"
        split_rows = {split: read_jsonl(fold_dir / f"{split}.jsonl") for split in ("train", "val", "test")}
        split_ids = {name: {row["image_id"] for row in rows} for name, rows in split_rows.items()}
        if any(split_ids[a] & split_ids[b] for i, a in enumerate(("train", "val", "test"))
               for b in ("train", "val", "test")[i + 1:]):
            raise ValueError(f"Overlapping train/validation/test IDs in fold {fold}")
        for split, rows in split_rows.items():
            fold_records.append({"fold": fold, "split": split, "n_images": len(rows), "sha256": sha256(fold_dir / f"{split}.jsonl")})
        train_rows, test_rows = split_rows["train"], split_rows["test"]
        by_category: dict[str, list[list[float]]] = {}
        for row in train_rows:
            by_category.setdefault(row["category"], []).append([float(row[f"human_p_{label}"]) for label in LABELS])
        priors = {cat: np.mean(np.asarray(values, dtype=float), axis=0) for cat, values in by_category.items()}

        for arm in ARMS[:-1]:
            pred_path = (Path(__file__).resolve().parents[1] / "outputs/soft_entropy" / f"fold_{fold:02d}" / "predictions.jsonl") if arm == "soft_entropy" else run_root / f"fold_{fold:02d}" / arm / "predictions.jsonl"
            preds = read_jsonl(pred_path)
            if len(preds) != len(test_rows):
                raise ValueError(f"Expected {len(test_rows)} predictions at {pred_path}; got {len(preds)}")
            if len({row["image_id"] for row in preds}) != len(preds):
                raise ValueError(f"Duplicate prediction IDs at {pred_path}")
            if {row["image_id"] for row in preds} != split_ids["test"]:
                raise ValueError(f"Test prediction IDs do not match fold {fold}")
            summary = json.loads((pred_path.parent / "run_summary.json").read_text(encoding="utf-8"))
            if summary["model_revision"] != "cc594898137f460bfe9f0759e9844b3ce807cfb5":
                raise ValueError(f"Unexpected model revision at {pred_path.parent}")
            reference_args = json.loads((run_root / f"fold_{fold:02d}" / "soft_ce_only" / "training_args.json").read_text())
            for setting in ["seed", "lora_r", "lora_alpha", "lora_dropout", "lr", "weight_decay", "epochs", "patience", "min_delta", "batch_size", "grad_accum", "precision", "model_revision"]:
                if summary[setting] != reference_args[setting]:
                    raise ValueError(f"Mismatched {setting} for {arm} fold {fold}")
            if arm == "soft_entropy" and (summary["hard_weight"] != 0 or summary["entropy_weight"] != 0.1):
                raise ValueError("Wrong isolated entropy objective")
            truth = {r["image_id"]: np.asarray([r[f"human_p_{label}"] for label in LABELS]) for r in test_rows}
            for row in preds:
                p = np.asarray(row["human_probs"], dtype=float)
                q = np.asarray(row["predicted_probs"], dtype=float)
                if p.shape != (5,) or q.shape != (5,) or not np.isfinite(p).all() or not np.isfinite(q).all():
                    raise ValueError(f"Invalid probability vector for {row['image_id']}")
                if not np.isclose(p.sum(), 1.0, atol=1e-5) or not np.isclose(q.sum(), 1.0, atol=1e-5):
                    raise ValueError(f"Probability vector does not sum to one for {row['image_id']}")
                if not np.allclose(p, truth[row["image_id"]], atol=1e-6):
                    raise ValueError("Prediction human distribution differs from fold input")
                prediction_sets[arm][row["image_id"]] = row

        for row in test_rows:
            image_path = image_root / row["image_file"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            image_hashes[row["image_id"]] = sha256(image_path)
            p = [float(row[f"human_p_{label}"]) for label in LABELS]
            if not np.isclose(sum(p), 1.0, atol=1e-5):
                raise ValueError(f"Invalid human response distribution for {row['image_id']}")
            test_counts[row["image_id"]] = test_counts.get(row["image_id"], 0) + 1
            prediction_sets["category_prior"][row["image_id"]] = {
                "image_id": row["image_id"], "category": row["category"], "fold": fold,
                "human_probs": p, "predicted_probs": priors[row["category"]],
            }

    if set(test_counts.values()) != {1} or len(test_counts) != 200:
        raise ValueError(f"OOF test coverage is not exactly once for 200 images: n={len(test_counts)}")
    ids = sorted(test_counts)
    p_by_id = {image_id: np.asarray(prediction_sets["category_prior"][image_id]["human_probs"], dtype=float)
               for image_id in ids}
    category_by_id = {image_id: prediction_sets["category_prior"][image_id]["category"] for image_id in ids}
    arrays = {
        arm: np.asarray([prediction_sets[arm][image_id]["predicted_probs"] for image_id in ids], dtype=float)
        for arm in ARMS
    }
    p = np.asarray([p_by_id[image_id] for image_id in ids], dtype=float)
    categories = np.asarray([category_by_id[image_id] for image_id in ids])

    point = {arm: score(p, arrays[arm]) for arm in ARMS}
    boot_values = {arm: {metric: [] for metric in METRICS} for arm in ARMS}
    comparisons = (
        ("soft_entropy", "soft_ce_only"),
        ("soft_entropy", "majority_target"),
        ("soft_target", "soft_entropy"),
        ("soft_entropy", "category_prior"),
        ("soft_target", "majority_target"),
        ("soft_target", "soft_ce_only"),
        ("soft_ce_only", "majority_target"),
        ("soft_target", "category_prior"),
        ("soft_ce_only", "category_prior"),
        ("majority_target", "category_prior"),
    )
    diff_values = {
        (left, right, metric): []
        for left, right in comparisons
        for metric in METRICS
    }
    for _ in range(args.bootstrap):
        idx = stratified_indices(categories, rng)
        scores = {arm: score(p[idx], arrays[arm][idx]) for arm in ARMS}
        for arm in ARMS:
            for metric in METRICS:
                boot_values[arm][metric].append(scores[arm][metric])
        for left, right, metric in diff_values:
            diff_values[(left, right, metric)].append(scores[left][metric] - scores[right][metric])

    metric_rows = []
    for arm in ARMS:
        row = {"predictor": arm, "n_images": len(ids)}
        for metric in METRICS:
            samples = np.asarray(boot_values[arm][metric], dtype=float)
            row[metric] = point[arm][metric]
            row[f"{metric}_ci95_low"] = float(np.quantile(samples, 0.025))
            row[f"{metric}_ci95_high"] = float(np.quantile(samples, 0.975))
        metric_rows.append(row)
    difference_rows = []
    for left, right, metric in diff_values:
        samples = np.asarray(diff_values[(left, right, metric)], dtype=float)
        difference_rows.append({
            "comparison": f"{left} - {right}", "metric": metric,
            "difference": point[left][metric] - point[right][metric],
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "bootstrap_replicates": args.bootstrap,
        })
    per_category_rows = []
    for category in sorted(set(categories.tolist())):
        idx = np.flatnonzero(categories == category)
        for arm in ARMS:
            values = score(p[idx], arrays[arm][idx])
            per_category_rows.append({"category": category, "predictor": arm, "n_images": len(idx), **values})

    prediction_rows = []
    for i, image_id in enumerate(ids):
        p_vec = p[i]
        for arm in ARMS:
            q_vec = arrays[arm][i]
            prediction_rows.append({
                "image_id": image_id,
                "category": categories[i],
                "fold": prediction_sets["category_prior"][image_id]["fold"],
                "predictor": arm,
                "human_probs": json.dumps(p_vec.tolist()),
                "predicted_probs": json.dumps(q_vec.tolist()),
                "predicted_label": LABELS[int(np.argmax(q_vec))],
                "image_sha256": image_hashes[image_id],
            })

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "condition_metrics.csv", metric_rows)
    write_csv(out / "paired_differences.csv", difference_rows)
    write_csv(out / "per_category_metrics.csv", per_category_rows)
    write_csv(out / "oof_predictions.csv", prediction_rows)
    write_csv(out / "fold_split_manifest.csv", fold_records)
    (out / "image_hashes.json").write_text(json.dumps(image_hashes, indent=2), encoding="utf-8")
    validation = {
        "n_expected_images": 200,
        "n_oof_test_predictions_per_arm": {arm: len(prediction_sets[arm]) for arm in ARMS[:-1]},
        "n_valid_oof_ids": len(ids),
        "each_image_tested_once": set(test_counts.values()) == {1},
        "fold_split_ids_disjoint": True,
        "train_test_data_hashes": fold_records,
        "image_hashes_count": len(image_hashes),
        "condition_metrics": point,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "scope": "fixed 200-image subset; 5-fold out-of-fold; same Qwen checkpoint, folds, image inputs, fixed LoRA settings, and fold seeds; soft-loss bundle vs pure soft cross-entropy vs soft cross-entropy plus 0.1 entropy MSE (no hard-label term) vs plurality-label cross-entropy; validation-only objective-specific early stopping; inherited hyperparameter selection remains selection-aware",
    }
    (out / "analysis_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
