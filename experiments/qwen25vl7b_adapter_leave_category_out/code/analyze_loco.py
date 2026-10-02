from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


LABELS = ["functionality", "aesthetics", "usability", "symbolism", "unclear"]
ARMS = ["soft_target", "majority_target", "global_soft_prior"]
METRICS = ["tie_aware_accuracy", "hass", "brier_sum", "cross_entropy_nats", "jsd_bits"]
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_root", required=True, type=Path)
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--image_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=8419)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    split_hashes = []
    prediction_sets: dict[str, dict[str, dict]] = {arm: {} for arm in ARMS}
    test_counts: dict[str, int] = {}
    category_by_id: dict[str, str] = {}
    human_by_id: dict[str, list[float]] = {}
    image_hashes: dict[str, str] = {}
    held_out_categories: dict[str, str] = {}
    run_metadata: dict[int, dict[str, dict]] = {}

    for fold in range(1, 9):
        fold_dir = args.fold_root / f"fold_{fold:02d}"
        splits = {name: read_jsonl(fold_dir / f"{name}.jsonl") for name in ("train", "val", "test")}
        split_ids = {name: {row["image_id"] for row in rows} for name, rows in splits.items()}
        if any(split_ids[left] & split_ids[right] for i, left in enumerate(("train", "val", "test"))
               for right in ("train", "val", "test")[i + 1:]):
            raise ValueError(f"Train/validation/test IDs overlap in fold {fold}")
        if tuple(len(splits[name]) for name in ("train", "val", "test")) != (140, 35, 25):
            raise ValueError(f"Unexpected split sizes in fold {fold}")

        held_out = (fold_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
        held_category_set = {row["category"] for row in (json.loads(line) for line in held_out)}
        if len(held_category_set) != 1:
            raise ValueError(f"Test fold {fold} contains multiple categories: {held_category_set}")
        held_category = next(iter(held_category_set))
        held_out_categories[held_category] = str(fold)
        train_categories = {row["category"] for row in splits["train"]}
        val_categories = {row["category"] for row in splits["val"]}
        if held_category in train_categories or held_category in val_categories:
            raise ValueError(f"Held-out category {held_category} leaked into train/validation")
        if len(train_categories) != 7 or train_categories != val_categories:
            raise ValueError(f"Expected same seven train/validation categories in fold {fold}")

        for split in ("train", "val", "test"):
            path = fold_dir / f"{split}.jsonl"
            split_hashes.append({
                "fold": fold,
                "held_out_category": held_category,
                "split": split,
                "n_images": len(splits[split]),
                "sha256": sha256(path),
            })

        train_probs = np.asarray([
            [float(row[f"human_p_{label}"]) for label in LABELS]
            for row in splits["train"]
        ], dtype=float)
        if not np.isfinite(train_probs).all() or not np.allclose(train_probs.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Invalid training response distribution in fold {fold}")
        global_prior = train_probs.mean(axis=0)

        run_metadata[fold] = {}
        for arm, mode in (("soft_target", "soft"), ("majority_target", "majority")):
            pred_dir = args.run_root / f"fold_{fold:02d}" / arm
            pred_path = pred_dir / "predictions.jsonl"
            preds = read_jsonl(pred_path)
            test_rows = splits["test"]
            test_id_set = split_ids["test"]
            if len(preds) != 25 or len({row["image_id"] for row in preds}) != 25:
                raise ValueError(f"Expected 25 unique predictions at {pred_path}")
            if {row["image_id"] for row in preds} != test_id_set:
                raise ValueError(f"Predictions do not match held-out test IDs in fold {fold}")
            summary = json.loads((pred_dir / "run_summary.json").read_text(encoding="utf-8"))
            if summary["target_mode"] != mode or summary["model_revision"] != MODEL_REVISION:
                raise ValueError(f"Unexpected mode or checkpoint revision at {pred_dir}")
            run_metadata[fold][arm] = summary
            for row in preds:
                p = np.asarray(row["human_probs"], dtype=float)
                q = np.asarray(row["predicted_probs"], dtype=float)
                if p.shape != (5,) or q.shape != (5,) or not np.isfinite(p).all() or not np.isfinite(q).all():
                    raise ValueError(f"Invalid probabilities for {row['image_id']}")
                if not np.isclose(p.sum(), 1.0, atol=1e-5) or not np.isclose(q.sum(), 1.0, atol=1e-5):
                    raise ValueError(f"Probabilities do not sum to one for {row['image_id']}")
                prediction_sets[arm][row["image_id"]] = row

        if run_metadata[fold]["soft_target"]["seed"] != run_metadata[fold]["majority_target"]["seed"]:
            raise ValueError(f"Paired arm seeds differ in fold {fold}")
        if run_metadata[fold]["soft_target"]["test_images"] != 25 or run_metadata[fold]["majority_target"]["test_images"] != 25:
            raise ValueError(f"Run summary test count is not 25 in fold {fold}")

        for row in splits["test"]:
            image_id = row["image_id"]
            image_path = args.image_root / row["image_file"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            image_hashes[image_id] = sha256(image_path)
            p = [float(row[f"human_p_{label}"]) for label in LABELS]
            if not np.isclose(sum(p), 1.0, atol=1e-5):
                raise ValueError(f"Invalid test response distribution for {image_id}")
            test_counts[image_id] = test_counts.get(image_id, 0) + 1
            category_by_id[image_id] = row["category"]
            human_by_id[image_id] = p
            prediction_sets["global_soft_prior"][image_id] = {
                "image_id": image_id,
                "category": row["category"],
                "fold": fold,
                "human_probs": p,
                "predicted_probs": global_prior.tolist(),
            }

    if set(test_counts.values()) != {1} or len(test_counts) != 200:
        raise ValueError(f"Each of 200 images must appear in exactly one held-out test fold: {len(test_counts)}")
    if len(held_out_categories) != 8:
        raise ValueError(f"Expected one held-out fold for each of eight categories: {held_out_categories}")

    ids = sorted(test_counts)
    categories = np.asarray([category_by_id[image_id] for image_id in ids])
    p = np.asarray([human_by_id[image_id] for image_id in ids], dtype=float)
    arrays = {
        arm: np.asarray([prediction_sets[arm][image_id]["predicted_probs"] for image_id in ids], dtype=float)
        for arm in ARMS
    }
    point = {arm: score(p, arrays[arm]) for arm in ARMS}
    bootstrap_values = {arm: {metric: [] for metric in METRICS} for arm in ARMS}
    comparisons = (("soft_target", "majority_target"), ("soft_target", "global_soft_prior"),
                   ("majority_target", "global_soft_prior"))
    difference_values = {(left, right, metric): [] for left, right in comparisons for metric in METRICS}
    for _ in range(args.bootstrap):
        idx = stratified_indices(categories, rng)
        estimates = {arm: score(p[idx], arrays[arm][idx]) for arm in ARMS}
        for arm in ARMS:
            for metric in METRICS:
                bootstrap_values[arm][metric].append(estimates[arm][metric])
        for left, right in comparisons:
            for metric in METRICS:
                difference_values[(left, right, metric)].append(estimates[left][metric] - estimates[right][metric])

    metric_rows = []
    for arm in ARMS:
        row = {"predictor": arm, "n_images": len(ids)}
        for metric in METRICS:
            samples = np.asarray(bootstrap_values[arm][metric], dtype=float)
            row[metric] = point[arm][metric]
            row[f"{metric}_ci95_low"] = float(np.quantile(samples, 0.025))
            row[f"{metric}_ci95_high"] = float(np.quantile(samples, 0.975))
        metric_rows.append(row)

    paired_rows = []
    for left, right in comparisons:
        for metric in METRICS:
            samples = np.asarray(difference_values[(left, right, metric)], dtype=float)
            paired_rows.append({
                "comparison": f"{left} - {right}",
                "metric": metric,
                "difference": point[left][metric] - point[right][metric],
                "ci95_low": float(np.quantile(samples, 0.025)),
                "ci95_high": float(np.quantile(samples, 0.975)),
                "bootstrap_replicates": args.bootstrap,
            })

    category_rows = []
    for category in sorted(set(categories.tolist())):
        idx = np.flatnonzero(categories == category)
        for arm in ARMS:
            category_rows.append({"category": category, "predictor": arm, "n_images": len(idx),
                                  **score(p[idx], arrays[arm][idx])})

    prediction_rows = []
    for image_id in ids:
        for arm in ARMS:
            row = prediction_sets[arm][image_id]
            prediction_rows.append({
                "image_id": image_id,
                "category": category_by_id[image_id],
                "fold": row["fold"],
                "predictor": arm,
                "human_probs": json.dumps(human_by_id[image_id]),
                "predicted_probs": json.dumps(row["predicted_probs"]),
                "predicted_label": LABELS[int(np.argmax(row["predicted_probs"]))],
                "image_sha256": image_hashes[image_id],
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "condition_metrics.csv", metric_rows)
    write_csv(args.output_dir / "paired_differences.csv", paired_rows)
    write_csv(args.output_dir / "per_category_metrics.csv", category_rows)
    write_csv(args.output_dir / "oof_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "fold_split_manifest.csv", split_hashes)
    (args.output_dir / "image_hashes.json").write_text(json.dumps(image_hashes, indent=2), encoding="utf-8")
    validation = {
        "n_expected_images": 200,
        "n_predictions_per_arm": {arm: len(prediction_sets[arm]) for arm in ARMS},
        "each_image_tested_once": set(test_counts.values()) == {1},
        "category_held_out_once_each": len(held_out_categories) == 8,
        "held_out_categories": held_out_categories,
        "held_out_categories_absent_from_train_and_validation": True,
        "split_ids_disjoint": True,
        "train_validation_test_sizes_per_fold": {"train": 140, "validation": 35, "test": 25},
        "test_categories_count": {category: int(np.sum(categories == category)) for category in sorted(set(categories))},
        "image_hashes_count": len(image_hashes),
        "model_revision": MODEL_REVISION,
        "paired_seeds_match": True,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "scope": "fixed 200-image human-rated subset; leave-one-category-out task-specific adapter training; one FLUX generator; same 30-rater pool; intervals condition on eight fitted outer-fold models",
    }
    (args.output_dir / "analysis_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "run_metrics.json").write_text(
        json.dumps({"conditions": point, "paired_differences": paired_rows}, indent=2), encoding="utf-8"
    )
    print(f"validated and scored {len(ids)} held-out images in {len(held_out_categories)} categories")


if __name__ == "__main__":
    main()
