from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SPLITS = ("train", "val", "test")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fold-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=8419)
    args = parser.parse_args()

    source_dir = args.source_fold_root / "fold_01"
    source_rows = [row for split in SPLITS for row in read_jsonl(source_dir / f"{split}.jsonl")]
    rows_by_id = {row["image_id"]: row for row in source_rows}
    if len(rows_by_id) != len(source_rows) or len(rows_by_id) != 200:
        raise ValueError(f"Expected 200 unique source images, found {len(rows_by_id)} rows / {len(source_rows)} records")

    categories = sorted({row["category"] for row in rows_by_id.values()})
    if len(categories) != 8:
        raise ValueError(f"Expected eight categories, found {categories}")
    grouped = {
        category: sorted(
            (row for row in rows_by_id.values() if row["category"] == category),
            key=lambda row: row["image_id"],
        )
        for category in categories
    }
    if any(len(grouped[category]) != 25 for category in categories):
        raise ValueError("Each category must contain exactly 25 human-rated images")

    args.output_root.mkdir(parents=True, exist_ok=True)
    split_manifest: list[dict] = []
    fold_summaries: list[dict] = []
    for fold, held_out in enumerate(categories, start=1):
        fold_dir = args.output_root / f"fold_{fold:02d}"
        partitions: dict[str, list[dict]] = {"train": [], "val": [], "test": list(grouped[held_out])}
        for category_index, category in enumerate(categories):
            if category == held_out:
                continue
            category_rows = list(grouped[category])
            rng = np.random.default_rng(args.seed + 100 * fold + category_index)
            permutation = rng.permutation(len(category_rows))
            shuffled = [category_rows[int(index)] for index in permutation]
            partitions["train"].extend(shuffled[:20])
            partitions["val"].extend(shuffled[20:])

        for split in SPLITS:
            partitions[split].sort(key=lambda row: row["image_id"])
            digest = write_jsonl(fold_dir / f"{split}.jsonl", partitions[split])
            for row in partitions[split]:
                split_manifest.append({
                    "fold": fold,
                    "held_out_category": held_out,
                    "split": split,
                    "image_id": row["image_id"],
                    "category": row["category"],
                    "image_file": row["image_file"],
                })
        train_categories = {row["category"] for row in partitions["train"]}
        val_categories = {row["category"] for row in partitions["val"]}
        if held_out in train_categories or held_out in val_categories:
            raise AssertionError(f"Held-out category leaked into train/validation: {held_out}")
        fold_summaries.append({
            "fold": fold,
            "held_out_category": held_out,
            "train_images": len(partitions["train"]),
            "validation_images": len(partitions["val"]),
            "test_images": len(partitions["test"]),
            "train_categories": sorted(train_categories),
            "validation_categories": sorted(val_categories),
            "test_categories": sorted({row["category"] for row in partitions["test"]}),
            "split_sha256": {
                split: hashlib.sha256((fold_dir / f"{split}.jsonl").read_bytes()).hexdigest()
                for split in SPLITS
            },
        })

    with (args.output_root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_manifest[0].keys()))
        writer.writeheader()
        writer.writerows(split_manifest)
    (args.output_root / "dataset_summary.json").write_text(
        json.dumps({
            "n_images": len(rows_by_id),
            "categories": categories,
            "images_per_category": {category: len(grouped[category]) for category in categories},
            "outer_cv": "leave-one-human-rated-category-out",
            "held_out_category_is_absent_from_train_and_validation": True,
            "split_rule": "per non-held-out category, deterministic 20 train / 5 validation; held-out category 25 test",
            "seed": args.seed,
            "folds": fold_summaries,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {len(fold_summaries)} folds and {len(split_manifest)} split assignments to {args.output_root}")


if __name__ == "__main__":
    main()
