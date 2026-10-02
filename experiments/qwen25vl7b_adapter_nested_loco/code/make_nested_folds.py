from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_fold_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    args = parser.parse_args()

    manifest = {
        "design": "For each outer held-out category, inner leave-one-category-out tuning over the other seven categories.",
        "inner_validation_images_per_fold": 25,
        "inner_training_images_per_fold": 150,
        "outer_category_used_in_inner_splits": False,
        "outer_test_images_used_for_configuration_selection": False,
        "folds": [],
    }
    for outer in range(1, 9):
        source = args.source_fold_root / f"fold_{outer:02d}"
        train_rows = read_jsonl(source / "train.jsonl")
        validation_rows = read_jsonl(source / "val.jsonl")
        outer_test = read_jsonl(source / "test.jsonl")
        outer_test_categories = {row["category"] for row in outer_test}
        if len(outer_test_categories) != 1 or len(outer_test) != 25:
            raise ValueError(f"Unexpected outer test fold {outer}: {outer_test_categories}, n={len(outer_test)}")
        outer_category = next(iter(outer_test_categories))
        outer_train = train_rows + validation_rows
        outer_ids = {row["image_id"] for row in outer_train}
        test_ids = {row["image_id"] for row in outer_test}
        if outer_ids & test_ids or len(outer_ids) != 175:
            raise ValueError(f"Outer train/test overlap or wrong sample size in fold {outer}")
        counts = Counter(row["category"] for row in outer_train)
        categories = sorted(counts)
        if outer_category in counts or len(categories) != 7 or any(counts[c] != 25 for c in categories):
            raise ValueError(f"Outer category leaked or unexpected counts in fold {outer}: {counts}")
        fold_record = {
            "outer_fold": outer,
            "outer_held_out_category": outer_category,
            "n_outer_train_validation": len(outer_train),
            "inner_categories": [],
            "outer_test_ids_sha256": hashlib.sha256("\n".join(sorted(test_ids)).encode()).hexdigest(),
        }
        for inner, held_category in enumerate(categories, start=1):
            inner_validation = sorted(
                (row for row in outer_train if row["category"] == held_category),
                key=lambda row: row["image_id"],
            )
            inner_train = sorted(
                (row for row in outer_train if row["category"] != held_category),
                key=lambda row: row["image_id"],
            )
            if len(inner_train) != 150 or len(inner_validation) != 25:
                raise ValueError(f"Unexpected inner split size: outer={outer}, inner={inner}")
            train_ids = {row["image_id"] for row in inner_train}
            validation_ids = {row["image_id"] for row in inner_validation}
            if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
                raise ValueError(f"Image/category leakage in outer={outer}, inner={inner}")
            out = args.output_root / f"outer_{outer:02d}" / f"inner_{inner:02d}"
            write_jsonl(out / "train.jsonl", inner_train)
            write_jsonl(out / "val.jsonl", inner_validation)
            fold_record["inner_categories"].append({
                "inner_fold": inner,
                "held_out_category": held_category,
                "train_images": len(inner_train),
                "validation_images": len(inner_validation),
                "train_ids_sha256": hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest(),
                "validation_ids_sha256": hashlib.sha256("\n".join(sorted(validation_ids)).encode()).hexdigest(),
            })
        manifest["folds"].append(fold_record)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "nested_folds_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("created 56 nested folds; 150 train and 25 validation images per fold")


if __name__ == "__main__":
    main()
