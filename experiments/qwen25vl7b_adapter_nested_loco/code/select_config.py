from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

LABELS = ["functionality", "aesthetics", "usability", "symbolism", "unclear"]
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_fold_root", required=True, type=Path)
    parser.add_argument("--nested_data_root", required=True, type=Path)
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--config_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    configs = json.loads(args.config_file.read_text(encoding="utf-8"))
    if len(configs) != 3:
        raise ValueError(f"Expected three pre-existing sweep configurations, found {len(configs)}")
    score_rows = []
    selected = {}
    validation = {
        "outer_folds": 8,
        "inner_folds_per_outer": 7,
        "configs_per_inner_fold": len(configs),
        "selection_metric": "macro-averaged category-level soft cross-entropy (nats), minimized",
        "outer_test_used_in_configuration_selection": False,
        "outer_test_category_present_in_inner_training_or_validation": False,
        "all_175_outer_training_images_scored_once_per_config_and_fold": True,
        "model_revision": MODEL_REVISION,
    }

    for outer in range(1, 9):
        source = args.source_fold_root / f"fold_{outer:02d}"
        outer_test = read_jsonl(source / "test.jsonl")
        outer_categories = {row["category"] for row in outer_test}
        if len(outer_categories) != 1:
            raise ValueError(f"Outer test is not a single category in fold {outer}")
        outer_category = next(iter(outer_categories))
        outer_train = read_jsonl(source / "train.jsonl") + read_jsonl(source / "val.jsonl")
        expected_categories = sorted({row["category"] for row in outer_train})
        if len(outer_train) != 175 or outer_category in expected_categories or len(expected_categories) != 7:
            raise ValueError(f"Outer training scope is invalid in fold {outer}")
        outer_ids = {row["image_id"] for row in outer_test}
        candidate_scores = {config["config_id"]: [] for config in configs}
        candidate_categories = {config["config_id"]: set() for config in configs}
        candidate_ids = {config["config_id"]: set() for config in configs}
        candidate_seeds = {config["config_id"]: set() for config in configs}

        nested_outer = args.nested_data_root / f"outer_{outer:02d}"
        for inner_dir in sorted(nested_outer.glob("inner_*")):
            inner = int(inner_dir.name.split("_")[1])
            inner_val = read_jsonl(inner_dir / "val.jsonl")
            held_categories = {row["category"] for row in inner_val}
            if len(inner_val) != 25 or len(held_categories) != 1:
                raise ValueError(f"Unexpected inner validation fold {inner_dir}")
            held_category = next(iter(held_categories))
            inner_train = read_jsonl(inner_dir / "train.jsonl")
            train_categories = {row["category"] for row in inner_train}
            expected_outer_ids = {row["image_id"] for row in outer_train}
            inner_train_ids = {row["image_id"] for row in inner_train}
            expected_ids = {row["image_id"] for row in inner_val}
            if (held_category == outer_category or outer_category in train_categories
                    or held_category in train_categories or len(inner_train) != 150
                    or len(train_categories) != 6 or inner_train_ids & expected_ids
                    or inner_train_ids & outer_ids or expected_ids & outer_ids
                    or inner_train_ids | expected_ids != expected_outer_ids):
                raise ValueError(f"Outer/inner category or image leakage in {inner_dir}")
            expected_by_id = {row["image_id"]: row for row in inner_val}
            seed_by_config = {}
            for config in configs:
                config_id = config["config_id"]
                pred_dir = args.run_root / "inner" / f"outer_{outer:02d}" / f"inner_{inner:02d}" / config_id
                pred_path = pred_dir / "validation_predictions.jsonl"
                summary_path = pred_dir / "run_summary.json"
                if not pred_path.is_file() or not summary_path.is_file():
                    raise FileNotFoundError(f"Incomplete inner run: {pred_path}")
                predictions = read_jsonl(pred_path)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if (summary["target_mode"] != "soft" or summary["evaluation_split"] != "validation"
                        or summary["model_revision"] != MODEL_REVISION or summary["train_images"] != 150
                        or summary["validation_images"] != 25 or summary["evaluation_images"] != 25
                        or summary["test_images"] != 0):
                    raise ValueError(f"Unexpected inner run metadata: {summary_path}")
                if len(predictions) != 25 or {row["image_id"] for row in predictions} != expected_ids:
                    raise ValueError(f"Inner predictions do not exactly match validation IDs: {pred_path}")
                args_path = pred_dir / "training_args.json"
                train_args = json.loads(args_path.read_text(encoding="utf-8"))
                expected_args = {
                    "lora_r": config["rank"], "lora_alpha": config["alpha"],
                    "lora_dropout": config["dropout"], "lr": config["learning_rate"],
                    "hard_weight": config["hard_weight"], "entropy_weight": config["entropy_weight"],
                    "epochs": config["max_epochs"],
                }
                for name, expected in expected_args.items():
                    actual = train_args[name]
                    if isinstance(expected, float):
                        if abs(float(actual) - expected) > 1e-12:
                            raise ValueError(f"Unexpected {name} for {config_id}: {actual}")
                    elif actual != expected:
                        raise ValueError(f"Unexpected {name} for {config_id}: {actual}")
                seed_by_config[config_id] = int(summary["seed"])
                ids = candidate_ids[config_id]
                categories_seen = candidate_categories[config_id]
                losses = []
                for row in predictions:
                    image_id = row["image_id"]
                    if image_id in ids or row["category"] != held_category or image_id in outer_ids:
                        raise ValueError(f"Repeated, leaked, or miscategorized prediction: {image_id}")
                    source_row = expected_by_id[image_id]
                    p = [float(source_row[f"human_p_{label}"]) for label in LABELS]
                    q = [float(x) for x in row["predicted_probs"]]
                    row_p = [float(x) for x in row["human_probs"]]
                    if (len(q) != 5 or len(row_p) != 5
                            or any(not math.isfinite(x) for x in q + row_p)
                            or any(x < 0.0 for x in q)):
                        raise ValueError(f"Invalid probability vector: {image_id}")
                    if abs(sum(q) - 1.0) > 1e-5 or abs(sum(p) - 1.0) > 1e-5:
                        raise ValueError(f"Probability vector does not sum to one: {image_id}")
                    if any(abs(a - b) > 1e-6 for a, b in zip(p, row_p)):
                        raise ValueError(f"Prediction human target differs from source: {image_id}")
                    losses.append(-sum(a * math.log(max(b, 1e-12)) for a, b in zip(p, q)))
                    ids.add(image_id)
                category_ce = sum(losses) / len(losses)
                candidate_scores[config_id].append(category_ce)
                categories_seen.add(held_category)
                candidate_seeds[config_id].add(int(summary["seed"]))
                score_rows.append({
                    "outer_fold": outer,
                    "outer_held_out_category": outer_category,
                    "inner_fold": inner,
                    "inner_validation_category": held_category,
                    "config_id": config_id,
                    "n_images": len(predictions),
                    "soft_cross_entropy_nats": category_ce,
                    "best_epoch": summary["best_epoch"],
                    "inner_seed": summary["seed"],
                })
            if len(set(seed_by_config.values())) != 1:
                raise ValueError(f"Candidate configuration seeds are not paired in outer={outer}, inner={inner}")

        expected_id_set = {row["image_id"] for row in outer_train}
        seed_sets = [candidate_seeds[config["config_id"]] for config in configs]
        if any(seed_set != seed_sets[0] for seed_set in seed_sets[1:]):
            raise ValueError(f"Inner seed sets differ across candidates in outer fold {outer}")
        ranked = []
        for config in configs:
            config_id = config["config_id"]
            values = candidate_scores[config_id]
            if (len(values) != 7 or candidate_categories[config_id] != set(expected_categories)
                    or candidate_ids[config_id] != expected_id_set or len(candidate_seeds[config_id]) != 7):
                raise ValueError(f"Incomplete category-level inner coverage for outer fold {outer}: {config_id}")
            mean_ce = sum(values) / len(values)
            ranked.append((mean_ce, config_id))
        ranked.sort(key=lambda item: (item[0], item[1]))
        winner_ce, winner_id = ranked[0]
        config_lookup = {config["config_id"]: config for config in configs}
        selected[str(outer)] = {
            **config_lookup[winner_id],
            "outer_fold": outer,
            "outer_held_out_category": outer_category,
            "inner_mean_soft_cross_entropy_nats": winner_ce,
            "inner_ranking": [
                {"config_id": config_id, "mean_soft_cross_entropy_nats": mean_ce}
                for mean_ce, config_id in ranked
            ],
            "inner_category_count": 7,
            "inner_image_count_per_config": 175,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_rows.sort(key=lambda row: (row["outer_fold"], row["inner_fold"], row["config_id"]))
    with (args.output_dir / "inner_selection_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    (args.output_dir / "selected_configs.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "inner_selection_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("selected one configuration per outer fold using only its seven non-test categories")


if __name__ == "__main__":
    main()
