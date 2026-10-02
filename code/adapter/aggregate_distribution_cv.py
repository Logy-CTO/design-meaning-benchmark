from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "accuracy", "macro_f1", "cohen_kappa", "hass", "soft_jsd",
    "soft_jsd_alignment", "entropy_mae", "brier_to_human_distribution",
    "aggregate_distribution_jsd", "category_distribution_jsd_mean", "temperature",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    rows = []
    for fold in range(1, 6):
        payload = json.loads((root / f"fold_{fold:02d}" / "distribution_test.summary.json").read_text())
        rows.append({"fold": fold, **{key: payload[key] for key in METRICS}})
    frame = pd.DataFrame(rows)
    evaluation = root / "evaluation"; evaluation.mkdir(exist_ok=True)
    frame.to_csv(evaluation / "distribution_cv_fold_metrics.csv", index=False)
    aggregate = {key: {"mean": float(frame[key].mean()), "std": float(frame[key].std(ddof=0))} for key in METRICS}
    (evaluation / "cv_summary.json").write_text(json.dumps({"adapter_five_fold": aggregate}, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
