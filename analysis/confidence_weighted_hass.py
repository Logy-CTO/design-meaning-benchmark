from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import reanalysis as ra
from extended_audit import load_maps

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "reproduced_results"
N_BOOT = 5000
SEED = 20260925


def main() -> None:
    human = pd.read_csv(ra.DATA / "human_responses_30x200.csv")
    probs, _, _ = ra.response_distribution(human)
    ids = sorted(probs.index)
    confidence = human.groupby("image_id").confidence.mean().reindex(ids).to_numpy(dtype=float) / 5.0
    categories = human.groupby("image_id").category.first().reindex(ids)
    maps, _ = load_maps(human, ids, probs)
    rng = np.random.default_rng(SEED)
    bootstrap = ra.bootstrap_indices(categories, rng)
    rows = []
    for name, prediction_map in maps.items():
        selected_support = np.asarray([
            probs.loc[image_id, prediction_map[image_id]] for image_id in ids
        ], dtype=float)
        point = float(np.average(selected_support, weights=confidence))
        draws = np.asarray([
            np.average(selected_support[index], weights=confidence[index])
            for index in bootstrap
        ])
        rows.append({
            "predictor": name, "n_images": len(ids),
            "confidence_weighted_HASS": point,
            "ci95_low": float(np.quantile(draws, .025)),
            "ci95_high": float(np.quantile(draws, .975)),
        })
    pd.DataFrame(rows).to_csv(OUT / "confidence_weighted_hass_ci.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
