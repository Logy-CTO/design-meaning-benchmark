from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
import reanalysis as base  # noqa: E402


def main() -> None:
    human = pd.read_csv(base.DATA / "human_responses_30x200.csv")
    probs, winner_mask, ties = base.response_distribution(human)
    image_ids = list(probs.index)
    categories = human.groupby("image_id").category.first().reindex(image_ids)
    mean_confidence = human.groupby("image_id").confidence.mean().reindex(image_ids).to_numpy(float) / 5.0

    raw = pd.read_csv(base.DATA / "vlm_predictions_full1600.csv")
    raw = raw[raw.image_id.isin(image_ids)]
    models = {
        name: dict(zip(group.image_id, group.design_intent))
        for name, group in raw.groupby("model", sort=False)
    }
    model_a = "GPT-5.5"
    model_b = "InternVL3 (38B, 2025)"
    if set(models[model_a]) != set(image_ids) or set(models[model_b]) != set(image_ids):
        raise ValueError("The selected model predictions do not cover the exact 200-image subset")

    _, accuracy_a, hass_a = base.score_predictions(model_a, models[model_a], probs, winner_mask, ties)
    _, accuracy_b, hass_b = base.score_predictions(model_b, models[model_b], probs, winner_mask, ties)
    p = probs.loc[image_ids].to_numpy(float)
    selected_a = np.asarray([models[model_a][image_id] for image_id in image_ids], dtype=object)
    selected_b = np.asarray([models[model_b][image_id] for image_id in image_ids], dtype=object)
    labels = list(probs.columns)
    support_a = np.asarray([p[i, labels.index(selected_a[i])] for i in range(len(image_ids))])
    support_b = np.asarray([p[i, labels.index(selected_b[i])] for i in range(len(image_ids))])
    weighted_a = float(np.average(support_a, weights=mean_confidence))
    weighted_b = float(np.average(support_b, weights=mean_confidence))

    rng = np.random.default_rng(base.SEED)
    bootstrap = base.bootstrap_indices(categories, rng)
    rows = []
    for metric, left, right in (
        ("tie-aware accuracy", accuracy_a, accuracy_b),
        ("HASS", hass_a, hass_b),
    ):
        delta = left - right
        draws = delta[bootstrap].mean(axis=1)
        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "difference_a_minus_b": float(delta.mean()),
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "bootstrap_unit": "images within category; fixed 30 raters",
            "n_boot": base.N_BOOT,
            "seed": base.SEED,
        })

    # Confidence-weighted selected-label support is a ratio, so recompute its
    # numerator and denominator for each resample rather than resampling scores alone.
    delta_num = (support_a - support_b) * mean_confidence
    delta_ci = np.asarray([
        delta_num[index].sum() / mean_confidence[index].sum()
        for index in bootstrap
    ])
    rows.append({
        "model_a": model_a,
        "model_b": model_b,
        "metric": "confidence-weighted HASS",
        "difference_a_minus_b": weighted_a - weighted_b,
        "ci95_low": float(np.quantile(delta_ci, 0.025)),
        "ci95_high": float(np.quantile(delta_ci, 0.975)),
        "bootstrap_unit": "images within category; fixed 30 raters",
        "n_boot": base.N_BOOT,
        "seed": base.SEED,
    })
    pd.DataFrame(rows).to_csv(PROJECT / "reproduced_results" / "paired_top_model_differences.csv", index=False)


if __name__ == "__main__":
    main()
