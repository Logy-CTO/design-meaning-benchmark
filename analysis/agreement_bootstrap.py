"""Category-stratified image bootstrap intervals for inter-rater agreement.

Fleiss kappa matches the archived calculation. Nominal alpha uses pairable
coincidences, correcting the archive's within-item self-pairing bias. Images
are resampled within each observed category with all 30 ratings together.
The resulting intervals condition on the observed rater panel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SUBMISSION = PROJECT
INPUT = SUBMISSION / "data" / "human_responses_30x200.csv"
EXPECTED = SUBMISSION / "expected_results" / "human30_human_agreement_summary.csv"
OUTPUT = PROJECT / "reproduced_results" / "human_agreement_bootstrap.csv"
N_BOOT = 5000
SEED = 20260925


def agreement_estimates(counts: np.ndarray) -> tuple[float, float]:
    """Return Fleiss kappa and standard finite-sample nominal alpha."""
    matrix = np.asarray(counts, dtype=float)
    n_raters = matrix.sum(axis=1)
    valid = n_raters > 1
    matrix = matrix[valid]
    n_raters = n_raters[valid]

    pooled = matrix.sum(axis=0) / matrix.sum()
    item_agreement = (np.sum(matrix * matrix, axis=1) - n_raters) / (
        n_raters * (n_raters - 1)
    )
    expected_agreement = float(np.sum(pooled * pooled))
    fleiss = float((np.mean(item_agreement) - expected_agreement) / (1 - expected_agreement))

    total_by_label = matrix.sum(axis=0)
    total = float(total_by_label.sum())
    expected_disagreement = float((total**2-np.sum(total_by_label**2))/(total*(total-1)))
    observed_by_item = (n_raters**2-np.sum(matrix**2,axis=1))/(n_raters-1)
    observed_disagreement = float(observed_by_item.sum()/total)
    alpha = float(1 - observed_disagreement / expected_disagreement)
    return fleiss, alpha


def main() -> None:
    human = pd.read_csv(INPUT)
    label_order = ["functionality", "aesthetics", "usability", "symbolism", "unclear"]
    image_meta = human.groupby("image_id").agg(category=("category", "first"))
    counts = human.pivot_table(
        index="image_id", columns="design_intent", values="rater", aggfunc="count", fill_value=0
    ).reindex(columns=label_order, fill_value=0)
    counts = counts.reindex(image_meta.index)

    ratings_per_image = human.groupby("image_id").size().reindex(counts.index)
    category_sizes = image_meta["category"].value_counts().to_dict()
    if len(counts) != 200 or human["rater"].nunique() != 30:
        raise ValueError("Expected 200 images and 30 distinct raters.")
    if not (ratings_per_image == 30).all():
        raise ValueError("Each image must have exactly 30 observed ratings.")
    if len(category_sizes) != 8 or set(category_sizes.values()) != {25}:
        raise ValueError("Expected 25 images in each of the eight categories.")

    observed = agreement_estimates(counts.to_numpy())
    archived = pd.read_csv(EXPECTED).iloc[0]
    archived_values = (float(archived["fleiss_kappa"]), float(archived["krippendorff_alpha_nominal"]))
    if not np.isclose(observed[0], archived_values[0], rtol=0, atol=1e-10):
        raise ValueError(f"Fleiss kappa changed relative to the archive: {observed[0]} vs {archived_values[0]}")

    rng = np.random.default_rng(SEED)
    categories = image_meta["category"].to_numpy()
    strata = [np.flatnonzero(categories == category) for category in sorted(category_sizes)]
    sampled_indices = np.concatenate(
        [rng.choice(indices, size=(N_BOOT, len(indices)), replace=True) for indices in strata],
        axis=1,
    )
    sampled_counts = counts.to_numpy()[sampled_indices]
    n_raters = sampled_counts.sum(axis=2)

    pooled = sampled_counts.sum(axis=1) / n_raters.sum(axis=1, keepdims=True)
    item_agreement = (np.sum(sampled_counts * sampled_counts, axis=2) - n_raters) / (
        n_raters * (n_raters - 1)
    )
    expected_agreement = np.sum(pooled * pooled, axis=1)
    boot_fleiss = (item_agreement.mean(axis=1) - expected_agreement) / (1 - expected_agreement)

    totals = sampled_counts.sum(axis=1)
    total=n_raters.sum(axis=1)
    expected_disagreement = (total**2-np.sum(totals**2,axis=1))/(total*(total-1))
    observed_by_item = (n_raters**2-np.sum(sampled_counts**2,axis=2))/(n_raters-1)
    observed_disagreement = observed_by_item.sum(axis=1)/total
    boot_alpha = 1 - observed_disagreement / expected_disagreement
    # With complete equal-sized panels, finite-sample alpha and Fleiss kappa have this exact relation.
    assert np.allclose(boot_alpha,1-(total-1)/total*(1-boot_fleiss),rtol=0,atol=1e-12)

    rows = []
    for name, estimate, replicates in (
        ("fleiss_kappa", observed[0], boot_fleiss),
        ("krippendorff_alpha_nominal", observed[1], boot_alpha),
    ):
        low, high = np.quantile(replicates, [0.025, 0.975])
        rows.append(
            {
                "measure": name,
                "estimate": estimate,
                "ci95_percentile_low": float(low),
                "ci95_percentile_high": float(high),
                "bootstrap_replicates": N_BOOT,
                "seed": SEED,
                "resampling_unit": "image within category",
                "conditioning": "fixed observed set of 30 raters",
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False, float_format="%.10f")
    correction={"archived_alpha":archived_values[1],"standard_nominal_alpha":observed[1],
                "reason":"Archive used within-item proportions without excluding self-pairs and omitted the finite-sample expected-disagreement adjustment.",
                "definition_source":"https://stat.ethz.ch/CRAN/web/packages/matrixCorr/refman/matrixCorr.html#krippendorff_alpha",
                "primary_rows_unchanged":6000,"bootstrap_equal_panel_relation_max_error":float(np.max(np.abs(boot_alpha-(1-(total-1)/total*(1-boot_fleiss)))))}
    (OUTPUT.parent/'nominal_alpha_correction.json').write_text(json.dumps(correction,indent=2),encoding='utf-8')
    print(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "results": rows,"correction":correction}, indent=2))


if __name__ == "__main__":
    main()
