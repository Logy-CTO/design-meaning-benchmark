from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'reproduced_results'
LABELS = ['functionality', 'aesthetics', 'usability', 'symbolism', 'unclear']
SEED = 20260925
N_BOOT = 5000


def main():
    human = pd.read_csv(DATA / 'human_responses_30x200.csv')
    counts = human.pivot_table(index='image_id', columns='design_intent', values='rater', aggfunc='count', fill_value=0).reindex(columns=LABELS, fill_value=0)
    categories = human.groupby('image_id').category.first().reindex(counts.index)
    ids = list(counts.index)
    matrix = counts.to_numpy(dtype=int)
    max_counts = matrix.max(axis=1)
    winners = matrix == max_counts[:, None]
    n_winners = winners.sum(axis=1)
    tieaware_oracle_acc = 1.0 / n_winners
    oracle_hass = max_counts / matrix.sum(axis=1)

    # Expected metrics for a uniformly random five-class prediction under the paper's fractional tie scoring.
    random_acc = np.full(len(ids), 1.0 / len(LABELS))
    random_hass = np.full(len(ids), 1.0 / len(LABELS))

    # Leave one observed participant out per image. Predict with the plurality among the other 29;
    # score whether that rater's label is in the plurality set, with fractional credit for ties.
    loo_acc = np.zeros(len(ids), dtype=float)
    loo_support = np.zeros(len(ids), dtype=float)
    rater_ids = sorted(human.rater.unique())
    for j, image_id in enumerate(ids):
        rows = human[human.image_id == image_id].set_index('rater')
        per_rater_acc, per_rater_support = [], []
        for rater in rater_ids:
            observed = rows.loc[rater, 'design_intent']
            other = matrix[j].copy()
            other[LABELS.index(observed)] -= 1
            top = np.flatnonzero(other == other.max())
            per_rater_acc.append((1.0 / len(top)) if LABELS.index(observed) in top else 0.0)
            per_rater_support.append(other[LABELS.index(observed)] / other.sum())
        loo_acc[j] = float(np.mean(per_rater_acc))
        loo_support[j] = float(np.mean(per_rater_support))

    by_cat = [np.flatnonzero(categories.to_numpy() == c) for c in sorted(categories.unique())]
    rng = np.random.default_rng(SEED)
    boot = np.concatenate([rng.choice(ix, size=(N_BOOT, len(ix)), replace=True) for ix in by_cat], axis=1)

    rows = []
    metrics = {
        'Uniform random (expected)': (random_acc, random_hass, 'Expected under uniform five-class predictions'),
        'Human leave-one-rater-out': (loo_acc, loo_support, 'Each of 30 observed raters scored against the other 29 on the same image'),
        'In-sample plurality oracle': (tieaware_oracle_acc, oracle_hass, 'Descriptive upper reference computed from all 30 labels; not held-out'),
    }
    for name, (acc, hass, interpretation) in metrics.items():
        rows.append({
            'reference': name,
            'n_images': len(ids),
            'fractional_plurality_accuracy': float(acc.mean()),
            'accuracy_ci95_low': float(np.quantile(acc[boot].mean(axis=1), .025)),
            'accuracy_ci95_high': float(np.quantile(acc[boot].mean(axis=1), .975)),
            'human_support': float(hass.mean()),
            'support_ci95_low': float(np.quantile(hass[boot].mean(axis=1), .025)),
            'support_ci95_high': float(np.quantile(hass[boot].mean(axis=1), .975)),
            'interpretation': interpretation,
        })
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'human_reference_baselines.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == '__main__':
    main()
