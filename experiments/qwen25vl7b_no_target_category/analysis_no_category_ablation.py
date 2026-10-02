from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'analysis'))
import reanalysis as ra

DATA = PROJECT / 'data'
EXP = Path(__file__).resolve().parent
OUT = EXP / 'analysis'
LABELS = ra.LABELS


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    human = pd.read_csv(DATA / 'human_responses_30x200.csv')
    probs, winner_mask, ties = ra.response_distribution(human)
    ids = sorted(probs.index)
    category = human.groupby('image_id').category.first().reindex(ids)
    confidence = human.groupby('image_id').confidence.mean().reindex(ids) / 5.0

    # Reconstruct exactly the train-only category priors used in the principal audit.
    category_pred = {}
    for fold in range(1, 6):
        folder = DATA / 'adapter_folds' / f'fold_{fold:02d}'
        train = [json.loads(line) for line in (folder / 'train.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        test = [json.loads(line) for line in (folder / 'test.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        by_cat = {}
        for row in train:
            by_cat.setdefault(row['category'], []).append(row['image_id'])
        q = {cat: probs.loc[train_ids].mean(axis=0).to_numpy() for cat, train_ids in by_cat.items()}
        for row in test:
            category_pred[row['image_id']] = LABELS[int(np.argmax(q[row['category']]))]

    full_vlm = pd.read_csv(DATA / 'vlm_predictions_full1600.csv')
    supplied = full_vlm[full_vlm.model == 'Qwen2.5-VL (7B, 2025)'].set_index('image_id').design_intent.to_dict()
    no_cat_df = pd.read_csv(EXP / 'outputs' / 'qwen25_vl_7b_no_target_category_200.csv')
    no_cat = no_cat_df.set_index('image_id').design_intent.to_dict()
    if len(no_cat) != 200 or set(no_cat) != set(ids):
        raise ValueError(f'Expected one no-category prediction for every human-rated image; got {len(no_cat)}')

    predictions = {
        'Category-only soft prior (train only)': category_pred,
        'Qwen2.5-VL-7B (category supplied)': supplied,
        'Qwen2.5-VL-7B (category omitted)': no_cat,
    }
    vectors = {}
    scores = []
    for name, pred in predictions.items():
        row, acc_i, hass_i = ra.score_predictions(name, pred, probs, winner_mask, ties, confidence)
        vectors[name] = (acc_i, hass_i)
        scores.append(row)

    seed = 20260925
    rng = np.random.default_rng(seed)
    boot = ra.bootstrap_indices(category, rng)
    ordered_conf = confidence.reindex(ids).to_numpy(float)
    boot_w = ordered_conf[boot]
    for row in scores:
        name = row['predictor']
        acc_i, hass_i = vectors[name]
        row['accuracy_ci95_low'] = float(np.quantile(acc_i[boot].mean(axis=1), .025))
        row['accuracy_ci95_high'] = float(np.quantile(acc_i[boot].mean(axis=1), .975))
        row['HASS_ci95_low'] = float(np.quantile(hass_i[boot].mean(axis=1), .025))
        row['HASS_ci95_high'] = float(np.quantile(hass_i[boot].mean(axis=1), .975))
        wh = hass_i[boot] * boot_w
        row['confidence_weighted_HASS_ci95_low'] = float(np.quantile(wh.sum(axis=1) / boot_w.sum(axis=1), .025))
        row['confidence_weighted_HASS_ci95_high'] = float(np.quantile(wh.sum(axis=1) / boot_w.sum(axis=1), .975))
    pd.DataFrame(scores).to_csv(OUT / 'no_category_comparison.csv', index=False)

    diff_rows = []
    contrasts = [
        ('Qwen2.5-VL-7B (category supplied)', 'Qwen2.5-VL-7B (category omitted)'),
        ('Category-only soft prior (train only)', 'Qwen2.5-VL-7B (category omitted)'),
    ]
    for base, test in contrasts:
        for metric_ix, metric in ((0, 'fractional_plurality_accuracy'), (1, 'HASS')):
            d = vectors[base][metric_ix] - vectors[test][metric_ix]
            ci = np.quantile(d[boot].mean(axis=1), [.025, .975])
            diff_rows.append({'difference': f'{base} - {test}', 'metric': metric, 'point_estimate': float(d.mean()), 'ci95_low': float(ci[0]), 'ci95_high': float(ci[1])})
        base_h = vectors[base][1]
        test_h = vectors[test][1]
        d_num = (base_h[boot] * boot_w).sum(axis=1) / boot_w.sum(axis=1) - (test_h[boot] * boot_w).sum(axis=1) / boot_w.sum(axis=1)
        diff_rows.append({'difference': f'{base} - {test}', 'metric': 'confidence_weighted_HASS', 'point_estimate': float(np.average(base_h - test_h, weights=ordered_conf)), 'ci95_low': float(np.quantile(d_num, .025)), 'ci95_high': float(np.quantile(d_num, .975))})
    pd.DataFrame(diff_rows).to_csv(OUT / 'no_category_paired_differences.csv', index=False)

    shift = pd.crosstab(pd.Series([supplied[i] for i in ids], name='category_supplied'), pd.Series([no_cat[i] for i in ids], name='category_omitted')).reindex(index=LABELS, columns=LABELS, fill_value=0)
    shift.to_csv(OUT / 'prediction_shift_matrix.csv')
    change = pd.DataFrame({'image_id': ids, 'category': category.values, 'category_supplied': [supplied[i] for i in ids], 'category_omitted': [no_cat[i] for i in ids]})
    change['changed'] = change.category_supplied != change.category_omitted
    change.to_csv(OUT / 'prediction_changes_by_image.csv', index=False)
    group = change.groupby('category').agg(n=('image_id', 'count'), changed=('changed', 'sum')).reset_index()
    group['fraction_changed'] = group.changed / group.n
    for label in LABELS:
        group[f'no_category_{label}'] = group.category.map(change[change.category_omitted == label].groupby('category').size()).fillna(0).astype(int)
    group.to_csv(OUT / 'prediction_changes_by_category.csv', index=False)

    prompt_file = EXP / 'code' / 'schema' / 's2_8plus5.py'
    meta = {
        'experiment': 'Qwen2.5-VL-7B target-category-omitted prompt ablation',
        'model_id': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'model_revision': 'cc594898137f460bfe9f0759e9844b3ce807cfb5',
        'precision': 'bfloat16',
        'decoding': {'do_sample': False, 'max_new_tokens': 220},
        'images': 200,
        'categories': 8,
        'selection': 'sample_index 1-25 per category; fixed subset, selection rationale absent in archive',
        'prompt_key': 's2_descriptive_fields_no_target_category',
        'prompt_source_sha256': hashlib.sha256(prompt_file.read_bytes()).hexdigest(),
        'original_prompt_sha256': '26aa07eb586ba5321485b0c04ed3262ddbef885b9807ec7467654a357ae4c930',
        'runtime': {'torch': '2.8.0+cu128', 'transformers': '5.12.1', 'mistral_common_isolated': '1.12.0', 'gpu': 'NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition'},
        'output_rows': int(len(no_cat_df)),
        'invalid_or_missing_design_intent': int((~no_cat_df.design_intent.isin(LABELS)).sum()),
    }
    (OUT / 'run_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(pd.DataFrame(scores).to_string(index=False))
    print('\nPaired differences')
    print(pd.DataFrame(diff_rows).to_string(index=False))
    print('\nPrediction consistency:', float(np.mean([supplied[i] == no_cat[i] for i in ids])))
    print('\nShift matrix')
    print(shift.to_string())

if __name__ == '__main__':
    main()

