"""Score matched category/image conditions without converting failures to a label."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import reanalysis as ra

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/qwen_visual_category_controls'
OUT=EXP/'analysis';OUT.mkdir(exist_ok=True)
meta=json.loads((EXP/'outputs/run_metadata.json').read_text())
assert meta['status']=='completed' and meta['rows']==1600
assert meta['revision']=='cc594898137f460bfe9f0759e9844b3ce807cfb5'
assert meta['model']=='Qwen/Qwen2.5-VL-7B-Instruct' and meta['precision']=='bfloat16'
assert meta['decoding']=='greedy' and meta['max_new_tokens']==12
records=[json.loads(x) for x in (EXP/'outputs/predictions.jsonl').read_text().splitlines()]
human=pd.read_csv(ra.DATA/'human_responses_30x200.csv')
p,w,t=ra.response_distribution(human);ids=sorted(p.index)
cats=human.groupby('image_id').category.first().reindex(ids)
weights=human.groupby('image_id').confidence.mean().reindex(ids).to_numpy()/5
conditions=[o+'__'+c for o in meta['orders'] for c in meta['conditions']]
assert len(records)==len({(r['condition'],r['image_id']) for r in records})==1600
assert {(r['condition'],r['image_id']) for r in records}=={(c,i) for c in conditions for i in ids}
manifest=pd.read_csv(ROOT/'experiments/qwen_prompt_stability/data/master_image_manifest.csv').set_index('image_id')
for i in ids:
    path=ROOT/'images'/manifest.loc[i,'image_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['images'][i]['image_sha256']
for r in records:
    i=r['image_id'];c=r['input_condition'];donor=r['supplied_image_id']
    assert r['category']==cats.loc[i]
    if c=='category_only':assert donor is None and r['supplied_image_sha256'] is None
    else:
        assert donor==(meta['images'][i]['shuffled_donor'] if c=='shuffled_image_category' else i)
        assert r['supplied_image_sha256']==meta['images'][donor]['image_sha256']
        if c=='shuffled_image_category':assert donor!=i and cats.loc[donor]==cats.loc[i]
    assert r['category_value']==('[not supplied]' if c=='image_only' else manifest.loc[i,'product'])
    prompt=meta['prompt_templates'][r['label_order']].replace('<CATEGORY>',r['category_value'])
    assert hashlib.sha256(prompt.encode()).hexdigest()==r['prompt_sha256']
    normalized=r['raw_response'].strip().lower().strip('.\"\' ')
    assert r['predicted_label']==(normalized if normalized in ra.LABELS else None)
boot=ra.bootstrap_indices(cats,np.random.default_rng(20260927))
arrays={};labels={};metrics=[];bycat=[]
for cond in conditions:
    rows={r['image_id']:r for r in records if r['condition']==cond}
    labels[cond]=[rows[i]['predicted_label'] for i in ids]
    # Invalid outputs receive zero task success and zero selected-label support.
    a=np.array([ra.plurality_credit(rows[i]['predicted_label'],w.loc[i]) for i in ids])
    h=np.array([p.loc[i,rows[i]['predicted_label']] if rows[i]['predicted_label'] else 0 for i in ids])
    arrays[cond]={'accuracy':a,'HASS':h}
    row={'condition':cond,'n':200,'invalid':sum(x is None for x in labels[cond]),'unique_model_queries':sum(not r['category_inference_reused'] for r in rows.values())}
    for name,arr in arrays[cond].items():
        ci=ra.confidence_interval(arr,boot);row.update({name:float(arr.mean()),name+'_low':ci[0],name+'_high':ci[1]})
    row['confidence_weighted_HASS']=float(np.average(h,weights=weights));metrics.append(row)
    for cat in sorted(cats.unique()):
        ix=np.flatnonzero(cats.to_numpy()==cat)
        bycat.append({'condition':cond,'category':cat,'accuracy':float(a[ix].mean()),'HASS':float(h[ix].mean())})
pairs=[]
comparisons=[]
for order in meta['orders']:
    comparisons.extend((order+'__image_category',order+'__'+right) for right in ['image_only','category_only','shuffled_image_category'])
comparisons.extend(('canonical__'+c,'reverse__'+c) for c in meta['conditions'])
for left,right in comparisons:
    row={'left':left,'right':right,'n':200,'labels_changed':sum(a!=b for a,b in zip(labels[left],labels[right]))}
    for metric in ['accuracy','HASS']:
        d=arrays[left][metric]-arrays[right][metric];ci=ra.confidence_interval(d,boot)
        row.update({'delta_'+metric:float(d.mean()),metric+'_low':ci[0],metric+'_high':ci[1]})
    pairs.append(row)
pd.DataFrame(metrics).to_csv(OUT/'condition_metrics.csv',index=False)
pd.DataFrame(pairs).to_csv(OUT/'paired_differences.csv',index=False)
pd.DataFrame(bycat).to_csv(OUT/'per_category_metrics.csv',index=False)
validation={'n_records':len(records),'image_hashes_checked':200,'human_rows':len(human),'bootstrap_replicates':5000,'bootstrap_seed':20260927,'invalid_outputs':sum(r['predicted_label'] is None for r in records),'unique_model_queries':sum(r['unique_model_queries'] for r in metrics),'scope':'Post hoc fixed 200 images; matched neutral label-only prompt, not the original multi-field benchmark prompt. Category-only uses eight unique queries per label order, expanded over 25 images each. Paired category-stratified image bootstrap conditions on these eight categories, the panel, deterministic outputs and the fixed shuffle; it does not estimate prompt or category population uncertainty. Invalid labels are retained as failures, never recoded as unclear. Label-order analyses are sensitivity checks, not independent replications.','files':{f:hashlib.sha256((EXP/'outputs'/f).read_bytes()).hexdigest() for f in ['run_metadata.json','predictions.jsonl']}}
(OUT/'validation.json').write_text(json.dumps(validation,indent=2))
print(pd.DataFrame(metrics).to_string(index=False));print(pd.DataFrame(pairs).to_string(index=False))
