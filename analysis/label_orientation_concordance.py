"""Within-image ordinal concordance in the preserved 30-rater survey.

An exploratory, same-survey association, not independent construct validation.
Original response files are never modified and the primary sample remains 6000.
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/human_responses_30x200.csv'
OUT=ROOT/'reproduced_results'
OUT.mkdir(parents=True,exist_ok=True)
LABELS=['functionality','aesthetics','usability','symbolism','unclear']
human=pd.read_csv(DATA)
assert len(human)==6000 and human.rater.nunique()==30 and human.image_id.nunique()==200
assert not human.duplicated(['rater','image_id']).any()
score=pd.to_numeric(human.function_aesthetic_score,errors='raise')
assert score.notna().all() and score.isin(range(1,6)).all()
human=human.assign(orientation_score=score)
summary=[]
for label in LABELS:
    values=human[human.design_intent==label]
    summary.append({'label':label,'responses':len(values),'images':values.image_id.nunique(),
                    'median_orientation':float(values.orientation_score.median()),
                    'share_orientation_4_or_5':float((values.orientation_score>=4).mean()),
                    **{'score_'+str(s)+'_count':int((values.orientation_score==s).sum()) for s in range(1,6)}})
pd.DataFrame(summary).to_csv(OUT/'label_orientation_descriptives.csv',index=False)

def contrasts(frame):
    rows=[]
    for image_id,group in frame.groupby('image_id',sort=True):
        f=group.loc[group.design_intent=='functionality','orientation_score'].to_numpy()
        a=group.loc[group.design_intent.isin(['aesthetics','symbolism']),'orientation_score'].to_numpy()
        if len(f)==0 or len(a)==0:continue
        differences=a[:,None]-f[None,:]
        rows.append({'image_id':image_id,'category':group.category.iloc[0],
                     'n_functionality':len(f),'n_aesthetics_or_symbolism':len(a),
                     'ordinal_delta':float(np.sign(differences).mean()),
                     'probability_aesthetic_symbolic_higher':float((differences>0).mean()),
                     'probability_tied':float((differences==0).mean())})
    return pd.DataFrame(rows)

def estimate(frame,seed):
    values=contrasts(frame)
    assert len(values)>0
    rng=np.random.default_rng(seed)
    bootstrap=np.zeros(5000)
    n_by_category={}
    for category,group in values.groupby('category',sort=True):
        x=group.ordinal_delta.to_numpy()
        n_by_category[category]=len(x)
        bootstrap+=x[rng.integers(0,len(x),size=(5000,len(x)))].sum(axis=1)
    bootstrap/=len(values)
    return {'eligible_images':len(values),'eligible_images_by_category':n_by_category,
            'mean_within_image_ordinal_delta':float(values.ordinal_delta.mean()),
            'ci95_low':float(np.quantile(bootstrap,.025)),
            'ci95_high':float(np.quantile(bootstrap,.975)),
            'mean_probability_aesthetic_symbolic_higher':float(values.probability_aesthetic_symbolic_higher.mean()),
            'mean_probability_tied':float(values.probability_tied.mean()),
            'bootstrap_replicates':5000,'seed':seed},values

primary,per_image=estimate(human,20260927)
per_image.to_csv(OUT/'label_orientation_per_image.csv',index=False)
pd.DataFrame([dict(analysis='all responses',**{k:v for k,v in primary.items() if k!='eligible_images_by_category'})]).to_csv(OUT/'label_orientation_concordance.csv',index=False)
print(json.dumps(primary,indent=2))
