"""Matched input ablation with two fixed label orders and within-category shuffle."""
from pathlib import Path
import csv,json,hashlib,time,random
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'qwen_prompt_stability'
MODEL='Qwen/Qwen2.5-VL-7B-Instruct';REV='cc594898137f460bfe9f0759e9844b3ce807cfb5'
LABELS=['functionality','aesthetics','usability','symbolism','unclear']
DEFINITIONS={
'functionality':'purpose, operation, or task performance',
'aesthetics':'visual style, beauty, material expression, color, form, or atmosphere',
'usability':'user interaction, comfort, handling, ease of use, or embodied experience',
'symbolism':'cultural meaning, lifestyle, identity, status, history, or period character',
'unclear':'no single dominant interpretation'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prompt(category,order):
    return ('Which perceived product meaning is most strongly supported by the supplied information?\n'
       +'\n'.join('- '+x+': '+DEFINITIONS[x]+'.' for x in order)
       +'\n\nProduct category: '+category
       +'\n\nReturn exactly one lowercase label from the list above and no other text.')
def main():
    out=ROOT/'outputs';out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((BASE/'data/master_image_manifest.csv').open(encoding='utf-8-sig')))
    rows.sort(key=lambda r:(r['category'],int(r['sample_index'])))
    assert len(rows)==200
    groups={c:[r for r in rows if r['category']==c] for c in sorted({r['category'] for r in rows})}
    # One predeclared cyclic shift is a derangement within every category.
    donors={r['image_id']:g[(i+7)%len(g)] for g in groups.values() for i,r in enumerate(g)}
    image_root=BASE/'images/human_subset_200'
    manifest={r['image_id']:{'category':r['category'],'image_sha256':sha(image_root/r['image_file']),'shuffled_donor':donors[r['image_id']]['image_id']} for r in rows}
    done={};file=out/'predictions.jsonl'
    if file.exists():
        for line in file.read_text().splitlines():
            r=json.loads(line);done[(r['condition'],r['image_id'])]=r
    proc=AutoProcessor.from_pretrained(MODEL,revision=REV)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,revision=REV,torch_dtype=torch.bfloat16,device_map='auto',attn_implementation='sdpa').eval()
    orders={'canonical':LABELS,'reverse':list(reversed(LABELS))}
    meta={'model':MODEL,'revision':REV,'precision':'bfloat16','images':manifest,'prompt_templates':{k:prompt('<CATEGORY>',v) for k,v in orders.items()},'conditions':['image_category','image_only','category_only','shuffled_image_category'],'orders':orders,'decoding':'greedy','max_new_tokens':12,'shuffle_rule':'cyclic +7 among 25 sorted sample indices within category; fixed before output inspection','runtime':{'torch':torch.__version__,'transformers':__import__('transformers').__version__,'gpu':torch.cuda.get_device_name(0)},'status':'running'}
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    start=time.time();cache={}
    for order_name,order in orders.items():
      for condition in meta['conditions']:
       for i,row in enumerate(rows):
        cond=order_name+'__'+condition;key=(cond,row['image_id'])
        if key in done:continue
        category=row['product'] if condition!='image_only' else '[not supplied]'
        text=prompt(category,order);donor=donors[row['image_id']] if condition=='shuffled_image_category' else row
        supplied_image=condition!='category_only';image_path=image_root/donor['image_file']
        cachekey=(order_name,category)
        if not supplied_image and cachekey in cache:raw=cache[cachekey];reused=True
        else:
            content=[];kwargs={}
            if supplied_image:
                image=Image.open(image_path).convert('RGB');content.append({'type':'image','image':image});kwargs['images']=[image]
            content.append({'type':'text','text':text})
            chat=proc.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
            inputs=proc(text=[chat],padding=True,return_tensors='pt',**kwargs).to(model.device)
            with torch.inference_mode():gen=model.generate(**inputs,max_new_tokens=12,do_sample=False)
            raw=proc.batch_decode(gen[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
            if not supplied_image:cache[cachekey]=raw
            reused=False
        label=raw.strip().lower().strip('."\' ')
        record={'condition':cond,'input_condition':condition,'label_order':order_name,'image_id':row['image_id'],'category':row['category'],'category_value':category,'supplied_image_id':donor['image_id'] if supplied_image else None,'supplied_image_sha256':sha(image_path) if supplied_image else None,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'raw_response':raw,'predicted_label':label if label in LABELS else None,'category_inference_reused':reused}
        with file.open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
        done[key]=record
        print(f'{cond} {i+1}/200 {label}',flush=True)
    meta.update(status='completed',rows=len(done),elapsed_seconds=time.time()-start,invalid=sum(r['predicted_label'] is None for r in done.values()))
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2));print('COMPLETE '+json.dumps({k:meta[k] for k in ['rows','elapsed_seconds','invalid']}),flush=True)
if __name__=='__main__':main()
