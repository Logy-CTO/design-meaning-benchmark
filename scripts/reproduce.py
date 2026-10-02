"""Recalculate study results from saved predictions without model inference."""
from pathlib import Path
import argparse, subprocess, sys, json
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduced_results'
OUT.mkdir(exist_ok=True)

def run(script,*args):
    print('Running',script,flush=True)
    result=subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (OUT/(Path(script).stem+'.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:
        print(result.stderr)
        raise SystemExit(result.returncode)

def compare():
    report=[]
    for reference in sorted((ROOT/'expected_results').glob('*.csv')):
        actual=OUT/reference.name
        if not actual.exists():continue
        expected=pd.read_csv(reference);result=pd.read_csv(actual)
        if list(expected.columns)!=list(result.columns) or len(expected)!=len(result):
            raise RuntimeError('Shape/columns differ: '+reference.name)
        numeric=expected.select_dtypes(include='number').columns
        nonnumeric=[c for c in expected if c not in numeric]
        if nonnumeric and not expected[nonnumeric].fillna('').equals(result[nonnumeric].fillna('')):
            raise RuntimeError('Row identities differ: '+reference.name)
        delta=0.0
        if len(numeric):
            a=expected[numeric].to_numpy(float);b=result[numeric].to_numpy(float)
            if not np.allclose(a,b,rtol=1e-7,atol=1e-8,equal_nan=True):
                raise RuntimeError('Numeric mismatch: '+reference.name)
            finite=np.isfinite(a)&np.isfinite(b)
            if finite.any():delta=float(np.abs(a[finite]-b[finite]).max())
        report.append({'file':reference.name,'rows':len(expected),'max_absolute_difference':delta})
    (OUT/'reference_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Reference comparisons passed:',len(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--extended',action='store_true')
    args=parser.parse_args()
    for name in ['agreement_bootstrap','reanalysis','human_reference_baselines','extended_audit','confidence_weighted_hass','paired_top_model_differences','label_orientation_concordance']:
        run('analysis/'+name+'.py')
    compare()
    if args.extended:
        run('experiments/qwen25vl7b_no_target_category/analysis_no_category_ablation.py')
        run('analysis/prompt_stability_analysis.py')
        run('analysis/analyze_matched_visual_controls.py')
        nested=ROOT/'experiments/qwen25vl7b_adapter_nested_loco'
        run('experiments/qwen25vl7b_adapter_nested_loco/code/select_config.py','--source_fold_root',ROOT/'experiments/qwen25vl7b_adapter_leave_category_out/data/loco_folds','--nested_data_root',nested/'data/nested_folds','--run_root',nested/'outputs','--config_file',nested/'configs.json','--output_dir',OUT/'nested_selection')
        specifications=[
            ('qwen25vl7b_adapter_target_ablation','analyze_ablation.py',ROOT/'data/adapter_folds',ROOT/'experiments/qwen25vl7b_adapter_target_ablation/outputs'),
            ('qwen_visual_category_controls','analyze_four_objectives.py',ROOT/'data/adapter_folds',ROOT/'experiments/qwen25vl7b_adapter_target_ablation/outputs'),
            ('qwen25vl7b_adapter_leave_category_out','analyze_loco.py',ROOT/'experiments/qwen25vl7b_adapter_leave_category_out/data/loco_folds',ROOT/'experiments/qwen25vl7b_adapter_leave_category_out/outputs'),
            ('qwen25vl7b_adapter_nested_loco','analyze_nested_outer.py',ROOT/'experiments/qwen25vl7b_adapter_leave_category_out/data/loco_folds',ROOT/'experiments/qwen25vl7b_adapter_nested_loco/outputs')]
        for experiment,script,folds,outputs in specifications:
            run('experiments/'+experiment+'/code/'+script,'--fold_root',folds,'--run_root',outputs,'--image_root',ROOT/'images','--output_dir',OUT/experiment)
    print('Reproduction completed. Outputs:',OUT,flush=True)
