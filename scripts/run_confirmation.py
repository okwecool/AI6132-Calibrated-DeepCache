"""Frozen-policy confirmation on 100 unseen prompts plus predefined budget ablation."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
os.environ['TORCH_HOME']=str(ROOT/'.cache/torch')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import gc
import hashlib
import json
import random
import shutil
import time
import numpy as np
from scripts.run_pilot import Runner,load_pipe,write_json,log
from src.cache_policy import calibrated_schedule,uniform_schedule

OUT=ROOT/'results/confirmation'
SOURCE=ROOT/'results/round2'


def prepare():
    r2=json.loads((ROOT/'manifests/round2.json').read_text(encoding='utf-8'))
    pilot=json.loads((ROOT/'manifests/pilot.json').read_text(encoding='utf-8'))
    raw=ROOT/'data/raw/captions_val2014.json'
    if hashlib.sha256(raw.read_bytes()).hexdigest()!=r2['source_sha256']:
        raise RuntimeError('Caption source changed')
    excluded={r['image_id'] for m in [pilot,r2] for split in ['calibration','validation','test'] for r in m[split]}
    data=json.loads(raw.read_text(encoding='utf-8'))
    captions={}
    for a in sorted(data['annotations'],key=lambda x:x['id']):
        captions.setdefault(a['image_id'],{'image_id':a['image_id'],'annotation_id':a['id'],'prompt':a['caption'].strip()})
    ids=sorted(captions);random.Random(6132).shuffle(ids)
    chosen=[i for i in ids if i not in excluded][:100]
    manifest={k:r2[k] for k in ['source','dataset','source_sha256','selection','split_seed']}
    manifest.update({'calibration':r2['calibration'],'validation':r2['validation'],'test':[captions[i] for i in chosen],'test_seeds':[2026],'ablation_image_ids':chosen[:16],'excluded_previous_image_ids':sorted(excluded),'protocol':{'kind':'frozen_confirmation','calibration_and_validation_reused_from':'manifests/round2.json','primary_comparison':'calibrated minus uniform, mean LPIPS to identical-latent DDIM50','uncertainty':'paired prompt-cluster bootstrap, two-sided 95% interval','secondary_metrics':['CLIP cosine','latency','memory'],'budgets':[10,17,25],'budget_ablation_subset':'first 16 test prompts before any new outputs are inspected','no_new_parameter_selection':True}})
    assert len(chosen)==100 and not set(chosen)&excluded
    path=ROOT/'manifests/confirmation.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=manifest:
        raise RuntimeError('Refusing to overwrite a different confirmation manifest')
    write_json(path,manifest)
    original=json.loads((SOURCE/'policies.json').read_text(encoding='utf-8'))
    selected=json.loads((SOURCE/'selection.json').read_text(encoding='utf-8'))
    if selected['selected_new_cost']!='step_candidate':
        raise RuntimeError('Frozen experiment expects the round2 step-MSE selection')
    with np.load(SOURCE/'diagnostics/aggregate_costs.npz') as z:
        cost=z['step_cost'].copy()
    budgets={}
    for budget,interval in [(10,5),(17,3),(25,2)]:
        budgets[str(budget)]={'uniform':uniform_schedule(50,interval),'calibrated':calibrated_schedule(cost,budget,max_gap=6)}
        for schedule in budgets[str(budget)].values():assert len(schedule)==budget and schedule[0]==0
    assert budgets['17']['calibrated']==original['schedules'][selected['selected_new_cost']]
    policies={**original,'budgets':budgets,'frozen_from':'results/round2/policies.json','new_calibration_seconds':0,'note':'Budget 10/25 schedules derive only from frozen round2 calibration costs; no confirmation outcomes used.'}
    write_json(OUT/'policies.json',policies)
    write_json(OUT/'selection.json',{**selected,'reused_frozen_selection':True,'no_selection_on_confirmation_data':True})
    (OUT/'diagnostics').mkdir(parents=True,exist_ok=True)
    shutil.copyfile(SOURCE/'diagnostics/aggregate_costs.npz',OUT/'diagnostics/aggregate_costs.npz')
    labels={'ddim50':'DDIM 50 (reference)','uniform':'DeepCache uniform','paper_nonuniform':'DeepCache paper nonuniform','calibrated':'Frozen error-calibrated policy','dpmpp_matched':'Frozen DPM-Solver++ 20 steps'}
    write_json(OUT/'experiment_config.json',{'labels':labels,'kind':'frozen_confirmation','manifest':'manifests/confirmation.json','primary_comparison':'calibrated minus uniform LPIPS','test_methods_predefined':list(labels),'ablation_budgets_predefined':[10,17,25],'ablation_prompts':chosen[:16],'frozen_source_policy_sha256':hashlib.sha256((SOURCE/'policies.json').read_bytes()).hexdigest(),'frozen_source_selection_sha256':hashlib.sha256((SOURCE/'selection.json').read_bytes()).hexdigest()})
    env=json.loads((SOURCE/'environment.json').read_text(encoding='utf-8'))
    env.update({'round':3,'kind':'frozen_confirmation','manifest':'manifests/confirmation.json','test_prompts':100,'test_seeds':[2026],'new_calibration':False,'new_validation':False,'ablation_prompts':16,'ablation_budgets':[10,17,25]})
    write_json(OUT/'environment.json',env)
    return manifest,policies,selected,list(labels)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest,policies,selected,methods=prepare()
    start=time.perf_counter()
    runner=Runner(load_pipe(),OUT)
    runner.generate(manifest['calibration'][0],7,'warmup',split='warmup',steps=5,save=False)
    jobs=[(item,2026,m) for item in manifest['test'] for m in methods]
    random.Random(361106).shuffle(jobs)
    schedules={'uniform':policies['budgets']['17']['uniform'],'paper_nonuniform':policies['schedules'][selected['selected_paper']],'calibrated':policies['budgets']['17']['calibrated']}
    for index,(item,seed,method) in enumerate(jobs,1):
        if method=='ddim50':runner.generate(item,seed,method)
        elif method=='dpmpp_matched':runner.generate(item,seed,method,scheduler='dpmpp',steps=selected['dpmpp_steps'])
        else:runner.generate(item,seed,method,refresh=schedules[method])
        if index%25==0:log(f'Confirmation progress {index}/{len(jobs)}')
    write_json(OUT/'generation_complete.json',{'test_images':len(jobs),'expected_pairs':100,'methods':methods,'manifest':'manifests/confirmation.json','kind':'frozen_confirmation'})
    subset=manifest['test'][:16]
    ablation=[(item,budget,method) for item in subset for budget in [10,25] for method in ['uniform','calibrated']]
    random.Random(361107).shuffle(ablation)
    for item,budget,method in ablation:
        runner.generate(item,2026,f'{method}_b{budget}',split='ablation',refresh=policies['budgets'][str(budget)][method])
    write_json(OUT/'ablation_complete.json',{'new_images':len(ablation),'subset_prompts':16,'budgets':[10,17,25],'reuse_test_budget17_and_reference':True,'total_generation_seconds_this_invocation':time.perf_counter()-start})
    log('Frozen confirmation and budget ablation generation complete')


if __name__=='__main__':main()
