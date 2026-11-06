"""Second iteration: diagnose cache error, validate two revised costs, freeze and test."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
os.environ['TORCH_HOME']=str(ROOT/'.cache/torch')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import csv
import gc
import hashlib
import json
import random
import time
import numpy as np
import torch
from scipy.stats import spearmanr
from scripts.run_pilot import Runner,load_pipe,write_json,log,lpips_image
from src.cache_policy import ScheduledDeepCache,feature_interval_cost,calibrated_schedule,uniform_schedule,paper_nonuniform_schedule
from src.error_calibration import guided_prediction,ddim_epsilon_coefficient,interval_cost_from_errors,residualized_rank_correlation

OUT=ROOT/'results/round2'
STEPS,BUDGET,MAX_GAP=50,17,6


def prepare_manifest():
    previous=json.loads((ROOT/'manifests/pilot.json').read_text(encoding='utf-8'))
    raw=ROOT/'data/raw/captions_val2014.json'
    if hashlib.sha256(raw.read_bytes()).hexdigest()!=previous['source_sha256']:
        raise RuntimeError('Caption source has changed')
    data=json.loads(raw.read_text(encoding='utf-8'))
    captions={}
    for a in sorted(data['annotations'],key=lambda x:x['id']):
        captions.setdefault(a['image_id'],{'image_id':a['image_id'],'annotation_id':a['id'],'prompt':a['caption'].strip()})
    ids=sorted(captions)
    random.Random(6132).shuffle(ids)
    old_ids={r['image_id'] for split in ['calibration','validation','test'] for r in previous[split]}
    available=[i for i in ids if i not in old_ids]
    manifest={k:previous[k] for k in ['source','dataset','source_sha256','selection']}
    manifest.update({'split_seed':6132,'excluded_round1_image_ids':sorted(old_ids),'calibration':[captions[i] for i in available[:8]],'validation':[captions[i] for i in available[8:16]],'test':[captions[i] for i in available[16:32]],'test_seeds':[2026,6132]})
    groups=[{r['image_id'] for r in manifest[s]} for s in ['calibration','validation','test']]
    assert not any(groups[i]&groups[j] for i in range(3) for j in range(i))
    assert not old_ids.intersection(set.union(*groups))
    path=ROOT/'manifests/round2.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=manifest:
        raise RuntimeError('Refusing to overwrite a different round2 manifest')
    write_json(path,manifest)
    return manifest


@torch.inference_mode()
def collect_trace(runner,item):
    path=OUT/'diagnostics'/f"trace_{item['image_id']}.npz"
    if path.exists():
        with np.load(path) as z:return {k:z[k] for k in z.files}
    inputs,predictions,timesteps,context=[],[],[],[]
    def before(module,args,kwargs):
        inputs.append(args[0].detach().cpu())
        timesteps.append(int(args[1]))
        if not context:context.append(kwargs['encoder_hidden_states'].detach().cpu())
    def after(module,args,output):
        predictions.append((output[0] if isinstance(output,tuple) else output.sample).detach().cpu())
    handles=[runner.pipe.unet.register_forward_pre_hook(before,with_kwargs=True),runner.pipe.unet.register_forward_hook(after)]
    try:
        _,_,pooled=runner.generate(item,42,'reference_trace',split='calibration',trace=True,resume=False)
    finally:
        for h in handles:h.remove()
    result={'inputs':torch.stack(inputs).numpy(),'predictions':torch.stack(predictions).numpy(),'timesteps':np.array(timesteps),'context':context[0].numpy(),'pooled':pooled}
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,**result)
    return result


@torch.inference_mode()
def probe_trace(runner,item,trace):
    path=OUT/'diagnostics'/f"probes_{item['image_id']}.npz"
    if path.exists():
        with np.load(path) as z:return {k:z[k] for k in z.files}
    pipe=runner.pipe
    xs=torch.from_numpy(trace['inputs']).cuda()
    references=torch.from_numpy(trace['predictions']).cuda()
    ts=torch.from_numpy(trace['timesteps']).cuda()
    context=torch.from_numpy(trace['context']).cuda()
    # A resumed process may not have run the reference pipeline this invocation.
    from diffusers import DDIMScheduler
    pipe.scheduler=DDIMScheduler.from_config(runner.scheduler_config,clip_sample=False)
    pipe.scheduler.set_timesteps(STEPS,device='cuda')
    coefficients=np.array([ddim_epsilon_coefficient(pipe.scheduler,t) for t in trace['timesteps']])
    noise=np.full((STEPS,STEPS),np.nan)
    relative=np.full_like(noise,np.nan)
    feature=np.full_like(noise,np.nan)
    prime_max=0.;probe_calls=0
    g=trace['pooled'].astype(np.float32)
    for a in range(STEPS):
        with ScheduledDeepCache(pipe,[0],branch=0):
            full=pipe.unet(xs[a],ts[a],encoder_hidden_states=context,return_dict=False)[0]
            prime_max=max(prime_max,float((full-references[a]).abs().max()))
            noise[a,a]=relative[a,a]=feature[a,a]=0.
            for i in range(a+1,min(a+MAX_GAP,STEPS)):
                stale=pipe.unet(xs[i],ts[i],encoder_hidden_states=context,return_dict=False)[0]
                exact=guided_prediction(references[i])
                diff=guided_prediction(stale)-exact
                value=float(diff.square().mean())
                noise[a,i]=value
                relative[a,i]=value/(float(exact.square().mean())+1e-8)
                feature[a,i]=float((np.abs(g[i]-g[a]).mean(axis=-1)/(np.abs(g[i]).mean(axis=-1)+1e-6)).mean())
                probe_calls+=1
    torch.cuda.synchronize()
    if prime_max>1e-4:raise RuntimeError(f'Full-path replay mismatch: {prime_max}')
    result={'noise_mse':noise,'relative_noise_mse':relative,'step_mse':noise*coefficients[None,:]**2,'feature_drift':feature,'epsilon_coefficients':coefficients,'prime_max_error':np.array(prime_max),'partial_probe_calls':np.array(probe_calls)}
    np.savez(path,**result)
    log(f"Diagnosed COCO {item['image_id']}: full replay max error {prime_max:.6g}, {probe_calls} partial probes")
    return result


def diagnose(runner,manifest):
    start=time.perf_counter()
    traces,probes=[],[]
    for item in manifest['calibration']:
        trace=collect_trace(runner,item)
        traces.append(trace['pooled'])
        probes.append(probe_trace(runner,item,trace))
    feature_cost,_=feature_interval_cost(traces)
    # Every prompt has the same probe mask; ordinary mean preserves unmeasured
    # NaNs without warning on intentionally unsupported long intervals.
    noise=np.mean(np.stack([p['noise_mse'] for p in probes]),axis=0)
    step=np.mean(np.stack([p['step_mse'] for p in probes]),axis=0)
    noise_cost=interval_cost_from_errors(noise,MAX_GAP)
    step_cost=interval_cost_from_errors(step,MAX_GAP)
    schedules={'uniform':uniform_schedule(STEPS,3),'feature_calibrated':calibrated_schedule(feature_cost,BUDGET,MAX_GAP),'noise_candidate':calibrated_schedule(noise_cost,BUDGET,MAX_GAP),'step_candidate':calibrated_schedule(step_cost,BUDGET,MAX_GAP)}
    schedules.update({f'paper_c{c}':paper_nonuniform_schedule(STEPS,BUDGET,c,1.4) for c in [15,20]})
    for s in schedules.values():assert len(s)==BUDGET and s[0]==0
    rows=[]
    for item,p in zip(manifest['calibration'],probes):
        for a in range(STEPS):
            for i in range(a+1,min(a+MAX_GAP,STEPS)):
                rows.append({'image_id':item['image_id'],'refresh_at':a,'target_step':i,'age':i-a,'feature_drift':float(p['feature_drift'][a,i]),'noise_mse':float(p['noise_mse'][a,i]),'relative_noise_mse':float(p['relative_noise_mse'][a,i]),'step_mse':float(p['step_mse'][a,i])})
    with (OUT/'diagnostic_probes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    x=np.array([r['feature_drift'] for r in rows]);target=np.array([r['target_step'] for r in rows]);ages=np.array([r['age'] for r in rows])
    associations={}
    for field in ['noise_mse','relative_noise_mse','step_mse']:
        y=np.array([r[field] for r in rows])
        associations[field]={'overall_spearman':float(spearmanr(x,y).statistic),'rank_correlation_after_step_and_age_fixed_effects':residualized_rank_correlation(x,y,target,ages),'per_age_spearman':{str(age):float(spearmanr(x[ages==age],y[ages==age]).statistic) for age in range(1,MAX_GAP)}}
    write_json(OUT/'diagnostic_summary.json',{'probe_cases':len(rows),'calibration_prompts':len(traces),'seconds_this_invocation':time.perf_counter()-start,'full_replay_max_error':max(float(p['prime_max_error']) for p in probes),'associations':associations,'warning':'Counterfactuals use the uncached reference latent trajectory; they are local errors, not full rollout/ground-truth image quality. Fixed effects are exploratory, not causal.'})
    write_json(OUT/'policies.json',{'steps':STEPS,'budget':BUDGET,'max_gap':MAX_GAP,'schedules':schedules,'candidate_rule':'Two predefined new costs: guided-noise MSE and DDIM one-step latent-update MSE; select using validation LPIPS only','implementation':'Same official helper branch 0; signed-real-power paper schedule adaptation retained from round1'})
    np.savez(OUT/'diagnostics/aggregate_costs.npz',feature_cost=feature_cost,noise_cost=noise_cost,step_cost=step_cost,noise_mse=noise,step_mse=step,epsilon_coefficients=probes[0]['epsilon_coefficients'])
    log('Diagnosis and candidate calibration complete')
    return schedules


def run_validation_and_test(runner,manifest,schedules):
    import lpips
    metric=lpips.LPIPS(net='alex').eval().cpu()
    errors={name:[] for name in schedules if name!='uniform'}
    uniform_times=[];dpm_times=[]
    for item in manifest['validation']:
        _,reference,_=runner.generate(item,2026,'ddim50',split='validation')
        row,_,_=runner.generate(item,2026,'uniform',split='validation',refresh=schedules['uniform'])
        uniform_times.append(row['seconds'])
        for name in errors:
            _,image,_=runner.generate(item,2026,name,split='validation',refresh=schedules[name])
            errors[name].append(lpips_image(metric,reference,image))
        row,_,_=runner.generate(item,2026,'dpmpp20',split='validation',scheduler='dpmpp',steps=20)
        dpm_times.append(row['seconds'])
    del metric;gc.collect()
    means={k:float(np.mean(v)) for k,v in errors.items()}
    chosen=min(['noise_candidate','step_candidate'],key=lambda k:means[k])
    paper=min(['paper_c15','paper_c20'],key=lambda k:means[k])
    dpm_steps=max(5,min(45,round(20*np.median(uniform_times)/np.median(dpm_times))))
    selection={'validation_lpips':means,'selected_new_cost':chosen,'selected_paper':paper,'dpmpp_steps':dpm_steps,'frozen_before_test':True,'criterion':'Mean validation LPIPS to identical-latent DDIM50; CLIP test results do not influence selection'}
    write_json(OUT/'selection.json',selection)
    log(f'Frozen selection: {selection}')
    methods=['ddim50','uniform','paper_nonuniform','feature_calibrated','calibrated','dpmpp_matched']
    jobs=[(item,seed,m) for item in manifest['test'] for seed in manifest['test_seeds'] for m in methods]
    random.Random(261106).shuffle(jobs)
    for item,seed,method in jobs:
        if method=='ddim50':runner.generate(item,seed,method)
        elif method=='dpmpp_matched':runner.generate(item,seed,method,scheduler='dpmpp',steps=dpm_steps)
        else:
            key=chosen if method=='calibrated' else paper if method=='paper_nonuniform' else method
            runner.generate(item,seed,method,refresh=schedules[key])
    write_json(OUT/'generation_complete.json',{'test_images':len(jobs),'expected_pairs':len(manifest['test'])*len(manifest['test_seeds']),'methods':methods,'manifest':'manifests/round2.json','round':2})
    log('Round 2 held-out generation complete')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['diagnose','all'],default='all');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=prepare_manifest()
    environment=json.loads((ROOT/'results/pilot/environment.json').read_text(encoding='utf-8'))
    environment.update({'round':2,'manifest':'manifests/round2.json','calibration_prompts':8,'validation_prompts':8,'test_prompts':16,'test_seeds':[2026,6132],'calibration_probe_max_age':5,'new_test_ids_disjoint_from_all_round1_ids':True})
    write_json(OUT/'environment.json',environment)
    labels={'ddim50':'DDIM 50 (reference)','uniform':'DeepCache uniform','paper_nonuniform':'DeepCache paper nonuniform','feature_calibrated':'Feature-drift calibration (old proxy)','calibrated':'Error calibration (new proxy)','dpmpp_matched':'DPM-Solver++ (latency calibrated)'}
    write_json(OUT/'experiment_config.json',{'labels':labels,'round':2,'manifest':'manifests/round2.json','policy_candidates_predefined':['noise_candidate','step_candidate'],'test_methods_predefined':list(labels)})
    runner=Runner(load_pipe(),OUT)
    runner.generate(manifest['calibration'][0],7,'warmup',split='warmup',steps=5,save=False)
    if not (OUT/'diagnostic_summary.json').exists():schedules=diagnose(runner,manifest)
    else:schedules=json.loads((OUT/'policies.json').read_text(encoding='utf-8'))['schedules']
    if args.stage=='all':run_validation_and_test(runner,manifest,schedules)


if __name__=='__main__':main()
