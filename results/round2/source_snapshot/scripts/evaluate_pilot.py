"""Evaluate held-out pilot pairs and render standalone report figures."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HOME'] = str(ROOT/'.cache/huggingface')
os.environ['TORCH_HOME'] = str(ROOT/'.cache/torch')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import csv
import json
import math
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import lpips
from transformers import CLIPModel, CLIPProcessor
from scripts.run_pilot import lpips_image, write_json

OUT = ROOT/'results/pilot'
METHODS = ['ddim50','uniform','paper_nonuniform','calibrated','dpmpp_matched']
LABELS = {'ddim50':'DDIM 50 (reference)','uniform':'DeepCache uniform','paper_nonuniform':'DeepCache nonuniform (paper)','calibrated':'DeepCache calibrated (ours)','dpmpp_matched':'DPM-Solver++ (latency calibrated)'}


def prompt_cluster_ci(rows, field, method_a='calibrated', method_b='uniform'):
    by = defaultdict(dict)
    for r in rows:
        by[(r['image_id'],r['seed'])][r['method']] = r
    groups = defaultdict(list)
    for (image_id, seed), pair in by.items():
        groups[image_id].append(pair[method_a][field]-pair[method_b][field])
    values = np.array([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(6132)
    boot = rng.choice(values,size=(5000,len(values)),replace=True).mean(axis=1)
    return {'mean_difference':float(values.mean()),'cluster_bootstrap_95pct_ci':np.quantile(boot,[.025,.975]).tolist(),'clusters':len(values),'comparison':f'{method_a} minus {method_b}','note':'Exploratory paired prompt-cluster bootstrap; very small pilot, not confirmatory evidence'}


def main():
    global OUT,METHODS,LABELS
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',default='results/pilot')
    parser.add_argument('--plots-only',action='store_true',help='Reuse saved quality scores; only redraw figures')
    args=parser.parse_args()
    OUT=(ROOT/args.run_dir).resolve()
    complete=json.loads((OUT/'generation_complete.json').read_text(encoding='utf-8'))
    METHODS=complete['methods']
    if (OUT/'experiment_config.json').exists():
        LABELS.update(json.loads((OUT/'experiment_config.json').read_text(encoding='utf-8'))['labels'])
    manifest=json.loads((ROOT/complete.get('manifest','manifests/pilot.json')).read_text(encoding='utf-8'))
    if args.plots_only:
        saved=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
        render_plots(json.loads((OUT/'evaluated_rows.json').read_text(encoding='utf-8')),saved['methods'])
        return
    torch.set_num_threads(4)
    complete = json.loads((OUT/'generation_complete.json').read_text())
    unique = {}
    for line in (OUT/'measurements.jsonl').read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r['split']=='test': unique[r['key']] = r
    rows = sorted(unique.values(), key=lambda r:(r['image_id'],r['seed'],r['method']))
    if len(rows)!=complete['test_images']: raise RuntimeError('Incomplete test outputs')
    clip = CLIPModel.from_pretrained(ROOT/'models/clip',local_files_only=True).eval().to('cuda')
    processor = CLIPProcessor.from_pretrained(ROOT/'models/clip',local_files_only=True)
    with torch.inference_mode():
        for start in range(0,len(rows),8):
            group = rows[start:start+8]
            ims = [Image.open(ROOT/r['path']).convert('RGB') for r in group]
            batch = processor(text=[r['prompt'] for r in group],images=ims,return_tensors='pt',padding=True,truncation=True).to('cuda')
            output = clip(**batch)
            scores = (F.normalize(output.image_embeds,dim=-1)*F.normalize(output.text_embeds,dim=-1)).sum(-1).cpu().tolist()
            for r,s in zip(group,scores): r['clip_cosine']=s
    del clip,processor
    torch.cuda.empty_cache()
    metric = lpips.LPIPS(net='alex').eval().cpu()
    references = {(r['image_id'],r['seed']):r for r in rows if r['method']=='ddim50'}
    for r in rows:
        reference = Image.open(ROOT/references[(r['image_id'],r['seed'])]['path']).convert('RGB')
        image = Image.open(ROOT/r['path']).convert('RGB')
        r['lpips_to_ddim50'] = lpips_image(metric,reference,image)
        mse = float(np.mean((np.asarray(reference,dtype=np.float64)-np.asarray(image,dtype=np.float64))**2))
        r['psnr_to_ddim50'] = 10*math.log10(255**2/mse) if mse>0 else None
    write_json(OUT/'evaluated_rows.json',rows)
    with (OUT/'evaluated_rows.csv').open('w',encoding='utf-8-sig',newline='') as f:
        flat = [{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in rows]
        writer=csv.DictWriter(f,fieldnames=list(flat[0]))
        writer.writeheader();writer.writerows(flat)
    reference_time = np.median([r['seconds'] for r in rows if r['method']=='ddim50'])
    summary = {}
    for method in METHODS:
        subset=[r for r in rows if r['method']==method]
        summary[method]={'n':len(subset),'median_seconds':float(np.median([r['seconds'] for r in subset])),'mean_seconds':float(np.mean([r['seconds'] for r in subset])),'median_unet_seconds':float(np.median([r['unet_seconds'] for r in subset])),'max_peak_allocated_mib':max(r['peak_allocated_mib'] for r in subset),'max_peak_reserved_mib':max(r['peak_reserved_mib'] for r in subset),'mean_clip_cosine':float(np.mean([r['clip_cosine'] for r in subset])),'mean_lpips_to_ddim50':float(np.mean([r['lpips_to_ddim50'] for r in subset])),'full_refresh_count':subset[0]['full_refresh_count'],'unet_calls':subset[0]['unet_calls']}
        summary[method]['speedup_vs_ddim50']=float(reference_time/summary[method]['median_seconds'])
    comparators=['uniform','paper_nonuniform']+(['feature_calibrated'] if 'feature_calibrated' in METHODS else [])
    differences={f'calibrated_minus_{b}':{field:prompt_cluster_ci(rows,field,'calibrated',b) for field in ['lpips_to_ddim50','clip_cosine','seconds']} for b in comparators}
    write_json(OUT/'summary.json',{'methods':summary,'paired_differences':differences,'clip_metric':'Raw cosine of normalized CLIP ViT-B/32 embeddings, no scaling; not directly comparable with other CLIPScore protocols','quality_reference':'DDIM50 output with identical prompt and initial latent, not COCO image ground truth','timing':'One measured generation per prompt/seed/method after warmup; randomized order; no repeat-based latency confidence claim','sample_size':{'test_prompts':len(manifest['test']),'test_seeds':len(manifest['test_seeds']),'calibration_prompts':len(manifest['calibration']),'validation_prompts':len(manifest['validation'])},'manifest':complete.get('manifest','manifests/pilot.json')})

    render_plots(rows,summary)
    print(json.dumps(summary,indent=2),flush=True)


def render_plots(rows,summary):
    complete=json.loads((OUT/'generation_complete.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/complete.get('manifest','manifests/pilot.json')).read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(13,5.1),layout='constrained')
    colors=plt.get_cmap('tab10').colors
    for i,m in enumerate(METHODS):
        s=summary[m]
        axes[0].scatter(s['median_seconds'],s['mean_lpips_to_ddim50'],color=colors[i],s=65,label=LABELS[m])
        axes[1].scatter(s['median_seconds'],s['mean_clip_cosine'],color=colors[i],s=65,label=LABELS[m])
    axes[0].set(xlabel='Median end-to-end generation time (s)',ylabel='Mean LPIPS to DDIM50 (lower = closer)',title='Approximation error vs. latency')
    axes[1].set(xlabel='Median end-to-end generation time (s)',ylabel='Mean CLIP text-image cosine (higher)',title='Text alignment vs. latency')
    for ax in axes: ax.grid(alpha=.25)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,fontsize=8)
    fig.suptitle(f"RTX 5060 Laptop 8GB | SD1.5 512px | {len(manifest['test'])} held-out COCO captions x {len(manifest['test_seeds'])} seeds",fontsize=11)
    fig.savefig(OUT/'quality_latency.png',dpi=170);plt.close(fig)

    saved=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    comparators=['uniform','paper_nonuniform']+(['feature_calibrated'] if 'feature_calibrated' in METHODS else [])
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    for ax,field,title in [(axes[0],'lpips_to_ddim50','LPIPS difference (negative = closer to reference)'),(axes[1],'clip_cosine','CLIP cosine difference (positive = higher alignment)')]:
        for i,base in enumerate(comparators):
            d=saved['paired_differences'][f'calibrated_minus_{base}'][field]
            mean=d['mean_difference'];lo,hi=d['cluster_bootstrap_95pct_ci']
            ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=4,color='tab:blue')
        ax.axvline(0,color='gray',linestyle='--',linewidth=1)
        ax.set(yticks=list(range(len(comparators))),yticklabels=[{'uniform':'Uniform','paper_nonuniform':'Paper nonuniform','feature_calibrated':'Old feature proxy'}[k] for k in comparators],xlabel=title)
        ax.set_ylim(-.5,len(comparators)-.5);ax.grid(axis='x',alpha=.25)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.suptitle(f"Calibrated policy minus baseline | {len(manifest['test'])} prompt clusters | exploratory 95% intervals",fontsize=11)
    fig.savefig(OUT/'paired_differences.png',dpi=170);plt.close(fig)

    policies=json.loads((OUT/'policies.json').read_text())['schedules']
    selection=json.loads((OUT/'selection.json').read_text())
    fig,axes=plt.subplots(2,1,figsize=(10,5.4),layout='constrained',sharex=True)
    if (OUT/'diagnostics/aggregate_costs.npz').exists():
        z=np.load(OUT/'diagnostics/aggregate_costs.npz')
        # Fixed age=1 makes denoising phases comparable without mixing cache ages.
        for label,values in [('Noise MSE',np.array([z['noise_mse'][i-1,i] for i in range(1,50)])),('DDIM update MSE',np.array([z['step_mse'][i-1,i] for i in range(1,50)]))]:
            axes[0].plot(np.arange(1,50),values/(values.max()+1e-12),label=label)
        axes[0].legend()
        axes[0].set(ylabel='Each curve / its maximum',title='Measured one-step cache error on calibration inputs (age=1)')
        shown=[('Uniform',policies['uniform']),('Paper nonuniform',policies[selection['selected_paper']]),('Feature proxy',policies['feature_calibrated']),('Error proxy',policies[selection['selected_new_cost']])]
    else:
        drift=np.load(OUT/'calibration/feature_cost.npz')['drift']
        axes[0].plot(np.arange(1,50),[drift[i-1,i] for i in range(1,50)])
        axes[0].set(ylabel='Normalized feature drift',title='Calibration-set one-step feature drift (pooled proxy)')
        shown=[('Uniform',policies['uniform']),('Paper nonuniform',policies[selection['selected_paper']]),('Calibrated',policies['calibrated'])]
    for y,(label,steps) in enumerate(shown):
        axes[1].scatter(steps,[y]*len(steps),marker='|',s=200,label=label)
    axes[1].set(yticks=list(range(len(shown))),yticklabels=[label for label,_ in shown],xlabel='Denoising call index (50 total)',title='Refresh positions: exactly 17 per schedule')
    axes[1].set_ylim(-.35,len(shown)-.65)
    fig.savefig(OUT/'refresh_schedules.png',dpi=170);plt.close(fig)

    # Fixed first four test images, chosen before looking at quality, not cherry-picked.
    font_path='C:/Windows/Fonts/arial.ttf'
    font=ImageFont.truetype(font_path,16)
    small=ImageFont.truetype(font_path,13)
    lookup={(r['image_id'],r['seed'],r['method']):r for r in rows}
    groups=[manifest['test'][i:i+4] for i in range(0,len(manifest['test']),4)]
    for page,items in enumerate(groups,start=1):
        thumb=256; caption_h=48; label_h=40
        canvas=Image.new('RGB',(thumb*len(METHODS),(thumb+caption_h)*len(items)+label_h),'white')
        draw=ImageDraw.Draw(canvas)
        short={'ddim50':'DDIM50','uniform':'Uniform cache','paper_nonuniform':'Paper nonuniform','feature_calibrated':'Feature proxy (old)','calibrated':'Error proxy (new)' if 'feature_calibrated' in METHODS else 'Calibrated (ours)','dpmpp_matched':'DPM-Solver++'}
        headers=[short[m] for m in METHODS]
        for i,label in enumerate(headers):draw.text((i*thumb+8,10),label,font=font,fill='black')
        for j,item in enumerate(items):
            y=label_h+j*(thumb+caption_h)
            for i,m in enumerate(METHODS):
                r=lookup[(item['image_id'],manifest['test_seeds'][0],m)]
                im=Image.open(ROOT/r['path']).convert('RGB').resize((thumb,thumb),Image.Resampling.LANCZOS)
                canvas.paste(im,(i*thumb,y))
                draw.text((i*thumb+5,y+thumb+2),f"{r['seconds']:.2f}s | CLIP {r['clip_cosine']:.3f} | LPIPS {r['lpips_to_ddim50']:.3f}",font=small,fill='black')
            draw.text((5,y+thumb+23),f"COCO {item['image_id']} | {item['prompt'][:165]}",font=small,fill='black')
        canvas.save(OUT/f'comparison_{page}.jpg',quality=93)


if __name__=='__main__':
    main()
