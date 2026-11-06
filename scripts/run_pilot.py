"""Run a small reproducible SD1.5/DeepCache pilot. All assets load locally."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['TORCH_HOME'] = str(ROOT / '.cache/torch')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import contextlib
import gc
import importlib.metadata
import json
import platform
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from DeepCache import DeepCacheSDHelper
from src.cache_policy import ScheduledDeepCache, uniform_schedule, paper_nonuniform_schedule, feature_interval_cost, calibrated_schedule

OUT = ROOT / 'results/pilot'
STEPS, BUDGET, CFG = 50, 17, 7.5


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def load_pipe():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    pipe = StableDiffusionPipeline.from_pretrained(ROOT / 'models/sd15', torch_dtype=torch.float16, variant='fp16', use_safetensors=True, safety_checker=None, requires_safety_checker=False, local_files_only=True)
    pipe = pipe.to('cuda')
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def cache_size(helper):
    if helper is None:
        return 0
    seen = {}
    def visit(x):
        if torch.is_tensor(x):
            s = x.untyped_storage()
            seen[(str(x.device), s.data_ptr())] = s.nbytes()
        elif isinstance(x, dict):
            for v in x.values(): visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x: visit(v)
    visit(helper.cached_output)
    return sum(seen.values())


class Runner:
    def __init__(self, pipe, output_dir=None):
        self.pipe = pipe
        self.scheduler_config = dict(pipe.scheduler.config)
        self.output_dir = Path(output_dir) if output_dir is not None else OUT
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.output_dir / 'measurements.jsonl'
        self.previous = {}
        if self.raw_path.exists():
            for line in self.raw_path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                self.previous[row['key']] = row

    @torch.inference_mode()
    def generate(self, item, seed, method, split='test', scheduler='ddim', steps=STEPS, refresh=None, official_interval=None, trace=False, save=True, resume=True):
        key = f"{split}/{method}/{item['image_id']}_{seed}"
        file = self.output_dir / 'images' / (key + '.png')
        if resume and key in self.previous and file.exists() and not trace:
            return self.previous[key], Image.open(file).convert('RGB'), None
        if scheduler == 'ddim':
            self.pipe.scheduler = DDIMScheduler.from_config(self.scheduler_config, clip_sample=False)
        else:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.scheduler_config, algorithm_type='dpmsolver++', solver_order=2)
        latent = torch.randn((1, 4, 64, 64), generator=torch.Generator('cpu').manual_seed(seed), dtype=torch.float32).half().cuda()
        helper = None
        if official_interval is not None:
            helper = DeepCacheSDHelper(self.pipe)
            helper.set_params(cache_interval=official_interval, cache_branch_id=0)
            helper.enable()
        elif refresh is not None:
            helper = ScheduledDeepCache(self.pipe, refresh, branch=0)
            helper.__enter__()
        events, features = [], []
        def pre_hook(module, args):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            events.append((start, end))
        def post_hook(module, args, output):
            events[-1][1].record()
        handles = [self.pipe.unet.register_forward_pre_hook(pre_hook), self.pipe.unet.register_forward_hook(post_hook)]
        if trace:
            def feature_hook(module, args, output):
                pooled = F.adaptive_avg_pool2d(output.float(), (8, 8))
                features.append(pooled.flatten(1).cpu())
            # For official branch 0, this resnet's output is reused in partial calls.
            handles.append(self.pipe.unet.up_blocks[-1].resnets[-2].register_forward_hook(feature_hook))
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        start = time.perf_counter()
        try:
            image = self.pipe(prompt=item['prompt'], num_inference_steps=steps, guidance_scale=CFG, latents=latent, output_type='pil', eta=0.0).images[0]
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            calls = len(events)
            if isinstance(helper, ScheduledDeepCache):
                refresh_log = helper.refresh_log.copy()
            elif helper:
                refresh_log = list(range(0, calls, official_interval))
            else:
                refresh_log = list(range(calls))
            row = {'key': key, 'split': split, 'method': method, 'image_id': item['image_id'], 'annotation_id': item.get('annotation_id'), 'prompt': item['prompt'], 'seed': seed, 'scheduler': scheduler, 'num_steps': steps, 'cfg': CFG, 'seconds': seconds, 'unet_seconds': sum(a.elapsed_time(b) for a,b in events) / 1000, 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20, 'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20, 'initial_allocated_mib': before/2**20, 'cache_storage_mib': cache_size(helper)/2**20, 'unet_calls': calls, 'full_refresh_count': len(refresh_log), 'refresh_steps': refresh_log, 'path': file.relative_to(ROOT).as_posix(), 'feature_trace': trace}
            if not np.isfinite(np.asarray(image)).all() or np.asarray(image).std() < 1:
                raise RuntimeError('Invalid or nearly constant generated image')
        finally:
            for h in handles: h.remove()
            if helper is not None:
                helper.disable()
        if save:
            file.parent.mkdir(parents=True, exist_ok=True)
            image.save(file)
            with self.raw_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            self.previous[key] = row
        log(f"{key}: {row['seconds']:.2f}s, peak {row['peak_allocated_mib']:.0f}MiB, refresh {row['full_refresh_count']}/{calls}")
        array = torch.stack(features).numpy() if features else None
        return row, image, array


def lpips_image(model, reference, output):
    def tensor(im):
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().unsqueeze(0)/127.5-1
    with torch.inference_mode():
        return float(model(tensor(reference), tensor(output)).item())


def smoke(runner):
    item = {'image_id': 'smoke', 'prompt': 'A photograph of a red bicycle leaning against a stone wall.'}
    _, base, _ = runner.generate(item, 42, 'base', split='smoke', steps=6, resume=False)
    _, identity, _ = runner.generate(item, 42, 'cache_interval1', split='smoke', steps=6, official_interval=1, resume=False)
    _, official, _ = runner.generate(item, 42, 'official_interval3', split='smoke', steps=6, official_interval=3, resume=False)
    _, scheduled, _ = runner.generate(item, 42, 'scheduled_interval3', split='smoke', steps=6, refresh=[0,3], resume=False)
    identity_diff = np.abs(np.asarray(base).astype(int)-np.asarray(identity).astype(int)).max()
    helper_diff = np.abs(np.asarray(official).astype(int)-np.asarray(scheduled).astype(int)).max()
    checks = {'interval1_pixel_max_error': int(identity_diff), 'scheduled_vs_official_pixel_max_error': int(helper_diff)}
    write_json(OUT / 'smoke_checks.json', checks)
    if identity_diff != 0 or helper_diff != 0:
        raise RuntimeError(f'Cache equivalence smoke check failed: {checks}')
    log('Official helper equivalence checks passed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['smoke', 'all'], default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT/'manifests/pilot.json').read_text(encoding='utf-8')) if args.stage=='all' else {}
    env = {'python': sys.version, 'platform': platform.platform(), 'gpu': torch.cuda.get_device_name(0), 'device_capability': torch.cuda.get_device_capability(0), 'cuda': torch.version.cuda, 'packages': {n: importlib.metadata.version(n) for n in ['torch','torchvision','diffusers','DeepCache','transformers','huggingface-hub','lpips']}, 'precision': 'fp16', 'attention': 'PyTorch SDPA default', 'tf32_matmul': True, 'resolution': [512,512], 'batch_size': 1, 'model_source': json.loads((ROOT/'models/sd15/source.json').read_text()), 'steps': STEPS, 'budget': BUDGET, 'calibration_feature': 'up_blocks[-1].resnets[-2] output, spatial mean pooled 8x8, CFG branches normalized separately'}
    write_json(OUT/'environment.json', env)
    runner = Runner(load_pipe())
    if not (OUT/'smoke_checks.json').exists() or args.stage=='smoke':
        smoke(runner)
    if args.stage=='smoke': return
    warm = {'image_id': 'warmup', 'prompt': manifest['calibration'][0]['prompt']}
    runner.generate(warm, 7, 'warmup', split='warmup', steps=5, save=False)

    calibration_start = time.perf_counter()
    traces = []
    for item in manifest['calibration']:
        trace_file = OUT/'calibration'/f"{item['image_id']}.npy"
        if trace_file.exists():
            traces.append(np.load(trace_file))
        else:
            _, _, trace = runner.generate(item, 42, 'full_trace', split='calibration', trace=True)
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(trace_file, trace)
            traces.append(trace)
    cost, drift = feature_interval_cost(traces)
    calibrated = calibrated_schedule(cost, BUDGET, max_gap=6)
    np.savez(OUT/'calibration/feature_cost.npz', cost=cost, drift=drift)
    calibration_seconds = time.perf_counter()-calibration_start
    # Real-valued implementation of the paper schedule; two validation candidates.
    nonuniform_candidates = {f'paper_c{c}_p14': paper_nonuniform_schedule(STEPS,BUDGET,c,1.4) for c in [15,20]}
    policies = {'uniform': uniform_schedule(STEPS,3), 'calibrated': calibrated, **nonuniform_candidates}
    for schedule in policies.values():
        assert len(schedule)==BUDGET and schedule[0]==0
    write_json(OUT/'policies.json', {'steps':STEPS,'budget':BUDGET,'calibration_seconds_this_invocation':calibration_seconds,'schedules':policies,'paper_implementation_note':'Signed-real-power Eq7 adaptation; not bit-exact legacy pipeline reproduction','surrogate':'mean normalized L1 of spatially pooled actual cache-boundary features on uncached trajectories'})

    import lpips
    lpips_model = lpips.LPIPS(net='alex').eval().cpu()
    errors = {name:[] for name in policies}
    validation_times = {'uniform':[], 'dpmpp20':[]}
    for item in manifest['validation']:
        _, ref, _ = runner.generate(item, 2026, 'ddim50', split='validation')
        for name, schedule in policies.items():
            row, im, _ = runner.generate(item,2026,name,split='validation',refresh=schedule)
            errors[name].append(lpips_image(lpips_model,ref,im))
            if name=='uniform': validation_times[name].append(row['seconds'])
        row, _, _ = runner.generate(item,2026,'dpmpp20',split='validation',scheduler='dpmpp',steps=20)
        validation_times['dpmpp20'].append(row['seconds'])
    selected_paper = min(nonuniform_candidates, key=lambda k:np.mean(errors[k]))
    # DPM-Solver++ step selection uses only latency calibration, not test quality.
    dpm_steps = max(5,min(45,round(20*np.median(validation_times['uniform'])/np.median(validation_times['dpmpp20']))))
    selection = {'validation_lpips':{k:float(np.mean(v)) for k,v in errors.items()},'selected_paper':selected_paper,'dpmpp_steps':dpm_steps,'dpmpp_selection':'rounded 20 * validation median uniform latency / validation median DPM++20 latency; test time need not match exactly','calibrated_test_parameters_frozen':True}
    write_json(OUT/'selection.json',selection)
    del lpips_model
    gc.collect()
    log(f'Selection frozen: {selection}')

    methods = ['ddim50','uniform','paper_nonuniform','calibrated','dpmpp_matched']
    jobs = [(item,seed,method) for item in manifest['test'] for seed in manifest['test_seeds'] for method in methods]
    random.Random(6132).shuffle(jobs)
    for item, seed, method in jobs:
        if method=='ddim50': runner.generate(item,seed,method)
        elif method=='dpmpp_matched': runner.generate(item,seed,method,scheduler='dpmpp',steps=dpm_steps)
        else:
            schedule = policies[selected_paper] if method=='paper_nonuniform' else policies[method]
            runner.generate(item,seed,method,refresh=schedule)
    write_json(OUT/'generation_complete.json', {'test_images':len(jobs),'expected_pairs':len(manifest['test'])*len(manifest['test_seeds']),'methods':methods})
    log('Pilot generation complete')


if __name__=='__main__':
    main()
