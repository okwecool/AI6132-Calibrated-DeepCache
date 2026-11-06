"""Preserve interrupted runs and remeasure only demonstrably paused timings."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import hashlib
import numpy as np
from PIL import Image
from scripts.run_pilot import Runner,load_pipe,write_json

OUT=ROOT/'results/confirmation'


def main():
    rows=[json.loads(s) for s in (OUT/'measurements.jsonl').read_text(encoding='utf-8').splitlines()]
    affected=[r for r in rows if r['split'] in ['test','ablation'] and r['seconds']>60 and r['seconds']-r['unet_seconds']>60]
    audit={'rule':'wall seconds >60 AND wall minus GPU-event U-Net seconds >60; remeasure once with identical settings, retain original quality images and raw rows','repairs':[]}
    if affected:
        runner=Runner(load_pipe(),OUT)
        runner.generate(affected[0],7,'timing_warmup',split='warmup',steps=5,save=False)
        for r in affected:
            measured,image,_=runner.generate(r,r['seed'],r['method'],split='timing_repair',scheduler=r['scheduler'],steps=r['num_steps'],refresh=r['refresh_steps'] if r['full_refresh_count']<r['unet_calls'] else None)
            if measured['seconds']>60:raise RuntimeError('Remeasurement was interrupted; refusing to use its timing')
            original=Image.open(ROOT/r['path']).convert('RGB')
            equal=bool(np.array_equal(np.asarray(original),np.asarray(image)))
            if not equal:raise RuntimeError('Timing-only rerun changed pixels; investigate before summarizing')
            audit['repairs'].append({'key':r['key'],'original_seconds':r['seconds'],'original_unet_seconds':r['unet_seconds'],'replacement_seconds':measured['seconds'],'replacement_unet_seconds':measured['unet_seconds'],'original_image_sha256':hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest(),'replacement_row_key':measured['key'],'pixels_identical':equal})
    write_json(OUT/'timing_repairs.json',audit)
    print(f"Timing audit complete: {len(audit['repairs'])} interruption(s) remeasured, original rows preserved",flush=True)


if __name__=='__main__':main()
