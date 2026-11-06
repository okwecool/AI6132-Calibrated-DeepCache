"""Download pinned public model assets and official COCO caption annotations."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['TORCH_HOME'] = str(ROOT / '.cache/torch')
os.environ['HF_HUB_DISABLE_XET'] = '1'

import json
import random
import hashlib
import requests
from huggingface_hub import model_info, snapshot_download


def main():
    target = ROOT / 'models/sd15'
    repo = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    # Pin the exact weights used in the experiments, including on fresh setups.
    info = model_info(repo, revision='451f4fe16113bff5a5d2269ed5ad43b0592e9a14')
    files = [f.rfilename for f in info.siblings]
    wanted = ['model_index.json', 'README.md']
    for folder in ['unet', 'vae', 'text_encoder']:
        weights = [f for f in files if f.startswith(folder + '/') and f.endswith('.fp16.safetensors')]
        if len(weights) != 1:
            raise RuntimeError(f'Expected one fp16 weight file for {folder}: {weights}')
        wanted += weights + [folder + '/config.json']
    wanted += [f for f in files if f.startswith(('tokenizer/', 'scheduler/', 'feature_extractor/')) and f.endswith(('.json', '.txt'))]
    wanted += [f for f in files if 'license' in f.lower()]
    print('SD15 revision', info.sha, 'files', wanted, flush=True)
    snapshot_download(repo, revision=info.sha, local_dir=target, allow_patterns=wanted, max_workers=3)
    (target / 'source.json').write_text(json.dumps({'repo_id': repo, 'revision': info.sha, 'files': wanted}, indent=2), encoding='utf-8')

    clip_repo = 'openai/clip-vit-base-patch32'
    clip_info = model_info(clip_repo, revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268')
    clip_files = ['config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt', 'pytorch_model.bin', 'README.md']
    print('CLIP revision', clip_info.sha, flush=True)
    snapshot_download(clip_repo, revision=clip_info.sha, local_dir=ROOT / 'models/clip', allow_patterns=clip_files, max_workers=3)
    (ROOT / 'models/clip/source.json').write_text(json.dumps({'repo_id': clip_repo, 'revision': clip_info.sha}, indent=2), encoding='utf-8')

    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / 'captions_val2014.json'
    # Official COCO caption evaluation repository: small public eval subset.
    # The 2017 all-annotation archive stalled; this pilot needs captions only.
    url = 'https://raw.githubusercontent.com/tylin/coco-caption/master/annotations/captions_val2014.json'
    if not archive.exists():
        part = archive.with_suffix('.json.part')
        print('Downloading official COCO caption evaluation annotations', flush=True)
        with requests.get(url, stream=True, timeout=(20, 40)) as r:
            r.raise_for_status()
            with part.open('wb') as f:
                for chunk in r.iter_content(256 * 1024):
                    f.write(chunk)
        part.replace(archive)
    data = json.loads(archive.read_text(encoding='utf-8'))
    captions = {}
    for ann in sorted(data['annotations'], key=lambda a: a['id']):
        captions.setdefault(ann['image_id'], {'image_id': ann['image_id'], 'annotation_id': ann['id'], 'prompt': ann['caption'].strip()})
    ids = sorted(captions)
    random.Random(6132).shuffle(ids)
    manifest = {'source': url, 'dataset': 'COCO 2014 official caption-evaluation subset', 'source_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(), 'available_images':len(captions), 'split_seed': 6132, 'selection': 'lowest annotation id per image, shuffled image ids', 'calibration': [captions[i] for i in ids[:4]], 'validation': [captions[i] for i in ids[4:8]], 'test': [captions[i] for i in ids[8:16]], 'test_seeds': [2026, 6132]}
    out = ROOT / 'manifests/pilot.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Pilot manifest', out, flush=True)

    import torchvision.models as models
    models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    print('Assets ready', flush=True)


if __name__ == '__main__':
    main()
