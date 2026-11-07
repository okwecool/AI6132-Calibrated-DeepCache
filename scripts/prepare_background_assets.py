"""Download and fingerprint the pinned SD inpainting checkpoint."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "sd15-inpaint"
REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"
REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"


def main():
    from huggingface_hub import snapshot_download

    files = [
        "README.md", "model_index.json", "scheduler/scheduler_config.json",
        "tokenizer/merges.txt", "tokenizer/vocab.json",
        "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json",
        "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
        "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
    ]
    snapshot_download(REPO, revision=REVISION, local_dir=MODEL_DIR,
                      allow_patterns=files, max_workers=3, token=False)
    hashes = {}
    for filename in files:
        digest = hashlib.sha256()
        with (MODEL_DIR / filename).open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        hashes[filename] = digest.hexdigest()
    source = {"repo_id": REPO, "revision": REVISION, "sha256": hashes}
    temporary = MODEL_DIR / "source.json.tmp"
    temporary.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    temporary.replace(MODEL_DIR / "source.json")
    print(f"Prepared {MODEL_DIR} at {REVISION}", flush=True)


if __name__ == "__main__":
    main()
