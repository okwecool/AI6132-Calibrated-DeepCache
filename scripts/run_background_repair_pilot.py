"""Compare local inpainting with and without the product conditioning cue."""

import argparse
from pathlib import Path
import random
import time

from run_background import MODEL, ROOT, BackgroundRunner
from run_background_m3 import digest, read_json, write_json

import numpy as np
from PIL import Image, ImageChops
import torch

from src.background_repair import composite_repair, hide_product, repair_masks

MANIFEST = ROOT / "manifests/product_background_repair_pilot.json"
OUTPUT = ROOT / "results/product_background/repair_pilot"
METHODS = ("visible_product", "hidden_product")


def read_picture(path, mode):
    with Image.open(path) as image:
        return image.convert(mode)


def load_case(case, feather):
    folder = ROOT / case["source"]
    original = read_picture(folder / "input.png", "RGB")
    source = read_picture(folder / "composed.png", "RGB")
    foreground = read_picture(folder / "foreground.png", "L")
    if source.size != (512, 512) or original.size != source.size or foreground.size != source.size:
        raise ValueError(f"{case['name']}: expected the shared 512×512 canvas")
    protected = np.asarray(foreground) == 255
    if not np.array_equal(np.asarray(original)[protected], np.asarray(source)[protected]):
        raise RuntimeError(f"{case['name']}: first pass changed the protected product")
    repair, alpha = repair_masks(foreground, case["polygon"], feather)
    return original, source, foreground, repair, alpha


@torch.inference_mode()
def repair_case(pipe, images, case, method, seed, steps, cfg):
    _, source, foreground, repair, alpha = images
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    context = source if method == "visible_product" else hide_product(
        source, foreground, repair, case["fallback_rgb"],
    )
    tensor = pipe.image_processor.preprocess(context).to("cuda", dtype=pipe.vae.dtype)
    edit = pipe.mask_processor.preprocess(repair).to("cuda", dtype=pipe.vae.dtype)
    condition = pipe.vae.encode(tensor * (edit < 0.5)).latent_dist.mode()
    condition *= pipe.vae.config.scaling_factor
    noise = torch.randn(condition.shape, generator=torch.Generator("cpu").manual_seed(seed))
    noise = noise.to("cuda", dtype=pipe.unet.dtype)
    calls = []

    def count_forward(module, args):
        calls.append(1)

    handle = pipe.unet.register_forward_pre_hook(count_forward)
    try:
        generated = pipe(
            prompt=case["prompt"], negative_prompt=case["negative_prompt"],
            image=source, mask_image=repair, masked_image_latents=condition, latents=noise,
            height=512, width=512, strength=1.0, eta=0.0, guidance_scale=cfg,
            num_inference_steps=steps,
        ).images[0]
    finally:
        handle.remove()
    result = composite_repair(source, generated, foreground, repair, alpha)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if len(calls) != steps:
        raise RuntimeError(f"Expected {steps} repair U-Net calls, found {len(calls)}")
    changed = np.any(np.asarray(result) != np.asarray(source), axis=-1)
    return result, {
        "seconds": seconds, "unet_calls": len(calls),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "repair_pixels": int((np.asarray(repair) == 255).sum()),
        "changed_pixels": int(changed.sum()),
        "outside_repair_pixel_max_error": 0, "foreground_pixel_max_error": 0,
    }


def comparison_sheet(cases, seed, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = [("Source photo", "input.png"), ("First pass", "first_pass.png"),
               ("Repair region", "repair_mask.png"), ("Local repair", "visible_product.png"),
               ("Local repair with hidden product cue", "hidden_product.png")]
    fig, axes = plt.subplots(len(cases), len(columns), figsize=(14, 2.65 * len(cases)),
                             squeeze=False, layout="constrained")
    for row, case in enumerate(cases):
        folder = output / "images" / case["name"]
        for col, (label, file) in enumerate(columns):
            ax = axes[row, col]
            path = folder / str(seed) / file if col >= 3 else folder / file
            with Image.open(path) as picture:
                ax.imshow(picture.convert("RGB"))
            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.text(0, -.04, case["name"].replace("_", " "), transform=ax.transAxes, fontsize=9)
            ax.axis("off")
    fig.savefig(output / f"comparison_seed{seed}.jpg", dpi=180, pil_kwargs={"quality": 93})
    plt.close(fig)


def run(output):
    torch.set_num_threads(4)
    protocol = read_json(MANIFEST)
    cases = protocol["cases"]
    inputs = {case["name"]: load_case(case, protocol["feather_pixels"]) for case in cases}
    setup = {
        "manifest_sha256": digest(MANIFEST), "model": read_json(MODEL / "source.json"),
        "inference_code_sha256": {
            str(path.relative_to(ROOT).as_posix()): digest(path)
            for path in (Path(__file__), ROOT / "src/background_repair.py", ROOT / "scripts/run_background.py")
        },
        "source_sha256": {case["name"]: {file: digest(ROOT / case["source"] / file)
                                          for file in ("input.png", "composed.png", "foreground.png")}
                           for case in cases},
    }
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "setup.json"
    if lock.exists() and read_json(lock) != setup:
        raise RuntimeError("Use a different output directory for a new protocol")
    if not lock.exists():
        if any(output.iterdir()):
            raise RuntimeError("The output directory has no matching setup")
        write_json(lock, setup)
    for case in cases:
        folder = output / "images" / case["name"]
        folder.mkdir(parents=True, exist_ok=True)
        for name, image in zip(("input", "first_pass", "foreground", "repair_mask", "alpha"), inputs[case["name"]]):
            image.save(folder / f"{name}.png")

    jobs = [(case, seed, method) for case in cases for seed in protocol["seeds"] for method in METHODS]
    random.Random(protocol["order_seed"]).shuffle(jobs)
    runner, measurements = None, []
    for case, seed, method in jobs:
        name = case["name"]
        folder = output / "images" / name / str(seed)
        image_path, marker = folder / f"{method}.png", folder / f"{method}.json"
        if image_path.is_file() and marker.is_file():
            row = read_json(marker)
            if (row["case"], row["seed"], row["method"]) != (name, seed, method):
                raise RuntimeError(f"Saved repair record differs from the job: {marker}")
        else:
            if runner is None:
                runner = BackgroundRunner()
                repair_case(runner.pipe, inputs[name], case, method, 0, 6, protocol["cfg"])
            repaired, timing = repair_case(
                runner.pipe, inputs[name], case, method, seed, protocol["steps"], protocol["cfg"],
            )
            row = {"case": name, "seed": seed, "method": method, **timing}
            folder.mkdir(parents=True, exist_ok=True)
            repaired.save(image_path)
            write_json(marker, row)
            print(f"{name}/{seed}/{method}: {row['seconds']:.3f} s", flush=True)
        with Image.open(image_path) as result:
            outside = np.asarray(inputs[name][3]) == 0
            difference = np.asarray(ImageChops.difference(result, inputs[name][1]))
            if np.any(difference[outside]):
                raise RuntimeError(f"Saved output changed outside the repair mask: {image_path}")
        measurements.append(row)
    for seed in protocol["seeds"]:
        comparison_sheet(cases, seed, output)
    write_json(output / "measurements.json", measurements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)
