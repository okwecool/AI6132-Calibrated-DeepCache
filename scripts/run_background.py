"""Run SD inpainting with an optional DeepCache policy, entirely offline."""

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from DeepCache import DeepCacheSDHelper
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw, ImageOps

from src.background_task import prepare_product, restore_product
from src.cache_policy import ScheduledDeepCache, uniform_schedule

MODEL = ROOT / "models" / "sd15-inpaint"
SIZE = 512
CFG = 7.5


class BackgroundRunner:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("The FP16 inpainting runner requires CUDA")
        if not (MODEL / "source.json").is_file():
            raise RuntimeError("Prepare the pinned model with scripts/prepare_background_assets.py")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL, variant="fp16", torch_dtype=torch.float16,
            use_safetensors=True, local_files_only=True, safety_checker=None, feature_extractor=None,
            requires_safety_checker=False,
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        if self.pipe.unet.config.in_channels != 9:
            raise RuntimeError("Expected the dedicated 9-channel inpainting U-Net")
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config, clip_sample=False,
        )

    @torch.inference_mode()
    def generate(self, source, foreground, prompt, seed, steps, *,
                 interval=None, refresh=None, inspect=False):
        if interval is not None and refresh is not None:
            raise ValueError("Choose either the official interval or a refresh list")
        pipe = self.pipe
        edit = ImageOps.invert(foreground)
        helper = nullcontext()
        official = None
        if interval is not None:
            official = DeepCacheSDHelper(pipe)
            official.set_params(cache_interval=interval, cache_branch_id=0)
        elif refresh is not None:
            helper = ScheduledDeepCache(pipe, refresh, branch=0)

        events = []
        refresh_log = []
        first_input = None
        final_latent = None

        def before_unet(module, args):
            nonlocal first_input
            if inspect and first_input is None:
                first_input = args[0].detach().clone()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            events.append((start, end))

        def after_unet(module, args, output):
            events[-1][1].record()

        def after_mid_resnet(module, args, output):
            # The official helper wraps the entire mid block. Its first resnet
            # runs only when that block is actually recomputed.
            refresh_log.append(len(events) - 1)

        def capture_latent(pipeline, index, timestep, tensors):
            nonlocal final_latent
            final_latent = tensors["latents"]
            return tensors

        original_forward = pipe.unet.forward
        hooks = [
            pipe.unet.register_forward_pre_hook(before_unet),
            pipe.unet.register_forward_hook(after_unet),
            pipe.unet.mid_block.resnets[0].register_forward_hook(after_mid_resnet),
        ]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            source_tensor = pipe.image_processor.preprocess(source).to("cuda", dtype=pipe.vae.dtype)
            mask_tensor = pipe.mask_processor.preprocess(edit).to("cuda", dtype=pipe.vae.dtype)
            masked = source_tensor * (mask_tensor < 0.5)
            conditioning = pipe.vae.encode(masked).latent_dist.mode() * pipe.vae.config.scaling_factor
            shape = (1, pipe.vae.config.latent_channels, *conditioning.shape[-2:])
            generator = torch.Generator("cpu").manual_seed(seed)
            noise = torch.randn(shape, generator=generator, dtype=torch.float32).to(
                "cuda", dtype=pipe.unet.dtype,
            )
            if official is not None:
                official.enable()
            with helper:
                raw = pipe(
                    prompt=prompt, image=source, mask_image=edit,
                    masked_image_latents=conditioning, latents=noise,
                    width=SIZE, height=SIZE, strength=1.0, eta=0.0,
                    guidance_scale=CFG, num_inference_steps=steps,
                    callback_on_step_end=capture_latent if inspect else None,
                ).images[0]
            composed = restore_product(source, foreground, raw)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
        finally:
            if official is not None and official.function_dict:
                official.disable()
            for hook in hooks:
                hook.remove()

        expected = list(range(steps))
        if interval is not None:
            expected = uniform_schedule(steps, interval)
        elif refresh is not None:
            expected = list(refresh)
        if len(events) != steps or refresh_log != expected or pipe.unet.forward != original_forward:
            raise RuntimeError(f"Unexpected cache execution: calls={len(events)}, refresh={refresh_log}")
        metrics = {
            "steps": len(events), "refresh_steps": refresh_log,
            "seconds": elapsed,
            "unet_seconds": sum(start.elapsed_time(end) for start, end in events) / 1000,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
        return raw, composed, metrics, first_input, final_latent


def smoke_cases():
    """Synthetic product silhouettes for engineering checks, separate from evaluation data."""
    bottle, box, bag, mug = [Image.new("L", (SIZE, SIZE)) for _ in range(4)]
    draw = ImageDraw.Draw(bottle)
    draw.rounded_rectangle((155, 140, 355, 445), 35, fill=255)
    draw.rectangle((220, 72, 290, 160), fill=255)
    ImageDraw.Draw(box).rounded_rectangle((115, 170, 405, 410), 14, fill=255)
    draw = ImageDraw.Draw(bag)
    draw.ellipse((155, 84, 357, 260), fill=255)
    draw.ellipse((180, 109, 332, 235), fill=0)
    draw.rounded_rectangle((111, 205, 401, 429), 38, fill=255)
    draw = ImageDraw.Draw(mug)
    draw.ellipse((292, 173, 406, 325), fill=255)
    draw.ellipse((311, 198, 381, 299), fill=0)
    draw.rounded_rectangle((125, 142, 329, 405), 45, fill=255)
    for name, color, mask, prompt in [
        ("bottle", "#b24336", bottle,
         "a red bottle on a light stone counter, soft studio lighting, product photography"),
        ("box", "#bb8842", box,
         "a brown gift box on a dark wooden tabletop, warm window light, product photography"),
        ("bag", "#325b87", bag,
         "a blue handbag on a cream fabric display stand, boutique product photography"),
        ("mug", "#41736a", mug,
         "a green ceramic mug on a grey concrete tabletop, soft morning light"),
    ]:
        image = Image.composite(Image.new("RGB", mask.size, color),
                                Image.new("RGB", mask.size, "#dedede"), mask)
        yield name, image, mask, prompt


def save_case(directory, source, foreground, raw, composed):
    directory.mkdir(parents=True, exist_ok=True)
    for name, image in [("input", source), ("foreground", foreground),
                        ("generated", raw), ("composed", composed)]:
        image.save(directory / f"{name}.png")


def max_pixel_delta(first, second):
    return int(np.abs(np.asarray(first, dtype=np.int16) - np.asarray(second, dtype=np.int16)).max())


def assert_restored(source, foreground, composed):
    protected = np.asarray(foreground) == 255
    if not np.array_equal(np.asarray(source)[protected], np.asarray(composed)[protected]):
        raise RuntimeError("The composite modified protected foreground pixels")


def compare_exact(first, second, label):
    pixels = max_pixel_delta(first[0], second[0])
    latent = float((first[4].float() - second[4].float()).abs().max().item())
    if pixels != 0 or latent != 0:
        raise RuntimeError(f"{label}: pixel max delta={pixels}, latent max delta={latent}")
    return {"pixel_max_delta": pixels, "latent_max_delta": latent}


def run_smoke(runner, output):
    checks = {}
    benchmark = None
    for index, (name, image, mask, prompt) in enumerate(smoke_cases()):
        source, foreground = prepare_product(image, mask)
        seed = 6132 + index
        options = {"steps": 6, "inspect": True}
        base = runner.generate(source, foreground, prompt, seed, **options)
        identity = runner.generate(source, foreground, prompt, seed, interval=1, **options)
        uniform = runner.generate(source, foreground, prompt, seed, interval=3, **options)
        scheduled = runner.generate(source, foreground, prompt, seed, refresh=[0, 3], **options)
        if tuple(base[3].shape) != (2, 9, 64, 64):
            raise RuntimeError(f"{name}: unexpected CFG inpainting input shape {base[3].shape}")
        # Diffusers uses torch nearest interpolation (top-left sample).
        edit_mask = np.asarray(foreground)[::8, ::8] == 0
        actual_mask = base[3][:, 4].cpu().numpy()
        if not np.all(actual_mask == edit_mask):
            raise RuntimeError(f"{name}: U-Net did not receive the expected background mask")
        for result in (identity, uniform, scheduled):
            if not torch.equal(base[3], result[3]):
                raise RuntimeError(f"{name}: initial noise or inpainting condition differs")
        assert_restored(source, foreground, scheduled[1])
        checks[name] = {
            "seed": seed, "prompt": prompt, "unet_input_shape": list(base[3].shape),
            "interval1_vs_uncached": compare_exact(base, identity, f"{name}/interval1"),
            "scheduled_vs_official": compare_exact(uniform, scheduled, f"{name}/schedule"),
            "uncached_vs_cached_pixel_delta": max_pixel_delta(base[0], scheduled[0]),
            "observed_refreshes": scheduled[2]["refresh_steps"],
        }
        if index == 0:
            benchmark = (source, foreground, prompt, seed, base)
        save_case(output / "images" / name / "smoke", source, foreground, *scheduled[:2])
        print(f"{name}: mask, shared inputs and cache equivalence passed", flush=True)

    source, foreground, prompt, seed, initial = benchmark
    repeated = runner.generate(source, foreground, prompt, seed, steps=6, inspect=True)
    isolation = compare_exact(initial, repeated, "first case after intervening cached cases")
    timings = {}
    for name, options in [("ddim50", {}), ("scheduled_uniform17", {"refresh": uniform_schedule(50, 3)})]:
        raw, composed, metrics, _, _ = runner.generate(
            source, foreground, prompt, seed, steps=50, **options,
        )
        assert_restored(source, foreground, composed)
        save_case(output / "images" / "bottle" / name, source, foreground, raw, composed)
        timings[name] = metrics
        print(f"{name}: {metrics['seconds']:.3f} s, {metrics['peak_allocated_mib']:.0f} MiB allocated",
              flush=True)
    return {"fixture_source": "procedurally drawn engineering fixtures",
            "checks": checks, "cache_isolation": isolation, "timings": timings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output", type=Path, default=ROOT / "results/product_background/m1")
    generate = commands.add_parser("generate")
    generate.add_argument("--image", type=Path, required=True)
    generate.add_argument("--foreground", type=Path, required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--seed", type=int, default=6132)
    generate.add_argument("--steps", type=int, default=50)
    generate.add_argument("--interval", type=int)
    generate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory to preserve the existing run")
    if args.command == "generate":
        # The pinned DDIM configuration starts at offset 1, so 1000 steps
        # would include timestep 1000, outside the training schedule.
        if not 1 <= args.steps <= 999 or (args.interval is not None and args.interval < 1):
            parser.error("Expected 1–999 sampling steps and a positive cache interval")
        with Image.open(args.image) as image, Image.open(args.foreground) as foreground:
            source, mask = prepare_product(image, foreground)
    runner = BackgroundRunner()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.command == "smoke":
        report = run_smoke(runner, args.output)
    else:
        refresh = None if args.interval is None else uniform_schedule(args.steps, args.interval)
        raw, composed, metrics, _, _ = runner.generate(
            source, mask, args.prompt, args.seed, args.steps, refresh=refresh,
        )
        assert_restored(source, mask, composed)
        save_case(args.output / "images", source, mask, raw, composed)
        report = {"prompt": args.prompt, "seed": args.seed, "metrics": metrics}
    report.update({
        "model": json.loads((MODEL / "source.json").read_text(encoding="utf-8")),
        "gpu": torch.cuda.get_device_name(),
        "versions": {package: importlib.metadata.version(package)
                     for package in ("torch", "diffusers", "DeepCache")},
        "configuration": {"resolution": SIZE, "cfg": CFG, "strength": 1.0, "eta": 0.0,
                          "dtype": "float16", "scheduler": dict(runner.pipe.scheduler.config)},
    })
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
