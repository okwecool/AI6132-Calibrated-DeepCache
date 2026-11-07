"""Calibrate and validate spatial DDIM cache costs on separate photo identities."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

from run_background import (
    CFG, MODEL, ROOT, BackgroundRunner, assert_restored, save_case, uniform_schedule,
)
from run_background_photo_pilot import load_inputs, prepare

import lpips
import numpy as np
from diffusers import DPMSolverMultistepScheduler
from PIL import Image, ImageDraw
import torch

from src.cache_policy import ScheduledDeepCache, calibrated_schedule
from src.error_calibration import ddim_epsilon_coefficient, guided_prediction, interval_cost_from_errors
from src.region_cost import boundary_crops, edit_regions, normalized_region_weights

MANIFEST = ROOT / "manifests/product_background_m2.json"
OUTPUT = ROOT / "results/product_background/m2"
OBJECTIVES = ("global", "edit", "edge1", "edge3")
os.environ["TORCH_HOME"] = str(ROOT / ".cache/torch")


@torch.inference_mode()
def collect_reference(runner, source, mask, prompt, seed, steps):
    inputs, timesteps, predictions, context = [], [], [], []

    def before(module, args, kwargs):
        inputs.append(args[0].cpu())
        timesteps.append(args[1].detach())
        if not context:
            context.append(kwargs["encoder_hidden_states"].detach())

    def after(module, args, output):
        predictions.append(output[0].cpu())

    unet = runner.pipe.unet
    handles = [unet.register_forward_pre_hook(before, with_kwargs=True),
               unet.register_forward_hook(after)]
    try:
        runner.generate(source, mask, prompt, seed, steps)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(inputs).cuda(), timesteps, torch.stack(predictions).cuda(), context[0]


@torch.inference_mode()
def measure_case(runner, source, mask, case, protocol):
    pipe = runner.pipe
    steps, max_gap = protocol["steps"], protocol["max_gap"]
    xs, timesteps, predictions, context = collect_reference(
        runner, source, mask, case["prompt"], protocol["calibration_seed"], steps,
    )
    weights = normalized_region_weights(mask, xs.shape[-2:], protocol["ring_radius"]).cuda()
    coefficients = np.array([ddim_epsilon_coefficient(pipe.scheduler, t) for t in timesteps])
    exact = [guided_prediction(p, CFG) for p in predictions]
    errors = np.full((len(OBJECTIVES), steps, steps), np.nan)
    replay_error = 0.0
    count = 0
    for start in range(steps):
        with ScheduledDeepCache(pipe, [0]):
            prime = pipe.unet(xs[start], timesteps[start], encoder_hidden_states=context,
                              return_dict=False)[0]
            replay_error = max(replay_error, (prime.float() - predictions[start].float()).abs().max().item())
            errors[:, start, start] = 0
            for target in range(start + 1, min(start + max_gap, steps)):
                stale = pipe.unet(xs[target], timesteps[target], encoder_hidden_states=context,
                                  return_dict=False)[0]
                mse = (guided_prediction(stale, CFG) - exact[target]).square().mean(dim=(0, 1))
                weighted = (weights * mse).sum(dim=(1, 2))
                errors[:, start, target] = weighted.cpu().numpy() * coefficients[target] ** 2
                count += 1
    if not np.isfinite(replay_error) or replay_error > 1e-4:
        raise RuntimeError(f"Full replay mismatch for {case['name']}: {replay_error}")
    return errors, {"case": case["name"], "replay_max_error": replay_error, "partial_probes": count}


def calibrate(runner, cases, inputs, protocol, output):
    started = time.perf_counter()
    measured, checks = [], []
    for case in cases:
        error, check = measure_case(runner, *inputs[case["name"]], case, protocol)
        measured.append(error)
        checks.append(check)
        print(f"Calibration {case['name']}: {check}", flush=True)
    means = np.mean(np.stack(measured), axis=0)
    costs = {name: interval_cost_from_errors(error, protocol["max_gap"])
             for name, error in zip(OBJECTIVES, means)}
    schedules = {"uniform17": uniform_schedule(protocol["steps"], 3)}
    schedules.update({name: calibrated_schedule(cost, protocol["budget"], protocol["max_gap"])
                      for name, cost in costs.items()})
    proxies = {}
    for name, schedule in schedules.items():
        stops = schedule[1:] + [protocol["steps"]]
        proxies[name] = {objective: float(cost[schedule, stops].sum())
                         for objective, cost in costs.items()}
    np.savez_compressed(output / "aggregate_costs.npz", **costs)
    result = {"checks": checks, "seconds": time.perf_counter() - started,
              "schedules": schedules, "calibration_proxy_costs": proxies}
    (output / "calibration.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def render_case(runner, inputs, case, seed, steps, refresh, output, method):
    source, mask = inputs[case["name"]]
    raw, composed, metrics, _, _ = runner.generate(
        source, mask, case["prompt"], seed, steps, refresh=refresh,
    )
    assert_restored(source, mask, composed)
    save_case(output / "images" / case["name"] / method, source, mask, raw, composed)
    print(f"Validation {case['name']}/{method}: {metrics['seconds']:.3f} s", flush=True)
    return {"case": case["name"], "method": method, **metrics}


def validate(runner, cases, inputs, protocol, schedules, output):
    seed = protocol["validation_seed"]
    jobs = [(case, name, refresh) for case in cases
            for name, refresh in {"ddim50": None, **schedules}.items()]
    random.Random(protocol["order_seed"]).shuffle(jobs)
    rows = [render_case(runner, inputs, case, seed, protocol["steps"], refresh, output, name)
            for case, name, refresh in jobs]

    ddim = runner.pipe.scheduler
    runner.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        ddim.config, algorithm_type="dpmsolver++", solver_order=2,
    )
    timings = {}
    try:
        match_cases = cases[:protocol["dpm_match_cases"]]
        for case in match_cases:
            for steps in protocol["dpm_candidates"]:
                row = render_case(runner, inputs, case, seed, steps, None, output, f"dpm{steps}")
                timings.setdefault(steps, []).append(row)
        names = {case["name"] for case in match_cases}
        target = statistics.median(row["seconds"] for row in rows
                                   if row["method"] == "uniform17" and row["case"] in names)
        choice = min(timings, key=lambda steps: abs(
            statistics.median(row["seconds"] for row in timings[steps]) - target))
        rows.extend(timings[choice])
        rows.extend(render_case(runner, inputs, case, seed, choice, None, output, f"dpm{choice}")
                    for case in cases[len(match_cases):])
    finally:
        runner.pipe.scheduler = ddim
    return rows, {"steps": choice, "target_seconds": target, "trials": timings}


@torch.inference_mode()
def reference_metrics(metric, reference, candidate, boxes, edit, ring):
    def tensors(picture, boxes):
        crops = [np.asarray(picture.crop(box)) for box in boxes]
        return torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).cuda().float() / 127.5 - 1

    diff = np.abs(np.asarray(reference, dtype=np.float32) - np.asarray(candidate, dtype=np.float32))
    full = [(0, 0, reference.width, reference.height)]
    return {
        "roi_lpips": float(metric(tensors(reference, boxes), tensors(candidate, boxes)).mean()),
        "ring_rgb_mae": float(diff[ring].mean()),
        "background_rgb_mae": float(diff[edit].mean()),
        "full_lpips": float(metric(tensors(reference, full), tensors(candidate, full)).mean()),
    }


@torch.inference_mode()
def evaluate(rows, cases, inputs, radius, output):
    metric = lpips.LPIPS(net="alex").eval().cuda()
    box_map = {case["name"]: boundary_crops(inputs[case["name"]][1]) for case in cases}

    for row in rows:
        name, method = row["case"], row["method"]
        folder = output / "images" / name
        edit, ring = edit_regions(inputs[name][1], radius)
        with Image.open(folder / "ddim50/composed.png") as reference, \
             Image.open(folder / method / "composed.png") as candidate:
            row.update(reference_metrics(metric, reference, candidate, box_map[name], edit, ring))
    summary = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {f"mean_{field}": statistics.mean(row[field] for row in subset)
                           for field in ("roi_lpips", "ring_rgb_mae", "background_rgb_mae", "full_lpips")}
        summary[method]["median_seconds"] = statistics.median(row["seconds"] for row in subset)
    return summary, box_map


def make_sheet(cases, methods, output, filename, tile=240):
    columns = [("Input", "ddim50/input.png"), ("Mask", "ddim50/foreground.png")]
    columns.extend((label, f"{method}/composed.png") for label, method in methods)
    sheet = Image.new("RGB", (len(columns) * tile, 30 + len(cases) * (tile + 21)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, (label, path) in enumerate(columns):
        draw.text((col * tile + 5, 7), label, fill="black")
        for row, case in enumerate(cases):
            y = 30 + row * (tile + 21)
            with Image.open(output / "images" / case / path) as picture:
                sheet.paste(picture.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS),
                            (col * tile, y + 19))
            if col == 0:
                draw.text((5, y + 3), case, fill="black")
    sheet.save(output / filename, quality=93)


def run(protocol, output):
    if output.exists():
        raise ValueError("Choose a new output directory")
    if not (Path(os.environ["TORCH_HOME"]) / "hub/checkpoints/alexnet-owt-7be5be79.pth").exists():
        raise RuntimeError("The local AlexNet weights are required for LPIPS")
    torch.set_num_threads(4)
    cases = protocol["cases"]
    calibration = [case for case in cases if case["split"] == "calibration"]
    validation = [case for case in cases if case["split"] == "validation"]
    if len({case["row_idx"] for case in cases}) != len(cases) or len(calibration) + len(validation) != len(cases):
        raise ValueError("Calibration and validation must use distinct rows")
    inputs = load_inputs(cases)
    runner = BackgroundRunner()
    output.mkdir(parents=True, exist_ok=False)
    first = calibration[0]
    runner.generate(*inputs[first["name"]], first["prompt"], 0, steps=6)
    result = calibrate(runner, calibration, inputs, protocol, output)
    rows, dpm = validate(runner, validation, inputs, protocol, result["schedules"], output)
    del runner
    torch.cuda.empty_cache()
    summary, crops = evaluate(rows, validation, inputs, protocol["ring_radius"], output)
    selected = min(("edit", "edge1", "edge3"), key=lambda name: summary[name]["mean_roi_lpips"])
    methods = [("DDIM50 reference", "ddim50"), ("Uniform 17", "uniform17"),
               ("Global cost", "global"), (f"Selected: {selected}", selected)]
    names = [case["name"] for case in validation]
    make_sheet(names[:4], methods, output, "validation_1.jpg")
    make_sheet(names[4:], methods, output, "validation_2.jpg")
    selected_rows = sorted((row for row in rows if row["method"] == selected),
                           key=lambda row: row["roi_lpips"])
    extremes = [selected_rows[0]["case"], selected_rows[-1]["case"]]
    make_sheet(extremes, methods, output, "reference_extremes.jpg", tile=280)
    report = {"manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
              "model": json.loads((MODEL / "source.json").read_text(encoding="utf-8")),
              "validation": rows, "summary": summary, "roi_boxes": crops, "dpm": dpm,
              "selected": selected, "reference_extremes": extremes}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen candidate {selected}; reference extrema {extremes}; DPM steps {dpm['steps']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    protocol = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.prepare:
        prepare(protocol)
    else:
        run(protocol, args.output)
