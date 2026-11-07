"""Run the frozen 21-refresh product-background ablation on the M3 inputs."""

import argparse
from pathlib import Path
import random

import numpy as np

from run_background_m3 import (
    IMAGE_FILES, MANIFEST, M2, OUTPUT as M3, ROOT, digest, load_inputs, read_json, write_json,
)

OUTPUT = ROOT / "results/product_background/ablation21"
METHODS = ("uniform21", "global21", "edge21")
STEPS = 50
BUDGET = 21
MAX_GAP = 6


def protocol_for_test(model):
    from src.cache_policy import calibrated_schedule

    manifest = read_json(MANIFEST)
    source = read_json(M3 / "setup.json")
    if source["manifest_sha256"] != digest(MANIFEST) or source["model"] != model:
        raise RuntimeError("The M3 reference differs from the test manifest or model")
    path = M2 / "aggregate_costs.npz"
    with np.load(path) as data:
        costs = {objective: data[objective] for objective in ("global", "edge3")}
    schedules = {
        "uniform21": [i * STEPS // BUDGET for i in range(BUDGET)],
        "global21": calibrated_schedule(costs["global"], BUDGET, MAX_GAP),
        "edge21": calibrated_schedule(costs["edge3"], BUDGET, MAX_GAP),
    }
    if len({tuple(schedule) for schedule in schedules.values()}) != len(METHODS):
        raise RuntimeError("This ablation requires three distinct schedules")
    proxy = {}
    for method, schedule in schedules.items():
        stops = schedule[1:] + [STEPS]
        proxy[method] = {name: float(cost[schedule, stops].sum()) for name, cost in costs.items()}
    return manifest, {
        "status": "exploratory reuse of the M3 identities and reference outputs",
        "manifest_sha256": digest(MANIFEST),
        "m3_setup_sha256": digest(M3 / "setup.json"),
        "m2_cost_sha256": digest(path),
        "model": model,
        "code_sha256": {str(file.relative_to(ROOT).as_posix()): digest(file)
                        for file in (Path(__file__), ROOT / "scripts/run_background.py",
                                     ROOT / "src/background_task.py", ROOT / "src/cache_policy.py")},
        "steps": STEPS, "budget": BUDGET, "max_gap": MAX_GAP,
        "seeds": manifest["seeds"], "row_ids": [case["row_idx"] for case in manifest["cases"]],
        "order_seed": 21701, "schedules": schedules, "calibration_proxy_costs": proxy,
        "primary": {"field": "roi_lpips", "comparison": "edge21 minus global21",
                    "cluster": "product row, both seeds averaged", "interval": "paired bootstrap, 5000 resamples"},
        "secondary": ["edge21 versus uniform21", "global21 versus uniform21",
                      "full LPIPS", "ring and background MAE", "latency", "memory"],
    }


def verify_reference(inputs, manifest):
    from PIL import Image, ImageChops

    for case in manifest["cases"]:
        name = case["name"]
        for seed in manifest["seeds"]:
            reference = M3 / "images" / name / str(seed) / "ddim50"
            if not (reference / "composed.png").is_file():
                raise RuntimeError(f"Missing M3 reference for {name}/{seed}")
            for picture, file in zip(inputs[name], ("input.png", "foreground.png")):
                with Image.open(reference / file) as baseline:
                    if ImageChops.difference(picture, baseline).getbbox() is not None:
                        raise RuntimeError(f"Prepared {file} differs from M3 for {name}/{seed}")


def run(output):
    from run_background import MODEL, BackgroundRunner, assert_restored, save_case
    import torch

    torch.set_num_threads(4)
    manifest, protocol = protocol_for_test(read_json(MODEL / "source.json"))
    inputs = load_inputs(manifest)
    verify_reference(inputs, manifest)
    output.mkdir(parents=True, exist_ok=True)
    locked = output / "protocol.json"
    if locked.exists():
        if read_json(locked) != protocol:
            raise RuntimeError("Choose a new output directory for a different protocol")
    else:
        if any(output.iterdir()):
            raise RuntimeError("The output directory has no matching protocol")
        write_json(locked, protocol)
    schedules = protocol["schedules"]
    jobs = [(case, seed, method) for case in manifest["cases"]
            for seed in protocol["seeds"] for method in METHODS]
    random.Random(protocol["order_seed"]).shuffle(jobs)
    runner, measurements = None, []
    for index, (case, seed, method) in enumerate(jobs, start=1):
        name = case["name"]
        folder = output / "images" / name / str(seed) / method
        marker = folder / "metrics.json"
        if marker.exists() and all((folder / name).is_file() for name in IMAGE_FILES):
            row = read_json(marker)
            if (row["image_id"], row["seed"], row["method"], row["refresh_steps"]) != (
                case["row_idx"], seed, method, schedules[method],
            ):
                raise RuntimeError(f"Saved metrics differ from the frozen job: {folder}")
        else:
            if runner is None:
                runner = BackgroundRunner()
                warmup = manifest["cases"][0]
                runner.generate(*inputs[warmup["name"]], warmup["prompt"], 0, steps=6)
            source, mask = inputs[name]
            raw, composed, metrics, _, _ = runner.generate(
                source, mask, case["prompt"], seed, STEPS, refresh=schedules[method],
            )
            assert_restored(source, mask, composed)
            save_case(folder, source, mask, raw, composed)
            row = {"case": name, "image_id": case["row_idx"], "seed": seed, "method": method, **metrics}
            write_json(marker, row)
            print(f"{index}/{len(jobs)} {name}/{seed}/{method}: {metrics['seconds']:.3f} s", flush=True)
        measurements.append(row)
    write_json(output / "measurements.json", measurements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)
