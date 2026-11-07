"""Run the frozen product-background policies on held-out photos."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/product_background_m3.json"
ASSETS = ROOT / "data/raw/product_background/finegrain_m3"
OUTPUT = ROOT / "results/product_background/m3"
M2 = ROOT / "results/product_background/m2"
METHODS = ("ddim50", "uniform17", "calibrated", "dpm20")
IMAGE_FILES = ("input.png", "foreground.png", "generated.png", "composed.png")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, contents):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def asset(case, key):
    return ASSETS / f"{case['row_idx']}_{key}.img"


def prepare(protocol):
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import pyarrow.parquet as parquet
    from huggingface_hub import snapshot_download

    source = protocol["source"]
    downloaded = Path(snapshot_download(
        source["repo"], repo_type="dataset", revision=source["revision"],
        allow_patterns=["*.parquet"], local_dir=ASSETS / "parquet", token=False,
    ))
    rows = []
    for file in sorted(downloaded.rglob("*.parquet")):
        rows.extend(parquet.read_table(file, columns=["image", "mask"]).to_pylist())
    for case in protocol["cases"]:
        for key in ("image", "mask"):
            raw = rows[case["row_idx"]][key]["bytes"]
            if hashlib.sha256(raw).hexdigest() != case[key]["sha256"]:
                raise RuntimeError(f"{case['name']} {key} does not match the manifest")
            path = asset(case, key)
            temporary = path.with_suffix(".part")
            temporary.write_bytes(raw)
            temporary.replace(path)


def load_inputs(protocol):
    from PIL import Image
    from src.background_task import prepare_product

    inputs = {}
    for case in protocol["cases"]:
        for key in ("image", "mask"):
            if digest(asset(case, key)) != case[key]["sha256"]:
                raise RuntimeError(f"{case['name']} {key} does not match the manifest")
        with Image.open(asset(case, "image")) as image, Image.open(asset(case, "mask")) as mask:
            inputs[case["name"]] = prepare_product(image, mask)
    return inputs


def frozen_setup(protocol, model):
    calibration_file, validation_file = M2 / "calibration.json", M2 / "report.json"
    calibration, validation = read_json(calibration_file), read_json(validation_file)
    if protocol["policy_source_sha256"] != {
        "calibration.json": digest(calibration_file), "report.json": digest(validation_file),
    }:
        raise RuntimeError("The M2 policy artifacts differ from the frozen manifest")
    if model != validation["model"]:
        raise RuntimeError("Model provenance differs from M2")
    selected = validation["selected"]
    schedule = calibration["schedules"][selected]
    uniform = calibration["schedules"]["uniform17"]
    dpm_steps = validation["dpm"]["steps"]
    if schedule != calibration["schedules"]["global"] or dpm_steps != 20:
        raise RuntimeError("The frozen M2 schedule or solver baseline has changed")
    return {
        "manifest_sha256": digest(MANIFEST),
        "source_hashes": protocol["policy_source_sha256"],
        "inference_code_sha256": {
            str(path.relative_to(ROOT).as_posix()): digest(path)
            for path in (Path(__file__), ROOT / "scripts/run_background.py",
                         ROOT / "src/background_task.py", ROOT / "src/cache_policy.py")
        },
        "model": model, "selected": selected, "refresh_steps": {
            "ddim50": list(range(protocol["steps"])), "uniform17": uniform,
            "calibrated": schedule, "dpm20": list(range(dpm_steps)),
        },
    }


def validate_split(protocol):
    previous = []
    for name in ("photo_pilot", "m2"):
        previous.extend(read_json(ROOT / f"manifests/product_background_{name}.json")["cases"])
    excluded = {case["row_idx"] for case in previous}
    cases = protocol["cases"]
    ids = {case["row_idx"] for case in cases}
    if len(ids) != len(cases) or ids & excluded or set(protocol["excluded_rows"]) != excluded:
        raise ValueError("The test photos must be distinct from earlier phases")
    if len({case["name"] for case in cases}) != len(cases):
        raise ValueError("The test case names must be unique")


def generate(protocol, output):
    from run_background import MODEL, BackgroundRunner, assert_restored, save_case
    from diffusers import DPMSolverMultistepScheduler
    from PIL import Image, ImageChops
    import torch

    validate_split(protocol)
    torch.set_num_threads(4)
    inputs = load_inputs(protocol)
    setup = frozen_setup(protocol, read_json(MODEL / "source.json"))
    output.mkdir(parents=True, exist_ok=True)
    setup_file = output / "setup.json"
    if setup_file.exists():
        if read_json(setup_file) != setup:
            raise RuntimeError("Choose a new output directory for a different protocol")
    else:
        if any(output.iterdir()):
            raise RuntimeError("The output directory has no matching run provenance")
        write_json(setup_file, setup)

    cases, seeds = protocol["cases"], protocol["seeds"]
    jobs = [(case, seed, method, False) for case in cases for seed in seeds for method in METHODS]
    rng = random.Random(protocol["order_seed"])
    rng.shuffle(jobs)
    repeats = [(case, seeds[0], method, True)
               for case in cases[:protocol["timing_repeat_cases"]] for method in METHODS]
    rng.shuffle(repeats)
    jobs.extend(repeats)
    rows, repeated = [], []
    runner = None
    for index, (case, seed, method, repeat) in enumerate(jobs, start=1):
        name = case["name"]
        folder = output / "images" / name / str(seed) / method
        if repeat:
            folder = folder / "timing_repeat"
        marker = folder / "metrics.json"
        expected = setup["refresh_steps"][method]
        if marker.exists() and all((folder / file).is_file() for file in IMAGE_FILES):
            row = read_json(marker)
            if (row["case"], row["seed"], row["method"], row["refresh_steps"]) != (name, seed, method, expected):
                raise RuntimeError(f"Saved metrics do not match the frozen job: {folder}")
        else:
            if runner is None:
                runner = BackgroundRunner()
                ddim = runner.pipe.scheduler
                dpm = DPMSolverMultistepScheduler.from_config(
                    ddim.config, algorithm_type="dpmsolver++", solver_order=2,
                )
                warmup = cases[0]
                runner.generate(*inputs[warmup["name"]], warmup["prompt"], 0, steps=6)
            source, mask = inputs[name]
            runner.pipe.scheduler = dpm if method == "dpm20" else ddim
            steps = 20 if method == "dpm20" else protocol["steps"]
            refresh = expected if method in ("uniform17", "calibrated") else None
            raw, composed, metrics, _, _ = runner.generate(
                source, mask, case["prompt"], seed, steps, refresh=refresh,
            )
            assert_restored(source, mask, composed)
            if metrics["refresh_steps"] != expected:
                raise RuntimeError(f"The actual refresh steps differ for {folder}")
            save_case(folder, source, mask, raw, composed)
            row = {"case": name, "image_id": case["row_idx"], "seed": seed, "method": method,
                   "gpu": torch.cuda.get_device_name(), **metrics}
            if repeat:
                with Image.open(folder.parent / "generated.png") as original:
                    row["repeated_pixels_equal"] = ImageChops.difference(raw, original).getbbox() is None
            write_json(marker, row)
            print(f"{index}/{len(jobs)} {name}/{seed}/{method}{' repeat' if repeat else ''}: "
                  f"{metrics['seconds']:.3f} s", flush=True)
        (repeated if repeat else rows).append(row)
    write_json(output / "measurements.json", {"primary": rows, "timing_repeats": repeated})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    protocol = read_json(MANIFEST)
    if args.prepare:
        prepare(protocol)
    else:
        generate(protocol, args.output)
