"""Compare prompt wording and caching on four public product photos."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/product_background_photo_pilot.json"
ASSETS = ROOT / "data/raw/product_background/finegrain_pilot"
OUTPUT = ROOT / "results/product_background/photo_pilot"


def asset_path(case, key):
    return ASSETS / f"{case['row_idx']}_{key}.img"


def prepare(manifest):
    import requests

    ASSETS.mkdir(parents=True, exist_ok=True)
    source = manifest["source"]
    for case in manifest["cases"]:
        response = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params={"dataset": source["repo"], "config": source["config"],
                    "split": source["split"], "offset": case["row_idx"], "length": 1},
            timeout=30,
        )
        response.raise_for_status()
        entry = response.json()["rows"][0]
        if entry["row_idx"] != case["row_idx"]:
            raise RuntimeError("Dataset returned a different row")
        for key in ("image", "mask"):
            response = requests.get(entry["row"][key]["src"], timeout=30)
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != case[key]["sha256"]:
                raise RuntimeError(f"Downloaded {key} differs from the frozen manifest")
            path = asset_path(case, key)
            temporary = path.with_suffix(".part")
            temporary.write_bytes(response.content)
            temporary.replace(path)


def comparison_sheet(cases, seed, output):
    columns = [
        ("Input", "background/ddim50/input.png"),
        ("Foreground mask", "background/ddim50/foreground.png"),
        ("Background only | DDIM", "background/ddim50/composed.png"),
        ("Background only | cache", "background/uniform17/composed.png"),
        ("Named product | DDIM", "named/ddim50/composed.png"),
        ("Named product | cache", "named/uniform17/composed.png"),
    ]
    tile = 256
    sheet = Image.new("RGB", (len(columns) * tile, 36 + len(cases) * (tile + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), f"Product background photo pilot | seed {seed}", fill="black")
    for column, (label, filename) in enumerate(columns):
        x = column * tile
        draw.text((x + 6, 22), label, fill="black")
        for row, case in enumerate(cases):
            y = 36 + row * (tile + 20)
            folder = output / "images" / case["name"] / str(seed)
            with Image.open(folder / filename) as picture:
                sheet.paste(picture.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS),
                            (x, y + 20))
            if column == 0:
                draw.text((x + 6, y + 4), case["name"], fill="black")
    sheet.save(output / "images" / f"comparison_seed{seed}.jpg", quality=92)


def load_inputs(cases):
    from run_background import prepare_product

    inputs = {}
    for case in cases:
        for key in ("image", "mask"):
            if hashlib.sha256(asset_path(case, key).read_bytes()).hexdigest() != case[key]["sha256"]:
                raise RuntimeError(f"{case['name']}: {key} does not match the frozen source")
        with Image.open(asset_path(case, "image")) as photo, Image.open(asset_path(case, "mask")) as mask:
            inputs[case["name"]] = prepare_product(photo, mask)
    return inputs


def run(manifest, output):
    from run_background import (
        MODEL, BackgroundRunner, assert_restored, save_case, uniform_schedule,
    )

    if output.exists():
        raise ValueError("Choose a new output directory")
    cases = manifest["cases"]
    inputs = load_inputs(cases)
    runner = BackgroundRunner()
    output.mkdir(parents=True, exist_ok=False)
    first = cases[0]
    runner.generate(*inputs[first["name"]], first["prompts"]["background"], 0, steps=6)
    steps = manifest["steps"]
    schedule = uniform_schedule(steps, manifest["cache_interval"])
    jobs = [(case, seed, wording, method)
            for case in cases for seed in manifest["seeds"]
            for wording in case["prompts"] for method in ("ddim50", "uniform17")]
    random.Random(manifest["order_seed"]).shuffle(jobs)
    records = []
    for case, seed, wording, method in jobs:
        source, foreground = inputs[case["name"]]
        raw, composed, metrics, _, _ = runner.generate(
            source, foreground, case["prompts"][wording], seed, steps=steps,
            refresh=schedule if method == "uniform17" else None,
        )
        assert_restored(source, foreground, composed)
        folder = output / "images" / case["name"] / str(seed) / wording / method
        save_case(folder, source, foreground, raw, composed)
        records.append({"case": case["name"], "seed": seed, "wording": wording,
                        "method": method, **metrics})
        print(f"{case['name']}/{seed}/{wording}/{method}: {metrics['seconds']:.3f} s", flush=True)

    pairs = []
    for case in cases:
        foreground = inputs[case["name"]][1]
        background = np.asarray(foreground) == 0
        ring = (np.asarray(foreground.filter(ImageFilter.MaxFilter(33))) == 255) & background
        for seed in manifest["seeds"]:
            for wording in case["prompts"]:
                folder = output / "images" / case["name"] / str(seed) / wording
                with Image.open(folder / "ddim50/generated.png") as base, \
                     Image.open(folder / "uniform17/generated.png") as cached:
                    delta = np.abs(np.asarray(base, dtype=np.float32) - np.asarray(cached, dtype=np.float32))
                pairs.append({"case": case["name"], "seed": seed, "wording": wording,
                              "background_rgb_mae": float(delta[background].mean()),
                              "boundary_rgb_mae": float(delta[ring].mean())})
    for seed in manifest["seeds"]:
        comparison_sheet(cases, seed, output)
    report = {
        "model": json.loads((MODEL / "source.json").read_text(encoding="utf-8")),
        "manifest": str(MANIFEST.relative_to(ROOT).as_posix()),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "runs": records, "pairs": pairs,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Fetch only the frozen source photos and masks")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    options = parser.parse_args()
    protocol = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if options.prepare:
        prepare(protocol)
    else:
        run(protocol, options.output)
