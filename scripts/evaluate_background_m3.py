"""Score the frozen M3 outputs and render report, review and demo assets."""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import random
import statistics

from run_background_m3 import (
    IMAGE_FILES, MANIFEST, METHODS, OUTPUT, ROOT, digest, read_json, validate_split, write_json,
)
from run_background_m2 import reference_metrics
from evaluate_pilot import prompt_cluster_ci

import lpips
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from src.region_cost import boundary_crops, edit_regions

LABELS = {
    "ddim50": "DDIM50 reference", "uniform17": "Uniform cache",
    "calibrated": "Calibrated cache", "dpm20": "DPM-Solver++ 20",
}
FIELDS = ("roi_lpips", "full_lpips", "ring_rgb_mae", "background_rgb_mae", "clip_cosine")


def folder(output, name, seed, method):
    return output / "images" / name / str(seed) / method


def verify_runs(output, protocol):
    validate_split(protocol)
    setup = read_json(output / "setup.json")
    if setup["manifest_sha256"] != digest(MANIFEST):
        raise RuntimeError("The test manifest differs from the generated run")
    measurements = read_json(output / "measurements.json")
    rows, repeated = measurements["primary"], measurements["timing_repeats"]
    cases = {case["name"]: case for case in protocol["cases"]}
    expected = {(name, seed, method) for name in cases
                for seed in protocol["seeds"] for method in METHODS}
    keys = {(row["case"], row["seed"], row["method"]) for row in rows}
    if keys != expected or len(rows) != len(expected):
        raise RuntimeError("The primary paired sample is incomplete")
    expected_repeats = {(case["name"], protocol["seeds"][0], method)
                        for case in protocol["cases"][:protocol["timing_repeat_cases"]]
                        for method in METHODS}
    if {(row["case"], row["seed"], row["method"]) for row in repeated} != expected_repeats or \
            len(repeated) != len(expected_repeats):
        raise RuntimeError("The timing repeats are incomplete")
    for row in rows + repeated:
        name, seed, method = row["case"], row["seed"], row["method"]
        if row["image_id"] != cases[name]["row_idx"] or row["refresh_steps"] != setup["refresh_steps"][method]:
            raise RuntimeError(f"Unexpected row or refresh log for {name}/{seed}/{method}")
    for name, seed, method in keys:
        if not all((folder(output, name, seed, method) / file).is_file() for file in IMAGE_FILES):
            raise RuntimeError(f"Missing test images for {name}/{seed}/{method}")
    return rows, repeated


@torch.inference_mode()
def score(rows, protocol, output):
    torch.set_num_threads(4)
    case_map = {case["name"]: case for case in protocol["cases"]}
    masks, boxes = {}, {}
    for name in case_map:
        path = folder(output, name, protocol["seeds"][0], "ddim50")
        with Image.open(path / "foreground.png") as foreground:
            masks[name] = edit_regions(foreground, protocol["ring_radius"])
            boxes[name] = boundary_crops(foreground)

    metric = lpips.LPIPS(net="alex").eval().cuda()
    for row in rows:
        name, seed, method = row["case"], row["seed"], row["method"]
        base = folder(output, name, seed, "ddim50") / "composed.png"
        result = folder(output, name, seed, method) / "composed.png"
        with Image.open(base) as reference, Image.open(result) as candidate:
            row.update(reference_metrics(metric, reference, candidate, boxes[name], *masks[name]))
    del metric
    torch.cuda.empty_cache()

    clip_dir = ROOT / "models/clip"
    model = CLIPModel.from_pretrained(clip_dir, local_files_only=True).eval().cuda()
    processor = CLIPProcessor.from_pretrained(clip_dir, local_files_only=True)
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        images = []
        for row in batch:
            path = folder(output, row["case"], row["seed"], row["method"]) / "composed.png"
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(text=[case_map[row["case"]]["prompt"] for row in batch],
                           images=images, return_tensors="pt", padding=True, truncation=True).to("cuda")
        features = model(**inputs)
        cosine = (F.normalize(features.image_embeds) * F.normalize(features.text_embeds)).sum(dim=1)
        for row, value in zip(batch, cosine.cpu().tolist()):
            row["clip_cosine"] = value
    return boxes


def summarize(rows, repeated, protocol):
    summary = {}
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        summary[method] = {f"mean_{field}": statistics.mean(row[field] for row in group) for field in FIELDS}
        summary[method].update({
            "n": len(group), "median_seconds": statistics.median(row["seconds"] for row in group),
            "median_unet_seconds": statistics.median(row["unet_seconds"] for row in group),
            "peak_allocated_mib": max(row["peak_allocated_mib"] for row in group),
            "peak_reserved_mib": max(row["peak_reserved_mib"] for row in group),
        })
    primary = prompt_cluster_ci(rows, "roi_lpips", "calibrated", "uniform17")
    primary["note"] = "Paired bootstrap over product rows with both seeds kept in each product cluster."
    primary["relative_reduction"] = -primary["mean_difference"] / summary["uniform17"]["mean_roi_lpips"]
    by_case = {case["name"]: {method: [] for method in METHODS} for case in protocol["cases"]}
    original = {}
    for row in rows:
        by_case[row["case"]][row["method"]].append(row["roi_lpips"])
        original[(row["case"], row["seed"], row["method"])] = row
    improvements = {name: statistics.mean(data["calibrated"]) - statistics.mean(data["uniform17"])
                    for name, data in by_case.items()}
    primary["improved_products"] = sum(value < 0 for value in improvements.values())
    primary["delta_by_product"] = improvements
    extremes = {"largest_improvement": min(improvements, key=improvements.get),
                "largest_regression": max(improvements, key=improvements.get)}
    extreme_seeds = {}
    for role, name in extremes.items():
        deltas = {seed: original[name, seed, "calibrated"]["roi_lpips"] -
                  original[name, seed, "uniform17"]["roi_lpips"] for seed in protocol["seeds"]}
        selector = min if role == "largest_improvement" else max
        extreme_seeds[role] = selector(deltas, key=deltas.get)
    timing_repeats = []
    for row in repeated:
        first = original[(row["case"], row["seed"], row["method"])]
        timing_repeats.append({**row, "first_seconds": first["seconds"],
                               "difference_seconds": row["seconds"] - first["seconds"]})
    secondary = {
        "calibrated_vs_uniform_full_lpips": prompt_cluster_ci(rows, "full_lpips", "calibrated", "uniform17"),
        "calibrated_vs_uniform_clip": prompt_cluster_ci(rows, "clip_cosine", "calibrated", "uniform17"),
        "calibrated_vs_dpm_roi": prompt_cluster_ci(rows, "roi_lpips", "calibrated", "dpm20"),
    }
    return {"methods": summary, "primary": primary, "secondary": secondary,
            "timing_repeats": timing_repeats, "paired_extremes": extremes,
            "paired_extreme_seeds": extreme_seeds}


def comparison_figure(output, samples, filename):
    columns = [("Input", "ddim50", "input.png"), ("Mask", "ddim50", "foreground.png")]
    columns.extend((LABELS[method], method, "composed.png") for method in METHODS)
    fig, axes = plt.subplots(len(samples), len(columns), figsize=(15, 2.54 * len(samples) + .5),
                             squeeze=False, layout="constrained")
    for row, (name, seed) in enumerate(samples):
        for col, (label, method, file) in enumerate(columns):
            ax = axes[row, col]
            with Image.open(folder(output, name, seed, method) / file) as image:
                ax.imshow(image.convert("RGB"))
            ax.axis("off")
            if row == 0:
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.text(0, -.03, f"{name.replace('_', ' ')} · seed {seed}",
                        transform=ax.transAxes, fontsize=10)
    fig.savefig(output / filename, dpi=180, pil_kwargs={"quality": 93})
    plt.close(fig)


def summary_figure(output, summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), layout="constrained")
    stats = summary["methods"]
    for index, method in enumerate(METHODS):
        axes[0].scatter(stats[method]["median_seconds"], stats[method]["mean_roi_lpips"],
                        label=LABELS[method], s=65, color=f"C{index}")
    axes[0].set(xlabel="Median pipeline time (s)", ylabel="Mean boundary ROI LPIPS",
                title="Reference deviation vs latency")
    axes[0].legend(fontsize=8)
    primary = summary["primary"]
    values = sorted(primary["delta_by_product"].values())
    axes[1].bar(range(1, len(values) + 1), values,
                color=["#227856" if value < 0 else "#b4493e" for value in values])
    axes[1].axhline(0, color="#333333", linewidth=.8)
    axes[1].set(xlabel="Held-out products, sorted by paired change", ylabel="Calibrated minus uniform ROI LPIPS",
                title="Two seeds averaged within each product")
    fig.savefig(output / "quality_latency.png", dpi=200)
    plt.close(fig)


def blind_review(output, protocol):
    cases = protocol["cases"][:8]
    seed = protocol["seeds"][0]
    rng = random.Random(71032)
    key, form = {}, []
    for start in (0, 4):
        group = cases[start:start + 4]
        sheet = Image.new("RGB", (3 * 320, 30 + len(group) * 354), "white")
        draw = ImageDraw.Draw(sheet)
        for col, label in enumerate(("Input", "Version A", "Version B")):
            draw.text((col * 320 + 8, 7), label, fill="black")
        for row, case in enumerate(group):
            name = case["name"]
            identifier = f"R{start + row + 1:02d}"
            options = ["uniform17", "calibrated"]
            rng.shuffle(options)
            key[identifier] = {"case": name, "seed": seed, "A": options[0], "B": options[1]}
            form.append([identifier, case["prompt"], "", "", ""])
            paths = [folder(output, name, seed, "ddim50") / "input.png"]
            paths.extend(folder(output, name, seed, method) / "composed.png" for method in options)
            y = 30 + row * 354
            draw.text((8, y), identifier, fill="black")
            for col, path in enumerate(paths):
                with Image.open(path) as image:
                    sheet.paste(image.convert("RGB").resize((320, 320), Image.Resampling.LANCZOS),
                                (col * 320, y + 19))
        sheet.save(output / f"blind_review_{start // 4 + 1}.jpg", quality=92)
    write_json(output / "blind_key.json", key)
    with (output / "blind_review.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "prompt", "background_preference_A_B_tie",
                         "boundary_preference_A_B_tie", "notes"])
        writer.writerows(form)


def write_gallery(output, rows, protocol):
    data = json.dumps({"cases": protocol["cases"], "rows": rows, "labels": LABELS,
                       "seeds": protocol["seeds"]}, ensure_ascii=False).replace("<", "\\u003c")
    html = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Product background replacement | M3</title>
<style>
body{font:16px system-ui,sans-serif;background:#f2f4f6;color:#21252b;margin:0 auto;max-width:1440px;padding:22px}
label{margin:0 12px 12px 0;display:inline-block}select{font:inherit;padding:4px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:12px}
figure{background:white;border-radius:9px;margin:0;padding:10px}img{width:100%;aspect-ratio:1;object-fit:contain}
figcaption{margin-top:5px}small{color:#464e59}
</style>
<h1>Product background replacement</h1>
<label>Product <select id="case"></select></label>
<label>Seed <select id="seed"></select></label>
<label>Method <select id="method"></select></label>
<p id="prompt"></p><p id="timing"></p><main id="pictures"></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById('data').textContent);
const selectors = ['case','seed','method'].map(id => document.getElementById(id));
const fill = (select, entries) => entries.forEach(([value, label]) => select.add(new Option(label, value)));
fill(selectors[0], data.cases.map(c => [c.name, c.name.replaceAll('_',' ')]));
fill(selectors[1], data.seeds.map(seed => [seed, seed]));
fill(selectors[2], Object.entries(data.labels));
selectors[2].value = 'calibrated';
function show(){
 const [name, seed, method] = selectors.map(select => select.value);
 const sample = data.cases.find(c => c.name === name);
 const row = data.rows.find(r => r.case === name && String(r.seed) === seed && r.method === method);
 document.getElementById('prompt').textContent = sample.prompt;
 document.getElementById('timing').textContent =
   row.seconds.toFixed(3)+' s | '+row.steps+' denoising steps | '+row.refresh_steps.length+' full refreshes';
 const panels = document.getElementById('pictures'); panels.replaceChildren();
 for(const [file,label] of [['input','Input'],['foreground','Foreground mask'],
                            ['generated','Raw generation'],['composed','Composite']]){
   const figure = document.createElement('figure'), img = document.createElement('img');
   img.src = ['images',name,seed,method,file+'.png'].map(encodeURIComponent).join('/');
   img.alt = label;
   const caption = document.createElement('figcaption'); caption.textContent = label;
   figure.append(img,caption); panels.append(figure);
 }
}
selectors.forEach(select => select.addEventListener('change', show)); show();
</script></html>
"""
    (output / "gallery.html").write_text(html.replace("__DATA__", data), encoding="utf-8")


def run(output, protocol):
    rows, repeated = verify_runs(output, protocol)
    boxes = score(rows, protocol, output)
    summary = summarize(rows, repeated, protocol)
    summary["roi_boxes"] = boxes
    summary["evaluator"] = {
        "lpips": {"package": version("lpips"), "backbone": "alex",
                  "alexnet_sha256": digest(ROOT / ".cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth")},
        "clip": read_json(ROOT / "models/clip/source.json"),
        "torch": version("torch"), "diffusers": version("diffusers"), "DeepCache": version("DeepCache"),
    }
    write_json(output / "scores.json", rows)
    write_json(output / "summary.json", summary)
    with (output / "scores.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "image_id", "seed", "method",
                                                    "steps", "seconds", "peak_allocated_mib", *FIELDS],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    seed = protocol["seeds"][0]
    samples = [(case["name"], seed) for case in protocol["cases"]]
    for start in range(0, len(samples), 4):
        comparison_figure(output, samples[start:start + 4], f"comparison_{start // 4 + 1}.jpg")
    extremes = [(name, summary["paired_extreme_seeds"][role])
                for role, name in summary["paired_extremes"].items()]
    comparison_figure(output, extremes, "paired_extremes.jpg")
    summary_figure(output, summary)
    blind_review(output, protocol)
    write_gallery(output, rows, protocol)
    print(json.dumps(summary["primary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output, read_json(MANIFEST))
