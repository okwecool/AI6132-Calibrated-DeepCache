"""Evaluate the three 21-refresh schedules against identical M3 references."""

import argparse
import csv
from pathlib import Path
import statistics

from run_background_ablation21 import MANIFEST, M2, M3, METHODS, OUTPUT, digest, read_json, write_json
from run_background_m2 import reference_metrics
from evaluate_pilot import prompt_cluster_ci

import lpips
import matplotlib.pyplot as plt
from PIL import Image
import torch

from src.region_cost import boundary_crops, edit_regions

FIELDS = ("roi_lpips", "full_lpips", "ring_rgb_mae", "background_rgb_mae")
LABELS = {"uniform21": "Uniform 21", "global21": "Global DP 21", "edge21": "Boundary DP 21"}


def image_folder(output, name, seed, method):
    return output / "images" / name / str(seed) / method


def verify_rows(output, manifest, protocol):
    if (protocol["manifest_sha256"] != digest(MANIFEST)
            or protocol["m3_setup_sha256"] != digest(M3 / "setup.json")
            or protocol["m2_cost_sha256"] != digest(M2 / "aggregate_costs.npz")):
        raise RuntimeError("The frozen inputs differ from the ablation protocol")
    case_ids = {case["name"]: case["row_idx"] for case in manifest["cases"]}
    rows = read_json(output / "measurements.json")
    expected = {(name, seed, method) for name in case_ids
                for seed in protocol["seeds"] for method in METHODS}
    actual = {(row["case"], row["seed"], row["method"]) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError("The ablation is missing some paired outputs")
    for row in rows:
        if (row["image_id"] != case_ids[row["case"]] or row["steps"] != protocol["steps"]
                or row["refresh_steps"] != protocol["schedules"][row["method"]]):
            raise RuntimeError(f"Unexpected refresh record for {row['case']}")
    return rows


def score(rows, manifest, protocol, output):
    torch.set_num_threads(4)
    metric = lpips.LPIPS(net="alex").eval().cuda()
    regions, boxes = {}, {}
    for case in manifest["cases"]:
        name = case["name"]
        reference = image_folder(M3, name, protocol["seeds"][0], "ddim50")
        with Image.open(reference / "foreground.png") as mask:
            regions[name] = edit_regions(mask, manifest["ring_radius"])
            boxes[name] = boundary_crops(mask)
    for row in rows:
        name, seed, method = row["case"], row["seed"], row["method"]
        reference = image_folder(M3, name, seed, "ddim50") / "composed.png"
        result = image_folder(output, name, seed, method) / "composed.png"
        with Image.open(reference) as base, Image.open(result) as candidate:
            row.update(reference_metrics(metric, base, candidate, boxes[name], *regions[name]))


def summarize(rows):
    methods = {}
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        methods[method] = {
            "samples": len(group),
            **{field: statistics.mean(row[field] for row in group) for field in FIELDS},
            "median_seconds": statistics.median(row["seconds"] for row in group),
            "median_unet_seconds": statistics.median(row["unet_seconds"] for row in group),
            "peak_allocated_mib": max(row["peak_allocated_mib"] for row in group),
            "peak_reserved_mib": max(row["peak_reserved_mib"] for row in group),
        }
    primary = prompt_cluster_ci(rows, "roi_lpips", "edge21", "global21")
    primary["note"] = "Exploratory paired bootstrap over the M3 product identities, with both seeds grouped."
    paired = {}
    for row in rows:
        paired.setdefault(row["case"], {}).setdefault(row["seed"], {})[row["method"]] = row["roi_lpips"]
    deltas = {name: {seed: values["edge21"] - values["global21"] for seed, values in seeds.items()}
              for name, seeds in paired.items()}
    by_product = {name: statistics.mean(by_seed.values()) for name, by_seed in deltas.items()}
    primary["relative_change"] = primary["mean_difference"] / methods["global21"]["roi_lpips"]
    primary["edge_better_products"] = sum(value < 0 for value in by_product.values())
    primary["edge_worse_products"] = sum(value > 0 for value in by_product.values())
    primary["delta_by_product"] = by_product
    extremes = {}
    for name, choose in (("lowest_delta", min), ("highest_delta", max)):
        case = choose(by_product, key=by_product.get)
        seed = choose(deltas[case], key=deltas[case].get)
        extremes[name] = {"case": case, "seed": seed, "roi_lpips_difference": deltas[case][seed]}
    secondary = {
        "edge_minus_uniform_roi": prompt_cluster_ci(rows, "roi_lpips", "edge21", "uniform21"),
        "global_minus_uniform_roi": prompt_cluster_ci(rows, "roi_lpips", "global21", "uniform21"),
        "edge_minus_global_full": prompt_cluster_ci(rows, "full_lpips", "edge21", "global21"),
        "edge_minus_global_seconds": prompt_cluster_ci(rows, "seconds", "edge21", "global21"),
    }
    return {"methods": methods, "primary": primary, "secondary": secondary, "paired_extremes": extremes}


def plot_results(output, protocol, summary):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), layout="constrained")
    for y, method in enumerate(METHODS):
        axes[0].scatter(protocol["schedules"][method], [y] * protocol["budget"], marker="|",
                        s=120, linewidths=1.8, color=f"C{y}")
    axes[0].set(yticks=range(len(METHODS)), yticklabels=[LABELS[method] for method in METHODS],
                xlabel="Denoising step index", xlim=(-1, 50), title="Equal refresh budget")
    axes[0].grid(axis="x", alpha=.2)
    values = sorted(summary["primary"]["delta_by_product"].values())
    axes[1].bar(range(1, len(values) + 1), values,
                color=["#227856" if value < 0 else "#b4493e" for value in values])
    axes[1].axhline(0, color="#333333", linewidth=.8)
    axes[1].set(xlabel="Products, sorted by paired change", ylabel="Boundary minus global ROI LPIPS",
                title="Two seeds averaged per product")
    fig.savefig(output / "ablation_summary.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(13, 5.8), layout="constrained")
    for index, row in enumerate(summary["paired_extremes"].values()):
        name, seed = row["case"], row["seed"]
        base = image_folder(M3, name, seed, "ddim50")
        columns = [("Input", base / "input.png"), ("DDIM50 reference", base / "composed.png")]
        columns.extend((LABELS[method], image_folder(output, name, seed, method) / "composed.png")
                       for method in METHODS)
        for col, (label, path) in enumerate(columns):
            ax = axes[index, col]
            with Image.open(path) as picture:
                ax.imshow(picture.convert("RGB"))
            ax.axis("off")
            if index == 0:
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.text(0, -.04, f"{name.replace('_', ' ')} · seed {seed}",
                        transform=ax.transAxes, fontsize=9)
    fig.savefig(output / "paired_cases.jpg", dpi=200, pil_kwargs={"quality": 93})
    plt.close(fig)


def run(output):
    manifest, protocol = read_json(MANIFEST), read_json(output / "protocol.json")
    rows = verify_rows(output, manifest, protocol)
    score(rows, manifest, protocol, output)
    summary = summarize(rows)
    write_json(output / "scores.json", rows)
    write_json(output / "summary.json", summary)
    with (output / "scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, extrasaction="ignore",
                                fieldnames=["case", "image_id", "seed", "method", "seconds", *FIELDS])
        writer.writeheader()
        writer.writerows(rows)
    plot_results(output, protocol, summary)
    estimate = summary["primary"]
    print(f"Edge minus global ROI LPIPS: {estimate['mean_difference']:.6f} "
          f"{estimate['cluster_bootstrap_95pct_ci']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)
