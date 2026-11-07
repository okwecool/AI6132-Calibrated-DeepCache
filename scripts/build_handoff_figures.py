"""Collect the report figures and build reusable figures from saved evidence."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PRODUCT = RESULTS / "product_background"
OUTPUT = ROOT / "docs" / "handoff_figures"
READY = OUTPUT / "ready_to_use"
COLORS = ["#657385", "#d18e3d", "#19736a", "#5565aa"]
TEXT = "#243247"
MUTED = "#4b596d"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": TEXT,
    "axes.labelcolor": TEXT, "axes.edgecolor": "#afbac6",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
})


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, sources, kind):
    with Image.open(path) as image:
        width, height = image.size
    return {
        "figure": path.relative_to(OUTPUT).as_posix(), "kind": kind,
        "source_paths": " | ".join(item.relative_to(ROOT).as_posix() for item in sources),
        "width_px": width, "height_px": height, "sha256": sha(path),
    }


def collect_existing():
    entries = []
    groups = {
        "text_to_image": (RESULTS, ("pilot", "round2", "confirmation")),
        "product_background": (
            PRODUCT, ("m1", "photo_pilot", "m2", "m3", "ablation21", "repair_pilot"),
        ),
    }
    for group, (parent, stages) in groups.items():
        for stage in stages:
            original = parent / stage
            candidates = sorted(file for file in original.iterdir()
                                if file.suffix.lower() in (".png", ".jpg", ".jpeg"))
            if stage == "photo_pilot":
                candidates.extend(sorted((original / "images").glob("comparison_seed*.jpg")))
            for source in candidates:
                category = "review_materials" if source.name.startswith("blind_review_") else "existing"
                target = OUTPUT / category / group / stage / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                entries.append(record(target, [source], "existing copy"))
    return entries


def save(fig, name):
    path = READY / name
    options = {"pil_kwargs": {"quality": 95}} if path.suffix == ".jpg" else {}
    fig.savefig(path, dpi=240, facecolor="white", **options)
    plt.close(fig)
    return path


def method_figure():
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.03, .94, "Calibrated DeepCache for background replacement",
            fontsize=20, weight="bold", va="center")
    ax.text(.03, .85, "OFFLINE  •  Estimate where stale features affect the DDIM update",
            color=COLORS[2], weight="bold")
    ax.text(.03, .44, "ONLINE  •  Run the pretrained inpainting model with a fixed refresh list",
            color=COLORS[2], weight="bold")
    x_positions = (.03, .36, .69)
    width, height = .28, .19
    rows = [
        (.56, [("Uncached trajectory", "Record model inputs and\nreference noise predictions"),
               ("Stale-cache probes", "Measure the noise residual\nand weight the DDIM update"),
               ("Budgeted schedule", "Use dynamic programming\nto choose B full refreshes")]),
        (.15, [("Editing inputs", "Product photo, foreground\nmask and background prompt"),
               ("Cached denoising", "Run 50 DDIM steps with\n17 full U-Net refreshes"),
               ("Protected composite", "Restore original product pixels\ninside the supplied mask")]),
    ]
    for y, boxes in rows:
        for x, (title, detail) in zip(x_positions, boxes):
            ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=.009,rounding_size=.013",
                                       facecolor="#eef4f5" if y > .5 else "#f2f4f8",
                                       edgecolor="#b2c4ce", linewidth=1.1))
            ax.text(x + .016, y + .135, title, fontsize=13, weight="bold", va="center")
            ax.text(x + .016, y + .064, detail, fontsize=10.5, linespacing=1.35, va="center")
        for x, next_x in zip(x_positions[:-1], x_positions[1:]):
            ax.add_patch(FancyArrowPatch((x + width + .012, y + .095), (next_x - .012, y + .095),
                                         arrowstyle="-|>", mutation_scale=13, color=MUTED))
    ax.text(.03, .048, "Calibration runs before deployment. At inference, refresh positions are read from the frozen list.",
            fontsize=10.5, color=MUTED)
    fig.subplots_adjust(left=.018, right=.982, top=.985, bottom=.025)
    path = save(fig, "fig01_method_overview.png")
    return record(path, [ROOT / "src/cache_policy.py", ROOT / "src/error_calibration.py",
                         PRODUCT / "m3" / "setup.json"], "redrawn")


def data_figure():
    manifests = ROOT / "manifests"
    pilot = manifests / "product_background_photo_pilot.json"
    m2 = manifests / "product_background_m2.json"
    m3 = manifests / "product_background_m3.json"
    m2_cases, cases = read_json(m2)["cases"], read_json(m3)["cases"]
    splits = [len(read_json(pilot)["cases"]), sum(x["split"] == "calibration" for x in m2_cases),
              sum(x["split"] == "validation" for x in m2_cases), len(cases)]
    assert len({x["row_idx"] for x in read_json(pilot)["cases"] + m2_cases + cases}) == sum(splits)
    counts = Counter(case["scene_family"] for case in cases)
    coverage = [100 * case["foreground_fraction_at_512"] for case in cases]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    fig.suptitle("Product image dataset and held-out coverage", x=.065, y=.965, ha="left",
                 fontsize=18, weight="bold")
    for ax, values, labels, title in (
        (axes[0], splits, ["Pilot", "Calib.", "Valid.", "Test"], "A  Product identity splits"),
        (axes[1], [counts[k] for k in ("studio", "interior", "outdoor")],
         ["Studio", "Interior", "Outdoor"], "B  Held-out prompt scenes"),
    ):
        bars = ax.bar(labels, values, color=COLORS[2], width=.66)
        ax.bar_label(bars, padding=4, fontsize=11)
        ax.set(title=title, ylim=(0, max(values) * 1.27), ylabel="Product identities")
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].hist(coverage, bins=[10, 20, 30, 40, 50, 60, 70],
                 color=COLORS[3], edgecolor="white", rwidth=.96)
    axes[2].set(title="C  Held-out foreground area", xlabel="Share of prepared canvas (%)",
                ylabel="Product identities", xlim=(9, 71))
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.text(.065, .045, "Finegrain Product Masks Lite  •  One prompt per held-out product  •  "
             "24 products × 2 seeds × 4 methods = 192 primary generations", fontsize=10, color=MUTED)
    fig.subplots_adjust(left=.065, right=.985, top=.78, bottom=.20, wspace=.4)
    return record(save(fig, "fig02_data_overview.png"), [pilot, m2, m3], "redrawn")


def main_results():
    source = PRODUCT / "m3/summary.json"
    data = read_json(source)
    methods = ["ddim50", "uniform17", "calibrated", "dpm20"]
    labels = ["DDIM50", "Uniform 17", "Calibrated 17", "DPM++ 20"]
    results = [data["methods"][method] for method in methods]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.65))
    fig.suptitle("Held-out product background replacement", x=.07, y=.97, ha="left",
                 fontsize=18, weight="bold")
    for ax, field, title, unit, fmt in [
        (axes[0], "median_seconds", "A  Median pipeline time", "Seconds per image", "%.3f"),
        (axes[1], "mean_roi_lpips", "B  Boundary reference deviation", "Mean ROI LPIPS", "%.5f"),
    ]:
        values = [row[field] for row in results]
        bars = ax.bar(labels, values, color=COLORS, width=.62)
        ax.bar_label(bars, fmt=fmt, padding=4, fontsize=10)
        ax.set(title=title, ylabel=unit, ylim=(0, max(values) * 1.22))
        ax.spines[["top", "right"]].set_visible(False)
    primary = data["primary"]
    low, high = primary["cluster_bootstrap_95pct_ci"]
    fig.text(.07, .085,
             f"Calibrated vs uniform: {100*primary['relative_reduction']:.1f}% lower boundary LPIPS, "
             f"paired difference {primary['mean_difference']:.5f}, 95% CI [{low:.5f}, {high:.5f}].",
             fontsize=10.5, weight="bold")
    fig.text(.07, .041, "24 held-out products, two seeds each. LPIPS is measured against the matching uncached DDIM50 output.",
             fontsize=10, color=MUTED)
    fig.subplots_adjust(left=.07, right=.985, top=.80, bottom=.22, wspace=.23)
    return record(save(fig, "fig03_m3_main_results.png"), [source], "redrawn")


def place(ax, path, title=None):
    with Image.open(path) as picture:
        ax.imshow(picture.convert("RGB"))
    if title:
        ax.set_title(title, fontsize=9, pad=4)
    ax.set_axis_off()


def application_cases():
    examples = [
        ("M2 validation  |  Boots", PRODUCT / "m2/images/black_boots", "global", "Clutter and reflections removed"),
        ("M3 held-out  |  Heeled shoes", PRODUCT / "m3/images/heeled_shoes/2026", "calibrated",
         "Studio display with preserved shoe detail"),
        ("M2 validation  |  Printed vest", PRODUCT / "m2/images/printed_vest", "global",
         "Pattern leaks into the background"),
        ("M3 held-out  |  Striped shorts", PRODUCT / "m3/images/striped_shorts/2026", "calibrated",
         "Extra trouser legs appear outside the mask"),
    ]
    fig = plt.figure(figsize=(14.5, 7.75))
    gs = fig.add_gridspec(2, 2, left=.025, right=.975, top=.855, bottom=.08, wspace=.075, hspace=.25)
    fig.suptitle("Product background replacement: successes and failures",
                 x=.025, y=.98, ha="left", fontsize=20, weight="bold")
    sources = []
    for index, (title, folder, method, detail) in enumerate(examples):
        group = gs[index // 2, index % 2]
        group_box = group.get_position(fig)
        fig.text(group_box.x0, group_box.y1 + .031, title, fontsize=12, weight="bold")
        fig.text(group_box.x0, group_box.y0 - .027, detail, fontsize=9.5, color=MUTED)
        panels = group.subgridspec(1, 3, wspace=.035)
        paths = [folder / "ddim50/input.png", folder / "ddim50/composed.png",
                 folder / method / "composed.png"]
        sources.extend(paths)
        for col, (label, path) in enumerate(zip(("Input", "Uncached DDIM50", "Calibrated"), paths)):
            place(fig.add_subplot(panels[0, col]), path, label)
    fig.text(.025, .016, "Finegrain Product Masks Lite  •  Visual examples selected after generation  •  Seed 2026",
             fontsize=9.5, color=MUTED)
    return record(save(fig, "fig04_application_cases.jpg"), sources, "composite")


def repair_cases():
    fig, axes = plt.subplots(2, 4, figsize=(12.8, 7.1))
    fig.suptitle("Local repair: a removable handle and a persistent shape error",
                 x=.026, y=.98, ha="left", fontsize=17.5, weight="bold")
    fig.subplots_adjust(left=.026, right=.974, top=.87, bottom=.09, wspace=.035, hspace=.30)
    sources = []
    for index, (name, title, detail) in enumerate([
        ("striped_cup", "A  Cup", "The added handle disappears for this seed."),
        ("striped_shorts", "B  Shorts", "Both repairs leave an unwanted structure below the product."),
    ]):
        folder = PRODUCT / "repair_pilot/images" / name
        paths = [folder / "first_pass.png", folder / "repair_mask.png",
                 folder / "2026/visible_product.png", folder / "2026/hidden_product.png"]
        sources.extend(paths)
        for col, (label, path) in enumerate(zip(
            ("First pass", "Marked repair mask", "Local repair", "Hidden product cue"), paths,
        )):
            place(axes[index, col], path, label if index == 0 else None)
        box = axes[index, 0].get_position()
        fig.text(.026, box.y1 + .037, title, fontsize=11.5, weight="bold")
        fig.text(.026, box.y0 - .028, detail, fontsize=10, color=MUTED)
    fig.text(.026, .02, "Diagnostic cases, seed 2026  •  All pixels outside the marked repair area are preserved.",
             fontsize=9.5, color=MUTED)
    return record(save(fig, "fig05_repair_cases.jpg"), sources, "composite")


def main():
    READY.mkdir(parents=True, exist_ok=True)
    entries = collect_existing()
    existing_count = len(entries)
    entries.extend([method_figure(), data_figure(), main_results(), application_cases(), repair_cases()])
    with (OUTPUT / "index.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    print(f"Collected {existing_count} existing figures and built {len(entries) - existing_count} new figures.")


if __name__ == "__main__":
    main()
