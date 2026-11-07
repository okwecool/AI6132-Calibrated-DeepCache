"""Prepare counterbalanced offline review forms and summarize human responses."""

import argparse
import base64
from collections import Counter
import csv
from io import BytesIO
import json
from pathlib import Path
import random
import re
import statistics

from PIL import Image

from run_background_m3 import MANIFEST, OUTPUT as M3, ROOT, digest, read_json, write_json

OUTPUT = ROOT / "results/product_background/blind_review"
DIMENSIONS = ("boundary", "background")
METHODS = ("calibrated", "uniform17")
FIELDS = ("rater_code", "form", "case_id", *DIMENSIONS, "both_unacceptable", "notes")


def png_data(path):
    with Image.open(path) as picture, BytesIO() as stream:
        clean = Image.new("RGB", picture.size)
        clean.paste(picture.convert("RGB"))
        clean.save(stream, format="PNG")
        return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def write_form(path, version, cases):
    payload = json.dumps({"form": version, "cases": cases, "fields": FIELDS},
                         ensure_ascii=False).replace("<", "\\u003c")
    html = """<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>商品背景替换盲评</title>
<style>
body{font:16px/1.55 system-ui,sans-serif;background:#f1f3f6;color:#192232;margin:0 auto;max-width:1300px;padding:24px}
section,header,footer{background:white;padding:20px;border-radius:12px;margin:0 0 16px}
header p{max-width:850px}h1,h2{margin:0 0 8px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
figure{margin:0}figure button{width:100%;padding:0;border:0;background:#f4f4f4;cursor:zoom-in}
img{display:block;width:100%;aspect-ratio:1;object-fit:contain}
label{display:block;margin:8px 0}input,select,textarea,button{font:inherit}
textarea{width:95%;max-width:700px}dialog{width:min(700px,86vw);padding:12px;border:0;border-radius:8px}
dialog button{display:block;margin:0 0 8px}
.export{background:#125f50;color:white;border:0;border-radius:6px;padding:10px 18px}
@media(max-width:730px){.grid{grid-template-columns:repeat(2,1fr)}body{padding:10px}}
</style>
<header>
<h1>商品背景替换盲评</h1>
<p>查看背景描述、输入与掩码，再比较结果 A 和 B。点击任意图片可放大。请独立评分，完成全部 8 个案例后下载 CSV。编号使用组内约定的匿名代号。</p>
<p>轮廓自然度关注新增商品部件、主体扩张、边缘光晕、纹理泄漏和接触阴影。背景符合度关注场景、材质、颜色及光照是否符合给定描述。两张相近时选平局。</p>
</header>
<form id="review">
<section><label>匿名编号 <input name="rater_code" autocomplete="off" required
 pattern="[A-Za-z0-9][A-Za-z0-9_\\-]{1,31}" placeholder="例如 Rater01"></label></section>
<div id="cards"></div>
<footer><button class="export" type="submit">下载评分 CSV</button>
<p id="status" role="status"></p></footer>
</form>
<dialog id="zoom"><button type="button">关闭放大图</button><img alt="放大图"></dialog>
<script type="application/json" id="data">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById('data').textContent);
const cards = document.getElementById('cards');
const zoom = document.getElementById('zoom');
const zoomImage = zoom.querySelector('img');
zoom.querySelector('button').onclick = () => zoom.close();
function figure(label, uri) {
 const panel = document.createElement('figure');
 const button = document.createElement('button');
 button.type = 'button';
 button.setAttribute('aria-label', '放大' + label);
 const img = new Image();
 img.src = uri; img.alt = label; img.loading = 'lazy';
 button.append(img);
 button.onclick = () => { zoomImage.src = uri; zoomImage.alt = label; zoom.showModal(); };
 const caption = document.createElement('figcaption');
 caption.textContent = label;
 panel.append(button, caption);
 return panel;
}
function selector(name, title) {
 const label = document.createElement('label');
 label.textContent = title + ' ';
 const select = document.createElement('select');
 select.required = true; select.name = name;
 for (const [text,value] of [['请选择',''],['A','A'],['B','B'],['平局','tie']]) {
  select.add(new Option(text,value));
 }
 label.append(select);
 return label;
}
for (const sample of data.cases) {
 const card = document.createElement('section');
 const title = document.createElement('h2'); title.textContent = sample.id;
 const prompt = document.createElement('p'); prompt.textContent = '背景描述　' + sample.prompt;
 const images = document.createElement('div'); images.className = 'grid';
 for (const [key,label] of [['input','输入'],['mask','前景掩码'],['A','结果 A'],['B','结果 B']]) {
  images.append(figure(label,sample.images[key]));
 }
 const bad = document.createElement('label');
 const checkbox = document.createElement('input');
 checkbox.type = 'checkbox'; checkbox.name = sample.id + '_both_unacceptable';
 bad.append(checkbox,' 两张均不可用');
 const note = document.createElement('label');
 note.textContent = '备注（选填） ';
 const textarea = document.createElement('textarea');
 textarea.name = sample.id + '_notes'; textarea.rows = 2; textarea.maxLength = 500;
 note.append(textarea);
 card.append(title, prompt, images, selector(sample.id + '_boundary','轮廓和背景衔接'),
             selector(sample.id + '_background','背景符合描述'), bad, note);
 cards.append(card);
}
const cell = value => {
 let text = String(value);
 if (/^[=+@-]/.test(text)) text = "'" + text;
 return '"' + text.replaceAll('"','""') + '"';
};
document.getElementById('review').addEventListener('submit',event => {
 event.preventDefault();
 const values = new FormData(event.currentTarget);
 const code = values.get('rater_code');
 const rows = [data.fields];
 for (const {id} of data.cases) {
  rows.push([code, data.form, id, values.get(id+'_boundary'), values.get(id+'_background'),
             values.has(id+'_both_unacceptable') ? 'yes' : 'no', values.get(id+'_notes')]);
 }
 const csv = '\\ufeff' + rows.map(row => row.map(cell).join(',')).join('\\r\\n') + '\\r\\n';
 const url = URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
 const link = document.createElement('a'); link.href = url;
 link.download = 'review_' + code + '.csv'; link.click();
 setTimeout(() => URL.revokeObjectURL(url), 1000);
 document.getElementById('status').textContent = '已导出。请将 CSV 交给组织者。';
});
</script>
</html>
"""
    path.write_text(html.replace("__DATA__", payload), encoding="utf-8")


def prepare(output):
    manifest = read_json(MANIFEST)
    setup = read_json(M3 / "setup.json")
    if setup["manifest_sha256"] != digest(MANIFEST):
        raise RuntimeError("The M3 results and manifest do not match")
    cases = manifest["cases"][:8]
    seed = manifest["seeds"][0]
    if len(cases) != 8 or seed != 2026:
        raise RuntimeError("The protocol requires eight fixed cases and seed 2026")

    packets = output / "images" / "packets"
    packets.mkdir(parents=True, exist_ok=True)
    (output / "images" / "responses").mkdir(exist_ok=True)
    rng = random.Random(613217)
    left = ["calibrated"] * 4 + ["uniform17"] * 4
    rng.shuffle(left)
    key = {"m3_manifest_sha256": digest(MANIFEST), "forms": {}}
    for form, reverse in (("1", False), ("2", True)):
        samples, mappings = [], {}
        for index, (case, left_method) in enumerate(zip(cases, left), start=1):
            identifier = f"R{index:02d}"
            right_method = next(method for method in METHODS if method != left_method)
            a, b = (right_method, left_method) if reverse else (left_method, right_method)
            folder = M3 / "images" / case["name"] / str(seed)
            samples.append({"id": identifier, "prompt": case["prompt"], "images": {
                "input": png_data(folder / "ddim50/input.png"),
                "mask": png_data(folder / "ddim50/foreground.png"),
                "A": png_data(folder / a / "composed.png"),
                "B": png_data(folder / b / "composed.png"),
            }})
            mappings[identifier] = {"case": case["name"], "row_idx": case["row_idx"],
                                    "seed": seed, "A": a, "B": b}
        rng.shuffle(samples)
        key["forms"][form] = mappings
        write_form(packets / f"review_form_{form}.html", form, samples)
    write_json(output / "organizer_key.json", key)


def summarize_responses(files, key):
    forms = key["forms"]
    votes = {case_id: {dimension: Counter() for dimension in DIMENSIONS}
             for case_id in forms["1"]}
    unacceptable = Counter()
    raters = {}
    for file in files:
        with file.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"{file.name}: unexpected response columns")
            rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError(f"{file.name}: malformed response row")
        codes = {row["rater_code"] for row in rows}
        versions = {row["form"] for row in rows}
        if len(codes) != 1 or len(versions) != 1:
            raise ValueError(f"{file.name}: expected one rater and one form")
        code, version = codes.pop(), versions.pop()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,31}", code) or version not in forms:
            raise ValueError(f"{file.name}: invalid anonymous code or form")
        if code.casefold() in raters:
            raise ValueError(f"{file.name}: duplicate anonymous rater")
        if len(rows) != len(votes) or {row["case_id"] for row in rows} != set(votes):
            raise ValueError(f"{file.name}: complete each case exactly once")
        raters[code.casefold()] = version
        for row in rows:
            case_id = row["case_id"]
            for dimension in DIMENSIONS:
                side = row[dimension]
                if side not in ("A", "B", "tie"):
                    raise ValueError(f"{file.name}/{case_id}: invalid {dimension} vote")
                choice = "tie" if side == "tie" else forms[version][case_id][side]
                votes[case_id][dimension][choice] += 1
            both = row["both_unacceptable"]
            if both not in ("yes", "no"):
                raise ValueError(f"{file.name}/{case_id}: invalid usability flag")
            unacceptable[case_id] += both == "yes"
    if not raters:
        raise ValueError("No human response CSV files found")
    dimensions = {}
    for dimension in DIMENSIONS:
        counts = sum((item[dimension] for item in votes.values()), Counter())
        dimensions[dimension] = {
            "counts": {label: counts[label] for label in (*METHODS, "tie")},
            "calibrated_share_tie_half": statistics.mean(
                (item[dimension]["calibrated"] + 0.5 * item[dimension]["tie"]) / len(raters)
                for item in votes.values()
            ),
        }
    return {
        "raters": len(raters), "form_counts": dict(Counter(raters.values())),
        "status": "pilot_only" if len(raters) < 3 else "descriptive_blind_review",
        "product_cases": len(votes), "ratings_per_dimension": len(raters) * len(votes),
        "primary_dimension": "boundary", "dimensions": dimensions,
        "both_unacceptable_total": sum(unacceptable.values()),
        "by_case": {case_id: {**item, "both_unacceptable": unacceptable[case_id]}
                    for case_id, item in votes.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output)
        print(args.output / "images" / "packets")
    else:
        key = read_json(args.output / "organizer_key.json")
        if key["m3_manifest_sha256"] != digest(MANIFEST):
            raise RuntimeError("The response key and M3 manifest do not match")
        files = sorted((args.output / "images" / "responses").glob("*.csv"))
        result = summarize_responses(files, key)
        target = args.output / "human_review_summary.json"
        write_json(target, result)
        print(target)


if __name__ == "__main__":
    main()
