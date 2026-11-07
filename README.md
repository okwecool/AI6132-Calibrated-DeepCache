# AI6132 DeepCache pilot

Small SD1.5 inference experiment comparing DDIM, DPM-Solver++, official uniform
DeepCache, a paper-equation nonuniform schedule, and our calibration-derived
fixed-budget refresh schedule. No model training is required.

## Local environment

The project `.venv` inherits the existing, GPU-tested environment at
`E:\Environment\Anaconda3\envs\env_cuda` through `--system-site-packages`.
New packages are installed only inside this project. This is a local convenience,
not a portable environment: on another machine install compatible CUDA PyTorch,
then the packages in `requirements.txt`. Tested package versions, GPU and model
revision are recorded in `results/pilot/environment.json`.

```powershell
& 'E:\Environment\Anaconda3\envs\env_cuda\python.exe' -m venv --system-site-packages --without-pip .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' scripts/prepare_assets.py
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& '.\.venv\Scripts\python.exe' scripts/run_pilot.py --stage smoke
& '.\.venv\Scripts\python.exe' scripts/run_pilot.py --stage all
& '.\.venv\Scripts\python.exe' scripts/evaluate_pilot.py
& '.\.venv\Scripts\python.exe' scripts/write_pilot_report.py
```

Only asset preparation needs network access. Model and evaluator caches are
stored inside the project. All experiment/evaluation code loads assets offline.
Asset preparation records Hugging Face revisions and an annotation SHA256.

## Experiment protocol

- SD1.5 FP16, 512x512, batch size 1, CFG 7.5; PyTorch SDPA; no CPU offload.
- COCO 2014 validation caption annotations from the official COCO caption
  evaluation repository; use lowest annotation ID per image, shuffle image IDs
  with seed 6132, split 4 calibration / 4 validation / 8 test images. The manifest
  preserves all captions and IDs. This pilot uses captions only, not reference
  photograph pixels. The earlier proposed COCO2017 all-annotation archive stalled
  during download; the 2014 annotation file is the actual dataset used here.
- Test seeds 2026 and 6132: 16 paired cases, 5 methods, 80 test images.
- DDIM uses 50 steps with eta=0; each cache policy uses the same branch 0 and
  exactly 17 full refreshes. Original module-cache logic comes from DeepCache.
- Nonuniform uses a signed-real-power interpretation of paper Eq.7, with two
  center candidates selected on validation LPIPS. It is not a bit-exact run of
  the legacy pipeline; the rounding/real-power adaptation is disclosed.
- Calibration uses pooled actual cache-boundary features on four uncached
  trajectories. A dynamic program optimizes additive normalized feature drift,
  with max interval length 6. Its objective is a proxy, not guaranteed image error.
- DPM-Solver++ step count is determined from validation latency, rounded to an
  integer, and then frozen. Actual test latency need not match cache latency.
- Test job order is seeded and shuffled. Initial latent tensors are generated on
  CPU from each seed and reused consistently across methods. Each method gets a
  fresh cache state. CUDA is synchronized for generation time; PNG disk writes
  and quality scoring are excluded. GPU-event U-Net timing is also recorded.
- Test timing is one measured generation per case/method, after kernel warmup.
  Sample medians are useful for this pilot but are not repeated-run latency CIs.
- CLIP is raw normalized ViT-B/32 text-image cosine. LPIPS/PSNR compare to the
  identical-prompt/latent DDIM50 output, not a ground-truth photo. These measure
  reference deviation; neither proves realism. No small-sample FID claim is made.
- Approximate uncertainty uses paired bootstrap over the eight prompt/image
  clusters, retaining both seeds together. This tiny pilot is exploratory.

## Implementation and output

- `src/cache_policy.py`: our schedule wrapper, nonuniform schedule construction,
  feature-drift objective and dynamic program.
- `scripts/run_pilot.py`: smoke checks, calibration, validation, randomized test.
- `scripts/evaluate_pilot.py`: LPIPS/CLIP, paired differences, tables and figures.
- `tests/test_policies.py`: schedule invariants and DP verification by independent
  exhaustive enumeration; numerical cache equivalence is checked on the real GPU.
- `results/pilot/measurements.jsonl`: raw timings/configurations for every image.
- `results/pilot/evaluated_rows.csv`: test outputs and quality scores.
- `results/pilot/summary.json`: method summaries and exploratory paired intervals.
- `results/pilot/quality_latency.png`, `refresh_schedules.png`, `comparison_*.jpg`:
  figures and fixed-order held-out examples.

The runner can resume saved images using their measurement keys. Use a new output
directory or explicitly archive old results before changing model/config/data;
do not combine incompatible experiments in the same directory.

## Attribution

DeepCache is by Xinyin Ma, Gongfan Fang and Xinchao Wang (CVPR 2024),
https://github.com/horseee/DeepCache, Apache-2.0. Its module wrapping/caching
implementation is inherited from the installed DeepCache 0.1.1 package. Our
changes concern refresh scheduling, instrumentation and calibration, not the
underlying cache architecture. See `third_party/DeepCache-LICENSE`.

Other components: Hugging Face Diffusers (Apache-2.0), LPIPS (BSD-2-Clause),
OpenAI CLIP (MIT), and pretrained SD1.5 weights (CreativeML Open RAIL-M).
Retain relevant upstream licenses and cite the original papers in the report.
Do not include `.venv`, caches, models or datasets in the final course code ZIP.

## Second iteration: measured-error calibration

The second experiment is isolated in `results/round2` and
`manifests/round2.json`. All 8 calibration, 8 validation and 16 test image IDs
are disjoint from all first-round IDs. Test uses 2 seeds: 32 paired cases,
6 methods, 192 test images. The old feature objective is recalibrated using the
same new calibration prompts as the revised objectives, to isolate the objective
change instead of comparing unrelated round means.

```powershell
& '.\.venv\Scripts\python.exe' scripts/run_round2.py --stage all
& '.\.venv\Scripts\python.exe' scripts/plot_round2_diagnostics.py
& '.\.venv\Scripts\python.exe' scripts/evaluate_pilot.py --run-dir results/round2
& '.\.venv\Scripts\python.exe' scripts/write_round2_report.py
```

`--stage diagnose` stops after offline probes and schedule construction.
The 8 calibration trajectories are replayed at identical inputs. For each cache
refresh point, partial inference is probed at ages 1 through 5, giving 1880 local
probes in total. Full-path replay is checked against saved reference predictions.
These are correlated trajectory measurements, not 1880 independent images.

Two revised objectives are predefined: CFG-guided noise MSE and the corresponding
DDIM one-step latent-update MSE. DDIM's analytical epsilon sensitivity is checked
against the actual scheduler update in unit tests. Both costs use the same
17-refresh budget and max gap 6. The two candidates are selected using validation
LPIPS only; the selected list is frozen before test generation. No extra U-Net
evaluations run during test-time inference. Offline diagnostic/selection time is
reported separately.

Counterfactual probes use the uncached reference latent trajectory. They measure
local errors and do not directly model accumulated errors on a cached rollout.
No claim of optimal final image quality follows from minimizing their sum.

## Frozen confirmation and budget ablation

The third run uses `manifests/confirmation.json` and `results/confirmation`.
It freezes the round2 step-MSE policy, paper-schedule parameters and DPM-Solver++
step count before generating 100 new prompts with one fixed seed. All 100 image
IDs exclude the 48 IDs previously used in any calibration, validation or test.
There are five methods, hence 500 main-test images. The predefined primary
comparison is calibrated minus uniform mean LPIPS, paired by prompt; CLIP,
latency, other baselines and budget analyses are secondary.

The first 16 manifest prompts are a fixed budget-ablation subset. Budgets 10, 17
and 25 use the same prompts and latent seeds. Existing main-test reference and
budget17 images are reused; budgets 10/25 add 64 images, for 564 total. The new
budget schedules derive solely from round2 calibration costs, without refitting
or using confirmation outcomes. No new training, calibration probes or model
downloads are needed.

```powershell
& '.\.venv\Scripts\python.exe' scripts/run_confirmation.py
& '.\.venv\Scripts\python.exe' scripts/repair_interrupted_timings.py
& '.\.venv\Scripts\python.exe' scripts/evaluate_pilot.py --run-dir results/confirmation
& '.\.venv\Scripts\python.exe' scripts/summarize_confirmation.py
```

Generation can resume using saved measurement keys. The evaluator requires both
main generation and ablation completion markers before loading quality models.
Primary and ablation rows are scored together but summarized separately, so the
16-prompt subset does not inflate the main independent sample count.

If a long system interruption is observed, timing repair preserves the original
raw row and quality image. It remeasures once only when wall time exceeds 60s
and wall-minus-GPU-event time also exceeds 60s; the rerun must reproduce pixels
exactly. The evaluated CSV retains original timing fields and a repair flag.

## Git baseline and review workflow

The initial Git commit records the existing pilot, measured-error calibration,
and frozen confirmation work as a single baseline. It does not reconstruct an
earlier commit history. Separate experiment manifests, reports, measurements,
source snapshots, and hashes retain the provenance of the three rounds.

Tracked files include the source code, tests, dependency list, course project
brief, project notes, manifests, compact calibration arrays, statistics, and
summary figures. Full generated image sets, large trajectory tensors, lecture
extracts/previews, downloaded models, raw datasets, environments, and caches are
ignored and remain on the local disk. A Git clone therefore contains the saved
analysis but not every asset needed to regenerate or rescore images. In
particular, completion markers record the original local run; they do not mean
that its generated images are present in a clone. Restore or regenerate the
missing assets before image-level evaluation. Use separate output directories
for new experiments so the baseline records are preserved.

Future changes are implemented and checked in the working tree, then presented
to the user for review. Create a commit only after the user explicitly approves
those changes. The initial-commit authorization does not cover later commits.
See `AGENTS.md` for the repository working rules.

## Product background replacement: design

The [product background replacement design](docs/product_background_replacement_design.md)
defines the research protocol. The M1 runner uses a dedicated SD inpainting
checkpoint, a supplied foreground mask, and optional DeepCache refreshes.
Prepare the pinned model once, then run inference offline.

```powershell
& '.\.venv\Scripts\python.exe' scripts/prepare_background_assets.py
& '.\.venv\Scripts\python.exe' scripts/run_background.py smoke --output results/product_background/recheck
& '.\.venv\Scripts\python.exe' scripts/run_background.py generate --image input.png --foreground mask.png --prompt "a bottle on a pale stone counter, studio lighting" --steps 50 --interval 3 --output results/product_background/demo
```

Use a foreground mask aligned with the source image. White pixels protect the
product and black pixels mark the editable background. The runner fits both to
the same 512×512 canvas and saves the prepared inputs, raw generation, and
composite. The composite restores the prepared foreground pixels exactly.
Omit `--interval` for uncached DDIM. Use a new output directory for each run.
See the [M1 validation report](results/product_background/m1/report.md) for
the GPU checks, timing scope, and the observed foreground boundary artifact.

The [real product photo pilot](results/product_background/photo_pilot/report.md)
compares two prompt styles and two cache settings on four Finegrain examples.
The manifest records the source rows, file hashes, prompts, and seeds.

```powershell
& '.\.venv\Scripts\python.exe' scripts/run_background_photo_pilot.py --prepare
& '.\.venv\Scripts\python.exe' scripts/run_background_photo_pilot.py --output results/product_background/photo_pilot_recheck
```
