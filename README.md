# Vision Bench

Local, reproducible comparisons of image tagging models. Run one model, keep its
complete predictions, then compare it with other saved runs without repeating inference.

The first release provides CAFormer B36 and PixAI v0.9 ONNX profiles, full-score
shards, per-category metrics, isolated workers, JSON comparison and a local HTML
report. VLM, resource collectors and dataset downloaders are not included in
this release.

## Install

Python 3.11+ is required. Choose **one** runtime extra per environment:

```console
python -m venv .venv
# Activate .venv, then:
python -m pip install ".[cpu]"
# Windows DirectML: python -m pip install ".[dml]"
# CUDA: python -m pip install ".[cuda]"
```

Different ONNX runtime distributions share a Python import. Use separate
environments for CPU, DirectML and CUDA. Availability of an extra does not prove
that a specific GPU/provider combination has been tested.

## Prepare local inputs

Use existing local images and model files. Vision Bench does not download or
redistribute model weights or datasets. Model packs for the bundled profiles
must match the recorded artifact hashes. A pack contains `model.onnx`,
`selected_tags.csv` and `preprocess.json`.

Create `samples.jsonl`, one object per image:

```json
{"id":"image-001","image":"images/001.jpg","sha256":"<image SHA-256>","tags":{"general":["1girl"],"character":[]}}
```

Create `dataset.json` next to the samples file:

```json
{
  "schema_version": 1,
  "id": "local-holdout",
  "revision": "v1",
  "annotation_kind": "silver",
  "image_root": ".",
  "samples": "samples.jsonl",
  "samples_sha256": "<SHA-256 of samples.jsonl>"
}
```

The `image_root` may point outside the manifest directory. Sample image paths
stay relative to that root, and `samples` stays relative to the manifest.
Every sample declares the same annotation categories, including empty lists.
Use `human` only for human-reviewed labels; `silver` means incomplete source labels.

```console
vision-bench validate --dataset ./dataset.json
vision-bench run --dataset ./dataset.json --profile builtin:caformer-b36 --model-root ./caformer-pack --output ./runs
vision-bench run --dataset ./dataset.json --profile builtin:pixai-v0.9 --model-root ./pixai-pack --output ./runs
```

Add `--provider DmlExecutionProvider` or `--provider CUDAExecutionProvider` to
select that provider explicitly. Whole-session fallback/retry is rejected; CPU graph
partitions remain allowed and are recorded. `--limit 128` selects a deterministic
prefix and fingerprints that exact subset. No large run starts implicitly.

## Inspect and compare

Each successful run contains `run.json`, `profile.json`, `samples.jsonl`,
`vocabulary.json`, `records.jsonl`, `scores-*.npz`, `metrics.json` and `report.html`.
The vocabulary preserves every model output, including categories not evaluated
by a particular dataset. Scores are float32, with no top-K or policy filtering.
Unnamed indexed outputs are retained as `null` labels with category `unnamed`;
their scores keep their original column positions and are not evaluated as tags.
Failed samples have NaN rows and a separate failure mask, never fake zero scores.

```console
vision-bench recompute ./runs/my-run
vision-bench compare ./runs/model-a ./runs/model-b --output ./comparison.json
```

Recomputation verifies artifact hashes, needs no images or model session, and
does not modify the run. Comparison uses category-consistent common vocabulary
and reports the number of jointly successful samples. Output files and run IDs
are never silently reused. An interrupted or failed run is diagnostic evidence,
not a completed experiment.

Comparison `micro_representable` and mAP evaluate only common outputs. Its
`micro_full_truth` includes positives outside that common vocabulary; it does
not use the model's other native predictions. For native full-vocabulary quality,
read each run's own metrics. Categories lacking thresholds in any compared model
are explicitly excluded, rather than presented as a one-model comparison.

F1 treats missing source annotations as negatives **for the metric only**.
Unlabeled predictions are not verified false positives. Full-truth recall/F1
include out-of-vocabulary positive labels; mAP is over represented labels with
positive support. Native rating heads retain their own labels; they are not
converted into application moderation levels. Operating thresholds are fixed
profile inputs, not fitted on the evaluation set.

Latency includes decoding, preprocessing and inference, excludes integrity
checks and model loading. The first five selected rows are excluded from the
warm summary by default. GPU/RAM sampling is not implemented; reports explicitly
mark GPU memory unavailable. No external service is started or stopped.

The first evaluator limits each selected score matrix to 512 MiB before starting
inference. Shards bound recording buffers, but metric computation currently
loads the selected matrix into RAM and needs additional working memory.
Static single-frame images only, up to 12 MP / 8192 pixels per edge. Preprocessing
uses original pixel orientation without an implicit EXIF rotation.

## Develop

```console
python -m pip install -e ".[dev,cpu]"
python -m pytest
python -m ruff check .
python -m build
```

The tests use tiny generated models and local temporary images. They do not
download weights or invoke a real GPU. Real model checks are separate manual runs.
No model weights or evaluation images are included in the package.
