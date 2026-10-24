# Vision Bench

Local, reproducible comparisons of image tagging models. Run one model, keep its
complete predictions, then compare it with other saved runs without repeating inference.

The package provides CAFormer B36 FP32/AMP FP16, PixAI v0.9, WD EVA02 v3, CL v2
and Camie v2 ONNX profiles, full-score shards, per-category metrics, isolated
workers, precision comparisons and optional Windows GPU memory sampling.
VLM and dataset downloaders are not included.

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
`selected_tags.csv` and, where declared, `preprocess.json`. CL uses
`model_vocabulary.json`, `model_metadata.json` and external `model.onnx.data`
weights. Camie uses `camie-tagger-v2.onnx` and `camie-tagger-v2-metadata.json`.
Every required artifact is hash-verified, including external CL weights.

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
# Other profiles: builtin:wd-eva02-v3, builtin:cl-v2, builtin:camie-v2,
# builtin:caformer-b36-amp-fp16
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
warm summary by default. GPU memory is unavailable unless sampling is requested
and produces sufficient observations. No external service is started or stopped.

Scores are loaded one shard at a time into temporary disk-backed matrices;
metrics use bounded column blocks. Recording shards are limited to 64 MiB of score data. Temporary disk
space is required for one full float32 matrix per open run; precision comparison
also uses a temporary error matrix. Set a suitable cache location when needed:

```console
vision-bench --cache-dir ./cache compare ./runs/model-a ./runs/model-b --output ./comparison.json
vision-bench --cache-dir ./cache precision ./runs/fp32 ./runs/fp16 --output ./precision.json
```

Precision pairs require the same pinned upstream repository/revision, exact
indexed vocabulary, preprocessing implementation, provider and thresholds.
Both runs must have the same successful sample mask; failed outcomes remain visible.
The report includes exact absolute-error percentiles, threshold flips, affected
samples and paired quality metrics. Numeric errors include anonymous outputs;
tag metrics and threshold decisions exclude them. A close aggregate F1 does not
prove decision-by-decision equivalence. No thresholds are optimized on this set.
The AMP FP16 profile runs an exported mixed-precision ONNX graph, not PyTorch.

```console
vision-bench run --dataset ./dataset.json --profile builtin:caformer-b36 --model-root ./caformer-pack --provider DmlExecutionProvider --measure-memory
```

On Windows, optional [PDH process counters](https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw)
sample the worker's dedicated/shared GPU memory. The actual worker reports its PID
before model loading; a virtual-environment launcher PID is recorded separately.
Raw samples are stored in `memory.jsonl`. The default interval is 250 ms; a shorter
or insufficient sampling span is not reported as a memory high-water estimate.
Unavailable/failed counters remain unavailable, never fake zero measurements.
Observed high water is a sampled process metric, not an instantaneous allocator
peak, and includes model loading plus inference. RAM collection is not implemented.

CUDA requires compatible CUDA/cuDNN DLLs, which can be loaded from NVIDIA
site packages. An optional `VISION_BENCH_CUDNN_DIR` selects
an explicit cuDNN directory in the worker environment; it never changes global
settings. See [ONNX Runtime DLL loading](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls).
Use separate runtime environments for provider-specific measurements.

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
