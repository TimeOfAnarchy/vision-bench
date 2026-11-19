# Image tagging guide

[Back to the project overview](../README.md).

## Prepare local inputs

Use existing local images and model files. Vision Bench does not download or
redistribute model weights or datasets. Model packs for the bundled profiles
must match the recorded artifact hashes. A pack contains `model.onnx`,
`selected_tags.csv` and, where declared, `preprocess.json`. CL uses
`model_vocabulary.json`, `model_metadata.json` and external `model.onnx.data`
weights. Camie uses `camie-tagger-v2.onnx` and `camie-tagger-v2-metadata.json`.
PixAI v1.0 Mexes FP16 uses `model_fp16.onnx`, `selected_tags.csv` and
`preprocess.json`. Its CSV indices are checked, category 1 is preserved as
`style`, and its output already contains probabilities. The profile uses the
export's documented thresholds and a 1008px input; FP16 is intended for GPU use.
Every required artifact is hash-verified, including external CL weights.

Create `samples.jsonl`, one object per image:

```json
{"id":"image-001","image":"images/001.jpg","sha256":"<image SHA-256>","tags":{"general":["blue_sky"],"character":[]}}
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
vision-bench run --dataset ./dataset.json --profile builtin:caformer-b36 --model-root ./caformer-pack --output ./runs --run-id model-a
vision-bench run --dataset ./dataset.json --profile builtin:pixai-v0.9 --model-root ./pixai-pack --output ./runs --run-id model-b
# Other profiles: builtin:wd-eva02-v3, builtin:cl-v2, builtin:camie-v2,
# builtin:caformer-b36-amp-fp16
# builtin:pixai-v1.0-mexes-fp16
```

Add `--provider DmlExecutionProvider` or `--provider CUDAExecutionProvider` to
select that provider explicitly. Whole-session fallback/retry is rejected; CPU graph
partitions remain allowed and are recorded. `--limit 128` selects a deterministic
prefix and fingerprints that exact subset. No large run starts implicitly.

## Inspect and compare

To turn saved tagger runs into a standalone HTML/Markdown/JSON experiment report
with model, dataset, software and hardware provenance, see
[publishing results](publishing-results.md). Exporting does not publish anything.

Each successful run contains `run.json`, `profile.json`, `samples.jsonl`,
`vocabulary.json`, `records.jsonl`, `scores-*.npz`, `metrics.json` and `report.html`.
The vocabulary preserves every model output, including categories not evaluated
by a particular dataset. Scores are float32, with no top-K or policy filtering.
Unnamed indexed outputs are retained as `null` labels with category `unnamed`;
their scores keep their original column positions and are not evaluated as tags.
Failed samples have NaN rows and a separate failure mask, never fake zero scores.

```console
vision-bench recompute ./runs/model-a
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
positive support. Native output categories retain their original semantics.
Operating thresholds are fixed profile inputs, not fitted on the evaluation set.

Each compared category also gets a `bootstrap` block: 95% percentile intervals
for every run's micro, macro-F1 and mAP metrics, and for each run's difference
from the first listed run (the reference). Every resample draws the paired
successful samples with replacement and applies the same draw to every run, so
differences stay paired. `estimate` is the observed value on all paired samples.
A difference interval that includes zero means this selection does not separate
the models. Intervals reflect only which images were selected: labels,
thresholds and models stay fixed, and label noise is not covered.

```console
vision-bench compare ./runs/model-a ./runs/model-b --output ./comparison.json --bootstrap-resamples 2000 --bootstrap-seed 1
```

The default is 1,000 resamples with seed 0; `--bootstrap-resamples 0` disables
intervals. The seed and resample count are recorded in the comparison.

Fixed-threshold F1 mixes ranking quality with each profile's threshold choice.
Each compared run therefore also has a `threshold_analysis` over the same common
outputs and paired samples, micro-averaged over sample-label pairs:

- `true_positive` and `unlabeled_positive` at every threshold from 0 to 1 in
  steps of 0.001 (`score >= threshold` in float32). The index of a profile
  threshold reproduces its fixed-threshold micro counts.
- `precision_at_recall` and `recall_at_precision` at 0.5/0.6/0.7/0.8 compare
  models at matched operating points; `null` means the target is unreachable.
- `best_f1` chooses its threshold on the evaluation set itself. It shows how much
  a threshold change could gain, not a deployable operating point.
- `calibration` bins scores into ten equal-width bins with mean score and
  annotated positive rate, plus expected calibration error. Most pairs score
  near zero, so the overall error is tiny; `upper_bins` repeats it for scores
  >= 0.1, where tags are actually decided. Incomplete silver labels understate
  the true positive rate, so treat calibration as a relative signal.

Micro metrics weight images by how many tags they have. Each compared run also
has a `per_image` block on the same common outputs that weights every image once,
closer to what one tagged image looks like:

- per-image precision, recall and F1 (common-output and full-truth) at the
  profile threshold, averaged over the images that define each metric; the
  `images` count beside each value is its denominator;
- `sample_recall_at_5/10/20`: annotated common-output positives among each
  image's k highest scores, divided by its annotated positives. Positives tied at
  the k-th score share the remaining slots evenly, so column order never matters;
- full-truth F1 percentiles (p10/p25/median), images without any prediction,
  images with annotated common-output positives but no hits (a model miss), and
  images whose positives are all outside the common vocabulary (a coverage gap,
  which full-truth recall also counts).

The image-averaged metrics also get paired bootstrap intervals and differences.

Macro-F1 and mAP average labels equally, so labels seen once or twice make them
noisy. `by_support` regroups each run's per-label results by how many paired
images carry the label (1, 2-4, 5-19, 20-99, 100+): label and occurrence counts,
macro-F1, mAP and micro precision/recall/F1 per stratum. Strata come from the
shared annotations, so they hold the same labels for every compared run. Labels
without annotated positives, and their unlabeled predictions, are in no stratum.

## Audit silver labels

Every metric above counts an unlabeled prediction as an error, although many are
tags the source simply omitted. `label-audit` exports a uniform random sample of
unlabeled predictions per run (common outputs, profile thresholds, paired
samples) as a JSONL review packet; `label-audit-score` turns reviewed verdicts
into an audited precision estimate:

```console
vision-bench label-audit ./runs/model-a ./runs/model-b --category general --per-run 100 --output ./audit.jsonl
vision-bench label-audit-score ./audit.jsonl ./runs/model-a ./runs/model-b --output ./audit-result.json
```

Each line names the image, label, annotated tags, every run's score and the runs
whose sample includes it. A prediction sampled by several runs appears once, so
one verdict serves all of them. Set `human_review.verdict` to `present`,
`absent` or `unsure` (or leave it `null`). Lines are shuffled: reviewing any
prefix keeps each run's reviewed subset uniform.

Scoring rebuilds the sample from the runs, seed and category and refuses a
packet whose items differ, so pass the same runs in export order. Per run it
reports verdict counts, the present rate among `present`/`absent` verdicts
(Wilson 95% interval; `unsure` and pending lines are excluded), silver precision
and audited precision `(TP + unlabeled x present rate) / predicted`. Recall
cannot be corrected this way: tags missed by both the source and the model stay
unknown. The packet contains local image paths and sample identifiers; review it
locally and treat verdicts as one reviewer's judgement.

Latency includes decoding, preprocessing and inference, excludes integrity
checks and model loading. The first five selected rows are excluded from the
warm summary by default. GPU memory is unavailable unless sampling is requested
and produces sufficient observations. No external service is started or stopped.

Scores are loaded one shard at a time into temporary disk-backed matrices;
metrics use bounded column blocks. Recording shards are limited to 64 MiB of score
data. Temporary disk space is required for one full float32 matrix per open run;
precision comparison also uses a temporary error matrix. Set a suitable cache
location when needed:

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
peak, and includes model loading plus inference. Process RAM usage is not measured;
total host RAM is recorded separately as hardware inventory.

CUDA requires compatible CUDA/cuDNN DLLs, which can be loaded from NVIDIA
site packages. An optional `VISION_BENCH_CUDNN_DIR` selects
an explicit cuDNN directory in the worker environment; it never changes global
settings. See [ONNX Runtime DLL loading](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls).
Use separate runtime environments for provider-specific measurements.

### PyTorch CAFormer resolution experiments

An optional `torch` extra supports local CAFormer B36 safetensors with `timm`;
it never downloads weights or executes model-repository code. Use a CUDA-enabled
PyTorch runtime and a custom profile with `backend: "torch-caformer"`,
`provider: "TorchCUDA"`, `precision: "fp32"` or `"amp-fp16"`, and
`output_name: "logits"` / `output_kind: "logits"`. Pin the `model`, `vocabulary`,
`config` and `preprocess` artifacts by SHA-256. The config class count and ordered
tags must match the CSV, and weights load strictly without changing the head.

`caformer-b36` retains the original 512px bilinear padding followed by a 384px
bicubic resize. For controlled resolution experiments, use
`caformer-b36-direct-384`, `caformer-b36-direct-512` or
`caformer-b36-direct-640`: each resizes the original directly with bicubic
interpolation, preserves aspect ratio, uses white padding and ImageNet
normalization. The direct 384px path differs from the original two-stage path;
include an original-path control when assessing that preprocessing change.
Keep weights, precision and thresholds fixed across resolutions. AMP leaves
stored weights in FP32, uses FP16 autocast, and applies sigmoid in FP32.
PyTorch timings include the output transfer to CPU and CUDA synchronization;
cross-runtime timings describe the complete measured pipelines.

Static single-frame images only, up to 12 MP / 8192 pixels per edge. Preprocessing
uses original pixel orientation without an implicit EXIF rotation.
