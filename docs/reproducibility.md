# Reproducing and interpreting a run

Start with the [generated CPU example](../examples/README.md). No external data
or model weights are required. For real experiments, obtain images and models
under their respective terms, then create local manifests and hash-pinned profiles.
The software's MIT license does not grant rights to third-party weights, images,
source annotations or model outputs. No such assets are redistributed here.

## Inputs and measurements

Record dataset revision, exact selected IDs/image hashes, source annotation
provenance, model repository revision and every pack artifact hash. Bundled
profiles record particular exports; another export needs its own reviewed
profile. Keep category semantics and the selected images aligned. Record differences
in preprocessing, thresholds and providers: `compare` evaluates those complete
configured pipelines. Document threshold provenance; do not tune on the holdout.
Controlled precision comparisons additionally require the same pinned model
repository/revision, indexed vocabulary, preprocessing implementation, provider,
thresholds and successful-sample mask.

Record Python/package/runtime versions, OS, GPU/driver, power settings, competing
processes and warmup count with your experiment. Run one model per worker and
avoid competing GPU workloads. Share numerical tolerances and failed sample
counts alongside metrics. Reproducing a manifest does not guarantee identical
timings, device kernels or floating-point scores on another machine.

Latency covers decode/preprocessing/inference, excluding integrity checks and
model loading. WDDM memory sampling is Windows-only and reports an observed
process high water including loading, with dedicated/shared memory kept separate.
It is not an instantaneous allocator peak or server VRAM measurement. Host RAM
inventory describes total physical RAM; process RAM usage is not measured.
CUDA, DirectML, GPU memory and real VLM endpoints require separate manual
evidence; the CPU CI suite does not validate them. See [VLM constraints](vlm.md).

## Labels and conclusions

Incomplete source tags remain silver labels. Missing tags count as negatives
for the stated metric, without proving that unlabeled predictions are incorrect.
Synthetic example labels are algorithmic, not human annotations. Native output
categories retain their original semantics. Review exports are proposals until
separately reviewed; failure/unknown/abstention states remain explicit outcomes.

`recompute` verifies saved artifacts and compares metrics exactly without
inference or editing the original run. `compare` uses paired successful samples
and common vocabulary; keep denominators visible. `precision` requires aligned
export variants and reports numerical errors and threshold flips. Keep raw
shards, error records and original artifacts when publishing a derived report.

Run artifacts may contain local paths, source metadata, image identifiers,
prompts and server outputs. Review the exact files you choose to share. The
default Git exclusions keep local runs/models/datasets out of this source tree;
they do not anonymize externally shared artifacts.

## Automated checks

The GitHub workflow builds wheel and source archives, installs the wheel in a
fresh environment outside the checkout, runs tests shipped in the source archive,
and executes the generated example. It covers Python 3.11/3.13 on Linux and
Python 3.13 on Windows using CPU ONNX Runtime. It does not download model weights,
start inference servers or publish packages. Consult the actual
[CPU package check results](https://github.com/TimeOfAnarchy/vision-bench/actions/workflows/ci.yml)
for the tested commit; CPU CI does not establish real-model or GPU performance.
