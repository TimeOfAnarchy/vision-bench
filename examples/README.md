# Offline CPU example

From the repository root, activate a fresh Python 3.11+ environment and run:

```console
python -m pip install ".[cpu,demo]"
python examples/create_demo.py --output .local/demo-inputs
python -m vision_bench validate --dataset .local/demo-inputs/dataset.json
python -m vision_bench run --dataset .local/demo-inputs/dataset.json --profile .local/demo-inputs/profile.json --model-root .local/demo-inputs/model --output .local/demo-runs --run-id colors --warmup 1
python -m vision_bench recompute .local/demo-runs/colors
```

Open `.local/demo-runs/colors/report.html`. There are four successful rows and
three general outputs. Recomputation prints `{"matches": true}`. Use a new output
directory/run ID for another attempt; existing outputs are never replaced.

The script generates four solid-color images, algorithmic labels and a tiny
ONNX graph (channel means followed by sigmoid). It uses the CAFormer input
preprocessing contract, but **is not CAFormer or a trained image tagger**. There
are no external images, downloaded weights, credentials or server calls.
The generated inputs are covered by the repository's MIT license.

This example checks installation, input hashing, real CPU worker inference,
full-score recording, report generation and exact offline recomputation. Its
scores, perfect toy-label agreement and timings say nothing about real model
quality or performance. See [measurement guidance](../docs/reproducibility.md).
