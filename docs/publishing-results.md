# Publish an experiment report

Each tagger `run` freezes a dataset selection, model profile, complete scores,
outcomes and runtime metadata. `compare` computes paired quality on common
category vocabularies. `export-report` turns two or more completed compatible
runs into a portable, aggregate publication:

```console
vision-bench export-report ./runs/pixai ./runs/caformer --title "PixAI vs CAFormer" --dataset-description "Frozen holdout, source taxonomy annotations" --dataset-url https://example.org/my-dataset-selection --output .local/reports/my-comparison
```

Use your actual run directories and public selection URL; the names above do
not start either model or select a PixAI version. Models need their own reviewed,
hash-pinned profiles. Export verifies source artifacts, requires the same dataset
fingerprint/order/labels and does not run inference or modify the original runs.
An existing output directory is refused.

The new directory contains:

- `index.html`: standalone comparison page, with no scripts or remote assets;
- `README.md`: Markdown tables and Static Space configuration;
- `results.json`: aggregate values, dataset identity/hashes, model revisions,
  profile thresholds/artifact hashes, software versions and measured hardware;
- `SHA256SUMS`: hashes of the three report files.

New tagger runs capture CPU identifier, logical CPU count, RAM, OS and best-effort
NVIDIA model/VRAM/driver inventory. This probe runs after inference timings, has
a bounded timeout, and records no hostname, username, serial number or GPU UUID.
It describes visible devices, not verified session-to-device binding. A failed
or unavailable probe does not fail the model run. Loaded CUDA/cuDNN versions,
power settings and competing workloads are not automatically attested.

Historical runs cannot recover hardware retroactively: missing fields remain
`not_recorded`. `--hardware-note "Author-declared hardware and conditions"` adds
an explicitly attributed note without replacing measured metadata. Export never
reads the current machine's GPU to describe a historical run.

## Reading the tables

Paired quality uses only images successful in every model, and only common
category outputs. Full-truth scores also count positive source tags outside that
common vocabulary. Native quality uses each model's entire vocabulary and own
successful images; its denominator can differ. Missing-category comparisons are
retained as exclusions in JSON. Timing uses each run's own warm successful rows,
not the paired-quality mask. Compare timing only under matching conditions.

Silver annotations are incomplete source labels, not verified human ground truth.
A reported F1 is therefore not proof of moderation safety. WDDM GPU memory remains
a sampled process high water including model loading, not an allocator peak.

## Upload manually to Hugging Face

Open `index.html` locally and review its tables and the full aggregate JSON first.
Create a separate **Static / Blank Space** for this experiment, then upload all
four generated files to its root. The generated README configures `index.html`
as its application. Keep the package's download Space separate from individual
experiments. Creating or uploading a report is a manual step; this command never
logs in or publishes.

The export omits raw scores, per-label breakdowns, sample identities, images,
request/log files, local directories and arbitrary dataset source metadata.
It still contains model/run/dataset names and any description, URL or hardware
note you explicitly supply. Review these before sharing. A fingerprint alone
does not reconstruct which images were selected: supply a public, authorized
selection reference when others need exact reproduction. Dataset and image
redistribution rights remain separate from the tool's MIT license.
