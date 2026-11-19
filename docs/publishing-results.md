# Publish an experiment report

Each tagger `run` freezes a dataset selection, model profile, complete scores,
outcomes and runtime metadata. `compare` computes paired quality on common
category vocabularies, threshold-free operating points, calibration and
per-image quality, with
paired bootstrap intervals for each model's difference
from the first listed run (`--bootstrap-resamples`/`--bootstrap-seed` apply to
`export-report` too). `export-report` turns two or more completed compatible
runs into a portable, aggregate publication:

```console
vision-bench export-report ./runs/model-a ./runs/model-b --title "Autotagging comparison" --dataset-description "Frozen holdout, source taxonomy annotations" --output .local/reports/my-comparison
```

Use completed run directories, such as those in the [tagging guide](taggers.md).
Models need their own reviewed, hash-pinned profiles. Export verifies source
artifacts, requires the same dataset fingerprint/order/labels and does not run
inference or modify the original runs. An existing output directory is refused.
Add `--dataset-url` when you have a public, authorized reference for the exact
image selection. Omit it when that reference is unavailable and state the
reproduction limit in `--dataset-description`.

The new directory contains:

- `index.html`: standalone comparison page, with no scripts or remote assets;
- `README.md`: Markdown tables and Static Space configuration;
- `results.json`: aggregate values, dataset identity/hashes, model revisions,
  profile thresholds/artifact hashes, software versions and measured hardware;
- `SHA256SUMS`: hashes of the three report files.

New tagger runs capture CPU identifier, logical CPU count, total host RAM, OS and best-effort
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
A reported F1 describes agreement with those labels. WDDM GPU memory remains a
sampled process high water including model loading, not an allocator peak.

## Upload manually to Hugging Face

Open `index.html` locally and review its tables and the full aggregate JSON first.
Publish all four files together in a **Static Space**. Choose either layout:

- **Existing Space:** upload the report into a distinct subdirectory, such as
  `experiments/my-comparison/`. Preserve its relative links and the existing root
  README's application configuration. Add a link to the report from the homepage.
- **Dedicated Space:** upload the four files to its root. The generated README
  declares `sdk: static` and `app_file: index.html` for this layout.

This project's [Space](https://huggingface.co/spaces/TimeOfAnarchy/vision-bench)
uses the first layout. The tracked `site/index.html` is the homepage source;
experiment pages and assets are uploaded separately. The worked autotagging case
is under `experiments/autotagging/`, with its exported aggregate report in `data/`.
Those experiment artifacts are kept outside the source repository and package.

Use the Space's actual application URL for report links. This project's static
host is `https://timeofanarchy-vision-bench.static.hf.space/`. Packaged downloads
are maintained in [GitHub releases](https://github.com/TimeOfAnarchy/vision-bench/releases/latest).
Creating or uploading a report is a manual step; `export-report` never logs in or
publishes. Hosts may inject HTML into rendered pages; verify exported checksums
against the original uploaded files in the Space's Files view.

The export omits raw scores, per-label breakdowns, sample identities, images,
request/log files, local directories and arbitrary dataset source metadata.
It still contains model/run/dataset names and any description, URL or hardware
note you explicitly supply. Review these before sharing. A fingerprint alone
does not reconstruct which images were selected: supply a public, authorized
selection reference when others need exact reproduction. Dataset and image
redistribution rights remain separate from the tool's MIT license.
