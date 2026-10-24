# VLM experiments

Vision Bench sends single images to an existing, non-streaming
chat-completions HTTP endpoint, such as [llama.cpp's image-chat server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
It never loads server weights, starts services, switches models or applies an
application's moderation rules. HTTP client timings do not measure server VRAM,
model loading or isolated kernel performance.

## Task and model configuration

Copy the bundled examples into your own files:

```console
python -c "from importlib.resources import files; from pathlib import Path; p=files('vision_bench').joinpath('vlm_examples'); [Path(n).write_text(p.joinpath(n).read_text('utf-8'), encoding='utf-8') for n in ('scene-task.json','local-profile.json')]"
```

`scene-task.json` is a neutral visible-scene classification task. Replace it
with your own task, giving it a new revision when prompts, output schema or
comparison fields change. The task contains:

- `schema_version`, `id`, `revision`;
- `system_prompt` and `user_prompt`, sent verbatim without source annotations;
- `output_schema`, an object schema using JSON Schema draft 2020-12;
- `comparison_fields`, required top-level output properties used for exact agreement.

Only embedded schema references are supported; validation never downloads remote
schemas. Validation uses [jsonschema](https://python-jsonschema.readthedocs.io/en/stable/validate/).
No format checker is enabled, so `format` is descriptive rather than an enforced
constraint. `unknown`, nulls and abstentions must be explicitly represented by
the task schema; the benchmark never converts them into another label.

Edit `local-profile.json` to set the exact server model ID and a meaningful
`model_revision` (ideally a pinned revision or weight hash). Revision is declared
by the caller; HTTP response model IDs are checked, but server weights are not
independently attested. The profile also pins:

- image `max_side` (64–2048) and `jpeg_quality` (1–100);
- generation `max_tokens`, `temperature`, optional `top_p`, `seed`, and optional
  `chat_template_kwargs` for servers that support them;
- `response_format`: `none`, `json_object`, or `json_schema`.

For `json_schema`, the request uses `{"type":"json_schema","json_schema":{"name":"vision_output","strict":true,"schema":...}}`.
The bundled profile selects this mode. `json_object` alone does not guarantee
that a particular server/model returns content accepted by the task schema.
Not every server supports every option. Unsupported options remain visible as
HTTP errors; Vision Bench never silently changes your requested configuration.
For reasoning-capable llama.cpp models, `"chat_template_kwargs":{"enable_thinking":false}`
can be included in `generation` when supported by the model template.

## Run

```console
vision-bench vlm-run --dataset ./dataset.json --task ./scene-task.json --profile ./local-profile.json --endpoint http://127.0.0.1:8080/v1/chat/completions --limit 8 --repeats 2 --output ./runs --run-id scene-model-a
```

The dataset format is the same as for taggers. For a VLM-only dataset,
`"tags":{"general":[]}` is sufficient. Tags are retained for provenance and
are never sent to the model. The default selects eight images with two repeats;
larger selections require an explicit limit. Requests run sequentially, repeating
the same payload and seed, so agreement measures repeat stability for that exact
configuration, not robustness over different seeds.

Images are hash-checked, limited to 64 MiB / 12 MP / 8192 pixels per edge, composited
on white, downscaled with Lanczos and encoded once as JPEG. Pixel orientation is
preserved without an implicit EXIF rotation. The exact prepared image is retained
with its hash. Single-frame images only. Input preparation is excluded from
HTTP prediction latency.

`--api-key-env VARIABLE_NAME` reads a bearer token from the process environment.
Credentials are not accepted in profiles or URLs and are not included in saved
request metadata. The endpoint must have no URL credentials, query or fragment.
The client bypasses ambient HTTP proxies and refuses redirects.

Retries default to zero. `--retries 1` permits one additional request, at most
three retries. Only network failures and HTTP 429/500/502/503/504 are retried, with
short exponential backoff. A network timeout may leave server work in progress;
retrying can generate additional work. Malformed JSON, schema failures, refusals,
truncations and other HTTP errors are never retried or silently repaired.
`--timeout` is a socket I/O timeout per HTTP attempt, not a wall-clock deadline
for the entire run. Responses are limited to 4 MiB per attempt.

## Artifacts and offline analysis

Each run contains `run.json`, the exact `task.json` and `profile.json`, normalized
`samples.jsonl`, `records.jsonl`, `input-*.jpg`, `response-*.bin` and `metrics.json`.
Every HTTP attempt is retained, including transient errors before eventual
success. Request hashes exclude credentials. Raw response bytes include the
server's output envelope, usage and any reasoning fields; parsing uses only the
assistant's final textual content. No complete request containing image base64
or authorization headers is written.

The parser requires one matching-model assistant choice ending in `stop`, then
strict JSON conforming to the task schema. Duplicate keys, non-finite numbers,
markdown fences and extra prose are errors. Explicit refusal and truncation are
separate outcomes. An unstructured refusal is a parsing failure; it is not guessed
from prose. A completed run means all planned predictions were attempted; it may
contain zero valid outputs. Interrupted runs remain failed diagnostic artifacts.

```console
vision-bench vlm-recompute ./runs/scene-model-a
vision-bench vlm-compare ./runs/scene-model-a ./runs/scene-model-b --output ./vlm-comparison.json
vision-bench vlm-review ./runs/scene-model-a ./runs/scene-model-b --output ./human-review.jsonl
```

Offline commands require no model runtime, images outside the run, or network.
They verify file hashes and reparse raw responses against the saved task before
computing results. Run IDs and output files are never overwritten.

Metrics report every final outcome, HTTP attempts/retries, first-attempt successes,
valid-output coverage and successful prediction latency including backoff.
Stability counts a sample as eligible only when all repeats are valid and there
are at least two repeats. Successful-pair agreement is accompanied by its coverage
of all planned pairs. Missing outputs have no synthetic predictions.

Cross-run comparison requires identical task fingerprints, dataset ordering and
repeat counts. It compares aligned sample/repeat positions on jointly valid
outputs and includes each run's errors and coverage. Generation and image recipes
may differ between profiles; this is a comparison of those complete recipes.
Agreement is not accuracy, and source tags do not establish ground truth for an
arbitrary VLM task.

The review export keeps predictions under `proposed_output`, flags disagreements
and errors, and leaves reviewer, acceptance and corrected output empty. Images and
raw response references are relative to the identified run directories. Exports
are review packets, not automatically accepted human labels.
