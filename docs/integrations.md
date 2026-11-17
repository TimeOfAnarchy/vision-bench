# Library integrations and conversion tools

[Back to the project overview](../README.md).

## CPU session and image limits

Python integrations can limit CPU work with
`create_session(path, provider, cpu_threads=4, allow_spinning=False)`, passed as
the `OnnxTagger` session factory. The thread count limits intra-operator CPU
parallelism; disabling spinning prevents active waiting in both ONNX thread
pools. Omitted options retain the runtime defaults. These settings do not limit
GPU utilization or change the selected primary provider.
`OnnxTagger(..., max_image_edge=16384)` optionally accepts tall/narrow images up
to that edge length. The default remains 8192 pixels and the 12 MP area limit
always applies. This changes decode admission, not model preprocessing or pixels.

## ONNX precision conversion

`scripts/convert_onnx_fp16.py` creates a separate derived ONNX pack, retaining
float32 I/O, the source/derived hashes, converter versions and explicitly retained
float32 nodes. It refuses an existing output directory and never changes the
source pack. Conversion dependencies belong to a separate tools directory selected
by `--tools-dir`; choose `--converter common` or `--converter onnxruntime`.
Use `--help` for head selection and float32 operation/node exclusions. Validate
the converted graph with real inference and precision comparison before using it;
file size alone is not a GPU memory or numerical-accuracy measurement.

## VLM transport diagnostics

Library integrations can use `vision_bench.vlm_transport.post_chat_with_headers`
to retain the HTTP status, response bytes and a bounded allowlist of diagnostic
headers (`x-generation-id`, `retry-after`, `x-request-id`, `content-type`). Cookies
and authorization headers are excluded. `post_chat` returns `(status, body)`;
`post_chat_with_headers` returns `(status, body, headers)`. Neither function
performs implicit POST retries. `TransportError` retains the received HTTP status
and diagnostic headers when reading an already-started response fails.
