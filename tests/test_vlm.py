from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files

import pytest
from PIL import Image

from vision_bench.cli import main
from vision_bench.errors import BenchmarkError
from vision_bench.io import content_hash, file_hash, read_jsonl, read_object, write_json, write_jsonl
from vision_bench.vlm_contracts import parse_response, task_validator, validate_profile
from vision_bench.vlm_results import (
    compare_vlm_runs,
    compute_vlm_metrics,
    export_vlm_review,
    load_vlm_run,
    vlm_inventory,
)
from vision_bench.vlm_run import run_vlm
from vision_bench.vlm_transport import post_chat, post_chat_with_headers, validate_endpoint


@pytest.fixture
def vlm_inputs(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    samples = []
    for index in range(3):
        path = root / f"image-{index}.png"
        Image.new("RGBA", (30, 12), (255, index * 80, 0, 180)).save(path)
        samples.append(
            {"id": f"image:{index}", "image": path.name, "sha256": file_hash(path), "tags": {"general": []}}
        )
    write_jsonl(root / "samples.jsonl", samples)
    write_json(
        root / "dataset.json",
        {
            "schema_version": 1,
            "id": "tiny-vlm",
            "revision": "v1",
            "annotation_kind": "silver",
            "samples": "samples.jsonl",
            "samples_sha256": file_hash(root / "samples.jsonl"),
        },
    )
    task = {
        "schema_version": 1,
        "id": "observations",
        "revision": "v1",
        "system_prompt": "Return JSON only.",
        "user_prompt": "Classify visible evidence as a, b or unknown.",
        "comparison_fields": ["label"],
        "output_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["label"],
            "properties": {"label": {"enum": ["a", "b", "unknown"]}, "note": {"type": "string"}},
        },
    }
    profile = {
        "schema_version": 1,
        "id": "tiny-model",
        "model": "tiny-model",
        "model_revision": "fixture-v1",
        "response_format": "json_object",
        "image": {"max_side": 64, "jpeg_quality": 90},
        "generation": {"temperature": 0, "max_tokens": 64, "seed": 42},
    }
    write_json(root / "task.json", task)
    write_json(root / "profile.json", profile)
    return {
        "dataset_path": root / "dataset.json",
        "task_path": root / "task.json",
        "profile_path": root / "profile.json",
        "output_root": tmp_path / "runs",
    }


def response(label="a", *, content=None, finish="stop", refusal=None, model="tiny-model"):
    return {
        "model": model,
        "choices": [
            {
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "refusal": refusal,
                    "content": content if content is not None else json.dumps({"label": label}),
                },
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }


@contextmanager
def server(replies):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append({"payload": payload, "authorization": self.headers.get("Authorization")})
            reply = replies[min(len(received) - 1, len(replies) - 1)]
            status, body = reply[:2]
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for name, value in (reply[2] if len(reply) == 3 else {}).items():
                self.send_header(name, value)
            if status == 307:
                self.send_header("Location", "/redirected")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: httpd.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions", received
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(2)


@pytest.mark.parametrize(
    "body,outcome",
    [
        (response("unknown"), "ok"),
        (response(content='```json\n{"label":"a"}\n```'), "output_json_error"),
        (response(content='{"label":"a","label":"b"}'), "output_json_error"),
        (response(content='{"label":NaN}'), "output_json_error"),
        (response(content='{"label":1e999}'), "output_json_error"),
        (response(content='{"label":"unseen"}'), "schema_error"),
        (response(content="{}"), "schema_error"),
        (response(finish="length"), "truncated"),
        (response(refusal="Cannot answer."), "refusal"),
        (response(finish="content_filter"), "refusal"),
        (response(model="different-model"), "model_mismatch"),
        (response(finish="tool_calls"), "response_shape_error"),
        ({"model": "tiny-model", "choices": []}, "response_shape_error"),
        (b"not JSON", "response_json_error"),
    ],
)
def test_strict_output_parsing_preserves_unknown_and_distinguishes_failures(vlm_inputs, body, outcome):
    task = read_object(vlm_inputs["task_path"])
    result = parse_response(
        body if isinstance(body, bytes) else json.dumps(body).encode(), task, "tiny-model"
    )
    assert result["outcome"] == outcome
    if outcome == "ok":
        assert result["output"] == {"label": "unknown"}
    else:
        assert result["output"] is None
    if outcome == "model_mismatch":
        assert result["reported_model"] == "different-model"


def test_real_http_history_repeat_stability_and_offline_raw_reparse(vlm_inputs, monkeypatch, tmp_path):
    replies = [
        (200, response()),
        (503, {"error": "busy"}),
        (200, response()),
        (200, response("b")),
        (200, response()),
        (200, response(content="invalid")),
        (200, response(refusal="Cannot answer.")),
    ]
    monkeypatch.setenv("BENCH_TEST_AUTH", "fixture-secret-never-record")
    with server(replies) as (endpoint, received):
        path = run_vlm(**vlm_inputs, endpoint=endpoint, retries=1, repeats=2, api_key_env="BENCH_TEST_AUTH")
    assert len(received) == 7
    assert received[0]["authorization"] == "Bearer fixture-secret-never-record"
    payload = received[0]["payload"]
    assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "tags" not in json.dumps(payload) and payload["stream"] is False
    assert payload["response_format"] == {"type": "json_object"}
    before = {item.name: item.read_bytes() for item in path.iterdir()}
    assert all(b"fixture-secret-never-record" not in value for value in before.values())
    monkeypatch.setattr(
        "vision_bench.vlm_run.post_chat", lambda *args, **kwargs: pytest.fail("Offline network call")
    )
    metrics = compute_vlm_metrics(path)
    assert metrics == read_object(path / "metrics.json")
    assert metrics["outcomes"] == {"ok": 4, "output_json_error": 1, "refusal": 1}
    assert metrics["attempts"] == 7 and metrics["retried_predictions"] == 1
    assert metrics["first_attempt_successes"] == 3
    assert metrics["stability"] == {
        "comparison_fields": ["label"],
        "eligible_samples": 2,
        "stable_samples": 1,
        "stable_sample_fraction": 0.5,
        "planned_pairs": 3,
        "successful_pairs": 2,
        "equal_pairs": 1,
        "pair_coverage": 2 / 3,
        "agreement_on_successful_pairs": 0.5,
    }
    output = tmp_path / "review.jsonl"
    assert export_vlm_review([path], output) == 3
    review = read_jsonl(output)
    assert review[1]["disagreement"] and review[2]["has_errors"]
    assert all(
        item["human_review"]["corrected_output"] is None and item["human_review"]["accepted"] is None
        for item in review
    )
    with pytest.raises(FileExistsError):
        export_vlm_review([path], output)
    assert before == {item.name: item.read_bytes() for item in path.iterdir()}


def test_failures_do_not_disappear_and_invalid_outputs_are_not_retried(vlm_inputs):
    with server([(200, response(content="no JSON"))]) as (endpoint, received):
        path = run_vlm(**vlm_inputs, endpoint=endpoint, repeats=2, retries=3, limit=1)
    assert len(received) == 2
    metrics = read_object(path / "metrics.json")
    assert read_object(path / "run.json")["state"] == "completed"
    assert metrics["valid_output_fraction"] == 0
    assert metrics["latency"]["median_ms"] is None
    assert metrics["stability"]["successful_pairs"] == 0
    assert metrics["stability"]["pair_coverage"] == 0
    assert metrics["stability"]["stable_sample_fraction"] is None


def test_non_assistant_message_is_not_a_prediction(vlm_inputs):
    body = response()
    body["choices"][0]["message"]["role"] = "user"
    assert (
        parse_response(json.dumps(body).encode(), read_object(vlm_inputs["task_path"]), "tiny-model")[
            "outcome"
        ]
        == "response_shape_error"
    )


def test_input_hash_errors_never_send_images_and_transport_errors_are_bounded(vlm_inputs):
    (vlm_inputs["dataset_path"].parent / "image-0.png").write_bytes(b"changed")
    with server([(401, {"error": "unauthorized"})]) as (endpoint, received):
        path = run_vlm(**vlm_inputs, endpoint=endpoint, repeats=1, retries=3)
    assert len(received) == 2
    assert compute_vlm_metrics(path)["outcomes"] == {"http_error": 2, "image_error": 1}
    with server([(503, {"error": "busy"})]) as (endpoint, received):
        path = run_vlm(**vlm_inputs, endpoint=endpoint, repeats=1, retries=1, limit=2)
    assert len(received) == 2
    assert compute_vlm_metrics(path)["outcomes"] == {"http_error": 1, "image_error": 1}


def test_transport_blocks_redirects_and_bounds_response_size(monkeypatch):
    with server([(307, b"redirect")]) as (endpoint, received):
        status, body = post_chat(endpoint, {}, timeout=1)
    assert status == 307 and body == b"redirect" and len(received) == 1
    monkeypatch.setattr("vision_bench.vlm_transport.MAX_RESPONSE_BYTES", 32)
    from vision_bench.vlm_transport import TransportError

    with server([(200, b"x" * 100)]) as (endpoint, _):
        with pytest.raises(TransportError) as error:
            post_chat(endpoint, {}, timeout=1)
    assert error.value.kind == "response_too_large" and len(error.value.body) == 32


def test_transport_retains_only_bounded_diagnostic_headers_including_http_errors(monkeypatch):
    headers = {
        "X-Generation-Id": "fixture-generation",
        "Retry-After": "60",
        "Set-Cookie": "private",
        "X-Request-Id": "x" * 1100,
    }
    with server([(429, b"limited", headers)]) as (endpoint, received):
        status, body, saved = post_chat_with_headers(endpoint, {}, timeout=1)
    assert status == 429 and body == b"limited" and len(received) == 1
    assert saved["x-generation-id"] == "fixture-generation" and saved["retry-after"] == "60"
    assert len(saved["x-request-id"]) == 1024 and "set-cookie" not in saved
    monkeypatch.setattr("vision_bench.vlm_transport.MAX_RESPONSE_BYTES", 8)
    from vision_bench.vlm_transport import TransportError

    with server([(200, b"too-long-response", headers)]) as (endpoint, _):
        with pytest.raises(TransportError) as error:
            post_chat_with_headers(endpoint, {}, timeout=1)
    assert error.value.headers["x-generation-id"] == "fixture-generation"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://user:pass@example.test/chat",
        "http://example.test/chat?token=secret",
        "file:///tmp/chat",
        "http://example.test/chat#fragment",
    ],
)
def test_endpoint_credentials_and_ambiguous_targets_rejected(endpoint):
    with pytest.raises(BenchmarkError):
        validate_endpoint(endpoint)


def test_task_schema_is_validated_without_external_resources_and_profile_cannot_override_protocol(vlm_inputs):
    task = read_object(vlm_inputs["task_path"])
    task["output_schema"]["properties"]["label"] = {"$ref": "https://example.test/schema"}
    with pytest.raises(BenchmarkError, match="document-local"):
        task_validator(task)
    task = read_object(vlm_inputs["task_path"])
    task["comparison_fields"] = ["note"]
    with pytest.raises(BenchmarkError, match="required"):
        task_validator(task)
    profile = read_object(vlm_inputs["profile_path"])
    profile["generation"]["messages"] = []
    with pytest.raises(BenchmarkError, match="supported"):
        validate_profile(profile)
    examples = files("vision_bench").joinpath("vlm_examples")
    task_validator(json.loads(examples.joinpath("scene-task.json").read_text()))
    validate_profile(json.loads(examples.joinpath("local-profile.json").read_text()))


def test_embedded_schema_references_resolve_without_retrieval(vlm_inputs):
    task = read_object(vlm_inputs["task_path"])
    task["output_schema"]["$defs"] = {"label": {"enum": ["a", "b", "unknown"]}}
    task["output_schema"]["properties"]["label"] = {"$ref": "#/$defs/label"}
    validator = task_validator(task)
    assert validator.is_valid({"label": "unknown"})
    assert not validator.is_valid({"label": "different"})


def test_comparison_aligns_all_repeats_and_refuses_task_drift(vlm_inputs):
    with server([(200, response())]) as (endpoint, _):
        a = run_vlm(**vlm_inputs, endpoint=endpoint, run_id="a", limit=1)
    with server([(200, response()), (200, response("b"))]) as (endpoint, _):
        b = run_vlm(**vlm_inputs, endpoint=endpoint, run_id="b", limit=1)
    values = compare_vlm_runs([a, b])
    assert values["planned_predictions"] == values["paired_successful"] == 2
    assert values["equal_predictions"] == 1 and values["agreement_on_paired_successes"] == 0.5
    task = read_object(b / "task.json")
    task["revision"] = "v2"
    write_json(b / "task.json", task)
    manifest = read_object(b / "run.json")
    manifest["task_fingerprint"] = content_hash(task)
    manifest["artifacts"] = vlm_inventory(b)
    write_json(b / "run.json", manifest)
    with pytest.raises(BenchmarkError, match="identical task"):
        compare_vlm_runs([a, b])


def test_raw_integrity_and_forged_derived_output_are_rejected(vlm_inputs):
    with server([(200, response())]) as (endpoint, _):
        path = run_vlm(**vlm_inputs, endpoint=endpoint, repeats=1, limit=1)
    records = read_jsonl(path / "records.jsonl")
    records[0]["output"] = {"label": "b"}
    write_jsonl(path / "records.jsonl", records)
    with pytest.raises(BenchmarkError, match="integrity"):
        load_vlm_run(path)
    manifest = read_object(path / "run.json")
    manifest["artifacts"] = vlm_inventory(path)
    write_json(path / "run.json", manifest)
    with pytest.raises(BenchmarkError, match="raw response"):
        load_vlm_run(path)


@pytest.mark.parametrize("change", ["request", "retry_after_success", "grid"])
def test_saved_request_and_attempt_contracts_are_rechecked(vlm_inputs, change):
    with server([(200, response())]) as (endpoint, _):
        path = run_vlm(**vlm_inputs, endpoint=endpoint, repeats=1, limit=1)
    records = read_jsonl(path / "records.jsonl")
    manifest = read_object(path / "run.json")
    if change == "request":
        records[0]["request_sha256"] = "0" * 64
        expected = "request does not match"
    elif change == "grid":
        records[0]["repeat"] = 1
        expected = "order or timing"
    else:
        import copy

        manifest["retries"] = 1
        extra = copy.deepcopy(records[0]["attempts"][0])
        extra["attempt"] = 1
        extra["raw_response"] = "response-000000-00-01.bin"
        (path / extra["raw_response"]).write_bytes(
            (path / records[0]["attempts"][0]["raw_response"]).read_bytes()
        )
        records[0]["attempts"].append(extra)
        expected = "non-retryable"
    write_jsonl(path / "records.jsonl", records)
    manifest["artifacts"] = vlm_inventory(path)
    write_json(path / "run.json", manifest)
    with pytest.raises(BenchmarkError, match=expected):
        load_vlm_run(path)


def test_vlm_cli_and_immutable_run_ids(vlm_inputs, tmp_path, capsys):
    with server([(200, response())]) as (endpoint, _):
        args = [
            "vlm-run",
            "--dataset",
            str(vlm_inputs["dataset_path"]),
            "--task",
            str(vlm_inputs["task_path"]),
            "--profile",
            str(vlm_inputs["profile_path"]),
            "--endpoint",
            endpoint,
            "--output",
            str(vlm_inputs["output_root"]),
            "--run-id",
            "cli",
            "--limit",
            "1",
            "--repeats",
            "1",
        ]
        assert main(args) == 0
        assert main(args) == 2
    path = vlm_inputs["output_root"] / "cli"
    assert main(["vlm-recompute", str(path)]) == 0
    assert main(["vlm-review", str(path), "--output", str(tmp_path / "review.jsonl")]) == 0
    assert "matches" in capsys.readouterr().out


def test_closed_endpoint_network_errors_respect_retry_budget(vlm_inputs):
    with server([(200, response())]) as (endpoint, _):
        pass
    path = run_vlm(**vlm_inputs, endpoint=endpoint, repeats=1, retries=1, limit=1, timeout=0.2)
    metrics = compute_vlm_metrics(path)
    # Windows may reach the short socket deadline before reporting connection refusal.
    assert metrics["outcomes"] in ({"network_error": 1}, {"timeout": 1})
    assert metrics["attempts"] == 2


@pytest.mark.parametrize("wrapped", [False, True])
def test_socket_timeouts_are_distinct_and_have_bounded_retries(vlm_inputs, monkeypatch, wrapped):
    import urllib.error

    class Opener:
        def open(self, *args, **kwargs):
            if wrapped:
                raise urllib.error.URLError(TimeoutError("fixture timeout"))
            raise TimeoutError("fixture timeout")

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    path = run_vlm(**vlm_inputs, endpoint="http://127.0.0.1:1/chat", repeats=1, retries=1, limit=1)
    metrics = compute_vlm_metrics(path)
    assert metrics["outcomes"] == {"timeout": 1} and metrics["attempts"] == 2


@pytest.mark.parametrize("status", [200, 503])
@pytest.mark.parametrize("error,outcome", [(TimeoutError, "timeout"), (OSError, "network_error")])
def test_response_read_failures_preserve_received_status_and_recompute(
    vlm_inputs, monkeypatch, status, error, outcome
):
    class Response:
        code = status
        headers = {"x-request-id": "fixture-request"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            raise error("Response failed after receiving headers")

    class Opener:
        def open(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    path = run_vlm(**vlm_inputs, endpoint="http://127.0.0.1:1/chat", repeats=1, retries=1, limit=1)
    manifest, _, _, _, records = load_vlm_run(path)
    assert manifest["state"] == "completed"
    assert [attempt["http_status"] for attempt in records[0]["attempts"]] == [status, status]
    metrics = compute_vlm_metrics(path)
    assert metrics["outcomes"] == {outcome: 1}
    assert metrics["attempts"] == 2
    assert metrics == read_object(path / "metrics.json")


@pytest.mark.parametrize("status", [None, True, 0, 600, 200.0, "200"])
def test_saved_http_outcomes_require_a_valid_integer_status(vlm_inputs, status):
    with server([(200, response())]) as (endpoint, _):
        path = run_vlm(**vlm_inputs, endpoint=endpoint, repeats=1, limit=1)
    records = read_jsonl(path / "records.jsonl")
    records[0]["attempts"][0]["http_status"] = status
    write_jsonl(path / "records.jsonl", records)
    manifest = read_object(path / "run.json")
    manifest["artifacts"] = vlm_inventory(path)
    write_json(path / "run.json", manifest)
    with pytest.raises(BenchmarkError, match="HTTP status"):
        load_vlm_run(path)


def test_interrupt_preserves_a_failed_diagnostic_run(vlm_inputs, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("vision_bench.vlm_run.post_chat", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_vlm(**vlm_inputs, endpoint="http://127.0.0.1:1/chat", run_id="interrupted", limit=1)
    manifest = read_object(vlm_inputs["output_root"] / "interrupted/run.json")
    assert manifest["state"] == "failed" and manifest["error"] == {"type": "KeyboardInterrupt"}
