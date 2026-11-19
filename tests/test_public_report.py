from __future__ import annotations

import json
import subprocess

import pytest

from vision_bench.cli import main
from vision_bench.errors import BenchmarkError
from vision_bench.io import file_hash, read_object, write_json
from vision_bench.public_report import export_report, public_environment
from vision_bench.runner import run_benchmark


@pytest.fixture
def report_runs(local_inputs):
    values = local_inputs
    result = []
    for name in ("model-a", "model-b"):
        result.append(
            run_benchmark(
                dataset_path=values["dataset"],
                profile_path=values["profile"],
                model_root=values["root"],
                output_root=values["output"],
                run_id=name,
                warmup=1,
            )
        )
    return result


def test_public_export_retains_provenance_but_not_private_artifacts(report_runs, tmp_path, monkeypatch):
    first = read_object(report_runs[0] / "run.json")
    first["runtime"].pop("environment", None)  # Historical runs must not inherit the exporter's hardware.
    first["runtime"]["model"]["cuda_cudnn_directory"] = "C:/private-user/cudnn"
    first["runtime"]["model"].update(backend="torch-caformer", torch="2.11.0", precision="amp-fp16")
    first["dataset"]["source"] = {"private_source_path": "C:/private-user/annotations"}
    write_json(report_runs[0] / "run.json", first)
    before = [{p.name: file_hash(p) for p in run.iterdir() if p.is_file()} for run in report_runs]
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("Export must not launch inference or hardware probes"),
    )
    output = tmp_path / "publication"
    export_report(
        report_runs,
        output=output,
        title="Models <script>alert(1)</script>",
        dataset_url="https://example.org/dataset/v1",
        dataset_description="A frozen holdout",
        hardware_note="Historical hardware, supplied by author",
    )
    assert {p.name for p in output.iterdir()} == {"README.md", "index.html", "results.json", "SHA256SUMS"}
    data = read_object(output / "results.json")
    assert data["comparison"]["paired_successful"] == 3
    assert data["dataset"]["selected_count"] == 3
    assert data["dataset"]["annotation_kind"] == "silver"
    assert data["runs"][0]["runtime"]["environment"] == {"status": "not_recorded"}
    assert data["runs"][0]["runtime"]["model"]["onnxruntime"]
    assert data["runs"][0]["runtime"]["model"]["precision"] == "amp-fp16"
    assert "PyTorch 2.11.0" in (output / "index.html").read_text()
    assert data["runs"][0]["performance"]["warm"]["count"] == 2
    assert data["runs"][0]["artifact_hashes"]["model"]
    assert data["runs"][0]["metrics_version"] == 2
    assert data["runs"][0]["normalization"]
    assert data["runs"][0]["model_outputs"]["total"] == 3
    assert "author supplied" in data["author_hardware_note"]["provenance"]
    text = "\n".join(p.read_text() for p in output.iterdir())
    for private in (
        "C:/private-user",
        str(report_runs[0]),
        "image:0",
        "image-0.png",
        "launcher_pid",
        "cuda_cudnn_directory",
        "per_label",
    ):
        assert private not in text
    assert "<script>" not in (output / "index.html").read_text()
    assert "&lt;script&gt;" in (output / "index.html").read_text()
    readme = (output / "README.md").read_text()
    assert "sdk: static" in readme
    assert "each model's complete vocabulary" in readme
    assert "Paired differences: general (minus " in readme
    assert data["comparison"]["bootstrap"]["resamples"] == 1000
    assert "Threshold-free quality: general" in readme
    published = data["comparison"]["categories"]["general"]["runs"][0]["threshold_analysis"]
    assert "true_positive" not in published and "best_f1" in published
    assert "&lt;script&gt;" in readme
    assert "GPU sampling status" in (output / "index.html").read_text()
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert digest == file_hash(output / name)
    assert before == [{p.name: file_hash(p) for p in run.iterdir() if p.is_file()} for run in report_runs]
    with pytest.raises(BenchmarkError, match="new directory"):
        export_report(report_runs, output=output, title="Do not overwrite")


def test_public_export_rejects_corrupted_inputs_before_creating_output(report_runs, tmp_path):
    (report_runs[1] / "metrics.json").write_text("{}")
    with pytest.raises(BenchmarkError, match="integrity"):
        export_report(report_runs, output=tmp_path / "report", title="Invalid")
    assert not (tmp_path / "report").exists()


def test_public_report_keeps_paired_native_and_timing_denominators_distinct(local_inputs, tmp_path):
    values = local_inputs
    runs = []
    for name in ("complete", "with-failure"):
        if runs:
            (values["root"] / "image-1.png").write_bytes(b"changed source")
        runs.append(
            run_benchmark(
                dataset_path=values["dataset"],
                profile_path=values["profile"],
                model_root=values["root"],
                output_root=values["output"],
                run_id=name,
                warmup=1,
            )
        )
    export_report(runs, output=tmp_path / "report", title="Different successful samples")
    report = read_object(tmp_path / "report/results.json")
    assert report["comparison"]["paired_successful"] == 2
    assert [r["native_quality"]["general"]["sample_count"] for r in report["runs"]] == [3, 2]
    assert [r["performance"]["warm"]["count"] for r in report["runs"]] == [2, 1]
    assert [r["outcomes"]["failed"] for r in report["runs"]] == [0, 1]


@pytest.mark.parametrize(
    "url", ["javascript:alert(1)", "https://user:secret@example.org/data", "file:///private/data"]
)
def test_public_export_rejects_unsafe_dataset_links(tmp_path, url):
    with pytest.raises(BenchmarkError, match="HTTP"):
        export_report([], output=tmp_path / "report", title="Example", dataset_url=url)
    assert not (tmp_path / "report").exists()


def test_public_export_cli_and_measured_environment_whitelist(report_runs, tmp_path, capsys):
    assert (
        main(
            [
                "export-report",
                *map(str, report_runs),
                "--title",
                "Fixture comparison",
                "--output",
                str(tmp_path / "report"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["report"].endswith("index.html")
    result = public_environment(
        {
            "cpu": "Example CPU",
            "hostname": "private-host",
            "gpu_inventory": {
                "status": "observed",
                "devices": [{"name": "Example GPU", "driver": "1.2", "vram_mib": 1234, "uuid": "private-id"}],
            },
        }
    )
    assert result["cpu"] == "Example CPU"
    assert result["gpu_inventory"]["devices"][0] == {"name": "Example GPU", "driver": "1.2", "vram_mib": 1234}
    assert "private" not in json.dumps(result)
