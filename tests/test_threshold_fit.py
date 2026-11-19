import json

import numpy as np
import pytest
from PIL import Image

from vision_bench.cli import main
from vision_bench.compare import compare_runs
from vision_bench.io import file_hash, read_object, write_json, write_jsonl
from vision_bench.label_audit import export_label_audit
from vision_bench.public_report import export_report
from vision_bench.regressions import label_regressions
from vision_bench.runner import recompute
from vision_bench.threshold_fit import _fit_category


def test_fit_category_uses_successful_rows_and_inclusive_grid():
    scores = np.array([[0.4, 0.6], [0.3, 0.1], [0.7, 0.9], [np.nan, np.nan]], dtype=np.float32)
    threshold, overrides, positives = _fit_category(
        scores, np.array([0, 1, 2]), [0, 1], ["a", "b"], [{"a"}, set(), {"a", "b"}], 2
    )
    assert (threshold, overrides, positives) == (0.301, {"a": 0.301}, 3)


def test_cli_fits_profile_and_evaluates_disjoint_cpu_holdout(local_inputs, tmp_path, capsys):
    values = local_inputs
    root = values["root"]

    def dataset(name, samples):
        samples_path = root / f"{name}.jsonl"
        write_jsonl(samples_path, samples)
        manifest = read_object(values["dataset"])
        manifest.update(id=name, samples=samples_path.name, samples_sha256=file_hash(samples_path))
        path = root / f"{name}.json"
        write_json(path, manifest)
        return path

    def run(dataset_path, profile, run_id):
        return main(
            [
                "run",
                "--dataset",
                str(dataset_path),
                "--profile",
                str(profile),
                "--model-root",
                str(root),
                "--output",
                str(values["output"]),
                "--run-id",
                run_id,
                "--provider",
                "CPUExecutionProvider",
            ]
        )

    samples = values["samples"]
    samples[0]["tags"]["general"].append("extra")
    calibration = dataset("calibration", samples)
    assert run(calibration, values["profile"], "calibration") == 0
    cal_run = values["output"] / "calibration"
    assert read_object(cal_run / "run.json")["runtime"]["model"]["primary_provider"] == "CPUExecutionProvider"
    source_profile = (cal_run / "profile.json").read_bytes()
    fitted = tmp_path / "fitted.json"
    command = ["fit-thresholds", str(cal_run), "--output", str(fitted), "--per-tag", "--min-support", "1"]
    assert main(command) == 0
    assert main(command) == 2  # Exclusive output creation.
    capsys.readouterr()
    profile = read_object(fitted)
    assert (cal_run / "profile.json").read_bytes() == source_profile
    assert profile["id"] == "tiny-fitted"
    assert profile["thresholds"]["general"] == 0.101
    assert profile["tag_thresholds"]["general"] == {"extra": 0.0, "shared": 0.0}
    provenance = profile["threshold_fit"]
    assert provenance["source_run_id"] == "calibration"
    assert provenance["calibration_image_sha256"] == sorted(sample["sha256"] for sample in samples)
    assert provenance["dataset_fingerprint"] == read_object(cal_run / "run.json")["dataset"]["fingerprint"]

    higher_support = tmp_path / "min-two.json"
    assert (
        main(
            [
                "fit-thresholds",
                str(cal_run),
                "--output",
                str(higher_support),
                "--per-tag",
                "--min-support",
                "2",
            ]
        )
        == 0
    )
    assert read_object(higher_support)["tag_thresholds"]["general"] == {"shared": 0.0}
    category_only = tmp_path / "category.json"
    assert main(["fit-thresholds", str(cal_run), "--output", str(category_only)]) == 0
    assert read_object(category_only)["thresholds"] == profile["thresholds"]
    assert "tag_thresholds" not in read_object(category_only)

    holdout_samples = []
    for index, sample in enumerate(samples):
        image = root / f"holdout-{index}.png"
        Image.new("RGB", (13, 7), (50, 120, index)).save(image)
        holdout_samples.append(dict(sample, id=f"holdout:{index}", image=image.name, sha256=file_hash(image)))
    duplicate = dict(samples[0], id="new-id", image="renamed.png")
    mixed = dataset("mixed", [duplicate, holdout_samples[1]])
    assert run(mixed, fitted, "mixed") == 2
    assert "overlap by 1 image" in capsys.readouterr().err
    assert not (values["output"] / "mixed").exists()

    holdout = dataset("holdout", holdout_samples)
    assert run(holdout, fitted, "holdout-fitted") == 0
    assert run(holdout, values["profile"], "holdout-base") == 0
    fit_run, base_run = (values["output"] / name for name in ("holdout-fitted", "holdout-base"))
    assert read_object(fit_run / "run.json")["runtime"]["model"]["primary_provider"] == "CPUExecutionProvider"
    native, matches = recompute(fit_run)
    assert matches
    assert native["categories"]["general"]["micro_representable"]["f1"] == 0.8
    report = compare_runs([fit_run, base_run], bootstrap_resamples=8)["categories"]["general"]
    assert report["runs"][0]["metrics"]["tag_thresholds"] == profile["tag_thresholds"]["general"]
    assert report["runs"][0]["metrics"]["micro_representable"]["f1"] == 0.8
    assert report["runs"][1]["metrics"]["micro_representable"]["f1"] == pytest.approx(6 / 7)
    assert report["bootstrap"]["runs"][0]["metrics"]["micro_representable_f1"]["estimate"] == 0.8
    assert report["runs"][0]["per_image"]["metrics"]["sample_f1_representable"]["value"] == pytest.approx(
        7 / 9
    )

    regressions = label_regressions(base_run, fit_run, resamples=8, min_support=1)
    entries = {entry["label"]: entry for entry in regressions["categories"]["general"]["labels"]}
    assert (entries["extra"]["baseline"]["f1"], entries["extra"]["candidate"]["f1"]) == (0.0, 0.5)
    packet = tmp_path / "audit.jsonl"
    assert export_label_audit([fit_run, base_run], category="general", output=packet, per_run=3) == 2
    assert {json.loads(line)["label"] for line in packet.read_text(encoding="utf-8").splitlines()} == {
        "extra"
    }

    report_dir = tmp_path / "aggregate"
    export_report([fit_run, base_run], output=report_dir, title="Synthetic thresholds", bootstrap_resamples=0)
    public_run = read_object(report_dir / "results.json")["runs"][0]
    assert public_run["tag_thresholds"] == profile["tag_thresholds"]
    assert public_run["threshold_fit"]["dataset_fingerprint"] == provenance["dataset_fingerprint"]
    assert "calibration_image_sha256" not in public_run["threshold_fit"]
