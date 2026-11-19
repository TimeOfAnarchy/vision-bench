import json
from contextlib import contextmanager

import numpy as np
import pytest

from vision_bench import label_audit
from vision_bench.cli import main
from vision_bench.errors import BenchmarkError
from vision_bench.io import read_jsonl, read_object, write_json
from vision_bench.label_audit import export_label_audit, score_label_audit


@pytest.fixture
def fake_runs(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    labels = [f"tag-{index}" for index in range(10)]
    samples = [
        {
            "id": f"image:{index}",
            "image": f"{index}.png",
            "sha256": "0" * 64,
            "tags": {"general": [label for label in labels if rng.random() < 0.3]},
        }
        for index in range(40)
    ]
    vocabulary = {"labels": labels, "categories": ["general"] * len(labels)}
    runs, paths = {}, []
    for run_id in ("a", "b"):
        path = tmp_path / run_id
        path.mkdir()
        write_json(path / "profile.json", {"id": f"profile-{run_id}", "thresholds": {"general": 0.5}})
        manifest = {"run_id": run_id, "dataset": {"fingerprint": "fixture"}}
        scores = rng.random((40, 10)).astype(np.float32)
        runs[path.resolve()] = (manifest, samples, vocabulary, scores, np.ones(40, dtype=bool), [])
        paths.append(path)

    @contextmanager
    def load(path, cache_dir=None):
        yield runs[path.resolve()]

    monkeypatch.setattr(label_audit, "load_results", load)
    return paths, runs


def review(path, verdicts):
    items = read_jsonl(path)
    for item, verdict in zip(items, verdicts, strict=False):
        item["human_review"]["verdict"] = verdict
    path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
    return items


def test_export_samples_unlabeled_predictions_uniformly_and_deterministically(fake_runs, tmp_path):
    paths, runs = fake_runs
    output = tmp_path / "audit.jsonl"
    count = export_label_audit(paths, category="general", output=output, per_run=6, seed=3)
    items = read_jsonl(output)
    assert count == len(items) <= 12
    for path in paths:
        manifest, samples, vocabulary, scores, _, _ = runs[path.resolve()]
        mine = [item for item in items if manifest["run_id"] in item["sampled_by"]]
        assert len(mine) == 6
        for item in mine:
            row = int(item["id"].split(":")[1])
            assert item["label"] not in samples[row]["tags"]["general"]
            assert scores[row, vocabulary["labels"].index(item["label"])] >= 0.5
            assert item["scores"][manifest["run_id"]] == float(
                scores[row, vocabulary["labels"].index(item["label"])]
            )
    assert items[0]["human_review"] == {"reviewer": None, "verdict": None, "notes": ""}
    again = tmp_path / "again.jsonl"
    export_label_audit(paths, category="general", output=again, per_run=6, seed=3)
    assert again.read_bytes() == output.read_bytes()
    with pytest.raises(FileExistsError):
        export_label_audit(paths, category="general", output=output, per_run=6, seed=3)


def test_score_corrects_precision_with_reviewed_present_rate(fake_runs, tmp_path):
    paths, runs = fake_runs
    output = tmp_path / "audit.jsonl"
    export_label_audit(paths, category="general", output=output, per_run=8, seed=1)
    items = review(output, ["present", "absent", "unsure", "present"])  # The rest stay pending.
    result = score_label_audit(output, paths)
    for run in result["runs"]:
        mine = [item for item in items if run["run_id"] in item["sampled_by"]]
        verdicts = [item["human_review"]["verdict"] for item in mine]
        assert run["verdicts"]["present"] == verdicts.count("present")
        assert run["verdicts"]["pending"] == verdicts.count(None)
        assert run["sampled"] == 8
        predicted = run["true_positive"] + run["unlabeled_positive"]
        assert run["silver_precision"] == run["true_positive"] / predicted
        rate = run["present_rate"]
        if rate is None:
            assert run["audited_precision"] is None
            continue
        assert rate["low"] <= rate["estimate"] <= rate["high"]
        expected = (run["true_positive"] + run["unlabeled_positive"] * rate["estimate"]) / predicted
        assert run["audited_precision"]["estimate"] == pytest.approx(expected)
        assert run["audited_precision"]["low"] >= run["silver_precision"]


def test_score_refuses_edited_packets_and_unknown_verdicts(fake_runs, tmp_path):
    paths, _ = fake_runs
    output = tmp_path / "audit.jsonl"
    export_label_audit(paths, category="general", output=output, per_run=4, seed=0)
    with pytest.raises(BenchmarkError, match="same order"):
        score_label_audit(output, list(reversed(paths)))
    items = read_jsonl(output)
    items[0]["label"] = "something-else"
    edited = tmp_path / "edited.jsonl"
    edited.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
    with pytest.raises(BenchmarkError, match="does not match"):
        score_label_audit(edited, paths)
    review(output, ["maybe"])
    with pytest.raises(BenchmarkError, match="verdicts"):
        score_label_audit(output, paths)


def test_cli_audits_real_run_with_low_threshold(local_inputs, tmp_path, capsys):
    values = local_inputs
    profile = read_object(values["profile"])
    profile["thresholds"]["general"] = 0.05  # "extra" (0.1) becomes an unlabeled prediction.
    write_json(values["profile"], profile)
    run = values["output"] / "low"
    assert (
        main(
            [
                "run",
                "--dataset",
                str(values["dataset"]),
                "--profile",
                str(values["profile"]),
                "--model-root",
                str(values["root"]),
                "--output",
                str(values["output"]),
                "--run-id",
                "low",
            ]
        )
        == 0
    )
    capsys.readouterr()
    packet = tmp_path / "audit.jsonl"
    command = ["label-audit", str(run), "--category", "general", "--per-run", "2", "--output", str(packet)]
    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["items"] == 2
    review(packet, ["present", "present"])
    result = tmp_path / "audit-result.json"
    assert main(["label-audit-score", str(packet), str(run), "--output", str(result)]) == 0
    scored = read_object(result)["runs"][0]
    assert (scored["true_positive"], scored["unlabeled_positive"]) == (3, 3)
    assert scored["silver_precision"] == 0.5
    assert scored["audited_precision"]["estimate"] == 1.0
