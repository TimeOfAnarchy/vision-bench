import json
from contextlib import contextmanager

import numpy as np
import pytest

from vision_bench import regressions
from vision_bench.cli import main
from vision_bench.errors import BenchmarkError
from vision_bench.metrics import evaluate_columns
from vision_bench.regressions import benjamini_hochberg, label_regressions


def make_runs(tmp_path, monkeypatch, candidate_scores):
    rng = np.random.default_rng(0)
    labels = [f"tag-{index}" for index in range(6)]
    truth = rng.random((200, 6)) < [0.3, 0.3, 0.2, 0.2, 0.015, 0.0]
    samples = [
        {
            "id": f"image:{row}",
            "image": f"{row}.png",
            "sha256": "0" * 64,
            "tags": {"general": [label for label, keep in zip(labels, truth[row], strict=True) if keep]},
        }
        for row in range(200)
    ]
    baseline = np.clip(truth * 0.5 + rng.random((200, 6)) * 0.5, 0, 1).astype(np.float32)
    vocabulary = {"labels": labels, "categories": ["general"] * 6}
    runs, paths = {}, []
    for run_id, scores in (("base", baseline), ("cand", candidate_scores(baseline, truth, rng))):
        path = tmp_path / run_id
        path.mkdir()
        (path / "profile.json").write_text(
            json.dumps({"id": f"profile-{run_id}", "thresholds": {"general": 0.5}}), encoding="utf-8"
        )
        manifest = {"run_id": run_id, "dataset": {"fingerprint": "fixture"}}
        runs[path.resolve()] = (manifest, samples, vocabulary, scores, np.ones(200, dtype=bool), [])
        paths.append(path)

    @contextmanager
    def load(path, cache_dir=None):
        yield runs[path.resolve()]

    monkeypatch.setattr(regressions, "load_results", load)
    return paths, labels, baseline, truth, samples


def test_identical_runs_have_zero_differences_and_no_verdicts(tmp_path, monkeypatch):
    paths, labels, baseline, truth, samples = make_runs(tmp_path, monkeypatch, lambda base, *_: base.copy())
    report = label_regressions(*paths, resamples=200, block_size=4)
    general = report["categories"]["general"]
    assert general["summary"]["ap"] == {"regression": 0, "improvement": 0, "no_evidence": 4, "untested": 1}
    assert general["supported_labels"] == 5  # tag-5 has no annotated positives.
    expected = evaluate_columns(
        baseline, np.arange(200), list(range(6)), labels, [set(s["tags"]["general"]) for s in samples], 0.5
    )
    by_label = {entry["label"]: entry for entry in expected["per_label"]}
    for entry in general["labels"]:
        assert entry["baseline"]["ap"] == pytest.approx(by_label[entry["label"]]["ap"])
        assert entry["baseline"]["f1"] == pytest.approx(by_label[entry["label"]]["f1"])
        assert entry["ap_difference"]["estimate"] == 0
        if entry["support"] >= 5:
            assert entry["ap_difference"]["p_value"] == 1.0
        else:
            assert entry["ap_difference"]["verdict"] == "untested"
        assert "examples" not in entry


def test_broken_tag_is_flagged_with_examples(tmp_path, monkeypatch):
    def candidate(base, truth, rng):
        scores = base.copy()
        scores[:, 0] = rng.random(200).astype(np.float32)  # tag-0 loses its signal.
        return scores

    paths, *_ = make_runs(tmp_path, monkeypatch, candidate)
    general = label_regressions(*paths, resamples=300)["categories"]["general"]
    worst = general["labels"][0]
    assert worst["label"] == "tag-0"
    assert worst["ap_difference"]["verdict"] == "regression"
    assert worst["ap_difference"]["high"] < 0 and worst["ap_difference"]["q_value"] <= 0.05
    assert general["summary"]["ap"]["regression"] == 1
    drops = worst["examples"]["annotated_with_largest_drop"]
    assert len(drops) == 3 and all(item["candidate"] < item["baseline"] for item in drops)
    assert all(
        item["candidate"] > item["baseline"] for item in worst["examples"]["unlabeled_with_largest_rise"]
    )


def test_benjamini_hochberg_and_argument_checks(tmp_path, monkeypatch):
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.5]) == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.5])
    assert benjamini_hochberg([]) == []
    paths, *_ = make_runs(tmp_path, monkeypatch, lambda base, *_: base.copy())
    with pytest.raises(BenchmarkError, match="distinct"):
        label_regressions(paths[0], paths[0])
    with pytest.raises(BenchmarkError, match="False discovery"):
        label_regressions(*paths, fdr=1.5)


def test_cli_writes_regression_report(local_inputs, tmp_path, capsys):
    values = local_inputs
    paths = []
    for run_id in ("a", "b"):
        command = ["run", "--dataset", str(values["dataset"]), "--profile", str(values["profile"])]
        command += [
            "--model-root",
            str(values["root"]),
            "--output",
            str(values["output"]),
            "--run-id",
            run_id,
        ]
        assert main(command) == 0
        paths.append(values["output"] / run_id)
    capsys.readouterr()
    output = tmp_path / "regressions.json"
    command = ["regressions", *map(str, paths), "--output", str(output), "--min-support", "1"]
    assert main(command) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["categories"]["general"]["ap"]["regression"] == 0
    assert json.loads(output.read_text(encoding="utf-8"))["kind"] == "label_regressions"
    assert main(command) == 2  # Existing reports are never replaced.
