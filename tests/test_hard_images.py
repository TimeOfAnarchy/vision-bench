import json
from contextlib import contextmanager

import numpy as np
import pytest

from vision_bench import hard_images
from vision_bench.cli import main
from vision_bench.io import read_jsonl, write_json


def test_rankings_use_paired_rows_and_deduplicate_images(tmp_path, monkeypatch, capsys):
    samples = [
        {"id": sample_id, "image": f"{sample_id}.png", "sha256": f"{index:064x}", "tags": {"general": tags}}
        for index, (sample_id, tags) in enumerate(
            [
                ("disagree", ["a", "outside"]),
                ("miss", ["b"]),
                ("correct", ["a"]),
                ("undefined", []),
                ("failed", ["b"]),
            ]
        )
    ]
    vocab = {"labels": ["a", "b", "c"], "categories": ["general"] * 3}
    runs = {}
    for index, first_scores in enumerate(([0.8, 0.9, 0.1], [0.8, 0.1, 0.9], [0.0, 0.0, 0.0])):
        run_id = f"model-{index}"
        path = tmp_path / run_id
        path.mkdir()
        write_json(path / "profile.json", {"id": run_id, "thresholds": {"general": 0.5}})
        scores = np.array([first_scores, [0.9, 0, 0], [0.9, 0, 0], [0, 0, 0], first_scores], dtype=np.float32)
        valid = np.ones(len(samples), dtype=np.bool_)
        if index == 1:
            valid[-1] = False
            scores[-1, :] = np.nan
        runs[path] = (
            {"run_id": run_id, "dataset": {"fingerprint": "dataset"}},
            samples,
            vocab,
            scores,
            valid,
            [],
        )

    @contextmanager
    def load(path, cache_dir=None):
        yield runs[path]

    monkeypatch.setattr(hard_images, "load_results", load)
    command = ["hard-images", *(str(path) for path in runs), "--category", "general", "--top", "1"]
    output = tmp_path / "hard.jsonl"
    assert main([*command, "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["items"] == 2
    disagree, miss = read_jsonl(output)
    assert [disagree["id"], miss["id"]] == ["disagree", "miss"]
    assert disagree["review"]["paired_images"] == 4
    assert disagree["mean_pairwise_jaccard_distance"] == pytest.approx(8 / 9)
    assert disagree["ranked_by"] == [
        {"kind": "disagreement", "rank": 1},
        {"kind": "lowest_f1", "rank": 1, "run_index": 2},
    ]
    assert disagree["annotations_outside_common_vocabulary"] == ["outside"]
    assert disagree["runs"][0]["common_f1"] == pytest.approx(2 / 3)
    assert disagree["runs"][0]["full_truth_f1"] == 0.5
    assert disagree["runs"][0]["unlabeled_predictions"] == ["b"]
    assert disagree["runs"][2]["missed_annotations"] == ["a"]
    assert miss["mean_pairwise_jaccard_distance"] == 0
    assert miss["ranked_by"] == [
        {"kind": "lowest_f1", "rank": 1, "run_index": 0},
        {"kind": "lowest_f1", "rank": 1, "run_index": 1},
    ]
    assert miss["runs"][0]["common_f1"] == 0

    again = tmp_path / "again.jsonl"
    assert main([*command, "--output", str(again)]) == 0
    assert again.read_bytes() == output.read_bytes()
    assert main([*command, "--output", str(output)]) == 2
    assert again.read_bytes() == output.read_bytes()

    all_output = tmp_path / "all.jsonl"
    assert hard_images.export_hard_images(list(runs), category="general", output=all_output, top=10) == 3
    assert {item["id"] for item in read_jsonl(all_output)} == {"disagree", "miss", "correct"}
