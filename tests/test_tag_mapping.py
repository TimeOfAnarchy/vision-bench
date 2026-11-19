from contextlib import contextmanager

import numpy as np
import pytest

from vision_bench import compare, hard_images
from vision_bench.cli import main
from vision_bench.errors import BenchmarkError
from vision_bench.io import file_hash, read_jsonl, read_object, write_json
from vision_bench.tag_mapping import load_tag_mapping


def test_compare_mapping_normalizes_truth_and_retains_native_decisions(tmp_path, monkeypatch):
    samples = [
        {
            "id": str(index),
            "image": f"{index}.png",
            "sha256": f"{index:064x}",
            "tags": {"general": tags, "character": ["kitty_ears"]},
        }
        for index, tags in enumerate(
            (
                ["kitty_ears", "cat_ears", "old_color", "shared"],
                ["fox_ears", "shared"],
                ["animal_ears", "shared"],
                ["shared"],
            )
        )
    ]
    vocabularies = [
        {
            "labels": ["kitty_ears", "cat_ears", "old_color", "shared", "hero"],
            "categories": ["general"] * 4 + ["character"],
        },
        {"labels": ["animal_ears", "color", "shared", "hero"], "categories": ["general"] * 3 + ["character"]},
    ]
    scores = [
        [
            [0.65, 0.9, 0.8, 0.9, 0.6],
            [0.55, 0.9, 0.1, 0.9, 0.6],
            [0.2, 0.9, 0.1, 0.9, 0.6],
            [0.2, 0.9, 0.1, 0.9, 0.6],
        ],
        [[0.8, 0.8, 0.9, 0.9], [0.85, 0.1, 0.9, 0.9], [0.2, 0.1, 0.9, 0.9], [0.1, 0.1, 0.9, 0.9]],
    ]
    runs = {}
    for index, name in enumerate(("left", "right")):
        path = tmp_path / name
        path.mkdir()
        config = {"id": name, "thresholds": {"general": 0.5, "character": 0.5}}
        if index == 0:
            config["tag_thresholds"] = {"general": {"kitty_ears": 0.6, "cat_ears": 0.95}}
        write_json(path / "profile.json", config)
        runs[path] = (
            {"run_id": name, "dataset": {"fingerprint": "same"}},
            samples,
            vocabularies[index],
            np.asarray(scores[index], dtype=np.float32),
            np.ones(4, dtype=np.bool_),
            [],
        )

    @contextmanager
    def load(path, cache_dir=None):
        yield runs[path]

    monkeypatch.setattr(compare, "load_results", load)
    monkeypatch.setattr(hard_images, "load_results", load)
    paths = list(runs)
    mapping = tmp_path / "mapping.json"
    write_json(
        mapping,
        {
            "schema_version": 1,
            "categories": {
                "general": {
                    "aliases": {"Kitty Ears": "cat_ears", "old_color": "color"},
                    "implications": {
                        "cat_ears": ["furry_ears"],
                        "furry_ears": ["animal_ears"],
                        "fox_ears": ["animal_ears"],
                    },
                }
            },
        },
    )

    strict = compare.compare_runs(paths, bootstrap_resamples=0)
    aliases = compare.compare_runs(paths, bootstrap_resamples=0, tag_mapping=load_tag_mapping(mapping))
    assert "tag_mapping" not in strict
    assert strict["categories"]["general"]["label_count"] == 1
    assert aliases["categories"]["general"]["label_count"] == 2
    assert aliases["categories"]["general"]["runs"][0]["metrics"]["coverage"]["positive_occurrences"] == 8

    output = tmp_path / "mapped.json"
    command = [
        "compare",
        *(str(path) for path in paths),
        "--output",
        str(output),
        "--bootstrap-resamples",
        "8",
    ]
    assert main([*command, "--include-implications"]) == 2
    assert main([*command, "--tag-mapping", str(mapping), "--include-implications"]) == 0
    result = read_object(output)
    assert result["tag_mapping"]["sha256"] == file_hash(mapping)
    assert result["tag_mapping"]["mode"] == "aliases_and_implications"
    assert (
        result["categories"]["character"]["vocabulary_sha256"]
        == strict["categories"]["character"]["vocabulary_sha256"]
    )
    category = result["categories"]["general"]
    assert category["label_count"] == 3
    assert category["strict_vocabulary"]["represented_occurrences"] == 4
    left = category["runs"][0]
    metrics = left["metrics"]
    assert metrics["tag_thresholds"] == {"cat_ears": 0.95, "kitty_ears": 0.6}
    assert metrics["coverage"]["represented_occurrences"] == 8
    parent = next(tag for tag in metrics["per_label"] if tag["label"] == "animal_ears")
    assert (parent["true_positive"], parent["unlabeled_positive"], parent["false_negative"]) == (1, 0, 2)
    assert parent["ap"] == 0.75
    assert left["threshold_analysis"]["true_positive"][900] == 7
    assert metrics["micro_representable"]["f1"] == pytest.approx(6 / 7)
    assert category["bootstrap"]["runs"][0]["metrics"]["micro_representable_f1"]["estimate"] == pytest.approx(
        6 / 7
    )
    assert left["per_image"]["metrics"]["sample_f1_representable"]["value"] == pytest.approx(5 / 6)

    packet = tmp_path / "hard.jsonl"
    assert (
        main(
            [
                "hard-images",
                *(str(path) for path in paths),
                "--category",
                "general",
                "--top",
                "1",
                "--tag-mapping",
                str(mapping),
                "--include-implications",
                "--output",
                str(packet),
            ]
        )
        == 0
    )
    disagree = read_jsonl(packet)[0]
    assert disagree["review"]["tag_mapping"] == result["tag_mapping"]
    assert disagree["review"]["vocabulary_sha256"] == category["vocabulary_sha256"]
    assert disagree["id"] == "1"
    assert disagree["annotated"] == ["fox_ears", "shared"]
    assert disagree["common_annotations"] == ["animal_ears", "shared"]
    assert disagree["runs"][0]["missed_annotations"] == ["animal_ears"]
    assert [entry["tag"] for entry in disagree["runs"][1]["predictions"]] == ["animal_ears", "shared"]


@pytest.mark.parametrize(
    "rules",
    [
        {"aliases": {"a": "b", "b": "a"}},
        {"implications": {"a": ["b"], "b": ["a"]}},
        {"aliases": {"a": "b"}, "implications": {"a": ["b"]}},
        {"implications": {"a": "b"}},
        {"aliases": {"A": "b", "a": "c"}},
    ],
)
def test_mapping_rejects_invalid_rules(tmp_path, rules):
    path = tmp_path / "mapping.json"
    write_json(path, {"schema_version": 1, "categories": {"general": rules}})
    with pytest.raises(BenchmarkError):
        load_tag_mapping(path)


def test_implications_are_transitive_and_directional(tmp_path):
    path = tmp_path / "mapping.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "categories": {
                "general": {
                    "aliases": {"old": "new", "new": "current"},
                    "implications": {"current": ["parent", "ancestor"], "parent": ["ancestor"]},
                }
            },
        },
    )
    rules = load_tag_mapping(path, include_implications=True).rules["general"]
    assert rules.expand({"old", "new", "current"}) == {"current", "parent", "ancestor"}
    assert rules.expand({"parent"}) == {"parent", "ancestor"}
