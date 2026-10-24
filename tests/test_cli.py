import json

from vision_bench.cli import main
from vision_bench.io import read_object, write_json


def test_cli_runs_validates_recomputes_and_compares_without_overwriting(local_inputs, capsys):
    values = local_inputs
    assert main(["validate", "--dataset", str(values["dataset"])]) == 0
    assert json.loads(capsys.readouterr().out)["samples"] == 3
    paths = []
    for run_id in ("a", "b"):
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
                    run_id,
                ]
            )
            == 0
        )
        paths.append(values["output"] / run_id)
        capsys.readouterr()
    assert main(["recompute", str(paths[0])]) == 0
    assert json.loads(capsys.readouterr().out)["matches"] is True
    output = values["output"] / "new-directory" / "comparison.json"
    command = ["compare", *(str(path) for path in paths), "--output", str(output)]
    assert main(command) == 0
    original = output.read_bytes()
    assert main(command) == 2
    assert output.read_bytes() == original


def test_comparison_explicitly_excludes_missing_threshold_category(local_inputs):
    from vision_bench.compare import compare_runs
    from vision_bench.runner import run_benchmark

    values = local_inputs
    first = run_benchmark(
        dataset_path=values["dataset"],
        profile_path=values["profile"],
        model_root=values["root"],
        output_root=values["output"],
        run_id="with-character",
    )
    profile = read_object(values["profile"])
    del profile["thresholds"]["character"]
    profile["id"] = "without-character"
    write_json(values["profile"], profile)
    second = run_benchmark(
        dataset_path=values["dataset"],
        profile_path=values["profile"],
        model_root=values["root"],
        output_root=values["output"],
        run_id="without-character",
    )
    result = compare_runs([first, second])
    assert "character" not in result["categories"]
    assert result["excluded_categories"]["character"] == {
        "reason": "missing_threshold",
        "profiles": ["without-character"],
    }
