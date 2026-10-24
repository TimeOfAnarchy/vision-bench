"""Verify built packages and run the shipped CPU tests/example outside the checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tarfile
import venv
import zipfile
from pathlib import Path


def check_distribution(dist: Path, work: Path) -> None:
    (wheel,) = dist.resolve().glob("*.whl")
    (source,) = dist.resolve().glob("*.tar.gz")
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert any(name.endswith("/licenses/LICENSE") for name in names), "Missing wheel license"
        for resource in ("profiles/caformer-b36.json", "vlm_examples/scene-task.json"):
            assert f"vision_bench/{resource}" in names, f"Missing package resource: {resource}"
    with tarfile.open(source) as archive:
        names += archive.getnames()
        archive.extractall(work / "source", filter="data")
    for name in names:
        parts = Path(name).parts
        assert not any(
            part in {".local", ".venv", "AGENTS.md", "roadmap.md", "runs", "models", "datasets"}
            for part in parts
        ), name
        assert not any(part.startswith("validation-") for part in parts), name
    (extracted,) = (work / "source").iterdir()
    venv.EnvBuilder(with_pip=True).create(work / "env")
    python = work / "env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"

    def run(*args: str, cwd: Path = work) -> None:
        subprocess.run([str(python), *map(str, args)], cwd=cwd, env=env, check=True)

    run("-m", "pip", "install", f"{wheel}[cpu,dev]")
    run("-m", "pip", "check")
    run(
        "-I",
        "-c",
        "import importlib.util, sys; from pathlib import Path; import vision_bench; "
        "assert Path(vision_bench.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); "
        "assert importlib.util.find_spec('django') is None",
    )
    run("-I", "-m", "pytest", "-q", str(extracted / "tests"))
    run(str(extracted / "examples/create_demo.py"), "--output", "inputs")
    run("-I", "-m", "vision_bench", "validate", "--dataset", "inputs/dataset.json")
    run(
        "-I",
        "-m",
        "vision_bench",
        "run",
        "--dataset",
        "inputs/dataset.json",
        "--profile",
        "inputs/profile.json",
        "--model-root",
        "inputs/model",
        "--output",
        "runs",
        "--run-id",
        "colors",
        "--warmup",
        "1",
    )
    run("-I", "-m", "vision_bench", "recompute", "runs/colors")
    metrics = json.loads((work / "runs/colors/metrics.json").read_text("utf-8"))
    assert metrics["outcomes"] == {"attempted": 4, "successful": 4, "failed": 0}
    assert (work / "runs/colors/report.html").is_file()
    print("Installed-wheel tests and generated CPU example passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--work", type=Path, required=True, help="New temporary verification directory")
    args = parser.parse_args()
    check_distribution(args.dist, args.work)
