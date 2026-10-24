from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .errors import BenchmarkError


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="vision-bench", description="Independent local image tagger benchmarks"
    )
    root.add_argument("--version", action="version", version=__version__)
    modes = root.add_subparsers(dest="command", required=True)
    validate = modes.add_parser("validate", help="Verify dataset structure and image hashes")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument("--limit", type=int)
    run = modes.add_parser("run", help="Run one model in an isolated process")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--profile", required=True, help="Profile JSON or builtin:profile-id")
    run.add_argument("--model-root", type=Path, required=True)
    run.add_argument("--output", type=Path, default=Path("runs"))
    run.add_argument("--run-id")
    run.add_argument(
        "--provider", choices=["CPUExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider"]
    )
    run.add_argument("--limit", type=int)
    run.add_argument("--shard-size", type=int, default=16)
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--timeout", type=float, default=600, help="Total worker deadline in seconds")
    recalc = modes.add_parser(
        "recompute", help="Verify artifacts and compare recomputed metrics without inference"
    )
    recalc.add_argument("run", type=Path)
    compare = modes.add_parser("compare", help="Compare saved runs on paired samples and common vocabularies")
    compare.add_argument("runs", type=Path, nargs="+")
    compare.add_argument(
        "--output", type=Path, required=True, help="New JSON report; existing files are never replaced"
    )
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "validate":
            from .dataset import load_dataset

            data = load_dataset(args.dataset, limit=args.limit, verify_images=True)
            result = {"ok": True, "samples": len(data.samples), "fingerprint": data.fingerprint}
        elif args.command == "run":
            from .runner import run_benchmark

            path = run_benchmark(
                dataset_path=args.dataset,
                profile_path=args.profile,
                model_root=args.model_root,
                output_root=args.output,
                provider=args.provider,
                run_id=args.run_id,
                limit=args.limit,
                shard_size=args.shard_size,
                warmup=args.warmup,
                timeout=args.timeout,
            )
            result = {"run": str(path), "state": "completed", "report": str(path / "report.html")}
        elif args.command == "recompute":
            from .runner import recompute

            _, matches = recompute(args.run)
            result = {"matches": matches}
            print(json.dumps(result))
            return 0 if matches else 1
        else:
            from .compare import compare_runs

            result = compare_runs(args.runs)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
            result = {"comparison": str(args.output), "paired_successful": result["paired_successful"]}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (BenchmarkError, OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        print(f"vision-bench: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
