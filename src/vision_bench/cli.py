from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .errors import BenchmarkError


def add_bootstrap_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="Paired bootstrap resamples for 95%% intervals; 0 disables them",
    )
    command.add_argument("--bootstrap-seed", type=int, default=0, help="Seed for bootstrap resampling")


def add_tag_mapping_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--tag-mapping", type=Path, help="Versioned JSON with category-scoped tag aliases")
    command.add_argument(
        "--include-implications", action="store_true", help="Apply implication rules from the mapping"
    )


def tag_mapping_from_args(args: argparse.Namespace):
    if args.tag_mapping is not None:
        from .tag_mapping import load_tag_mapping

        return load_tag_mapping(args.tag_mapping, include_implications=args.include_implications)
    if args.include_implications:
        raise BenchmarkError("--include-implications requires --tag-mapping.")
    return None


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="vision-bench", description="Independent local image tagger benchmarks"
    )
    root.add_argument("--version", action="version", version=__version__)
    root.add_argument("--cache-dir", type=Path, help="Directory for temporary disk-backed score matrices")
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
        "--provider",
        choices=["CPUExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider", "TorchCUDA"],
    )
    run.add_argument("--limit", type=int)
    run.add_argument("--shard-size", type=int, default=16)
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--timeout", type=float, default=600, help="Total worker deadline in seconds")
    run.add_argument(
        "--measure-memory",
        action="store_true",
        help="Sample the actual worker's WDDM process memory on Windows",
    )
    run.add_argument("--memory-interval-ms", type=int, default=250)
    recalc = modes.add_parser(
        "recompute", help="Verify artifacts and compare recomputed metrics without inference"
    )
    recalc.add_argument("run", type=Path)
    fit = modes.add_parser("fit-thresholds", help="Fit a new profile from a separate saved calibration run")
    fit.add_argument("run", type=Path)
    fit.add_argument("--output", type=Path, required=True, help="New model profile JSON")
    fit.add_argument(
        "--profile-id", help="ID for the fitted profile; defaults to the original ID with -fitted"
    )
    fit.add_argument(
        "--per-tag", action="store_true", help="Fit supported tags in addition to category fallbacks"
    )
    fit.add_argument("--min-support", type=int, default=20, help="Minimum annotated images per fitted tag")
    compare = modes.add_parser("compare", help="Compare saved runs on paired samples and common vocabularies")
    compare.add_argument("runs", type=Path, nargs="+")
    compare.add_argument(
        "--output", type=Path, required=True, help="New JSON report; existing files are never replaced"
    )
    add_bootstrap_arguments(compare)
    add_tag_mapping_arguments(compare)
    precision = modes.add_parser(
        "precision", help="Compare two precision/export variants at identical operating points"
    )
    precision.add_argument("baseline", type=Path)
    precision.add_argument("candidate", type=Path)
    precision.add_argument("--output", type=Path, required=True)
    publication = modes.add_parser(
        "export-report", help="Export a portable public tagger comparison without images or local paths"
    )
    publication.add_argument("runs", type=Path, nargs="+")
    publication.add_argument("--output", type=Path, required=True)
    publication.add_argument("--title", required=True)
    publication.add_argument("--dataset-url")
    publication.add_argument("--dataset-description", default="")
    publication.add_argument(
        "--hardware-note",
        default="",
        help="Explicit author annotation for historical runs; not measured hardware",
    )
    add_bootstrap_arguments(publication)
    vlm = modes.add_parser("vlm-run", help="Run a versioned image-chat task against an external server")
    vlm.add_argument("--dataset", type=Path, required=True)
    vlm.add_argument("--task", type=Path, required=True)
    vlm.add_argument("--profile", type=Path, required=True)
    vlm.add_argument("--endpoint", required=True, help="Full HTTP(S) chat-completions URL")
    vlm.add_argument("--output", type=Path, default=Path("runs"))
    vlm.add_argument("--run-id")
    vlm.add_argument("--limit", type=int, default=8)
    vlm.add_argument("--repeats", type=int, default=2)
    vlm.add_argument("--retries", type=int, default=0)
    vlm.add_argument(
        "--timeout", type=float, default=120, help="Socket I/O timeout in seconds per HTTP attempt"
    )
    vlm.add_argument("--api-key-env", help="Read a bearer token from this environment variable")
    vlm_recalc = modes.add_parser(
        "vlm-recompute", help="Verify VLM raw responses and recompute offline metrics"
    )
    vlm_recalc.add_argument("run", type=Path)
    vlm_compare = modes.add_parser("vlm-compare", help="Compare aligned VLM runs offline")
    vlm_compare.add_argument("runs", type=Path, nargs="+")
    vlm_compare.add_argument("--output", type=Path, required=True)
    vlm_review = modes.add_parser(
        "vlm-review", help="Export proposed outputs for human review, without creating gold labels"
    )
    vlm_review.add_argument("runs", type=Path, nargs="+")
    vlm_review.add_argument("--output", type=Path, required=True)
    regressions = modes.add_parser(
        "regressions", help="Find tags that significantly regressed or improved between two runs"
    )
    regressions.add_argument("baseline", type=Path)
    regressions.add_argument("candidate", type=Path)
    regressions.add_argument("--output", type=Path, required=True)
    regressions.add_argument(
        "--min-support", type=int, default=5, help="Annotated images needed to test a tag"
    )
    regressions.add_argument(
        "--fdr", type=float, default=0.05, help="Benjamini-Hochberg false discovery rate"
    )
    add_bootstrap_arguments(regressions)
    audit = modes.add_parser(
        "label-audit", help="Export a uniform sample of unlabeled tagger predictions for human review"
    )
    audit.add_argument("runs", type=Path, nargs="+")
    audit.add_argument("--category", required=True)
    audit.add_argument("--output", type=Path, required=True, help="New JSONL review packet")
    audit.add_argument("--per-run", type=int, default=100, help="Unlabeled predictions sampled per run")
    audit.add_argument("--seed", type=int, default=0)
    audit_score = modes.add_parser(
        "label-audit-score", help="Verify a reviewed label audit and estimate audited precision"
    )
    audit_score.add_argument("audit", type=Path)
    audit_score.add_argument("runs", type=Path, nargs="+", help="The exported runs, in export order")
    audit_score.add_argument("--output", type=Path, required=True)
    hard = modes.add_parser("hard-images", help="Export images with disagreeing predictions or low F1")
    hard.add_argument("runs", type=Path, nargs="+")
    hard.add_argument("--category", required=True)
    hard.add_argument("--output", type=Path, required=True, help="New JSONL review packet")
    hard.add_argument("--top", type=int, default=25, help="Images to select per ranking")
    add_tag_mapping_arguments(hard)
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
                measure_memory=args.measure_memory,
                memory_interval_ms=args.memory_interval_ms,
                cache_dir=args.cache_dir,
            )
            result = {"run": str(path), "state": "completed", "report": str(path / "report.html")}
        elif args.command == "vlm-run":
            from .vlm_run import run_vlm

            path = run_vlm(
                dataset_path=args.dataset,
                task_path=args.task,
                profile_path=args.profile,
                endpoint=args.endpoint,
                output_root=args.output,
                run_id=args.run_id,
                limit=args.limit,
                repeats=args.repeats,
                retries=args.retries,
                timeout=args.timeout,
                api_key_env=args.api_key_env,
            )
            result = {"run": str(path), "state": "completed"}
        elif args.command in {"recompute", "vlm-recompute"}:
            if args.command == "vlm-recompute":
                from .io import read_object
                from .vlm_results import compute_vlm_metrics

                matches = compute_vlm_metrics(args.run) == read_object(args.run / "metrics.json")
            else:
                from .runner import recompute

                _, matches = recompute(args.run, cache_dir=args.cache_dir)
            result = {"matches": matches}
            print(json.dumps(result))
            return 0 if matches else 1
        elif args.command == "export-report":
            from .public_report import export_report

            path = export_report(
                args.runs,
                output=args.output,
                title=args.title,
                dataset_url=args.dataset_url,
                dataset_description=args.dataset_description,
                hardware_note=args.hardware_note,
                cache_dir=args.cache_dir,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
            )
            result = {"report": str(path / "index.html"), "directory": str(path)}
        elif args.command == "fit-thresholds":
            from .threshold_fit import fit_thresholds

            profile = fit_thresholds(
                args.run,
                output=args.output,
                profile_id=args.profile_id,
                per_tag=args.per_tag,
                min_support=args.min_support,
                cache_dir=args.cache_dir,
            )
            result = {
                "profile": str(args.output),
                "fitted_categories": profile["threshold_fit"]["fitted_categories"],
            }
        elif args.command == "vlm-review":
            from .vlm_results import export_vlm_review

            result = {"review": str(args.output), "samples": export_vlm_review(args.runs, args.output)}
        elif args.command == "regressions":
            from .regressions import label_regressions

            report = label_regressions(
                args.baseline,
                args.candidate,
                resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed,
                min_support=args.min_support,
                fdr=args.fdr,
                cache_dir=args.cache_dir,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            result = {
                "report": str(args.output),
                "categories": {
                    category: {
                        "ap": values["summary"]["ap"],
                        "f1": values["summary"]["f1"],
                        "worst_ap": [
                            {"label": entry["label"], "difference": entry["ap_difference"]["estimate"]}
                            for entry in values["labels"]
                            if entry["ap_difference"]["verdict"] == "regression"
                        ][:5],
                    }
                    for category, values in report["categories"].items()
                },
            }
        elif args.command == "label-audit":
            from .label_audit import export_label_audit

            count = export_label_audit(
                args.runs,
                category=args.category,
                output=args.output,
                per_run=args.per_run,
                seed=args.seed,
                cache_dir=args.cache_dir,
            )
            result = {"review": str(args.output), "items": count}
        elif args.command == "label-audit-score":
            from .label_audit import score_label_audit

            scored = score_label_audit(args.audit, args.runs, cache_dir=args.cache_dir)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(scored, output, ensure_ascii=False, indent=2, allow_nan=False)
            result = {
                "result": str(args.output),
                "runs": {
                    run["run_id"]: run["audited_precision"] and run["audited_precision"]["estimate"]
                    for run in scored["runs"]
                },
            }
        elif args.command == "hard-images":
            from .hard_images import export_hard_images

            count = export_hard_images(
                args.runs,
                category=args.category,
                output=args.output,
                top=args.top,
                tag_mapping=tag_mapping_from_args(args),
                cache_dir=args.cache_dir,
            )
            result = {"review": str(args.output), "items": count}
        else:
            if args.command == "compare":
                from .compare import compare_runs

                result = compare_runs(
                    args.runs,
                    cache_dir=args.cache_dir,
                    bootstrap_resamples=args.bootstrap_resamples,
                    bootstrap_seed=args.bootstrap_seed,
                    tag_mapping=tag_mapping_from_args(args),
                )
            elif args.command == "precision":
                from .precision import compare_precision

                result = compare_precision(args.baseline, args.candidate, cache_dir=args.cache_dir)
            else:
                from .vlm_results import compare_vlm_runs

                result = compare_vlm_runs(args.runs)
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
