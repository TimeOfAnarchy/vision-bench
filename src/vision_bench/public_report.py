"""Portable aggregate experiment reports; raw run artifacts remain private."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .compare import compare_runs
from .errors import BenchmarkError
from .io import file_hash, read_object, write_json

NOTES = [
    "Paired quality compares the same successful samples and common category vocabulary. Full-truth recall/F1 also count source positives outside that vocabulary.",
    "Bootstrap intervals resample the paired images with replacement (same draw for every model) and cover sampling variation of this selection only, not label noise, other datasets or threshold choice. A difference interval that includes zero is not evidence of a real difference.",
    "Threshold-free rows use the same common outputs: best F1 picks its threshold on this evaluation set (an optimistic diagnostic, not a deployable operating point); precision at fixed recall compares models at matched operating points. Calibration error uses incomplete silver labels as the positive rate.",
    "Per-image rows average each image once (micro metrics weight images by tag count). Recall @ top 10 counts annotated common-output positives among each image's ten highest scores; no-hit images have annotated common-output positives but none predicted at the profile threshold (images whose positives are all outside the common vocabulary are counted separately in the JSON).",
    "Native quality uses each model's complete vocabulary and its own successful samples; its denominators differ from the paired comparison.",
    "Silver source annotations are incomplete. Unlabeled predictions are not verified false positives; native output categories retain their original semantics.",
    "Latency covers decoding, preprocessing and inference, excluding integrity checks and loading. Timing uses each run's own successful warm rows, not the paired quality subset.",
    "Timing differences are not model speedups unless hardware, provider, runtime and measurement conditions match. Power settings and competing workloads are not automatically recorded.",
    "GPU memory is a sampled WDDM process high water including loading, not an instantaneous allocator peak. GPU inventory does not establish which device a session used.",
    "Hardware is captured in new runs only. Missing historical fields remain unknown. CUDA/cuDNN loaded-library versions are not attested by an ONNX Runtime version or driver version.",
    "This aggregate report contains no sample identities, images or raw scores. A selection fingerprint verifies identity but does not reconstruct the dataset; publish an authorized selection reference separately.",
]


def pick(value: dict, keys: tuple[str, ...]) -> dict:
    return {key: value[key] for key in keys if key in value}


def aggregate_quality(value: dict) -> dict:
    result = pick(
        value,
        (
            "sample_count",
            "label_count",
            "threshold",
            "macro_f1_supported",
            "mean_average_precision_supported",
        ),
    )
    for key in ("micro_full_truth", "micro_representable"):
        result[key] = pick(
            value[key], ("true_positive", "unlabeled_positive", "false_negative", "precision", "recall", "f1")
        )
    result["coverage"] = pick(
        value["coverage"],
        (
            "truth_labels",
            "represented_truth_labels",
            "positive_occurrences",
            "represented_occurrences",
            "oov_occurrences",
            "occurrence_recall_ceiling",
        ),
    )
    return result


def public_environment(value: dict | None) -> dict:
    if not isinstance(value, dict):
        return {"status": "not_recorded"}
    result = pick(
        value, ("collector", "os", "os_release", "architecture", "cpu", "logical_cpus", "ram_bytes")
    )
    gpu = value.get("gpu_inventory", {})
    result["gpu_inventory"] = {
        **pick(gpu, ("status", "scope")),
        "devices": [pick(device, ("name", "vram_mib", "driver")) for device in gpu.get("devices", [])],
    }
    return result


def export_report(
    paths: list[Path],
    *,
    output: Path,
    title: str,
    dataset_url: str | None = None,
    dataset_description: str = "",
    hardware_note: str = "",
    cache_dir: Path | None = None,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 0,
) -> Path:
    if not title.strip() or len(title) > 160:
        raise BenchmarkError("Report title must contain 1-160 characters.")
    if dataset_url:
        url = urlsplit(dataset_url)
        if url.scheme not in {"https", "http"} or not url.hostname or url.username or url.password:
            raise BenchmarkError("Dataset URL must be an HTTP(S) URL without credentials.")
    output = output.resolve()
    if output.exists():
        raise BenchmarkError("Report output must be a new directory.")
    comparison = compare_runs(
        paths, cache_dir=cache_dir, bootstrap_resamples=bootstrap_resamples, bootstrap_seed=bootstrap_seed
    )
    for category in comparison["categories"].values():
        for row in category["runs"]:
            row["metrics"] = aggregate_quality(row["metrics"])
            for curve in ("true_positive", "unlabeled_positive"):
                row["threshold_analysis"].pop(curve)
    runs = []
    for path in paths:
        manifest, profile, metrics = [
            read_object(path / name) for name in ("run.json", "profile.json", "metrics.json")
        ]
        if "metrics.json" not in {item["path"] for item in manifest["artifacts"]}:
            raise BenchmarkError("Native metrics must be covered by the verified artifact inventory.")
        if (
            metrics["profile_id"] != profile["id"]
            or metrics["dataset_fingerprint"] != comparison["dataset_fingerprint"]
        ):
            raise BenchmarkError("Native metrics disagree with the compared inputs.")
        runtime = manifest["runtime"]
        model = runtime.get("model", {})
        source = profile.get("source", {})
        repository = source.get("repository")
        if repository is not None and not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
            repository = None
        performance = metrics["performance"]
        memory = pick(
            performance["gpu_memory"],
            (
                "status",
                "method",
                "scope",
                "requested_interval_ms",
                "valid_samples",
                "valid_span_seconds",
                "dedicated_observed_high_water_bytes",
                "shared_observed_high_water_bytes",
            ),
        )
        runs.append(
            {
                "run_id": manifest["run_id"],
                "profile_id": profile["id"],
                "run_manifest_sha256": file_hash(path / "run.json"),
                "started_at": manifest["started_at"],
                "completed_at": manifest.get("completed_at"),
                "model_source": {
                    "repository": repository,
                    **{
                        key: source[key]
                        if isinstance(source.get(key), str) and re.fullmatch(r"[\w.+-]{1,160}", source[key])
                        else None
                        for key in ("revision", "precision")
                    },
                },
                "artifact_hashes": {role: item["sha256"] for role, item in profile["artifacts"].items()},
                "profile_fingerprint": manifest["profile_fingerprint"],
                "metrics_version": metrics.get("metrics_version"),
                "normalization": manifest.get("normalization"),
                "model_outputs": pick(metrics.get("model_outputs", {}), ("total", "named", "unnamed")),
                "thresholds": profile["thresholds"],
                "preprocess": profile["preprocess"],
                "runtime": {
                    **pick(runtime, ("platform", "preprocess_version", "preprocess_code_sha256")),
                    "python": runtime.get("python", "not recorded").split()[0],
                    "libraries": pick(runtime.get("libraries", {}), ("vision-bench", "numpy", "Pillow")),
                    "model": pick(
                        model,
                        (
                            "onnxruntime",
                            "backend",
                            "torch",
                            "timm",
                            "cuda",
                            "cudnn",
                            "precision",
                            "weights_dtype",
                            "strict_state_dict",
                            "tf32",
                            "primary_provider",
                            "providers",
                            "input_shape",
                            "output_name",
                            "output_kind",
                            "cpu_graph_partitions_allowed",
                        ),
                    ),
                    "environment": public_environment(runtime.get("environment")),
                },
                "outcomes": metrics["outcomes"],
                "performance": {
                    "model_load_ms": metrics["model_load_ms"],
                    "warm": pick(performance["warm"], ("count", "median_ms", "p95_ms", "images_per_second")),
                    "warmup_excluded_rows": performance["warmup_excluded_rows"],
                    "gpu_memory": memory,
                },
                "native_quality": {
                    category: aggregate_quality(values) for category, values in metrics["categories"].items()
                },
            }
        )
    original_dataset = read_object(paths[0] / "run.json")["dataset"]
    payload = {
        "schema_version": 1,
        "kind": "public_tagger_comparison",
        "exporter_version": __version__,
        "title": title,
        "dataset": {
            **pick(
                original_dataset,
                (
                    "id",
                    "revision",
                    "annotation_kind",
                    "fingerprint",
                    "manifest_sha256",
                    "samples_sha256",
                    "selected_count",
                ),
            ),
            "public_reference_url": dataset_url,
            "author_description": dataset_description,
        },
        "author_hardware_note": {
            "text": hardware_note,
            "provenance": "author supplied; not machine-attested",
        },
        "runs": runs,
        "comparison": comparison,
        "limitations": NOTES,
    }
    page, readme = render(payload)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "results.json", payload)
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "README.md").write_text(readme, encoding="utf-8")
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{file_hash(output / name)}  {name}\n" for name in ("index.html", "README.md", "results.json")
        ),
        encoding="utf-8",
    )
    return output


def render(data: dict) -> tuple[str, str]:
    def text(value):
        return html.escape(str(value))

    def md(value):
        return re.sub(r"([\\`*_{}\[\]()#+.!|])", r"\\\1", html.escape(str(value), quote=False)).replace(
            "\n", " "
        )

    def number(value):
        return "n/a" if value is None else f"{value:.4f}"

    def point(value, name):
        return "n/a" if value is None else f"{value[name]:.4f} ({value['threshold']:.3f})"

    def interval(value):
        if value["estimate"] is None or value["low"] is None:
            return "n/a"
        return f"{value['estimate']:+.4f} [{value['low']:+.4f}, {value['high']:+.4f}]"

    def table(headers, rows):
        head = "<tr>" + "".join(f"<th scope='col'>{text(v)}</th>" for v in headers) + "</tr>"
        body = "".join("<tr>" + "".join(f"<td>{text(v)}</td>" for v in row) + "</tr>" for row in rows)
        markdown = "| " + " | ".join(map(md, headers)) + " |\n|" + "---|" * len(headers) + "\n"
        markdown += "".join("| " + " | ".join(map(md, row)) + " |\n" for row in rows)
        return (
            f"<div class='scroll'><table><thead>{head}</thead><tbody>{body}</tbody></table></div>",
            markdown,
        )

    sections, markdown = [], []

    def section(title, headers, rows):
        h, m = table(headers, rows)
        sections.append(f"<h2>{text(title)}</h2>{h}")
        markdown.append(f"## {title}\n\n{m}\n")

    section(
        "Run outcomes and performance",
        [
            "Model",
            "Successful / attempted",
            "Warm rows",
            "Warm median ms",
            "Warm p95 ms",
            "Load ms",
            "Dedicated GPU MiB",
            "GPU sampling status",
        ],
        [
            [
                r["profile_id"],
                f"{r['outcomes']['successful']} / {r['outcomes']['attempted']}",
                r["performance"]["warm"]["count"],
                number(r["performance"]["warm"]["median_ms"]),
                number(r["performance"]["warm"]["p95_ms"]),
                number(r["performance"]["model_load_ms"]),
                number(r["performance"]["gpu_memory"].get("dedicated_observed_high_water_bytes") / 1024**2)
                if r["performance"]["gpu_memory"].get("dedicated_observed_high_water_bytes") is not None
                else r["performance"]["gpu_memory"].get("status", "not_recorded"),
                r["performance"]["gpu_memory"].get("status", "not_recorded"),
            ]
            for r in data["runs"]
        ],
    )
    for category, result in data["comparison"]["categories"].items():
        section(
            f"Paired quality: {category} ({data['comparison']['paired_successful']} samples, {result['label_count']} common outputs)",
            [
                "Model",
                "Full-truth F1",
                "Full-truth recall",
                "Common-output F1",
                "Common-output precision",
                "mAP",
            ],
            [
                [
                    r["profile_id"],
                    number(r["metrics"]["micro_full_truth"]["f1"]),
                    number(r["metrics"]["micro_full_truth"]["recall"]),
                    number(r["metrics"]["micro_representable"]["f1"]),
                    number(r["metrics"]["micro_representable"]["precision"]),
                    number(r["metrics"]["mean_average_precision_supported"]),
                ]
                for r in result["runs"]
            ],
        )
        section(
            f"Threshold-free quality: {category} (common outputs, micro)",
            [
                "Model",
                "Best F1 (threshold)",
                "Precision @ recall 0.6",
                "Recall @ precision 0.6",
                "Calibration error (scores >= 0.1)",
            ],
            [
                [
                    r["profile_id"],
                    point(r["threshold_analysis"]["best_f1"], "f1"),
                    point(r["threshold_analysis"]["precision_at_recall"]["0.6"], "precision"),
                    point(r["threshold_analysis"]["recall_at_precision"]["0.6"], "recall"),
                    number(
                        r["threshold_analysis"]["calibration"]["upper_bins"]["expected_calibration_error"]
                    ),
                ]
                for r in result["runs"]
            ],
        )
        section(
            f"Per-image quality: {category} (each image weighted once)",
            [
                "Model",
                "Per-image F1 (full truth)",
                "Per-image recall (full truth)",
                "Per-image precision",
                "Recall @ top 10",
                "F1 p10 / median",
                "No hits / images with common positives",
            ],
            [
                [
                    r["profile_id"],
                    number(r["per_image"]["metrics"]["sample_f1_full_truth"]["value"]),
                    number(r["per_image"]["metrics"]["sample_recall_full_truth"]["value"]),
                    number(r["per_image"]["metrics"]["sample_precision"]["value"]),
                    number(r["per_image"]["metrics"]["sample_recall_at_10"]["value"]),
                    f"{number(r['per_image']['sample_f1_full_truth_quantiles']['p10'])} / "
                    f"{number(r['per_image']['sample_f1_full_truth_quantiles']['p50'])}",
                    f"{r['per_image']['images_with_common_positives_but_no_hits']} / "
                    f"{r['per_image']['metrics']['sample_recall_representable']['images']}",
                ]
                for r in result["runs"]
            ],
        )
        if "bootstrap" not in result:
            continue
        profiles = {r["run_id"]: r["profile_id"] for r in result["runs"]}
        section(
            f"Paired differences: {category} (minus {profiles[result['bootstrap']['runs'][0]['run_id']]}, 95% bootstrap interval)",
            ["Model", "Full-truth F1", "Common-output F1", "Common-output precision", "mAP", "Per-image F1"],
            [
                [profiles[r["run_id"]]]
                + [
                    interval(r["metrics"][name])
                    for name in (
                        "micro_full_truth_f1",
                        "micro_representable_f1",
                        "micro_representable_precision",
                        "mean_average_precision_supported",
                        "sample_f1_full_truth",
                    )
                ]
                for r in result["bootstrap"]["differences_from_reference"]
            ],
        )
    section(
        "Native full-vocabulary quality (own successful samples)",
        ["Model", "Category", "Full-truth F1", "Recall", "mAP"],
        [
            [
                r["profile_id"],
                category,
                number(values["micro_full_truth"]["f1"]),
                number(values["micro_full_truth"]["recall"]),
                number(values["mean_average_precision_supported"]),
            ]
            for r in data["runs"]
            for category, values in r["native_quality"].items()
        ],
    )
    section(
        "Hardware and software",
        [
            "Model",
            "CPU",
            "RAM GiB",
            "NVIDIA inventory / driver",
            "Provider",
            "Runtime",
            "Python",
            "Vision Bench",
        ],
        [
            [
                r["profile_id"],
                r["runtime"]["environment"].get("cpu") or "not recorded",
                number(r["runtime"]["environment"].get("ram_bytes") / 1024**3)
                if r["runtime"]["environment"].get("ram_bytes")
                else "not recorded",
                "; ".join(
                    f"{d['name']} / {d['driver']}"
                    for d in r["runtime"]["environment"].get("gpu_inventory", {}).get("devices", [])
                )
                or "not recorded",
                r["runtime"]["model"].get("primary_provider", "not recorded"),
                f"PyTorch {r['runtime']['model']['torch']}"
                if "torch" in r["runtime"]["model"]
                else f"ONNX Runtime {r['runtime']['model'].get('onnxruntime', 'not recorded')}",
                r["runtime"].get("python", "not recorded").split(" ")[0],
                r["runtime"]["libraries"].get("vision-bench", "not recorded"),
            ]
            for r in data["runs"]
        ],
    )
    dataset = data["dataset"]
    description = f"Dataset: {dataset['id']} · revision {dataset['revision']} · {dataset['selected_count']} selected · {dataset['annotation_kind']} annotations"
    reference = (
        ""
        if not dataset["public_reference_url"]
        else f"<p><a href='{text(dataset['public_reference_url'])}'>Public dataset / selection reference</a></p>"
    )
    details = html.escape(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    notes = "".join(f"<li>{text(note)}</li>" for note in NOTES)
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{text(data["title"])}</title><style>body{{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;line-height:1.6;color:#202630}}h1{{line-height:1.2}}.scroll,pre{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #ccd2df}}pre{{padding:16px;background:#f2f4f8}}li{{margin:8px 0}}</style>
<h1>{text(data["title"])}</h1><p>{text(description)}</p><p>{text(dataset["author_description"])}</p>{reference}
<p>Hardware note (author supplied): {text(data["author_hardware_note"]["text"] or "none")}</p>
{"".join(sections)}<h2>Interpretation and limits</h2><ul>{notes}</ul>
<h2>Model revisions, hashes, thresholds and complete aggregate data</h2><details><summary>Show experiment details</summary><pre>{details}</pre></details>
<p><a href="results.json">Download aggregate JSON</a> · <a href="SHA256SUMS">Checksums</a></p></html>"""
    readme = f"---\ntitle: {json.dumps(data['title'], ensure_ascii=False)}\nsdk: static\napp_file: index.html\n---\n\n# {md(data['title'])}\n\n{md(description)}\n\n{md(dataset['author_description'])}\n\n"
    readme += f"Public dataset reference: {md(dataset['public_reference_url'] or 'not supplied')}\n\n"
    readme += f"Hardware note (author supplied): {md(data['author_hardware_note']['text'] or 'none')}\n\n"
    readme += (
        "\n".join(markdown)
        + "## Interpretation and limits\n\n"
        + "\n".join(f"- {md(note)}" for note in NOTES)
    )
    readme += "\n\nFull revisions, artifact hashes, thresholds and aggregate metrics: [results.json](results.json).\n"
    return page, readme
