from __future__ import annotations

import html
import json
from pathlib import Path


def write_report(path: Path, metrics: dict) -> None:
    """Self-contained report: no scripts, remote assets, or unescaped model/dataset text."""
    rows = []
    for category, values in metrics["categories"].items():
        micro = values["micro_full_truth"]
        ap = values["mean_average_precision_supported"]
        rows.append(
            f"<tr><td>{html.escape(category)}</td><td>{micro['f1']:.4f}</td>"
            f"<td>{micro['recall']:.4f}</td><td>{'—' if ap is None else f'{ap:.4f}'}</td>"
            f"<td>{values['coverage']['oov_occurrences']}</td></tr>"
        )
    title = html.escape(str(metrics["profile_id"]))
    warm = metrics["performance"]["warm"]
    timing = (
        "No warm samples"
        if warm["median_ms"] is None
        else f"Warm median: {warm['median_ms']:.2f} ms; p95: {warm['p95_ms']:.2f} ms"
    )
    memory = metrics["performance"]["gpu_memory"]
    if memory["status"] in {"observed", "partial"}:

        def mib(value):
            return "unavailable" if value is None else f"{value / 1024**2:.1f} MiB"

        memory_text = (
            f"WDDM observed dedicated high water: {mib(memory['dedicated_observed_high_water_bytes'])}; "
            f"shared: {mib(memory['shared_observed_high_water_bytes'])}. Sampled, not an instantaneous peak."
        )
    else:
        memory_text = f"GPU memory: {memory['status']} ({memory.get('reason', 'not recorded')})."
    payload = html.escape(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
    path.write_text(
        f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vision Bench · {title}</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;color:#202630}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}
pre{{overflow:auto;background:#f3f5f7;padding:16px}}.note{{line-height:1.6;color:#4b5563}}</style>
<h1>{title}</h1><p>{metrics["outcomes"]["successful"]} / {metrics["outcomes"]["attempted"]} successful samples</p>
<p>{html.escape(timing)}</p><p>{html.escape(memory_text)}</p>
<p class="note">Annotation kind: {html.escape(metrics["annotation_kind"])}. Unlabeled predictions are not
verified false positives. F1 includes out-of-vocabulary positive labels; mAP uses represented labels.
Thresholds are configured operating points, not optimized on this evaluation set.</p>
<table><thead><tr><th>Category</th><th>Full F1</th><th>Full recall</th><th>mAP</th><th>OOV positives</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table><h2>Details and performance</h2><pre>{payload}</pre></html>
""",
        encoding="utf-8",
    )
