#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinela_models.regional import get_region_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a regional benchmark report from evaluation artifacts.")
    parser.add_argument("--region", default="south_america")
    parser.add_argument("--config", default=str(ROOT / "configs" / "regional_goes.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--eval-json", default=None)
    parser.add_argument("--history-json", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def manifest_counts(path: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    splits: dict[str, int] = {}
    total = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total += 1
            counts[row.get("label_name", "")] = counts.get(row.get("label_name", ""), 0) + 1
            splits[row.get("split", "")] = splits.get(row.get("split", ""), 0) + 1
    return {"total": total, "by_label": counts, "by_split": splits}


def main() -> None:
    args = parse_args()
    cfg = get_region_config(args.region, args.config)
    region_root = cfg.region_root(args.data_root)
    eval_path = Path(args.eval_json) if args.eval_json else region_root / "evaluation" / "best_test.json"
    history_path = Path(args.history_json) if args.history_json else ROOT / "models" / "regional" / cfg.name / "history.json"
    manifest_path = Path(args.manifest) if args.manifest else region_root / "manifest.csv"
    out_dir = Path(args.out_dir) if args.out_dir else region_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluation = load_json(eval_path)
    history = load_json(history_path) if history_path.exists() else []
    counts = manifest_counts(manifest_path)
    best_epoch = history[-1] if history else {}
    report = {
        "region": cfg.name,
        "dataset": counts,
        "evaluation": evaluation,
        "training_last_epoch": best_epoch,
        "nasa_firms_baseline": {
            "role": "bootstrap_label_source_and_holdout_reference",
            "lead_time_claim": "not_measured",
            "caveat": (
                "This report evaluates a GOES model against FIRMS-derived bootstrap labels. "
                "It is not an independent NASA lead-time benchmark and should not be marketed as earlier-than-NASA evidence."
            ),
            "next_benchmark_needed": (
                "Run a pre-FIRMS replay scan where GOES windows before each FIRMS timestamp are evaluated, "
                "then compare the first model alert time against the FIRMS acquisition time."
            ),
        },
    }
    json_path = out_dir / "regional_benchmark.json"
    md_path = out_dir / "regional_benchmark.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = [
        f"# Regional Benchmark: {cfg.name}",
        "",
        "## Dataset",
        "",
        f"- Total samples: {counts['total']}",
        f"- By label: `{json.dumps(counts['by_label'], sort_keys=True)}`",
        f"- By split: `{json.dumps(counts['by_split'], sort_keys=True)}`",
        "",
        "## Test Metrics",
        "",
        f"- Fire precision: {evaluation.get('precision', 0):.4f}",
        f"- Fire recall: {evaluation.get('recall', 0):.4f}",
        f"- Fire F1: {evaluation.get('f1', 0):.4f}",
        f"- Binary labels: `{json.dumps(evaluation.get('labels', {}), sort_keys=True)}`",
        f"- Original label counts: `{json.dumps(evaluation.get('original_class_counts', {}), sort_keys=True)}`",
        f"- Class-wise: `{json.dumps(evaluation.get('class_wise', {}), sort_keys=True)}`",
        "",
        "## NASA FIRMS Context",
        "",
        "FIRMS is being used here as a bootstrap label source and holdout reference.",
        "This is not yet an independent lead-time benchmark against NASA FIRMS.",
        "",
        "To support an earlier-than-NASA claim, run a separate pre-FIRMS replay benchmark: scan GOES imagery before each FIRMS acquisition time, record the first model alert, then compare that timestamp to the FIRMS timestamp.",
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
