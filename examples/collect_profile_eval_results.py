"""Collect profile hyperedge pool evaluation summaries into CSV/JSON.

Example:
    python examples/collect_profile_eval_results.py \
      --root outputs/formal_profile_eval \
      --out-prefix outputs/formal_profile_eval/summary

It scans subdirectories for profile_hyperedge_pool_summary.json and writes:
    summary.csv
    summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


SUMMARY_NAME = "profile_hyperedge_pool_summary.json"


def flatten_summary(path: Path, root: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rel = path.parent.relative_to(root).as_posix()
    row: Dict[str, Any] = {
        "run": rel,
        "summary_path": path.as_posix(),
        "n": data.get("n", 0),
        "typing_mode": data.get("typing_mode", ""),
        "hit": data.get("hit", 0.0),
        "recall": data.get("recall", 0.0),
        "tokens": data.get("tokens", 0.0),
        "reward": data.get("reward", 0.0),
        "fallback_rate": data.get("fallback_rate", 0.0),
        "fast_channel_rate": round(1.0 - float(data.get("fallback_rate", 0.0)), 6),
        "num_edges": data.get("num_edges", 0),
        "active_edges": data.get("active_edges", 0),
        "discovery_buffer_size": data.get("discovery_buffer_size", 0),
    }
    edge_counts = data.get("edge_type_counts", {}) or {}
    for key, val in edge_counts.items():
        row[f"edge_{key}"] = val
    return row


def collect(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob(SUMMARY_NAME)):
        rows.append(flatten_summary(path, root))
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    preferred = [
        "run",
        "typing_mode",
        "n",
        "hit",
        "recall",
        "tokens",
        "reward",
        "fallback_rate",
        "fast_channel_rate",
        "num_edges",
        "active_edges",
        "discovery_buffer_size",
        "summary_path",
    ]
    fieldnames = [x for x in preferred if x in fieldnames] + [x for x in fieldnames if x not in preferred]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Evaluation output root directory.")
    parser.add_argument("--out-prefix", required=True, help="Output prefix, e.g. outputs/eval/summary.")
    args = parser.parse_args()

    root = Path(args.root)
    out_prefix = Path(args.out_prefix)
    rows = collect(root)
    write_csv(rows, out_prefix.with_suffix(".csv"))
    out_prefix.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"collected {len(rows)} summaries")
    print("wrote", out_prefix.with_suffix(".csv"))
    print("wrote", out_prefix.with_suffix(".json"))


if __name__ == "__main__":
    main()
