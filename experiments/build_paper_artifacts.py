"""Build paper tables and lightweight figures from completed LoCoMo runs.

Each input directory must contain the outputs produced by
``rejudge_trace_hypermem_style.py``.  The script deliberately consumes only
the three paper metrics, keeping retrieval diagnostics in their raw run files.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


METRICS = ("llm_acc", "retrieval_tokens", "retrieval_latency_ms")


def parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"run must use NAME=PATH, got: {spec}")
    name, raw_path = spec.split("=", 1)
    return name.strip(), Path(raw_path).expanduser()


def read_rows(path: Path, experiment: str) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: List[Dict[str, str]] = []
    for row in rows:
        out.append({"experiment": experiment, **{key: row.get(key, "") for key in ("method", "variant", "n", *METRICS)}})
    return out


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    fields = ["experiment", "method", "variant", "n", *METRICS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict[str, str]]) -> None:
    lines = [
        "# Paper Metrics",
        "",
        "| Experiment | Method | n | llm_acc | retrieval_tokens | retrieval_latency_ms |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['method']} | {row['n']} | "
            f"{float(row['llm_acc']):.4f} | {float(row['retrieval_tokens']):.1f} | "
            f"{float(row['retrieval_latency_ms']):.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_plot(out_dir: Path, rows: List[Dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for experiment in sorted({row["experiment"] for row in rows}):
        part = [row for row in rows if row["experiment"] == experiment]
        if not part:
            continue
        fig, axis = plt.subplots(figsize=(7.2, 4.8))
        for row in part:
            axis.scatter(float(row["retrieval_latency_ms"]), float(row["llm_acc"]), s=60)
            axis.annotate(row["method"], (float(row["retrieval_latency_ms"]), float(row["llm_acc"])), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axis.set_xlabel("Retrieval latency (ms)")
        axis.set_ylabel("LLM accuracy")
        axis.set_title(f"{experiment}: accuracy-latency trade-off")
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"{experiment.lower()}_pareto.png", dpi=220)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=OUTPUT_DIR; may be repeated")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    for spec in args.run:
        name, run_dir = parse_run(spec)
        summary = run_dir / "hypermem_style_summary.csv"
        if not summary.exists():
            raise FileNotFoundError(f"missing judge summary: {summary}")
        rows.extend(read_rows(summary, name))
    rows.sort(key=lambda row: (row["experiment"], row["method"], row["variant"]))
    write_csv(out_dir / "paper_metrics.csv", rows)
    write_markdown(out_dir / "paper_metrics.md", rows)
    try_plot(out_dir, rows)


if __name__ == "__main__":
    main()
