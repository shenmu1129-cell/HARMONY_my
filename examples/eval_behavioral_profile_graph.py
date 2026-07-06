"""Evaluate a saved graph as reward-guided behavioral-profile memory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import profile_centric_hypergraph_eval as base_eval  # noqa: E402
from hypermem.behavioral_profile import behavioral_profile_summary, write_behavioral_pool  # noqa: E402
from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory  # noqa: E402


def load_questions(path: str, max_questions: int = 0) -> List[Dict[str, Any]]:
    rows = base_eval.normalize_questions(base_eval.read_json_or_jsonl(Path(path)))
    return rows[:max_questions] if max_questions and len(rows) > max_questions else rows


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_eval(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.questions_json, args.max_questions)
    train_q, test_q = base_eval.split_questions(questions, args.train_ratio)
    print(f"[behavioral-eval] graph={args.memory_graph} questions={len(questions)} train={len(train_q)} test={len(test_q)}", flush=True)

    all_rows: List[Dict[str, Any]] = []
    trace_path = out_dir / "behavioral_profile_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as trace:
        baseline_memory = ProfileCentricHypergraphMemory.load(args.memory_graph)
        baseline_memory.learning_rate = args.learning_rate
        baseline_rows = base_eval.run_questions(
            baseline_memory, test_q, args, "embedding_only_behavioral_profile", False, False, trace
        )
        all_rows.extend(baseline_rows)

        learned_memory = ProfileCentricHypergraphMemory.load(args.memory_graph)
        learned_memory.learning_rate = args.learning_rate
        train_rows = base_eval.run_questions(
            learned_memory, train_q, args, "reward_guided_behavioral_train", True, True, trace
        )
        test_rows = base_eval.run_questions(
            learned_memory, test_q, args, "reward_guided_behavioral_frozen_test", True, False, trace
        )
        all_rows.extend(train_rows)
        all_rows.extend(test_rows)

    learned_memory.save(out_dir / "behavioral_trained_memory.json")
    baseline_memory.save(out_dir / "behavioral_embedding_only_memory.json")
    write_rows(out_dir / "behavioral_profile_results.csv", all_rows)

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        by_method.setdefault(row["method"], []).append(row)

    pool = write_behavioral_pool(
        learned_memory,
        out_dir / "high_value_behavioral_pool.json",
        top_k=args.pool_top_k,
        min_value=args.pool_min_value,
        min_utility=args.pool_min_utility,
        require_positive_feedback=args.pool_require_positive_feedback,
    )
    summary = {
        "pipeline": [
            "load_open_set_behavioral_profile_hyperedges",
            "keep_episodic_detail_facts_in_base_memory_graph",
            "train_reward_guided_hyperedge_utility_on_train_qa",
            "export_high_value_behavioral_memory_pool",
        ],
        "design_note": "Only recurring, stable, or reward-useful dimensions are promoted to behavioral profile edges; ordinary details stay in the base fact/tree path.",
        "memory_graph": args.memory_graph,
        "learning_rate": args.learning_rate,
        "num_questions": len(questions),
        "num_train_questions": len(train_q),
        "num_test_questions": len(test_q),
        "methods": {method: base_eval.summarize(rows) for method, rows in by_method.items()},
        "behavioral_profile": behavioral_profile_summary(learned_memory, top_k=args.pool_top_k),
        "high_value_pool_size": len(pool),
    }
    (out_dir / "behavioral_profile_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "behavioral_profile_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["method", "n", "accuracy", "hit", "recall", "tokens", "reward", "fallback_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, rows in by_method.items():
            writer.writerow({"method": method, **base_eval.summarize(rows)})

    print("Reward-Guided Behavioral Profile Memory Eval")
    print(json.dumps({k: v for k, v in summary.items() if k != "behavioral_profile"}, ensure_ascii=False, indent=2))
    print("wrote:", out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-graph", required=True)
    parser.add_argument("--questions-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--top-k-edges", type=int, default=3)
    parser.add_argument("--top-k-facts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--sufficiency-threshold", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=0.18)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--pool-top-k", type=int, default=50)
    parser.add_argument("--pool-min-value", type=float, default=0.0)
    parser.add_argument("--pool-min-utility", type=float, default=0.0)
    parser.add_argument("--pool-require-positive-feedback", action="store_true")
    args = parser.parse_args()
    args.embedding_dim = 512
    args.attach_threshold = 0.52
    args.discovery_threshold = 0.55
    args.construction_mode = "loaded_behavioral_graph"
    args.batch_size = 0
    args.canonical_threshold = 0.0
    args.max_edge_facts = 0
    args.max_auto_edge_pairs = 0
    args.min_feature_support = 1
    args.consolidate_every = 0
    return args


if __name__ == "__main__":
    run_eval(parse_args())
