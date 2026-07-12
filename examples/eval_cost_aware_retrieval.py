"""Evaluate cost-aware retrieval controllers on a saved behavioral hypergraph.

This script is intentionally reward-free. It compares retrieval cost/latency for:
profile, dual_path, query_adaptive, progressive, and budget.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import profile_centric_hypergraph_eval as base_eval  # noqa: E402
from hypermem.cost_aware_retrieval import retrieve_cost_aware  # noqa: E402
from hypermem.dual_path_retrieval import retrieve_dual_path  # noqa: E402
from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory  # noqa: E402
from hypermem.query_router import route_query, route_to_dict  # noqa: E402

METHOD_NAMES = {
    "profile": "profile_only",
    "dual_path": "dual_path_always",
    "query_adaptive": "query_adaptive_path_selection",
    "progressive": "progressive_expansion",
    "budget": "budget_aware_evidence",
}


def load_questions(path: str, max_questions: int = 0) -> List[Dict[str, Any]]:
    rows = base_eval.normalize_questions(base_eval.read_json_or_jsonl(Path(path)))
    return rows[:max_questions] if max_questions and len(rows) > max_questions else rows


def split_eval_questions(rows: Sequence[Dict[str, Any]], train_ratio: float, eval_split: str) -> List[Dict[str, Any]]:
    train_q, test_q = base_eval.split_questions(rows, train_ratio)
    if eval_split == "train":
        return list(train_q)
    if eval_split == "test":
        return list(test_q)
    return list(rows)


def retrieve(memory: ProfileCentricHypergraphMemory, question: str, args: argparse.Namespace, mode: str):
    if mode == "profile":
        return memory.retrieve(
            question,
            top_k_edges=args.top_k_edges,
            top_k_facts=args.top_k_facts,
            max_tokens=args.max_tokens,
            use_utility=False,
            fallback=not args.no_fallback,
            sufficiency_threshold=args.sufficiency_threshold,
        )
    if mode == "dual_path":
        return retrieve_dual_path(
            memory,
            question,
            top_k_edges=args.top_k_edges,
            top_k_facts=args.top_k_facts,
            max_tokens=args.max_tokens,
            use_utility=False,
            fallback=not args.no_fallback,
            sufficiency_threshold=args.sufficiency_threshold,
            top_k_topics=args.top_k_topics,
            top_k_episodes=args.top_k_episodes,
        )
    return retrieve_cost_aware(
        memory,
        question,
        strategy=mode,
        top_k_edges=args.top_k_edges,
        top_k_facts=args.top_k_facts,
        max_tokens=args.max_tokens,
        use_utility=False,
        fallback=not args.no_fallback,
        sufficiency_threshold=args.sufficiency_threshold,
        top_k_topics=args.top_k_topics,
        top_k_episodes=args.top_k_episodes,
        budget_ratio=args.budget_ratio,
        expansion_ratio=args.expansion_ratio,
        representative_facts_per_edge=args.representative_facts_per_edge,
    )


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    base = base_eval.summarize(rows)
    n = max(1, len(rows))
    base.update(
        {
            "latency_ms": round(sum(float(r.get("latency_ms", 0.0)) for r in rows) / n, 4),
            "selected_facts": round(sum(float(r.get("selected_facts", 0.0)) for r in rows) / n, 4),
            "selected_edges": round(sum(float(r.get("selected_edges", 0.0)) for r in rows) / n, 4),
        }
    )
    return base


def run_method(memory_path: str, questions: Sequence[Dict[str, Any]], args: argparse.Namespace, mode: str, trace_file) -> List[Dict[str, Any]]:
    memory = ProfileCentricHypergraphMemory.load(memory_path)
    method = METHOD_NAMES[mode]
    rows: List[Dict[str, Any]] = []
    started = base_eval.log_method_start(method, len(questions))
    iterator = base_eval.progress(questions, total=len(questions), desc=f"[qa] {method}", enabled=not args.no_progress)
    for idx, q in enumerate(iterator, start=1):
        route = route_query(q["question"])
        t0 = time.perf_counter()
        result = retrieve(memory, q["question"], args, mode)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        row, reward, hit, _ = base_eval.row_from_result(method, q, result, update_used=False)
        row.update(route_to_dict(route))
        row["retrieval_mode"] = mode
        row["channel"] = result.channel
        row["latency_ms"] = round(latency_ms, 4)
        row["selected_facts"] = len(result.selected_facts)
        row["selected_edges"] = len(result.selected_edges)
        rows.append(row)
        trace_file.write(
            json.dumps(
                {
                    **row,
                    "gold": q["gold"],
                    "evidence": [fact.content for fact in result.selected_facts],
                    "debug": result.debug_scores,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        if not args.no_progress and idx % 100 == 0:
            print(f"[qa] {method} {idx}/{len(questions)}", flush=True)
    base_eval.log_method_end(method, started, rows)
    return rows


def parse_modes(raw: str) -> List[str]:
    if raw == "all":
        return ["profile", "dual_path", "query_adaptive", "progressive", "budget"]
    if raw == "cost_aware":
        return ["query_adaptive", "progressive", "budget"]
    modes = [x.strip() for x in raw.split(",") if x.strip()]
    valid = set(METHOD_NAMES)
    unknown = [x for x in modes if x not in valid]
    if unknown:
        raise ValueError(f"unknown modes: {unknown}; valid={sorted(valid)} plus all/cost_aware")
    return modes


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_questions = load_questions(args.questions_json, args.max_questions)
    questions = split_eval_questions(all_questions, args.train_ratio, args.eval_split)
    modes = parse_modes(args.modes)
    print(
        f"[cost-aware-eval] graph={args.memory_graph} all_questions={len(all_questions)} "
        f"eval_split={args.eval_split} eval_questions={len(questions)} modes={modes}",
        flush=True,
    )

    all_rows: List[Dict[str, Any]] = []
    with (out_dir / "cost_aware_trace.jsonl").open("w", encoding="utf-8") as trace:
        for mode in modes:
            rows = run_method(args.memory_graph, questions, args, mode, trace)
            all_rows.extend(rows)

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        by_method.setdefault(row["method"], []).append(row)

    result_csv = out_dir / "cost_aware_results.csv"
    with result_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = []
        seen = set()
        for row in all_rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "memory_graph": args.memory_graph,
        "questions_json": args.questions_json,
        "num_all_questions": len(all_questions),
        "eval_split": args.eval_split,
        "num_eval_questions": len(questions),
        "modes": modes,
        "max_tokens": args.max_tokens,
        "top_k_edges": args.top_k_edges,
        "top_k_facts": args.top_k_facts,
        "top_k_topics": args.top_k_topics,
        "top_k_episodes": args.top_k_episodes,
        "budget_ratio": args.budget_ratio,
        "expansion_ratio": args.expansion_ratio,
        "representative_facts_per_edge": args.representative_facts_per_edge,
        "methods": {method: summarize(rows) for method, rows in by_method.items()},
    }
    (out_dir / "cost_aware_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_csv = out_dir / "cost_aware_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "n",
            "accuracy",
            "hit",
            "recall",
            "tokens",
            "reward",
            "fallback_rate",
            "latency_ms",
            "selected_facts",
            "selected_edges",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, rows in by_method.items():
            writer.writerow({"method": method, **summarize(rows)})

    print("Cost-aware Retrieval Eval")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote:", out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-graph", required=True)
    parser.add_argument("--questions-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--modes", default="all", help="all, cost_aware, or comma list: profile,dual_path,query_adaptive,progressive,budget")
    parser.add_argument("--eval-split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--top-k-edges", type=int, default=3)
    parser.add_argument("--top-k-facts", type=int, default=8)
    parser.add_argument("--top-k-topics", type=int, default=5)
    parser.add_argument("--top-k-episodes", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--sufficiency-threshold", type=float, default=0.10)
    parser.add_argument("--budget-ratio", type=float, default=0.65)
    parser.add_argument("--expansion-ratio", type=float, default=0.55)
    parser.add_argument("--representative-facts-per-edge", type=int, default=2)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
