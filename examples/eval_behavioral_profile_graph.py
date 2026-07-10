"""Cost-aware evaluation for saved behavioral-hypergraph memory.

Reward learning and dual-path-always retrieval are intentionally removed.
The evaluator compares lower-cost strategies for a Topic-Episode-Fact hierarchy
plus behavioral hyperedges on the same saved memory graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import profile_centric_hypergraph_eval as base_eval  # noqa: E402
from hypermem.cost_aware_retrieval import (  # noqa: E402
    retrieve_budget_aware,
    retrieve_progressive,
    retrieve_topic_episode_only,
)
from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory, ProfileRetrievalResult  # noqa: E402
from hypermem.query_router import route_query, route_to_dict  # noqa: E402

METHODS = ["profile_full", "topic_episode", "progressive", "budget", "adaptive_budget", "adaptive_tiny"]


def load_questions(path: str, max_questions: int = 0) -> List[Dict[str, Any]]:
    rows = base_eval.normalize_questions(base_eval.read_json_or_jsonl(Path(path)))
    return rows[:max_questions] if max_questions and len(rows) > max_questions else rows


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def method_list(text: str) -> List[str]:
    if text.strip().lower() == "all":
        return list(METHODS)
    out = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in out if x not in METHODS]
    if bad:
        raise ValueError(f"unknown methods={bad}; valid={METHODS}")
    return out or list(METHODS)


def debug_value(result: ProfileRetrievalResult, keys: Sequence[str], default: Any = 0) -> Any:
    for item in result.debug_scores:
        for key in keys:
            if key in item:
                return item.get(key, default)
    return default


def count_path(result: ProfileRetrievalResult, path_name: str) -> int:
    return sum(1 for item in result.debug_scores if item.get("path") == path_name)


def adaptive_retrieve(
    memory: ProfileCentricHypergraphMemory,
    question: str,
    args: argparse.Namespace,
    *,
    route: str,
    tiny: bool,
) -> ProfileRetrievalResult:
    max_tokens = args.tiny_budget_tokens if tiny else int(args.max_tokens * args.budget_ratio)
    top_k_facts = max(2, min(args.top_k_facts, 4 if tiny else 6))
    top_k_topics = max(1, min(args.top_k_topics, 2 if tiny else args.top_k_topics))
    top_k_episodes = max(2, min(args.top_k_episodes, 3 if tiny else args.top_k_episodes))

    if route == "episodic":
        return retrieve_topic_episode_only(
            memory,
            question,
            top_k_facts=top_k_facts,
            max_tokens=max_tokens,
            top_k_topics=top_k_topics,
            top_k_episodes=top_k_episodes,
        )

    if route == "behavioral":
        return retrieve_progressive(
            memory,
            question,
            top_k_edges=max(1, min(args.top_k_edges, 2 if tiny else args.top_k_edges)),
            top_k_facts=top_k_facts,
            max_tokens=max_tokens,
            use_utility=False,
            top_k_topics=top_k_topics,
            top_k_episodes=top_k_episodes,
            representative_facts_per_edge=1 if tiny else args.representative_facts_per_edge,
            expansion_ratio=1.0,
        )

    return retrieve_budget_aware(
        memory,
        question,
        top_k_edges=max(1, min(args.top_k_edges, 2 if tiny else args.top_k_edges)),
        top_k_facts=top_k_facts,
        max_tokens=max_tokens,
        use_utility=False,
        top_k_topics=top_k_topics,
        top_k_episodes=top_k_episodes,
        budget_ratio=1.0,
    )


def retrieve(
    memory: ProfileCentricHypergraphMemory,
    question: str,
    args: argparse.Namespace,
    *,
    method: str,
    route: str,
) -> ProfileRetrievalResult:
    if method == "profile_full":
        return memory.retrieve(
            question,
            top_k_edges=args.top_k_edges,
            top_k_facts=args.top_k_facts,
            max_tokens=args.max_tokens,
            use_utility=False,
            fallback=not args.no_fallback,
            sufficiency_threshold=args.sufficiency_threshold,
        )
    if method == "topic_episode":
        return retrieve_topic_episode_only(
            memory,
            question,
            top_k_facts=args.top_k_facts,
            max_tokens=int(args.max_tokens * args.budget_ratio),
            top_k_topics=args.top_k_topics,
            top_k_episodes=args.top_k_episodes,
        )
    if method == "progressive":
        return retrieve_progressive(
            memory,
            question,
            top_k_edges=args.top_k_edges,
            top_k_facts=args.top_k_facts,
            max_tokens=args.max_tokens,
            use_utility=False,
            top_k_topics=args.top_k_topics,
            top_k_episodes=args.top_k_episodes,
            representative_facts_per_edge=args.representative_facts_per_edge,
            expansion_ratio=args.expansion_ratio,
        )
    if method == "budget":
        return retrieve_budget_aware(
            memory,
            question,
            top_k_edges=args.top_k_edges,
            top_k_facts=args.top_k_facts,
            max_tokens=args.max_tokens,
            use_utility=False,
            top_k_topics=args.top_k_topics,
            top_k_episodes=args.top_k_episodes,
            budget_ratio=args.budget_ratio,
        )
    if method == "adaptive_budget":
        return adaptive_retrieve(memory, question, args, route=route, tiny=False)
    if method == "adaptive_tiny":
        return adaptive_retrieve(memory, question, args, route=route, tiny=True)
    raise ValueError(method)


def run_method(memory, questions, args, method: str, trace_file) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    started = base_eval.log_method_start(method, len(questions))
    iterator = base_eval.progress(questions, total=len(questions), desc=f"[qa] {method}", enabled=not args.no_progress)
    for idx, q in enumerate(iterator, start=1):
        route = route_query(q["question"])
        t0 = time.time()
        result = retrieve(memory, q["question"], args, method=method, route=route.route)
        elapsed_ms = round((time.time() - t0) * 1000.0, 3)
        row, _, _, _ = base_eval.row_from_result(method, q, result, update_used=False)
        row.update(route_to_dict(route))
        row.update({
            "strategy": method,
            "retrieval_ms": elapsed_ms,
            "candidate_facts": debug_value(result, ["candidate_facts"], len(result.selected_facts)),
            "expanded_edges": debug_value(result, ["expanded_edges"], len(result.selected_edges)),
            "expanded_topics": debug_value(result, ["expanded_topics"], count_path(result, "topic")),
            "expanded_episodes": debug_value(result, ["expanded_episodes"], count_path(result, "episode")),
            "selected_facts_debug": debug_value(result, ["selected_facts"], len(result.selected_facts)),
            "token_budget": debug_value(result, ["token_budget"], args.max_tokens),
        })
        rows.append(row)
        trace_file.write(json.dumps({
            **row,
            "gold": q["gold"],
            "debug": result.debug_scores,
            "evidence": [f.content for f in result.selected_facts],
        }, ensure_ascii=False) + "\n")
        trace_file.flush()
        if not args.no_progress and idx % 100 == 0:
            print(f"[qa] {method} {idx}/{len(questions)} avg={(time.time() - started) / idx:.4f}s/q", flush=True)
    base_eval.log_method_end(method, started, rows)
    return rows


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    base = base_eval.summarize(rows)
    n = max(1, len(rows))
    base.update({
        "retrieval_ms": round(sum(float(r.get("retrieval_ms", 0.0)) for r in rows) / n, 3),
        "num_facts": round(sum(float(r.get("num_facts", 0.0)) for r in rows) / n, 3),
        "candidate_facts": round(sum(float(r.get("candidate_facts", 0.0)) for r in rows) / n, 3),
        "expanded_edges": round(sum(float(r.get("expanded_edges", 0.0)) for r in rows) / n, 3),
        "expanded_topics": round(sum(float(r.get("expanded_topics", 0.0)) for r in rows) / n, 3),
        "expanded_episodes": round(sum(float(r.get("expanded_episodes", 0.0)) for r in rows) / n, 3),
    })
    return base


def route_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        route = str(row.get("route", "unknown"))
        out[route] = out.get(route, 0) + 1
    return out


def run_eval(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.questions_json, args.max_questions)
    if args.eval_scope == "test":
        _, eval_q = base_eval.split_questions(questions, args.train_ratio)
    else:
        eval_q = questions
    methods = method_list(args.methods)
    print(
        f"[cost-aware-eval] graph={args.memory_graph} questions={len(questions)} eval={len(eval_q)} "
        f"eval_scope={args.eval_scope} methods={methods}",
        flush=True,
    )

    memory = ProfileCentricHypergraphMemory.load(args.memory_graph)
    all_rows: List[Dict[str, Any]] = []
    with (out / "cost_aware_trace.jsonl").open("w", encoding="utf-8") as trace:
        for method in methods:
            all_rows.extend(run_method(memory, eval_q, args, method, trace))

    write_rows(out / "cost_aware_results.csv", all_rows)
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        by_method.setdefault(row["method"], []).append(row)

    summary = {
        "pipeline": [
            "load_saved_behavioral_hypergraph_memory",
            "reward_learning_removed",
            "dual_path_always_removed",
            "evaluate_profile_full_baseline",
            "evaluate_topic_episode_fact_budget_retrieval",
            "evaluate_progressive_hypergraph_expansion",
            "evaluate_value_per_token_budget_selection",
            "evaluate_query_adaptive_budget_selection",
        ],
        "design_note": "Reward learning and dual-path-always retrieval are disabled; this run focuses on lookup speed and token cost.",
        "memory_graph": args.memory_graph,
        "num_questions": len(questions),
        "num_eval_questions": len(eval_q),
        "eval_scope": args.eval_scope,
        "methods": {m: summarize(rs) for m, rs in by_method.items()},
        "route_counts": route_counts(all_rows),
    }
    (out / "cost_aware_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "behavioral_profile_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "method", "n", "accuracy", "hit", "recall", "tokens", "reward", "fallback_rate",
        "retrieval_ms", "num_facts", "candidate_facts", "expanded_edges", "expanded_topics", "expanded_episodes",
    ]
    for name in ["cost_aware_summary.csv", "behavioral_profile_summary.csv"]:
        with (out / name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for method, rows in by_method.items():
                writer.writerow({"method": method, **summarize(rows)})

    print("Cost-aware Hypergraph Memory Eval")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote:", out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-graph", required=True)
    parser.add_argument("--questions-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default="all", help=f"comma list or all; valid={METHODS}")
    parser.add_argument("--eval-scope", choices=["all", "test"], default="all")
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--top-k-edges", type=int, default=3)
    parser.add_argument("--top-k-facts", type=int, default=8)
    parser.add_argument("--top-k-topics", type=int, default=3)
    parser.add_argument("--top-k-episodes", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--budget-ratio", type=float, default=0.55)
    parser.add_argument("--expansion-ratio", type=float, default=0.45)
    parser.add_argument("--tiny-budget-tokens", type=int, default=110)
    parser.add_argument("--representative-facts-per-edge", type=int, default=2)
    parser.add_argument("--sufficiency-threshold", type=float, default=0.10)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")

    # Deprecated compatibility args from previous reward/dual-path evaluator.
    parser.add_argument("--retrieval-mode", default="", help="Deprecated/no-op. Use --methods.")
    parser.add_argument("--reward-modes", default="", help="Deprecated/no-op.")
    parser.add_argument("--learning-rate", type=float, default=0.18, help="Deprecated/no-op.")
    parser.add_argument("--allow-deactivate", action="store_true", help="Deprecated/no-op.")
    parser.add_argument("--pool-top-k", type=int, default=50, help="Deprecated/no-op.")
    parser.add_argument("--pool-min-value", type=float, default=0.0, help="Deprecated/no-op.")
    parser.add_argument("--pool-min-utility", type=float, default=0.0, help="Deprecated/no-op.")
    parser.add_argument("--pool-require-positive-feedback", action="store_true", help="Deprecated/no-op.")
    parser.add_argument("--dual-profile-weight", type=float, default=0.38, help="Deprecated/no-op.")
    parser.add_argument("--dual-topic-weight", type=float, default=0.32, help="Deprecated/no-op.")
    parser.add_argument("--dual-episode-weight", type=float, default=0.18, help="Deprecated/no-op.")
    parser.add_argument("--dual-alignment-weight", type=float, default=0.12, help="Deprecated/no-op.")
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
