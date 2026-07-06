"""Evaluate a saved graph as reward-guided behavioral-profile memory.

This script supports two retrieval modes:

1. profile: query -> behavioral hyperedges -> member facts.
2. dual_path: query -> behavioral hyperedges -> facts -> episodes, and
   query -> topics -> episodes -> facts, followed by evidence alignment.
"""

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
from hypermem.behavioral_profile import (  # noqa: E402
    behavioral_profile_summary,
    update_behavioral_edges_from_feedback,
    write_behavioral_pool,
)
from hypermem.dual_path_retrieval import retrieve_dual_path  # noqa: E402
from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory  # noqa: E402
from hypermem.query_router import route_query, route_to_dict  # noqa: E402


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


def _retrieve(memory: ProfileCentricHypergraphMemory, question: str, args: argparse.Namespace, *, use_utility: bool, retrieval_mode: str):
    if retrieval_mode == "dual_path":
        return retrieve_dual_path(
            memory,
            question,
            top_k_edges=args.top_k_edges,
            top_k_facts=args.top_k_facts,
            max_tokens=args.max_tokens,
            use_utility=use_utility,
            fallback=not args.no_fallback,
            sufficiency_threshold=args.sufficiency_threshold,
            top_k_topics=args.top_k_topics,
            top_k_episodes=args.top_k_episodes,
            profile_weight=args.dual_profile_weight,
            topic_weight=args.dual_topic_weight,
            episode_weight=args.dual_episode_weight,
            alignment_weight=args.dual_alignment_weight,
        )
    return memory.retrieve(
        question,
        top_k_edges=args.top_k_edges,
        top_k_facts=args.top_k_facts,
        max_tokens=args.max_tokens,
        use_utility=use_utility,
        fallback=not args.no_fallback,
        sufficiency_threshold=args.sufficiency_threshold,
    )


def run_behavioral_questions(
    memory: ProfileCentricHypergraphMemory,
    questions: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    method: str,
    *,
    use_utility: bool,
    update: bool,
    retrieval_mode: str,
    trace_file,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    started = base_eval.log_method_start(method, len(questions))
    iterator = base_eval.progress(questions, total=len(questions), desc=f"[qa] {method}", enabled=not args.no_progress)
    for idx, q in enumerate(iterator, start=1):
        route = route_query(q["question"])
        result = _retrieve(memory, q["question"], args, use_utility=use_utility, retrieval_mode=retrieval_mode)
        row, reward, hit, _ = base_eval.row_from_result(method, q, result, update_used=update and route.update_behavioral_edges)
        row.update(route_to_dict(route))
        row["retrieval_mode"] = retrieval_mode
        rows.append(row)
        if update:
            update_behavioral_edges_from_feedback(
                memory,
                result.selected_edges,
                reward=reward,
                hit=bool(hit),
                route=route.route,
                learning_rate=args.learning_rate,
                allow_deactivate=args.allow_deactivate,
            )
        trace_file.write(json.dumps({
            **row,
            "gold": q["gold"],
            "edge_debug": result.debug_scores,
            "evidence": [fact.content for fact in result.selected_facts],
        }, ensure_ascii=False) + "\n")
        trace_file.flush()
        if not args.no_progress and idx % 100 == 0:
            print(f"[qa] {method} {idx}/{len(questions)}", flush=True)
    base_eval.log_method_end(method, started, rows)
    return rows


def route_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        route = str(row.get("route", "unknown"))
        counts[route] = counts.get(route, 0) + 1
    return counts


def run_eval(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.questions_json, args.max_questions)
    train_q, test_q = base_eval.split_questions(questions, args.train_ratio)
    print(
        f"[behavioral-eval] graph={args.memory_graph} questions={len(questions)} train={len(train_q)} test={len(test_q)} "
        f"retrieval_mode={args.retrieval_mode}",
        flush=True,
    )

    modes = ["profile", "dual_path"] if args.retrieval_mode == "both" else [args.retrieval_mode]
    all_rows: List[Dict[str, Any]] = []
    trace_path = out_dir / "behavioral_profile_trace.jsonl"
    trained_memories: Dict[str, ProfileCentricHypergraphMemory] = {}
    baseline_memories: Dict[str, ProfileCentricHypergraphMemory] = {}

    with trace_path.open("w", encoding="utf-8") as trace:
        for mode in modes:
            baseline_memory = ProfileCentricHypergraphMemory.load(args.memory_graph)
            baseline_memory.learning_rate = args.learning_rate
            baseline_memories[mode] = baseline_memory
            baseline_method = "embedding_only_behavioral_profile" if mode == "profile" else "dual_path_embedding_only"
            baseline_rows = run_behavioral_questions(
                baseline_memory,
                test_q,
                args,
                baseline_method,
                use_utility=False,
                update=False,
                retrieval_mode=mode,
                trace_file=trace,
            )
            all_rows.extend(baseline_rows)

            learned_memory = ProfileCentricHypergraphMemory.load(args.memory_graph)
            learned_memory.learning_rate = args.learning_rate
            trained_memories[mode] = learned_memory
            train_method = "reward_guided_behavioral_train" if mode == "profile" else "reward_guided_dual_path_train"
            test_method = "reward_guided_behavioral_frozen_test" if mode == "profile" else "reward_guided_dual_path_frozen_test"
            train_rows = run_behavioral_questions(
                learned_memory,
                train_q,
                args,
                train_method,
                use_utility=True,
                update=True,
                retrieval_mode=mode,
                trace_file=trace,
            )
            test_rows = run_behavioral_questions(
                learned_memory,
                test_q,
                args,
                test_method,
                use_utility=True,
                update=False,
                retrieval_mode=mode,
                trace_file=trace,
            )
            all_rows.extend(train_rows)
            all_rows.extend(test_rows)

    # Keep the profile-mode trained memory as the default legacy output. Also
    # save dual-path trained memory separately when present.
    if "profile" in trained_memories:
        trained_memories["profile"].save(out_dir / "behavioral_trained_memory.json")
    if "dual_path" in trained_memories:
        trained_memories["dual_path"].save(out_dir / "dual_path_trained_memory.json")
    if "profile" in baseline_memories:
        baseline_memories["profile"].save(out_dir / "behavioral_embedding_only_memory.json")

    write_rows(out_dir / "behavioral_profile_results.csv", all_rows)

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        by_method.setdefault(row["method"], []).append(row)

    pool_source = trained_memories.get("dual_path") or trained_memories.get("profile") or ProfileCentricHypergraphMemory.load(args.memory_graph)
    pool = write_behavioral_pool(
        pool_source,
        out_dir / "high_value_behavioral_pool.json",
        top_k=args.pool_top_k,
        min_value=args.pool_min_value,
        min_utility=args.pool_min_utility,
        require_positive_feedback=args.pool_require_positive_feedback,
    )
    summary = {
        "pipeline": [
            "load_post_hierarchy_open_set_behavioral_hyperedges",
            "profile_path_query_to_behavioral_hyperedges_to_member_facts_to_source_episodes",
            "topic_path_query_to_topics_to_episodes_to_facts",
            "episode_level_alignment_and_fact_fusion",
            "route_query_for_reward_update_only",
            "train_reward_guided_hyperedge_utility_on_behavioral_or_mixed_train_qa",
            "export_high_value_behavioral_memory_pool",
        ],
        "design_note": "Dual-path mode aligns evidence from behavioral profile hyperedges and topic-episode timeline retrieval. Existing profile-only retrieval is still evaluated for ablation.",
        "memory_graph": args.memory_graph,
        "retrieval_mode": args.retrieval_mode,
        "learning_rate": args.learning_rate,
        "num_questions": len(questions),
        "num_train_questions": len(train_q),
        "num_test_questions": len(test_q),
        "route_counts": route_counts(all_rows),
        "methods": {method: base_eval.summarize(rows) for method, rows in by_method.items()},
        "behavioral_profile": behavioral_profile_summary(pool_source, top_k=args.pool_top_k),
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
    parser.add_argument("--allow-deactivate", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--pool-top-k", type=int, default=50)
    parser.add_argument("--pool-min-value", type=float, default=0.0)
    parser.add_argument("--pool-min-utility", type=float, default=0.0)
    parser.add_argument("--pool-require-positive-feedback", action="store_true")
    parser.add_argument("--retrieval-mode", choices=["profile", "dual_path", "both"], default="both")
    parser.add_argument("--top-k-topics", type=int, default=3)
    parser.add_argument("--top-k-episodes", type=int, default=6)
    parser.add_argument("--dual-profile-weight", type=float, default=0.38)
    parser.add_argument("--dual-topic-weight", type=float, default=0.32)
    parser.add_argument("--dual-episode-weight", type=float, default=0.18)
    parser.add_argument("--dual-alignment-weight", type=float, default=0.12)
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
