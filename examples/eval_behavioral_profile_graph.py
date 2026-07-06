"""Evaluate a saved graph as reward-guided behavioral-profile memory.

This script supports two retrieval modes:

1. profile: query -> behavioral hyperedges -> member facts.
2. dual_path: query -> behavioral hyperedges -> facts -> episodes, and
   query -> topics -> episodes -> facts, followed by evidence alignment.

It also supports multiple reward update modes so that dual-path retrieval can be
compared under different credit-assignment assumptions without rebuilding the
memory graph.
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
from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory, ProfileRetrievalResult  # noqa: E402
from hypermem.query_router import route_query, route_to_dict  # noqa: E402

REWARD_MODES = ["answer", "contribution", "alignment", "conservative"]


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


def _selected_fact_ids(result: ProfileRetrievalResult) -> set[str]:
    return {fact.fact_id for fact in result.selected_facts}


def _edge_contribution(edge, result: ProfileRetrievalResult) -> float:
    selected = _selected_fact_ids(result)
    if not selected:
        return 0.0
    member_selected = selected & set(edge.member_fact_ids)
    return len(member_selected) / max(1, len(selected))


def _edge_alignment(edge, result: ProfileRetrievalResult) -> float:
    """Estimate whether this edge actually contributed to fused dual-path evidence."""
    contribution = _edge_contribution(edge, result)
    if contribution <= 0:
        return 0.0
    # The dual-path retriever writes fused_fact debug records with alignment_score.
    selected = _selected_fact_ids(result)
    aligned_fact_ids = set()
    for item in result.debug_scores:
        if item.get("path") != "fused_fact":
            continue
        if float(item.get("alignment_score", 0.0) or 0.0) > 0:
            fid = str(item.get("fact_id", ""))
            if fid in selected:
                aligned_fact_ids.add(fid)
    if not aligned_fact_ids:
        return contribution
    edge_aligned = aligned_fact_ids & set(edge.member_fact_ids)
    return 0.5 * contribution + 0.5 * (len(edge_aligned) / max(1, len(aligned_fact_ids)))


def _mode_reward(base_reward: float, *, hit: int, contribution: float, alignment: float, reward_mode: str) -> float:
    """Convert answer-level reward into an edge-level update signal."""
    if reward_mode == "answer":
        return base_reward
    if reward_mode == "contribution":
        if contribution <= 0:
            return 0.0
        return base_reward * contribution
    if reward_mode == "alignment":
        if alignment <= 0:
            return 0.0
        return base_reward * alignment
    if reward_mode == "conservative":
        if hit:
            return max(0.0, base_reward) * max(0.25, contribution)
        # Weak punishment only when the edge contributed evidence but the answer failed.
        return min(0.0, base_reward) * min(0.35, max(0.0, contribution))
    return base_reward


def apply_reward_update(
    memory: ProfileCentricHypergraphMemory,
    result: ProfileRetrievalResult,
    *,
    base_reward: float,
    hit: int,
    route: str,
    args: argparse.Namespace,
    reward_mode: str,
) -> Dict[str, Any]:
    stats = {
        "reward_mode": reward_mode,
        "updated_edges": 0,
        "skipped_edges": 0,
        "avg_edge_reward": 0.0,
        "avg_contribution": 0.0,
        "avg_alignment": 0.0,
    }
    edge_rewards: List[float] = []
    contributions: List[float] = []
    alignments: List[float] = []
    for edge in result.selected_edges:
        contribution = _edge_contribution(edge, result)
        alignment = _edge_alignment(edge, result)
        edge_reward = _mode_reward(base_reward, hit=hit, contribution=contribution, alignment=alignment, reward_mode=reward_mode)
        contributions.append(contribution)
        alignments.append(alignment)
        if reward_mode in {"contribution", "alignment", "conservative"} and abs(edge_reward) < 1e-12:
            stats["skipped_edges"] += 1
            continue
        update_behavioral_edges_from_feedback(
            memory,
            [edge],
            reward=edge_reward,
            hit=bool(hit),
            route=route,
            learning_rate=args.learning_rate,
            allow_deactivate=args.allow_deactivate,
        )
        stats["updated_edges"] += 1
        edge_rewards.append(edge_reward)
    if edge_rewards:
        stats["avg_edge_reward"] = round(sum(edge_rewards) / len(edge_rewards), 6)
    if contributions:
        stats["avg_contribution"] = round(sum(contributions) / len(contributions), 6)
    if alignments:
        stats["avg_alignment"] = round(sum(alignments) / len(alignments), 6)
    return stats


def run_behavioral_questions(
    memory: ProfileCentricHypergraphMemory,
    questions: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    method: str,
    *,
    use_utility: bool,
    update: bool,
    retrieval_mode: str,
    reward_mode: str,
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
        row["reward_mode"] = reward_mode
        update_stats: Dict[str, Any] = {
            "updated_edges": 0,
            "skipped_edges": 0,
            "avg_edge_reward": 0.0,
            "avg_contribution": 0.0,
            "avg_alignment": 0.0,
        }
        if update:
            update_stats = apply_reward_update(
                memory,
                result,
                base_reward=reward,
                hit=hit,
                route=route.route,
                args=args,
                reward_mode=reward_mode,
            )
        row.update(update_stats)
        rows.append(row)
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


def _reward_modes(args: argparse.Namespace) -> List[str]:
    if args.reward_modes == "all":
        return list(REWARD_MODES)
    modes = [x.strip() for x in args.reward_modes.split(",") if x.strip()]
    unknown = [x for x in modes if x not in REWARD_MODES]
    if unknown:
        raise ValueError(f"unknown reward modes: {unknown}; valid={REWARD_MODES}")
    return modes or ["answer"]


def run_eval(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.questions_json, args.max_questions)
    train_q, test_q = base_eval.split_questions(questions, args.train_ratio)
    reward_modes = _reward_modes(args)
    print(
        f"[behavioral-eval] graph={args.memory_graph} questions={len(questions)} train={len(train_q)} test={len(test_q)} "
        f"retrieval_mode={args.retrieval_mode} reward_modes={reward_modes}",
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
                reward_mode="none",
                trace_file=trace,
            )
            all_rows.extend(baseline_rows)

            for reward_mode in reward_modes:
                learned_memory = ProfileCentricHypergraphMemory.load(args.memory_graph)
                learned_memory.learning_rate = args.learning_rate
                key = f"{mode}:{reward_mode}"
                trained_memories[key] = learned_memory
                if mode == "profile":
                    train_method = f"reward_{reward_mode}_behavioral_train"
                    test_method = f"reward_{reward_mode}_behavioral_frozen_test"
                else:
                    train_method = f"reward_{reward_mode}_dual_path_train"
                    test_method = f"reward_{reward_mode}_dual_path_frozen_test"
                train_rows = run_behavioral_questions(
                    learned_memory,
                    train_q,
                    args,
                    train_method,
                    use_utility=True,
                    update=True,
                    retrieval_mode=mode,
                    reward_mode=reward_mode,
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
                    reward_mode=reward_mode,
                    trace_file=trace,
                )
                all_rows.extend(train_rows)
                all_rows.extend(test_rows)

    if "profile:answer" in trained_memories:
        trained_memories["profile:answer"].save(out_dir / "behavioral_trained_memory.json")
    if "dual_path:answer" in trained_memories:
        trained_memories["dual_path:answer"].save(out_dir / "dual_path_trained_memory.json")
    for key, memory in trained_memories.items():
        safe_key = key.replace(":", "_")
        memory.save(out_dir / f"trained_memory_{safe_key}.json")
    if "profile" in baseline_memories:
        baseline_memories["profile"].save(out_dir / "behavioral_embedding_only_memory.json")

    write_rows(out_dir / "behavioral_profile_results.csv", all_rows)

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        by_method.setdefault(row["method"], []).append(row)

    pool_source = (
        trained_memories.get("dual_path:alignment")
        or trained_memories.get("dual_path:contribution")
        or trained_memories.get("dual_path:answer")
        or trained_memories.get("profile:answer")
        or ProfileCentricHypergraphMemory.load(args.memory_graph)
    )
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
            "reward_mode_answer_or_contribution_or_alignment_or_conservative",
            "route_query_for_reward_update_only",
            "train_reward_guided_hyperedge_utility_on_behavioral_or_mixed_train_qa",
            "export_high_value_behavioral_memory_pool",
        ],
        "design_note": "Dual-path mode aligns evidence from behavioral profile hyperedges and topic-episode timeline retrieval. Multiple reward modes test credit assignment for profile-edge utility learning.",
        "memory_graph": args.memory_graph,
        "retrieval_mode": args.retrieval_mode,
        "reward_modes": reward_modes,
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
    parser.add_argument("--reward-modes", default="all", help="comma list from answer,contribution,alignment,conservative or all")
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
