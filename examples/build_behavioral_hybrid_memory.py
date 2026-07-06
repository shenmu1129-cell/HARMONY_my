"""Build the full hybrid memory pipeline.

Required stage order implemented here:

1. LLM extracts Topic -> Episode -> Fact hierarchy.
2. Behavioral profile hyperedges are induced from the extracted facts and
   episode/topic metadata.
3. A lightweight RL/bandit-style utility prior is initialized from hyperedge
   features; later QA feedback can update it through eval_behavioral_profile_graph.py.
4. Facts and behavioral hyperedges are embedded/indexed inside
   ProfileCentricHypergraphMemory.
5. Retrieval is performed by first selecting behavioral hyperedges and then
   selecting member facts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypermem.behavioral_profile import (  # noqa: E402
    behavioral_profile_summary,
    initialize_behavioral_priors,
    write_behavioral_pool,
)
from hypermem.llm_hierarchy_builder import (  # noqa: E402
    extract_topic_episode_fact_hierarchy,
    flatten_hierarchy_facts,
    save_hierarchy_outputs,
)
from hypermem.llm_profile_builder_en import build_english_llm_profile_hypergraph_from_rows  # noqa: E402
from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory  # noqa: E402


def read_json_or_jsonl(path: Path) -> List[Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    rows: List[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def multi_edge_stats(memory: ProfileCentricHypergraphMemory) -> Dict[str, Any]:
    memberships: Dict[str, List[str]] = {}
    for edge_id, edge in memory.edges.items():
        if edge.status != "active":
            continue
        for fact_id in edge.member_fact_ids:
            memberships.setdefault(fact_id, []).append(edge_id)
    multi = {fact_id: edge_ids for fact_id, edge_ids in memberships.items() if len(edge_ids) >= 2}
    active_edges = [edge for edge in memory.edges.values() if edge.status == "active"]
    facts_per_edge = [len(edge.member_fact_ids) for edge in active_edges]
    return {
        "num_membership_facts": len(memberships),
        "num_multi_edge_facts": len(multi),
        "multi_edge_fact_ratio": round(len(multi) / max(1, len(memory.facts)), 6),
        "avg_facts_per_active_edge": round(sum(facts_per_edge) / max(1, len(facts_per_edge)), 3),
        "max_facts_per_active_edge": max(facts_per_edge) if facts_per_edge else 0,
    }


def active_edge_table(memory: ProfileCentricHypergraphMemory) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for edge_id, edge in memory.edges.items():
        if edge.status != "active":
            continue
        meta = edge.metadata or {}
        rows.append({
            "edge_id": edge_id,
            "memory_layer": "behavioral_profile_hyperedge",
            "edge_type": edge.edge_type.value,
            "feature_name": meta.get("feature_name", ""),
            "feature_type": meta.get("feature_type", ""),
            "description": meta.get("feature_description", ""),
            "num_facts": len(edge.member_fact_ids),
            "utility_score": round(edge.utility_score, 6),
            "stability_score": round(edge.stability_score, 6),
            "confidence_score": round(edge.confidence_score, 6),
            "positive_triggers": meta.get("positive_triggers", []),
            "negative_triggers": meta.get("negative_triggers", []),
        })
    return sorted(rows, key=lambda row: (row["utility_score"], row["num_facts"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-json", required=True, help="Raw conversation/memory JSON or JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-memory", type=int, default=0)
    parser.add_argument("--hierarchy-batch-size", type=int, default=40)
    parser.add_argument("--behavior-batch-size", type=int, default=50)
    parser.add_argument("--canonical-threshold", type=float, default=0.72)
    parser.add_argument("--consolidate-every", type=int, default=4)
    parser.add_argument("--llm-consolidation-rounds", type=int, default=0)
    parser.add_argument("--max-edge-facts", type=int, default=160)
    parser.add_argument("--max-features-per-batch", type=int, default=12)
    parser.add_argument("--max-features-per-fact", type=int, default=4)
    parser.add_argument("--utility-prior-strength", type=float, default=0.08)
    parser.add_argument("--pool-top-k", type=int, default=50)
    parser.add_argument("--no-llm-hierarchy", action="store_true", help="Use fallback hierarchy wrapper if input is already fact-like or for debugging.")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = read_json_or_jsonl(Path(args.memory_json))
    if args.max_memory:
        raw_rows = raw_rows[: args.max_memory]

    print("[pipeline] stage=1 llm_topic_episode_fact_extraction", flush=True)
    hierarchy = extract_topic_episode_fact_hierarchy(
        raw_rows,
        batch_size=args.hierarchy_batch_size,
        use_llm=not args.no_llm_hierarchy,
        show_progress=not args.no_progress,
    )
    hierarchy_facts = save_hierarchy_outputs(hierarchy, out_dir)
    # Use a flattened copy with topic/episode context appended to metadata. The
    # text itself stays atomic so fact-level retrieval remains clean.
    behavior_rows = flatten_hierarchy_facts(hierarchy)

    print("[pipeline] stage=2 behavioral_hyperedge_induction_from_extracted_facts", flush=True)
    memory = ProfileCentricHypergraphMemory(user_id="behavioral_hybrid_memory")
    build_english_llm_profile_hypergraph_from_rows(
        memory,
        behavior_rows,
        batch_size=args.behavior_batch_size,
        canonical_threshold=args.canonical_threshold,
        consolidate_every=args.consolidate_every,
        llm_consolidation_rounds=args.llm_consolidation_rounds,
        max_edge_facts=args.max_edge_facts,
        max_features_per_batch=args.max_features_per_batch,
        max_features_per_fact=args.max_features_per_fact,
        show_progress=not args.no_progress,
    )

    print("[pipeline] stage=3 reward_guided_utility_prior_from_hyperedge_features", flush=True)
    initialize_behavioral_priors(memory, prior_strength=args.utility_prior_strength)

    print("[pipeline] stage=4 embedding_and_index_materialization", flush=True)
    graph_path = out_dir / "behavioral_hybrid_graph.json"
    memory.save(graph_path)

    print("[pipeline] stage=5 export_behavioral_pool_and_reports", flush=True)
    candidate_pool = write_behavioral_pool(
        memory,
        out_dir / "candidate_behavioral_pool.json",
        top_k=args.pool_top_k,
        min_value=0.0,
        min_utility=0.0,
    )
    behavioral_summary = behavioral_profile_summary(memory, top_k=args.pool_top_k)
    (out_dir / "behavioral_profile_summary.json").write_text(json.dumps(behavioral_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "active_behavioral_edges.json").write_text(json.dumps(active_edge_table(memory), ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "pipeline_order": [
            "llm_extract_topic_episode_fact",
            "induce_behavioral_profile_hyperedges_using_hierarchy_facts",
            "initialize_reward_guided_utility_prior_from_hyperedge_features",
            "embed_and_index_facts_and_behavioral_hyperedges",
            "retrieve_by_behavioral_hyperedge_then_member_facts",
        ],
        "num_raw_rows": len(raw_rows),
        "num_hierarchy_facts": len(hierarchy_facts),
        "num_memory_facts": len(memory.facts),
        "num_edges": len(memory.edges),
        "active_edges": memory.active_edge_count(),
        "edge_type_counts": memory.export().get("edge_type_counts", {}),
        "multi_edge_stats": multi_edge_stats(memory),
        "candidate_behavioral_pool_size": len(candidate_pool),
        "outputs": {
            "topic_episode_fact_tree": str(out_dir / "topic_episode_fact_tree.json"),
            "hierarchical_facts": str(out_dir / "hierarchical_facts.jsonl"),
            "behavioral_hybrid_graph": str(graph_path),
            "candidate_behavioral_pool": str(out_dir / "candidate_behavioral_pool.json"),
        },
        "args": vars(args),
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("wrote:", out_dir)


if __name__ == "__main__":
    main()
