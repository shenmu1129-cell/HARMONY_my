"""Build an English LLM-induced behavioral-profile hypergraph once and save it.

The graph stores open-set behavioral/profile hyperedges for recurring or stable
memory dimensions. Ordinary detail facts remain in the base fact memory and can
be served by the episodic/tree path in the full hybrid system.
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

from hypermem.behavioral_profile import behavioral_profile_summary, write_behavioral_pool  # noqa: E402
from hypermem.llm_profile_builder_en import build_english_llm_profile_hypergraph_from_rows  # noqa: E402
from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory  # noqa: E402


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        content = row.get("content") or row.get("text") or row.get("fact") or row.get("summary") or ""
        if not content:
            continue
        out.append(
            {
                "fact_id": row.get("fact_id") or row.get("id") or f"fact_{i+1:06d}",
                "content": str(content),
                "keywords": row.get("keywords") or [],
                "timestamp": row.get("timestamp") or row.get("time_index") or i + 1,
                "metadata": row,
            }
        )
    return out


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
        "sample_multi_edge_facts": dict(list(multi.items())[:20]),
    }


def edge_table(memory: ProfileCentricHypergraphMemory) -> List[Dict[str, Any]]:
    rows = []
    for edge_id, edge in memory.edges.items():
        if edge.status != "active":
            continue
        meta = edge.metadata or {}
        rows.append(
            {
                "edge_id": edge_id,
                "edge_type": edge.edge_type.value,
                "feature_name": meta.get("feature_name", ""),
                "feature_type": meta.get("feature_type", ""),
                "num_facts": len(edge.member_fact_ids),
                "description": meta.get("feature_description", ""),
                "memory_layer": "behavioral_profile_candidate",
            }
        )
    return sorted(rows, key=lambda row: row["num_facts"], reverse=True)


def all_edge_table(memory: ProfileCentricHypergraphMemory) -> List[Dict[str, Any]]:
    rows = []
    for edge_id, edge in memory.edges.items():
        meta = edge.metadata or {}
        rows.append(
            {
                "edge_id": edge_id,
                "status": edge.status,
                "merged_into": meta.get("merged_into", ""),
                "edge_type": edge.edge_type.value,
                "feature_name": meta.get("feature_name", ""),
                "feature_type": meta.get("feature_type", ""),
                "num_facts": len(edge.member_fact_ids),
                "description": meta.get("feature_description", ""),
                "memory_layer": "behavioral_profile_candidate",
            }
        )
    return sorted(rows, key=lambda row: (row["status"] != "active", -row["num_facts"], row["edge_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-memory", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--canonical-threshold", type=float, default=0.72)
    parser.add_argument("--consolidate-every", type=int, default=4)
    parser.add_argument("--llm-consolidation-rounds", type=int, default=0)
    parser.add_argument("--max-edge-facts", type=int, default=160)
    parser.add_argument("--max-features-per-batch", type=int, default=12)
    parser.add_argument("--max-features-per-fact", type=int, default=4)
    parser.add_argument("--pool-top-k", type=int, default=50)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = normalize_rows(read_jsonl(Path(args.memory_json)))
    if args.max_memory:
        rows = rows[: args.max_memory]

    memory = ProfileCentricHypergraphMemory(user_id="behavioral_profile_graph")
    build_english_llm_profile_hypergraph_from_rows(
        memory,
        rows,
        batch_size=args.batch_size,
        canonical_threshold=args.canonical_threshold,
        consolidate_every=args.consolidate_every,
        llm_consolidation_rounds=args.llm_consolidation_rounds,
        max_edge_facts=args.max_edge_facts,
        max_features_per_batch=args.max_features_per_batch,
        max_features_per_fact=args.max_features_per_fact,
        show_progress=not args.no_progress,
    )

    graph_path = out_dir / "profile_graph.json"
    report_path = out_dir / "build_report.json"
    edges_path = out_dir / "active_edges.json"
    all_edges_path = out_dir / "all_edges.json"
    candidate_pool_path = out_dir / "candidate_behavioral_pool.json"
    behavioral_summary_path = out_dir / "behavioral_profile_summary.json"

    memory.save(graph_path)
    candidate_pool = write_behavioral_pool(memory, candidate_pool_path, top_k=args.pool_top_k, min_value=0.0, min_utility=0.0)
    behavioral_summary = behavioral_profile_summary(memory, top_k=args.pool_top_k)
    behavioral_summary_path.write_text(json.dumps(behavioral_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "memory_design": "reward_guided_behavioral_profile_plus_episodic_tree",
        "design_note": "Behavioral/profile hyperedges store recurring or stable high-value dimensions; ordinary detail facts remain in the base fact/tree path.",
        "num_input_rows": len(rows),
        "num_facts": len(memory.facts),
        "num_edges": len(memory.edges),
        "active_edges": memory.active_edge_count(),
        "edge_type_counts": memory.export().get("edge_type_counts", {}),
        "multi_edge_stats": multi_edge_stats(memory),
        "candidate_behavioral_pool_size": len(candidate_pool),
        "build_args": vars(args),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    edges_path.write_text(json.dumps(edge_table(memory), ensure_ascii=False, indent=2), encoding="utf-8")
    all_edges_path.write_text(json.dumps(all_edge_table(memory), ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("wrote:", graph_path)
    print("wrote:", report_path)
    print("wrote:", edges_path)
    print("wrote:", all_edges_path)
    print("wrote:", candidate_pool_path)
    print("wrote:", behavioral_summary_path)


if __name__ == "__main__":
    main()
