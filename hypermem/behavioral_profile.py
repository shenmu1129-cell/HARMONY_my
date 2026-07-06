"""Reward-guided behavioral-profile memory utilities.

This module implements the scheme used by the third design direction:

1. keep ordinary episodic/detail facts in the base memory graph/tree;
2. induce open-set behavioral profile hyperedges for stable or recurring user patterns;
3. use QA/retrieval reward as a lightweight contextual-bandit signal;
4. promote repeatedly useful hyperedges into a high-value behavioral memory pool.

The functions here intentionally work on the existing ProfileCentricHypergraphMemory
objects so older profile-centric experiments remain backward compatible.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory, ProfileHyperedgeUnit, ProfileFact, estimate_tokens


def _feature_name(edge: ProfileHyperedgeUnit) -> str:
    return str((edge.metadata or {}).get("feature_name") or edge.summary[:80])


def _feature_type(edge: ProfileHyperedgeUnit) -> str:
    return str((edge.metadata or {}).get("feature_type") or edge.edge_type.value)


def _feature_description(edge: ProfileHyperedgeUnit) -> str:
    return str((edge.metadata or {}).get("feature_description") or edge.summary)


def behavioral_value(
    edge: ProfileHyperedgeUnit,
    facts: Dict[str, ProfileFact],
    *,
    max_access_count: Optional[int] = None,
    max_member_count: Optional[int] = None,
) -> float:
    """Compute a reward-guided behavioral value for a hyperedge.

    The value is deliberately not just semantic similarity. It favors hyperedges
    that are stable, coherent, repeatedly selected by queries, and rewarded by QA
    feedback, while mildly penalizing overly large/token-heavy edges.
    """
    max_access = max(1, int(max_access_count or edge.access_count or 1))
    max_members = max(1, int(max_member_count or len(edge.member_fact_ids) or 1))
    access_norm = math.log1p(edge.access_count) / math.log1p(max_access)
    hit_rate = edge.hit_count / max(1, edge.hit_count + edge.failure_count)
    member_norm = math.log1p(len(edge.member_fact_ids)) / math.log1p(max_members)
    token_penalty = min(1.0, edge.token_cost(facts) / 1200.0)
    value = (
        0.34 * edge.utility_score
        + 0.16 * hit_rate
        + 0.14 * access_norm
        + 0.12 * edge.stability_score
        + 0.10 * edge.confidence_score
        + 0.08 * edge.coherence_score
        + 0.06 * member_norm
        - 0.06 * token_penalty
    )
    return max(0.0, min(1.0, value))


def edge_to_behavioral_record(
    edge: ProfileHyperedgeUnit,
    facts: Dict[str, ProfileFact],
    *,
    value: Optional[float] = None,
    max_access_count: Optional[int] = None,
    max_member_count: Optional[int] = None,
    include_members: bool = True,
    max_members_preview: int = 30,
) -> Dict[str, Any]:
    if value is None:
        value = behavioral_value(edge, facts, max_access_count=max_access_count, max_member_count=max_member_count)
    member_facts = [facts[fid].content for fid in edge.member_fact_ids[:max_members_preview] if fid in facts]
    record: Dict[str, Any] = {
        "edge_id": edge.edge_id,
        "status": edge.status,
        "memory_layer": "behavioral_profile_pool",
        "edge_type": edge.edge_type.value,
        "feature_name": _feature_name(edge),
        "feature_type": _feature_type(edge),
        "description": _feature_description(edge),
        "num_facts": len(edge.member_fact_ids),
        "token_cost": edge.token_cost(facts),
        "behavioral_value": round(value, 6),
        "utility_score": round(edge.utility_score, 6),
        "stability_score": round(edge.stability_score, 6),
        "confidence_score": round(edge.confidence_score, 6),
        "coherence_score": round(edge.coherence_score, 6),
        "access_count": edge.access_count,
        "hit_count": edge.hit_count,
        "failure_count": edge.failure_count,
        "total_reward": round(edge.total_reward, 6),
        "last_reward": round(edge.last_reward, 6),
        "positive_triggers": (edge.metadata or {}).get("positive_triggers", []),
        "negative_triggers": (edge.metadata or {}).get("negative_triggers", []),
    }
    if include_members:
        record["member_fact_ids"] = edge.member_fact_ids[:max_members_preview]
        record["member_facts_preview"] = member_facts
    return record


def high_value_behavioral_pool(
    memory: ProfileCentricHypergraphMemory,
    *,
    top_k: int = 50,
    min_value: float = 0.50,
    min_utility: float = 0.50,
    min_facts: int = 2,
    require_positive_feedback: bool = False,
) -> List[Dict[str, Any]]:
    """Return high-value behavioral-profile hyperedges after reward updates.

    During a cold build, utility is usually near 0.5 and hit_count is 0, so this
    function behaves like a ranked structural pool. After training/evaluation
    feedback, it becomes a reward-guided high-value memory pool.
    """
    active = [edge for edge in memory.edges.values() if edge.status == "active"]
    max_access = max([edge.access_count for edge in active] or [1])
    max_members = max([len(edge.member_fact_ids) for edge in active] or [1])
    records: List[Dict[str, Any]] = []
    for edge in active:
        if len(edge.member_fact_ids) < min_facts:
            continue
        if require_positive_feedback and edge.hit_count <= 0:
            continue
        value = behavioral_value(edge, memory.facts, max_access_count=max_access, max_member_count=max_members)
        if value < min_value and edge.utility_score < min_utility:
            continue
        records.append(
            edge_to_behavioral_record(
                edge,
                memory.facts,
                value=value,
                max_access_count=max_access,
                max_member_count=max_members,
            )
        )
    records.sort(
        key=lambda row: (
            row["behavioral_value"],
            row["utility_score"],
            row["hit_count"],
            row["access_count"],
            row["num_facts"],
        ),
        reverse=True,
    )
    return records[:top_k]


def behavioral_profile_summary(memory: ProfileCentricHypergraphMemory, *, top_k: int = 20) -> Dict[str, Any]:
    active = [edge for edge in memory.edges.values() if edge.status == "active"]
    pool = high_value_behavioral_pool(memory, top_k=top_k, min_value=0.0, min_utility=0.0, min_facts=1)
    num_hits = sum(edge.hit_count for edge in active)
    num_failures = sum(edge.failure_count for edge in active)
    num_access = sum(edge.access_count for edge in active)
    return {
        "memory_design": "reward_guided_behavioral_profile_plus_episodic_tree",
        "behavioral_profile_definition": (
            "Open-set hyperedges store recurring, stable, or repeatedly useful user behavior/profile patterns. "
            "Ordinary episodic details remain available as fact-level memory and can be served by the base tree/RAG path."
        ),
        "num_facts": len(memory.facts),
        "num_edges": len(memory.edges),
        "active_edges": len(active),
        "total_edge_access_count": num_access,
        "total_edge_hits": num_hits,
        "total_edge_failures": num_failures,
        "avg_utility": round(sum(edge.utility_score for edge in active) / max(1, len(active)), 6),
        "avg_behavioral_value": round(sum(row["behavioral_value"] for row in pool) / max(1, len(pool)), 6),
        "top_behavioral_edges": pool[:top_k],
    }


def write_behavioral_pool(
    memory: ProfileCentricHypergraphMemory,
    output_path: str | Path,
    *,
    top_k: int = 50,
    min_value: float = 0.50,
    min_utility: float = 0.50,
    require_positive_feedback: bool = False,
) -> List[Dict[str, Any]]:
    pool = high_value_behavioral_pool(
        memory,
        top_k=top_k,
        min_value=min_value,
        min_utility=min_utility,
        require_positive_feedback=require_positive_feedback,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
    return pool
