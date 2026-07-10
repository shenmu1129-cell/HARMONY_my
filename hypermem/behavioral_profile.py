"""Reward-guided behavioral-profile memory utilities.

This module implements the third design direction:

1. ordinary episodic/detail facts stay in the base Topic-Episode-Fact tree;
2. open-set behavioral profile hyperedges are induced after hierarchy extraction;
3. QA/retrieval reward acts as a lightweight contextual-bandit signal;
4. repeatedly useful hyperedges are promoted into a high-value behavioral pool.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory, ProfileHyperedgeUnit, ProfileFact, clamp


def _feature_name(edge: ProfileHyperedgeUnit) -> str:
    return str((edge.metadata or {}).get("feature_name") or edge.summary[:80])


def _feature_type(edge: ProfileHyperedgeUnit) -> str:
    return str((edge.metadata or {}).get("feature_type") or edge.edge_type.value)


def _feature_description(edge: ProfileHyperedgeUnit) -> str:
    return str((edge.metadata or {}).get("feature_description") or edge.summary)


def update_behavioral_edge_utility(
    edge: ProfileHyperedgeUnit,
    *,
    reward: float,
    hit: bool,
    lr: float = 0.18,
    route: str = "behavioral",
    allow_deactivate: bool = False,
) -> None:
    """Bandit-style utility update for behavioral hyperedges.

    This is intentionally safer than ProfileHyperedgeUnit.update_utility for the
    hybrid setting. Episodic/detail queries should not deactivate behavioral
    edges; they either skip updates or only provide weak negative evidence.
    """
    reward = max(-1.0, min(1.0, reward))
    target = clamp((reward + 1.0) / 2.0)
    edge.utility_score = clamp((1.0 - lr) * edge.utility_score + lr * target)
    edge.total_reward += reward
    edge.last_reward = reward
    edge.access_count += 1
    if hit:
        edge.hit_count += 1
        edge.stability_score = clamp(edge.stability_score + lr * 0.10)
        edge.confidence_score = clamp(edge.confidence_score + lr * 0.08)
    else:
        edge.failure_count += 1
        # Mixed queries can be noisy; use a smaller penalty than pure behavioral failures.
        penalty_scale = 0.04 if route == "mixed" else 0.06
        edge.stability_score = clamp(edge.stability_score - lr * penalty_scale)
        edge.confidence_score = clamp(edge.confidence_score - lr * 0.04)
    if allow_deactivate and edge.failure_count >= 40 and edge.hit_count == 0 and edge.access_count >= 40:
        edge.status = "inactive"
    edge.updated_at = time.time()


def update_behavioral_edges_from_feedback(
    memory: ProfileCentricHypergraphMemory,
    selected_edges: List[ProfileHyperedgeUnit],
    *,
    reward: float,
    hit: bool,
    route: str,
    learning_rate: float = 0.18,
    allow_deactivate: bool = False,
) -> None:
    """Update selected behavioral edges only for behavioral or mixed queries."""
    if route not in {"behavioral", "mixed"}:
        return
    for edge in selected_edges:
        if edge.edge_id not in memory.edges:
            continue
        update_behavioral_edge_utility(
            edge,
            reward=reward,
            hit=hit,
            lr=learning_rate,
            route=route,
            allow_deactivate=allow_deactivate,
        )


def initialize_behavioral_priors(memory: ProfileCentricHypergraphMemory, *, prior_strength: float = 0.08) -> None:
    """Initialize cold-start utility from structural hyperedge features.

    This is not supervised training. It gives a small prior to coherent and
    evidence-supported hyperedges before QA reward is available.
    """
    active = [edge for edge in memory.edges.values() if edge.status == "active"]
    max_members = max([len(edge.member_fact_ids) for edge in active] or [1])
    for edge in active:
        support = math.log1p(len(edge.member_fact_ids)) / math.log1p(max_members)
        token_penalty = min(1.0, edge.token_cost(memory.facts) / 1200.0)
        prior = 0.50 + prior_strength * (0.45 * support + 0.35 * edge.confidence_score + 0.20 * edge.coherence_score - 0.20 * token_penalty)
        edge.utility_score = clamp(prior, lo=0.35, hi=0.75)
        edge.metadata.setdefault("utility_prior", {})
        edge.metadata["utility_prior"] = {
            "support": round(support, 6),
            "token_penalty": round(token_penalty, 6),
            "prior_strength": prior_strength,
            "initialized_utility": round(edge.utility_score, 6),
        }


def behavioral_value(
    edge: ProfileHyperedgeUnit,
    facts: Dict[str, ProfileFact],
    *,
    max_access_count: Optional[int] = None,
    max_member_count: Optional[int] = None,
) -> float:
    """Compute a reward-guided behavioral value for a hyperedge."""
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
        "memory_design": "llm_hierarchy_then_reward_guided_behavioral_profile_then_embedding_then_hyperedge_retrieval",
        "behavioral_profile_definition": (
            "Open-set behavioral hyperedges are induced after Topic-Episode-Fact extraction. "
            "Their utility is learned from behavioral/mixed query feedback, and retrieval first selects behavioral hyperedges before selecting member facts."
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
