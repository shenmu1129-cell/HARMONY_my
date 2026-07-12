"""
Policy hyperedge lifecycle management.

Implements the four lifecycle operations:
- Spawn: Add validated candidate edges to the topology
- Merge: Combine overlapping policy edges
- Reweight: Update edge weights with new feedback
- Prune: Remove edges with low marginal contribution

Reference: Paper Sections 4.7.4 – 4.7.7
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .types import (
    ActionType,
    HypergraphState,
    MDPAction,
    MDPState,
    PCHConfig,
    PolicyHyperedge,
    PolicyEdgeStatus,
    RetrievalTrajectory,
)


def spawn_policy_edges(
    validated_edges: List[PolicyHyperedge],
    hypergraph: HypergraphState,
    config: PCHConfig,
    current_step: int = 0,
) -> List[str]:
    """Spawn validated candidate edges into the topology.

    A candidate edge is spawned when:
    1. |C_val| >= n_min
    2. LCB_{1-delta}(A_hat) > tau_A
    3. Consistency > tau_C

    Returns list of spawned edge IDs.
    """
    spawned_ids: List[str] = []

    for edge in validated_edges:
        if edge.edge_id in hypergraph.policy_edges:
            # Edge already exists, update it
            existing = hypergraph.policy_edges[edge.edge_id]
            existing.advantage_mean = edge.advantage_mean
            existing.advantage_std = edge.advantage_std
            existing.advantage_lcb = edge.advantage_lcb
            existing.validation_query_count = edge.validation_query_count
            existing.validation_consistency = edge.validation_consistency
            if existing.status != PolicyEdgeStatus.ACTIVE:
                existing.status = PolicyEdgeStatus.ACTIVE
                spawned_ids.append(edge.edge_id)
            continue

        # Check capacity
        active_count = hypergraph.num_active_policy_edges()
        if active_count >= config.max_policy_edges:
            # Try to prune worst edge first
            _prune_worst_edge(hypergraph, config)

        if hypergraph.num_active_policy_edges() >= config.max_policy_edges:
            continue

        # Spawn the edge
        edge.status = PolicyEdgeStatus.ACTIVE
        edge.created_at_step = current_step
        hypergraph.policy_edges[edge.edge_id] = edge
        hypergraph.total_policy_edges_spawned += 1
        spawned_ids.append(edge.edge_id)

    return spawned_ids


def reweight_policy_edges(
    hypergraph: HypergraphState,
    new_trajectories: List[RetrievalTrajectory],
    config: PCHConfig,
    rho: float = 0.3,  # EMA update rate
) -> None:
    """Reweight policy edges based on new trajectory feedback.

    Updates: advantage estimate, success rate, evidence gain,
    cost saved, frequency, freshness, uncertainty.

    Uses exponential moving average:
    w^{(k+1)}(e_p) <- (1-rho) * w^{(k)}(e_p) + rho * w_hat_new(e_p)
    """
    for edge_id, edge in hypergraph.policy_edges.items():
        if edge.status != PolicyEdgeStatus.ACTIVE:
            continue

        # Count how many times this edge was used in new trajectories
        usage_count = 0
        total_return_with_edge = 0.0
        total_return_without_edge = 0.0

        for traj in new_trajectories:
            used = False
            for step in traj.steps:
                if step.action.edge_id == edge_id:
                    used = True
                    break
            if used:
                usage_count += 1
                total_return_with_edge += traj.total_return
            else:
                total_return_without_edge += traj.total_return

        n = max(1, len(new_trajectories))

        # New estimates
        new_success = usage_count / n
        new_gain = (
            (total_return_with_edge / max(1, usage_count))
            if usage_count > 0
            else 0.0
        )
        new_freshness = 1.0  # Reset on access

        # EMA update
        edge.success_rate = (1 - rho) * edge.success_rate + rho * new_success
        edge.avg_evidence_gain = (1 - rho) * edge.avg_evidence_gain + rho * new_gain
        edge.access_frequency += usage_count
        edge.freshness_score = (1 - rho) * edge.freshness_score + rho * new_freshness
        edge.uncertainty = (1 - rho) * edge.uncertainty + rho * (1.0 - new_success)


def prune_policy_edges(
    hypergraph: HypergraphState,
    trajectories: List[RetrievalTrajectory],
    config: PCHConfig,
    prune_threshold: float = -0.05,
) -> List[str]:
    """Prune policy edges with low approximate marginal contribution.

    Approximate marginal contribution:
    Delta_hat_topo(e_p) = avg_{q in Q(e_p)} [U(q, e_p) - U(q, a_q^{(2)})]

    Where a_q^{(2)} is the second-best action if e_p is removed.
    Prunes edges whose contribution is below threshold or whose
    source facts have become invalid.
    """
    pruned_ids: List[str] = []

    for edge_id, edge in list(hypergraph.policy_edges.items()):
        if edge.status == PolicyEdgeStatus.PRUNED:
            continue

        should_prune = False

        # Check 1: Low marginal contribution
        if edge.success_rate < 0.1 and edge.access_frequency > 5:
            should_prune = True

        # Check 2: Very negative advantage (consistently worse than baseline)
        if edge.advantage_lcb < prune_threshold and edge.validation_query_count > 3:
            should_prune = True

        # Check 3: Stale edge (no access for long time, low freshness)
        if edge.freshness_score < 0.1 and edge.access_frequency > 0:
            should_prune = True

        # Check 4: Deprecated source facts
        valid_facts = sum(
            1 for f in edge.fact_ids
            if f in hypergraph.fact_contents
        )
        if valid_facts < len(edge.fact_ids) * 0.5:
            should_prune = True

        if should_prune:
            edge.status = PolicyEdgeStatus.PRUNED
            hypergraph.total_policy_edges_pruned += 1
            pruned_ids.append(edge_id)

    # Remove pruned edges from the dictionary
    for eid in pruned_ids:
        del hypergraph.policy_edges[eid]

    return pruned_ids


def merge_policy_edges(
    hypergraph: HypergraphState,
    config: PCHConfig,
) -> List[str]:
    """Merge highly overlapping policy edges.

    For pairs (e_i, e_j) with high intent and evidence overlap,
    construct merged candidate e_ij. Accept if:
    J(e_ij) > J({e_i, e_j}) + tau_merge
    """
    merged_ids: List[str] = []
    active_edges = [
        (eid, edge) for eid, edge in hypergraph.policy_edges.items()
        if edge.status == PolicyEdgeStatus.ACTIVE
    ]

    if len(active_edges) < 2:
        return merged_ids

    # Compute pairwise overlap
    merged_pairs: Set[Tuple[str, str]] = set()

    for i in range(len(active_edges)):
        for j in range(i + 1, len(active_edges)):
            eid_i, edge_i = active_edges[i]
            eid_j, edge_j = active_edges[j]

            # Compute overlap metrics
            fact_overlap = len(set(edge_i.fact_ids) & set(edge_j.fact_ids))
            fact_union = len(set(edge_i.fact_ids) | set(edge_j.fact_ids))
            fact_jaccard = fact_overlap / max(1, fact_union)

            episode_overlap = len(set(edge_i.episode_ids) & set(edge_j.episode_ids))
            episode_union = len(set(edge_i.episode_ids) | set(edge_j.episode_ids))
            episode_jaccard = episode_overlap / max(1, episode_union)

            # Merge if both overlaps are high
            if fact_jaccard > 0.5 and episode_jaccard > 0.5:
                merged_id = _create_merged_edge(
                    edge_i, edge_j, hypergraph, config,
                )
                if merged_id:
                    merged_ids.append(merged_id)
                    merged_pairs.add((eid_i, eid_j))

    # Deprecate original edges that were merged
    for eid_i, eid_j in merged_pairs:
        if eid_i in hypergraph.policy_edges:
            hypergraph.policy_edges[eid_i].status = PolicyEdgeStatus.MERGED
        if eid_j in hypergraph.policy_edges:
            hypergraph.policy_edges[eid_j].status = PolicyEdgeStatus.MERGED

    return merged_ids


def _create_merged_edge(
    edge_i: PolicyHyperedge,
    edge_j: PolicyHyperedge,
    hypergraph: HypergraphState,
    config: PCHConfig,
) -> Optional[str]:
    """Create a merged policy edge from two overlapping edges."""
    merged_id = f"pe_merged_{len(hypergraph.policy_edges):03d}"

    # Union of facts, attributes, episodes
    merged_facts = list(dict.fromkeys(edge_i.fact_ids + edge_j.fact_ids))
    merged_attrs = list(set(edge_i.attribute_ids) | set(edge_j.attribute_ids))
    merged_episodes = list(set(edge_i.episode_ids) | set(edge_j.episode_ids))

    # Combined intent
    merged_intent = f"{edge_i.intent_prototype} + {edge_j.intent_prototype}"

    # Average advantage
    merged_advantage = (edge_i.advantage_mean + edge_j.advantage_mean) / 2.0

    # Estimated utility comparison
    utility_separate = edge_i.advantage_mean + edge_j.advantage_mean
    utility_merged = merged_advantage  # Simplified: assume additive

    if utility_merged <= utility_separate + config.merge_margin:
        return None

    # Create merged embedding
    if edge_i.embedding is not None and edge_j.embedding is not None:
        merged_emb = (edge_i.embedding + edge_j.embedding) / 2.0
    else:
        merged_emb = None

    merged = PolicyHyperedge(
        edge_id=merged_id,
        status=PolicyEdgeStatus.ACTIVE,
        intent_prototype=merged_intent,
        attribute_ids=merged_attrs,
        episode_ids=merged_episodes,
        fact_ids=merged_facts,
        compressed_path=edge_i.compressed_path + edge_j.compressed_path,
        advantage_mean=merged_advantage,
        parent_edge_ids=[edge_i.edge_id, edge_j.edge_id],
        embedding=merged_emb,
    )

    hypergraph.policy_edges[merged_id] = merged
    return merged_id


def _prune_worst_edge(hypergraph: HypergraphState, config: PCHConfig) -> Optional[str]:
    """Find and prune the worst-performing active policy edge."""
    worst_id = None
    worst_score = float("inf")

    for eid, edge in hypergraph.policy_edges.items():
        if edge.status != PolicyEdgeStatus.ACTIVE:
            continue
        # Score: combination of low advantage and low usage
        score = edge.advantage_lcb - 0.1 * edge.access_frequency
        if score < worst_score:
            worst_score = score
            worst_id = eid

    if worst_id is not None:
        hypergraph.policy_edges[worst_id].status = PolicyEdgeStatus.PRUNED
        hypergraph.total_policy_edges_pruned += 1
        del hypergraph.policy_edges[worst_id]

    return worst_id
