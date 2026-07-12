"""
Trajectory canonicalization and repeated subpath mining.

Converts high-return retrieval trajectories into structural signatures,
clusters compatible trajectories, and mines repeated subpaths
that can be compressed into candidate policy hyperedges.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

from .types import (
    ActionType,
    HypergraphState,
    PCHConfig,
    PolicyHyperedge,
    PolicyEdgeStatus,
    RetrievalTrajectory,
    StructuralHyperedge,
    TrajectorySignature,
    TrajectoryStep,
)


def canonicalize_trajectory(
    trajectory: RetrievalTrajectory,
    hypergraph: HypergraphState,
) -> TrajectorySignature:
    """Convert a trajectory into a canonical structural signature.

    Sig(tau) = (I(q), R(q), T(q), TypeSeq(tau))

    - I(q): Query intent prototype (simplified: first few words or keyword cluster)
    - R(q): Role/entity constraint (extracted from visited edges)
    - T(q): Temporal relation type
    - TypeSeq: Sequence of structural edge types visited
    """
    # Extract intent prototype from query
    query_words = trajectory.query_text.lower().split()
    # Simple heuristic: use wh-words or first meaningful words
    wh_words = {"what", "when", "where", "who", "why", "how", "which"}
    intent_words = [w for w in query_words if w in wh_words]
    if intent_words:
        intent = intent_words[0]
    elif len(query_words) > 0:
        intent = query_words[0]
    else:
        intent = "unknown"

    # Extract role constraint from visited edges
    role_parts: Set[str] = set()
    for step in trajectory.steps:
        if step.action.action_type == ActionType.SELECT_STRUCT:
            edge = hypergraph.structural_edges.get(step.action.edge_id)
            if edge and edge.role_constraints:
                role_parts.update(edge.role_constraints.values())
    role_constraint = "|".join(sorted(role_parts)) if role_parts else "any"

    # Extract temporal relation
    temporal_words = {"before", "after", "during", "when", "change", "evolve", "recent"}
    temporal_hits = [w for w in query_words if w in temporal_words]
    temporal_relation = temporal_hits[0] if temporal_hits else "none"

    # Extract edge type sequence
    edge_types: List[str] = []
    for step in trajectory.steps:
        if step.action.action_type == ActionType.SELECT_STRUCT:
            edge = hypergraph.structural_edges.get(step.action.edge_id)
            if edge:
                # Type based on member count and constraints
                if edge.role_constraints:
                    edge_types.append("constrained")
                elif len(edge.fact_ids) <= 3:
                    edge_types.append("small")
                else:
                    edge_types.append("large")
            else:
                edge_types.append("unknown")
        elif step.action.action_type == ActionType.SELECT_POLICY:
            edge_types.append("policy")
        elif step.action.action_type == ActionType.STOP:
            edge_types.append("stop")
        elif step.action.action_type == ActionType.FALLBACK:
            edge_types.append("fallback")

    signature = TrajectorySignature(
        intent_prototype=intent,
        role_constraint=role_constraint,
        temporal_relation=temporal_relation,
        edge_type_sequence=tuple(edge_types),
        path_length=len(edge_types),
    )
    trajectory.signature = signature
    return signature


def mine_repeated_subpaths(
    trajectories: List[RetrievalTrajectory],
    hypergraph: HypergraphState,
    config: PCHConfig,
) -> List[Tuple[List[str], List[RetrievalTrajectory]]]:
    """Mine repeated structural subpaths from trajectories.

    Finds subpaths of length >= min_subpath_length that appear
    in at least min_trajectory_support trajectories within
    compatible signature clusters.

    Returns:
        List of (subpath_edge_ids, supporting_trajectories) tuples.
    """
    # Group trajectories by compatible signatures
    clusters: Dict[str, List[RetrievalTrajectory]] = defaultdict(list)
    for traj in trajectories:
        if traj.signature is None:
            canonicalize_trajectory(traj, hypergraph)
        sig = traj.signature
        if sig is not None:
            # Use intent + role as cluster key (ignore temporal for clustering)
            cluster_key = f"{sig.intent_prototype}|{sig.role_constraint}"
            clusters[cluster_key].append(traj)

    # Mine subpaths within each cluster
    all_subpaths: List[Tuple[List[str], List[RetrievalTrajectory]]] = []

    for cluster_key, cluster_trajs in clusters.items():
        if len(cluster_trajs) < config.signature_cluster_min_size:
            continue

        # Extract structural edge sequences from each trajectory
        edge_sequences: List[List[str]] = []
        for traj in cluster_trajs:
            seq = []
            for step in traj.steps:
                if step.action.action_type == ActionType.SELECT_STRUCT:
                    seq.append(step.action.edge_id)
            if len(seq) >= config.min_subpath_length:
                edge_sequences.append(seq)

        # Find all subpaths of length >= min_subpath_length
        subpath_counts: Dict[Tuple[str, ...], List[RetrievalTrajectory]] = defaultdict(list)

        for traj_idx, seq in enumerate(edge_sequences):
            traj = cluster_trajs[traj_idx]
            for start in range(len(seq)):
                for end in range(start + config.min_subpath_length, len(seq) + 1):
                    subpath = tuple(seq[start:end])
                    if traj not in subpath_counts[subpath]:
                        subpath_counts[subpath].append(traj)

        # Filter by support
        for subpath, supporting_trajs in subpath_counts.items():
            if len(supporting_trajs) >= config.min_trajectory_support:
                all_subpaths.append((list(subpath), supporting_trajs))

    # Sort by support count (descending)
    all_subpaths.sort(key=lambda x: -len(x[1]))
    return all_subpaths


def construct_candidate_policy_edge(
    subpath_edge_ids: List[str],
    supporting_trajectories: List[RetrievalTrajectory],
    hypergraph: HypergraphState,
    edge_id_prefix: str = "pe",
) -> PolicyHyperedge:
    """Construct a candidate policy hyperedge from a repeated subpath.

    Aggregates facts from all structural edges in the subpath,
    computes initial advantage statistics from supporting trajectories.
    """
    # Collect all facts and attributes from subpath edges
    all_fact_ids: List[str] = []
    all_attribute_ids: List[str] = []
    all_episode_ids: List[str] = []
    intent_words: Set[str] = set()

    for eid in subpath_edge_ids:
        edge = hypergraph.structural_edges.get(eid)
        if edge is None:
            continue
        all_fact_ids.extend(edge.fact_ids)
        all_attribute_ids.extend(edge.attribute_ids)
        all_episode_ids.extend(edge.episode_ids)

    # Deduplicate while preserving order
    seen_f = set()
    unique_facts = []
    for f in all_fact_ids:
        if f not in seen_f:
            seen_f.add(f)
            unique_facts.append(f)

    # Extract intent prototype from supporting queries
    for traj in supporting_trajectories:
        for word in traj.query_text.lower().split():
            if len(word) > 3:
                intent_words.add(word)
    intent_prototype = " ".join(sorted(intent_words)[:5])

    # LOO: estimate positive-contribution facts
    positive_facts = _filter_positive_facts(
        unique_facts, supporting_trajectories, hypergraph,
    )

    # Generate edge ID
    edge_id = f"{edge_id_prefix}_{len(hypergraph.policy_edges):03d}"

    # Compute initial embedding as mean of member fact embeddings
    fact_embs = [
        hypergraph.fact_embeddings.get(f, np.zeros(256))
        for f in positive_facts
    ]
    embedding = np.mean(fact_embs, axis=0) if fact_embs else np.zeros(256)

    edge = PolicyHyperedge(
        edge_id=edge_id,
        status=PolicyEdgeStatus.CANDIDATE,
        intent_prototype=intent_prototype,
        attribute_ids=list(set(all_attribute_ids)),
        episode_ids=list(set(all_episode_ids)),
        fact_ids=positive_facts,
        structural_edge_ids=list(subpath_edge_ids),
        compressed_path=list(subpath_edge_ids),
        embedding=embedding,
        created_at_step=0,
    )

    return edge


def _filter_positive_facts(
    fact_ids: List[str],
    trajectories: List[RetrievalTrajectory],
    hypergraph: HypergraphState,
    min_contribution: float = 0.05,
) -> List[str]:
    """Leave-one-out fact contribution estimation.

    Keeps only facts with average positive contribution.
    For training, uses evidence gain as proxy for contribution.
    """
    if len(trajectories) < 2:
        return fact_ids

    contributions: Dict[str, List[float]] = defaultdict(list)

    for traj in trajectories:
        retrieved_facts: Set[str] = set()
        for step in traj.steps:
            if step.action.action_type in (ActionType.SELECT_STRUCT, ActionType.SELECT_POLICY):
                edge = (
                    hypergraph.structural_edges.get(step.action.edge_id)
                    or hypergraph.policy_edges.get(step.action.edge_id)
                )
                if edge is None:
                    continue
                retrieved_facts.update(
                    edge.fact_ids if hasattr(edge, 'fact_ids') else []
                )

        # For each fact, estimate contribution as:
        # whether the fact was retrieved in high-return trajectories
        for fid in fact_ids:
            if fid in retrieved_facts and traj.total_return > 0:
                contributions[fid].append(1.0)
            elif fid in retrieved_facts:
                contributions[fid].append(0.3)
            else:
                contributions[fid].append(0.0)

    # Keep facts with above-threshold average contribution
    positive = []
    for fid in fact_ids:
        contribs = contributions.get(fid, [0.0])
        avg_contrib = sum(contribs) / max(1, len(contribs))
        if avg_contrib > min_contribution:
            positive.append(fid)

    return positive
