"""
Held-out counterfactual advantage estimation.

Splits supporting queries into proposal and validation sets,
computes counterfactual advantage of candidate policy edges
relative to structural retrieval baseline.

Key implementation of paper Section 4.7:
- Proposal queries generate candidate edges
- Validation queries compute counterfactual advantage
- Lower Confidence Bound (LCB) determines Spawn eligibility
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .types import (
    ActionType,
    EdgeType,
    HypergraphState,
    MDPAction,
    MDPState,
    PCHConfig,
    PolicyHyperedge,
    PolicyEdgeStatus,
    RetrievalTrajectory,
)


def split_proposal_validation(
    trajectories: List[RetrievalTrajectory],
    config: PCHConfig,
) -> Tuple[List[RetrievalTrajectory], List[RetrievalTrajectory]]:
    """Split supporting trajectories into proposal and validation sets.

    Proposal set (70%): used to generate candidate edges.
    Validation set (30%): used to validate candidate edges.

    Returns (proposal_trajs, validation_trajs).
    """
    n = len(trajectories)
    if n < 2:
        return trajectories, []

    n_prop = max(1, int(n * config.proposal_ratio))
    n_val = n - n_prop

    # Ensure minimum validation size
    if n_val < config.min_validation_queries and n > config.min_validation_queries:
        n_val = config.min_validation_queries
        n_prop = n - n_val

    # Shuffle deterministically by query_id
    indices = sorted(range(n), key=lambda i: hash(trajectories[i].query_id))
    prop_indices = set(indices[:n_prop])
    val_indices = set(indices[n_prop:])

    prop_trajs = [t for i, t in enumerate(trajectories) if i in prop_indices]
    val_trajs = [t for i, t in enumerate(trajectories) if i in val_indices]

    return prop_trajs, val_trajs


def estimate_counterfactual_advantage(
    candidate_edge: PolicyHyperedge,
    validation_trajectories: List[RetrievalTrajectory],
    hypergraph: HypergraphState,
    config: PCHConfig,
) -> Tuple[float, float, float]:
    """Estimate counterfactual advantage on held-out validation queries.

    For each validation query:
    1. Simulate using the candidate policy edge
    2. Compare against the structural baseline (original trajectory)
    3. Compute advantage: U(q, policy_edge) - U(q, structural_baseline)

    Returns (mean_advantage, std_advantage, consistency).
    """
    if not validation_trajectories:
        return 0.0, 0.0, 0.0

    advantages: List[float] = []
    positive_count = 0

    for traj in validation_trajectories:
        # Structural baseline utility (from original trajectory)
        baseline_utility = _compute_trajectory_utility(traj, config)

        # Counterfactual utility: using policy edge + stop
        policy_utility = _simulate_policy_edge_utility(
            candidate_edge, traj, hypergraph, config,
        )

        advantage = policy_utility - baseline_utility
        advantages.append(advantage)
        if advantage > 0:
            positive_count += 1

    if not advantages:
        return 0.0, 0.0, 0.0

    mean_adv = float(np.mean(advantages))
    std_adv = float(np.std(advantages)) if len(advantages) > 1 else 0.1

    # Consistency: fraction of validation queries with positive advantage
    consistency = positive_count / len(advantages)

    return mean_adv, std_adv, consistency


def compute_advantage_lcb(
    mean_advantage: float,
    std_advantage: float,
    n_validation: int,
    config: PCHConfig,
) -> float:
    """Compute the Lower Confidence Bound of the advantage.

    LCB_{1-delta}(A) = mean_A - z_{1-delta} * std_A / sqrt(n)

    Where z_{1-delta} is the z-score for confidence level (1-delta).
    """
    if n_validation <= 1:
        return mean_advantage - std_advantage

    # Z-score for given confidence level
    # 90% -> 1.28, 95% -> 1.645, 99% -> 2.33
    confidence_to_z = {
        0.80: 0.84,
        0.85: 1.04,
        0.90: 1.28,
        0.95: 1.645,
        0.99: 2.33,
    }
    z = confidence_to_z.get(config.confidence_level, 1.28)

    standard_error = std_advantage / math.sqrt(n_validation)
    lcb = mean_advantage - z * standard_error

    return lcb


def _compute_trajectory_utility(
    trajectory: RetrievalTrajectory,
    config: PCHConfig,
) -> float:
    """Compute the total utility of a retrieval trajectory.

    U(q, tau) = sum of step rewards (discounted).
    """
    utility = 0.0
    for i, step in enumerate(trajectory.steps):
        utility += (config.gamma ** i) * step.reward
    return utility


def _simulate_policy_edge_utility(
    candidate_edge: PolicyHyperedge,
    trajectory: RetrievalTrajectory,
    hypergraph: HypergraphState,
    config: PCHConfig,
) -> float:
    """Simulate utility of using the candidate policy edge.

    Models the counterfactual: what if the query used the policy edge
    instead of the structural path?

    The virtual state transition T(s, F+(e_p)) is simulated by:
    1. Starting from initial state
    2. Executing SELECT_POLICY(candidate)
    3. Then STOP
    """
    if not trajectory.steps:
        return 0.0

    initial_state = trajectory.steps[0].state

    # Simulate policy edge retrieval
    edge_fact_ids = set(candidate_edge.fact_ids)
    initial_fact_ids = set(initial_state.evidence_fact_ids)
    new_facts = edge_fact_ids - initial_fact_ids
    dup_facts = edge_fact_ids & initial_fact_ids

    # Evidence gain
    evidence_gain = len(new_facts) / max(1, len(edge_fact_ids))
    duplicate_penalty = len(dup_facts) / max(1, len(edge_fact_ids))

    # Cost (policy edges are cheaper)
    visit_cost = 0.5
    latency_cost = 5.0
    token_cost = sum(
        len(hypergraph.fact_contents.get(f, "").split())
        for f in candidate_edge.fact_ids
    )

    # Virtual immediate reward
    virtual_reward = (
        evidence_gain * config.alpha_evidence
        - visit_cost * config.lambda_visit
        - latency_cost * config.lambda_latency
        - token_cost * config.lambda_token
        - duplicate_penalty * config.lambda_duplicate
    )

    # STOP reward after policy edge
    evidence_count = len(initial_fact_ids | edge_fact_ids)
    coverage_score = min(1.0, evidence_count / 5.0)
    stop_reward = coverage_score * config.alpha_evidence

    return virtual_reward + config.gamma * stop_reward


def validate_candidate_edges(
    candidate_edges: List[PolicyHyperedge],
    validation_trajectories: List[RetrievalTrajectory],
    hypergraph: HypergraphState,
    config: PCHConfig,
) -> List[PolicyHyperedge]:
    """Run full held-out validation for all candidate edges.

    Updates each candidate edge's advantage statistics in-place.
    Returns list of edges that pass the Spawn criteria.
    """
    validated: List[PolicyHyperedge] = []

    for edge in candidate_edges:
        mean_adv, std_adv, consistency = estimate_counterfactual_advantage(
            edge, validation_trajectories, hypergraph, config,
        )
        lcb = compute_advantage_lcb(mean_adv, std_adv, len(validation_trajectories), config)

        edge.advantage_mean = mean_adv
        edge.advantage_std = std_adv
        edge.advantage_lcb = lcb
        edge.validation_query_count = len(validation_trajectories)
        edge.validation_consistency = consistency

        # Check Spawn criteria
        if (
            edge.validation_query_count >= config.min_validation_queries
            and lcb > config.advantage_threshold
            and consistency > config.consistency_threshold
        ):
            validated.append(edge)

    return validated
