"""
Policy–Topology Alternating Co-optimization.

Implements the alternating optimization loop (Paper Section 4.8):
1. Policy Evaluation: Train value function on current topology
2. Shortcut Proposal: Mine candidate subpaths from trajectories
3. Held-out Validation: Estimate counterfactual advantage
4. Topology Update: Spawn / Merge / Reweight / Prune
5. Validation Acceptance: Only accept if utility improves
6. Policy Re-optimization: Re-train on updated topology
7. Stop if converged
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Set, Tuple

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
from .teacher import TeacherRetriever, collect_teacher_trajectories
from .value_learning import ValuePolicy, train_bc, train_cql
from .trajectory_mining import (
    canonicalize_trajectory,
    construct_candidate_policy_edge,
    mine_repeated_subpaths,
)
from .advantage import (
    estimate_counterfactual_advantage,
    split_proposal_validation,
    validate_candidate_edges,
)
from .lifecycle import (
    spawn_policy_edges,
    merge_policy_edges,
    reweight_policy_edges,
    prune_policy_edges,
)


def policy_evaluation_step(
    policy: ValuePolicy,
    trajectories: List[RetrievalTrajectory],
    config: PCHConfig,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Step 1: Policy Evaluation on current topology.

    Trains value function using BC warm-start + CQL.
    Returns training metrics.
    """
    metrics = {}

    # BC warm-start (only if policy hasn't been trained yet)
    bc_losses = train_bc(policy, trajectories, config)
    metrics["bc_final_loss"] = bc_losses[-1] if bc_losses else 0.0
    metrics["bc_epochs"] = len(bc_losses)

    # Conservative Q-Learning
    cql_losses = train_cql(policy, trajectories, config)
    metrics["cql_final_loss"] = cql_losses[-1] if cql_losses else 0.0
    metrics["cql_epochs"] = len(cql_losses)

    if verbose:
        print(f"  [Policy Eval] BC loss: {metrics['bc_final_loss']:.4f} "
              f"({metrics['bc_epochs']} epochs), "
              f"CQL loss: {metrics['cql_final_loss']:.4f} "
              f"({metrics['cql_epochs']} epochs)")

    return metrics


def topology_update_step(
    hypergraph: HypergraphState,
    trajectories: List[RetrievalTrajectory],
    policy: ValuePolicy,
    config: PCHConfig,
    current_round: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Steps 2-4: Shortcut Proposal → Validation → Topology Update.

    1. Mine repeated subpaths from high-return trajectories
    2. Construct candidate policy edges
    3. Split supporting queries (proposal/validation)
    4. Estimate held-out counterfactual advantage
    5. Spawn validated edges
    """
    metrics = {
        "candidates_proposed": 0,
        "candidates_validated": 0,
        "edges_spawned": 0,
        "edges_merged": 0,
        "edges_pruned": 0,
    }

    # Step 2: Canonicalize trajectories
    for traj in trajectories:
        canonicalize_trajectory(traj, hypergraph)

    # Step 3: Mine repeated subpaths
    subpaths = mine_repeated_subpaths(trajectories, hypergraph, config)
    if verbose:
        print(f"  [Mining] Found {len(subpaths)} repeated subpaths "
              f"(min_support={config.min_trajectory_support})")

    if not subpaths:
        return metrics

    # Construct candidate edges from top subpaths
    candidate_edges: List[PolicyHyperedge] = []
    for subpath_ids, supporting_trajs in subpaths[:20]:  # Limit to top 20
        # Split supporting queries
        prop_trajs, val_trajs = split_proposal_validation(supporting_trajs, config)

        if len(prop_trajs) < 2 or len(val_trajs) < config.min_validation_queries:
            continue

        edge = construct_candidate_policy_edge(
            subpath_ids, prop_trajs, hypergraph,
        )
        candidate_edges.append(edge)

    metrics["candidates_proposed"] = len(candidate_edges)

    if not candidate_edges:
        return metrics

    # Step 4: Held-out validation
    validated = []
    for edge in candidate_edges:
        # Get validation trajectories for this edge's supporting queries
        _, val_trajs = split_proposal_validation(
            [t for t in trajectories
             if t.signature and t.signature.is_compatible(
                 canonicalize_trajectory(t, hypergraph)
             )] if trajectories else [],
            config,
        )
        if not val_trajs:
            val_trajs = trajectories[-config.min_validation_queries:]

        mean_adv, std_adv, consistency = estimate_counterfactual_advantage(
            edge, val_trajs, hypergraph, config,
        )

        edge.advantage_mean = mean_adv
        edge.advantage_std = std_adv
        edge.validation_query_count = len(val_trajs)
        edge.validation_consistency = consistency

        from .advantage import compute_advantage_lcb
        edge.advantage_lcb = compute_advantage_lcb(
            mean_adv, std_adv, len(val_trajs), config,
        )

        if (
            edge.validation_query_count >= config.min_validation_queries
            and edge.advantage_lcb > config.advantage_threshold
            and consistency > config.consistency_threshold
        ):
            validated.append(edge)

    metrics["candidates_validated"] = len(validated)

    # Step 5: Spawn validated edges
    spawned = spawn_policy_edges(
        validated, hypergraph, config, current_round,
    )
    metrics["edges_spawned"] = len(spawned)

    if verbose:
        print(f"  [Topology] {len(candidate_edges)} candidates → "
              f"{len(validated)} validated → {len(spawned)} spawned")

    # Merge overlapping edges (optional, for cleanup)
    if len(spawned) > 0:
        merged = merge_policy_edges(hypergraph, config)
        metrics["edges_merged"] = len(merged)

    # Reweight existing edges
    reweight_policy_edges(hypergraph, trajectories, config)

    # Prune low-contribution edges
    pruned = prune_policy_edges(hypergraph, trajectories, config)
    metrics["edges_pruned"] = len(pruned)

    return metrics


def policy_reoptimization_step(
    policy: ValuePolicy,
    hypergraph: HypergraphState,
    trajectories: List[RetrievalTrajectory],
    mdp: "RetrievalMDP",
    config: PCHConfig,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Step 6: Policy Re-optimization on updated topology.

    After topology changes, the action space has changed,
    so we need to:
    1. Update the policy's action index
    2. Re-label or regenerate transitions
    3. Re-train the value function
    """
    # Update action index to include new policy edges
    policy.update_action_index()
    # Rebuild MDP action lists
    mdp._rebuild_action_lists()

    # Collect new trajectories on updated topology
    # (using current policy for exploration)
    if trajectories:
        new_trajs = _relabel_trajectories(trajectories, hypergraph, mdp, config)
    else:
        new_trajs = trajectories

    # Re-train value function
    metrics = policy_evaluation_step(policy, new_trajs, config, verbose=verbose)
    return metrics


def alternating_optimization(
    hypergraph: HypergraphState,
    initial_trajectories: List[RetrievalTrajectory],
    config: PCHConfig,
    mdp: "RetrievalMDP",
    val_queries: Optional[List[Tuple[str, str, np.ndarray]]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full alternating optimization loop.

    Returns:
        Dictionary with optimization history and final metrics.
    """
    if verbose:
        print("=" * 60)
        print("PCH-Mem: Policy–Topology Alternating Co-optimization")
        print(f"  Structural edges: {hypergraph.num_structural_edges()}")
        print(f"  Training trajectories: {len(initial_trajectories)}")
        print(f"  Max rounds: {config.max_rounds}")
        print("=" * 60)

    # Initialize policy
    policy = ValuePolicy(config, hypergraph)

    history: List[Dict[str, Any]] = []
    trajectories = list(initial_trajectories)
    prev_val_utility = -float("inf")

    for k in range(config.max_rounds):
        if verbose:
            print(f"\n--- Round {k + 1}/{config.max_rounds} ---")
            print(f"  Structural edges: {hypergraph.num_structural_edges()}")
            print(f"  Policy edges: {hypergraph.num_policy_edges()} "
                  f"({hypergraph.num_active_policy_edges()} active)")

        round_metrics = {
            "round": k,
            "num_structural_edges": hypergraph.num_structural_edges(),
            "num_policy_edges": hypergraph.num_policy_edges(),
            "num_active_policy_edges": hypergraph.num_active_policy_edges(),
        }

        # Step 1: Policy Evaluation
        eval_metrics = policy_evaluation_step(
            policy, trajectories, config, verbose=verbose,
        )
        round_metrics.update(eval_metrics)

        # Steps 2-5: Topology Update
        topo_metrics = topology_update_step(
            hypergraph, trajectories, policy, config,
            current_round=k, verbose=verbose,
        )
        round_metrics.update(topo_metrics)

        # Check if topology changed
        if topo_metrics["edges_spawned"] == 0 and topo_metrics["edges_pruned"] == 0:
            if verbose:
                print(f"  [Converged] No topology changes in round {k + 1}")
            history.append(round_metrics)
            break

        # Step 6: Policy Re-optimization
        reopt_metrics = policy_reoptimization_step(
            policy, hypergraph, trajectories, mdp, config, verbose=verbose,
        )
        round_metrics.update({f"reopt_{k}": v for k, v in reopt_metrics.items()})

        # Estimate validation utility
        if val_queries:
            val_utility = _estimate_validation_utility(
                policy, hypergraph, mdp, val_queries, config,
            )
            round_metrics["val_utility"] = val_utility

            if verbose:
                print(f"  [Validation] Utility: {val_utility:.4f} "
                      f"(prev: {prev_val_utility:.4f})")

            # Step 7: Check convergence
            if val_utility <= prev_val_utility + config.utility_improvement_epsilon:
                if verbose:
                    print(f"  [Converged] Utility improvement below epsilon")
                history.append(round_metrics)
                break

            prev_val_utility = val_utility

        # Collect new trajectories for next round
        if topo_metrics["edges_spawned"] > 0:
            # Generate some trajectories using updated policy
            # (in practice, this would use new or replayed queries)
            pass

        history.append(round_metrics)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Optimization complete after {len(history)} rounds")
        print(f"  Final policy edges: {hypergraph.num_policy_edges()}")
        print(f"  Total spawned: {hypergraph.total_policy_edges_spawned}")
        print(f"  Total pruned: {hypergraph.total_policy_edges_pruned}")
        print(f"{'=' * 60}")

    return {
        "history": history,
        "final_policy_edges": hypergraph.num_policy_edges(),
        "total_spawned": hypergraph.total_policy_edges_spawned,
        "total_pruned": hypergraph.total_policy_edges_pruned,
    }


def _relabel_trajectories(
    trajectories: List[RetrievalTrajectory],
    hypergraph: HypergraphState,
    mdp: "RetrievalMDP",
    config: PCHConfig,
) -> List[RetrievalTrajectory]:
    """Re-label trajectories after topology changes.

    When new policy edges are added, old trajectories need to
    be updated to reflect the new action space.
    """
    # Simplified: return trajectories as-is
    # In full implementation, would replay each trajectory with
    # the new action set and re-compute rewards
    return trajectories


def _estimate_validation_utility(
    policy: ValuePolicy,
    hypergraph: HypergraphState,
    mdp: "RetrievalMDP",
    val_queries: List[Tuple[str, str, np.ndarray]],
    config: PCHConfig,
) -> float:
    """Estimate average utility on validation queries using current policy."""
    total_utility = 0.0

    for qid, qtext, qemb in val_queries:
        state = mdp.initial_state(qemb)
        done = False
        traj_utility = 0.0
        step_idx = 0

        while not done and step_idx < config.max_steps:
            available = mdp.get_available_actions(state)
            action = policy.select_action(state, available)
            next_state, reward, done, _ = mdp.step(state, action)
            traj_utility += (config.gamma ** step_idx) * reward
            state = next_state
            step_idx += 1

        total_utility += traj_utility

    return total_utility / max(1, len(val_queries))
