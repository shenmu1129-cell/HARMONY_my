"""
PCH-Mem: Policy-Compiled Hypergraph Memory.

Advantage-driven policy hyperedge compilation for self-optimizing
long-term memory retrieval.

Core modules:
- types: data structures for policy hyperedges, trajectories, MDP
- mdp: budget-constrained retrieval MDP
- teacher: high-recall trajectory collection
- value_learning: BC + conservative Q-learning
- trajectory_mining: signature canonicalization, subpath mining
- advantage: held-out counterfactual advantage estimation
- lifecycle: Spawn / Merge / Reweight / Prune operations
- alternating: policy-topology alternating co-optimization
- online: Fast / Safe / Fallback layered retrieval
- pseudo_query: query-free initialization
"""

from .types import (
    PolicyHyperedge,
    RetrievalTrajectory,
    TrajectoryStep,
    StructuralHyperedge,
    HypergraphState,
    PCHConfig,
)
from .mdp import (
    RetrievalMDP,
    MDPState,
    MDPAction,
    compute_reward,
)
from .teacher import (
    TeacherRetriever,
    collect_teacher_trajectories,
)
from .value_learning import (
    QNetwork,
    train_bc,
    train_cql,
    ValuePolicy,
)
from .trajectory_mining import (
    canonicalize_trajectory,
    mine_repeated_subpaths,
    construct_candidate_policy_edge,
)
from .advantage import (
    split_proposal_validation,
    estimate_counterfactual_advantage,
    compute_advantage_lcb,
)
from .lifecycle import (
    spawn_policy_edges,
    reweight_policy_edges,
    prune_policy_edges,
)
from .alternating import (
    alternating_optimization,
    policy_evaluation_step,
    topology_update_step,
    policy_reoptimization_step,
)
from .online import (
    OnlineRetriever,
    fast_path_retrieve,
    safe_path_retrieve,
    fallback_retrieve,
)
from .pseudo_query import (
    generate_pseudo_queries,
    PseudoQueryGenerator,
)

__all__ = [
    "PolicyHyperedge",
    "RetrievalTrajectory",
    "TrajectoryStep",
    "StructuralHyperedge",
    "HypergraphState",
    "PCHConfig",
    "RetrievalMDP",
    "MDPState",
    "MDPAction",
    "compute_reward",
    "TeacherRetriever",
    "collect_teacher_trajectories",
    "QNetwork",
    "train_bc",
    "train_cql",
    "ValuePolicy",
    "canonicalize_trajectory",
    "mine_repeated_subpaths",
    "construct_candidate_policy_edge",
    "split_proposal_validation",
    "estimate_counterfactual_advantage",
    "compute_advantage_lcb",
    "spawn_policy_edges",
    "reweight_policy_edges",
    "prune_policy_edges",
    "alternating_optimization",
    "policy_evaluation_step",
    "topology_update_step",
    "policy_reoptimization_step",
    "OnlineRetriever",
    "fast_path_retrieve",
    "safe_path_retrieve",
    "fallback_retrieve",
    "generate_pseudo_queries",
    "PseudoQueryGenerator",
]
