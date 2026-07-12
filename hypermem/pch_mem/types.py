"""
PCH-Mem core data types.

Defines policy hyperedges, retrieval trajectories, MDP components,
and configuration for the PCH-Mem framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ── Hyperedge Types ──────────────────────────────────────────────


class EdgeType(str, Enum):
    """Types of hyperedges in the dual hypergraph."""
    STRUCTURAL = "structural"     # Built from dialogue content
    POLICY = "policy"             # Compiled from retrieval experience


class PolicyEdgeStatus(str, Enum):
    """Lifecycle status of a policy hyperedge."""
    CANDIDATE = "candidate"       # Proposed but not yet validated
    ACTIVE = "active"             # Validated and in topology
    DEPRECATED = "deprecated"     # Marginal contribution below threshold
    MERGED = "merged"             # Absorbed into another edge
    PRUNED = "pruned"             # Removed from topology


# ── Structural Components ────────────────────────────────────────


@dataclass
class StructuralHyperedge:
    """A structural hyperedge built from dialogue content.

    Encodes a local event or semantically coherent segment:
    {v_topic, v_episode, attributes, facts}
    """
    edge_id: str
    topic_id: str = ""
    episode_ids: List[str] = field(default_factory=list)
    fact_ids: List[str] = field(default_factory=list)
    attribute_ids: List[str] = field(default_factory=list)

    # Metadata
    role_constraints: Dict[str, str] = field(default_factory=dict)
    temporal_range: Optional[Tuple[str, str]] = None
    keywords: List[str] = field(default_factory=list)

    # Embedding (aggregated from member facts)
    embedding: Optional[np.ndarray] = None

    def member_count(self) -> int:
        return len(self.fact_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "topic_id": self.topic_id,
            "episode_ids": self.episode_ids,
            "fact_ids": self.fact_ids,
            "attribute_ids": self.attribute_ids,
            "role_constraints": self.role_constraints,
            "temporal_range": self.temporal_range,
            "keywords": self.keywords,
        }


@dataclass
class PolicyHyperedge:
    """A policy-compiled hyperedge — compressed multi-step retrieval shortcut.

    Connects: query_intent → attributes → episodes → source_facts
    as an atomic access unit.

    Unlike structural hyperedges, policy hyperedges are created from
    retrieval experience and carry advantage/utility statistics.
    """
    edge_id: str
    status: PolicyEdgeStatus = PolicyEdgeStatus.CANDIDATE

    # Connected components
    intent_prototype: str = ""           # Query intent description
    intent_embedding: Optional[np.ndarray] = None
    attribute_ids: List[str] = field(default_factory=list)
    episode_ids: List[str] = field(default_factory=list)
    fact_ids: List[str] = field(default_factory=list)
    structural_edge_ids: List[str] = field(default_factory=list)

    # Source path (compressed structural subpath)
    compressed_path: List[str] = field(default_factory=list)

    # Advantage statistics
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_lcb: float = 0.0          # Lower confidence bound
    validation_query_count: int = 0
    validation_consistency: float = 0.0

    # Utility weights (for Reweight)
    success_rate: float = 0.0
    avg_evidence_gain: float = 0.0
    avg_cost_saved: float = 0.0
    access_frequency: int = 0
    last_access_step: int = 0
    freshness_score: float = 1.0
    uncertainty: float = 1.0

    # Lifecycle metadata
    created_at_step: int = 0
    parent_edge_ids: List[str] = field(default_factory=list)

    # Embedding of the full policy edge
    embedding: Optional[np.ndarray] = None

    def compression_ratio(self) -> float:
        """Ratio of original path length to policy edge (1 access)."""
        path_len = len(self.compressed_path)
        return path_len / 1.0 if path_len > 0 else 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "status": self.status.value,
            "intent_prototype": self.intent_prototype,
            "attribute_ids": self.attribute_ids,
            "episode_ids": self.episode_ids,
            "fact_ids": self.fact_ids,
            "structural_edge_ids": self.structural_edge_ids,
            "compressed_path": self.compressed_path,
            "advantage_mean": self.advantage_mean,
            "advantage_std": self.advantage_std,
            "advantage_lcb": self.advantage_lcb,
            "validation_query_count": self.validation_query_count,
            "validation_consistency": self.validation_consistency,
            "success_rate": self.success_rate,
            "avg_evidence_gain": self.avg_evidence_gain,
            "avg_cost_saved": self.avg_cost_saved,
            "access_frequency": self.access_frequency,
            "freshness_score": self.freshness_score,
            "uncertainty": self.uncertainty,
        }


# ── Trajectory Types ─────────────────────────────────────────────


class ActionType(str, Enum):
    """Types of retrieval actions."""
    SELECT_STRUCT = "select_struct"       # Visit structural hyperedge
    SELECT_POLICY = "select_policy"       # Visit policy hyperedge
    STOP = "stop"                         # End retrieval
    FALLBACK = "fallback"                 # Trigger global retrieval


@dataclass
class MDPAction:
    """A retrieval action in the MDP."""
    action_type: ActionType
    edge_id: str = ""                    # Target hyperedge ID (empty for STOP/FALLBACK)
    edge_type: EdgeType = EdgeType.STRUCTURAL

    def is_terminal(self) -> bool:
        return self.action_type in (ActionType.STOP, ActionType.FALLBACK)

    def __hash__(self) -> int:
        return hash((self.action_type.value, self.edge_id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MDPAction):
            return False
        return self.action_type == other.action_type and self.edge_id == other.edge_id


@dataclass
class MDPState:
    """State representation for the retrieval MDP.

    s_t = [z_q, z_Et, c_t, b_t, u_t, h_t]
    """
    # Query embedding
    query_embedding: np.ndarray

    # Current evidence pool embedding (aggregated)
    evidence_embedding: np.ndarray

    # Constraint coverage: role, temporal, topic
    role_coverage: float = 0.0
    temporal_coverage: float = 0.0
    topic_coverage: float = 0.0

    # Remaining budgets
    remaining_visits: int = 5
    remaining_tokens: int = 500
    remaining_latency_ms: float = 1000.0

    # Uncertainty of current evidence
    evidence_uncertainty: float = 1.0

    # Path history
    visited_edges: List[str] = field(default_factory=list)
    collected_facts: List[str] = field(default_factory=list)
    step_count: int = 0

    # Current evidence pool set
    evidence_fact_ids: List[str] = field(default_factory=list)

    def to_feature_vector(self, dim: int = 256) -> np.ndarray:
        """Convert state to fixed-size feature vector for Q-network."""
        features = []

        # Query embedding (truncate or pad)
        q_emb = self.query_embedding
        if len(q_emb) > dim // 4:
            q_emb = q_emb[:dim // 4]
        features.extend(q_emb)

        # Evidence embedding
        e_emb = self.evidence_embedding
        if len(e_emb) > dim // 4:
            e_emb = e_emb[:dim // 4]
        features.extend(e_emb)

        # Scalar features
        scalar_features = [
            self.role_coverage,
            self.temporal_coverage,
            self.topic_coverage,
            self.remaining_visits / 10.0,          # Normalize
            self.remaining_tokens / 1000.0,
            self.remaining_latency_ms / 2000.0,
            self.evidence_uncertainty,
            self.step_count / 10.0,
            float(len(self.collected_facts)) / 20.0,
            float(len(self.visited_edges)) / 10.0,
        ]

        # Pad the evidence embedding to match expected size
        target_emb_size = dim // 4
        if len(e_emb) < target_emb_size:
            e_emb_padded = list(e_emb) + [0.0] * (target_emb_size - len(e_emb))
        else:
            e_emb_padded = list(e_emb)

        # Build final feature vector
        result = list(q_emb[:target_emb_size]) + e_emb_padded + scalar_features
        return np.array(result, dtype=np.float32)


@dataclass
class TrajectoryStep:
    """Single step in a retrieval trajectory."""
    state: MDPState
    action: MDPAction
    reward: float
    next_state: MDPState
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTrajectory:
    """Complete retrieval trajectory for one query."""
    query_id: str
    query_text: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_return: float = 0.0
    total_evidence_gain: float = 0.0
    total_cost: float = 0.0
    evidence_recall: float = 0.0

    # Canonical signature (computed after collection)
    signature: Optional[TrajectorySignature] = None

    @property
    def length(self) -> int:
        return len(self.steps)

    def append_step(self, step: TrajectoryStep) -> None:
        self.steps.append(step)
        self.total_return += step.reward

    def compute_return(self, gamma: float = 0.95) -> float:
        """Compute discounted return."""
        ret = 0.0
        for i, step in enumerate(self.steps):
            ret += (gamma ** i) * step.reward
        self.total_return = ret
        return ret


@dataclass
class TrajectorySignature:
    """Canonical structural signature of a trajectory.

    Sig(tau) = (I(q), R(q), T(q), TypeSeq(tau))
    """
    intent_prototype: str          # I(q): query intent prototype
    role_constraint: str           # R(q): role/entity constraint
    temporal_relation: str         # T(q): temporal relation type
    edge_type_sequence: Tuple[str, ...]  # TypeSeq: structural edge type sequence
    path_length: int = 0

    def to_key(self) -> str:
        """Generate a hashable key for signature matching."""
        return f"{self.intent_prototype}|{self.role_constraint}|{self.temporal_relation}|{'->'.join(self.edge_type_sequence)}"

    def is_compatible(self, other: "TrajectorySignature") -> bool:
        """Check if two signatures are compatible for clustering."""
        return (
            self.intent_prototype == other.intent_prototype
            and self.role_constraint == other.role_constraint
            and self.temporal_relation == other.temporal_relation
        )


# ── Hypergraph State ─────────────────────────────────────────────


@dataclass
class HypergraphState:
    """Complete state of the dual hypergraph for one conversation."""
    conversation_id: str

    # Structural components
    structural_edges: Dict[str, StructuralHyperedge] = field(default_factory=dict)
    fact_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    fact_contents: Dict[str, str] = field(default_factory=dict)
    fact_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Policy components
    policy_edges: Dict[str, PolicyHyperedge] = field(default_factory=dict)

    # Indexing
    structural_embedding_matrix: Optional[np.ndarray] = None
    structural_edge_list: List[str] = field(default_factory=list)
    policy_embedding_matrix: Optional[np.ndarray] = None
    policy_edge_list: List[str] = field(default_factory=list)

    # Evolution tracking
    optimization_round: int = 0
    total_policy_edges_spawned: int = 0
    total_policy_edges_pruned: int = 0

    def num_structural_edges(self) -> int:
        return len(self.structural_edges)

    def num_policy_edges(self) -> int:
        return len(self.policy_edges)

    def num_active_policy_edges(self) -> int:
        return sum(
            1 for e in self.policy_edges.values()
            if e.status == PolicyEdgeStatus.ACTIVE
        )


# ── Configuration ────────────────────────────────────────────────


@dataclass
class PCHConfig:
    """Configuration for PCH-Mem framework."""

    # ── MDP ──
    max_steps: int = 5
    max_visits: int = 8
    max_tokens: int = 600
    max_latency_ms: float = 2000.0
    gamma: float = 0.95

    # ── Reward weights ──
    alpha_evidence: float = 1.0      # Evidence gain weight
    beta_answer: float = 0.5          # Answer support weight
    lambda_visit: float = 0.05        # Visit cost
    lambda_latency: float = 0.001     # Latency cost (per ms)
    lambda_token: float = 0.002       # Token cost
    lambda_duplicate: float = 0.1     # Duplicate evidence penalty
    lambda_miss: float = 0.5          # Missing key evidence penalty

    # ── Teacher ──
    teacher_top_k_structural: int = 10
    teacher_top_k_facts: int = 20
    teacher_use_reranker: bool = True
    teacher_recall_budget: int = 8

    # ── Value Learning ──
    embedding_dim: int = 256
    hidden_dim: int = 128
    q_hidden_layers: int = 2
    bc_epochs: int = 50
    cql_epochs: int = 100
    cql_alpha: float = 0.5           # CQL conservative weight
    bc_alpha: float = 0.3            # BC regularization weight
    learning_rate: float = 3e-4
    batch_size: int = 64
    target_update_freq: int = 10

    # ── Trajectory Mining ──
    min_subpath_length: int = 2
    min_trajectory_support: int = 3   # n_min for subpath
    min_positive_contribution: float = 0.05
    signature_cluster_min_size: int = 3

    # ── Advantage Estimation ──
    proposal_ratio: float = 0.7       # 70% proposal, 30% validation
    min_validation_queries: int = 3    # n_min for validation
    advantage_threshold: float = 0.1   # tau_A for Spawn
    consistency_threshold: float = 0.6  # tau_C for Spawn
    confidence_level: float = 0.9     # 1-delta for LCB
    merge_margin: float = 0.05

    # ── Alternating Optimization ──
    max_rounds: int = 5
    utility_improvement_epsilon: float = 0.01
    topology_change_epsilon: float = 0.05
    max_policy_edges: int = 50

    # ── Online Retrieval ──
    fast_sufficiency_threshold: float = 0.7
    online_top_k_policy: int = 5
    online_top_k_structural: int = 5

    # ── Pseudo-Query Generation ──
    pseudo_queries_per_topic: int = 3
    pseudo_queries_per_episode: int = 2
    max_pseudo_queries: int = 50
