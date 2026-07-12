"""
Budget-constrained retrieval MDP.

Defines the Markov Decision Process for evidence retrieval
over the dual hypergraph (structural + policy hyperedges).

M_R(k) = (S(k), A(k), P(k), R, gamma, H)
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
    StructuralHyperedge,
)


class RetrievalMDP:
    """Budget-constrained retrieval MDP over the dual hypergraph."""

    def __init__(
        self,
        hypergraph: HypergraphState,
        config: PCHConfig,
    ):
        self.hg = hypergraph
        self.config = config

        # Pre-build action index
        self._structural_action_list: List[MDPAction] = []
        self._policy_action_list: List[MDPAction] = []
        self._rebuild_action_lists()

    def _rebuild_action_lists(self) -> None:
        """Rebuild action lists from current hypergraph state."""
        self._structural_action_list = [
            MDPAction(ActionType.SELECT_STRUCT, eid, EdgeType.STRUCTURAL)
            for eid in self.hg.structural_edges
        ]
        self._policy_action_list = [
            MDPAction(ActionType.SELECT_POLICY, pid, EdgeType.POLICY)
            for pid, edge in self.hg.policy_edges.items()
            if edge.status.value in ("active", "candidate")
        ]

    @property
    def terminal_actions(self) -> List[MDPAction]:
        return [
            MDPAction(ActionType.STOP),
            MDPAction(ActionType.FALLBACK),
        ]

    def initial_state(
        self,
        query_embedding: np.ndarray,
        role_constraint: str = "",
    ) -> MDPState:
        """Create initial MDP state for a query."""
        return MDPState(
            query_embedding=query_embedding.copy(),
            evidence_embedding=np.zeros_like(query_embedding),
            role_coverage=1.0 if not role_constraint else 0.0,
            temporal_coverage=0.0,
            topic_coverage=0.0,
            remaining_visits=self.config.max_visits,
            remaining_tokens=self.config.max_tokens,
            remaining_latency_ms=self.config.max_latency_ms,
            evidence_uncertainty=1.0,
            visited_edges=[],
            collected_facts=[],
            step_count=0,
            evidence_fact_ids=[],
        )

    def get_available_actions(
        self,
        state: MDPState,
        top_k_structural: int = 5,
        top_k_policy: int = 5,
    ) -> List[MDPAction]:
        """Get available actions for current state.

        Returns top-k structural edges + top-k policy edges + terminal actions.
        For simplicity, we return all edges filtered by budget constraints.
        """
        actions: List[MDPAction] = []

        # Policy edges (prioritized)
        for action in self._policy_action_list:
            if action.edge_id not in state.visited_edges:
                actions.append(action)

        # Structural edges
        for action in self._structural_action_list:
            if action.edge_id not in state.visited_edges:
                actions.append(action)

        # Limit candidates
        if len(actions) > top_k_structural + top_k_policy:
            # Simple: take first top_k of each
            policy_actions = [a for a in actions if a.edge_type == EdgeType.POLICY][:top_k_policy]
            struct_actions = [a for a in actions if a.edge_type == EdgeType.STRUCTURAL][:top_k_structural]
            actions = policy_actions + struct_actions

        # Always add terminal actions
        actions.extend(self.terminal_actions)
        return actions

    def step(
        self,
        state: MDPState,
        action: MDPAction,
    ) -> Tuple[MDPState, float, bool, Dict]:
        """Execute action and return (next_state, reward, done, info)."""
        info: Dict = {"action_type": action.action_type.value}

        if action.action_type == ActionType.STOP:
            return self._handle_stop(state, info)

        if action.action_type == ActionType.FALLBACK:
            return self._handle_fallback(state, info)

        if action.action_type == ActionType.SELECT_STRUCT:
            return self._handle_select_structural(state, action, info)

        if action.action_type == ActionType.SELECT_POLICY:
            return self._handle_select_policy(state, action, info)

        # Unknown action
        return state, 0.0, True, info

    def _handle_stop(self, state: MDPState, info: Dict) -> Tuple[MDPState, float, bool, Dict]:
        """Handle STOP action."""
        # Final reward based on evidence sufficiency
        evidence_count = len(state.evidence_fact_ids)
        coverage_score = min(1.0, evidence_count / 5.0)

        reward = coverage_score * self.config.alpha_evidence
        info["stop_reason"] = "explicit_stop"
        info["evidence_count"] = evidence_count

        return state, reward, True, info

    def _handle_fallback(self, state: MDPState, info: Dict) -> Tuple[MDPState, float, bool, Dict]:
        """Handle FALLBACK action (global retrieval).

        In practice, this would trigger BM25/dense/reranker.
        Here we model it as acquiring evidence at higher cost.
        """
        # Simulate fallback retrieving additional facts
        fallback_cost = 0.3  # Higher cost for fallback
        fallback_gain = 0.5  # Moderate evidence gain

        reward = (
            fallback_gain * self.config.alpha_evidence
            - fallback_cost * self.config.lambda_visit
            - 50.0 * self.config.lambda_latency  # Higher latency
            - 100.0 * self.config.lambda_token
        )

        next_state = MDPState(
            query_embedding=state.query_embedding.copy(),
            evidence_embedding=state.evidence_embedding.copy(),
            role_coverage=min(1.0, state.role_coverage + 0.5),
            temporal_coverage=min(1.0, state.temporal_coverage + 0.5),
            topic_coverage=min(1.0, state.topic_coverage + 0.5),
            remaining_visits=state.remaining_visits - 1,
            remaining_tokens=state.remaining_tokens - 100,
            remaining_latency_ms=state.remaining_latency_ms - 50.0,
            evidence_uncertainty=state.evidence_uncertainty * 0.8,
            visited_edges=state.visited_edges + ["FALLBACK"],
            collected_facts=state.collected_facts,
            step_count=state.step_count + 1,
            evidence_fact_ids=list(state.evidence_fact_ids),
        )

        info["fallback_used"] = True
        return next_state, reward, True, info

    def _handle_select_structural(
        self, state: MDPState, action: MDPAction, info: Dict
    ) -> Tuple[MDPState, float, bool, Dict]:
        """Handle SELECT_STRUCT action."""
        edge = self.hg.structural_edges.get(action.edge_id)
        if edge is None:
            return state, -0.1, False, {"error": f"Unknown edge {action.edge_id}"}

        # Compute evidence gain
        new_facts = [f for f in edge.fact_ids if f not in state.evidence_fact_ids]
        dup_facts = [f for f in edge.fact_ids if f in state.evidence_fact_ids]

        evidence_gain = len(new_facts) / max(1, len(edge.fact_ids))
        duplicate_penalty = len(dup_facts) / max(1, len(edge.fact_ids))

        # Cost components
        visit_cost = 1.0
        latency_cost = 15.0  # ms per structural edge visit
        token_cost = sum(
            len(self.hg.fact_contents.get(f, "").split())
            for f in edge.fact_ids
        )

        # Compute reward
        reward = (
            evidence_gain * self.config.alpha_evidence
            - visit_cost * self.config.lambda_visit
            - latency_cost * self.config.lambda_latency
            - token_cost * self.config.lambda_token
            - duplicate_penalty * self.config.lambda_duplicate
        )

        # Update evidence embedding (simple average)
        if new_facts:
            new_embs = [
                self.hg.fact_embeddings.get(f, np.zeros_like(state.query_embedding))
                for f in new_facts
            ]
            avg_new_emb = np.mean(new_embs, axis=0)
            if np.linalg.norm(state.evidence_embedding) == 0:
                updated_evidence_emb = avg_new_emb
            else:
                alpha = len(new_facts) / (len(state.evidence_fact_ids) + len(new_facts))
                updated_evidence_emb = (
                    (1 - alpha) * state.evidence_embedding + alpha * avg_new_emb
                )
        else:
            updated_evidence_emb = state.evidence_embedding.copy()

        # Update constraint coverage
        role_cov = min(1.0, state.role_coverage + (0.2 if edge.role_constraints else 0.0))
        temporal_cov = min(1.0, state.temporal_coverage + (0.3 if edge.temporal_range else 0.0))

        updated_fact_ids = list(set(state.evidence_fact_ids) | set(edge.fact_ids))

        next_state = MDPState(
            query_embedding=state.query_embedding.copy(),
            evidence_embedding=updated_evidence_emb,
            role_coverage=role_cov,
            temporal_coverage=temporal_cov,
            topic_coverage=min(1.0, state.topic_coverage + 0.15),
            remaining_visits=state.remaining_visits - 1,
            remaining_tokens=state.remaining_tokens - token_cost,
            remaining_latency_ms=state.remaining_latency_ms - latency_cost,
            evidence_uncertainty=state.evidence_uncertainty * 0.9,
            visited_edges=state.visited_edges + [action.edge_id],
            collected_facts=state.collected_facts + new_facts,
            step_count=state.step_count + 1,
            evidence_fact_ids=updated_fact_ids,
        )

        # Check budget exhaustion
        done = (
            next_state.remaining_visits <= 0
            or next_state.remaining_tokens <= 0
            or next_state.remaining_latency_ms <= 0
        )

        info["new_facts"] = len(new_facts)
        info["edge_fact_count"] = len(edge.fact_ids)
        info["budget_exhausted"] = done

        return next_state, reward, done, info

    def _handle_select_policy(
        self, state: MDPState, action: MDPAction, info: Dict
    ) -> Tuple[MDPState, float, bool, Dict]:
        """Handle SELECT_POLICY action.

        Policy edges provide direct access to source facts with lower cost.
        """
        edge = self.hg.policy_edges.get(action.edge_id)
        if edge is None:
            return state, -0.1, False, {"error": f"Unknown policy edge {action.edge_id}"}

        # Evidence gain (policy edges directly access source facts)
        new_facts = [f for f in edge.fact_ids if f not in state.evidence_fact_ids]
        dup_facts = [f for f in edge.fact_ids if f in state.evidence_fact_ids]

        evidence_gain = len(new_facts) / max(1, len(edge.fact_ids))
        duplicate_penalty = len(dup_facts) / max(1, len(edge.fact_ids))

        # Policy edges are cheaper than structural edges
        visit_cost = 0.5  # Half the cost of structural
        latency_cost = 5.0  # Lower latency
        token_cost = sum(
            len(self.hg.fact_contents.get(f, "").split())
            for f in edge.fact_ids
        )

        reward = (
            evidence_gain * self.config.alpha_evidence
            - visit_cost * self.config.lambda_visit
            - latency_cost * self.config.lambda_latency
            - token_cost * self.config.lambda_token
            - duplicate_penalty * self.config.lambda_duplicate
        )

        # Update state
        if new_facts:
            new_embs = [
                self.hg.fact_embeddings.get(f, np.zeros_like(state.query_embedding))
                for f in new_facts
            ]
            avg_new_emb = np.mean(new_embs, axis=0)
            if np.linalg.norm(state.evidence_embedding) == 0:
                updated_evidence_emb = avg_new_emb
            else:
                alpha = len(new_facts) / max(1, len(state.evidence_fact_ids) + len(new_facts))
                updated_evidence_emb = (
                    (1 - alpha) * state.evidence_embedding + alpha * avg_new_emb
                )
        else:
            updated_evidence_emb = state.evidence_embedding.copy()

        updated_fact_ids = list(set(state.evidence_fact_ids) | set(edge.fact_ids))

        next_state = MDPState(
            query_embedding=state.query_embedding.copy(),
            evidence_embedding=updated_evidence_emb,
            role_coverage=min(1.0, state.role_coverage + 0.3),
            temporal_coverage=min(1.0, state.temporal_coverage + 0.3),
            topic_coverage=min(1.0, state.topic_coverage + 0.2),
            remaining_visits=state.remaining_visits - 1,
            remaining_tokens=state.remaining_tokens - token_cost,
            remaining_latency_ms=state.remaining_latency_ms - latency_cost,
            evidence_uncertainty=state.evidence_uncertainty * 0.85,
            visited_edges=state.visited_edges + [action.edge_id],
            collected_facts=state.collected_facts + new_facts,
            step_count=state.step_count + 1,
            evidence_fact_ids=updated_fact_ids,
        )

        done = (
            next_state.remaining_visits <= 0
            or next_state.remaining_tokens <= 0
            or next_state.remaining_latency_ms <= 0
        )

        info["new_facts"] = len(new_facts)
        info["policy_edge_used"] = True
        info["compression_ratio"] = edge.compression_ratio()

        return next_state, reward, done, info


def compute_reward(
    evidence_gain: float,
    answer_support: float,
    visit_count: int,
    latency_ms: float,
    token_count: int,
    duplicate_count: int,
    miss_penalty: float,
    config: PCHConfig,
) -> float:
    """Standalone reward computation function.

    U(q, tau_q) = alpha*Q_evi + beta*Q_ans
                  - lambda_v*C_visit - lambda_l*C_latency
                  - lambda_k*C_token - lambda_d*C_dup
                  - lambda_m*C_miss
    """
    return (
        config.alpha_evidence * evidence_gain
        + config.beta_answer * answer_support
        - config.lambda_visit * visit_count
        - config.lambda_latency * latency_ms
        - config.lambda_token * token_count
        - config.lambda_duplicate * duplicate_count
        - config.lambda_miss * miss_penalty
    )
