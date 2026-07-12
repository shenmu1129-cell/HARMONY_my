"""
Online Fast/Safe/Fallback layered retrieval.

After offline policy–topology optimization, online queries are
routed through three layers (Paper Section 4.9):

1. Fast Path: Query → Policy Hyperedge → Source Facts → STOP
2. Safe Path: Query → Policy Hyperedge → One Structural Edge → Facts
3. Fallback: Structural Hypergraph → Global BM25/Dense → Reranker

All evidence is projected to source facts with role, time,
session, and row-level provenance.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

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
)
from .teacher import TeacherRetriever


class OnlineRetriever:
    """Online layered retrieval with Fast/Safe/Fallback paths."""

    def __init__(
        self,
        hypergraph: HypergraphState,
        policy: "ValuePolicy",
        mdp: "RetrievalMDP",
        teacher: TeacherRetriever,
        config: PCHConfig,
    ):
        self.hg = hypergraph
        self.policy = policy
        self.mdp = mdp
        self.teacher = teacher
        self.config = config

        # Pre-compute embedding matrix for ANN
        self._build_ann_index()

        # Statistics
        self.fast_path_count = 0
        self.safe_path_count = 0
        self.fallback_count = 0
        self.total_queries = 0

    def _build_ann_index(self) -> None:
        """Build ANN indices for policy and structural edges."""
        # Policy edge embeddings
        if self.hg.policy_edges:
            active_edges = [
                (eid, e) for eid, e in self.hg.policy_edges.items()
                if e.status == PolicyEdgeStatus.ACTIVE
            ]
            if active_edges:
                embs = []
                ids = []
                for eid, e in active_edges:
                    if e.embedding is not None:
                        embs.append(e.embedding)
                        ids.append(eid)
                if embs:
                    self.hg.policy_embedding_matrix = np.stack(embs)
                    self.hg.policy_edge_list = ids

        # Structural edge embeddings
        if self.hg.structural_edges:
            embs = []
            ids = []
            for eid, e in self.hg.structural_edges.items():
                if e.embedding is not None:
                    embs.append(e.embedding)
                    ids.append(eid)
            if embs:
                self.hg.structural_embedding_matrix = np.stack(embs)
                self.hg.structural_edge_list = ids

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
    ) -> Dict[str, Any]:
        """Execute online layered retrieval for a query.

        Returns:
            Dict with selected facts, path used, latency, and metrics.
        """
        start_time = time.perf_counter()
        self.total_queries += 1

        # Step 1: Query encoding and candidate recall
        policy_candidates = self._ann_search(
            query_embedding,
            self.hg.policy_embedding_matrix,
            self.hg.policy_edge_list,
            top_k=self.config.online_top_k_policy,
        )
        structural_candidates = self._ann_search(
            query_embedding,
            self.hg.structural_embedding_matrix,
            self.hg.structural_edge_list,
            top_k=self.config.online_top_k_structural,
        )

        # Step 2: Value-based action selection
        state = self.mdp.initial_state(query_embedding)
        available_actions = self._build_candidate_actions(
            policy_candidates, structural_candidates, state,
        )
        action = self.policy.select_action(state, available_actions)

        # Step 3: Execute retrieval path
        if action.action_type == ActionType.FALLBACK:
            result = self._fallback_retrieve(query, query_embedding)
        elif action.action_type == ActionType.SELECT_POLICY:
            result = self._policy_path_retrieve(
                query, query_embedding, action, state,
            )
        else:
            result = self._structural_path_retrieve(
                query, query_embedding, action, state,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        result["latency_ms"] = latency_ms
        result["query"] = query
        return result

    def _ann_search(
        self,
        query_emb: np.ndarray,
        matrix: Optional[np.ndarray],
        edge_list: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Approximate nearest neighbor search."""
        if matrix is None or len(edge_list) == 0:
            return []

        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        scores = matrix_norm @ query_norm

        order = np.argsort(-scores)
        return [
            (edge_list[i], float(scores[i]))
            for i in order[:top_k]
            if scores[i] > 0
        ]

    def _build_candidate_actions(
        self,
        policy_candidates: List[Tuple[str, float]],
        structural_candidates: List[Tuple[str, float]],
        state: MDPState,
    ) -> List[MDPAction]:
        """Build candidate action list from ANN results."""
        actions: List[MDPAction] = []

        for eid, _ in policy_candidates:
            if eid not in state.visited_edges:
                actions.append(MDPAction(ActionType.SELECT_POLICY, eid, EdgeType.POLICY))

        for eid, _ in structural_candidates:
            if eid not in state.visited_edges:
                actions.append(MDPAction(ActionType.SELECT_STRUCT, eid, EdgeType.STRUCTURAL))

        if not actions:
            actions.append(MDPAction(ActionType.FALLBACK))
        actions.append(MDPAction(ActionType.STOP))

        return actions

    def _policy_path_retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        action: MDPAction,
        state: MDPState,
    ) -> Dict[str, Any]:
        """Fast Path or Safe Path via policy edge."""
        edge = self.hg.policy_edges.get(action.edge_id)
        if edge is None:
            return self._fallback_retrieve(query, query_embedding)

        # Execute policy edge access
        next_state, reward, done, info = self.mdp.step(state, action)

        # Check evidence sufficiency
        evidence_facts = [
            self.hg.fact_contents.get(fid, "")
            for fid in next_state.evidence_fact_ids
            if fid in self.hg.fact_contents
        ]

        sufficiency = self._estimate_sufficiency(
            query, evidence_facts, next_state,
        )

        if sufficiency >= self.config.fast_sufficiency_threshold:
            # Fast Path: enough evidence, stop
            self.fast_path_count += 1
            return {
                "path": "fast",
                "policy_edge_id": action.edge_id,
                "selected_facts": next_state.evidence_fact_ids,
                "fact_contents": evidence_facts,
                "sufficiency": sufficiency,
                "evidence_count": len(next_state.evidence_fact_ids),
                "steps": 1,
            }
        else:
            # Safe Path: expand with one structural edge
            self.safe_path_count += 1
            struct_state, struct_reward, _, struct_info = self._safe_expand(
                query_embedding, next_state,
            )
            all_facts = list(set(
                next_state.evidence_fact_ids + struct_state.evidence_fact_ids
            ))
            evidence_facts = [
                self.hg.fact_contents.get(fid, "")
                for fid in all_facts
                if fid in self.hg.fact_contents
            ]
            return {
                "path": "safe",
                "policy_edge_id": action.edge_id,
                "structural_expansion_used": True,
                "selected_facts": all_facts,
                "fact_contents": evidence_facts,
                "sufficiency": sufficiency,
                "evidence_count": len(all_facts),
                "steps": 2,
            }

    def _structural_path_retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        action: MDPAction,
        state: MDPState,
    ) -> Dict[str, Any]:
        """Structural retrieval path."""
        next_state, reward, done, info = self.mdp.step(state, action)

        evidence_facts = [
            self.hg.fact_contents.get(fid, "")
            for fid in next_state.evidence_fact_ids
            if fid in self.hg.fact_contents
        ]

        return {
            "path": "structural",
            "edge_id": action.edge_id,
            "selected_facts": next_state.evidence_fact_ids,
            "fact_contents": evidence_facts,
            "sufficiency": self._estimate_sufficiency(query, evidence_facts, next_state),
            "evidence_count": len(next_state.evidence_fact_ids),
            "steps": 1,
        }

    def _safe_expand(
        self,
        query_embedding: np.ndarray,
        state: MDPState,
    ) -> Tuple[MDPState, float, bool, Dict]:
        """Execute one structural expansion from a policy edge state."""
        # Find adjacent structural edges (edges sharing facts)
        adjacent_edges = []
        current_facts = set(state.evidence_fact_ids)
        for eid, edge in self.hg.structural_edges.items():
            if eid not in state.visited_edges:
                shared = current_facts & set(edge.fact_ids)
                new_facts = set(edge.fact_ids) - current_facts
                if new_facts:
                    adjacent_edges.append((eid, len(new_facts), len(shared)))

        if adjacent_edges:
            # Pick edge with most new facts
            adjacent_edges.sort(key=lambda x: (-x[1], x[2]))
            best_eid = adjacent_edges[0][0]
            action = MDPAction(ActionType.SELECT_STRUCT, best_eid, EdgeType.STRUCTURAL)
            return self.mdp.step(state, action)

        return state, 0.0, True, {}

    def _fallback_retrieve(
        self, query: str, query_embedding: np.ndarray,
    ) -> Dict[str, Any]:
        """Global retrieval fallback using BM25 + dense."""
        self.fallback_count += 1

        # Use teacher's hybrid retrieval
        results = self.teacher.retrieve(query, query_embedding, top_k_facts=10)
        fact_ids = [fid for fid, _ in results]

        evidence_facts = [
            self.hg.fact_contents.get(fid, "")
            for fid in fact_ids
            if fid in self.hg.fact_contents
        ]

        return {
            "path": "fallback",
            "selected_facts": fact_ids,
            "fact_contents": evidence_facts,
            "sufficiency": 0.5,
            "evidence_count": len(fact_ids),
            "steps": 0,
            "fallback_used": True,
        }

    def _estimate_sufficiency(
        self,
        query: str,
        evidence_facts: List[str],
        state: MDPState,
    ) -> float:
        """Estimate evidence sufficiency for the query.

        Based on: evidence count, semantic coverage, constraint coverage.
        """
        if not evidence_facts:
            return 0.0

        # Evidence count score
        count_score = min(1.0, len(evidence_facts) / 5.0)

        # Constraint coverage
        constraint_coverage = (
            state.role_coverage * 0.3
            + state.temporal_coverage * 0.3
            + state.topic_coverage * 0.2
            + (1.0 - state.evidence_uncertainty) * 0.2
        )

        # Combined sufficiency
        return 0.5 * count_score + 0.5 * constraint_coverage

    def get_path_distribution(self) -> Dict[str, float]:
        """Get the distribution of retrieval paths used."""
        total = max(1, self.total_queries)
        return {
            "fast": self.fast_path_count / total,
            "safe": self.safe_path_count / total,
            "fallback": self.fallback_count / total,
            "fast_count": self.fast_path_count,
            "safe_count": self.safe_path_count,
            "fallback_count": self.fallback_count,
            "total": self.total_queries,
        }


# Convenience functions
def fast_path_retrieve(
    retriever: OnlineRetriever,
    query: str,
    query_embedding: np.ndarray,
) -> Dict[str, Any]:
    return retriever.retrieve(query, query_embedding)


def safe_path_retrieve(
    retriever: OnlineRetriever,
    query: str,
    query_embedding: np.ndarray,
) -> Dict[str, Any]:
    return retriever.retrieve(query, query_embedding)


def fallback_retrieve(
    retriever: OnlineRetriever,
    query: str,
    query_embedding: np.ndarray,
) -> Dict[str, Any]:
    return retriever._fallback_retrieve(query, query_embedding)
