"""
High-recall teacher for trajectory collection.

Uses a combination of BM25, dense retrieval, RRF fusion,
and structural hypergraph expansion to collect high-quality
retrieval trajectories for offline RL training.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .types import (
    ActionType,
    EdgeType,
    HypergraphState,
    MDPAction,
    MDPState,
    PCHConfig,
    RetrievalTrajectory,
    StructuralHyperedge,
    TrajectoryStep,
    TrajectorySignature,
)


class SimpleBM25:
    """Minimal BM25 index for fact-level retrieval."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_lengths: List[int] = []
        self.avg_dl: float = 0.0
        self.N: int = 0
        self._term_to_df: Dict[str, int] = {}
        self._term_to_postings: Dict[str, Dict[int, int]] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"[a-z0-9_]+", text)
        tokens.extend(re.findall(r"[一-鿿]", text))
        return tokens

    def index(self, doc_ids: List[str], doc_texts: List[str]) -> None:
        self.doc_ids = list(doc_ids)
        self.doc_texts = list(doc_texts)
        self.N = len(doc_ids)
        self.doc_lengths = []
        self._term_to_df = {}
        self._term_to_postings = {}

        for i, text in enumerate(doc_texts):
            tokens = self.tokenize(text)
            self.doc_lengths.append(len(tokens))
            tf = Counter(tokens)
            for term, count in tf.items():
                if term not in self._term_to_df:
                    self._term_to_df[term] = 0
                    self._term_to_postings[term] = {}
                self._term_to_df[term] += 1
                self._term_to_postings[term][i] = count

        self.avg_dl = sum(self.doc_lengths) / max(1, self.N)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        if self.N == 0:
            return []
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = np.zeros(self.N, dtype=np.float32)
        for token in set(query_tokens):
            df = self._term_to_df.get(token, 0)
            if df == 0:
                continue
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            postings = self._term_to_postings.get(token, {})
            qf = query_tokens.count(token)
            for doc_idx, tf in postings.items():
                dl = max(1, self.doc_lengths[doc_idx])
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
                scores[doc_idx] += idf * tf_norm * qf

        order = np.argsort(-scores)
        return [
            (self.doc_ids[i], float(scores[i]))
            for i in order[:top_k]
            if scores[i] > 0
        ]


class TeacherRetriever:
    """High-recall teacher that combines multiple retrieval strategies.

    Pipeline:
    1. BM25 over all facts
    2. Dense ANN over structural edge embeddings
    3. RRF fusion
    4. Structural hypergraph expansion
    5. High evidence budget for full trajectory collection
    """

    def __init__(
        self,
        hypergraph: HypergraphState,
        config: PCHConfig,
        embedding_dim: int = 256,
    ):
        self.hg = hypergraph
        self.config = config
        self.embedding_dim = embedding_dim

        # Build BM25 index over fact contents
        self.bm25 = SimpleBM25()
        fact_ids = list(self.hg.fact_contents.keys())
        fact_texts = [self.hg.fact_contents[fid] for fid in fact_ids]
        self.bm25.index(fact_ids, fact_texts)

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k_facts: int = 20,
    ) -> List[Tuple[str, float]]:
        """Hybrid retrieval with RRF fusion.

        Returns ranked list of (fact_id, score).
        """
        # BM25 retrieval
        bm25_results = self.bm25.search(query, top_k=50)
        bm25_scores = {fid: score for fid, score in bm25_results}

        # Dense retrieval over structural edges
        dense_results = self._dense_retrieval(query_embedding, top_k=20)
        dense_fact_scores: Dict[str, float] = {}
        for edge_id, score in dense_results:
            edge = self.hg.structural_edges.get(edge_id)
            if edge is None:
                continue
            for fid in edge.fact_ids:
                current = dense_fact_scores.get(fid, 0.0)
                dense_fact_scores[fid] = max(current, score)

        # RRF fusion
        rrf_scores: Dict[str, float] = {}
        # BM25 contribution (rank 1-based)
        for rank, (fid, _) in enumerate(bm25_results[:50], start=1):
            rrf_scores[fid] = rrf_scores.get(fid, 0.0) + 1.0 / (60 + rank)

        # Dense contribution
        dense_ranked = sorted(dense_fact_scores.items(), key=lambda x: -x[1])
        for rank, (fid, _) in enumerate(dense_ranked[:50], start=1):
            rrf_scores[fid] = rrf_scores.get(fid, 0.0) + 1.0 / (60 + rank)

        # Sort and return top-k
        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return ranked[:top_k_facts]

    def _dense_retrieval(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """Dense ANN retrieval over structural edge embeddings."""
        if not self.hg.structural_edge_list:
            return []

        edge_embeddings = []
        for eid in self.hg.structural_edge_list:
            edge = self.hg.structural_edges.get(eid)
            if edge is not None and edge.embedding is not None:
                edge_embeddings.append(edge.embedding)
            else:
                # Fallback: aggregate fact embeddings
                fact_embs = [
                    self.hg.fact_embeddings.get(fid, np.zeros(self.embedding_dim))
                    for fid in (edge.fact_ids if edge else [])
                ]
                emb = np.mean(fact_embs, axis=0) if fact_embs else np.zeros(self.embedding_dim)
                edge_embeddings.append(emb)

        if not edge_embeddings:
            return []

        matrix = np.stack(edge_embeddings)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        scores = matrix_norm @ query_norm

        ranked = sorted(
            zip(self.hg.structural_edge_list, scores),
            key=lambda x: -x[1],
        )
        return [(eid, float(s)) for eid, s in ranked[:top_k]]

    def collect_trajectory(
        self,
        query_id: str,
        query_text: str,
        query_embedding: np.ndarray,
        mdp: "RetrievalMDP",
        gold_fact_ids: Optional[Set[str]] = None,
    ) -> RetrievalTrajectory:
        """Collect a high-recall retrieval trajectory using teacher policy.

        The teacher greedily expands evidence until budget exhausted
        or evidence gain plateaus.
        """
        trajectory = RetrievalTrajectory(
            query_id=query_id,
            query_text=query_text,
        )

        state = mdp.initial_state(query_embedding)
        done = False

        while not done and state.step_count < self.config.max_steps:
            # Teacher policy: select structural edge with highest
            # expected evidence gain
            available = mdp.get_available_actions(state)
            struct_actions = [
                a for a in available
                if a.action_type == ActionType.SELECT_STRUCT
            ]

            if not struct_actions:
                action = MDPAction(ActionType.STOP)
            else:
                # Score structural edges by potential new evidence
                best_action = None
                best_score = -float("inf")
                for action in struct_actions:
                    edge = self.hg.structural_edges.get(action.edge_id)
                    if edge is None:
                        continue
                    new_facts = len([
                        f for f in edge.fact_ids
                        if f not in state.evidence_fact_ids
                    ])
                    # Prefer edges with more new evidence
                    score = new_facts
                    if score > best_score:
                        best_score = score
                        best_action = action

                if best_action is None or best_score == 0:
                    action = MDPAction(ActionType.STOP)
                else:
                    action = best_action

            # Execute action
            next_state, reward, done, info = mdp.step(state, action)

            step = TrajectoryStep(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                info=info,
            )
            trajectory.append_step(step)
            state = next_state

        # Compute total evidence recall if gold facts provided
        if gold_fact_ids:
            retrieved = set(trajectory.steps[-1].next_state.evidence_fact_ids if trajectory.steps else [])
            trajectory.evidence_recall = (
                len(retrieved & gold_fact_ids) / max(1, len(gold_fact_ids))
            )

        # Compute total cost
        trajectory.total_cost = sum(
            s.info.get("new_facts", 0) * 10  # rough token cost
            for s in trajectory.steps
        )

        return trajectory


def collect_teacher_trajectories(
    queries: List[Tuple[str, str, np.ndarray]],
    mdp: "RetrievalMDP",
    teacher: TeacherRetriever,
    gold_facts_map: Optional[Dict[str, Set[str]]] = None,
) -> List[RetrievalTrajectory]:
    """Collect teacher trajectories for a set of queries.

    Args:
        queries: List of (query_id, query_text, query_embedding)
        mdp: Retrieval MDP instance
        teacher: Teacher retriever
        gold_facts_map: Optional mapping from query_id to gold fact IDs

    Returns:
        List of collected trajectories
    """
    trajectories = []
    for qid, qtext, qemb in queries:
        gold = gold_facts_map.get(qid) if gold_facts_map else None
        traj = teacher.collect_trajectory(
            query_id=qid,
            query_text=qtext,
            query_embedding=qemb,
            mdp=mdp,
            gold_fact_ids=gold,
        )
        trajectories.append(traj)
    return trajectories
