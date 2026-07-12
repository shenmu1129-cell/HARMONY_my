"""
Minimal dynamic hierarchical HyperMem demo components.

This module is intentionally independent from the production six-stage
HyperMem pipeline. It keeps the same high-level ideas (topics, episodes,
facts, hyperedges, and coarse-to-fine retrieval) while adding variable-depth
segmentation and query-adaptive expansion for CPU-only experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DynamicMemoryNode:
    node_id: str
    node_type: str  # root/topic/subtopic/episode/subepisode/fact
    title: str
    summary: str
    text: str
    level: int
    parent_id: Optional[str]
    children: List[str] = field(default_factory=list)
    fact_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamicHyperedge:
    edge_id: str
    edge_type: str  # sibling_group / episode_fact / cross_episode_fact
    node_ids: List[str]
    parent_id: Optional[str]
    weight: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamicHierarchyMemory:
    nodes: Dict[str, DynamicMemoryNode]
    hyperedges: Dict[str, DynamicHyperedge]
    root_id: str

    @property
    def facts(self) -> Dict[str, DynamicMemoryNode]:
        return {
            node_id: node
            for node_id, node in self.nodes.items()
            if node.node_type == "fact"
        }


@dataclass
class DynamicRetrievalResult:
    query: str
    query_type: str
    retrieved_fact_ids: List[str]
    expanded_node_ids: List[str]
    expanded_path: List[str]
    context_tokens: int
    latency_ms: float
    expanded_depth: int
    expanded_breadth: int
    evidence_hit: Optional[bool] = None


@dataclass
class StructuredFact:
    fact_id: str
    text: str
    speaker: str
    time_index: int
    episode_id: str
    node_id: str
    keywords: List[str]
    entities: List[str]
    source_turn_ids: List[str]

    def searchable_text(self) -> str:
        parts = [
            f"[{self.speaker}]" if self.speaker else "",
            f"[time:{self.time_index}]",
            f"[episode:{self.episode_id}]",
            "keywords: " + " ".join(self.keywords) if self.keywords else "",
            "entities: " + " ".join(self.entities) if self.entities else "",
            self.text,
            " ".join(self.source_turn_ids),
        ]
        return " ".join(part for part in parts if part)


class SimpleTfidfIndex:
    """Small mixed Chinese/English TF-IDF index with cosine search."""

    def __init__(self) -> None:
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def fit(self, doc_ids: Sequence[str], documents: Sequence[str]) -> "SimpleTfidfIndex":
        self.doc_ids = list(doc_ids)
        self.documents = list(documents)
        tokenized = [self.tokenize(doc) for doc in self.documents]
        vocab = sorted({token for tokens in tokenized for token in tokens})
        self.vocab = {token: idx for idx, token in enumerate(vocab)}

        if not self.vocab:
            self.idf = np.zeros(0, dtype=np.float32)
            self.matrix = np.zeros((len(self.documents), 0), dtype=np.float32)
            return self

        df = np.zeros(len(self.vocab), dtype=np.float32)
        for tokens in tokenized:
            for token in set(tokens):
                df[self.vocab[token]] += 1.0

        n_docs = max(len(self.documents), 1)
        self.idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        rows = [self._vectorize_tokens(tokens) for tokens in tokenized]
        self.matrix = np.vstack(rows).astype(np.float32)
        return self

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.doc_ids or self.matrix.size == 0:
            return []
        query_vec = self.vectorize(query)
        if query_vec.size == 0 or np.linalg.norm(query_vec) == 0:
            return []
        scores = self.matrix @ query_vec
        order = np.argsort(-scores)
        return [
            (self.doc_ids[idx], float(scores[idx]))
            for idx in order[:top_k]
            if scores[idx] > 0.0
        ]

    def vectorize_many(self, texts: Sequence[str]) -> np.ndarray:
        if not self.vocab:
            self.fit([str(i) for i in range(len(texts))], texts)
        vectors = [self.vectorize(text) for text in texts]
        return np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, 0))

    def vectorize(self, text: str) -> np.ndarray:
        return self._vectorize_tokens(self.tokenize(text))

    def _vectorize_tokens(self, tokens: Sequence[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        if not tokens or not self.vocab:
            return vec
        counts = Counter(token for token in tokens if token in self.vocab)
        if not counts:
            return vec
        total = float(sum(counts.values()))
        for token, count in counts.items():
            vec[self.vocab[token]] = (count / total) * self.idf[self.vocab[token]]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        tokens: List[str] = []
        tokens.extend(re.findall(r"[a-z0-9_]+", text))
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
        tokens.extend(cjk_chars)
        tokens.extend(
            "".join(pair)
            for pair in zip(cjk_chars, cjk_chars[1:])
        )
        return tokens


class DynamicHierarchyBuilder:
    """Build variable-depth memory nodes and hyperedges using DP segmentation."""

    def __init__(
        self,
        max_depth: int = 5,
        min_segment_size: int = 2,
        max_leaf_tokens: int = 200,
        split_gain_threshold: float = 0.08,
        alpha: float = 0.05,
        beta: float = 0.1,
        boundary_gamma: float = 0.08,
    ) -> None:
        self.max_depth = max_depth
        self.min_segment_size = min_segment_size
        self.max_leaf_tokens = max_leaf_tokens
        self.split_gain_threshold = split_gain_threshold
        self.alpha = alpha
        self.beta = beta
        self.boundary_gamma = boundary_gamma
        self.nodes: Dict[str, DynamicMemoryNode] = {}
        self.hyperedges: Dict[str, DynamicHyperedge] = {}
        self._utterances: List[str] = []
        self._vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._next_node = 0
        self._next_edge = 0
        self._next_fact = 0
        self._stop_terms = {
            "the", "and", "for", "with", "that", "this", "from", "have", "has",
            "had", "was", "were", "are", "about", "what", "when", "where", "which",
            "how", "why", "did", "does", "would", "could", "should", "session",
            "observation",
        }

    def build(self, utterances: Sequence[str]) -> DynamicHierarchyMemory:
        self.nodes = {}
        self.hyperedges = {}
        self._next_node = 0
        self._next_edge = 0
        self._next_fact = 0
        self._utterances = [utt.strip() for utt in utterances if utt and utt.strip()]

        root_id = "root"
        full_text = "\n".join(self._utterances)
        self.nodes[root_id] = DynamicMemoryNode(
            node_id=root_id,
            node_type="root",
            title="Conversation Memory",
            summary=self._summarize(full_text),
            text=full_text,
            level=0,
            parent_id=None,
            metadata={"start": 0, "end": len(self._utterances)},
        )

        base_index = SimpleTfidfIndex().fit(
            [str(i) for i in range(len(self._utterances))],
            self._utterances,
        )
        self._vectors = base_index.vectorize_many(self._utterances)
        self._build_children(root_id, 0, len(self._utterances), 1)
        return DynamicHierarchyMemory(self.nodes, self.hyperedges, root_id)

    def _build_children(self, parent_id: str, start: int, end: int, level: int) -> None:
        segment_count = end - start
        segment_text = "\n".join(self._utterances[start:end])
        should_stop = (
            level >= self.max_depth
            or segment_count < self.min_segment_size * 2
            or estimate_tokens(segment_text) <= self.max_leaf_tokens
        )
        split_ranges: List[Tuple[int, int]] = []
        if not should_stop:
            split_ranges, gain = self._dp_split(start, end)
            if gain < self.split_gain_threshold or len(split_ranges) <= 1:
                split_ranges = []

        if not split_ranges:
            self._add_leaf(parent_id, start, end, level)
            return

        child_ids: List[str] = []
        for child_start, child_end in split_ranges:
            node_id = self._new_node_id(level)
            text = "\n".join(self._utterances[child_start:child_end])
            node = DynamicMemoryNode(
                node_id=node_id,
                node_type=self._node_type(level, is_leaf=False),
                title=self._title(text),
                summary=self._summarize(text),
                text=text,
                level=level,
                parent_id=parent_id,
                metadata={"start": child_start, "end": child_end},
            )
            self.nodes[node_id] = node
            self.nodes[parent_id].children.append(node_id)
            child_ids.append(node_id)
            self._build_children(node_id, child_start, child_end, level + 1)

        self._add_sibling_hyperedge(parent_id, child_ids)

    def _add_leaf(self, parent_id: str, start: int, end: int, level: int) -> str:
        text = "\n".join(self._utterances[start:end])
        node_id = self._new_node_id(level)
        leaf = DynamicMemoryNode(
            node_id=node_id,
            node_type=self._node_type(level, is_leaf=True),
            title=self._title(text),
            summary=self._summarize(text),
            text=text,
            level=level,
            parent_id=parent_id,
            metadata={"start": start, "end": end, "leaf": True},
        )
        self.nodes[node_id] = leaf
        self.nodes[parent_id].children.append(node_id)

        fact_ids = self._extract_facts(node_id, text, level + 1, start, end)
        leaf.fact_ids.extend(fact_ids)
        if fact_ids:
            self._add_hyperedge(
                "episode_fact",
                fact_ids,
                parent_id=node_id,
                weight=1.0,
                metadata={"start": start, "end": end},
            )
        return node_id

    def _extract_facts(
        self,
        parent_id: str,
        text: str,
        level: int,
        start: int,
        end: int,
    ) -> List[str]:
        parts = re.split(r"[\n。！？!?；;]+", text)
        facts: List[str] = []
        for raw in parts:
            sentence = raw.strip(" -\t")
            if len(sentence) < 8:
                continue
            facts.append(sentence)
        if not facts and text.strip():
            facts = [text.strip()]

        fact_ids: List[str] = []
        for idx, fact_text in enumerate(facts):
            fact_id = f"fact_{self._next_fact:03d}"
            self._next_fact += 1
            structured = self._structure_fact(fact_id, fact_text, parent_id, start, idx)
            self.nodes[fact_id] = DynamicMemoryNode(
                node_id=fact_id,
                node_type="fact",
                title=f"Fact {self._next_fact}",
                summary=structured.searchable_text(),
                text=structured.searchable_text(),
                level=level,
                parent_id=parent_id,
                metadata={
                    "fact_id": structured.fact_id,
                    "raw_text": structured.text,
                    "speaker": structured.speaker,
                    "time_index": structured.time_index,
                    "episode_id": structured.episode_id,
                    "node_id": structured.node_id,
                    "keywords": structured.keywords,
                    "entities": structured.entities,
                    "source_turn_ids": structured.source_turn_ids,
                    "start": start,
                    "end": end,
                    "order": idx,
                    "path": self.path_to(parent_id) + [parent_id],
                },
            )
            self.nodes[parent_id].children.append(fact_id)
            fact_ids.append(fact_id)
        return fact_ids

    def _structure_fact(
        self,
        fact_id: str,
        fact_text: str,
        parent_id: str,
        start: int,
        order: int,
    ) -> StructuredFact:
        source_turn_ids = re.findall(r"D\d+:\d+", fact_text)
        speaker = ""
        speaker_match = re.search(r"\]\s*(?:Observation\s+)?([A-Z][A-Za-z0-9_-]+)\s*:", fact_text)
        if speaker_match:
            speaker = speaker_match.group(1)
        time_index = start * 1000 + order
        session_match = re.search(r"session_(\d+)", fact_text.lower())
        if session_match:
            time_index = int(session_match.group(1)) * 1000 + order
        elif source_turn_ids:
            turn_match = re.match(r"D(\d+):(\d+)", source_turn_ids[0])
            if turn_match:
                time_index = int(turn_match.group(1)) * 1000 + int(turn_match.group(2))

        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", fact_text)
        counts = Counter(token.lower() for token in tokens if token.lower() not in self._stop_terms)
        keywords = [token for token, _ in counts.most_common(8)]
        entities = []
        for token in tokens:
            if token[:1].isupper() and token.lower() not in self._stop_terms and token not in entities:
                entities.append(token)
        if speaker and speaker not in entities:
            entities.insert(0, speaker)
        return StructuredFact(
            fact_id=fact_id,
            text=fact_text,
            speaker=speaker,
            time_index=time_index,
            episode_id=parent_id,
            node_id=parent_id,
            keywords=keywords,
            entities=entities[:8],
            source_turn_ids=source_turn_ids,
        )

    def _dp_split(self, start: int, end: int) -> Tuple[List[Tuple[int, int]], float]:
        n = end - start
        dp = [float("inf")] * (n + 1)
        prev = [-1] * (n + 1)
        dp[0] = 0.0

        for j in range(self.min_segment_size, n + 1):
            for i in range(0, j - self.min_segment_size + 1):
                if i > 0 and i < self.min_segment_size:
                    continue
                cost = self._segment_cost(start + i, start + j)
                if i > 0:
                    boundary = 1.0 - self._adjacent_similarity(start + i - 1, start + i)
                    cost -= self.boundary_gamma * boundary
                candidate = dp[i] + cost
                if candidate < dp[j]:
                    dp[j] = candidate
                    prev[j] = i

        if not math.isfinite(dp[n]):
            return [], 0.0

        ranges: List[Tuple[int, int]] = []
        cursor = n
        while cursor > 0 and prev[cursor] >= 0:
            i = prev[cursor]
            ranges.append((start + i, start + cursor))
            cursor = i
        ranges.reverse()

        whole_cost = self._segment_cost(start, end)
        split_cost = sum(self._segment_cost(i, j) for i, j in ranges)
        gain = (whole_cost - split_cost) / max(abs(whole_cost), 1e-6)
        return ranges, gain

    def _segment_cost(self, start: int, end: int) -> float:
        n = end - start
        if n <= 1:
            cohesion_cost = 0.0
        else:
            sims: List[float] = []
            for i in range(start, end):
                for j in range(i + 1, end):
                    sims.append(float(self._vectors[i] @ self._vectors[j]))
            avg_sim = sum(sims) / len(sims) if sims else 1.0
            cohesion_cost = 1.0 - avg_sim
        length_penalty = max(0.0, (n / max(self.min_segment_size, 1)) - 1.0) ** 2
        return cohesion_cost + self.alpha * length_penalty + self.beta

    def _adjacent_similarity(self, left: int, right: int) -> float:
        if left < 0 or right >= len(self._utterances):
            return 0.0
        return float(self._vectors[left] @ self._vectors[right])

    def _add_sibling_hyperedge(self, parent_id: str, child_ids: List[str]) -> None:
        if len(child_ids) <= 1:
            return
        self._add_hyperedge(
            "sibling_group",
            child_ids,
            parent_id=parent_id,
            weight=1.0,
            metadata={"child_count": len(child_ids)},
        )

    def _add_hyperedge(
        self,
        edge_type: str,
        node_ids: List[str],
        parent_id: Optional[str],
        weight: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        edge_id = f"dyn_edge_{self._next_edge:03d}"
        self._next_edge += 1
        self.hyperedges[edge_id] = DynamicHyperedge(
            edge_id=edge_id,
            edge_type=edge_type,
            node_ids=list(node_ids),
            parent_id=parent_id,
            weight=weight,
            metadata=metadata or {},
        )

    def _new_node_id(self, level: int) -> str:
        node_id = f"dyn_l{level}_{self._next_node:03d}"
        self._next_node += 1
        return node_id

    @staticmethod
    def _node_type(level: int, is_leaf: bool) -> str:
        if level <= 1:
            return "topic"
        if not is_leaf and level == 2:
            return "subtopic"
        if level == 2:
            return "episode"
        return "subepisode" if is_leaf else "subtopic"

    @staticmethod
    def _title(text: str) -> str:
        clean = re.sub(r"\s+", " ", text).strip()
        return clean[:28] + ("..." if len(clean) > 28 else "")

    @staticmethod
    def _summarize(text: str, max_chars: int = 120) -> str:
        clean = re.sub(r"\s+", " ", text).strip()
        return clean[:max_chars] + ("..." if len(clean) > max_chars else "")

    def path_to(self, node_id: str) -> List[str]:
        path: List[str] = []
        cursor: Optional[str] = node_id
        while cursor:
            path.append(cursor)
            cursor = self.nodes[cursor].parent_id if cursor in self.nodes else None
        return list(reversed(path))


class DynamicRetriever:
    """Query-adaptive coarse-to-fine retrieval over DynamicHierarchyMemory."""

    BREADTH_MARKERS = (
        "总结", "有哪些", "方向", "比较", "整体", "all", "summarize", "overview",
        "options", "choices",
    )
    DEPTH_MARKERS = (
        "为什么", "如何", "过程", "细节", "怎么设计", "why", "how", "reason",
        "detail", "design",
    )
    TEMPORAL_MARKERS = (
        "什么时候", "之前", "后来", "变化", "时间线", "逐渐", "before", "after",
        "timeline", "evolve", "evolution",
    )
    TEMPORAL_HIGH_CONFIDENCE_MARKERS = (
        "之前", "后来", "之后", "变化", "逐渐", "过程", "时间线", "什么时候", "先后",
        "before", "after", "later", "earlier", "change", "evolve", "timeline",
        "over time", "when",
    )

    def __init__(
        self,
        memory: DynamicHierarchyMemory,
        max_context_tokens: int = 1000,
        fallback_ratio: float = 0.3,
        enable_global_fact_fallback: bool = False,
        enable_temporal_expansion: bool = False,
        temporal_max_nodes: int = 5,
        temporal_facts_per_node: int = 2,
    ) -> None:
        self.memory = memory
        self.max_context_tokens = max_context_tokens
        self.fallback_ratio = min(max(fallback_ratio, 0.0), 0.9)
        self.enable_global_fact_fallback = enable_global_fact_fallback
        self.enable_temporal_expansion = enable_temporal_expansion
        self.temporal_max_nodes = temporal_max_nodes
        self.temporal_facts_per_node = temporal_facts_per_node
        self.node_index = SimpleTfidfIndex()
        self.fact_index = SimpleTfidfIndex()
        self._build_indexes()

    def retrieve(
        self,
        query: str,
        evidence_ids: Optional[Iterable[str]] = None,
        top_k: int = 5,
    ) -> DynamicRetrievalResult:
        start = time.perf_counter()
        query_type = self.detect_query_type(query)
        selected_nodes: List[str] = []
        selected_facts: List[str] = []
        path: List[str] = []

        candidates = self.node_index.search(query, top_k=top_k)
        if not candidates:
            candidates = [(self.memory.root_id, 0.0)]
        anchor_id = candidates[0][0]
        if anchor_id == self.memory.root_id and len(candidates) > 1:
            anchor_id = candidates[1][0]

        if query_type == "breadth":
            selected_nodes, selected_facts, path = self._retrieve_breadth(query, anchor_id, top_k)
        elif query_type == "depth":
            selected_nodes, selected_facts, path = self._retrieve_depth(query, anchor_id, top_k)
        elif query_type == "temporal":
            if self.enable_temporal_expansion and self._should_use_temporal_expansion(query, anchor_id):
                selected_nodes, selected_facts, path = self._retrieve_temporal_expanded(query, anchor_id, top_k)
            else:
                selected_nodes, selected_facts, path = self._retrieve_temporal(query, anchor_id, top_k)
        else:
            selected_nodes, selected_facts, path = self._retrieve_fact(query, anchor_id, top_k)

        if self.enable_global_fact_fallback:
            selected_nodes, selected_facts = self._apply_hybrid_budget(
                query, selected_nodes, selected_facts, top_k
            )
        else:
            selected_nodes, selected_facts = self._apply_budget(selected_nodes, selected_facts)
        context_tokens = self._context_tokens(selected_nodes, selected_facts)
        latency_ms = (time.perf_counter() - start) * 1000.0
        expected = set(evidence_ids or [])
        evidence_hit = bool(expected.intersection(selected_facts)) if expected else None

        return DynamicRetrievalResult(
            query=query,
            query_type=query_type,
            retrieved_fact_ids=selected_facts,
            expanded_node_ids=selected_nodes,
            expanded_path=path,
            context_tokens=context_tokens,
            latency_ms=latency_ms,
            expanded_depth=max((self.memory.nodes[nid].level for nid in selected_nodes), default=0),
            expanded_breadth=self._expanded_breadth(selected_nodes),
            evidence_hit=evidence_hit,
        )

    @classmethod
    def detect_query_type(cls, query: str) -> str:
        query_lower = query.lower()
        if any(marker in query_lower for marker in cls.TEMPORAL_MARKERS):
            return "temporal"
        if any(marker in query_lower for marker in cls.BREADTH_MARKERS):
            return "breadth"
        if any(marker in query_lower for marker in cls.DEPTH_MARKERS):
            return "depth"
        return "fact"

    def _build_indexes(self) -> None:
        node_ids: List[str] = []
        node_docs: List[str] = []
        fact_ids: List[str] = []
        fact_docs: List[str] = []
        for node_id, node in self.memory.nodes.items():
            if node.node_type == "fact":
                fact_ids.append(node_id)
                fact_docs.append(self._fact_search_text(node))
            else:
                node_ids.append(node_id)
                node_docs.append(" ".join([node.title, node.summary, node.text]))
        self.node_index.fit(node_ids, node_docs)
        self.fact_index.fit(fact_ids, fact_docs)

    def _retrieve_fact(
        self,
        query: str,
        anchor_id: str,
        top_k: int,
    ) -> Tuple[List[str], List[str], List[str]]:
        facts = [fact_id for fact_id, _ in self.fact_index.search(query, top_k=top_k)]
        nodes: List[str] = []
        for fact_id in facts[:top_k]:
            parent = self.memory.nodes[fact_id].parent_id
            if parent and parent not in nodes:
                nodes.append(parent)
        path = self._path(anchor_id)
        return nodes[:2], facts, path

    def _retrieve_breadth(
        self,
        query: str,
        anchor_id: str,
        top_k: int,
    ) -> Tuple[List[str], List[str], List[str]]:
        anchor = self.memory.nodes[anchor_id]
        parent_id = anchor_id if anchor.children else anchor.parent_id or self.memory.root_id
        children = [
            child_id for child_id in self.memory.nodes[parent_id].children
            if self.memory.nodes[child_id].node_type != "fact"
        ]
        if len(children) < 2 and anchor.parent_id:
            parent_id = anchor.parent_id
            children = [
                child_id for child_id in self.memory.nodes[parent_id].children
                if self.memory.nodes[child_id].node_type != "fact"
            ]
        ranked_children = self._rank_nodes(query, children)[: max(top_k, 3)]
        facts = self._top_facts_from_nodes(query, ranked_children or [parent_id], top_k=top_k)
        return [parent_id] + ranked_children, facts, self._path(parent_id)

    def _retrieve_depth(
        self,
        query: str,
        anchor_id: str,
        top_k: int,
    ) -> Tuple[List[str], List[str], List[str]]:
        cursor = anchor_id
        path = self._path(cursor)
        selected = list(path)
        while True:
            children = [
                child_id for child_id in self.memory.nodes[cursor].children
                if self.memory.nodes[child_id].node_type != "fact"
            ]
            if not children:
                break
            ranked = self._rank_nodes(query, children)
            if not ranked:
                break
            cursor = ranked[0]
            selected.append(cursor)
            path.append(cursor)
            if len(selected) >= self.memory.nodes[cursor].level + top_k:
                break
        facts = self._top_facts_from_nodes(query, [cursor], top_k=top_k)
        if len(facts) < top_k:
            more = [fact_id for fact_id, _ in self.fact_index.search(query, top_k=top_k)]
            facts.extend(fact_id for fact_id in more if fact_id not in facts)
        return selected, facts[:top_k], path

    def _retrieve_temporal(
        self,
        query: str,
        anchor_id: str,
        top_k: int,
    ) -> Tuple[List[str], List[str], List[str]]:
        topic_id = self._nearest_topic(anchor_id)
        descendants = self._descendant_nonfact_nodes(topic_id)
        ranked = set(self._rank_nodes(query, descendants)[: max(top_k, 4)])
        chronological = sorted(
            [node_id for node_id in descendants if node_id in ranked or self.memory.nodes[node_id].fact_ids],
            key=lambda node_id: self.memory.nodes[node_id].metadata.get("start", 0),
        )
        selected_nodes = [topic_id] + chronological[: max(top_k, 4)]
        facts = self._top_facts_from_nodes(query, selected_nodes, top_k=top_k)
        global_facts = [fact_id for fact_id, _ in self.fact_index.search(query, top_k=max(top_k * 2, 10))]
        facts = global_facts + [fact_id for fact_id in facts if fact_id not in global_facts]
        if len(facts) < top_k:
            ordered_facts = self._facts_under(topic_id)
            facts.extend(fact_id for fact_id in ordered_facts if fact_id not in facts)
        return selected_nodes, facts[:top_k], self._path(topic_id)

    def _retrieve_temporal_expanded(
        self,
        query: str,
        anchor_id: str,
        top_k: int,
    ) -> Tuple[List[str], List[str], List[str]]:
        topic_id = self._best_temporal_anchor(query, anchor_id)
        descendants = self._descendant_nonfact_nodes(topic_id)
        if not descendants:
            descendants = [topic_id]

        ranked_nodes = self._rank_nodes(query, descendants)
        ranked_set = set(ranked_nodes[: max(self.temporal_max_nodes * 3, self.temporal_max_nodes)])
        candidate_nodes = [
            node_id for node_id in descendants
            if node_id in ranked_set or self._node_has_query_terms(query, node_id)
        ]
        if not candidate_nodes:
            candidate_nodes = ranked_nodes[: self.temporal_max_nodes]

        selected_nodes = sorted(
            candidate_nodes[: max(self.temporal_max_nodes * 2, self.temporal_max_nodes)],
            key=lambda node_id: self._node_time_key(node_id),
        )[: self.temporal_max_nodes]

        selected_facts: List[str] = []
        for node_id in selected_nodes:
            node_facts = self._rank_facts(query, self._facts_under(node_id))
            node_facts = sorted(
                node_facts[: max(self.temporal_facts_per_node * 3, self.temporal_facts_per_node)],
                key=lambda fact_id: self._fact_time_key(fact_id),
            )[: self.temporal_facts_per_node]
            selected_facts.extend(fact_id for fact_id in node_facts if fact_id not in selected_facts)

        if len(selected_facts) < top_k:
            global_temporal = [
                fact_id for fact_id, _ in self.fact_index.search(query, top_k=max(top_k * 4, 30))
                if fact_id not in selected_facts
            ]
            selected_facts.extend(global_temporal)

        selected_facts = sorted(dict.fromkeys(selected_facts), key=self._fact_time_key)
        return [topic_id] + selected_nodes, selected_facts[: max(top_k, self.temporal_max_nodes * self.temporal_facts_per_node)], self._path(topic_id)

    def _should_use_temporal_expansion(self, query: str, anchor_id: str) -> bool:
        query_lower = query.lower()
        has_temporal_marker = any(
            marker in query_lower for marker in self.TEMPORAL_HIGH_CONFIDENCE_MARKERS
        )
        if not has_temporal_marker:
            return False

        topic_id = self._best_temporal_anchor(query, anchor_id)
        descendants = self._descendant_nonfact_nodes(topic_id)
        if not descendants:
            descendants = [topic_id]
        ranked_nodes = self._rank_nodes(query, descendants)
        candidate_nodes = ranked_nodes[: max(self.temporal_max_nodes * 3, 8)]
        time_buckets = set()
        for node_id in candidate_nodes:
            facts = self._facts_under(node_id)
            for fact_id in facts[: max(self.temporal_facts_per_node * 4, 4)]:
                time_index = int(self.memory.nodes[fact_id].metadata.get("time_index", 0))
                time_buckets.add(time_index // 1000 if time_index else self.memory.nodes[fact_id].metadata.get("start", 0))
                if len(time_buckets) >= 2:
                    return True
        return False

    def _rank_nodes(self, query: str, node_ids: Sequence[str]) -> List[str]:
        if not node_ids:
            return []
        query_vec = self.node_index.vectorize(query)
        scored = []
        for node_id in node_ids:
            node = self.memory.nodes[node_id]
            doc = " ".join([node.title, node.summary, node.text])
            score = float(self.node_index.vectorize(doc) @ query_vec)
            scored.append((node_id, score))
        scored.sort(key=lambda item: (-item[1], self.memory.nodes[item[0]].metadata.get("start", 0)))
        return [node_id for node_id, _ in scored]

    def _rank_facts(self, query: str, fact_ids: Sequence[str]) -> List[str]:
        if not fact_ids:
            return []
        query_vec = self.fact_index.vectorize(query)
        scored = []
        for fact_id in fact_ids:
            node = self.memory.nodes[fact_id]
            score = float(self.fact_index.vectorize(self._fact_search_text(node)) @ query_vec)
            scored.append((fact_id, score))
        scored.sort(key=lambda item: (-item[1], self._fact_time_key(item[0])))
        return [fact_id for fact_id, _ in scored]

    def _top_facts_from_nodes(self, query: str, node_ids: Sequence[str], top_k: int) -> List[str]:
        candidate_facts: List[str] = []
        for node_id in node_ids:
            candidate_facts.extend(self._facts_under(node_id))
        if not candidate_facts:
            return []
        ranked_global = [fact_id for fact_id, _ in self.fact_index.search(query, top_k=max(top_k * 3, 10))]
        selected = [fact_id for fact_id in ranked_global if fact_id in set(candidate_facts)]
        selected.extend(fact_id for fact_id in candidate_facts if fact_id not in selected)
        return selected[:top_k]

    def _facts_under(self, node_id: str) -> List[str]:
        facts: List[str] = []
        node = self.memory.nodes[node_id]
        facts.extend(node.fact_ids)
        for child_id in node.children:
            if self.memory.nodes[child_id].node_type == "fact":
                if child_id not in facts:
                    facts.append(child_id)
            else:
                facts.extend(self._facts_under(child_id))
        return sorted(
            dict.fromkeys(facts),
            key=lambda fact_id: (
                self.memory.nodes[fact_id].metadata.get("start", 0),
                self.memory.nodes[fact_id].metadata.get("order", 0),
            ),
        )

    def _descendant_nonfact_nodes(self, node_id: str) -> List[str]:
        descendants: List[str] = []
        for child_id in self.memory.nodes[node_id].children:
            if self.memory.nodes[child_id].node_type == "fact":
                continue
            descendants.append(child_id)
            descendants.extend(self._descendant_nonfact_nodes(child_id))
        return descendants

    def _nearest_topic(self, node_id: str) -> str:
        path = self._path(node_id)
        for candidate in reversed(path):
            if self.memory.nodes[candidate].node_type == "topic":
                return candidate
        return node_id if self.memory.nodes[node_id].node_type == "topic" else self.memory.root_id

    def _best_temporal_anchor(self, query: str, anchor_id: str) -> str:
        candidates = [
            node_id for node_id, _ in self.node_index.search(query, top_k=12)
            if self.memory.nodes[node_id].node_type in {"topic", "subtopic"}
        ]
        if candidates:
            return candidates[0]
        return self._nearest_topic(anchor_id)

    def _path(self, node_id: str) -> List[str]:
        path: List[str] = []
        cursor: Optional[str] = node_id
        while cursor:
            path.append(cursor)
            cursor = self.memory.nodes[cursor].parent_id if cursor in self.memory.nodes else None
        return list(reversed(path))

    def _apply_budget(
        self,
        node_ids: List[str],
        fact_ids: List[str],
    ) -> Tuple[List[str], List[str]]:
        return self._apply_budget_limit(node_ids, fact_ids, self.max_context_tokens)

    def _apply_budget_limit(
        self,
        node_ids: List[str],
        fact_ids: List[str],
        token_limit: int,
    ) -> Tuple[List[str], List[str]]:
        kept_nodes: List[str] = []
        kept_facts: List[str] = []
        tokens = 0
        for node_id in dict.fromkeys(node_ids):
            node_tokens = estimate_tokens(self.memory.nodes[node_id].summary)
            if tokens + node_tokens <= token_limit:
                kept_nodes.append(node_id)
                tokens += node_tokens
        for fact_id in dict.fromkeys(fact_ids):
            fact_tokens = estimate_tokens(self.memory.nodes[fact_id].text)
            if tokens + fact_tokens <= token_limit:
                kept_facts.append(fact_id)
                tokens += fact_tokens
        return kept_nodes, kept_facts

    def _apply_hybrid_budget(
        self,
        query: str,
        node_ids: List[str],
        fact_ids: List[str],
        top_k: int,
    ) -> Tuple[List[str], List[str]]:
        dynamic_limit = int(self.max_context_tokens * (1.0 - self.fallback_ratio))
        fallback_limit = max(0, self.max_context_tokens - dynamic_limit)
        kept_nodes, kept_dynamic_facts = self._apply_budget_limit(node_ids, fact_ids, dynamic_limit)

        used_tokens = self._context_tokens(kept_nodes, kept_dynamic_facts)
        fallback_budget = max(0, min(fallback_limit, self.max_context_tokens - used_tokens))
        fallback_facts: List[str] = []
        fallback_tokens = 0
        dynamic_set = set(kept_dynamic_facts)
        for fact_id, _score in self.fact_index.search(query, top_k=max(top_k * 8, 50)):
            if fact_id in dynamic_set or fact_id in fallback_facts:
                continue
            fact_tokens = estimate_tokens(self.memory.nodes[fact_id].text)
            if fallback_tokens + fact_tokens <= fallback_budget:
                fallback_facts.append(fact_id)
                fallback_tokens += fact_tokens

        merged_facts = self._merge_fact_scores(query, kept_dynamic_facts, fallback_facts)
        return self._apply_budget_limit(kept_nodes, merged_facts, self.max_context_tokens)

    def _merge_fact_scores(
        self,
        query: str,
        dynamic_facts: Sequence[str],
        fallback_facts: Sequence[str],
    ) -> List[str]:
        query_vec = self.fact_index.vectorize(query)
        scores: Dict[str, float] = {}
        for rank, fact_id in enumerate(dynamic_facts):
            scores[fact_id] = max(scores.get(fact_id, 0.0), 1.0 + 1.0 / (rank + 1))
        for rank, fact_id in enumerate(fallback_facts):
            sim = float(self.fact_index.vectorize(self._fact_search_text(self.memory.nodes[fact_id])) @ query_vec)
            scores[fact_id] = max(scores.get(fact_id, 0.0), sim + 0.5 / (rank + 1))
        return [
            fact_id for fact_id, _ in sorted(
                scores.items(),
                key=lambda item: (-item[1], self._fact_time_key(item[0])),
            )
        ]

    def _context_tokens(self, node_ids: Sequence[str], fact_ids: Sequence[str]) -> int:
        text = "\n".join(
            [self.memory.nodes[node_id].summary for node_id in node_ids]
            + [self.memory.nodes[fact_id].text for fact_id in fact_ids]
        )
        return estimate_tokens(text)

    def _fact_search_text(self, node: DynamicMemoryNode) -> str:
        metadata = node.metadata
        parts = [
            metadata.get("speaker", ""),
            str(metadata.get("time_index", "")),
            metadata.get("episode_id", ""),
            " ".join(metadata.get("keywords", []) or []),
            " ".join(metadata.get("entities", []) or []),
            " ".join(metadata.get("source_turn_ids", []) or []),
            metadata.get("raw_text", node.text),
        ]
        return " ".join(str(part) for part in parts if part)

    def _fact_time_key(self, fact_id: str) -> Tuple[int, int]:
        node = self.memory.nodes[fact_id]
        return (
            int(node.metadata.get("time_index", node.metadata.get("start", 0))),
            int(node.metadata.get("order", 0)),
        )

    def _node_time_key(self, node_id: str) -> Tuple[int, int]:
        node = self.memory.nodes[node_id]
        return (
            int(node.metadata.get("start", 0)),
            int(node.metadata.get("end", 0)),
        )

    def _node_has_query_terms(self, query: str, node_id: str) -> bool:
        query_terms = set(SimpleTfidfIndex.tokenize(query))
        if not query_terms:
            return False
        node_terms = set(SimpleTfidfIndex.tokenize(self.memory.nodes[node_id].text))
        return bool(query_terms.intersection(node_terms))

    def _expanded_breadth(self, node_ids: Sequence[str]) -> int:
        by_parent: Dict[Optional[str], int] = defaultdict(int)
        for node_id in node_ids:
            by_parent[self.memory.nodes[node_id].parent_id] += 1
        return max(by_parent.values(), default=0)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for mixed Chinese/English demo text."""
    ascii_words = re.findall(r"[A-Za-z0-9_]+", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    other = max(0, len(text) - sum(len(word) for word in ascii_words) - len(cjk_chars))
    return len(ascii_words) + max(1, math.ceil(len(cjk_chars) / 2)) + math.ceil(other / 4)
