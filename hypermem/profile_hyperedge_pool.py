"""
User-profile guided dynamic hyperedge pool for long-term conversational memory.

This module adds a fast personalized retrieval channel on top of the original
Topic / Episode / Fact memory base.  It does not call LLMs or external APIs.
The pool keeps high-utility profile hyperedges such as user preferences, goals,
habits, current state, domain knowledge and temporal evolution chains.

Core idea:
    base memory nodes  ->  reward-guided user-profile hyperedge pool
    query              ->  profile fast channel first
                       ->  fallback to original facts/path when insufficient
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    """Small multilingual tokenizer for demo/eval without extra dependencies."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def keyword_overlap(a: str | Sequence[str], b: str | Sequence[str]) -> float:
    if isinstance(a, str):
        a_set = set(tokenize(a))
    else:
        a_set = {str(x).lower() for x in a if str(x).strip()}
    if isinstance(b, str):
        b_set = set(tokenize(b))
    else:
        b_set = {str(x).lower() for x in b if str(x).strip()}
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / max(1, len(a_set | b_set))


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class NodeType(str, Enum):
    TOPIC = "topic"
    EPISODE = "episode"
    FACT = "fact"


class ProfileHyperedgeType(str, Enum):
    PREFERENCE = "preference"
    GOAL = "goal"
    HABIT = "habit"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    CURRENT_STATE = "current_state"
    TEMPORAL_EVOLUTION = "temporal_evolution"
    EVIDENCE_GROUP = "evidence_group"
    SUPERSEDE = "supersede"
    OTHER = "other"


@dataclass
class MemoryNode:
    """A lightweight view of Topic / Episode / Fact nodes used by the pool."""

    node_id: str
    node_type: NodeType
    content: str
    keywords: List[str] = field(default_factory=list)
    timestamp: Optional[float] = None
    topic_id: str = ""
    episode_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        return " ".join([self.content, " ".join(self.keywords)]).strip()


@dataclass
class ProfileHyperedge:
    """A reusable personalized high-order memory unit."""

    edge_id: str
    edge_type: ProfileHyperedgeType
    summary: str
    member_node_ids: List[str]
    keywords: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    time_span: Tuple[Optional[float], Optional[float]] = (None, None)

    # Reward-maintained utility signals.
    profile_score: float = 0.5
    utility_score: float = 0.5
    freshness_score: float = 0.5
    coherence_score: float = 0.5
    access_count: int = 0
    hit_count: int = 0
    failure_count: int = 0
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def score(self, query: str, now: Optional[float] = None) -> float:
        """Score edge for profile fast-channel retrieval."""
        if self.status != "active":
            return 0.0
        text_score = keyword_overlap(query, self.summary) * 0.55
        kw_score = keyword_overlap(query, self.keywords) * 0.25
        utility = self.utility_score * 0.12
        profile = self.profile_score * 0.05
        freshness = self.freshness_score * 0.03
        return clamp(text_score + kw_score + utility + profile + freshness)

    def apply_feedback(self, hit: bool, answer_quality: float = 1.0, lr: float = 0.15) -> None:
        """Reward update: useful edges are promoted; noisy edges decay softly."""
        self.access_count += 1
        if hit:
            self.hit_count += 1
            target = clamp(0.6 + 0.4 * answer_quality)
            self.utility_score = clamp((1 - lr) * self.utility_score + lr * target)
            self.profile_score = clamp(self.profile_score + lr * 0.25)
        else:
            self.failure_count += 1
            self.utility_score = clamp(self.utility_score - lr * 0.35)
            self.profile_score = clamp(self.profile_score - lr * 0.15)
        if self.failure_count >= 5 and self.hit_count == 0:
            self.status = "inactive"
        self.updated_at = time.time()


@dataclass
class RetrievalResult:
    query: str
    channel: str
    evidence_nodes: List[MemoryNode]
    hyperedges: List[ProfileHyperedge]
    score: float
    tokens: int
    sufficient: bool
    fallback_used: bool = False

    def evidence_text(self) -> str:
        return "\n".join(f"- {n.content}" for n in self.evidence_nodes)


class UserProfileHyperedgePool:
    """
    Reward-guided profile hyperedge pool.

    The pool is a fast personalized channel.  It should not replace the full
    memory base.  When evidence is insufficient, callers should fallback to the
    original HyperMem path or global fact retrieval.
    """

    def __init__(self, user_id: str = "default_user") -> None:
        self.user_id = user_id
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: Dict[str, ProfileHyperedge] = {}
        self._edge_counter = 0

    # ---------------------------- ingestion ----------------------------
    def add_node(self, node: MemoryNode) -> None:
        self.nodes[node.node_id] = node

    def add_fact(
        self,
        content: str,
        node_id: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        timestamp: Optional[float] = None,
        topic_id: str = "",
        episode_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        promote: bool = True,
    ) -> MemoryNode:
        node_id = node_id or f"fact_{len(self.nodes) + 1:05d}"
        kw = keywords or self._extract_keywords(content)
        node = MemoryNode(
            node_id=node_id,
            node_type=NodeType.FACT,
            content=content,
            keywords=kw,
            timestamp=timestamp or time.time(),
            topic_id=topic_id,
            episode_ids=episode_ids or [],
            metadata=metadata or {},
        )
        self.add_node(node)
        if promote:
            self.promote_node(node)
        return node

    def promote_node(self, node: MemoryNode) -> ProfileHyperedge:
        """Create or attach node to the best profile hyperedge."""
        edge_type = self.classify_profile_type(node.content)
        candidates = [e for e in self.edges.values() if e.edge_type == edge_type and e.status == "active"]
        best: Optional[ProfileHyperedge] = None
        best_score = 0.0
        for edge in candidates:
            s = max(keyword_overlap(node.content, edge.summary), keyword_overlap(node.keywords, edge.keywords))
            if s > best_score:
                best_score, best = s, edge
        if best is not None and best_score >= 0.08:
            self.attach_node(best.edge_id, node.node_id)
            return best
        return self.create_edge(edge_type, [node.node_id], summary=self._edge_summary(edge_type, [node]))

    def create_edge(
        self,
        edge_type: ProfileHyperedgeType,
        member_node_ids: List[str],
        summary: str,
        keywords: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProfileHyperedge:
        self._edge_counter += 1
        edge_id = f"profile_edge_{self._edge_counter:05d}"
        member_nodes = [self.nodes[nid] for nid in member_node_ids if nid in self.nodes]
        ts = [n.timestamp for n in member_nodes if n.timestamp is not None]
        kw = keywords or self._extract_keywords(" ".join([summary] + [n.text() for n in member_nodes]))
        edge = ProfileHyperedge(
            edge_id=edge_id,
            edge_type=edge_type,
            summary=summary,
            member_node_ids=list(dict.fromkeys(member_node_ids)),
            keywords=kw,
            time_span=(min(ts) if ts else None, max(ts) if ts else None),
            profile_score=self._initial_profile_score(edge_type),
            utility_score=0.55,
            freshness_score=0.7,
            coherence_score=self._coherence(member_nodes),
            metadata=metadata or {},
        )
        self.edges[edge_id] = edge
        return edge

    def attach_node(self, edge_id: str, node_id: str) -> None:
        edge = self.edges[edge_id]
        if node_id not in edge.member_node_ids:
            edge.member_node_ids.append(node_id)
        node = self.nodes[node_id]
        edge.summary = self._edge_summary(edge.edge_type, [self.nodes[n] for n in edge.member_node_ids if n in self.nodes])
        edge.keywords = self._extract_keywords(edge.summary + " " + " ".join(node.keywords))
        ts = [self.nodes[n].timestamp for n in edge.member_node_ids if n in self.nodes and self.nodes[n].timestamp]
        edge.time_span = (min(ts) if ts else None, max(ts) if ts else None)
        edge.coherence_score = self._coherence([self.nodes[n] for n in edge.member_node_ids if n in self.nodes])
        edge.freshness_score = clamp(edge.freshness_score + 0.08)
        edge.updated_at = time.time()

    # ---------------------------- retrieval ----------------------------
    def retrieve_edges(self, query: str, top_k: int = 3, min_score: float = 0.03) -> List[Tuple[ProfileHyperedge, float]]:
        scored = [(edge, edge.score(query)) for edge in self.edges.values()]
        scored = [(e, s) for e, s in scored if s >= min_score]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve_fast_channel(
        self,
        query: str,
        top_k_edges: int = 3,
        max_tokens: int = 450,
        sufficiency_threshold: float = 0.18,
        fallback_nodes: Optional[Sequence[MemoryNode]] = None,
    ) -> RetrievalResult:
        """Retrieve profile hyperedges first, then fallback when insufficient."""
        edge_scores = self.retrieve_edges(query, top_k=top_k_edges)
        selected_edges = [e for e, _ in edge_scores]
        selected_nodes = self._expand_edges(selected_edges, query=query, max_tokens=max_tokens)
        score = sum(s for _, s in edge_scores) / max(1, len(edge_scores))
        sufficient = score >= sufficiency_threshold and bool(selected_nodes)
        fallback_used = False

        if not sufficient and fallback_nodes:
            fallback_used = True
            selected_nodes = self._rank_nodes(query, fallback_nodes)[: max(1, max_tokens // 30)]
            score = max(score, keyword_overlap(query, " ".join(n.text() for n in selected_nodes)))
            sufficient = bool(selected_nodes)

        for edge in selected_edges:
            edge.access_count += 1

        return RetrievalResult(
            query=query,
            channel="profile_fast_channel" if not fallback_used else "profile_fast_channel+fallback",
            evidence_nodes=selected_nodes,
            hyperedges=selected_edges,
            score=score,
            tokens=self._estimate_tokens(selected_nodes),
            sufficient=sufficient,
            fallback_used=fallback_used,
        )

    def update_rewards(self, edge_ids: Sequence[str], hit: bool, answer_quality: float = 1.0) -> None:
        for edge_id in edge_ids:
            if edge_id in self.edges:
                self.edges[edge_id].apply_feedback(hit=hit, answer_quality=answer_quality)

    # ---------------------------- profile helpers ----------------------------
    def classify_profile_type(self, text: str) -> ProfileHyperedgeType:
        t = (text or "").lower()
        if any(x in t for x in ["喜欢", "偏好", "希望", "习惯", "prefer", "style", "usually", "常常"]):
            return ProfileHyperedgeType.PREFERENCE
        if any(x in t for x in ["目标", "投稿", "aaai", "acl", "goal", "aim", "target"]):
            return ProfileHyperedgeType.GOAL
        if any(x in t for x in ["流程", "workflow", "codex", "prompt", "跑实验", "实验结果"]):
            return ProfileHyperedgeType.HABIT
        if any(x in t for x in ["当前", "现在", "最终", "latest", "current", "finally", "决定"]):
            return ProfileHyperedgeType.CURRENT_STATE
        if any(x in t for x in ["后来", "之前", "变化", "演化", "earlier", "later", "evolve", "change"]):
            return ProfileHyperedgeType.TEMPORAL_EVOLUTION
        if any(x in t for x in ["hypermem", "memory", "locomo", "超图", "超边", "强化学习", "retrieval", "rag"]):
            return ProfileHyperedgeType.DOMAIN_KNOWLEDGE
        return ProfileHyperedgeType.OTHER

    def build_from_texts(self, texts: Iterable[str], user_id: Optional[str] = None) -> None:
        if user_id:
            self.user_id = user_id
        for i, text in enumerate(texts):
            cleaned = str(text).strip()
            if cleaned:
                self.add_fact(cleaned, node_id=f"fact_{i + 1:05d}", timestamp=float(i + 1))

    def export_profile(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "active_edges": sum(1 for e in self.edges.values() if e.status == "active"),
            "edge_type_counts": self.edge_type_counts(),
            "top_edges": [
                {
                    "edge_id": e.edge_id,
                    "type": e.edge_type.value,
                    "summary": e.summary,
                    "utility_score": round(e.utility_score, 4),
                    "profile_score": round(e.profile_score, 4),
                    "members": e.member_node_ids,
                }
                for e in sorted(self.edges.values(), key=lambda x: x.utility_score, reverse=True)[:10]
            ],
        }

    def edge_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for edge in self.edges.values():
            counts[edge.edge_type.value] = counts.get(edge.edge_type.value, 0) + 1
        return counts

    # ---------------------------- persistence ----------------------------
    def save(self, path: str | Path) -> None:
        data = {
            "user_id": self.user_id,
            "edge_counter": self._edge_counter,
            "nodes": {k: self._jsonify(asdict(v)) for k, v in self.nodes.items()},
            "edges": {k: self._jsonify(asdict(v)) for k, v in self.edges.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "UserProfileHyperedgePool":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        pool = cls(user_id=data.get("user_id", "default_user"))
        pool._edge_counter = int(data.get("edge_counter", 0))
        for node_id, raw in data.get("nodes", {}).items():
            raw["node_type"] = NodeType(raw["node_type"])
            pool.nodes[node_id] = MemoryNode(**raw)
        for edge_id, raw in data.get("edges", {}).items():
            raw["edge_type"] = ProfileHyperedgeType(raw["edge_type"])
            raw["time_span"] = tuple(raw.get("time_span", (None, None)))
            pool.edges[edge_id] = ProfileHyperedge(**raw)
        return pool

    # ---------------------------- internals ----------------------------
    def _expand_edges(self, edges: Sequence[ProfileHyperedge], query: str, max_tokens: int) -> List[MemoryNode]:
        candidates: List[MemoryNode] = []
        for edge in edges:
            candidates.extend(self.nodes[nid] for nid in edge.member_node_ids if nid in self.nodes)
        ranked = self._rank_nodes(query, candidates)
        selected: List[MemoryNode] = []
        tokens = 0
        seen = set()
        for node in ranked:
            if node.node_id in seen:
                continue
            t = max(1, len(tokenize(node.content)))
            if selected and tokens + t > max_tokens:
                break
            selected.append(node)
            seen.add(node.node_id)
            tokens += t
        return selected

    def _rank_nodes(self, query: str, nodes: Sequence[MemoryNode]) -> List[MemoryNode]:
        return sorted(nodes, key=lambda n: keyword_overlap(query, n.text()), reverse=True)

    def _estimate_tokens(self, nodes: Sequence[MemoryNode]) -> int:
        return sum(max(1, len(tokenize(n.content))) for n in nodes)

    def _extract_keywords(self, text: str, max_keywords: int = 12) -> List[str]:
        toks = tokenize(text)
        stop = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for", "with", "这个", "那个"}
        counts: Dict[str, int] = {}
        for tok in toks:
            if tok in stop:
                continue
            counts[tok] = counts.get(tok, 0) + 1
        return [w for w, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_keywords]]

    def _edge_summary(self, edge_type: ProfileHyperedgeType, nodes: Sequence[MemoryNode]) -> str:
        contents = [n.content.strip() for n in nodes if n.content.strip()]
        if not contents:
            return f"{edge_type.value} hyperedge"
        joined = "；".join(contents[:4])
        if len(contents) > 4:
            joined += f"；... (+{len(contents)-4} more)"
        return f"[{edge_type.value}] {joined}"

    def _initial_profile_score(self, edge_type: ProfileHyperedgeType) -> float:
        if edge_type in {ProfileHyperedgeType.PREFERENCE, ProfileHyperedgeType.GOAL, ProfileHyperedgeType.CURRENT_STATE}:
            return 0.75
        if edge_type in {ProfileHyperedgeType.HABIT, ProfileHyperedgeType.DOMAIN_KNOWLEDGE, ProfileHyperedgeType.TEMPORAL_EVOLUTION}:
            return 0.65
        return 0.45

    def _coherence(self, nodes: Sequence[MemoryNode]) -> float:
        if len(nodes) <= 1:
            return 0.8
        vals = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                vals.append(keyword_overlap(nodes[i].text(), nodes[j].text()))
        return clamp(sum(vals) / max(1, len(vals)) + 0.35)

    def _jsonify(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, dict):
            return {k: self._jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._jsonify(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._jsonify(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return 0.0
        return obj
