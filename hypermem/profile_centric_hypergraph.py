"""Profile-centric hypergraph memory.

New main flow:
    conversation facts -> user-profile hyperedges -> reward utility learning
    -> profile hyperedge retrieval -> fact evidence selection.

This module intentionally implements the lightweight bandit-style variant:
- no LLM calls;
- no external embedding service;
- lexical overlap is used as a local embedding proxy for smoke tests;
- each profile hyperedge owns a learnable utility score.

For formal experiments, replace ``keyword_overlap`` with a real embedding
similarity while keeping the same retrieval/update interfaces.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
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


def estimate_tokens(items: str | Sequence[str]) -> int:
    if isinstance(items, str):
        return max(1, len(tokenize(items)))
    return sum(max(1, len(tokenize(str(x)))) for x in items)


class ProfileEdgeType(str, Enum):
    PREFERENCE = "preference"
    GOAL = "goal"
    HABIT = "habit"
    CURRENT_STATE = "current_state"
    TEMPORAL_EVOLUTION = "temporal_evolution"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    TOOL_USAGE = "tool_usage"
    WRITING_STYLE = "writing_style"
    RESEARCH_FOCUS = "research_focus"
    AUTO_DISCOVERED = "auto_discovered"
    OTHER = "other"


@dataclass
class ProfileFact:
    fact_id: str
    content: str
    keywords: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        return " ".join([self.content, " ".join(self.keywords)]).strip()


@dataclass
class ProfileHyperedgeUnit:
    """A first-class user-profile hyperedge with learnable utility."""

    edge_id: str
    edge_type: ProfileEdgeType
    summary: str
    member_fact_ids: List[str]
    keywords: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    utility_score: float = 0.50
    freshness_score: float = 0.50
    stability_score: float = 0.50
    confidence_score: float = 0.50
    coherence_score: float = 0.50

    access_count: int = 0
    hit_count: int = 0
    failure_count: int = 0
    total_reward: float = 0.0
    last_reward: float = 0.0
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def token_cost(self, facts: Dict[str, ProfileFact]) -> int:
        return estimate_tokens([facts[fid].content for fid in self.member_fact_ids if fid in facts])

    def relevance(self, query: str) -> float:
        return 0.65 * keyword_overlap(query, self.summary) + 0.35 * keyword_overlap(query, self.keywords)

    def type_match(self, query_type: ProfileEdgeType) -> float:
        if query_type == self.edge_type:
            return 1.0
        compatible = {
            ProfileEdgeType.GOAL: {ProfileEdgeType.RESEARCH_FOCUS, ProfileEdgeType.CURRENT_STATE},
            ProfileEdgeType.RESEARCH_FOCUS: {ProfileEdgeType.GOAL, ProfileEdgeType.DOMAIN_KNOWLEDGE},
            ProfileEdgeType.PREFERENCE: {ProfileEdgeType.WRITING_STYLE, ProfileEdgeType.HABIT},
            ProfileEdgeType.HABIT: {ProfileEdgeType.TOOL_USAGE, ProfileEdgeType.PREFERENCE},
            ProfileEdgeType.CURRENT_STATE: {ProfileEdgeType.TEMPORAL_EVOLUTION, ProfileEdgeType.GOAL},
        }
        return 0.35 if query_type in compatible and self.edge_type in compatible[query_type] else 0.0

    def score(
        self,
        query: str,
        facts: Dict[str, ProfileFact],
        query_type: ProfileEdgeType,
        use_utility: bool = True,
        min_relevance: float = 0.01,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        if self.status != "active":
            return 0.0, {}
        weights = weights or {}
        relevance = self.relevance(query)
        token_penalty = min(1.0, self.token_cost(facts) / 900.0)
        type_match = self.type_match(query_type)
        if relevance < min_relevance:
            return 0.0, {
                "relevance": round(relevance, 6),
                "utility": round(self.utility_score if use_utility else 0.0, 6),
                "freshness": round(self.freshness_score, 6),
                "stability": round(self.stability_score, 6),
                "confidence": round(self.confidence_score, 6),
                "type_match": round(type_match, 6),
                "token_penalty": round(token_penalty, 6),
            }
        utility = self.utility_score if use_utility else 0.0
        score = (
            weights.get("relevance", 0.58) * relevance
            + weights.get("utility", 0.18) * utility
            + weights.get("freshness", 0.07) * self.freshness_score
            + weights.get("stability", 0.07) * self.stability_score
            + weights.get("confidence", 0.05) * self.confidence_score
            + weights.get("type_match", 0.08) * type_match
            - weights.get("token_cost", 0.03) * token_penalty
        )
        return clamp(score), {
            "relevance": round(relevance, 6),
            "utility": round(utility, 6),
            "freshness": round(self.freshness_score, 6),
            "stability": round(self.stability_score, 6),
            "confidence": round(self.confidence_score, 6),
            "type_match": round(type_match, 6),
            "token_penalty": round(token_penalty, 6),
        }

    def update_utility(self, reward: float, hit: bool, lr: float = 0.18) -> None:
        reward = clamp(reward, lo=-1.0, hi=1.0)
        normalized = clamp((reward + 1.0) / 2.0)
        self.utility_score = clamp((1.0 - lr) * self.utility_score + lr * normalized)
        self.total_reward += reward
        self.last_reward = reward
        if hit:
            self.hit_count += 1
            self.stability_score = clamp(self.stability_score + lr * 0.10)
            self.confidence_score = clamp(self.confidence_score + lr * 0.08)
        else:
            self.failure_count += 1
            self.stability_score = clamp(self.stability_score - lr * 0.06)
            self.confidence_score = clamp(self.confidence_score - lr * 0.05)
        if self.failure_count >= 8 and self.hit_count == 0:
            self.status = "inactive"
        self.updated_at = time.time()


@dataclass
class ProfileRetrievalResult:
    query: str
    channel: str
    selected_edges: List[ProfileHyperedgeUnit]
    selected_facts: List[ProfileFact]
    score: float
    tokens: int
    fallback_used: bool
    sufficient: bool
    debug_scores: List[Dict[str, Any]] = field(default_factory=list)

    def evidence_text(self) -> str:
        return "\n".join(f"- {f.content}" for f in self.selected_facts)


class ProfileCentricHypergraphMemory:
    """Profile-centric hypergraph memory with reward-guided utility learning."""

    def __init__(
        self,
        user_id: str = "default_user",
        attach_threshold: float = 0.06,
        discovery_threshold: float = 0.08,
        learning_rate: float = 0.18,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.user_id = user_id
        self.attach_threshold = attach_threshold
        self.discovery_threshold = discovery_threshold
        self.learning_rate = learning_rate
        self.weights = weights or {}
        self.facts: Dict[str, ProfileFact] = {}
        self.edges: Dict[str, ProfileHyperedgeUnit] = {}
        self.discovery_buffer: List[str] = []
        self._edge_counter = 0

    # ----------------------------- construction -----------------------------
    def add_fact(
        self,
        content: str,
        fact_id: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        promote: bool = True,
    ) -> ProfileFact:
        fid = fact_id or f"fact_{len(self.facts) + 1:06d}"
        fact = ProfileFact(
            fact_id=fid,
            content=str(content),
            keywords=keywords or self.extract_keywords(content),
            timestamp=float(timestamp if timestamp is not None else len(self.facts) + 1),
            metadata=metadata or {},
        )
        self.facts[fid] = fact
        if promote:
            self.promote_fact(fact)
        return fact

    def build_from_rows(self, rows: Sequence[Dict[str, Any]]) -> None:
        for i, row in enumerate(rows):
            content = row.get("content") or row.get("text") or row.get("fact") or row.get("summary") or ""
            if not content:
                continue
            fact_id = row.get("fact_id") or row.get("id") or f"fact_{i + 1:06d}"
            raw_keywords = row.get("keywords", [])
            keywords = [str(x) for x in raw_keywords] if isinstance(raw_keywords, list) else []
            self.add_fact(
                content=str(content),
                fact_id=str(fact_id),
                keywords=keywords,
                timestamp=float(row.get("timestamp") or row.get("time_index") or i + 1),
                metadata=dict(row),
            )
        self.discover_auto_edges()

    def promote_fact(self, fact: ProfileFact) -> ProfileHyperedgeUnit:
        edge_type, confidence, matched = self.infer_profile_type(fact.content)
        if edge_type == ProfileEdgeType.OTHER or confidence < 0.45:
            edge = self.attach_to_best_edge(fact, allowed_types=[ProfileEdgeType.AUTO_DISCOVERED])
            if edge is not None:
                return edge
            self.discovery_buffer.append(fact.fact_id)
            return self.create_edge(
                ProfileEdgeType.AUTO_DISCOVERED,
                [fact.fact_id],
                summary=self.edge_summary(ProfileEdgeType.AUTO_DISCOVERED, [fact]),
                confidence=0.40,
                metadata={"source": "low_confidence_seed", "rule_confidence": confidence, "matched_rules": matched},
            )

        edge = self.attach_to_best_edge(fact, allowed_types=[edge_type])
        if edge is not None:
            edge.metadata.setdefault("matched_rules", [])
            edge.metadata["matched_rules"] = sorted(set(edge.metadata["matched_rules"]) | set(matched))
            edge.confidence_score = clamp(max(edge.confidence_score, confidence))
            return edge

        return self.create_edge(
            edge_type,
            [fact.fact_id],
            summary=self.edge_summary(edge_type, [fact]),
            confidence=confidence,
            metadata={"source": "profile_seed", "rule_confidence": confidence, "matched_rules": matched},
        )

    def attach_to_best_edge(self, fact: ProfileFact, allowed_types: Optional[Sequence[ProfileEdgeType]] = None) -> Optional[ProfileHyperedgeUnit]:
        allowed = set(allowed_types or [])
        candidates = [e for e in self.edges.values() if e.status == "active" and (not allowed or e.edge_type in allowed)]
        best: Optional[ProfileHyperedgeUnit] = None
        best_score = 0.0
        for edge in candidates:
            s = max(keyword_overlap(fact.text(), edge.summary), keyword_overlap(fact.keywords, edge.keywords))
            if s > best_score:
                best_score, best = s, edge
        if best is None or best_score < self.attach_threshold:
            return None
        if fact.fact_id not in best.member_fact_ids:
            best.member_fact_ids.append(fact.fact_id)
        self.refresh_edge(best)
        return best

    def discover_auto_edges(self) -> None:
        active_auto = [e for e in self.edges.values() if e.edge_type == ProfileEdgeType.AUTO_DISCOVERED and e.status == "active"]
        changed = True
        while changed:
            changed = False
            for i, a in enumerate(list(active_auto)):
                if a.status != "active":
                    continue
                for b in active_auto[i + 1 :]:
                    if b.status != "active":
                        continue
                    sim = max(keyword_overlap(a.summary, b.summary), keyword_overlap(a.keywords, b.keywords))
                    if sim >= self.discovery_threshold:
                        for fid in b.member_fact_ids:
                            if fid not in a.member_fact_ids:
                                a.member_fact_ids.append(fid)
                        b.status = "merged"
                        self.refresh_edge(a)
                        changed = True
                active_auto = [e for e in active_auto if e.status == "active"]

    def create_edge(self, edge_type: ProfileEdgeType, member_fact_ids: List[str], summary: str, confidence: float = 0.50, metadata: Optional[Dict[str, Any]] = None) -> ProfileHyperedgeUnit:
        self._edge_counter += 1
        edge_id = f"profile_hyperedge_{self._edge_counter:06d}"
        facts = [self.facts[fid] for fid in member_fact_ids if fid in self.facts]
        max_ts = max([f.timestamp for f in self.facts.values()] or [1.0])
        now = max([f.timestamp for f in facts], default=max_ts)
        edge = ProfileHyperedgeUnit(
            edge_id=edge_id,
            edge_type=edge_type,
            summary=summary,
            member_fact_ids=list(dict.fromkeys(member_fact_ids)),
            keywords=self.extract_keywords(" ".join([summary] + [f.text() for f in facts])),
            utility_score=0.50,
            freshness_score=clamp(0.35 + 0.65 * (now / max_ts)),
            stability_score=clamp(0.45 + 0.08 * math.log1p(len(facts))),
            confidence_score=confidence,
            coherence_score=self.coherence(facts),
            metadata=metadata or {},
        )
        self.edges[edge.edge_id] = edge
        return edge

    def refresh_edge(self, edge: ProfileHyperedgeUnit) -> None:
        facts = [self.facts[fid] for fid in edge.member_fact_ids if fid in self.facts]
        edge.summary = self.edge_summary(edge.edge_type, facts)
        edge.keywords = self.extract_keywords(" ".join([edge.summary] + [f.text() for f in facts]))
        edge.coherence_score = self.coherence(facts)
        edge.stability_score = clamp(0.45 + 0.08 * math.log1p(len(facts)))
        if facts:
            max_ts = max([f.timestamp for f in self.facts.values()] or [1.0])
            edge.freshness_score = clamp(0.35 + 0.65 * (max(f.timestamp for f in facts) / max_ts))
        edge.updated_at = time.time()

    # ----------------------------- retrieval -----------------------------
    def retrieve(self, query: str, top_k_edges: int = 3, top_k_facts: int = 8, max_tokens: int = 450, use_utility: bool = True, fallback: bool = True, sufficiency_threshold: float = 0.10) -> ProfileRetrievalResult:
        query_type, _, _ = self.infer_profile_type(query)
        scored_edges: List[Tuple[ProfileHyperedgeUnit, float, Dict[str, float]]] = []
        for edge in self.edges.values():
            score, parts = edge.score(query, self.facts, query_type=query_type, use_utility=use_utility, weights=self.weights)
            if score > 0:
                scored_edges.append((edge, score, parts))
        scored_edges.sort(key=lambda x: x[1], reverse=True)
        selected = scored_edges[:top_k_edges]
        selected_edges = [e for e, _, _ in selected]
        edge_score = sum(s for _, s, _ in selected) / max(1, len(selected))
        facts = self.select_facts(query, selected_edges, top_k_facts=top_k_facts, max_tokens=max_tokens)
        fallback_used = False
        if (not facts or edge_score < sufficiency_threshold) and fallback:
            fallback_used = True
            facts = self.global_fact_retrieval(query, top_k=top_k_facts, max_tokens=max_tokens)
        for edge in selected_edges:
            edge.access_count += 1
        return ProfileRetrievalResult(
            query=query,
            channel="profile_hyperedge_to_fact" if not fallback_used else "profile_hyperedge_to_fact+fallback",
            selected_edges=selected_edges,
            selected_facts=facts,
            score=edge_score,
            tokens=estimate_tokens([f.content for f in facts]),
            fallback_used=fallback_used,
            sufficient=bool(facts),
            debug_scores=[{"edge_id": e.edge_id, "edge_type": e.edge_type.value, "score": round(s, 6), **parts, "summary": e.summary, "members": len(e.member_fact_ids)} for e, s, parts in selected],
        )

    def select_facts(self, query: str, edges: Sequence[ProfileHyperedgeUnit], top_k_facts: int = 8, max_tokens: int = 450) -> List[ProfileFact]:
        candidates: Dict[str, float] = {}
        for edge in edges:
            for fid in edge.member_fact_ids:
                fact = self.facts.get(fid)
                if fact is None:
                    continue
                fact_score = 0.70 * keyword_overlap(query, fact.text()) + 0.30 * edge.relevance(query) + 0.08 * edge.utility_score
                candidates[fid] = max(candidates.get(fid, 0.0), fact_score)
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        selected: List[ProfileFact] = []
        tokens = 0
        for fid, _ in ranked:
            fact = self.facts[fid]
            tok = estimate_tokens(fact.content)
            if selected and tokens + tok > max_tokens:
                continue
            selected.append(fact)
            tokens += tok
            if len(selected) >= top_k_facts or tokens >= max_tokens:
                break
        return selected

    def global_fact_retrieval(self, query: str, top_k: int = 8, max_tokens: int = 450) -> List[ProfileFact]:
        scored = [(f, keyword_overlap(query, f.text())) for f in self.facts.values()]
        scored = [(f, s) for f, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected: List[ProfileFact] = []
        tokens = 0
        for fact, _ in scored:
            tok = estimate_tokens(fact.content)
            if selected and tokens + tok > max_tokens:
                continue
            selected.append(fact)
            tokens += tok
            if len(selected) >= top_k or tokens >= max_tokens:
                break
        return selected

    def update_from_feedback(self, result: ProfileRetrievalResult, reward: float, hit: bool) -> None:
        for edge in result.selected_edges:
            edge.update_utility(reward=reward, hit=hit, lr=self.learning_rate)

    # ----------------------------- utilities -----------------------------
    def infer_profile_type(self, text: str) -> Tuple[ProfileEdgeType, float, List[str]]:
        t = (text or "").lower()
        rule_bank: List[Tuple[ProfileEdgeType, List[str]]] = [
            (ProfileEdgeType.RESEARCH_FOCUS, ["hypermem", "memory", "locomo", "a-mem", "memo", "超图", "超边", "rag", "retrieval"]),
            (ProfileEdgeType.GOAL, ["目标", "投稿", "aaai", "acl", "论文", "创新", "竞争力", "goal", "target"]),
            (ProfileEdgeType.PREFERENCE, ["喜欢", "偏好", "希望", "不喜欢", "prefer", "style", "更希望", "想要"]),
            (ProfileEdgeType.WRITING_STYLE, ["审稿人", "风险", "不足", "论文写法", "润色", "表达", "reviewer", "writing"]),
            (ProfileEdgeType.HABIT, ["经常", "通常", "习惯", "流程", "workflow", "每次", "先", "再"]),
            (ProfileEdgeType.TOOL_USAGE, ["github", "codex", "服务器", "conda", "bash", "脚本", "clone", "commit", "pull"]),
            (ProfileEdgeType.CURRENT_STATE, ["当前", "现在", "最新", "最终", "决定", "主线", "目前", "正在"]),
            (ProfileEdgeType.TEMPORAL_EVOLUTION, ["之前", "后来", "变化", "演化", "转向", "早期", "现在改成", "evolve"]),
            (ProfileEdgeType.DOMAIN_KNOWLEDGE, ["强化学习", "embedding", "向量", "检索", "训练", "评估", "dataset", "baseline"]),
        ]
        best_type = ProfileEdgeType.OTHER
        best_matches: List[str] = []
        for edge_type, kws in rule_bank:
            matches = [kw for kw in kws if kw.lower() in t]
            if len(matches) > len(best_matches):
                best_type = edge_type
                best_matches = matches
        if not best_matches:
            return ProfileEdgeType.OTHER, 0.0, []
        return best_type, clamp(0.35 + 0.15 * len(best_matches), hi=0.95), best_matches

    def edge_summary(self, edge_type: ProfileEdgeType, facts: Sequence[ProfileFact]) -> str:
        prefix = {
            ProfileEdgeType.RESEARCH_FOCUS: "用户研究方向",
            ProfileEdgeType.GOAL: "用户长期目标",
            ProfileEdgeType.PREFERENCE: "用户偏好",
            ProfileEdgeType.WRITING_STYLE: "用户写作/分析风格",
            ProfileEdgeType.HABIT: "用户工作习惯",
            ProfileEdgeType.TOOL_USAGE: "用户工具使用习惯",
            ProfileEdgeType.CURRENT_STATE: "用户当前状态",
            ProfileEdgeType.TEMPORAL_EVOLUTION: "用户想法演化",
            ProfileEdgeType.DOMAIN_KNOWLEDGE: "用户领域知识",
            ProfileEdgeType.AUTO_DISCOVERED: "自动发现的用户画像维度",
            ProfileEdgeType.OTHER: "其他用户画像",
        }.get(edge_type, edge_type.value)
        body = "；".join([f.content.strip() for f in facts[:4] if f.content.strip()])
        return f"{prefix}: {body}" if body else prefix

    def extract_keywords(self, text: str, max_keywords: int = 18) -> List[str]:
        toks = tokenize(text)
        stop = {"the", "and", "or", "to", "of", "in", "a", "is", "are", "for", "with", "我", "你", "他", "的", "了", "是", "在", "和", "也", "就", "都", "这个", "那个"}
        counts: Dict[str, int] = {}
        for tok in toks:
            if tok in stop:
                continue
            counts[tok] = counts.get(tok, 0) + 1
        return [k for k, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_keywords]]

    def coherence(self, facts: Sequence[ProfileFact]) -> float:
        if len(facts) <= 1:
            return 0.55
        sims: List[float] = []
        for i, a in enumerate(facts):
            for b in facts[i + 1 :]:
                sims.append(keyword_overlap(a.text(), b.text()))
        return clamp((sum(sims) / len(sims) if sims else 0.2) + 0.35)

    def edge_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for edge in self.edges.values():
            counts[edge.edge_type.value] = counts.get(edge.edge_type.value, 0) + 1
        return counts

    def export(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "num_facts": len(self.facts),
            "num_edges": len(self.edges),
            "active_edges": sum(1 for e in self.edges.values() if e.status == "active"),
            "edge_type_counts": self.edge_type_counts(),
            "avg_utility": round(sum(e.utility_score for e in self.edges.values()) / max(1, len(self.edges)), 6),
            "top_edges": [
                {
                    "edge_id": e.edge_id,
                    "edge_type": e.edge_type.value,
                    "summary": e.summary,
                    "utility_score": round(e.utility_score, 6),
                    "freshness_score": round(e.freshness_score, 6),
                    "stability_score": round(e.stability_score, 6),
                    "confidence_score": round(e.confidence_score, 6),
                    "hit_count": e.hit_count,
                    "failure_count": e.failure_count,
                    "members": e.member_fact_ids[:20],
                }
                for e in sorted(self.edges.values(), key=lambda x: x.utility_score, reverse=True)[:20]
            ],
        }

    def save(self, path: str | Path) -> None:
        data = {
            "user_id": self.user_id,
            "attach_threshold": self.attach_threshold,
            "discovery_threshold": self.discovery_threshold,
            "learning_rate": self.learning_rate,
            "weights": self.weights,
            "edge_counter": self._edge_counter,
            "facts": {fid: asdict(f) for fid, f in self.facts.items()},
            "edges": {eid: self._jsonify(asdict(e)) for eid, e in self.edges.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ProfileCentricHypergraphMemory":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        mem = cls(data.get("user_id", "default_user"), data.get("attach_threshold", 0.06), data.get("discovery_threshold", 0.08), data.get("learning_rate", 0.18), data.get("weights") or {})
        mem._edge_counter = int(data.get("edge_counter", 0))
        for fid, row in data.get("facts", {}).items():
            mem.facts[fid] = ProfileFact(**row)
        for eid, row in data.get("edges", {}).items():
            row["edge_type"] = ProfileEdgeType(row["edge_type"])
            mem.edges[eid] = ProfileHyperedgeUnit(**row)
        return mem

    def _jsonify(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {k: self._jsonify(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._jsonify(v) for v in value]
        return value
