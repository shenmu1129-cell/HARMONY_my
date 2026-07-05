"""
Verifier-guided adaptive retrieval controller for the dynamic hierarchy demo.

This module is retrieval-only. It does not call an LLM, generate answers, or
touch HyperMem stages 5/6. The controller starts with cheap anchor fact search,
uses a rule verifier to decide what is missing, and adaptively expands memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .dynamic_hierarchy import (
    DynamicHierarchyMemory,
    DynamicRetriever,
    SimpleTfidfIndex,
    estimate_tokens,
)


TEMPORAL_KEYWORDS = {
    "before", "after", "later", "earlier", "change", "evolve", "timeline",
    "over time", "when", "之前", "后来", "之后", "变化", "逐渐", "过程", "时间线",
    "什么时候", "先后",
}
SUMMARY_KEYWORDS = {
    "summarize", "summary", "overview", "overall", "all", "list", "what are",
    "哪些", "总结", "整体", "有哪些", "概括",
}
WHY_HOW_KEYWORDS = {
    "why", "how", "reason", "because", "explain", "为什么", "如何", "怎么", "原因",
}
COMPARE_KEYWORDS = {
    "compare", "versus", "vs", "different", "difference", "same", "比较", "区别", "差异",
}
COREFERENCE_KEYWORDS = {
    "this", "that", "these", "those", "it", "they", "them", "之前说的", "这个", "那个",
    "这些", "那些", "这件事", "那件事",
}


@dataclass(frozen=True)
class MemoryAction:
    name: str
    budget_tokens: int = 0
    max_items: int = 0


@dataclass
class EvidenceItem:
    fact_id: str
    score: float
    source_action: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceVerifierOutput:
    sufficient: bool
    confidence: float
    missing_type: str
    suggested_action: str
    features: Dict[str, Any]


@dataclass
class AdaptiveStepTrace:
    step: int
    action: str
    added_fact_ids: List[str]
    added_node_ids: List[str]
    context_tokens: int
    verifier: EvidenceVerifierOutput


@dataclass
class AdaptiveRetrievalResult:
    query: str
    query_type: str
    selected_fact_ids: List[str]
    selected_node_ids: List[str]
    action_sequence: List[str]
    verifier_outputs: List[EvidenceVerifierOutput]
    steps: List[AdaptiveStepTrace]
    context_tokens: int
    latency_ms: float
    stopped_reason: str


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


class EvidenceVerifier:
    """Rule verifier over retrieval confidence and structured fact metadata."""

    def __init__(
        self,
        high_score_threshold: float = 0.28,
        low_score_threshold: float = 0.12,
        margin_threshold: float = 0.05,
    ) -> None:
        self.high_score_threshold = high_score_threshold
        self.low_score_threshold = low_score_threshold
        self.margin_threshold = margin_threshold

    def verify(
        self,
        query: str,
        evidence: Sequence[EvidenceItem],
        query_type: str,
        context_tokens: int,
        max_context_tokens: int,
        step: int,
    ) -> EvidenceVerifierOutput:
        scores = sorted((item.score for item in evidence), reverse=True)
        top1 = scores[0] if scores else 0.0
        top2 = scores[1] if len(scores) > 1 else 0.0
        margin = top1 - top2
        score_entropy = self._entropy(scores[:8])
        time_buckets = {
            int(item.metadata.get("time_index", item.metadata.get("start", 0))) // 1000
            for item in evidence
            if item.metadata.get("time_index", item.metadata.get("start", None)) is not None
        }
        node_count = len({item.metadata.get("node_id") or item.metadata.get("episode_id") for item in evidence})

        has_temporal = contains_any(query, TEMPORAL_KEYWORDS) or query_type == "temporal"
        has_summary = contains_any(query, SUMMARY_KEYWORDS) or query_type == "breadth"
        has_reason = contains_any(query, WHY_HOW_KEYWORDS) or query_type == "depth"
        has_compare = contains_any(query, COMPARE_KEYWORDS)
        has_coref = contains_any(query, COREFERENCE_KEYWORDS)

        confidence = self._confidence(top1, margin, len(evidence), node_count, score_entropy)
        features = {
            "top1_score": top1,
            "top2_score": top2,
            "score_margin": margin,
            "score_entropy": score_entropy,
            "evidence_count": len(evidence),
            "node_count": node_count,
            "time_coverage": len(time_buckets),
            "context_tokens": context_tokens,
            "max_context_tokens": max_context_tokens,
            "has_temporal_keyword": int(has_temporal),
            "has_summary_keyword": int(has_summary),
            "has_reason_keyword": int(has_reason),
            "has_compare_keyword": int(has_compare),
            "has_coreference_keyword": int(has_coref),
        }

        if has_temporal and len(time_buckets) < 2:
            return EvidenceVerifierOutput(False, confidence, "temporal", "temporal_expand", features)
        if has_summary and (len(evidence) < 6 or node_count < 2):
            return EvidenceVerifierOutput(False, confidence, "summary", "parent_summary_expand", features)
        if has_compare and len(evidence) < 4:
            return EvidenceVerifierOutput(False, confidence, "multi_hop", "sibling_fact_expand", features)
        if has_reason and (len(evidence) < 4 or confidence < 0.42):
            return EvidenceVerifierOutput(False, confidence, "reason", "sibling_fact_expand", features)
        if has_coref and top1 < self.high_score_threshold:
            return EvidenceVerifierOutput(False, confidence, "coreference", "parent_summary_expand", features)
        if top1 < self.low_score_threshold or len(evidence) < 2:
            return EvidenceVerifierOutput(False, confidence, "low_confidence", "global_fact_search_large", features)

        simple_query = not (has_temporal or has_summary or has_reason or has_compare)
        sufficient = (
            confidence >= 0.62
            and top1 >= self.high_score_threshold
            and margin >= self.margin_threshold
            and context_tokens <= max_context_tokens
            and len(evidence) >= 4
            and (simple_query or len(evidence) >= 5)
        )
        if sufficient:
            return EvidenceVerifierOutput(True, confidence, "none", "stop", features)
        return EvidenceVerifierOutput(False, confidence, "low_confidence", "global_fact_search_large", features)

    @staticmethod
    def _entropy(scores: Sequence[float]) -> float:
        positive = [max(score, 0.0) for score in scores]
        total = sum(positive)
        if total <= 0:
            return 0.0
        return -sum((score / total) * math.log((score / total) + 1e-12) for score in positive)

    @staticmethod
    def _confidence(top1: float, margin: float, evidence_count: int, node_count: int, entropy: float) -> float:
        score_part = min(1.0, top1 / 0.35)
        margin_part = min(1.0, max(0.0, margin) / 0.12)
        count_part = min(1.0, evidence_count / 5.0)
        node_part = min(1.0, node_count / 3.0)
        entropy_penalty = min(0.2, entropy / 15.0)
        return max(0.0, min(1.0, 0.50 * score_part + 0.20 * margin_part + 0.20 * count_part + 0.10 * node_part - entropy_penalty))


class AdaptiveMemoryController:
    """Multi-step memory action controller over DynamicHierarchyMemory."""

    ACTIONS = {
        "global_fact_search_small": MemoryAction("global_fact_search_small", budget_tokens=360, max_items=12),
        "global_fact_search_large": MemoryAction("global_fact_search_large", budget_tokens=800, max_items=32),
        "parent_summary_expand": MemoryAction("parent_summary_expand", budget_tokens=160, max_items=4),
        "sibling_fact_expand": MemoryAction("sibling_fact_expand", budget_tokens=420, max_items=16),
        "temporal_expand": MemoryAction("temporal_expand", budget_tokens=520, max_items=12),
        "dynamic_path_expand": MemoryAction("dynamic_path_expand", budget_tokens=500, max_items=12),
        "compress_context": MemoryAction("compress_context"),
        "stop": MemoryAction("stop"),
    }

    def __init__(
        self,
        memory: DynamicHierarchyMemory,
        max_context_tokens: int = 800,
        max_steps: int = 4,
        verifier: Optional[EvidenceVerifier] = None,
    ) -> None:
        self.memory = memory
        self.max_context_tokens = max_context_tokens
        self.max_steps = max_steps
        self.verifier = verifier or EvidenceVerifier()
        self.retriever = DynamicRetriever(memory, max_context_tokens=max_context_tokens)

    def run(self, query: str, top_k: int = 8) -> AdaptiveRetrievalResult:
        start = time.perf_counter()
        query_type = DynamicRetriever.detect_query_type(query)
        evidence: Dict[str, EvidenceItem] = {}
        selected_nodes: Dict[str, float] = {}
        steps: List[AdaptiveStepTrace] = []
        action_sequence: List[str] = []
        verifier_outputs: List[EvidenceVerifierOutput] = []
        last_verifier: Optional[EvidenceVerifierOutput] = None

        for step_idx in range(self.max_steps):
            action_name = self._choose_action(step_idx, last_verifier, action_sequence)
            action_sequence.append(action_name)
            if action_name == "stop":
                break

            added_facts, added_nodes = self._execute_action(action_name, query, evidence, selected_nodes, top_k)
            self._compress(evidence, selected_nodes)
            context_tokens = self._context_tokens(list(selected_nodes), list(evidence))
            verifier_output = self.verifier.verify(
                query,
                self._ordered_evidence(evidence),
                query_type,
                context_tokens,
                self.max_context_tokens,
                step_idx + 1,
            )
            verifier_outputs.append(verifier_output)
            steps.append(AdaptiveStepTrace(
                step=step_idx + 1,
                action=action_name,
                added_fact_ids=added_facts,
                added_node_ids=added_nodes,
                context_tokens=context_tokens,
                verifier=verifier_output,
            ))
            last_verifier = verifier_output
            if verifier_output.sufficient:
                action_sequence.append("stop")
                break

        if not action_sequence or action_sequence[-1] not in {"stop", "compress_context"}:
            action_sequence.append("compress_context")
            self._compress(evidence, selected_nodes)

        selected_fact_ids = [item.fact_id for item in self._ordered_evidence(evidence)]
        selected_node_ids = [
            node_id for node_id, _score in sorted(
                selected_nodes.items(),
                key=lambda item: (-item[1], self.memory.nodes[item[0]].metadata.get("start", 0)),
            )
        ]
        context_tokens = self._context_tokens(selected_node_ids, selected_fact_ids)
        return AdaptiveRetrievalResult(
            query=query,
            query_type=query_type,
            selected_fact_ids=selected_fact_ids,
            selected_node_ids=selected_node_ids,
            action_sequence=action_sequence,
            verifier_outputs=verifier_outputs,
            steps=steps,
            context_tokens=context_tokens,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            stopped_reason="sufficient" if last_verifier and last_verifier.sufficient else "max_steps",
        )

    def _choose_action(
        self,
        step_idx: int,
        last_verifier: Optional[EvidenceVerifierOutput],
        previous_actions: Sequence[str],
    ) -> str:
        if step_idx == 0:
            return "global_fact_search_small"
        if last_verifier and last_verifier.sufficient:
            return "stop"
        missing = last_verifier.missing_type if last_verifier else "low_confidence"
        plan = {
            "temporal": ["temporal_expand", "global_fact_search_large"],
            "summary": ["parent_summary_expand", "sibling_fact_expand", "global_fact_search_large"],
            "multi_hop": ["sibling_fact_expand", "dynamic_path_expand", "global_fact_search_large"],
            "reason": ["sibling_fact_expand", "dynamic_path_expand", "global_fact_search_large"],
            "coreference": ["global_fact_search_large"],
            "low_confidence": ["global_fact_search_large"],
        }.get(missing, ["global_fact_search_large"])
        for action_name in plan:
            if action_name not in previous_actions:
                return action_name
        return "stop"

    def _execute_action(
        self,
        action_name: str,
        query: str,
        evidence: Dict[str, EvidenceItem],
        selected_nodes: Dict[str, float],
        top_k: int,
    ) -> Tuple[List[str], List[str]]:
        if action_name == "global_fact_search_small":
            return self._global_fact_search(query, evidence, selected_nodes, self.ACTIONS[action_name])
        if action_name == "global_fact_search_large":
            return self._global_fact_search(query, evidence, selected_nodes, self.ACTIONS[action_name])
        if action_name == "parent_summary_expand":
            return self._parent_summary_expand(evidence, selected_nodes)
        if action_name == "sibling_fact_expand":
            return self._sibling_fact_expand(query, evidence, selected_nodes, self.ACTIONS[action_name])
        if action_name == "temporal_expand":
            return self._temporal_expand(query, evidence, selected_nodes, self.ACTIONS[action_name])
        if action_name == "dynamic_path_expand":
            return self._dynamic_path_expand(query, evidence, selected_nodes, top_k, self.ACTIONS[action_name])
        if action_name == "compress_context":
            self._compress(evidence, selected_nodes)
            return [], []
        return [], []

    def _global_fact_search(
        self,
        query: str,
        evidence: Dict[str, EvidenceItem],
        selected_nodes: Dict[str, float],
        action: MemoryAction,
    ) -> Tuple[List[str], List[str]]:
        added: List[str] = []
        added_nodes: List[str] = []
        tokens = 0
        for fact_id, score in self.retriever.fact_index.search(query, top_k=max(action.max_items * 8, 80)):
            if fact_id in evidence:
                continue
            node = self.memory.nodes[fact_id]
            fact_tokens = estimate_tokens(node.text)
            if tokens + fact_tokens > action.budget_tokens:
                continue
            tokens += fact_tokens
            if self._add_fact(fact_id, score, action.name, evidence):
                added.append(fact_id)
            if len(added) >= action.max_items:
                break
        return added, added_nodes

    def _parent_summary_expand(
        self,
        evidence: Dict[str, EvidenceItem],
        selected_nodes: Dict[str, float],
    ) -> Tuple[List[str], List[str]]:
        added_nodes: List[str] = []
        anchors = self._anchor_fact_ids(evidence, limit=4)
        for rank, fact_id in enumerate(anchors):
            cursor = self.memory.nodes[fact_id].parent_id
            score = max(0.05, evidence[fact_id].score * (0.8 - rank * 0.08))
            while cursor and cursor in self.memory.nodes:
                if self.memory.nodes[cursor].node_type != "root" and cursor not in selected_nodes:
                    selected_nodes[cursor] = score
                    added_nodes.append(cursor)
                cursor = self.memory.nodes[cursor].parent_id
        return [], added_nodes[: self.ACTIONS["parent_summary_expand"].max_items]

    def _sibling_fact_expand(
        self,
        query: str,
        evidence: Dict[str, EvidenceItem],
        selected_nodes: Dict[str, float],
        action: MemoryAction,
    ) -> Tuple[List[str], List[str]]:
        candidate_nodes: List[str] = []
        for fact_id in self._anchor_fact_ids(evidence, limit=5):
            parent_id = self.memory.nodes[fact_id].parent_id
            grandparent_id = self.memory.nodes[parent_id].parent_id if parent_id else None
            siblings = self.memory.nodes[grandparent_id].children if grandparent_id else [parent_id]
            for node_id in siblings:
                if node_id and self.memory.nodes[node_id].node_type != "fact" and node_id not in candidate_nodes:
                    candidate_nodes.append(node_id)
        ranked_facts = self.retriever._rank_facts(query, self._facts_from_nodes(candidate_nodes))
        return self._add_ranked_facts(ranked_facts, query, action, evidence, selected_nodes)

    def _temporal_expand(
        self,
        query: str,
        evidence: Dict[str, EvidenceItem],
        selected_nodes: Dict[str, float],
        action: MemoryAction,
    ) -> Tuple[List[str], List[str]]:
        anchor_id = self._best_anchor_node(query, evidence)
        topic_id = self.retriever._best_temporal_anchor(query, anchor_id)
        descendants = self.retriever._descendant_nonfact_nodes(topic_id) or [topic_id]
        ranked_nodes = self.retriever._rank_nodes(query, descendants)
        buckets: Dict[int, List[str]] = {}
        for node_id in ranked_nodes[:20]:
            facts = self.retriever._rank_facts(query, self.retriever._facts_under(node_id))
            for fact_id in facts[:4]:
                bucket = int(self.memory.nodes[fact_id].metadata.get("time_index", 0)) // 1000
                buckets.setdefault(bucket, []).append(fact_id)
        ordered_facts: List[str] = []
        for _bucket, fact_ids in sorted(buckets.items()):
            ordered_facts.extend(fact_ids[:2])
        if not ordered_facts:
            ordered_facts = self.retriever._rank_facts(query, self.retriever._facts_under(topic_id))
        if topic_id not in selected_nodes:
            selected_nodes[topic_id] = 0.15
        return self._add_ranked_facts(ordered_facts, query, action, evidence, selected_nodes)

    def _dynamic_path_expand(
        self,
        query: str,
        evidence: Dict[str, EvidenceItem],
        selected_nodes: Dict[str, float],
        top_k: int,
        action: MemoryAction,
    ) -> Tuple[List[str], List[str]]:
        result = self.retriever.retrieve(query, top_k=max(top_k, action.max_items // 2))
        added_nodes: List[str] = []
        for rank, node_id in enumerate(result.expanded_node_ids):
            if node_id not in selected_nodes:
                selected_nodes[node_id] = max(0.05, 0.25 - rank * 0.02)
                added_nodes.append(node_id)
        added_facts, more_nodes = self._add_ranked_facts(result.retrieved_fact_ids, query, action, evidence, selected_nodes)
        return added_facts, added_nodes + more_nodes

    def _add_ranked_facts(
        self,
        fact_ids: Sequence[str],
        query: str,
        action: MemoryAction,
        evidence: Dict[str, EvidenceItem],
        selected_nodes: Dict[str, float],
    ) -> Tuple[List[str], List[str]]:
        added: List[str] = []
        added_nodes: List[str] = []
        tokens = 0
        score_by_fact = dict(self.retriever.fact_index.search(query, top_k=max(len(fact_ids) * 3, 80)))
        for rank, fact_id in enumerate(dict.fromkeys(fact_ids)):
            if fact_id not in self.memory.nodes or self.memory.nodes[fact_id].node_type != "fact":
                continue
            score = float(score_by_fact.get(fact_id, max(0.01, 0.12 - rank * 0.004)))
            fact_tokens = estimate_tokens(self.memory.nodes[fact_id].text)
            if tokens + fact_tokens > action.budget_tokens:
                continue
            tokens += fact_tokens
            if self._add_fact(fact_id, score, action.name, evidence):
                added.append(fact_id)
            parent_id = self.memory.nodes[fact_id].parent_id
            if parent_id and parent_id not in selected_nodes:
                selected_nodes[parent_id] = score * 0.25
                added_nodes.append(parent_id)
            if len(added) >= action.max_items:
                break
        return added, added_nodes

    def _add_fact(
        self,
        fact_id: str,
        score: float,
        source_action: str,
        evidence: Dict[str, EvidenceItem],
    ) -> bool:
        node = self.memory.nodes[fact_id]
        if fact_id in evidence:
            if score > evidence[fact_id].score:
                evidence[fact_id].score = score
                evidence[fact_id].source_action = source_action
            return False
        evidence[fact_id] = EvidenceItem(
            fact_id=fact_id,
            score=score,
            source_action=source_action,
            text=node.text,
            metadata=dict(node.metadata),
        )
        return True

    def _compress(self, evidence: Dict[str, EvidenceItem], selected_nodes: Dict[str, float]) -> None:
        ordered_items = self._ordered_evidence(evidence)
        ordered_nodes = [
            node_id for node_id, _score in sorted(
                selected_nodes.items(),
                key=lambda item: (-item[1], self.memory.nodes[item[0]].metadata.get("start", 0)),
            )
        ]
        kept_nodes: Dict[str, float] = {}
        kept_evidence: Dict[str, EvidenceItem] = {}
        tokens = 0
        for node_id in ordered_nodes:
            node_tokens = estimate_tokens(self.memory.nodes[node_id].summary)
            if tokens + node_tokens <= self.max_context_tokens:
                kept_nodes[node_id] = selected_nodes[node_id]
                tokens += node_tokens
        for item in ordered_items:
            fact_tokens = estimate_tokens(item.text)
            if tokens + fact_tokens <= self.max_context_tokens:
                kept_evidence[item.fact_id] = item
                tokens += fact_tokens
        selected_nodes.clear()
        selected_nodes.update(kept_nodes)
        evidence.clear()
        evidence.update(kept_evidence)

    def _ordered_evidence(self, evidence: Dict[str, EvidenceItem]) -> List[EvidenceItem]:
        return sorted(
            evidence.values(),
            key=lambda item: (-item.score, self.memory.nodes[item.fact_id].metadata.get("time_index", 0)),
        )

    def _anchor_fact_ids(self, evidence: Dict[str, EvidenceItem], limit: int = 3) -> List[str]:
        return [item.fact_id for item in self._ordered_evidence(evidence)[:limit]]

    def _best_anchor_node(self, query: str, evidence: Dict[str, EvidenceItem]) -> str:
        anchors = self._anchor_fact_ids(evidence, limit=1)
        if anchors:
            parent_id = self.memory.nodes[anchors[0]].parent_id
            if parent_id:
                return parent_id
        candidates = self.retriever.node_index.search(query, top_k=3)
        return candidates[0][0] if candidates else self.memory.root_id

    def _facts_from_nodes(self, node_ids: Sequence[str]) -> List[str]:
        fact_ids: List[str] = []
        for node_id in node_ids:
            if node_id in self.memory.nodes and self.memory.nodes[node_id].node_type != "fact":
                fact_ids.extend(self.retriever._facts_under(node_id))
        return list(dict.fromkeys(fact_ids))

    def _context_tokens(self, node_ids: Sequence[str], fact_ids: Sequence[str]) -> int:
        node_texts = [
            self.memory.nodes[node_id].summary
            for node_id in node_ids
            if node_id in self.memory.nodes and self.memory.nodes[node_id].node_type != "fact"
        ]
        fact_texts = [
            self.memory.nodes[fact_id].text
            for fact_id in fact_ids
            if fact_id in self.memory.nodes and self.memory.nodes[fact_id].node_type == "fact"
        ]
        return estimate_tokens("\n".join(node_texts + fact_texts))


def trace_to_dict(result: AdaptiveRetrievalResult, memory: DynamicHierarchyMemory) -> Dict[str, Any]:
    return {
        "query_type": result.query_type,
        "action_sequence": result.action_sequence,
        "verifier_outputs": [
            {
                "sufficient": output.sufficient,
                "confidence": output.confidence,
                "missing_type": output.missing_type,
                "suggested_action": output.suggested_action,
                "features": output.features,
            }
            for output in result.verifier_outputs
        ],
        "selected_evidence": [
            {
                "fact_id": fact_id,
                "text": memory.nodes[fact_id].text,
                "source_turn_ids": memory.nodes[fact_id].metadata.get("source_turn_ids", []),
                "time_index": memory.nodes[fact_id].metadata.get("time_index"),
                "node_id": memory.nodes[fact_id].metadata.get("node_id"),
            }
            for fact_id in result.selected_fact_ids
        ],
        "selected_nodes": result.selected_node_ids,
        "context_tokens": result.context_tokens,
        "latency_ms": result.latency_ms,
        "stopped_reason": result.stopped_reason,
    }
