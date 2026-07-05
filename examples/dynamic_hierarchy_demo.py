#!/usr/bin/env python
"""
Run a CPU-only dynamic hierarchy demo:

    python examples/dynamic_hierarchy_demo.py

The script compares a fixed HyperMem-style topic -> episode -> fact baseline
with an experimental variable-depth hierarchy plus query-adaptive expansion.
It does not call vLLM, OpenAI APIs, or the production HyperMem stages.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import statistics
import sys
import time
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "hypermem" / "experimental" / "dynamic_hierarchy.py"
spec = importlib.util.spec_from_file_location("dynamic_hierarchy", MODULE_PATH)
dynamic_hierarchy = importlib.util.module_from_spec(spec)
sys.modules["dynamic_hierarchy"] = dynamic_hierarchy
assert spec and spec.loader
spec.loader.exec_module(dynamic_hierarchy)

DynamicHierarchyBuilder = dynamic_hierarchy.DynamicHierarchyBuilder
DynamicRetriever = dynamic_hierarchy.DynamicRetriever
DynamicRetrievalResult = dynamic_hierarchy.DynamicRetrievalResult
SimpleTfidfIndex = dynamic_hierarchy.SimpleTfidfIndex
estimate_tokens = dynamic_hierarchy.estimate_tokens


TOY_DIALOGUE = [
    "用户最初在读 HyperMem，想理解为什么它用 topic、episode、fact 三层结构来组织长期对话记忆。",
    "用户注意到 HyperMem 的 hyperedge 可以把同一 topic 下分散出现的 episodes 连起来，避免普通图只表达 pairwise relation。",
    "用户担心自己还没有完全讲清楚 coarse-to-fine retrieval，也担心 topic 到 episode 再到 fact 的固定路径会漏掉跨事件证据。",
    "用户顺手提到电脑风扇有点吵，键盘手感一般，想晚点再处理这些生活问题。",
    "第二次讨论时，用户想把强化学习加入 HyperMem，让系统学习什么时候扩展 topic、什么时候直接拿 fact。",
    "用户设想 RL policy 可以根据 query type、token budget 和 evidence hit 来优化 retrieval action。",
    "用户还提到奖励函数可以结合答案正确率、上下文 token 数、检索延迟和是否覆盖关键 fact。",
    "中间用户又说自己想换机械键盘，但这和 HyperMem 论文方向没有直接关系。",
    "第三次讨论 AAAI 创新性时，用户担心 HyperMem 原始三层结构已经比较完整，单纯换 embedding 或 reranker 不够新。",
    "用户认为可选创新点包括学习型检索策略、动态记忆组织、跨 episode fact 聚合，以及更细粒度的预算控制。",
    "用户特别担心方向不够，是因为固定 Topic-Episode-Fact 可能被审稿人认为只是工程化层次设计，缺少机制创新。",
    "后来用户开始提出动态分层想法：简单内容只保留浅层，复杂内容可以继续拆 subtopic 或 sub-episode。",
    "用户希望动态规划根据语义一致性和长度惩罚自动切分连续对话，不再强制所有内容都塞进三层。",
    "用户还希望查询时能自适应展开：总结类问题横向展开多个 sibling，细节类问题纵向深入到 leaf。",
    "对于时间线问题，用户希望沿同一 topic 下的层级路径和时间顺序展开，展示想法如何从理解超图变成 RL，再变成动态分层。",
    "动态分层具体想解决的问题，是固定三层在复杂长对话里粒度不匹配，导致 token 浪费或相关 fact 被过度压缩。",
    "用户最后决定先做最小可运行 demo，不接 GPT 生成答案，只验证 retrieval、token、latency 和 evidence hit。",
]


QUERY_SPECS = [
    {
        "query": "我之前为什么担心 HyperMem 方向不够？",
        "expected_fragments": ["方向不够", "固定 Topic-Episode-Fact", "机制创新"],
    },
    {
        "query": "我想在 HyperMem 里加入什么学习方法？",
        "expected_fragments": ["强化学习", "RL policy", "reward"],
    },
    {
        "query": "我对 HyperMem 的想法是怎么逐渐变化的？",
        "expected_fragments": ["最初", "强化学习", "动态分层", "时间顺序"],
    },
    {
        "query": "这个方向有哪些可选创新点？",
        "expected_fragments": ["可选创新点", "学习型检索", "动态记忆组织"],
    },
    {
        "query": "动态分层具体想解决什么问题？",
        "expected_fragments": ["固定三层", "粒度不匹配", "token 浪费"],
    },
]


@dataclass
class FixedRetrievalResult:
    query: str
    query_type: str
    retrieved_fact_ids: List[str]
    expanded_node_ids: List[str]
    context_tokens: int
    latency_ms: float
    expanded_depth: int
    expanded_breadth: int
    evidence_hit: Optional[bool]


class FixedHyperMemStyleBaseline:
    """Tiny fixed topic -> episode -> fact retrieval baseline for comparison."""

    def __init__(self, utterances: List[str], max_context_tokens: int = 1000) -> None:
        self.max_context_tokens = max_context_tokens
        self.topics = {
            "topic_hypermem_research": {
                "title": "HyperMem Research",
                "summary": "用户围绕 HyperMem 的三层超图记忆、RL 改进、AAAI 创新性和动态分层想法反复讨论。",
                "episode_ids": ["ep_understanding", "ep_rl", "ep_aaai", "ep_dynamic"],
            },
            "topic_life_noise": {
                "title": "Life and Devices",
                "summary": "用户穿插讨论电脑风扇、键盘等生活设备问题。",
                "episode_ids": ["ep_life"],
            },
        }
        self.episodes = {
            "ep_understanding": {
                "summary": "理解 HyperMem 超图、topic/episode/fact 和 coarse-to-fine retrieval。",
                "text": "\n".join(utterances[0:4]),
                "topic_id": "topic_hypermem_research",
            },
            "ep_rl": {
                "summary": "讨论在 HyperMem 中加入强化学习优化检索动作。",
                "text": "\n".join(utterances[4:8]),
                "topic_id": "topic_hypermem_research",
            },
            "ep_aaai": {
                "summary": "讨论 AAAI 创新性，担心固定三层不够新，并列出可能创新点。",
                "text": "\n".join(utterances[8:11]),
                "topic_id": "topic_hypermem_research",
            },
            "ep_dynamic": {
                "summary": "讨论动态规划分层、查询自适应展开和最小 demo。",
                "text": "\n".join(utterances[11:17]),
                "topic_id": "topic_hypermem_research",
            },
            "ep_life": {
                "summary": "电脑风扇和机械键盘等生活问题。",
                "text": "\n".join([utterances[3], utterances[7]]),
                "topic_id": "topic_life_noise",
            },
        }
        self.facts: Dict[str, Dict[str, str]] = {}
        for ep_id, episode in self.episodes.items():
            for idx, sentence in enumerate(episode["text"].split("\n")):
                fact_id = f"fixed_fact_{ep_id}_{idx}"
                self.facts[fact_id] = {
                    "text": sentence,
                    "episode_id": ep_id,
                    "topic_id": episode["topic_id"],
                }
        self.topic_index = SimpleTfidfIndex().fit(
            list(self.topics),
            [item["title"] + " " + item["summary"] for item in self.topics.values()],
        )
        self.episode_index = SimpleTfidfIndex().fit(
            list(self.episodes),
            [item["summary"] + " " + item["text"] for item in self.episodes.values()],
        )
        self.fact_index = SimpleTfidfIndex().fit(
            list(self.facts),
            [item["text"] for item in self.facts.values()],
        )

    def retrieve(
        self,
        query: str,
        evidence_ids: Optional[Iterable[str]] = None,
        top_k: int = 5,
    ) -> FixedRetrievalResult:
        start = time.perf_counter()
        query_type = DynamicRetriever.detect_query_type(query)
        topic_hits = self.topic_index.search(query, top_k=1) or [("topic_hypermem_research", 0.0)]
        topic_id = topic_hits[0][0]
        topic_episode_ids = self.topics[topic_id]["episode_ids"]

        ranked_episode_ids = [
            ep_id for ep_id, _ in self.episode_index.search(query, top_k=len(self.episodes))
            if ep_id in topic_episode_ids
        ]
        if query_type in {"breadth", "temporal"}:
            expanded_episode_ids = topic_episode_ids
        else:
            expanded_episode_ids = ranked_episode_ids[:2] or topic_episode_ids[:2]

        candidate_facts = [
            fact_id for fact_id, fact in self.facts.items()
            if fact["episode_id"] in set(expanded_episode_ids)
        ]
        ranked_facts = [
            fact_id for fact_id, _ in self.fact_index.search(query, top_k=len(self.facts))
            if fact_id in set(candidate_facts)
        ]
        selected_facts = ranked_facts[:top_k]
        selected_nodes = [topic_id] + expanded_episode_ids
        selected_nodes, selected_facts = self._apply_budget(selected_nodes, selected_facts)
        expected = set(evidence_ids or [])
        return FixedRetrievalResult(
            query=query,
            query_type=query_type,
            retrieved_fact_ids=selected_facts,
            expanded_node_ids=selected_nodes,
            context_tokens=self._context_tokens(selected_nodes, selected_facts),
            latency_ms=(time.perf_counter() - start) * 1000.0,
            expanded_depth=3,
            expanded_breadth=len(expanded_episode_ids),
            evidence_hit=bool(expected.intersection(selected_facts)) if expected else None,
        )

    def evidence_ids_for(self, fragments: Iterable[str]) -> List[str]:
        return [
            fact_id for fact_id, fact in self.facts.items()
            if any(fragment in fact["text"] for fragment in fragments)
        ]

    def _apply_budget(self, node_ids: List[str], fact_ids: List[str]) -> tuple[List[str], List[str]]:
        kept_nodes: List[str] = []
        kept_facts: List[str] = []
        tokens = 0
        for node_id in node_ids:
            text = self.topics[node_id]["summary"] if node_id in self.topics else self.episodes[node_id]["text"]
            node_tokens = estimate_tokens(text)
            if tokens + node_tokens <= self.max_context_tokens:
                kept_nodes.append(node_id)
                tokens += node_tokens
        for fact_id in fact_ids:
            fact_tokens = estimate_tokens(self.facts[fact_id]["text"])
            if tokens + fact_tokens <= self.max_context_tokens:
                kept_facts.append(fact_id)
                tokens += fact_tokens
        return kept_nodes, kept_facts

    def _context_tokens(self, node_ids: List[str], fact_ids: List[str]) -> int:
        texts = []
        for node_id in node_ids:
            texts.append(self.topics[node_id]["summary"] if node_id in self.topics else self.episodes[node_id]["text"])
        texts.extend(self.facts[fact_id]["text"] for fact_id in fact_ids)
        return estimate_tokens("\n".join(texts))


def dynamic_evidence_ids(memory, fragments: Iterable[str]) -> List[str]:
    return [
        fact_id for fact_id, node in memory.facts.items()
        if any(fragment in node.text for fragment in fragments)
    ]


def print_result_pair(
    query: str,
    fixed: FixedRetrievalResult,
    dynamic: DynamicRetrievalResult,
    dynamic_retriever: DynamicRetriever,
) -> None:
    print("=" * 88)
    print(f"Query: {query}")
    print(f"Query type: {dynamic.query_type}")
    print("\nFixed HyperMem-style retrieval")
    print(f"  retrieved_fact_ids: {fixed.retrieved_fact_ids}")
    print(f"  expanded_node_ids: {fixed.expanded_node_ids}")
    print(f"  context_tokens: {fixed.context_tokens}")
    print(f"  latency_ms: {fixed.latency_ms:.2f}")
    print(f"  evidence_hit: {fixed.evidence_hit}")

    print("\nDynamic hierarchy retrieval")
    print(f"  retrieved_fact_ids: {dynamic.retrieved_fact_ids}")
    print(f"  expanded_node_ids: {dynamic.expanded_node_ids}")
    print(f"  hierarchy_path: {dynamic.expanded_path}")
    print(f"  context_tokens: {dynamic.context_tokens}")
    print(f"  baseline_fixed_tokens: {fixed.context_tokens}")
    print(f"  latency_ms: {dynamic.latency_ms:.2f}")
    print(f"  evidence_hit: {dynamic.evidence_hit}")
    print("  retrieved_facts:")
    for fact_id in dynamic.retrieved_fact_ids:
        print(f"    - {fact_id}: {dynamic_retriever.memory.nodes[fact_id].text}")


def summarize(method: str, results) -> None:
    print(f"{method}:")
    print(f"- avg_context_tokens: {statistics.mean(r.context_tokens for r in results):.1f}")
    print(f"- avg_latency_ms: {statistics.mean(r.latency_ms for r in results):.2f}")
    print(f"- evidence_hit_rate: {statistics.mean(1.0 if r.evidence_hit else 0.0 for r in results):.2f}")
    print(f"- avg_expanded_depth: {statistics.mean(r.expanded_depth for r in results):.1f}")
    print(f"- avg_expanded_breadth: {statistics.mean(r.expanded_breadth for r in results):.1f}")


def main() -> None:
    builder = DynamicHierarchyBuilder(
        max_depth=5,
        min_segment_size=2,
        max_leaf_tokens=80,
        split_gain_threshold=0.04,
        alpha=0.05,
        beta=0.1,
    )
    dynamic_memory = builder.build(TOY_DIALOGUE)
    dynamic_retriever = DynamicRetriever(dynamic_memory, max_context_tokens=420)
    fixed_baseline = FixedHyperMemStyleBaseline(TOY_DIALOGUE, max_context_tokens=760)

    print("Dynamic hierarchy stats:")
    non_fact_nodes = [node for node in dynamic_memory.nodes.values() if node.node_type != "fact"]
    print(f"- nodes: {len(dynamic_memory.nodes)}")
    print(f"- non_fact_nodes: {len(non_fact_nodes)}")
    print(f"- facts: {len(dynamic_memory.facts)}")
    print(f"- hyperedges: {len(dynamic_memory.hyperedges)}")
    print(f"- max_depth: {max(node.level for node in dynamic_memory.nodes.values())}")
    print()

    fixed_results: List[FixedRetrievalResult] = []
    dynamic_results: List[DynamicRetrievalResult] = []

    for spec in QUERY_SPECS:
        query = spec["query"]
        fixed_expected = fixed_baseline.evidence_ids_for(spec["expected_fragments"])
        dynamic_expected = dynamic_evidence_ids(dynamic_memory, spec["expected_fragments"])
        fixed = fixed_baseline.retrieve(query, evidence_ids=fixed_expected)
        dynamic = dynamic_retriever.retrieve(query, evidence_ids=dynamic_expected)
        fixed_results.append(fixed)
        dynamic_results.append(dynamic)
        print_result_pair(query, fixed, dynamic, dynamic_retriever)

    print("=" * 88)
    print("Per-query metrics:")
    print("method\tquery_type\tretrieved_fact_ids\texpanded_node_ids\texpanded_depth\texpanded_breadth\tcontext_tokens\tlatency_ms\tevidence_hit")
    for fixed, dynamic in zip(fixed_results, dynamic_results):
        print(
            "fixed\t"
            f"{fixed.query_type}\t{fixed.retrieved_fact_ids}\t{fixed.expanded_node_ids}\t"
            f"{fixed.expanded_depth}\t{fixed.expanded_breadth}\t{fixed.context_tokens}\t"
            f"{fixed.latency_ms:.2f}\t{fixed.evidence_hit}"
        )
        print(
            "dynamic\t"
            f"{dynamic.query_type}\t{dynamic.retrieved_fact_ids}\t{dynamic.expanded_node_ids}\t"
            f"{dynamic.expanded_depth}\t{dynamic.expanded_breadth}\t{dynamic.context_tokens}\t"
            f"{dynamic.latency_ms:.2f}\t{dynamic.evidence_hit}"
        )

    print("\nSummary table:")
    summarize("Fixed HyperMem-style retrieval", fixed_results)
    summarize("Dynamic hierarchy retrieval", dynamic_results)

    print("\nLoCoMo upgrade sketch:")
    print("- replace TOY_DIALOGUE with LoCoMo conversation turns or existing HyperMem episode texts")
    print("- map gold evidence spans to expected fact ids for evidence_hit/evidence_recall")
    print("- optionally swap SimpleTfidfIndex for HyperMem stage3 BM25+dense indexes")
    print("- keep DynamicRetriever actions as the policy surface for later RL optimization")


if __name__ == "__main__":
    main()
