#!/usr/bin/env python
"""
Retrieval-only LoCoMo comparison for the experimental dynamic hierarchy.

This script does not run HyperMem's LLM extraction/answering stages. It uses
the LoCoMo conversations already converted to `data/locomo10.json`, builds:

1. a fixed HyperMem-style topic -> session episode -> utterance fact memory;
2. the experimental dynamic hierarchy over the same session texts.

Then it measures evidence hit against LoCoMo evidence dia_ids.

Run:
    /opt/miniconda3/envs/wwt310/bin/python examples/locomo_retrieval_eval.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "hypermem" / "experimental" / "dynamic_hierarchy.py"
spec = importlib.util.spec_from_file_location("dynamic_hierarchy", MODULE_PATH)
dynamic_hierarchy = importlib.util.module_from_spec(spec)
sys.modules["dynamic_hierarchy"] = dynamic_hierarchy
assert spec and spec.loader
spec.loader.exec_module(dynamic_hierarchy)

DynamicHierarchyBuilder = dynamic_hierarchy.DynamicHierarchyBuilder
DynamicRetriever = dynamic_hierarchy.DynamicRetriever
SimpleTfidfIndex = dynamic_hierarchy.SimpleTfidfIndex
estimate_tokens = dynamic_hierarchy.estimate_tokens


@dataclass
class RetrievalMetrics:
    method: str
    query: str
    category: Any
    query_type: str
    retrieved_fact_ids: List[str]
    expanded_node_ids: List[str]
    context_tokens: int
    latency_ms: float
    evidence_hit: bool
    evidence_recall: float
    expanded_depth: int
    expanded_breadth: int


class FixedSessionHyperMemBaseline:
    """Fixed 3-layer baseline: one topic, session episodes, utterance facts."""

    def __init__(self, item: Dict[str, Any], max_context_tokens: int = 1200) -> None:
        conversation = item["conversation"]
        self.max_context_tokens = max_context_tokens
        self.topic_id = "topic_conversation"
        self.topics = {
            self.topic_id: {
                "summary": (
                    f"Conversation between {conversation.get('speaker_a', 'Speaker A')} "
                    f"and {conversation.get('speaker_b', 'Speaker B')} across sessions."
                ),
            }
        }
        self.episodes: Dict[str, Dict[str, Any]] = {}
        self.facts: Dict[str, Dict[str, Any]] = {}

        for session_key in session_keys(conversation):
            messages = conversation[session_key]
            episode_id = f"episode_{session_key}"
            lines = [format_message(msg) for msg in messages]
            lines.extend(observation_lines(item, session_key))
            self.episodes[episode_id] = {
                "summary": f"{session_key} with {len(messages)} turns.",
                "text": "\n".join(lines),
                "session": session_key,
                "topic_id": self.topic_id,
            }
            for msg in messages:
                dia_id = msg.get("dia_id", "")
                fact_id = f"fact_{dia_id}"
                self.facts[fact_id] = {
                    "text": format_message(msg),
                    "episode_id": episode_id,
                    "topic_id": self.topic_id,
                    "dia_id": dia_id,
                }
            for obs_idx, obs_line in enumerate(observation_lines(item, session_key)):
                dia_ids = extract_dia_ids(obs_line)
                dia_id = dia_ids[0] if dia_ids else f"{session_key}_obs_{obs_idx}"
                fact_id = f"fact_obs_{session_key}_{obs_idx}"
                self.facts[fact_id] = {
                    "text": obs_line,
                    "episode_id": episode_id,
                    "topic_id": self.topic_id,
                    "dia_id": dia_id,
                }

        self.episode_index = SimpleTfidfIndex().fit(
            list(self.episodes),
            [episode["summary"] + "\n" + episode["text"] for episode in self.episodes.values()],
        )
        self.fact_index = SimpleTfidfIndex().fit(
            list(self.facts),
            [fact["text"] for fact in self.facts.values()],
        )

    def retrieve(self, query: str, evidence_ids: Sequence[str], top_k: int = 8) -> RetrievalMetrics:
        start = time.perf_counter()
        query_type = DynamicRetriever.detect_query_type(query)
        ranked_episodes = [episode_id for episode_id, _ in self.episode_index.search(query, top_k=8)]
        if query_type in {"breadth", "temporal"}:
            expanded_episode_ids = ranked_episodes[:5]
        else:
            expanded_episode_ids = ranked_episodes[:2]
        if not expanded_episode_ids:
            expanded_episode_ids = list(self.episodes)[:2]

        allowed_facts = {
            fact_id
            for fact_id, fact in self.facts.items()
            if fact["episode_id"] in set(expanded_episode_ids)
        }
        ranked_facts = [
            fact_id for fact_id, _ in self.fact_index.search(query, top_k=max(top_k * 6, 30))
            if fact_id in allowed_facts
        ]
        ranked_facts.extend(fact_id for fact_id in allowed_facts if fact_id not in ranked_facts)
        selected_nodes, selected_facts = self._apply_budget(
            [self.topic_id] + expanded_episode_ids,
            ranked_facts[:top_k],
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        hit, recall = score_evidence(
            evidence_ids,
            [self.facts[fact_id]["text"] for fact_id in selected_facts],
        )
        return RetrievalMetrics(
            method="fixed_hypermem_style",
            query=query,
            category=None,
            query_type=query_type,
            retrieved_fact_ids=selected_facts,
            expanded_node_ids=selected_nodes,
            context_tokens=self._context_tokens(selected_nodes, selected_facts),
            latency_ms=latency_ms,
            evidence_hit=hit,
            evidence_recall=recall,
            expanded_depth=3,
            expanded_breadth=len(expanded_episode_ids),
        )

    def _apply_budget(self, node_ids: List[str], fact_ids: List[str]) -> Tuple[List[str], List[str]]:
        kept_nodes: List[str] = []
        kept_facts: List[str] = []
        tokens = 0
        for node_id in node_ids:
            if node_id in self.topics:
                text = self.topics[node_id]["summary"]
            else:
                text = self.episodes[node_id]["text"]
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

    def _context_tokens(self, node_ids: Sequence[str], fact_ids: Sequence[str]) -> int:
        texts: List[str] = []
        for node_id in node_ids:
            texts.append(self.topics[node_id]["summary"] if node_id in self.topics else self.episodes[node_id]["text"])
        texts.extend(self.facts[fact_id]["text"] for fact_id in fact_ids)
        return estimate_tokens("\n".join(texts))


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def session_keys(conversation: Dict[str, Any]) -> List[str]:
    def session_num(key: str) -> int:
        return int(key.split("_")[1])

    return sorted(
        [
            key for key in conversation
            if key.startswith("session_") and not key.endswith("_date_time")
        ],
        key=session_num,
    )


def format_message(message: Dict[str, Any]) -> str:
    return f"[{message.get('dia_id', '')}] {message.get('speaker', '')}: {message.get('text', '')}"


def observation_lines(item: Dict[str, Any], session_key: str) -> List[str]:
    observations = item.get("observation", {})
    session_observation = observations.get(f"{session_key}_observation", {})
    lines: List[str] = []
    for speaker, entries in session_observation.items():
        for entry in entries:
            if not isinstance(entry, list) or not entry:
                continue
            text = str(entry[0])
            dia_ids = entry[1:]
            dia_text = ", ".join(str(dia_id) for dia_id in dia_ids)
            lines.append(f"[{dia_text}] Observation {speaker}: {text}")
    return lines


def extract_dia_ids(text: str) -> List[str]:
    import re

    return re.findall(r"D\d+:\d+", text)


def build_session_texts(item: Dict[str, Any]) -> List[str]:
    conversation = item["conversation"]
    texts: List[str] = []
    for session_key in session_keys(conversation):
        session_time = conversation.get(f"{session_key}_date_time", "")
        lines = [f"{session_key} at {session_time}"]
        lines.extend(observation_lines(item, session_key))
        lines.extend(format_message(msg) for msg in conversation[session_key])
        texts.append("\n".join(lines))
    return texts


def score_evidence(evidence_ids: Sequence[str], retrieved_texts: Sequence[str]) -> Tuple[bool, float]:
    expected = {str(eid) for eid in evidence_ids}
    if not expected:
        return False, 0.0
    joined = "\n".join(retrieved_texts)
    found = {eid for eid in expected if eid in joined}
    return bool(found), len(found) / len(expected)


def dynamic_retrieve(
    retriever: Any,
    query: str,
    evidence_ids: Sequence[str],
    category: Any,
    top_k: int,
    method: str = "dynamic_hierarchy",
) -> RetrievalMetrics:
    result = retriever.retrieve(query, evidence_ids=None, top_k=top_k)
    fact_texts = [
        retriever.memory.nodes[fact_id].text
        for fact_id in result.retrieved_fact_ids
    ]
    hit, recall = score_evidence(evidence_ids, fact_texts)
    return RetrievalMetrics(
        method=method,
        query=query,
        category=category,
        query_type=result.query_type,
        retrieved_fact_ids=result.retrieved_fact_ids,
        expanded_node_ids=result.expanded_node_ids,
        context_tokens=result.context_tokens,
        latency_ms=result.latency_ms,
        evidence_hit=hit,
        evidence_recall=recall,
        expanded_depth=result.expanded_depth,
        expanded_breadth=result.expanded_breadth,
    )


def global_fact_only_retrieve(
    retriever: Any,
    query: str,
    evidence_ids: Sequence[str],
    category: Any,
    budget: int,
    top_k: int,
    method: str,
) -> RetrievalMetrics:
    start = time.perf_counter()
    selected_facts: List[str] = []
    tokens = 0
    for fact_id, _score in retriever.fact_index.search(query, top_k=max(top_k * 16, 120)):
        fact_tokens = estimate_tokens(retriever.memory.nodes[fact_id].text)
        if tokens + fact_tokens <= budget:
            selected_facts.append(fact_id)
            tokens += fact_tokens
    fact_texts = [retriever.memory.nodes[fact_id].text for fact_id in selected_facts]
    hit, recall = score_evidence(evidence_ids, fact_texts)
    return RetrievalMetrics(
        method=method,
        query=query,
        category=category,
        query_type=DynamicRetriever.detect_query_type(query),
        retrieved_fact_ids=selected_facts,
        expanded_node_ids=[],
        context_tokens=tokens,
        latency_ms=(time.perf_counter() - start) * 1000.0,
        evidence_hit=hit,
        evidence_recall=recall,
        expanded_depth=0,
        expanded_breadth=0,
    )


def aggregate(method: str, rows: Sequence[RetrievalMetrics]) -> Dict[str, Any]:
    if not rows:
        return {
            "method": method,
            "num_questions": 0,
            "evidence_hit_rate": 0.0,
            "avg_evidence_recall": 0.0,
            "avg_context_tokens": 0.0,
            "avg_latency_ms": 0.0,
            "hit_per_1k_tokens": 0.0,
            "recall_per_1k_tokens": 0.0,
            "avg_expanded_depth": 0.0,
            "avg_expanded_breadth": 0.0,
        }
    hit_rate = statistics.mean(1.0 if row.evidence_hit else 0.0 for row in rows)
    recall = statistics.mean(row.evidence_recall for row in rows)
    avg_tokens = statistics.mean(row.context_tokens for row in rows)
    token_units = avg_tokens / 1000.0 if avg_tokens else 0.0
    return {
        "method": method,
        "num_questions": len(rows),
        "evidence_hit_rate": hit_rate,
        "avg_evidence_recall": recall,
        "avg_context_tokens": avg_tokens,
        "avg_latency_ms": statistics.mean(row.latency_ms for row in rows),
        "hit_per_1k_tokens": hit_rate / token_units if token_units else 0.0,
        "recall_per_1k_tokens": recall / token_units if token_units else 0.0,
        "avg_expanded_depth": statistics.mean(row.expanded_depth for row in rows),
        "avg_expanded_breadth": statistics.mean(row.expanded_breadth for row in rows),
    }


def category_aggregate(method: str, category: Any, rows: Sequence[RetrievalMetrics]) -> Dict[str, Any]:
    summary = aggregate(method, rows)
    return {
        "method": method,
        "category": category,
        "num_questions": summary["num_questions"],
        "evidence_hit_rate": summary["evidence_hit_rate"],
        "avg_evidence_recall": summary["avg_evidence_recall"],
        "avg_context_tokens": summary["avg_context_tokens"],
        "avg_latency_ms": summary["avg_latency_ms"],
        "hit_per_1k_tokens": summary["hit_per_1k_tokens"],
        "recall_per_1k_tokens": summary["recall_per_1k_tokens"],
    }


def summarize(name: str, rows: Sequence[RetrievalMetrics]) -> None:
    if not rows:
        print(f"{name}: no rows")
        return
    summary = aggregate(name, rows)
    print(f"{name}:")
    print(f"- questions: {summary['num_questions']}")
    print(f"- evidence_hit_rate: {summary['evidence_hit_rate']:.4f}")
    print(f"- avg_evidence_recall: {summary['avg_evidence_recall']:.4f}")
    print(f"- avg_context_tokens: {summary['avg_context_tokens']:.1f}")
    print(f"- avg_latency_ms: {summary['avg_latency_ms']:.2f}")
    print(f"- evidence_hit_per_1k_tokens: {summary['hit_per_1k_tokens']:.4f}")
    print(f"- recall_per_1k_tokens: {summary['recall_per_1k_tokens']:.4f}")
    print(f"- avg_expanded_depth: {summary['avg_expanded_depth']:.2f}")
    print(f"- avg_expanded_breadth: {summary['avg_expanded_breadth']:.2f}")


def summarize_by_category(name: str, rows: Sequence[RetrievalMetrics]) -> None:
    print(f"\n{name} by category:")
    categories = sorted({row.category for row in rows}, key=lambda value: str(value))
    for category in categories:
        subset = [row for row in rows if row.category == category]
        print(
            f"- category={category}: n={len(subset)}, "
            f"hit={statistics.mean(1.0 if row.evidence_hit else 0.0 for row in subset):.4f}, "
            f"recall={statistics.mean(row.evidence_recall for row in subset):.4f}, "
            f"tokens={statistics.mean(row.context_tokens for row in subset):.1f}"
        )


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def method_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for budget in parse_int_list(args.dynamic_budget_list):
        configs.append({
            "method": f"dynamic_{budget}",
            "budget": budget,
            "fallback": False,
            "temporal": False,
        })
    if args.enable_global_fallback:
        for budget in parse_int_list(args.hybrid_budget_list):
            configs.append({
                "method": f"dynamic_hybrid_{budget}",
                "budget": budget,
                "fallback": True,
                "temporal": False,
            })
    if args.enable_temporal_expansion:
        for budget in parse_int_list(args.hybrid_temporal_budget_list):
            configs.append({
                "method": f"dynamic_hybrid_temporal_gated_{budget}" if args.enable_global_fallback else f"dynamic_temporal_gated_{budget}",
                "budget": budget,
                "fallback": args.enable_global_fallback,
                "temporal": True,
                "fallback_ratio": args.fallback_ratio,
            })
    for config in configs:
        config.setdefault("fallback_ratio", args.fallback_ratio)
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for config in configs:
        if config["method"] in seen:
            continue
        seen.add(config["method"])
        deduped.append(config)
    return deduped


def fallback_ratio_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    configs = []
    for ratio in [float(item.strip()) for item in args.fallback_ratio_list.split(",") if item.strip()]:
        ratio_label = str(ratio).rstrip("0").rstrip(".")
        configs.append({
            "method": f"dynamic_hybrid_600_fr{ratio_label}",
            "budget": 600,
            "fallback": True,
            "temporal": False,
            "fallback_ratio": ratio,
        })
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data" / "locomo10.json")
    parser.add_argument("--limit-conv", type=int, default=10)
    parser.add_argument("--max-questions", type=int, default=0, help="0 means all non-category-5 questions")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-context-tokens", type=int, default=1200)
    parser.add_argument("--fixed-budget-list", default="300,400,600,800,1000")
    parser.add_argument("--global-fact-budget-list", default="400,600,800")
    parser.add_argument("--dynamic-budget-list", default="200,400,600,800,1000")
    parser.add_argument("--enable-global-fallback", action="store_true", default=True)
    parser.add_argument("--fallback-ratio", type=float, default=0.3)
    parser.add_argument("--fallback-ratio-list", default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--hybrid-budget-list", default="400,600")
    parser.add_argument("--enable-temporal-expansion", action="store_true", default=True)
    parser.add_argument("--temporal-max-nodes", type=int, default=5)
    parser.add_argument("--temporal-facts-per-node", type=int, default=2)
    parser.add_argument("--hybrid-temporal-budget-list", default="600")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    args = parser.parse_args()

    dataset = load_dataset(args.data_path)[: args.limit_conv]
    rows_by_method: Dict[str, List[RetrievalMetrics]] = {"fixed_default": []}
    fixed_budgets = parse_int_list(args.fixed_budget_list)
    global_fact_budgets = parse_int_list(args.global_fact_budget_list)
    for budget in fixed_budgets:
        rows_by_method[f"fixed_{budget}"] = []
    for budget in global_fact_budgets:
        rows_by_method[f"global_fact_only_{budget}"] = []

    configs = method_configs(args)
    ratio_configs = fallback_ratio_configs(args)
    all_dynamic_configs = configs + [
        config for config in ratio_configs
        if config["method"] not in {base["method"] for base in configs}
    ]
    for config in configs:
        rows_by_method[config["method"]] = []
    ratio_rows_by_method: Dict[str, List[RetrievalMetrics]] = {
        config["method"]: [] for config in ratio_configs
    }

    for conv_idx, item in enumerate(dataset):
        session_texts = build_session_texts(item)
        fixed_default = FixedSessionHyperMemBaseline(item, max_context_tokens=args.max_context_tokens)
        fixed_by_budget = {
            budget: FixedSessionHyperMemBaseline(item, max_context_tokens=budget)
            for budget in fixed_budgets
        }
        memory = DynamicHierarchyBuilder(
            max_depth=5,
            min_segment_size=2,
            max_leaf_tokens=260,
            split_gain_threshold=0.04,
            alpha=0.05,
            beta=0.1,
        ).build(session_texts)
        retrievers = {
            config["method"]: DynamicRetriever(
                memory,
                max_context_tokens=config["budget"],
                fallback_ratio=config["fallback_ratio"],
                enable_global_fact_fallback=config["fallback"],
                enable_temporal_expansion=config["temporal"],
                temporal_max_nodes=args.temporal_max_nodes,
                temporal_facts_per_node=args.temporal_facts_per_node,
            )
            for config in all_dynamic_configs
        }
        global_fact_retriever = DynamicRetriever(memory, max_context_tokens=args.max_context_tokens)

        qa_items = [qa for qa in item.get("qa", []) if qa.get("category") != 5 and qa.get("question")]
        if args.max_questions:
            qa_items = qa_items[: args.max_questions]

        for qa in qa_items:
            query = qa["question"]
            evidence_ids = [str(eid) for eid in qa.get("evidence", [])]
            category = qa.get("category")
            fixed_result = fixed_default.retrieve(query, evidence_ids, top_k=args.top_k)
            fixed_result.category = category
            fixed_result.method = "fixed_default"
            rows_by_method["fixed_default"].append(fixed_result)
            for budget, fixed in fixed_by_budget.items():
                fixed_budget_result = fixed.retrieve(query, evidence_ids, top_k=args.top_k)
                fixed_budget_result.category = category
                fixed_budget_result.method = f"fixed_{budget}"
                rows_by_method[f"fixed_{budget}"].append(fixed_budget_result)

            for budget in global_fact_budgets:
                method = f"global_fact_only_{budget}"
                rows_by_method[method].append(
                    global_fact_only_retrieve(
                        global_fact_retriever,
                        query,
                        evidence_ids,
                        category,
                        budget,
                        args.top_k,
                        method,
                    )
                )

            for config in configs:
                method = config["method"]
                dynamic_result = dynamic_retrieve(
                    retrievers[method],
                    query,
                    evidence_ids,
                    category,
                    args.top_k,
                    method=method,
                )
                rows_by_method[method].append(dynamic_result)
            for config in ratio_configs:
                method = config["method"]
                ratio_rows_by_method[method].append(
                    dynamic_retrieve(
                        retrievers[method],
                        query,
                        evidence_ids,
                        category,
                        args.top_k,
                        method=method,
                    )
                )

        print(
            f"processed conv={conv_idx} sample_id={item.get('sample_id')} "
            f"sessions={len(session_texts)} questions={len(qa_items)} "
            f"dynamic_nodes={len(memory.nodes)} dynamic_facts={len(memory.facts)}"
        )

    print("\nSummary table:")
    preferred_order = [
        "fixed_default",
        "fixed_300",
        "fixed_400",
        "fixed_600",
        "fixed_800",
        "fixed_1000",
        "global_fact_only_400",
        "global_fact_only_600",
        "global_fact_only_800",
        "dynamic_200",
        "dynamic_400",
        "dynamic_600",
        "dynamic_800",
        "dynamic_1000",
        "dynamic_hybrid_400",
        "dynamic_hybrid_600",
        "dynamic_hybrid_temporal_gated_600",
    ]
    summary_rows = [
        aggregate(method, rows_by_method[method])
        for method in preferred_order
        if method in rows_by_method
    ]
    fields = [
        "method",
        "num_questions",
        "evidence_hit_rate",
        "avg_evidence_recall",
        "avg_context_tokens",
        "avg_latency_ms",
        "hit_per_1k_tokens",
        "recall_per_1k_tokens",
        "avg_expanded_depth",
        "avg_expanded_breadth",
    ]
    print("\t".join(fields[:8]))
    for row in summary_rows:
        print(
            f"{row['method']}\t{row['num_questions']}\t"
            f"{row['evidence_hit_rate']:.4f}\t{row['avg_evidence_recall']:.4f}\t"
            f"{row['avg_context_tokens']:.1f}\t{row['avg_latency_ms']:.2f}\t"
            f"{row['hit_per_1k_tokens']:.4f}\t{row['recall_per_1k_tokens']:.4f}"
        )

    main_category_methods = [
        method for method in preferred_order
        if method in rows_by_method and method in {
            "fixed_default",
            "fixed_400",
            "fixed_600",
            "global_fact_only_400",
            "global_fact_only_600",
            "dynamic_400",
            "dynamic_600",
            "dynamic_hybrid_400",
            "dynamic_hybrid_600",
            "dynamic_hybrid_temporal_gated_600",
        }
    ]
    for method in main_category_methods:
        rows = rows_by_method[method]
        summarize_by_category(method, rows)

    category_rows: List[Dict[str, Any]] = []
    for method in main_category_methods:
        rows = rows_by_method[method]
        categories = sorted({row.category for row in rows}, key=lambda value: str(value))
        for category in categories:
            subset = [row for row in rows if row.category == category]
            category_rows.append(category_aggregate(method, category, subset))

    ratio_summary_rows = [
        aggregate(method, rows)
        for method, rows in ratio_rows_by_method.items()
    ]

    sweep_path = args.output_dir / "locomo_dynamic_budget_sweep.csv"
    category_path = args.output_dir / "locomo_retrieval_by_category.csv"
    ratio_path = args.output_dir / "locomo_fallback_ratio_sweep.csv"
    write_csv(sweep_path, summary_rows, fields)
    write_csv(
        category_path,
        category_rows,
        [
            "method",
            "category",
            "num_questions",
            "evidence_hit_rate",
            "avg_evidence_recall",
            "avg_context_tokens",
            "avg_latency_ms",
            "hit_per_1k_tokens",
            "recall_per_1k_tokens",
        ],
    )
    write_csv(ratio_path, ratio_summary_rows, fields)
    print(f"\nSaved summary CSV: {sweep_path}")
    print(f"Saved category CSV: {category_path}")
    print(f"Saved fallback ratio CSV: {ratio_path}")

    print("\nSample misses where fixed hit but dynamic missed:")
    fixed_rows = rows_by_method["fixed_default"]
    dynamic_rows = rows_by_method.get("dynamic_600") or next(
        (rows for method, rows in rows_by_method.items() if method.startswith("dynamic_")),
        [],
    )
    misses = [
        (fixed, dynamic)
        for fixed, dynamic in zip(fixed_rows, dynamic_rows)
        if fixed.evidence_hit and not dynamic.evidence_hit
    ][:5]
    for fixed, dynamic in misses:
        print(f"- q={fixed.query!r} category={fixed.category} fixed_tokens={fixed.context_tokens} dynamic_tokens={dynamic.context_tokens}")


if __name__ == "__main__":
    main()
