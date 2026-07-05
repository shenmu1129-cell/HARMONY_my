#!/usr/bin/env python
"""
Retrieval-only evaluation for the verifier-guided adaptive memory controller.

This script does not call any LLM, does not generate answers, and does not run
HyperMem stages 5/6.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCOMO_EVAL_PATH = REPO_ROOT / "examples" / "locomo_retrieval_eval.py"
locomo_spec = importlib.util.spec_from_file_location("locomo_retrieval_eval", LOCOMO_EVAL_PATH)
locomo_eval = importlib.util.module_from_spec(locomo_spec)
sys.modules["locomo_retrieval_eval"] = locomo_eval
assert locomo_spec and locomo_spec.loader
locomo_spec.loader.exec_module(locomo_eval)

from hypermem.experimental.adaptive_controller import (  # noqa: E402
    COREFERENCE_KEYWORDS,
    SUMMARY_KEYWORDS,
    TEMPORAL_KEYWORDS,
    WHY_HOW_KEYWORDS,
    AdaptiveMemoryController,
    trace_to_dict,
)


DynamicHierarchyBuilder = locomo_eval.DynamicHierarchyBuilder
DynamicRetriever = locomo_eval.DynamicRetriever
FixedSessionHyperMemBaseline = locomo_eval.FixedSessionHyperMemBaseline
RetrievalMetrics = locomo_eval.RetrievalMetrics
SimpleTfidfIndex = locomo_eval.SimpleTfidfIndex
build_session_texts = locomo_eval.build_session_texts
global_fact_only_retrieve = locomo_eval.global_fact_only_retrieve
load_dataset = locomo_eval.load_dataset
score_evidence = locomo_eval.score_evidence
write_csv = locomo_eval.write_csv


COMPARE_KEYWORDS = {
    "compare", "versus", "vs", "different", "difference", "same", "比较", "区别", "差异",
}


def contains_any(text: str, keywords: Iterable[str]) -> int:
    lowered = text.lower()
    return int(any(keyword in lowered for keyword in keywords))


def reward_for(hit: bool, recall: float, tokens: int, hit_weight: float, token_penalty: float) -> float:
    return float(recall) + hit_weight * float(hit) - token_penalty * float(tokens) / 1000.0


def metric_row_from_retrieval(
    method: str,
    conv_id: int,
    question_id: int,
    question: str,
    category: Any,
    metrics: RetrievalMetrics,
    hit_weight: float,
    token_penalty: float,
    steps: int = 1,
) -> Dict[str, Any]:
    reward = reward_for(metrics.evidence_hit, metrics.evidence_recall, metrics.context_tokens, hit_weight, token_penalty)
    return {
        "method": method,
        "conv_id": conv_id,
        "question_id": question_id,
        "category": category,
        "question": question,
        "query_type": metrics.query_type,
        "evidence_hit": int(metrics.evidence_hit),
        "evidence_recall": metrics.evidence_recall,
        "context_tokens": metrics.context_tokens,
        "latency_ms": metrics.latency_ms,
        "steps": steps,
        "reward": reward,
    }


def aggregate(method: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "method": method,
            "num_questions": 0,
            "evidence_hit_rate": 0.0,
            "avg_evidence_recall": 0.0,
            "avg_context_tokens": 0.0,
            "avg_steps": 0.0,
            "avg_reward": 0.0,
            "avg_latency_ms": 0.0,
            "hit_per_1k_tokens": 0.0,
            "recall_per_1k_tokens": 0.0,
        }
    hit_rate = statistics.mean(float(row["evidence_hit"]) for row in rows)
    recall = statistics.mean(float(row["evidence_recall"]) for row in rows)
    avg_tokens = statistics.mean(float(row["context_tokens"]) for row in rows)
    token_units = avg_tokens / 1000.0 if avg_tokens else 0.0
    return {
        "method": method,
        "num_questions": len(rows),
        "evidence_hit_rate": hit_rate,
        "avg_evidence_recall": recall,
        "avg_context_tokens": avg_tokens,
        "avg_steps": statistics.mean(float(row["steps"]) for row in rows),
        "avg_reward": statistics.mean(float(row["reward"]) for row in rows),
        "avg_latency_ms": statistics.mean(float(row["latency_ms"]) for row in rows),
        "hit_per_1k_tokens": hit_rate / token_units if token_units else 0.0,
        "recall_per_1k_tokens": recall / token_units if token_units else 0.0,
    }


def category_aggregate(method: str, category: Any, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary = aggregate(method, rows)
    return {
        "method": method,
        "category": category,
        "num_questions": summary["num_questions"],
        "evidence_hit_rate": summary["evidence_hit_rate"],
        "avg_evidence_recall": summary["avg_evidence_recall"],
        "avg_context_tokens": summary["avg_context_tokens"],
        "avg_steps": summary["avg_steps"],
        "avg_reward": summary["avg_reward"],
    }


def build_query_features(
    conv_id: int,
    question_id: int,
    category: Any,
    query: str,
    global_retriever: Any,
) -> Dict[str, Any]:
    tokens = SimpleTfidfIndex.tokenize(query)
    fact_scores = [score for _fact_id, score in global_retriever.fact_index.search(query, top_k=8)]
    topic_scores = [score for _node_id, score in global_retriever.node_index.search(query, top_k=5)]
    top1_fact = fact_scores[0] if fact_scores else 0.0
    top2_fact = fact_scores[1] if len(fact_scores) > 1 else 0.0
    top_topic = topic_scores[0] if topic_scores else 0.0
    top_topic_2 = topic_scores[1] if len(topic_scores) > 1 else 0.0
    return {
        "conv_id": conv_id,
        "question_id": question_id,
        "category": category,
        "query_length": len(query),
        "num_terms": len(tokens),
        "has_temporal_keyword": contains_any(query, TEMPORAL_KEYWORDS),
        "has_summary_keyword": contains_any(query, SUMMARY_KEYWORDS),
        "has_why_how_keyword": contains_any(query, WHY_HOW_KEYWORDS),
        "has_compare_keyword": contains_any(query, COMPARE_KEYWORDS),
        "has_coreference_keyword": contains_any(query, COREFERENCE_KEYWORDS),
        "top1_global_fact_score": top1_fact,
        "top2_global_fact_score": top2_fact,
        "top1_top2_margin": top1_fact - top2_fact,
        "top_topic_score": top_topic,
        "top_topic_margin": top_topic - top_topic_2,
        "predicted_query_type": DynamicRetriever.detect_query_type(query),
    }


def feature_matrix(feature_rows: Sequence[Dict[str, Any]], type_vocab: Sequence[str]) -> np.ndarray:
    numeric_fields = [
        "query_length",
        "num_terms",
        "has_temporal_keyword",
        "has_summary_keyword",
        "has_why_how_keyword",
        "has_compare_keyword",
        "has_coreference_keyword",
        "top1_global_fact_score",
        "top2_global_fact_score",
        "top1_top2_margin",
        "top_topic_score",
        "top_topic_margin",
        "category",
    ]
    matrix = []
    for row in feature_rows:
        values = [float(row[field]) for field in numeric_fields]
        values.extend(1.0 if str(row["predicted_query_type"]) == query_type else 0.0 for query_type in type_vocab)
        matrix.append(values)
    return np.array(matrix, dtype=np.float64)


def rows_by_key(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    return {
        (int(row["conv_id"]), int(row["question_id"])): row
        for row in rows
    }


def train_override_logreg(
    fixed_rows: Sequence[Dict[str, Any]],
    global_rows: Sequence[Dict[str, Any]],
    feature_rows: Sequence[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    fixed_by_key = rows_by_key(fixed_rows)
    global_by_key = rows_by_key(global_rows)
    type_vocab = sorted({str(row["predicted_query_type"]) for row in feature_rows})
    conv_ids = sorted({int(row["conv_id"]) for row in feature_rows})
    selected_rows: List[Dict[str, Any]] = []

    for heldout_conv in conv_ids:
        train_features = [row for row in feature_rows if int(row["conv_id"]) != heldout_conv]
        test_features = [row for row in feature_rows if int(row["conv_id"]) == heldout_conv]
        x_train = feature_matrix(train_features, type_vocab)
        x_test = feature_matrix(test_features, type_vocab)
        y_train = []
        for feature in train_features:
            key = (int(feature["conv_id"]), int(feature["question_id"]))
            gain = float(global_by_key[key]["reward"]) - float(fixed_by_key[key]["reward"])
            y_train.append(int(gain > threshold))

        if len(set(y_train)) < 2:
            predictions = np.full(len(test_features), int(y_train[0]) if y_train else 0)
        else:
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
            model.fit(x_train_scaled, y_train)
            predictions = model.predict(x_test_scaled)

        for feature, prediction in zip(test_features, predictions):
            key = (int(feature["conv_id"]), int(feature["question_id"]))
            source = global_by_key[key] if int(prediction) else fixed_by_key[key]
            row = dict(source)
            row["method"] = "override_logreg"
            row["steps"] = 1
            selected_rows.append(row)
    selected_rows.sort(key=lambda row: (int(row["conv_id"]), int(row["question_id"])))
    return selected_rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data" / "locomo10.json")
    parser.add_argument("--limit-conv", type=int, default=0, help="0 means all available conversations")
    parser.add_argument("--max-questions", type=int, default=0, help="0 means all non-category-5 questions")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--adaptive-max-context-tokens", type=int, default=800)
    parser.add_argument("--adaptive-max-steps", type=int, default=4)
    parser.add_argument("--reward-hit-weight", type=float, default=0.2)
    parser.add_argument("--reward-token-penalty", type=float, default=0.1)
    parser.add_argument("--override-threshold", type=float, default=0.03)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    args = parser.parse_args()

    dataset_all = load_dataset(args.data_path)
    dataset = dataset_all[: args.limit_conv] if args.limit_conv else dataset_all

    rows_by_method: Dict[str, List[Dict[str, Any]]] = {
        "fixed_400": [],
        "global_fact_only_800": [],
        "adaptive_controller_v1": [],
    }
    feature_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    for conv_id, item in enumerate(dataset):
        session_texts = build_session_texts(item)
        fixed_400 = FixedSessionHyperMemBaseline(item, max_context_tokens=400)
        memory = DynamicHierarchyBuilder(
            max_depth=5,
            min_segment_size=2,
            max_leaf_tokens=260,
            split_gain_threshold=0.04,
            alpha=0.05,
            beta=0.1,
        ).build(session_texts)
        global_retriever = DynamicRetriever(memory, max_context_tokens=1000)
        controller = AdaptiveMemoryController(
            memory,
            max_context_tokens=args.adaptive_max_context_tokens,
            max_steps=args.adaptive_max_steps,
        )

        qa_items = [qa for qa in item.get("qa", []) if qa.get("category") != 5 and qa.get("question")]
        if args.max_questions:
            qa_items = qa_items[: args.max_questions]

        for question_id, qa in enumerate(qa_items):
            query = qa["question"]
            category = qa.get("category")
            evidence_ids = [str(eid) for eid in qa.get("evidence", [])]
            feature_rows.append(build_query_features(conv_id, question_id, category, query, global_retriever))

            fixed_metrics = fixed_400.retrieve(query, evidence_ids, top_k=args.top_k)
            fixed_metrics.category = category
            rows_by_method["fixed_400"].append(metric_row_from_retrieval(
                "fixed_400",
                conv_id,
                question_id,
                query,
                category,
                fixed_metrics,
                args.reward_hit_weight,
                args.reward_token_penalty,
            ))

            global_metrics = global_fact_only_retrieve(
                global_retriever,
                query,
                evidence_ids,
                category,
                800,
                args.top_k,
                "global_fact_only_800",
            )
            rows_by_method["global_fact_only_800"].append(metric_row_from_retrieval(
                "global_fact_only_800",
                conv_id,
                question_id,
                query,
                category,
                global_metrics,
                args.reward_hit_weight,
                args.reward_token_penalty,
            ))

            adaptive_result = controller.run(query, top_k=args.top_k)
            adaptive_texts = [memory.nodes[fact_id].text for fact_id in adaptive_result.selected_fact_ids]
            adaptive_hit, adaptive_recall = score_evidence(evidence_ids, adaptive_texts)
            adaptive_reward = reward_for(
                adaptive_hit,
                adaptive_recall,
                adaptive_result.context_tokens,
                args.reward_hit_weight,
                args.reward_token_penalty,
            )
            rows_by_method["adaptive_controller_v1"].append({
                "method": "adaptive_controller_v1",
                "conv_id": conv_id,
                "question_id": question_id,
                "category": category,
                "question": query,
                "query_type": adaptive_result.query_type,
                "evidence_hit": int(adaptive_hit),
                "evidence_recall": adaptive_recall,
                "context_tokens": adaptive_result.context_tokens,
                "latency_ms": adaptive_result.latency_ms,
                "steps": len(adaptive_result.steps),
                "reward": adaptive_reward,
            })

            trace_payload = trace_to_dict(adaptive_result, memory)
            trace_payload.update({
                "conv_id": conv_id,
                "question_id": question_id,
                "question": query,
                "category": category,
                "evidence_hit": int(adaptive_hit),
                "evidence_recall": adaptive_recall,
                "reward": adaptive_reward,
            })
            trace_rows.append(trace_payload)

        print(
            f"processed conv={conv_id} sample_id={item.get('sample_id')} "
            f"questions={len(qa_items)} dynamic_nodes={len(memory.nodes)} dynamic_facts={len(memory.facts)}"
        )

    rows_by_method["override_logreg"] = train_override_logreg(
        rows_by_method["fixed_400"],
        rows_by_method["global_fact_only_800"],
        feature_rows,
        threshold=args.override_threshold,
    )

    summary_order = [
        "fixed_400",
        "global_fact_only_800",
        "override_logreg",
        "adaptive_controller_v1",
    ]
    summary_rows = [aggregate(method, rows_by_method[method]) for method in summary_order]

    category_rows: List[Dict[str, Any]] = []
    for method in summary_order:
        rows = rows_by_method[method]
        for category in sorted({row["category"] for row in rows}, key=lambda value: str(value)):
            subset = [row for row in rows if row["category"] == category]
            category_rows.append(category_aggregate(method, category, subset))

    result_fields = [
        "method",
        "num_questions",
        "evidence_hit_rate",
        "avg_evidence_recall",
        "avg_context_tokens",
        "avg_steps",
        "avg_reward",
        "avg_latency_ms",
        "hit_per_1k_tokens",
        "recall_per_1k_tokens",
    ]
    category_fields = [
        "method",
        "category",
        "num_questions",
        "evidence_hit_rate",
        "avg_evidence_recall",
        "avg_context_tokens",
        "avg_steps",
        "avg_reward",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "adaptive_controller_results.csv", summary_rows, result_fields)
    write_csv(args.output_dir / "adaptive_controller_by_category.csv", category_rows, category_fields)
    write_jsonl(args.output_dir / "adaptive_controller_action_trace.jsonl", trace_rows)

    print("\nAdaptive controller retrieval-only summary:")
    print("\t".join(result_fields))
    for row in summary_rows:
        print(
            f"{row['method']}\t{row['num_questions']}\t"
            f"{row['evidence_hit_rate']:.4f}\t{row['avg_evidence_recall']:.4f}\t"
            f"{row['avg_context_tokens']:.1f}\t{row['avg_steps']:.2f}\t"
            f"{row['avg_reward']:.4f}\t{row['avg_latency_ms']:.2f}\t"
            f"{row['hit_per_1k_tokens']:.4f}\t{row['recall_per_1k_tokens']:.4f}"
        )
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
