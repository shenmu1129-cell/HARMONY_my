#!/usr/bin/env python
"""
Policy routing prototype for retrieval-only LoCoMo action selection.

This script reuses the retrieval baselines from `locomo_retrieval_eval.py`.
It does not call any LLM, does not generate answers, and does not run
HyperMem stages 5/6.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


REPO_ROOT = Path(__file__).resolve().parent.parent

LOCOMO_EVAL_PATH = REPO_ROOT / "examples" / "locomo_retrieval_eval.py"
locomo_spec = importlib.util.spec_from_file_location("locomo_retrieval_eval", LOCOMO_EVAL_PATH)
locomo_eval = importlib.util.module_from_spec(locomo_spec)
sys.modules["locomo_retrieval_eval"] = locomo_eval
assert locomo_spec and locomo_spec.loader
locomo_spec.loader.exec_module(locomo_eval)

DynamicHierarchyBuilder = locomo_eval.DynamicHierarchyBuilder
DynamicRetriever = locomo_eval.DynamicRetriever
FixedSessionHyperMemBaseline = locomo_eval.FixedSessionHyperMemBaseline
RetrievalMetrics = locomo_eval.RetrievalMetrics
build_session_texts = locomo_eval.build_session_texts
dynamic_retrieve = locomo_eval.dynamic_retrieve
global_fact_only_retrieve = locomo_eval.global_fact_only_retrieve
load_dataset = locomo_eval.load_dataset
score_evidence = locomo_eval.score_evidence
write_csv = locomo_eval.write_csv


ACTION_SETS = {
    "minimal": [
        "fixed_400",
        "global_fact_only_400",
        "global_fact_only_800",
    ],
    "compact": [
        "fixed_400",
        "global_fact_only_400",
        "global_fact_only_800",
        "dynamic_hybrid_600_fr0.5",
    ],
}

OVERRIDE_THRESHOLDS = [0.03, 0.05, 0.08]
REWARD_TOKEN_PENALTIES = [0.00, 0.02, 0.05, 0.10, 0.15]

TEMPORAL_KEYWORDS = {
    "before", "after", "later", "earlier", "when", "timeline", "change", "evolve",
    "over time", "之前", "后来", "之后", "变化", "逐渐", "过程", "时间线", "什么时候", "先后",
}
SUMMARY_KEYWORDS = {"summarize", "summary", "overall", "all", "list", "哪些", "总结", "整体", "有哪些"}
WHY_HOW_KEYWORDS = {"why", "how", "reason", "because", "为什么", "如何", "怎么", "原因"}
COMPARE_KEYWORDS = {"compare", "versus", "vs", "different", "same", "比较", "区别", "差异"}
PREFERENCE_KEYWORDS = {"prefer", "favorite", "like", "enjoy", "want", "would", "偏好", "喜欢", "想"}


def contains_any(query: str, keywords: Iterable[str]) -> int:
    q = query.lower()
    return int(any(keyword in q for keyword in keywords))


def parse_float_list(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def selected_actions(action_set: str) -> List[str]:
    if action_set not in ACTION_SETS:
        raise ValueError(f"unknown action set: {action_set}")
    return ACTION_SETS[action_set]


def entropy(scores: Sequence[float]) -> float:
    positive = np.array([max(0.0, score) for score in scores], dtype=np.float64)
    total = float(positive.sum())
    if total <= 0:
        return 0.0
    probs = positive / total
    return float(-(probs * np.log(probs + 1e-12)).sum())


def reward_for(row: Dict[str, Any], hit_weight: float, token_penalty: float) -> float:
    return (
        float(row["evidence_recall"])
        + hit_weight * float(row["evidence_hit"])
        - token_penalty * float(row["context_tokens"]) / 1000.0
    )


def metric_summary(rows: Sequence[Dict[str, Any]], reward_col: str = "reward") -> Dict[str, float]:
    if not rows:
        return {"hit_rate": 0.0, "avg_recall": 0.0, "avg_tokens": 0.0, "avg_reward": 0.0}
    return {
        "hit_rate": float(np.mean([float(row["evidence_hit"]) for row in rows])),
        "avg_recall": float(np.mean([float(row["evidence_recall"]) for row in rows])),
        "avg_tokens": float(np.mean([float(row["context_tokens"]) for row in rows])),
        "avg_reward": float(np.mean([float(row[reward_col]) for row in rows])),
    }


def build_query_features(
    conv_id: int,
    question_id: int,
    category: Any,
    query: str,
    global_retriever: Any,
) -> Dict[str, Any]:
    words = locomo_eval.SimpleTfidfIndex.tokenize(query)
    fact_hits = global_retriever.fact_index.search(query, top_k=10)
    fact_scores = [score for _fact_id, score in fact_hits]
    top1_fact = fact_scores[0] if len(fact_scores) >= 1 else 0.0
    top2_fact = fact_scores[1] if len(fact_scores) >= 2 else 0.0

    topic_hits = global_retriever.node_index.search(query, top_k=5)
    topic_scores = [score for _node_id, score in topic_hits]
    top_topic = topic_scores[0] if len(topic_scores) >= 1 else 0.0
    top_topic_2 = topic_scores[1] if len(topic_scores) >= 2 else 0.0
    predicted_type = DynamicRetriever.detect_query_type(query)

    return {
        "conv_id": conv_id,
        "question_id": question_id,
        "category": category,
        "question": query,
        "query_length": len(query),
        "num_words_chars": len(words),
        "has_temporal_keyword": contains_any(query, TEMPORAL_KEYWORDS),
        "has_summary_keyword": contains_any(query, SUMMARY_KEYWORDS),
        "has_why_how_keyword": contains_any(query, WHY_HOW_KEYWORDS),
        "has_compare_keyword": contains_any(query, COMPARE_KEYWORDS),
        "has_preference_keyword": contains_any(query, PREFERENCE_KEYWORDS),
        "top1_global_fact_score": top1_fact,
        "top2_global_fact_score": top2_fact,
        "top1_top2_margin": top1_fact - top2_fact,
        "global_fact_score_entropy": entropy(fact_scores),
        "top_topic_score": top_topic,
        "top_topic_margin": top_topic - top_topic_2,
        "predicted_query_type": predicted_type,
    }


def build_action_matrix(
    dataset: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    action_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    actions = set(selected_actions(args.action_set))

    for conv_id, item in enumerate(dataset):
        session_texts = build_session_texts(item)
        fixed_400 = (
            FixedSessionHyperMemBaseline(item, max_context_tokens=400)
            if "fixed_400" in actions
            else None
        )
        memory = DynamicHierarchyBuilder(
            max_depth=5,
            min_segment_size=2,
            max_leaf_tokens=260,
            split_gain_threshold=0.04,
            alpha=0.05,
            beta=0.1,
        ).build(session_texts)
        global_retriever = DynamicRetriever(memory, max_context_tokens=1000)
        hybrid_05 = None
        if "dynamic_hybrid_600_fr0.5" in actions:
            hybrid_05 = DynamicRetriever(
                memory,
                max_context_tokens=600,
                fallback_ratio=0.5,
                enable_global_fact_fallback=True,
            )

        qa_items = [qa for qa in item.get("qa", []) if qa.get("category") != 5 and qa.get("question")]
        if args.max_questions:
            qa_items = qa_items[: args.max_questions]

        for question_id, qa in enumerate(qa_items):
            query = qa["question"]
            category = qa.get("category")
            evidence_ids = [str(eid) for eid in qa.get("evidence", [])]
            feature_rows.append(build_query_features(conv_id, question_id, category, query, global_retriever))

            metrics_by_action: Dict[str, RetrievalMetrics] = {}
            if fixed_400 is not None:
                metrics_by_action["fixed_400"] = fixed_400.retrieve(query, evidence_ids, top_k=args.top_k)
            if "global_fact_only_400" in actions:
                metrics_by_action["global_fact_only_400"] = global_fact_only_retrieve(
                    global_retriever,
                    query,
                    evidence_ids,
                    category,
                    400,
                    args.top_k,
                    "global_fact_only_400",
                )
            if "global_fact_only_800" in actions:
                metrics_by_action["global_fact_only_800"] = global_fact_only_retrieve(
                    global_retriever,
                    query,
                    evidence_ids,
                    category,
                    800,
                    args.top_k,
                    "global_fact_only_800",
                )
            if hybrid_05 is not None:
                metrics_by_action["dynamic_hybrid_600_fr0.5"] = dynamic_retrieve(
                    hybrid_05,
                    query,
                    evidence_ids,
                    category,
                    args.top_k,
                    "dynamic_hybrid_600_fr0.5",
                )
            for action, metrics in metrics_by_action.items():
                action_rows.append({
                    "conv_id": conv_id,
                    "question_id": question_id,
                    "category": category,
                    "question": query,
                    "action": action,
                    "evidence_hit": int(metrics.evidence_hit),
                    "evidence_recall": metrics.evidence_recall,
                    "context_tokens": metrics.context_tokens,
                    "latency_ms": metrics.latency_ms,
                })

        print(f"processed conv={conv_id} sample_id={item.get('sample_id')} questions={len(qa_items)}")

    return action_rows, feature_rows


def rows_by_question(action_rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        grouped[(int(row["conv_id"]), int(row["question_id"]))].append(row)
    return grouped


def compute_oracle(
    action_rows: List[Dict[str, Any]],
    hit_weight: float,
    token_penalty: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    for row in action_rows:
        row["reward"] = reward_for(row, hit_weight, token_penalty)

    grouped = rows_by_question(action_rows)
    oracle_rows: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        best = max(rows, key=lambda row: float(row["reward"]))
        oracle_rows.append(dict(best))

    by_action = defaultdict(list)
    for row in action_rows:
        by_action[row["action"]].append(row)
    action_rewards = {
        action: float(np.mean([float(row["reward"]) for row in rows]))
        for action, rows in by_action.items()
    }
    best_single_action = max(action_rewards, key=action_rewards.get)
    distribution = dict(Counter(row["action"] for row in oracle_rows))
    oracle_metrics = metric_summary(oracle_rows)

    summary = {
        "best_single_action": best_single_action,
        "best_single_action_reward": action_rewards[best_single_action],
        "oracle_policy_reward": oracle_metrics["avg_reward"],
        "oracle_hit_rate": oracle_metrics["hit_rate"],
        "oracle_recall": oracle_metrics["avg_recall"],
        "oracle_avg_tokens": oracle_metrics["avg_tokens"],
        "oracle_action_distribution": json.dumps(distribution, ensure_ascii=False, sort_keys=True),
    }
    return oracle_rows, summary


def feature_matrix(feature_rows: Sequence[Dict[str, Any]], type_vocab: Sequence[str] | None = None) -> Tuple[np.ndarray, List[str]]:
    numeric_fields = [
        "query_length",
        "num_words_chars",
        "has_temporal_keyword",
        "has_summary_keyword",
        "has_why_how_keyword",
        "has_compare_keyword",
        "has_preference_keyword",
        "top1_global_fact_score",
        "top2_global_fact_score",
        "top1_top2_margin",
        "global_fact_score_entropy",
        "top_topic_score",
        "top_topic_margin",
    ]
    if type_vocab is None:
        type_vocab = sorted({str(row["predicted_query_type"]) for row in feature_rows})
    matrix = []
    for row in feature_rows:
        values = [float(row[field]) for field in numeric_fields]
        values.extend(1.0 if row["predicted_query_type"] == query_type else 0.0 for query_type in type_vocab)
        values.append(float(row["category"]))
        matrix.append(values)
    names = numeric_fields + [f"type_{query_type}" for query_type in type_vocab] + ["category"]
    return np.array(matrix, dtype=np.float64), names


def action_lookup(action_rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int, str], Dict[str, Any]]:
    return {
        (int(row["conv_id"]), int(row["question_id"]), str(row["action"])): row
        for row in action_rows
    }


def evaluate_chosen_actions(
    name: str,
    feature_rows: Sequence[Dict[str, Any]],
    chosen_actions: Sequence[str],
    oracle_action_by_question: Dict[Tuple[int, int], str],
    lookup: Dict[Tuple[int, int, str], Dict[str, Any]],
    split_name: str,
) -> Dict[str, Any]:
    selected = []
    oracle_labels = []
    for row, action in zip(feature_rows, chosen_actions):
        key = (int(row["conv_id"]), int(row["question_id"]))
        selected.append(lookup[(key[0], key[1], action)])
        oracle_labels.append(oracle_action_by_question[key])
    summary = metric_summary(selected)
    return {
        "split": split_name,
        "model": name,
        "policy_reward": summary["avg_reward"],
        "policy_hit_rate": summary["hit_rate"],
        "policy_recall": summary["avg_recall"],
        "policy_avg_tokens": summary["avg_tokens"],
        "action_accuracy": float(accuracy_score(oracle_labels, list(chosen_actions))),
        "chosen_action_distribution": json.dumps(dict(Counter(chosen_actions)), sort_keys=True),
    }


def split_feature_rows(feature_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    conv_ids = sorted({int(row["conv_id"]) for row in feature_rows})
    if not conv_ids:
        return [], [], "empty"
    if max(conv_ids) >= 9:
        train_ids = {conv_id for conv_id in conv_ids if conv_id <= 7}
        test_ids = {conv_id for conv_id in conv_ids if conv_id >= 8}
        split_name = "conv0-7_train_conv8-9_test"
    else:
        test_count = max(1, math.ceil(len(conv_ids) * 0.2))
        test_ids = set(conv_ids[-test_count:])
        train_ids = set(conv_ids[:-test_count])
        if not train_ids:
            train_ids = set(conv_ids[:1])
        if train_ids == test_ids and len(conv_ids) > 1:
            train_ids = set(conv_ids[:-1])
            test_ids = {conv_ids[-1]}
        split_name = f"conv{min(train_ids)}-{max(train_ids)}_train_conv{min(test_ids)}-{max(test_ids)}_test"
    return (
        [row for row in feature_rows if int(row["conv_id"]) in train_ids],
        [row for row in feature_rows if int(row["conv_id"]) in test_ids],
        split_name,
    )


def train_supervised_policies(
    action_rows: Sequence[Dict[str, Any]],
    feature_rows: Sequence[Dict[str, Any]],
    oracle_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    oracle_action_by_question = {
        (int(row["conv_id"]), int(row["question_id"])): str(row["action"])
        for row in oracle_rows
    }
    train_features, test_features, split_name = split_feature_rows(feature_rows)
    type_vocab = sorted({str(row["predicted_query_type"]) for row in feature_rows})
    x_train, _ = feature_matrix(train_features, type_vocab)
    x_test, _ = feature_matrix(test_features, type_vocab)
    y_train = [oracle_action_by_question[(int(row["conv_id"]), int(row["question_id"]))] for row in train_features]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    lookup = action_lookup(action_rows)

    results = []
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(x_train_scaled, y_train)
    results.append(evaluate_chosen_actions(
        "supervised_policy_logreg",
        test_features,
        logreg.predict(x_test_scaled),
        oracle_action_by_question,
        lookup,
        split_name,
    ))

    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=13, class_weight="balanced")
    rf.fit(x_train, y_train)
    results.append(evaluate_chosen_actions(
        "supervised_policy_rf",
        test_features,
        rf.predict(x_test),
        oracle_action_by_question,
        lookup,
        split_name,
    ))
    return results


def train_bandit_regressor(
    action_rows: Sequence[Dict[str, Any]],
    feature_rows: Sequence[Dict[str, Any]],
    oracle_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    oracle_action_by_question = {
        (int(row["conv_id"]), int(row["question_id"])): str(row["action"])
        for row in oracle_rows
    }
    train_features, test_features, split_name = split_feature_rows(feature_rows)
    type_vocab = sorted({str(row["predicted_query_type"]) for row in feature_rows})
    x_train, _ = feature_matrix(train_features, type_vocab)
    x_test, _ = feature_matrix(test_features, type_vocab)
    lookup = action_lookup(action_rows)

    reward_by_key = {
        (int(row["conv_id"]), int(row["question_id"]), str(row["action"])): float(row["reward"])
        for row in action_rows
    }
    actions = sorted({str(row["action"]) for row in action_rows})
    predictions = {}
    for action in actions:
        y_train = [
            reward_by_key[(int(row["conv_id"]), int(row["question_id"]), action)]
            for row in train_features
        ]
        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=17)
        model.fit(x_train, y_train)
        predictions[action] = model.predict(x_test)

    chosen = []
    for idx in range(len(test_features)):
        chosen.append(max(actions, key=lambda action: predictions[action][idx]))
    return evaluate_chosen_actions(
        "bandit_rf_regressor",
        test_features,
        chosen,
        oracle_action_by_question,
        lookup,
        split_name,
    )


def baseline_final_rows(
    action_rows: Sequence[Dict[str, Any]],
    oracle_rows: Sequence[Dict[str, Any]],
    feature_rows: Sequence[Dict[str, Any]],
    supervised_rows: Sequence[Dict[str, Any]],
    bandit_row: Dict[str, Any],
) -> List[Dict[str, Any]]:
    _train_features, test_features, _split_name = split_feature_rows(feature_rows)
    test_keys = {
        (int(row["conv_id"]), int(row["question_id"]))
        for row in test_features
    }
    rows = []
    for method in [
        "fixed_400",
        "global_fact_only_600",
        "global_fact_only_800",
        "dynamic_hybrid_600_fr0.5",
    ]:
        selected = [
            row for row in action_rows
            if row["action"] == method and (int(row["conv_id"]), int(row["question_id"])) in test_keys
        ]
        summary = metric_summary(selected)
        rows.append({
            "method": method,
            "hit_rate": summary["hit_rate"],
            "avg_recall": summary["avg_recall"],
            "avg_tokens": summary["avg_tokens"],
            "avg_reward": summary["avg_reward"],
        })
    oracle_test_rows = [
        row for row in oracle_rows
        if (int(row["conv_id"]), int(row["question_id"])) in test_keys
    ]
    oracle_summary = metric_summary(oracle_test_rows)
    rows.append({
        "method": "oracle_policy",
        "hit_rate": oracle_summary["hit_rate"],
        "avg_recall": oracle_summary["avg_recall"],
        "avg_tokens": oracle_summary["avg_tokens"],
        "avg_reward": oracle_summary["avg_reward"],
    })
    for row in supervised_rows:
        rows.append({
            "method": row["model"],
            "hit_rate": row["policy_hit_rate"],
            "avg_recall": row["policy_recall"],
            "avg_tokens": row["policy_avg_tokens"],
            "avg_reward": row["policy_reward"],
        })
    rows.append({
        "method": bandit_row["model"],
        "hit_rate": bandit_row["policy_hit_rate"],
        "avg_recall": bandit_row["policy_recall"],
        "avg_tokens": bandit_row["policy_avg_tokens"],
        "avg_reward": bandit_row["policy_reward"],
    })
    return rows


def rows_for_features(
    feature_rows: Sequence[Dict[str, Any]],
    chosen_actions: Sequence[str],
    lookup: Dict[Tuple[int, int, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    selected = []
    for feature, action in zip(feature_rows, chosen_actions):
        key = (int(feature["conv_id"]), int(feature["question_id"]), action)
        selected.append(lookup[key])
    return selected


def evaluate_policy_choice(
    method: str,
    fold: str,
    feature_rows: Sequence[Dict[str, Any]],
    chosen_actions: Sequence[str],
    lookup: Dict[Tuple[int, int, str], Dict[str, Any]],
    threshold: str = "",
    selected_action: str = "",
) -> Dict[str, Any]:
    selected = rows_for_features(feature_rows, chosen_actions, lookup)
    summary = metric_summary(selected)
    return {
        "fold": fold,
        "method": method,
        "threshold": threshold,
        "selected_action": selected_action,
        "hit_rate": summary["hit_rate"],
        "avg_recall": summary["avg_recall"],
        "avg_tokens": summary["avg_tokens"],
        "avg_reward": summary["avg_reward"],
        "override_rate": float(np.mean([action != "fixed_400" for action in chosen_actions])) if chosen_actions else 0.0,
        "chosen_action_distribution": json.dumps(dict(Counter(chosen_actions)), sort_keys=True),
    }


def feature_key(row: Dict[str, Any]) -> Tuple[int, int]:
    return int(row["conv_id"]), int(row["question_id"])


def best_rows_for_features(
    feature_rows: Sequence[Dict[str, Any]],
    grouped_rows: Dict[Tuple[int, int], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    return [
        max(grouped_rows[feature_key(row)], key=lambda item: float(item["reward"]))
        for row in feature_rows
    ]


def best_single_action_for_features(
    feature_rows: Sequence[Dict[str, Any]],
    actions: Sequence[str],
    lookup: Dict[Tuple[int, int, str], Dict[str, Any]],
) -> str:
    scores = {}
    for action in actions:
        rows = rows_for_features(feature_rows, [action] * len(feature_rows), lookup)
        scores[action] = float(np.mean([float(row["reward"]) for row in rows])) if rows else -1e9
    return max(scores, key=scores.get)


def oracle_distribution_rows(
    oracle_rows: Sequence[Dict[str, Any]],
    token_penalty: float,
) -> List[Dict[str, Any]]:
    counts = Counter(str(row["action"]) for row in oracle_rows)
    total = sum(counts.values()) or 1
    return [
        {
            "token_penalty": token_penalty,
            "action": action,
            "count": count,
            "share": count / total,
        }
        for action, count in sorted(counts.items())
    ]


def category_oracle_distribution_rows(
    oracle_rows: Sequence[Dict[str, Any]],
    token_penalty: float,
) -> List[Dict[str, Any]]:
    by_category: Dict[str, Counter] = defaultdict(Counter)
    for row in oracle_rows:
        by_category[str(row["category"])][str(row["action"])] += 1
    rows = []
    for category, counts in sorted(by_category.items()):
        total = sum(counts.values()) or 1
        for action, count in sorted(counts.items()):
            rows.append({
                "token_penalty": token_penalty,
                "category": category,
                "action": action,
                "count": count,
                "share": count / total,
            })
    return rows


def reward_penalty_sweep(
    action_rows: List[Dict[str, Any]],
    hit_weight: float,
    token_penalties: Sequence[float],
) -> List[Dict[str, Any]]:
    rows = []
    for penalty in token_penalties:
        _oracle_rows, summary = compute_oracle(action_rows, hit_weight, penalty)
        rows.append({
            "token_penalty": penalty,
            **summary,
        })
    return rows


def train_override_predictions(
    model_name: str,
    x_train: np.ndarray,
    y_train: Sequence[int],
    x_test: np.ndarray,
) -> np.ndarray:
    if len(set(y_train)) < 2:
        return np.full(len(x_test), int(y_train[0]) if y_train else 0)

    if model_name == "override_logreg":
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(x_train_scaled, y_train)
        return model.predict(x_test_scaled)

    if model_name == "override_rf":
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=23,
            class_weight="balanced",
        )
        model.fit(x_train, y_train)
        return model.predict(x_test)

    if model_name == "override_hgb":
        model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            l2_regularization=0.01,
            random_state=29,
        )
        sample_weight = compute_sample_weight(class_weight="balanced", y=np.array(y_train))
        model.fit(x_train, y_train, sample_weight=sample_weight)
        return model.predict(x_test)

    raise ValueError(f"unknown override model: {model_name}")


def override_fold_rows(
    train_features: Sequence[Dict[str, Any]],
    test_features: Sequence[Dict[str, Any]],
    x_train: np.ndarray,
    x_test: np.ndarray,
    actions: Sequence[str],
    lookup: Dict[Tuple[int, int, str], Dict[str, Any]],
    grouped_rows: Dict[Tuple[int, int], List[Dict[str, Any]]],
    fold: str,
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    rows = []
    alternative_actions = [action for action in actions if action != "fixed_400"]
    best_alt_action = best_single_action_for_features(train_features, alternative_actions, lookup)
    for threshold in thresholds:
        labels = []
        for feature in train_features:
            key = feature_key(feature)
            fixed_reward = float(lookup[(key[0], key[1], "fixed_400")]["reward"])
            best_reward = float(max(grouped_rows[key], key=lambda row: float(row["reward"]))["reward"])
            labels.append(int(best_reward - fixed_reward > threshold))
        for model_name in ["override_logreg", "override_rf", "override_hgb"]:
            predictions = train_override_predictions(model_name, x_train, labels, x_test)
            chosen_actions = [
                best_alt_action if int(prediction) else "fixed_400"
                for prediction in predictions
            ]
            rows.append(evaluate_policy_choice(
                model_name,
                fold,
                test_features,
                chosen_actions,
                lookup,
                threshold=f"{threshold:.2f}",
                selected_action=best_alt_action,
            ))
    return rows


def reward_regression_fold_row(
    model_name: str,
    train_features: Sequence[Dict[str, Any]],
    test_features: Sequence[Dict[str, Any]],
    x_train: np.ndarray,
    x_test: np.ndarray,
    actions: Sequence[str],
    lookup: Dict[Tuple[int, int, str], Dict[str, Any]],
    fold: str,
) -> Dict[str, Any]:
    predictions = {}
    for action in actions:
        y_train = [
            float(lookup[(int(row["conv_id"]), int(row["question_id"]), action)]["reward"])
            for row in train_features
        ]
        if model_name == "reward_regression_rf":
            model = RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=3,
                random_state=31,
            )
        elif model_name == "reward_regression_hgb":
            model = HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.05,
                l2_regularization=0.01,
                random_state=37,
            )
        else:
            raise ValueError(f"unknown reward regression model: {model_name}")
        model.fit(x_train, y_train)
        predictions[action] = model.predict(x_test)

    chosen_actions = []
    for idx in range(len(test_features)):
        chosen_actions.append(max(actions, key=lambda action: predictions[action][idx]))
    return evaluate_policy_choice(model_name, fold, test_features, chosen_actions, lookup)


def run_loocv(
    action_rows: Sequence[Dict[str, Any]],
    feature_rows: Sequence[Dict[str, Any]],
    actions: Sequence[str],
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    lookup = action_lookup(action_rows)
    grouped_rows = rows_by_question(action_rows)
    type_vocab = sorted({str(row["predicted_query_type"]) for row in feature_rows})
    conv_ids = sorted({int(row["conv_id"]) for row in feature_rows})
    fold_rows: List[Dict[str, Any]] = []

    for heldout_conv in conv_ids:
        train_features = [row for row in feature_rows if int(row["conv_id"]) != heldout_conv]
        test_features = [row for row in feature_rows if int(row["conv_id"]) == heldout_conv]
        if not train_features:
            train_features = test_features
        fold = f"leave_conv_{heldout_conv}_out"
        x_train, _ = feature_matrix(train_features, type_vocab)
        x_test, _ = feature_matrix(test_features, type_vocab)

        for action in ["fixed_400", "global_fact_only_800"]:
            if action in actions:
                fold_rows.append(evaluate_policy_choice(
                    action,
                    fold,
                    test_features,
                    [action] * len(test_features),
                    lookup,
                    selected_action=action,
                ))

        best_train_action = best_single_action_for_features(train_features, actions, lookup)
        fold_rows.append(evaluate_policy_choice(
            "best_single_action",
            fold,
            test_features,
            [best_train_action] * len(test_features),
            lookup,
            selected_action=best_train_action,
        ))

        oracle_actions = [
            str(row["action"])
            for row in best_rows_for_features(test_features, grouped_rows)
        ]
        fold_rows.append(evaluate_policy_choice(
            "oracle_policy",
            fold,
            test_features,
            oracle_actions,
            lookup,
        ))

        fold_rows.extend(override_fold_rows(
            train_features,
            test_features,
            x_train,
            x_test,
            actions,
            lookup,
            grouped_rows,
            fold,
            thresholds,
        ))

        for model_name in ["reward_regression_rf", "reward_regression_hgb"]:
            fold_rows.append(reward_regression_fold_row(
                model_name,
                train_features,
                test_features,
                x_train,
                x_test,
                actions,
                lookup,
                fold,
            ))

    return fold_rows


def aggregate_loocv_rows(fold_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        grouped[(str(row["method"]), str(row["threshold"]))].append(row)

    summary_rows = []
    for (method, threshold), rows in sorted(grouped.items()):
        result = {
            "method": method,
            "threshold": threshold,
            "num_folds": len(rows),
        }
        for metric in ["hit_rate", "avg_recall", "avg_tokens", "avg_reward", "override_rate"]:
            values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
            result[f"{metric}_mean"] = float(values.mean()) if len(values) else 0.0
            result[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) else 0.0
        result["selected_action_distribution"] = json.dumps(
            dict(Counter(str(row["selected_action"]) for row in rows if row["selected_action"])),
            sort_keys=True,
        )
        summary_rows.append(result)
    return summary_rows


def final_table_from_loocv(summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def pick(method: str) -> Dict[str, Any]:
        candidates = [row for row in summary_rows if row["method"] == method]
        if not candidates:
            return {
                "method": method,
                "hit_rate": 0.0,
                "avg_recall": 0.0,
                "avg_tokens": 0.0,
                "avg_reward": 0.0,
                "override_rate": 0.0,
            }
        best = max(candidates, key=lambda row: float(row["avg_reward_mean"]))
        return {
            "method": method,
            "hit_rate": best["hit_rate_mean"],
            "avg_recall": best["avg_recall_mean"],
            "avg_tokens": best["avg_tokens_mean"],
            "avg_reward": best["avg_reward_mean"],
            "override_rate": best["override_rate_mean"],
        }

    return [
        pick("fixed_400"),
        pick("global_fact_only_800"),
        pick("best_single_action"),
        pick("oracle_policy"),
        pick("override_logreg"),
        pick("override_rf"),
        pick("override_hgb"),
        pick("reward_regression_rf"),
        pick("reward_regression_hgb"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data" / "locomo10.json")
    parser.add_argument("--limit-conv", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--action-set", choices=sorted(ACTION_SETS), default="compact")
    parser.add_argument("--reward-hit-weight", type=float, default=0.2)
    parser.add_argument("--reward-token-penalty", type=float, default=0.1)
    parser.add_argument(
        "--reward-token-penalty-list",
        type=str,
        default=",".join(f"{value:.2f}" for value in REWARD_TOKEN_PENALTIES),
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    args = parser.parse_args()

    dataset_all = load_dataset(args.data_path)
    dataset = dataset_all[: args.limit_conv] if args.limit_conv else dataset_all
    actions = selected_actions(args.action_set)
    action_rows, feature_rows = build_action_matrix(dataset, args)
    token_penalties = parse_float_list(args.reward_token_penalty_list)
    penalty_sweep_rows = reward_penalty_sweep(action_rows, args.reward_hit_weight, token_penalties)
    oracle_rows, oracle_summary = compute_oracle(
        action_rows,
        args.reward_hit_weight,
        args.reward_token_penalty,
    )
    oracle_summary = {
        "action_set": args.action_set,
        "num_conversations": len(dataset),
        "num_questions": len(feature_rows),
        "reward_hit_weight": args.reward_hit_weight,
        "reward_token_penalty": args.reward_token_penalty,
        **oracle_summary,
    }
    oracle_action_dist_rows = oracle_distribution_rows(oracle_rows, args.reward_token_penalty)
    oracle_category_dist_rows = category_oracle_distribution_rows(oracle_rows, args.reward_token_penalty)

    fold_rows = run_loocv(action_rows, feature_rows, actions, OVERRIDE_THRESHOLDS)
    loocv_summary_rows = aggregate_loocv_rows(fold_rows)
    final_rows = final_table_from_loocv(loocv_summary_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    action_matrix_fields = [
        "conv_id", "question_id", "category", "question", "action",
        "evidence_hit", "evidence_recall", "context_tokens", "latency_ms",
    ]
    write_csv(
        args.output_dir / "policy_action_matrix.csv",
        [{field: row[field] for field in action_matrix_fields} for row in action_rows],
        action_matrix_fields,
    )
    write_csv(args.output_dir / "policy_query_features.csv", feature_rows, [
        "conv_id", "question_id", "category", "question", "query_length",
        "num_words_chars", "has_temporal_keyword", "has_summary_keyword",
        "has_why_how_keyword", "has_compare_keyword", "has_preference_keyword",
        "top1_global_fact_score", "top2_global_fact_score", "top1_top2_margin",
        "global_fact_score_entropy", "top_topic_score", "top_topic_margin",
        "predicted_query_type",
    ])
    write_csv(args.output_dir / "policy_oracle_summary.csv", [oracle_summary], [
        "action_set", "num_conversations", "num_questions", "reward_hit_weight",
        "reward_token_penalty", "best_single_action", "best_single_action_reward", "oracle_policy_reward",
        "oracle_hit_rate", "oracle_recall", "oracle_avg_tokens", "oracle_action_distribution",
    ])
    write_csv(args.output_dir / "policy_oracle_action_distribution.csv", oracle_action_dist_rows, [
        "token_penalty", "action", "count", "share",
    ])
    write_csv(args.output_dir / "policy_oracle_category_distribution.csv", oracle_category_dist_rows, [
        "token_penalty", "category", "action", "count", "share",
    ])
    write_csv(args.output_dir / "policy_reward_penalty_sweep.csv", penalty_sweep_rows, [
        "token_penalty", "best_single_action", "best_single_action_reward", "oracle_policy_reward",
        "oracle_hit_rate", "oracle_recall", "oracle_avg_tokens", "oracle_action_distribution",
    ])
    fold_fields = [
        "fold", "method", "threshold", "selected_action", "hit_rate", "avg_recall",
        "avg_tokens", "avg_reward", "override_rate", "chosen_action_distribution",
    ]
    write_csv(args.output_dir / "policy_loocv_fold_results.csv", fold_rows, fold_fields)
    write_csv(
        args.output_dir / "policy_override_results.csv",
        [row for row in fold_rows if str(row["method"]).startswith("override_")],
        fold_fields,
    )
    write_csv(
        args.output_dir / "policy_reward_regression_results.csv",
        [row for row in fold_rows if str(row["method"]).startswith("reward_regression_")],
        fold_fields,
    )
    summary_fields = [
        "method", "threshold", "num_folds",
        "hit_rate_mean", "hit_rate_std",
        "avg_recall_mean", "avg_recall_std",
        "avg_tokens_mean", "avg_tokens_std",
        "avg_reward_mean", "avg_reward_std",
        "override_rate_mean", "override_rate_std",
        "selected_action_distribution",
    ]
    write_csv(args.output_dir / "policy_loocv_summary.csv", loocv_summary_rows, summary_fields)
    write_csv(args.output_dir / "policy_bandit_results.csv", [
        row for row in loocv_summary_rows if str(row["method"]).startswith("reward_regression_")
    ], summary_fields)
    write_csv(args.output_dir / "policy_supervised_results.csv", [
        row for row in loocv_summary_rows if str(row["method"]).startswith("override_")
    ], summary_fields)
    write_csv(args.output_dir / "policy_routing_final_table.csv", final_rows, [
        "method", "hit_rate", "avg_recall", "avg_tokens", "avg_reward", "override_rate",
    ])

    print("Policy oracle summary:")
    print(json.dumps(oracle_summary, indent=2, ensure_ascii=False))
    print("\nFinal table:")
    for row in final_rows:
        print(
            f"{row['method']}\t"
            f"hit={float(row['hit_rate']):.4f}\t"
            f"recall={float(row['avg_recall']):.4f}\t"
            f"tokens={float(row['avg_tokens']):.1f}\t"
            f"reward={float(row['avg_reward']):.4f}\t"
            f"override={float(row['override_rate']):.3f}"
        )
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
