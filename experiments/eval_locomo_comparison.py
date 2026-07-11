from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import eval_longmemeval_mini as base
from eval_longmemeval_mini import (
    BM25Index,
    LLMClient,
    MethodConfig,
    ProfileFact,
    ProfileRetrievalResult,
    QwenEmbeddingClient,
    QwenRerankerClient,
    build_memory,
    estimate_tokens,
    generate_and_judge,
    keyword_overlap,
    load_examples,
    pack_ranked,
    retrieve_method,
    retrieval_metrics,
    write_csv,
)
from eval_rl_memory_complexity import complexity_bin, memory_stats, size_bin
from hypermem.subquery_planner import (
    ROUTE_GLOBAL_COMPACT,
    ROUTE_HIERARCHICAL,
    ROUTE_ROLE_SHORTCUT,
    ROUTE_ROLE_TEMPORAL,
    QueryPlan,
    RouteCandidate,
    Subquery,
    candidate_routes,
    decompose_query,
    lexical_overlap,
)


@dataclass(frozen=True)
class RetrievalAction:
    name: str
    method: MethodConfig
    role_gate: bool = False
    snippet_words: int = 96


def infer_query_roles(question: str, rows: Sequence[Dict[str, Any]]) -> List[str]:
    roles = sorted({str(row.get("role") or "").strip() for row in rows if str(row.get("role") or "").strip()})
    q = question.lower()
    matched: List[str] = []
    for role in roles:
        if re.search(rf"\b{re.escape(role.lower())}\b", q):
            matched.append(role)
    return matched


def role_gated_rows(question: str, rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    roles = infer_query_roles(question, rows)
    # Keep all rows. Hard role filtering dropped answer evidence when the answer was
    # said by the conversation partner; role actions are now soft routing choices.
    return list(rows), roles


def query_snippet(text: str, query: str, max_words: int = 96) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    qterms = {t for t in base.tokenize(query) if len(t) > 2}
    best_i = 0
    best_score = -1
    step = max(1, max_words // 3)
    for i in range(0, max(1, len(words) - max_words + 1), step):
        chunk = " ".join(words[i : i + max_words])
        score = len(qterms & set(base.tokenize(chunk)))
        if score > best_score:
            best_i = i
            best_score = score
    return " ".join(words[best_i : best_i + max_words])


def source_snippet_result(
    ret: ProfileRetrievalResult,
    query: str,
    method_name: str,
    max_words: int,
    max_tokens: int | None = None,
) -> ProfileRetrievalResult:
    facts: List[ProfileFact] = []
    for fact in ret.selected_facts:
        meta = dict(fact.metadata)
        raw = str(meta.get("raw_content") or fact.content)
        date = str(meta.get("date") or "")
        role = str(meta.get("role") or "")
        content = f"[source={meta.get('row_id', fact.fact_id)} date={date} role={role}] {query_snippet(raw, query, max_words)}"
        meta["display_content"] = content
        facts.append(
            ProfileFact(
                fact_id=fact.fact_id,
                content=content,
                keywords=fact.keywords,
                timestamp=fact.timestamp,
                embedding=fact.embedding,
                metadata=meta,
            )
        )
    if max_tokens is not None:
        packed: List[ProfileFact] = []
        used_tokens = 0
        for fact in facts:
            fact_tokens = estimate_tokens(fact.content)
            if packed and used_tokens + fact_tokens > max_tokens:
                continue
            packed.append(fact)
            used_tokens += fact_tokens
        facts = packed
    debug = [dict(x) for x in ret.debug_scores]
    if debug:
        debug[0]["method"] = method_name
        debug[0]["snippet_words"] = max_words
    return ProfileRetrievalResult(
        query=ret.query,
        channel=method_name,
        selected_edges=ret.selected_edges,
        selected_facts=facts,
        score=ret.score,
        tokens=estimate_tokens([f.content for f in facts]),
        fallback_used=ret.fallback_used,
        sufficient=ret.sufficient,
        debug_scores=debug,
    )


def evidence_block_display_first(ret: ProfileRetrievalResult, max_chars: int = 4200) -> str:
    lines: List[str] = []
    facts = (
        list(ret.selected_facts)
        if any((fact.metadata or {}).get("display_content") for fact in ret.selected_facts)
        else base.sorted_evidence_facts(ret)
    )
    for i, fact in enumerate(facts[:12], start=1):
        meta = fact.metadata or {}
        text = str(meta.get("display_content") or meta.get("raw_content") or fact.content).replace("\n", " ").strip()
        if meta.get("display_content"):
            lines.append(f"[{i}] {text[:520]}")
        else:
            date = str(meta.get("date") or "")
            role = str(meta.get("role") or "")
            lines.append(f"[{i}] {date} {role}: {text[:360]}")
    return "\n".join(lines)[:max_chars]


base.evidence_block = evidence_block_display_first
generate_and_judge.__globals__["evidence_block"] = evidence_block_display_first


def action_space() -> List[RetrievalAction]:
    return [
        RetrievalAction(
            "RoleCompact",
            MethodConfig("RoleCompact", graph_gate="hypermem_full", top_k_facts=10, max_tokens=650, initial_candidates=55, topic_top_k=5, episode_top_k=10, lambda_prop=0.5),
            role_gate=True,
            snippet_words=72,
        ),
        RetrievalAction(
            "RoleBalanced",
            MethodConfig("RoleBalanced", graph_gate="hypermem_full", top_k_facts=14, max_tokens=850, initial_candidates=70, topic_top_k=6, episode_top_k=12, lambda_prop=0.5),
            role_gate=True,
            snippet_words=80,
        ),
        RetrievalAction(
            "FullCompact",
            MethodConfig("FullCompact", graph_gate="hypermem_full", top_k_facts=10, max_tokens=650, initial_candidates=55, topic_top_k=5, episode_top_k=10, lambda_prop=0.5),
            role_gate=False,
            snippet_words=72,
        ),
        RetrievalAction(
            "FullBalanced",
            MethodConfig("FullBalanced", graph_gate="hypermem_full", top_k_facts=14, max_tokens=850, initial_candidates=70, topic_top_k=6, episode_top_k=12, lambda_prop=0.5),
            role_gate=False,
            snippet_words=80,
        ),
        RetrievalAction(
            "FullRecall",
            MethodConfig("FullRecall", graph_gate="hypermem_full", top_k_facts=18, max_tokens=1100, initial_candidates=90, topic_top_k=8, episode_top_k=16, lambda_prop=0.5),
            role_gate=False,
            snippet_words=88,
        ),
        RetrievalAction(
            "RoleTemporal",
            MethodConfig("RoleTemporal", graph_gate="hypermem_full", top_k_facts=18, max_tokens=1100, initial_candidates=90, topic_top_k=8, episode_top_k=16, lambda_prop=0.5),
            role_gate=True,
            snippet_words=88,
        ),
    ]


def hypermem_config() -> MethodConfig:
    return MethodConfig(
        "HyperMem-Flow",
        graph_gate="hypermem_full",
        top_k_facts=30,
        max_tokens=1800,
        initial_candidates=100,
        topic_top_k=15,
        episode_top_k=20,
        lambda_prop=0.5,
    )


class ActionBandit:
    def __init__(self, actions: Sequence[RetrievalAction], seed: int = 7) -> None:
        self.actions = list(actions)
        self.rng = random.Random(seed)
        self.alpha: Dict[str, List[float]] = {}
        self.beta: Dict[str, List[float]] = {}

    def bucket(self, example: Any) -> str:
        stats = memory_stats(example)
        role_state = "role" if infer_query_roles(example.question, example.rows) else "norole"
        return f"{size_bin(stats)}:{complexity_bin(example, stats)}:{role_state}"

    def _ensure(self, bucket: str) -> None:
        if bucket in self.alpha:
            return
        self.alpha[bucket] = [1.0] * len(self.actions)
        self.beta[bucket] = [1.0] * len(self.actions)
        for i, action in enumerate(self.actions):
            if action.name in {"FullCompact", "FullBalanced"}:
                self.alpha[bucket][i] += 0.3
            if bucket.endswith(":role") and action.role_gate:
                self.alpha[bucket][i] += 0.35
            if "multi_count" in bucket and action.name in {"RoleBalanced", "FullRecall"}:
                self.alpha[bucket][i] += 0.6
            if "temporal" in bucket and action.name in {"FullBalanced", "FullRecall", "RoleBalanced"}:
                self.alpha[bucket][i] += 0.4
            if "preference" in bucket and action.name in {"FullCompact", "FullBalanced", "RoleCompact"}:
                self.alpha[bucket][i] += 0.45

    def select(self, example: Any, train: bool) -> int:
        bucket = self.bucket(example)
        self._ensure(bucket)
        if train:
            vals = [self.rng.betavariate(a, b) for a, b in zip(self.alpha[bucket], self.beta[bucket])]
        else:
            vals = [a / max(1e-9, a + b) for a, b in zip(self.alpha[bucket], self.beta[bucket])]
        return max(range(len(self.actions)), key=lambda i: (vals[i], -self.actions[i].method.max_tokens))

    def update(self, example: Any, action_idx: int, reward: float) -> None:
        bucket = self.bucket(example)
        self._ensure(bucket)
        reward = max(0.0, min(1.0, reward))
        self.alpha[bucket][action_idx] += reward
        self.beta[bucket][action_idx] += 1.0 - reward

    def dump(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for bucket in sorted(self.alpha):
            out[bucket] = {
                self.actions[i].name: round(self.alpha[bucket][i] / max(1e-9, self.alpha[bucket][i] + self.beta[bucket][i]), 4)
                for i in range(len(self.actions))
            }
        return out


def reward(example: Any, metrics: Dict[str, Any], ret: ProfileRetrievalResult) -> float:
    comp = complexity_bin(example, memory_stats(example))
    latency = float(ret.debug_scores[0].get("latency_ms", 0.0)) if ret.debug_scores else 0.0
    first_answer_rank = None
    for idx, fact in enumerate(ret.selected_facts, start=1):
        if fact.metadata.get("has_answer"):
            first_answer_rank = idx
            break
    if comp == "multi_count":
        score = 0.18 * metrics["fact_hit"] + 0.50 * metrics["answer_turn_recall"] + 0.24 * metrics["all_answer_turns_hit"]
        token_target = 1100.0
    elif comp == "temporal":
        score = 0.20 * metrics["fact_hit"] + 0.50 * metrics["answer_turn_recall"] + 0.22 * metrics["all_answer_turns_hit"]
        token_target = 950.0
    else:
        score = 0.30 * metrics["fact_hit"] + 0.46 * metrics["answer_turn_recall"] + 0.16 * metrics["all_answer_turns_hit"]
        token_target = 800.0
    if first_answer_rank == 1:
        score += 0.14
    elif first_answer_rank is not None and first_answer_rank <= 3:
        score += 0.09
    elif first_answer_rank is not None and first_answer_rank <= 5:
        score += 0.04
    elif metrics["fact_hit"]:
        score -= 0.06
    score -= min(0.22, max(0.0, ret.tokens - token_target) / token_target * 0.16)
    score -= min(0.08, latency / 5000.0 * 0.05)
    return score


def conversation_key(example: Any) -> str:
    qid = str(example.qid)
    if "::qa_" in qid:
        return qid.split("::qa_", 1)[0]
    if "::" in qid:
        return qid.split("::", 1)[0]
    return qid


def conversation_balanced_sample(examples: Sequence[Any], max_examples: int, seed: int) -> List[Any]:
    """Round-robin sample QA items across conversations before applying a cap."""
    if max_examples <= 0 or len(examples) <= max_examples:
        return list(examples)
    rng = random.Random(seed)
    groups: Dict[str, List[Any]] = {}
    for example in examples:
        groups.setdefault(conversation_key(example), []).append(example)
    for items in groups.values():
        rng.shuffle(items)
    keys = sorted(groups)
    rng.shuffle(keys)
    selected: List[Any] = []
    while keys and len(selected) < max_examples:
        next_keys: List[str] = []
        for key in keys:
            if groups[key] and len(selected) < max_examples:
                selected.append(groups[key].pop())
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def split_examples(
    examples: Sequence[Any],
    train_size: int,
    test_size: int,
    seed: int,
    split_unit: str = "conversation",
) -> Tuple[List[Any], List[Any]]:
    rng = random.Random(seed)
    if split_unit == "conversation":
        groups: Dict[str, List[Any]] = {}
        for example in examples:
            groups.setdefault(conversation_key(example), []).append(example)
        group_keys = list(groups)
        rng.shuffle(group_keys)
        train: List[Any] = []
        test: List[Any] = []
        for key in group_keys:
            target = train if len(train) < train_size else test
            target.extend(groups[key])
            if len(train) >= train_size and len(test) >= test_size:
                break
        return train[:train_size], test[:test_size]
    pool = list(examples)
    rng.shuffle(pool)
    train = pool[: min(train_size, len(pool))]
    train_ids = {ex.qid for ex in train}
    test = [ex for ex in pool if ex.qid not in train_ids][:test_size]
    return train, test


def run_action(
    example: Any,
    action: RetrievalAction,
    method_name: str,
    qwen_embed: QwenEmbeddingClient,
    qwen_reranker: QwenRerankerClient | None,
    memory: Any | None = None,
    format_sources: bool = True,
) -> ProfileRetrievalResult:
    rows, gated_roles = role_gated_rows(example.question, example.rows) if action.role_gate else (list(example.rows), [])
    memory = memory or build_memory(rows)
    ret = retrieve_method(
        example,
        memory,
        action.method,
        qwen_embed=qwen_embed,
        qwen_reranker=qwen_reranker,
        role_hints=gated_roles,
    )
    if format_sources:
        ret = source_snippet_result(ret, example.question, method_name, action.snippet_words, action.method.max_tokens)
    ret.channel = method_name
    if ret.debug_scores:
        ret.debug_scores[0]["method"] = method_name
        ret.debug_scores[0]["action"] = action.name
        ret.debug_scores[0]["role_gate"] = action.role_gate
        ret.debug_scores[0]["gated_roles"] = ",".join(gated_roles)
    return ret


ROUTE_TO_ACTION = {
    ROUTE_GLOBAL_COMPACT: "FullCompact",
    ROUTE_ROLE_SHORTCUT: "RoleBalanced",
    ROUTE_HIERARCHICAL: "FullBalanced",
    ROUTE_ROLE_TEMPORAL: "RoleTemporal",
}


class SubqueryRouteBandit:
    """Small Thompson-style policy over a subquery-route action space.

    The posterior is contextual only at a coarse, auditable level.  The online
    evidence-gain update is used at deployment time; supervised retrieval
    feedback is used only in the benchmark training phase.
    """

    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self.alpha: Dict[str, Dict[str, float]] = {}
        self.beta: Dict[str, Dict[str, float]] = {}

    def bucket(self, task: Subquery, has_role_hint: bool) -> str:
        return f"{task.intent}:{'role' if has_role_hint else 'global'}"

    def _ensure(self, bucket: str) -> None:
        if bucket in self.alpha:
            return
        self.alpha[bucket] = {
            ROUTE_GLOBAL_COMPACT: 1.2,
            ROUTE_HIERARCHICAL: 1.3,
            ROUTE_ROLE_SHORTCUT: 1.2,
            ROUTE_ROLE_TEMPORAL: 1.1,
        }
        self.beta[bucket] = {route: 1.0 for route in self.alpha[bucket]}

    def posterior(self, task: Subquery, route: str, has_role_hint: bool, train: bool) -> float:
        bucket = self.bucket(task, has_role_hint)
        self._ensure(bucket)
        alpha = self.alpha[bucket].get(route, 1.0)
        beta = self.beta[bucket].get(route, 1.0)
        return self.rng.betavariate(alpha, beta) if train else alpha / max(1e-9, alpha + beta)

    def update(self, task: Subquery, route: str, has_role_hint: bool, reward_value: float) -> None:
        bucket = self.bucket(task, has_role_hint)
        self._ensure(bucket)
        reward_value = max(0.0, min(1.0, reward_value))
        self.alpha[bucket][route] = self.alpha[bucket].get(route, 1.0) + reward_value
        self.beta[bucket][route] = self.beta[bucket].get(route, 1.0) + 1.0 - reward_value

    def dump(self) -> Dict[str, Dict[str, float]]:
        return {
            bucket: {
                route: round(alpha / max(1e-9, alpha + self.beta[bucket][route]), 4)
                for route, alpha in routes.items()
            }
            for bucket, routes in sorted(self.alpha.items())
        }


def choose_subquery_route(
    task: Subquery,
    routes: Sequence[RouteCandidate],
    *,
    has_role_hint: bool,
    completed_texts: Sequence[str],
    policy: SubqueryRouteBandit,
    train: bool,
) -> Tuple[RouteCandidate, float, Dict[str, float]]:
    """Score V(i, j) using evidence prior, dependency unlock, cost and redundancy."""
    scored: List[Tuple[float, RouteCandidate, Dict[str, float]]] = []
    for route in routes:
        posterior = policy.posterior(task, route.route, has_role_hint, train=train)
        dependency_gain = 0.16 * len(task.depends_on)
        redundancy = max([lexical_overlap(task.text, text) for text in completed_texts] or [0.0])
        value = (
            0.43 * route.expected_evidence
            + 0.27 * posterior
            + dependency_gain
            - 0.00013 * route.token_cost
            - 0.00055 * route.search_cost
            - 0.20 * redundancy
        )
        parts = {
            "expected_evidence": round(route.expected_evidence, 6),
            "policy_posterior": round(posterior, 6),
            "dependency_gain": round(dependency_gain, 6),
            "redundancy": round(redundancy, 6),
            "value": round(value, 6),
        }
        scored.append((value, route, parts))
    _value, selected, parts = max(scored, key=lambda item: (item[0], -item[1].token_cost, item[1].route))
    return selected, parts["value"], parts


def merge_subquery_evidence(
    example: Any,
    step_results: Sequence[Tuple[Subquery, RouteCandidate, ProfileRetrievalResult]],
    qwen_reranker: QwenRerankerClient | None,
    started: float,
) -> ProfileRetrievalResult:
    """Deduplicate route outputs and rerank source facts once against the full query."""
    candidates: Dict[str, Tuple[ProfileFact, float, int]] = {}
    for _task, _route, ret in step_results:
        for rank, fact in enumerate(ret.selected_facts, start=1):
            fact_id = fact.fact_id
            previous = candidates.get(fact_id)
            route_score = 1.0 / rank
            if previous is None:
                candidates[fact_id] = (fact, route_score, 1)
            else:
                candidates[fact_id] = (previous[0], previous[1] + route_score, previous[2] + 1)
    facts = [item[0] for item in candidates.values()]
    if qwen_reranker and facts:
        rerank_scores = qwen_reranker.rerank(example.question, [fact.content for fact in facts])
    else:
        rerank_scores = [candidates[fact.fact_id][1] for fact in facts]
    ranked: List[Tuple[ProfileFact, float]] = []
    for fact, rerank_score in zip(facts, rerank_scores):
        _saved, route_score, support_count = candidates[fact.fact_id]
        score = float(rerank_score) + 0.08 * route_score + 0.05 * (support_count - 1)
        ranked.append((fact, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    selected = pack_ranked(ranked, top_k=18, max_tokens=1100)
    mean_score = sum(score for _fact, score in ranked[: max(1, len(selected))]) / max(1, len(selected))
    return base.result(example.question, "HARMONY-Mem", selected, [], mean_score, started, len(facts))


def run_harmony_subquery(
    example: Any,
    policy: SubqueryRouteBandit,
    qwen_embed: QwenEmbeddingClient,
    qwen_reranker: QwenRerankerClient | None,
    *,
    atomic_action: RetrievalAction | None = None,
    train: bool = False,
    max_steps: int = 3,
    token_budget: int = 1250,
    search_budget: int = 280,
) -> ProfileRetrievalResult:
    """Execute bounded sequential (subquery, route) scheduling for HARMONY."""
    started = time.time()
    plan: QueryPlan = decompose_query(example.question, max_subqueries=max_steps)
    # Most long-memory questions are atomic.  Reusing the trained, cheap
    # single-path controller here is the intended gate behavior; the
    # subquery scheduler is reserved for genuine dependency-bearing queries.
    if not plan.decomposed and atomic_action is not None:
        ret = run_action(
            example,
            atomic_action,
            "HARMONY-Mem",
            qwen_embed,
            qwen_reranker,
        )
        if ret.debug_scores:
            ret.debug_scores[0].update(
                {
                    "planner": {
                        "decomposed": False,
                        "confidence": round(plan.confidence, 6),
                        "signals": plan.matched_signals,
                        "relations": [],
                        "subquery_count": 1,
                        "atomic_gate": "legacy_single_path",
                    },
                    "scheduler_steps": [
                        {
                            "step": 1,
                            "subquery_id": plan.subqueries[0].subquery_id,
                            "subquery": plan.subqueries[0].text,
                            "intent": plan.subqueries[0].intent,
                            "route": "legacy_single_path",
                            "action": atomic_action.name,
                            "new_fact_count": len(ret.selected_facts),
                            "returned_fact_count": len(ret.selected_facts),
                            "marginal_evidence_gain": 1.0,
                            "retrieved_tokens": ret.tokens,
                        }
                    ],
                    "stopped_reason": "atomic_single_path",
                }
            )
        return ret
    available_roles = sorted({str(row.get("role") or "").strip() for row in example.rows if str(row.get("role") or "").strip()})
    memory = build_memory(example.rows)
    query_roles = {role.lower() for role in infer_query_roles(example.question, example.rows)}
    has_role_hint = bool(
        query_roles
        and any(query_roles & {str(role).lower() for role in fact.metadata.get("role_mentions", [])} for fact in memory.facts.values())
    )
    pending = {task.subquery_id: task for task in plan.subqueries}
    completed: List[Subquery] = []
    completed_texts: List[str] = []
    attempted_routes: Dict[str, set[str]] = {}
    seen_fact_ids: set[str] = set()
    step_results: List[Tuple[Subquery, RouteCandidate, ProfileRetrievalResult]] = []
    step_trace: List[Dict[str, Any]] = []
    used_tokens = 0
    used_search = 0
    stopped_reason = "all_subqueries_resolved"

    for step_idx in range(max_steps):
        ready = [
            task for task in pending.values()
            if all(dep in {done.subquery_id for done in completed} for dep in task.depends_on)
        ]
        if not ready:
            stopped_reason = "dependency_blocked"
            break
        options: List[Tuple[float, Subquery, RouteCandidate, Dict[str, float]]] = []
        for task in ready:
            for route in candidate_routes(task, has_role_hint=has_role_hint):
                if route.route in attempted_routes.get(task.subquery_id, set()):
                    continue
                if used_tokens + route.token_cost > token_budget or used_search + route.search_cost > search_budget:
                    continue
                selected, value, parts = choose_subquery_route(
                    task,
                    [route],
                    has_role_hint=has_role_hint,
                    completed_texts=completed_texts,
                    policy=policy,
                    train=train,
                )
                options.append((value, task, selected, parts))
        if not options:
            stopped_reason = "budget_exhausted"
            break
        _value, task, route, value_parts = max(options, key=lambda item: (item[0], -item[2].token_cost, item[2].route))
        action = {item.name: item for item in action_space()}[ROUTE_TO_ACTION[route.route]]
        # Keep a broad, inexpensive candidate pool from each route.  Qwen
        # reranking happens once after all paths are fused; truncating to a
        # route's final context size before that stage loses complementary
        # evidence on multi-hop questions.
        provisional_method = replace(
            action.method,
            top_k_facts=max(36, action.method.top_k_facts * 3),
            max_tokens=max(2400, action.method.max_tokens * 2),
        )
        provisional_action = replace(action, method=provisional_method)
        sub_example = replace(example, question=task.text)
        ret = run_action(
            sub_example,
            provisional_action,
            f"HARMONY-Mem/{route.route}",
            qwen_embed,
            # Route outputs are provisional.  Reranking each route and then
            # reranking their union doubles latency for atomic questions, so
            # HARMONY performs its expensive Qwen rerank once after fusion.
            None,
            memory=memory,
            format_sources=False,
        )
        new_ids = [fact.fact_id for fact in ret.selected_facts if fact.fact_id not in seen_fact_ids]
        novelty = len(new_ids) / max(1, len(ret.selected_facts))
        confidence = max(0.0, min(1.0, float(ret.score)))
        marginal_gain = 0.72 * novelty + 0.28 * confidence
        policy.update(task, route.route, has_role_hint, marginal_gain)
        seen_fact_ids.update(new_ids)
        used_tokens += route.token_cost
        used_search += int(ret.debug_scores[0].get("candidate_facts", route.search_cost)) if ret.debug_scores else route.search_cost
        step_results.append((task, route, ret))
        step_trace.append(
            {
                "step": step_idx + 1,
                "subquery_id": task.subquery_id,
                "subquery": task.text,
                "intent": task.intent,
                "route": route.route,
                "action": action.name,
                "value": value_parts,
                "new_fact_count": len(new_ids),
                "returned_fact_count": len(ret.selected_facts),
                "marginal_evidence_gain": round(marginal_gain, 6),
                "estimated_incremental_tokens": route.token_cost,
                "retrieved_tokens": ret.tokens,
                "candidate_facts": int(ret.debug_scores[0].get("candidate_facts", 0)) if ret.debug_scores else 0,
            }
        )
        attempted_routes.setdefault(task.subquery_id, set()).add(route.route)
        retry_atomic = (
            task.intent in {"atomic_causal", "atomic_temporal"}
            and len(attempted_routes[task.subquery_id]) < 2
            and step_idx + 1 < max_steps
        )
        if not retry_atomic:
            completed.append(task)
            completed_texts.append(task.text)
            pending.pop(task.subquery_id, None)
        if step_idx > 0 and marginal_gain < 0.10 and len(seen_fact_ids) >= 6:
            stopped_reason = "low_marginal_gain"
            break

    merged = merge_subquery_evidence(example, step_results, qwen_reranker, started)
    merged = source_snippet_result(merged, example.question, "HARMONY-Mem", 88, 1100)
    merged.channel = "HARMONY-Mem"
    if merged.debug_scores:
        merged.debug_scores[0].update(
            {
                "planner": {
                    "decomposed": plan.decomposed,
                    "confidence": round(plan.confidence, 6),
                    "signals": plan.matched_signals,
                    "relations": plan.relations,
                    "subquery_count": len(plan.subqueries),
                    "role_hints": [role for role in available_roles if role.lower() in example.question.lower()],
                    "role_hyperedge_available": has_role_hint,
                },
                "scheduler_steps": step_trace,
                "stopped_reason": stopped_reason,
                "scheduler_tokens": used_tokens,
                "scheduler_search_cost": used_search,
            }
        )
    return merged


def run_method_config(
    example: Any,
    method: MethodConfig,
    qwen_embed: QwenEmbeddingClient | None,
    qwen_reranker: QwenRerankerClient | None,
) -> ProfileRetrievalResult:
    memory = build_memory(example.rows)
    return retrieve_method(example, memory, method, qwen_embed=qwen_embed, qwen_reranker=qwen_reranker)


STOPWORDS = {
    "about", "after", "again", "also", "before", "being", "could", "every", "first", "from",
    "have", "into", "last", "like", "more", "than", "that", "their", "there", "these",
    "they", "this", "what", "when", "where", "which", "while", "with", "would",
}


def salient_terms(text: str) -> set[str]:
    terms = {t for t in base.tokenize(text) if len(t) > 3 and t not in STOPWORDS}
    entities = {m.lower() for m in re.findall(r"\b[A-Z][a-z]{2,}\b", text)}
    return terms | entities


def fact_scores(
    example: Any,
    facts: Sequence[ProfileFact],
    qwen_embed: QwenEmbeddingClient,
) -> Tuple[List[Tuple[ProfileFact, float]], List[Tuple[ProfileFact, float]], List[float], Dict[str, List[float]]]:
    if not facts:
        return [], [], [], {}
    bm25 = BM25Index(facts)
    bm25_rank = [(fact, bm25.score(example.question, i)) for i, fact in enumerate(facts)]
    bm25_rank.sort(key=lambda x: x[1], reverse=True)
    vectors = qwen_embed.embed([example.question] + [f.content for f in facts])
    qvec = vectors[0]
    emb = {fact.fact_id: vec for fact, vec in zip(facts, vectors[1:])}
    dense_rank = [(fact, qwen_embed.cosine(qvec, emb[fact.fact_id])) for fact in facts]
    dense_rank.sort(key=lambda x: x[1], reverse=True)
    return bm25_rank, dense_rank, qvec, emb


def finalize_baseline_result(
    example: Any,
    method_name: str,
    ranked: Sequence[Tuple[ProfileFact, float]],
    started: float,
    candidate_count: int,
    top_k: int = 20,
    max_tokens: int = 1250,
) -> ProfileRetrievalResult:
    selected = pack_ranked(ranked, top_k, max_tokens)
    score = sum(s for _f, s in ranked[: max(1, top_k)]) / max(1, min(len(ranked), top_k))
    ret = base.result(example.question, method_name, selected, [], score, started, candidate_count)
    ret.channel = method_name
    if ret.debug_scores:
        ret.debug_scores[0]["method"] = method_name
    return ret


def retrieve_mem0_lite(
    example: Any,
    qwen_embed: QwenEmbeddingClient,
    qwen_reranker: QwenRerankerClient | None,
) -> ProfileRetrievalResult:
    started = time.time()
    memory = build_memory(example.rows)
    facts = list(memory.facts.values())
    bm25_rank, dense_rank, _qvec, _emb = fact_scores(example, facts, qwen_embed)
    fused = base.rrf_fuse([
        [(f.fact_id, s) for f, s in bm25_rank[:100]],
        [(f.fact_id, s) for f, s in dense_rank[:100]],
    ])
    q_terms = salient_terms(example.question)
    by_id = {f.fact_id: f for f in facts}
    candidates = [by_id[fid] for fid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:90] if fid in by_id]
    rerank_scores = qwen_reranker.rerank(example.question, [f.content for f in candidates]) if qwen_reranker and candidates else [fused.get(f.fact_id, 0.0) for f in candidates]
    ranked: List[Tuple[ProfileFact, float]] = []
    temporal_q = bool(re.search(r"\b(when|date|before|after|first|last|days?|weeks?|months?)\b", example.question.lower()))
    for fact, rr in zip(candidates, rerank_scores):
        f_terms = salient_terms(fact.content)
        entity_boost = len(q_terms & f_terms) * 0.08
        temporal_boost = 0.04 if temporal_q and str(fact.metadata.get("date") or "") else 0.0
        ranked.append((fact, float(rr) + entity_boost + temporal_boost + 0.05 * keyword_overlap(example.question, fact.content)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return finalize_baseline_result(example, "Mem0-Lite", ranked, started, len(candidates), top_k=20, max_tokens=1250)


def retrieve_amem_lite(
    example: Any,
    qwen_embed: QwenEmbeddingClient,
    qwen_reranker: QwenRerankerClient | None,
) -> ProfileRetrievalResult:
    started = time.time()
    memory = build_memory(example.rows)
    facts = list(memory.facts.values())
    bm25_rank, dense_rank, _qvec, _emb = fact_scores(example, facts, qwen_embed)
    seed_ids = {f.fact_id for f, _s in bm25_rank[:18]} | {f.fact_id for f, _s in dense_rank[:18]}
    by_id = {f.fact_id: f for f in facts}
    seed_terms = {fid: salient_terms(by_id[fid].content) for fid in seed_ids if fid in by_id}
    candidate_ids = set(seed_ids)
    for fid in list(seed_ids):
        fact = by_id.get(fid)
        if not fact:
            continue
        sid = str(fact.metadata.get("session_id") or "")
        role = str(fact.metadata.get("role") or "")
        terms = seed_terms.get(fid, set())
        for other in facts:
            if other.fact_id in candidate_ids:
                continue
            same_session = str(other.metadata.get("session_id") or "") == sid
            same_role = role and str(other.metadata.get("role") or "") == role
            linked = bool(terms & salient_terms(other.content))
            if same_session or (same_role and linked):
                candidate_ids.add(other.fact_id)
            if len(candidate_ids) >= 95:
                break
    candidates = [by_id[fid] for fid in candidate_ids if fid in by_id]
    rerank_scores = qwen_reranker.rerank(example.question, [f.content for f in candidates]) if qwen_reranker and candidates else [keyword_overlap(example.question, f.content) for f in candidates]
    ranked = []
    q_terms = salient_terms(example.question)
    for fact, rr in zip(candidates, rerank_scores):
        link_boost = 0.05 * len(q_terms & salient_terms(fact.content))
        ranked.append((fact, float(rr) + link_boost + 0.04 * keyword_overlap(example.question, fact.content)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return finalize_baseline_result(example, "A-Mem-Lite", ranked, started, len(candidates), top_k=20, max_tokens=1250)


def retrieve_hipporag_lite(
    example: Any,
    qwen_embed: QwenEmbeddingClient,
    qwen_reranker: QwenRerankerClient | None,
) -> ProfileRetrievalResult:
    started = time.time()
    memory = build_memory(example.rows)
    facts = list(memory.facts.values())
    bm25_rank, dense_rank, _qvec, _emb = fact_scores(example, facts, qwen_embed)
    seed_score: Dict[str, float] = {}
    for rank, (fact, _s) in enumerate(bm25_rank[:60], start=1):
        seed_score[fact.fact_id] = seed_score.get(fact.fact_id, 0.0) + 1.0 / (60 + rank)
    for rank, (fact, _s) in enumerate(dense_rank[:60], start=1):
        seed_score[fact.fact_id] = seed_score.get(fact.fact_id, 0.0) + 1.0 / (60 + rank)
    terms = {f.fact_id: salient_terms(f.content) for f in facts}
    by_session: Dict[str, List[str]] = {}
    for f in facts:
        by_session.setdefault(str(f.metadata.get("session_id") or ""), []).append(f.fact_id)
    p = {f.fact_id: seed_score.get(f.fact_id, 0.0) for f in facts}
    total = sum(p.values()) or 1.0
    p = {fid: val / total for fid, val in p.items()}
    for _ in range(6):
        nxt = {fid: 0.15 * seed_score.get(fid, 0.0) / total for fid in p}
        for fact in facts:
            fid = fact.fact_id
            neigh = set(by_session.get(str(fact.metadata.get("session_id") or ""), [])[:80])
            f_terms = terms[fid]
            for other in facts:
                if other.fact_id != fid and f_terms & terms[other.fact_id]:
                    neigh.add(other.fact_id)
                if len(neigh) >= 80:
                    break
            share = 0.85 * p.get(fid, 0.0) / max(1, len(neigh))
            for nid in neigh:
                nxt[nid] = nxt.get(nid, 0.0) + share
        p = nxt
    by_id = {f.fact_id: f for f in facts}
    candidates = [by_id[fid] for fid, _ in sorted(p.items(), key=lambda x: x[1], reverse=True)[:90] if fid in by_id]
    rerank_scores = qwen_reranker.rerank(example.question, [f.content for f in candidates]) if qwen_reranker and candidates else [p.get(f.fact_id, 0.0) for f in candidates]
    ranked = [(fact, 0.70 * float(rr) + 0.30 * p.get(fact.fact_id, 0.0)) for fact, rr in zip(candidates, rerank_scores)]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return finalize_baseline_result(example, "HippoRAG-Lite", ranked, started, len(candidates), top_k=20, max_tokens=1250)


def retrieve_lightrag_lite(
    example: Any,
    qwen_embed: QwenEmbeddingClient,
    qwen_reranker: QwenRerankerClient | None,
) -> ProfileRetrievalResult:
    started = time.time()
    memory = build_memory(example.rows)
    facts = list(memory.facts.values())
    episodes, topics = base.build_episode_topic_docs(facts)
    topic_docs = [(tid, t["text"]) for tid, t in topics.items()]
    ep_docs = [(eid, e["text"]) for eid, e in episodes.items()]
    topic_bm25 = base.bm25_rank_docs(example.question, topic_docs, min(40, len(topic_docs))) if topic_docs else []
    ep_bm25 = base.bm25_rank_docs(example.question, ep_docs, min(60, len(ep_docs))) if ep_docs else []
    topic_ids = [tid for tid, _ in topic_bm25[:8]]
    ep_ids = {eid for eid, _ in ep_bm25[:12]}
    ep_ids |= {eid for tid in topic_ids for eid in topics.get(tid, {}).get("episode_ids", [])}
    candidate_ids = {f.fact_id for eid in ep_ids for f in episodes.get(eid, {}).get("facts", [])}
    if len(candidate_ids) < 60:
        candidate_ids |= {f.fact_id for f in facts[:60]}
    by_id = {f.fact_id: f for f in facts}
    candidates = [by_id[fid] for fid in candidate_ids if fid in by_id]
    bm25_rank, dense_rank, _qvec, _emb = fact_scores(example, candidates, qwen_embed)
    fused = base.rrf_fuse([
        [(f.fact_id, s) for f, s in bm25_rank[:80]],
        [(f.fact_id, s) for f, s in dense_rank[:80]],
    ])
    candidates = [by_id[fid] for fid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:80] if fid in by_id]
    rerank_scores = qwen_reranker.rerank(example.question, [f.content for f in candidates]) if qwen_reranker and candidates else [fused.get(f.fact_id, 0.0) for f in candidates]
    ranked = [(fact, float(rr) + 0.04 * keyword_overlap(example.question, fact.content)) for fact, rr in zip(candidates, rerank_scores)]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return finalize_baseline_result(example, "LightRAG-Lite", ranked, started, len(candidates), top_k=20, max_tokens=1250)


def add_row(
    rows: List[Dict[str, Any]],
    method_name: str,
    example: Any,
    ret: ProfileRetrievalResult,
    train_seconds: float = 0.0,
) -> Dict[str, Any]:
    metrics = retrieval_metrics(example, ret)
    row = {
        "method": method_name,
        "qid": example.qid,
        "qtype": example.qtype,
        "complexity_bin": complexity_bin(example, memory_stats(example)),
        "size_bin": size_bin(memory_stats(example)),
        "question": example.question,
        "gold": example.answer,
        **metrics,
        "retrieval_tokens": ret.tokens,
        "retrieval_ms": ret.debug_scores[0].get("latency_ms", 0.0) if ret.debug_scores else 0.0,
        "num_facts": len(ret.selected_facts),
        "action": ret.debug_scores[0].get("action", "") if ret.debug_scores else "",
        "role_gate": ret.debug_scores[0].get("role_gate", False) if ret.debug_scores else False,
        "gated_roles": ret.debug_scores[0].get("gated_roles", "") if ret.debug_scores else "",
        "train_seconds": round(train_seconds, 4),
    }
    rows.append(row)
    return row


def summarize_compare(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for method in sorted({str(r["method"]) for r in rows}):
        part = [r for r in rows if r["method"] == method]
        judged = [float(r["judge_score"]) for r in part if r.get("judge_score") not in (None, "")]
        out.append(
            {
                "method": method,
                "n": len(part),
                "llm_acc": sum(judged) / len(judged) if judged else "",
                "llm_n": len(judged),
                "fact_hit": sum(float(r["fact_hit"]) for r in part) / len(part),
                "answer_recall": sum(float(r["answer_turn_recall"]) for r in part) / len(part),
                "all_hit": sum(float(r["all_answer_turns_hit"]) for r in part) / len(part),
                "avg_tokens": sum(float(r["retrieval_tokens"]) for r in part) / len(part),
                "avg_ms": sum(float(r["retrieval_ms"]) for r in part) / len(part),
                "p50_ms": statistics.median([float(r["retrieval_ms"]) for r in part]),
                "train_seconds": max(float(r.get("train_seconds") or 0.0) for r in part),
            }
        )
    return out


def qtype_counts(examples: Sequence[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ex in examples:
        counts[str(ex.qtype)] = counts.get(str(ex.qtype), 0) + 1
    return dict(sorted(counts.items()))


def fixed_role_balanced_action() -> RetrievalAction:
    return RetrievalAction(
        "RoleBalanced",
        MethodConfig("RoleBalanced", graph_gate="hypermem_full", top_k_facts=14, max_tokens=850, initial_candidates=70, topic_top_k=6, episode_top_k=12, lambda_prop=0.5),
        role_gate=True,
        snippet_words=80,
    )


def fixed_no_role_action() -> RetrievalAction:
    return RetrievalAction(
        "FullBalanced",
        MethodConfig("FullBalanced", graph_gate="hypermem_full", top_k_facts=14, max_tokens=850, initial_candidates=70, topic_top_k=6, episode_top_k=12, lambda_prop=0.5),
        role_gate=False,
        snippet_words=80,
    )


def fixed_full_recall_action() -> RetrievalAction:
    return RetrievalAction(
        "FullRecall",
        MethodConfig("FullRecall", graph_gate="hypermem_full", top_k_facts=18, max_tokens=1100, initial_candidates=90, topic_top_k=8, episode_top_k=16, lambda_prop=0.5),
        role_gate=False,
        snippet_words=88,
    )


def build_method_runs(
    ex: Any,
    wanted: set[str],
    harmony_action: RetrievalAction | None,
    harmony_policy: SubqueryRouteBandit | None,
    qwen_embed: QwenEmbeddingClient,
    qwen_reranker: QwenRerankerClient | None,
) -> List[Tuple[str, ProfileRetrievalResult]]:
    method_runs: List[Tuple[str, ProfileRetrievalResult]] = []
    if "HARMONY-Mem" in wanted and harmony_policy is not None:
        method_runs.append(("HARMONY-Mem", run_harmony_subquery(ex, harmony_policy, qwen_embed, qwen_reranker, atomic_action=harmony_action)))
    if "HARMONY-LegacyRouter" in wanted and harmony_action is not None:
        method_runs.append(("HARMONY-LegacyRouter", run_action(ex, harmony_action, "HARMONY-LegacyRouter", qwen_embed, qwen_reranker)))
    if "HARMONY-NoRL-RoleBalanced" in wanted:
        method_runs.append(("HARMONY-NoRL-RoleBalanced", run_action(ex, fixed_role_balanced_action(), "HARMONY-NoRL-RoleBalanced", qwen_embed, qwen_reranker)))
    if "HARMONY-NoRole" in wanted:
        method_runs.append(("HARMONY-NoRole", run_action(ex, fixed_no_role_action(), "HARMONY-NoRole", qwen_embed, qwen_reranker)))
    if "HARMONY-NoRL-FullRecall" in wanted:
        method_runs.append(("HARMONY-NoRL-FullRecall", run_action(ex, fixed_full_recall_action(), "HARMONY-NoRL-FullRecall", qwen_embed, qwen_reranker)))
    if "HyperMem-Flow" in wanted:
        method_runs.append(("HyperMem-Flow", run_method_config(ex, hypermem_config(), qwen_embed, qwen_reranker)))
    if "BM25-turn" in wanted:
        method_runs.append(("BM25-turn", run_method_config(ex, MethodConfig("BM25-turn", graph_gate="bm25", top_k_facts=8, max_tokens=520), None, None)))
    if "QwenEmb-turn" in wanted:
        method_runs.append(("QwenEmb-turn", run_method_config(ex, MethodConfig("QwenEmb-turn", graph_gate="qwen_dense", top_k_facts=12, max_tokens=800, initial_candidates=55), qwen_embed, qwen_reranker)))
    if "Mem0-Lite" in wanted:
        method_runs.append(("Mem0-Lite", retrieve_mem0_lite(ex, qwen_embed, qwen_reranker)))
    if "A-Mem-Lite" in wanted:
        method_runs.append(("A-Mem-Lite", retrieve_amem_lite(ex, qwen_embed, qwen_reranker)))
    if "HippoRAG-Lite" in wanted:
        method_runs.append(("HippoRAG-Lite", retrieve_hipporag_lite(ex, qwen_embed, qwen_reranker)))
    if "LightRAG-Lite" in wanted:
        method_runs.append(("LightRAG-Lite", retrieve_lightrag_lite(ex, qwen_embed, qwen_reranker)))
    # Qwen services exhibit a per-question warm-up/queue effect.  A fixed
    # method order would systematically charge that cost to HARMONY, which is
    # first in the paper table.  Rotate deterministically by qid so every
    # method occupies each position across the split.
    if len(method_runs) > 1:
        match = re.search(r"::qa_(\d+)$", str(ex.qid))
        offset = int(match.group(1)) % len(method_runs) if match else sum(ord(char) for char in str(ex.qid)) % len(method_runs)
        method_runs = method_runs[offset:] + method_runs[:offset]
    return method_runs


def run_test_example_worker(payload: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    qi = int(payload["qi"])
    ex = payload["example"]
    wanted = set(payload["wanted"])
    action_name = payload.get("harmony_action_name")
    actions = {action.name: action for action in action_space()}
    harmony_action = actions.get(action_name) if action_name else None
    qwen_embed = QwenEmbeddingClient(base_url=str(payload["qwen_embedding_url"]))
    qwen_reranker = (
        QwenRerankerClient(base_url=str(payload["qwen_reranker_url"]))
        if payload.get("use_qwen_reranker")
        else None
    )
    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    train_seconds = float(payload.get("train_seconds") or 0.0)
    method_runs = build_method_runs(ex, wanted, harmony_action, SubqueryRouteBandit(seed=int(payload.get("seed", 7))), qwen_embed, qwen_reranker)
    for method_name, ret in method_runs:
        row = add_row(rows, method_name, ex, ret, train_seconds if method_name == "HARMONY-Mem" else 0.0)
        traces.append(
            {
                **row,
                "evidence": [f.content for f in ret.selected_facts],
                "planner": ret.debug_scores[0].get("planner", {}) if ret.debug_scores else {},
                "scheduler": ret.debug_scores[0].get("scheduler_steps", []) if ret.debug_scores else [],
            }
        )
    return qi, rows, traces


def update_subquery_policy_from_result(
    example: Any,
    ret: ProfileRetrievalResult,
    policy: SubqueryRouteBandit,
    supervised_reward: float,
) -> None:
    if not ret.debug_scores:
        return
    steps = ret.debug_scores[0].get("scheduler_steps", [])
    if not isinstance(steps, list):
        return
    plan = decompose_query(example.question)
    task_by_id = {task.subquery_id: task for task in plan.subqueries}
    has_role_hint = bool(infer_query_roles(example.question, example.rows))
    for step in steps:
        task = task_by_id.get(str(step.get("subquery_id") or ""))
        route = str(step.get("route") or "")
        if task and route:
            marginal = float(step.get("marginal_evidence_gain") or 0.0)
            policy.update(task, route, has_role_hint, 0.65 * supervised_reward + 0.35 * marginal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=260)
    parser.add_argument("--train-size", type=int, default=60)
    parser.add_argument("--test-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-unit", choices=["conversation", "question"], default="conversation")
    parser.add_argument("--question-pattern", default="", help="Optional case-insensitive regex used to select evaluation questions.")
    parser.add_argument(
        "--methods",
        default="HARMONY-Mem,HARMONY-LegacyRouter,HyperMem-Flow,Mem0-Lite,A-Mem-Lite,HippoRAG-Lite,LightRAG-Lite,BM25-turn,QwenEmb-turn",
    )
    parser.add_argument("--qwen-embedding-url", default="http://localhost:11810/v1/embeddings")
    parser.add_argument("--qwen-reranker-url", default="http://localhost:12810")
    parser.add_argument("--use-qwen-reranker", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-llm-judge", type=int, default=0)
    parser.add_argument("--reader-model", default="deepseek-chat")
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--reader-mode", default="temporal")
    parser.add_argument("--skip-empty-gold", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {m.strip() for m in args.methods.split(",") if m.strip()}
    examples = load_examples(Path(args.data), max_examples=0)
    if args.question_pattern:
        question_pattern = re.compile(args.question_pattern, flags=re.I)
        examples = [example for example in examples if question_pattern.search(example.question)]
    examples = conversation_balanced_sample(examples, args.max_examples, args.seed)
    if args.skip_empty_gold:
        examples = [ex for ex in examples if str(ex.answer).strip()]
    train, test = split_examples(examples, args.train_size, args.test_size, args.seed, split_unit=args.split_unit)
    (out_dir / "split_info.json").write_text(
        json.dumps(
            {
                "max_examples": args.max_examples,
                "train_size": len(train),
                "test_size": len(test),
                "seed": args.seed,
                "split_unit": args.split_unit,
                "question_pattern": args.question_pattern,
                "conversation_keys": sorted({conversation_key(ex) for ex in examples}),
                "train_qtypes": qtype_counts(train),
                "test_qtypes": qtype_counts(test),
                "train_qids": [ex.qid for ex in train],
                "test_qids": [ex.qid for ex in test],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    qwen_embed = QwenEmbeddingClient(base_url=args.qwen_embedding_url)
    qwen_reranker = QwenRerankerClient(base_url=args.qwen_reranker_url) if args.use_qwen_reranker else None

    actions = action_space()
    router = ActionBandit(actions, seed=args.seed)
    subquery_policy = SubqueryRouteBandit(seed=args.seed)
    train_started = time.time()
    if "HARMONY-Mem" in wanted or "HARMONY-LegacyRouter" in wanted:
        for ex in train:
            action_idx = router.select(ex, train=True)
            ret = run_action(ex, actions[action_idx], "HARMONY-AtomicRouter/train", qwen_embed, qwen_reranker)
            router.update(ex, action_idx, reward(ex, retrieval_metrics(ex, ret), ret))
    if "HARMONY-Mem" in wanted:
        for ex in train:
            atomic_action = actions[router.select(ex, train=False)]
            ret = run_harmony_subquery(ex, subquery_policy, qwen_embed, qwen_reranker, atomic_action=atomic_action, train=True)
            update_subquery_policy_from_result(ex, ret, subquery_policy, reward(ex, retrieval_metrics(ex, ret), ret))
    train_seconds = time.time() - train_started
    (out_dir / "router_state.json").write_text(
        json.dumps({"legacy_router": router.dump(), "subquery_policy": subquery_policy.dump()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cache_path = out_dir / "llm_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    reader = LLMClient(model=args.reader_model) if args.max_llm_judge > 0 else None
    judge = LLMClient(model=args.judge_model) if args.max_llm_judge > 0 else None

    harmony_plan = {
        ex.qid: actions[router.select(ex, train=False)].name
        for ex in test
        if "HARMONY-Mem" in wanted or "HARMONY-LegacyRouter" in wanted
    }

    rows: List[Dict[str, Any]] = []
    trace_path = out_dir / "trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as trace:
        can_parallel = args.num_workers > 1 and args.max_llm_judge <= 0 and "HARMONY-Mem" not in wanted
        if can_parallel:
            payloads = [
                {
                    "qi": qi,
                    "example": ex,
                    "wanted": sorted(wanted),
                    "harmony_action_name": harmony_plan.get(ex.qid),
                    "qwen_embedding_url": args.qwen_embedding_url,
                    "qwen_reranker_url": args.qwen_reranker_url,
                    "use_qwen_reranker": bool(args.use_qwen_reranker),
                    "train_seconds": train_seconds,
                    "seed": args.seed,
                }
                for qi, ex in enumerate(test, start=1)
            ]
            max_workers = max(1, min(args.num_workers, len(payloads), os.cpu_count() or args.num_workers))
            completed: Dict[int, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(run_test_example_worker, payload) for payload in payloads]
                for fut in as_completed(futures):
                    qi, part_rows, part_traces = fut.result()
                    completed[qi] = (part_rows, part_traces)
                    print(f"[done] {len(completed)}/{len(test)}", flush=True)
            for qi in sorted(completed):
                part_rows, part_traces = completed[qi]
                rows.extend(part_rows)
                for item in part_traces:
                    trace.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            for qi, ex in enumerate(test, start=1):
                action = {a.name: a for a in actions}.get(harmony_plan.get(ex.qid, ""))
                method_runs = build_method_runs(ex, wanted, action, subquery_policy, qwen_embed, qwen_reranker)
                for method_name, ret in method_runs:
                    row = add_row(rows, method_name, ex, ret, train_seconds if method_name == "HARMONY-Mem" else 0.0)
                    if reader is not None and judge is not None and qi <= args.max_llm_judge:
                        key = f"{ex.qid}::{method_name}::reader={reader.model}::judge={judge.model}::mode={args.reader_mode}::locomo_main_v1"
                        judged = generate_and_judge(reader, judge, ex, ret, cache, key, args.reader_mode)
                        row.update({k: v for k, v in judged.items() if k != "judge_raw"})
                        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
                    trace.write(
                        json.dumps(
                            {
                                **row,
                                "evidence": [f.content for f in ret.selected_facts],
                                "planner": ret.debug_scores[0].get("planner", {}) if ret.debug_scores else {},
                                "scheduler": ret.debug_scores[0].get("scheduler_steps", []) if ret.debug_scores else [],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                print(f"[done] {qi}/{len(test)}", flush=True)

    write_csv(out_dir / "locomo_results.csv", rows)
    compare = summarize_compare(rows)
    write_csv(out_dir / "locomo_compare.csv", compare)
    print((out_dir / "locomo_compare.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
