from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from eval_longmemeval_mini import (
    BM25Index,
    LLMClient,
    MethodConfig,
    ProfileFact,
    ProfileRetrievalResult,
    QwenEmbeddingClient,
    QwenRerankerClient,
    ThompsonRouter,
    build_memory,
    estimate_tokens,
    generate_and_judge,
    keyword_overlap,
    load_examples,
    pack_ranked,
    result,
    retrieval_metrics,
    rrf_fuse,
    summarize,
    write_csv,
)


def ablation_actions() -> List[MethodConfig]:
    return [
        MethodConfig("A1-compact", graph_gate="hypermem_full", top_k_facts=12, max_tokens=800, initial_candidates=55, topic_top_k=4, episode_top_k=8, lambda_prop=0.5),
        MethodConfig("A2-balanced", graph_gate="hypermem_full", top_k_facts=20, max_tokens=1250, initial_candidates=70, topic_top_k=6, episode_top_k=12, lambda_prop=0.5),
        MethodConfig("A3-recall", graph_gate="hypermem_full", top_k_facts=24, max_tokens=1500, initial_candidates=85, topic_top_k=8, episode_top_k=16, lambda_prop=0.5),
        MethodConfig("A4-broad", graph_gate="hypermem_full", top_k_facts=28, max_tokens=1700, initial_candidates=100, topic_top_k=10, episode_top_k=18, lambda_prop=0.5),
    ]


def retrieve_full(
    example: Any,
    memory: Any,
    action: MethodConfig,
    qwen_embed: QwenEmbeddingClient,
    qwen_reranker: QwenRerankerClient | None,
    method_name: str,
) -> ProfileRetrievalResult:
    from eval_longmemeval_mini import retrieve_method

    ret = retrieve_method(example, memory, action, qwen_embed=qwen_embed, qwen_reranker=qwen_reranker)
    ret.channel = method_name
    if ret.debug_scores:
        ret.debug_scores[0]["method"] = method_name
        ret.debug_scores[0]["action"] = action.name
    return ret


def retrieve_flat_rrf(
    example: Any,
    memory: Any,
    action: MethodConfig,
    qwen_embed: QwenEmbeddingClient,
    qwen_reranker: QwenRerankerClient | None,
) -> ProfileRetrievalResult:
    started = time.time()
    facts = list(memory.facts.values())
    docs = [(f.fact_id, f.content) for f in facts]
    bm25 = BM25Index(facts)
    bm25_rank = [(fact.fact_id, bm25.score(example.question, i)) for i, fact in enumerate(facts)]
    bm25_rank.sort(key=lambda x: x[1], reverse=True)

    vectors = qwen_embed.embed([example.question] + [text for _, text in docs])
    qvec, dvecs = vectors[0], vectors[1:]
    dense_rank = [(fact.fact_id, qwen_embed.cosine(qvec, vec)) for fact, vec in zip(facts, dvecs)]
    dense_rank.sort(key=lambda x: x[1], reverse=True)
    fused = rrf_fuse([bm25_rank[: action.initial_candidates], dense_rank[: action.initial_candidates]])
    candidate_ids = [fid for fid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[: action.initial_candidates]]
    by_id = {f.fact_id: f for f in facts}
    candidates = [by_id[fid] for fid in candidate_ids if fid in by_id]
    if qwen_reranker is not None and candidates:
        rerank_scores = qwen_reranker.rerank(example.question, [f.content for f in candidates])
    else:
        rerank_scores = [fused.get(f.fact_id, 0.0) for f in candidates]

    ranked: List[Tuple[ProfileFact, float]] = []
    for fact, score in zip(candidates, rerank_scores):
        ranked.append((fact, float(score) + 0.05 * keyword_overlap(example.question, fact.content)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    selected = pack_ranked(ranked, action.top_k_facts, action.max_tokens)
    avg = sum(s for _, s in ranked[: max(1, action.top_k_facts)]) / max(1, min(len(ranked), action.top_k_facts))
    return result(example.question, "NoHyperedge-FlatRRF", selected, [], avg, started, len(candidates))


def to_summary_leaf(ret: ProfileRetrievalResult) -> ProfileRetrievalResult:
    groups: Dict[str, List[ProfileFact]] = {}
    for fact in ret.selected_facts:
        sid = str(fact.metadata.get("session_id") or "unknown")
        groups.setdefault(sid, []).append(fact)

    summary_facts: List[ProfileFact] = []
    for idx, (sid, facts) in enumerate(groups.items(), start=1):
        first = facts[0]
        summary_hint = str(first.metadata.get("session_summary") or "").strip()
        if summary_hint:
            content = summary_hint
        else:
            snippets = [str(f.metadata.get("raw_content") or f.content).replace("\n", " ").strip() for f in facts[:4]]
            content = f"Compressed summary for {sid}: " + " ".join(snippets)
        metadata = dict(first.metadata)
        metadata["raw_content"] = content
        metadata["has_answer"] = any(bool(f.metadata.get("has_answer")) for f in facts)
        metadata["summary_leaf"] = True
        summary_facts.append(
            ProfileFact(
                fact_id=f"summary::{sid}::{idx}",
                content=content[:1800],
                timestamp=float(idx),
                embedding=[],
                metadata=metadata,
            )
        )

    out = ProfileRetrievalResult(
        query=ret.query,
        channel="NoSourcePreserve-SummaryLeaf",
        selected_edges=ret.selected_edges,
        selected_facts=summary_facts,
        score=ret.score,
        tokens=estimate_tokens([f.content for f in summary_facts]),
        fallback_used=ret.fallback_used,
        sufficient=bool(summary_facts),
        debug_scores=[dict(ret.debug_scores[0], method="NoSourcePreserve-SummaryLeaf") if ret.debug_scores else {"method": "NoSourcePreserve-SummaryLeaf"}],
    )
    return out


def reward_for_router(metrics: Dict[str, Any], ret: ProfileRetrievalResult) -> float:
    latency = float(ret.debug_scores[0].get("latency_ms", 0.0)) if ret.debug_scores else 0.0
    reward = 0.55 * metrics["fact_hit"] + 0.30 * metrics["answer_turn_recall"] + 0.15 * metrics["all_answer_turns_hit"]
    reward -= min(0.16, ret.tokens / 9000.0)
    reward -= min(0.06, latency / 14000.0)
    return reward


def avg(rows: Sequence[Dict[str, Any]], key: str) -> float:
    return sum(float(r.get(key, 0.0)) for r in rows) / max(1, len(rows))


def judged_avg(rows: Sequence[Dict[str, Any]]) -> float | str:
    judged = [float(r.get("judge_score", 0.0)) for r in rows if "judge_score" in r]
    if not judged:
        return ""
    return sum(judged) / len(judged)


def rebuild_stratified_split(examples: Sequence[Any], train_size: int, base_train_size: int) -> Tuple[List[Any], List[Any]]:
    base_train = list(examples[: min(base_train_size, train_size, len(examples))])
    selected_ids = {ex.qid for ex in base_train}
    groups: Dict[str, List[Any]] = {}
    for ex in examples[len(base_train) :]:
        groups.setdefault(str(ex.qtype), []).append(ex)
    ordered_types = sorted(groups, key=lambda key: (0 if "multi" in key else 1 if "temporal" in key else 2, key))
    train = list(base_train)
    while len(train) < min(train_size, len(examples)):
        progressed = False
        for qtype in ordered_types:
            bucket = groups.get(qtype) or []
            while bucket and bucket[0].qid in selected_ids:
                bucket.pop(0)
            if not bucket:
                continue
            ex = bucket.pop(0)
            train.append(ex)
            selected_ids.add(ex.qid)
            progressed = True
            if len(train) >= train_size:
                break
        if not progressed:
            break
    test = [ex for ex in examples if ex.qid not in selected_ids]
    return train, test


def interleave_by_qtype(examples: Sequence[Any]) -> List[Any]:
    groups: Dict[str, List[Any]] = {}
    for ex in examples:
        groups.setdefault(str(ex.qtype), []).append(ex)
    ordered_types = sorted(groups, key=lambda key: (0 if "temporal" in key else 1 if "multi" in key else 2, key))
    out: List[Any] = []
    while True:
        progressed = False
        for qtype in ordered_types:
            bucket = groups.get(qtype) or []
            if not bucket:
                continue
            out.append(bucket.pop(0))
            progressed = True
        if not progressed:
            break
    return out


def clone_with_method(ret: ProfileRetrievalResult, method_name: str) -> ProfileRetrievalResult:
    debug = [dict(item) for item in ret.debug_scores]
    if debug:
        debug[0]["method"] = method_name
    return ProfileRetrievalResult(
        query=ret.query,
        channel=method_name,
        selected_edges=ret.selected_edges,
        selected_facts=ret.selected_facts,
        score=ret.score,
        tokens=ret.tokens,
        fallback_used=ret.fallback_used,
        sufficient=ret.sufficient,
        debug_scores=debug,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=40)
    parser.add_argument("--train-size", type=int, default=10)
    parser.add_argument("--max-llm-judge", type=int, default=10)
    parser.add_argument("--stratified-rebuild", action="store_true")
    parser.add_argument("--base-train-size", type=int, default=10)
    parser.add_argument("--interleave-test", action="store_true")
    parser.add_argument("--reader-model", default="deepseek-chat")
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--reader-mode", default="temporal")
    parser.add_argument("--qwen-embedding-url", default="http://localhost:11810/v1/embeddings")
    parser.add_argument("--qwen-reranker-url", default="http://localhost:12810")
    parser.add_argument("--use-qwen-reranker", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(Path(args.data), max_examples=args.max_examples)
    if args.stratified_rebuild:
        train, test = rebuild_stratified_split(examples, args.train_size, args.base_train_size)
    else:
        train = examples[: min(args.train_size, len(examples))]
        test = examples[min(args.train_size, len(examples)) :]
    if args.interleave_test:
        test = interleave_by_qtype(test)
    actions = ablation_actions()
    qwen_embed = QwenEmbeddingClient(base_url=args.qwen_embedding_url)
    qwen_reranker = QwenRerankerClient(base_url=args.qwen_reranker_url) if args.use_qwen_reranker else None
    router = ThompsonRouter(actions, seed=29)
    action_counts: Dict[str, int] = {action.name: 0 for action in actions}

    for ex in train:
        memory = build_memory(ex.rows)
        arm_idx = router.select(ex, train=True)
        ret = retrieve_full(ex, memory, actions[arm_idx], qwen_embed, qwen_reranker, "Full-HG-RL-4A")
        metrics = retrieval_metrics(ex, ret)
        router.update(ex, arm_idx, reward_for_router(metrics, ret))

    cache_path = out_dir / "llm_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    reader = LLMClient(model=args.reader_model) if args.max_llm_judge > 0 else None
    judge = LLMClient(model=args.judge_model) if args.max_llm_judge > 0 else None
    rows: List[Dict[str, Any]] = []

    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as trace:
        for qi, ex in enumerate(test, start=1):
            memory = build_memory(ex.rows)
            method_runs: List[Tuple[str, ProfileRetrievalResult]] = []

            arm_idx = router.select(ex, train=False)
            action_counts[actions[arm_idx].name] += 1
            full_ret = retrieve_full(ex, memory, actions[arm_idx], qwen_embed, qwen_reranker, "Full-HG-RL-4A")
            method_runs.append(("Full-HG-RL-4A", full_ret))
            method_runs.append(("NoSourcePreserve-SummaryLeaf", to_summary_leaf(full_ret)))

            fixed_rets: Dict[str, ProfileRetrievalResult] = {}
            for action in actions:
                method_name = f"NoRL-Fixed-{action.name}"
                fixed_ret = retrieve_full(ex, memory, action, qwen_embed, qwen_reranker, method_name)
                fixed_rets[action.name] = fixed_ret
                method_runs.append((method_name, fixed_ret))

            rule_action = "A3-recall" if "multi" in ex.qtype else "A1-compact"
            if rule_action in fixed_rets:
                method_runs.append(("QTypeRule-A1Temp-A3Multi", clone_with_method(fixed_rets[rule_action], "QTypeRule-A1Temp-A3Multi")))

            method_runs.append(("NoHyperedge-FlatRRF", retrieve_flat_rrf(ex, memory, actions[2], qwen_embed, qwen_reranker)))

            for method_name, ret in method_runs:
                metrics = retrieval_metrics(ex, ret)
                row = {
                    "method": method_name,
                    "qid": ex.qid,
                    "qtype": ex.qtype,
                    "question": ex.question,
                    "gold": ex.answer,
                    **metrics,
                    "retrieval_tokens": ret.tokens,
                    "retrieval_ms": ret.debug_scores[0].get("latency_ms", 0.0) if ret.debug_scores else 0.0,
                    "num_facts": len(ret.selected_facts),
                    "action": ret.debug_scores[0].get("action", "") if ret.debug_scores else "",
                }
                if reader is not None and judge is not None and qi <= args.max_llm_judge:
                    cache_key = f"{ex.qid}::{method_name}::reader={reader.model}::judge={judge.model}::mode={args.reader_mode}::ablation_v2"
                    judged = generate_and_judge(reader, judge, ex, ret, cache, cache_key, args.reader_mode)
                    row.update({k: v for k, v in judged.items() if k != "judge_raw"})
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
                rows.append(row)
                trace.write(json.dumps({**row, "evidence": [f.content for f in ret.selected_facts]}, ensure_ascii=False) + "\n")
            print(f"[done] {qi}/{len(test)}", flush=True)

    write_csv(out_dir / "module_ablation_results.csv", rows)
    summary = summarize(rows)
    write_csv(out_dir / "module_ablation_summary.csv", summary)

    extra_rows: List[Dict[str, Any]] = []
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    full = by_method.get("Full-HG-RL-4A", [])
    for method, part in by_method.items():
        extra_rows.append(
            {
                "method": method,
                "n": len(part),
                "llm_acc": judged_avg(part),
                "fact_hit": avg(part, "fact_hit"),
                "answer_recall": avg(part, "answer_turn_recall"),
                "avg_tokens": avg(part, "retrieval_tokens"),
                "avg_ms": avg(part, "retrieval_ms"),
                "p50_ms": statistics.median([float(r.get("retrieval_ms", 0.0)) for r in part]) if part else 0.0,
                "delta_acc_vs_full": (judged_avg(part) - judged_avg(full)) if full and judged_avg(part) != "" and judged_avg(full) != "" else "",
                "token_ratio_vs_full": avg(part, "retrieval_tokens") / max(1e-9, avg(full, "retrieval_tokens")) if full else "",
            }
        )
    write_csv(out_dir / "module_ablation_compare.csv", extra_rows)
    (out_dir / "action_counts.json").write_text(json.dumps(action_counts, ensure_ascii=False, indent=2), encoding="utf-8")
    split_info = {
        "train_qtypes": {qtype: sum(1 for ex in train if ex.qtype == qtype) for qtype in sorted({ex.qtype for ex in train})},
        "test_qtypes": {qtype: sum(1 for ex in test if ex.qtype == qtype) for qtype in sorted({ex.qtype for ex in test})},
        "train_qids": [ex.qid for ex in train],
        "test_qids": [ex.qid for ex in test],
    }
    (out_dir / "split_info.json").write_text(json.dumps(split_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print((out_dir / "module_ablation_summary.csv").read_text(encoding="utf-8"))
    print("ACTION_COUNTS", json.dumps(action_counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
