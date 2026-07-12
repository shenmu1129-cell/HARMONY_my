#!/usr/bin/env python3
"""
PCH-Mem Comprehensive Experiment on LoCoMo.

Full-scale verification using the existing experiment infrastructure:
- train=60, test=80 (same as existing baselines)
- Qwen3-Embedding-4B + Qwen3-Reranker-4B
- GPT-4 judge via OpenRouter
- All baselines + PCH-Mem

Run:
  python experiments/run_pch_mem_comprehensive.py \
    --data data/locomo10.json \
    --output-dir outputs/pch_mem_comprehensive \
    --train-size 60 --test-size 80 \
    --max-llm-judge 80 \
    --methods "BM25-turn,QwenEmb-turn,HyperMem-Flow,Mem0-Lite,PCH-Mem"
"""

from __future__ import annotations

import argparse, csv, hashlib, json, os, random, statistics, sys, time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.eval_longmemeval_mini import (
    MethodConfig as OrigMethodConfig,
    QwenEmbeddingClient, QwenRerankerClient,
    build_memory, estimate_tokens, load_examples,
    pack_ranked, retrieve_method, retrieval_metrics,
    write_csv, generate_and_judge,
    BM25Index, ProfileFact, ProfileRetrievalResult,
    LLMClient, retrieve_hypermem_full,
)
from experiments.eval_locomo_comparison import (
    split_examples, qtype_counts,
    source_snippet_result,
    retrieve_mem0_lite, retrieve_amem_lite,
    retrieve_hipporag_lite, retrieve_lightrag_lite,
)

from hypermem.pch_mem.types import (
    PolicyHyperedge, PolicyEdgeStatus, HypergraphState, PCHConfig,
)


# ═══════════════════════════════════════════════════════════════════
# PCH-Mem Retrieval Method
# ═══════════════════════════════════════════════════════════════════

def _hash_embed(text: str, dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for w in text.lower().split():
        d = hashlib.md5(w.encode()).hexdigest()
        idx = int(d[:8], 16) % dim
        sign = 1.0 if int(d[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def build_pch_policy_edges(
    train_examples: List[Any],
    memory: Any,
    qwen_embed: Any,
    max_edges: int = 15,
) -> Tuple[List[PolicyHyperedge], np.ndarray, List[str]]:
    """Build policy hyperedges from training examples.

    For each training example, note which facts were the gold evidence.
    Bundle frequently co-occurring facts into policy hyperedges.
    """
    # Collect gold evidence patterns from training data
    fact_cooccur: Dict[Tuple[str, ...], int] = Counter()
    fact_gold_count: Dict[str, int] = Counter()

    method = OrigMethodConfig("train", graph_gate="hypermem_full", top_k_facts=20, max_tokens=1200,
                              initial_candidates=90, topic_top_k=8, episode_top_k=16, lambda_prop=0.5)

    for ex in train_examples[:60]:
        try:
            ret = retrieve_hypermem_full(ex, memory, method, qwen_embed, None)
            gold_ids = set(getattr(ex, 'evidence_ids', []) or [])
            # Map memory fact_ids to our fact content
            retrieved = [f.fact_id for f in ret.selected_facts[:20]]
            # Find which retrieved facts overlap with gold
            gold_hits = [fid for fid in retrieved if fid in gold_ids or any(
                gid in fid for gid in gold_ids
            )]
            if len(gold_hits) >= 2:
                fact_cooccur[tuple(sorted(gold_hits)[:6])] += 1
            for fid in gold_hits:
                fact_gold_count[fid] += 1
        except Exception:
            continue

    # Create policy edges from top co-occurring fact bundles
    pch_edges = []
    for fact_tuple, count in fact_cooccur.most_common(max_edges):
        if count < 2:
            continue
        pe = PolicyHyperedge(
            edge_id=f"pe_full_{len(pch_edges):03d}",
            status=PolicyEdgeStatus.ACTIVE,
            fact_ids=list(fact_tuple),
            advantage_mean=count / max(1, len(train_examples)),
            advantage_lcb=(count - 1) / max(1, len(train_examples)),
            validation_query_count=len(train_examples),
            validation_consistency=count / max(1, sum(fact_cooccur.values())),
        )
        pch_edges.append(pe)

    # Also create PEs from individual high-frequency gold facts
    top_gold_facts = [fid for fid, c in fact_gold_count.most_common(max_edges * 2) if c >= 2]
    # Group by co-occurrence in same training example
    for fid in top_gold_facts[:max_edges]:
        # Find other facts that co-occur with this one
        co_occurring = []
        for ft, cnt in fact_cooccur.items():
            if fid in ft:
                co_occurring.extend(f for f in ft if f != fid)
        if co_occurring:
            bundle = [fid] + list(dict.fromkeys(co_occurring))[:4]
            pe = PolicyHyperedge(
                edge_id=f"pe_single_{len(pch_edges):03d}",
                status=PolicyEdgeStatus.ACTIVE,
                fact_ids=bundle,
                advantage_mean=fact_gold_count[fid] / max(1, len(train_examples)),
                validation_query_count=len(train_examples),
            )
            pch_edges.append(pe)

    # Build lookup structures
    pe_embeddings = np.zeros((len(pch_edges), 256), dtype=np.float32)
    pe_ids = []
    for i, pe in enumerate(pch_edges):
        embs = [_hash_embed(str(fid)) for fid in pe.fact_ids]
        pe_embeddings[i] = np.mean(embs, axis=0) if embs else np.zeros(256)
        pe_ids.append(pe.edge_id)

    return pch_edges, pe_embeddings, pe_ids


def retrieve_pch_mem(
    example: Any,
    memory: Any,
    method: Any,
    qwen_embed: Any,
    qwen_reranker: Any | None,
    pch_edges: List[PolicyHyperedge],
    pe_embeddings: np.ndarray,
    pe_ids: List[str],
) -> ProfileRetrievalResult:
    """PCH-Mem retrieval: policy edges + full pipeline fallback.

    Strategy:
    1. Try to match query against policy edges (fast O(1) lookup)
    2. If matched, use PE facts as primary evidence
    3. Supplement with lightweight BM25 for coverage
    4. If no match, fall back to full HyperMem pipeline
    """
    started = time.time()

    # Query embedding for PE matching
    try:
        qemb = qwen_embed.embed([example.question])[0] if qwen_embed else _hash_embed(example.question)
    except Exception:
        qemb = _hash_embed(example.question)

    # Try PE matching
    pe_matched_facts: List[str] = []
    if len(pe_embeddings) > 0:
        qemb_norm = np.array(qemb) / (np.linalg.norm(np.array(qemb)) + 1e-8)
        pe_norm = pe_embeddings / (np.linalg.norm(pe_embeddings, axis=1, keepdims=True) + 1e-8)
        scores = pe_norm @ qemb_norm
        top_idx = int(np.argmax(scores))
        if scores[top_idx] > 0.3:
            pe = pch_edges[top_idx]
            pe_matched_facts = list(pe.fact_ids)

    if pe_matched_facts:
        # Fast path: PE facts + BM25 supplement
        facts = list(memory.facts.values())
        bm25 = BM25Index(facts)
        bm25_results = bm25.search(example.question, top_k=50)
        bm25_fact_ids = [fid for fid, _ in bm25_results]

        # Merge: PE first (pre-verified), then BM25
        merged_ids = pe_matched_facts + [f for f in bm25_fact_ids if f not in pe_matched_facts]
        top_ids = merged_ids[:max(method.top_k_facts, 20)]

        selected = [f for f in facts if f.fact_id in top_ids]
        if not selected:
            selected = [f for f in facts if f.fact_id in bm25_fact_ids[:method.top_k_facts]]

        elapsed = (time.time() - started) * 1000
        return ProfileRetrievalResult(
            query=example.question,
            channel="PCH-Mem/fast",
            selected_edges=[],
            selected_facts=selected[:method.top_k_facts],
            score=float(scores[top_idx]) if len(pe_embeddings) > 0 else 0.0,
            tokens=estimate_tokens([f.content for f in selected[:method.top_k_facts]]),
            fallback_used=False,
            sufficient=len(selected) >= 3,
            debug_scores=[{
                "path": "pch_mem_fast",
                "pe_facts": len(pe_matched_facts),
                "bm25_supplement": max(0, len(top_ids) - len(pe_matched_facts)),
                "total_facts": len(selected),
                "latency_ms": elapsed,
            }],
        )

    # Fallback: full HyperMem pipeline
    ret = retrieve_hypermem_full(example, memory, method, qwen_embed, qwen_reranker)
    ret.channel = "PCH-Mem/fallback"
    if ret.debug_scores:
        ret.debug_scores[0]["path"] = "pch_mem_fallback"
    return ret


# ═══════════════════════════════════════════════════════════════════
# Main Experiment Runner
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PCH-Mem Comprehensive Experiment")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=260)
    parser.add_argument("--train-size", type=int, default=60)
    parser.add_argument("--test-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", default="BM25-turn,QwenEmb-turn,HyperMem-Flow,Mem0-Lite,PCH-Mem")
    parser.add_argument("--qwen-embedding-url", default="http://localhost:11810/v1/embeddings")
    parser.add_argument("--qwen-reranker-url", default="http://localhost:12810")
    parser.add_argument("--use-qwen-reranker", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-llm-judge", type=int, default=80)
    parser.add_argument("--reader-model", default="gpt-4.1-mini")
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--reader-mode", default="temporal")
    parser.add_argument("--skip-empty-gold", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and split data
    examples = load_examples(Path(args.data), max_examples=args.max_examples)
    if args.skip_empty_gold:
        examples = [ex for ex in examples if str(ex.answer).strip()]
    train, test = split_examples(examples, args.train_size, args.test_size, args.seed)

    print("=" * 70)
    print("PCH-Mem Comprehensive Experiment")
    print(f"  Train: {len(train)}, Test: {len(test)}")
    print(f"  Methods: {args.methods}")
    print(f"  LLM Judge: {args.max_llm_judge} examples")
    print(f"  Reader: {args.reader_model}, Judge: {args.judge_model}")
    print("=" * 70)

    # Save split info
    (out_dir / "split_info.json").write_text(json.dumps({
        "train_size": len(train), "test_size": len(test),
        "seed": args.seed,
        "train_qids": [ex.qid for ex in train],
        "test_qids": [ex.qid for ex in test],
    }, indent=2), encoding="utf-8")

    # Init services
    qwen_embed = QwenEmbeddingClient(base_url=args.qwen_embedding_url)
    qwen_reranker = QwenRerankerClient(base_url=args.qwen_reranker_url) if args.use_qwen_reranker else None

    # Build memory for all examples
    print("\nBuilding memory from all dialogue rows...")
    all_rows = []
    for ex in examples[:args.max_examples]:
        all_rows.extend(ex.rows)
    memory = build_memory(all_rows)
    print(f"  Memory: {len(memory.facts)} facts, {len(memory.edges)} edges")

    # Build PCH-Mem policy edges from training data
    wanted_methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    pch_edges, pe_embeddings, pe_ids = [], np.zeros((0, 256)), []
    if "PCH-Mem" in wanted_methods:
        print("\nBuilding PCH-Mem policy edges from training data...")
        pch_edges, pe_embeddings, pe_ids = build_pch_policy_edges(train, memory, qwen_embed)
        print(f"  Created {len(pch_edges)} policy edges")

    # Setup LLM
    cache_path = out_dir / "llm_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    reader = LLMClient(model=args.reader_model) if args.max_llm_judge > 0 else None
    judge = LLMClient(model=args.judge_model) if args.max_llm_judge > 0 else None

    # Run baselines on test set
    rows: List[Dict[str, Any]] = []
    pch_method = OrigMethodConfig("PCH-Mem", graph_gate="hypermem_full", top_k_facts=20, max_tokens=1200,
                                  initial_candidates=90, topic_top_k=8, episode_top_k=16, lambda_prop=0.5)

    for qi, ex in enumerate(test):
        ex_label = f"[{qi+1}/{len(test)}]"
        print(f"\n{ex_label} Q: {ex.question[:100]}...")

        for method_name in sorted(wanted_methods):
            t0 = time.time()

            if method_name == "PCH-Mem":
                ret = retrieve_pch_mem(ex, memory, pch_method, qwen_embed, qwen_reranker,
                                       pch_edges, pe_embeddings, pe_ids)
            elif method_name == "HyperMem-Flow":
                hypermem_method = OrigMethodConfig("HyperMem-Flow", graph_gate="hypermem_full",
                                                   top_k_facts=20, max_tokens=1200,
                                                   initial_candidates=90, topic_top_k=8,
                                                   episode_top_k=16, lambda_prop=0.5)
                ret = retrieve_hypermem_full(ex, memory, hypermem_method, qwen_embed, qwen_reranker)
            elif method_name == "BM25-turn":
                bm25_method = OrigMethodConfig("BM25-turn", graph_gate="bm25", top_k_facts=8, max_tokens=520)
                ret = retrieve_method(ex, memory, bm25_method, None, None)
            elif method_name == "QwenEmb-turn":
                dense_method = OrigMethodConfig("QwenEmb-turn", graph_gate="qwen_dense", top_k_facts=12,
                                                max_tokens=800, initial_candidates=55)
                ret = retrieve_method(ex, memory, dense_method, qwen_embed, qwen_reranker)
            elif method_name == "Mem0-Lite":
                ret = retrieve_mem0_lite(ex, memory)
            elif method_name == "A-Mem-Lite":
                ret = retrieve_amem_lite(ex, memory)
            elif method_name == "HippoRAG-Lite":
                ret = retrieve_hipporag_lite(ex, memory)
            elif method_name == "LightRAG-Lite":
                ret = retrieve_lightrag_lite(ex, memory)
            else:
                print(f"  Unknown method: {method_name}, skipping")
                continue

            elapsed_ms = (time.time() - t0) * 1000

            # Snippet formatting for LLM
            ret = source_snippet_result(ret, ex.question, method_name, max_words=96)

            # LLM evaluation
            llm_acc = None
            if reader and judge and qi < args.max_llm_judge:
                cache_key = f"{ex.qid}::{method_name}::reader={reader.model}::judge={judge.model}::mode={args.reader_mode}::pch_v1"
                if cache_key not in cache:
                    try:
                        result = generate_and_judge(ex, ret, reader, judge, mode=args.reader_mode)
                        cache[cache_key] = {
                            "llm_acc": result.get("is_correct", None),
                            "llm_answer": str(result.get("answer", ""))[:500],
                        }
                    except Exception as e:
                        print(f"  LLM judge error: {e}")
                        cache[cache_key] = {"llm_acc": None, "llm_answer": ""}

                llm_acc = cache[cache_key].get("llm_acc")

            # Retrieval metrics
            rmetrics = retrieval_metrics(ex, ret)

            row = {
                "qid": ex.qid,
                "method": method_name,
                "question": ex.question[:200],
                "llm_acc": llm_acc,
                "fact_hit": rmetrics.get("fact_hit", 0),
                "answer_recall": rmetrics.get("answer_recall", 0),
                "all_hit": rmetrics.get("all_hit", 0),
                "avg_tokens": ret.tokens,
                "avg_ms": elapsed_ms,
                "p50_ms": elapsed_ms,
                "facts_returned": len(ret.selected_facts),
                "channel": ret.channel,
            }
            rows.append(row)

            status = f"acc={llm_acc}" if llm_acc is not None else "no_judge"
            print(f"    {method_name}: {status}, tokens={ret.tokens}, "
                  f"ms={elapsed_ms:.0f}, facts={len(ret.selected_facts)}, "
                  f"hit={rmetrics.get('fact_hit', 0):.2f}")

        # Save incrementally
        write_csv(out_dir / "pch_mem_results.csv", rows)

    # Save cache
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("COMPREHENSIVE RESULTS")
    print("=" * 70)

    by_method: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    summary_rows = []
    for method_name in sorted(by_method.keys()):
        mrows = by_method[method_name]
        n = len(mrows)
        n_judged = sum(1 for r in mrows if r["llm_acc"] is not None)
        acc = statistics.mean([r["llm_acc"] for r in mrows if r["llm_acc"] is not None]) if n_judged > 0 else 0
        fact_hit = statistics.mean([r["fact_hit"] for r in mrows])
        tokens = statistics.mean([r["avg_tokens"] for r in mrows])
        latency = statistics.mean([r["avg_ms"] for r in mrows])
        channels = Counter(r["channel"] for r in mrows)

        print(f"\n{method_name} (n={n}, judged={n_judged}):")
        print(f"  LLM Acc: {acc:.4f}")
        print(f"  Fact Hit: {fact_hit:.4f}")
        print(f"  Avg Tokens: {tokens:.0f}")
        print(f"  Avg Latency: {latency:.0f}ms")
        print(f"  Channels: {dict(channels)}")

        summary_rows.append({
            "method": method_name, "n": n, "llm_acc": round(acc, 4),
            "fact_hit": round(fact_hit, 4), "avg_tokens": round(tokens, 0),
            "avg_ms": round(latency, 0),
        })

    write_csv(out_dir / "pch_mem_summary.csv", summary_rows)
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
