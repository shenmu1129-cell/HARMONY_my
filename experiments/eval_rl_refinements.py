from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import eval_longmemeval_mini as base
from eval_longmemeval_mini import (
    LLMClient,
    MethodConfig,
    ProfileEdgeType,
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
    summarize,
    write_csv,
)
from eval_rl_memory_complexity import complexity_bin, memory_stats, size_bin


BEHAVIOR_CATEGORIES: Dict[str, Sequence[str]] = {
    "travel": ("trip", "travel", "flight", "hotel", "tokyo", "denver", "hawaii", "europe", "airport", "vacation"),
    "health": ("doctor", "health", "cough", "fitness", "gym", "run", "sleep", "appointment", "medicine"),
    "work": ("work", "job", "office", "company", "colleague", "meeting", "project", "google"),
    "family": ("family", "mother", "father", "brother", "sister", "parent", "grandparent", "birthday"),
    "shopping": ("buy", "bought", "order", "purchase", "gift", "device", "furniture", "jewelry"),
    "food": ("cook", "recipe", "restaurant", "food", "dinner", "coffee", "cocktail", "kitchen"),
    "entertainment": ("movie", "show", "festival", "game", "book", "music", "concert", "watch"),
    "home": ("home", "apartment", "bedroom", "furniture", "kitchen", "clean", "move", "studio"),
    "tech": ("phone", "ipad", "router", "thermostat", "nas", "device", "battery", "camera"),
    "finance": ("money", "spent", "cost", "raise", "charity", "price", "budget"),
    "art": ("paint", "painting", "art", "gallery", "craft", "design"),
}


def classify_text(text: str) -> str:
    toks = set(base.tokenize(text))
    best = ("misc", 0)
    for cat, words in BEHAVIOR_CATEGORIES.items():
        score = len(toks & set(words))
        if score > best[1]:
            best = (cat, score)
    return best[0]


def better_episode_topic_docs(facts: Sequence[ProfileFact]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    episodes, _old_topics = base.build_episode_topic_docs_original(facts)  # type: ignore[attr-defined]
    buckets: Dict[str, List[str]] = {}
    for sid, ep in episodes.items():
        cat = classify_text(ep["text"])
        if cat == "misc":
            date = str(ep.get("date") or "")
            cat = f"month_{date[:7]}" if len(date) >= 7 else "misc"
        buckets.setdefault(cat, []).append(sid)
    topics: Dict[str, Dict[str, Any]] = {}
    for idx, (cat, sids) in enumerate(sorted(buckets.items()), start=1):
        topic_id = f"topic_{idx:03d}_{cat}"
        preview = "\n".join(episodes[sid]["text"][:520] for sid in sids[:8])
        topics[topic_id] = {
            "id": topic_id,
            "episode_ids": sids,
            "text": f"Behavioral/topic cluster {cat}. Episodes: {', '.join(sids[:12])}\n{preview}",
        }
    return episodes, topics


def enable_better_topics() -> None:
    if not hasattr(base, "build_episode_topic_docs_original"):
        base.build_episode_topic_docs_original = base.build_episode_topic_docs  # type: ignore[attr-defined]
    base.build_episode_topic_docs = better_episode_topic_docs


def disable_better_topics() -> None:
    if hasattr(base, "build_episode_topic_docs_original"):
        base.build_episode_topic_docs = base.build_episode_topic_docs_original  # type: ignore[attr-defined]


def augment_behavior_edges(memory: Any) -> None:
    by_cat: Dict[str, List[str]] = {}
    for fact in memory.facts.values():
        raw = str(fact.metadata.get("raw_content") or fact.content)
        cat = classify_text(raw)
        if cat != "misc":
            by_cat.setdefault(cat, []).append(fact.fact_id)
    for cat, fids in by_cat.items():
        if len(fids) < 2:
            continue
        memory.create_edge(
            ProfileEdgeType.HABIT if cat in {"food", "health", "entertainment"} else ProfileEdgeType.PREFERENCE,
            fids,
            summary=f"Behavioral hyperedge about {cat}: recurring memories, preferences, and events.",
            confidence=0.58,
            metadata={"edge_kind": "behavior", "behavior_category": cat},
        )


def infer_query_roles(question: str, rows: Sequence[Dict[str, Any]]) -> List[str]:
    roles = sorted({str(row.get("role") or "").strip() for row in rows if str(row.get("role") or "").strip()})
    q = question.lower()
    matched: List[str] = []
    for role in roles:
        role_l = role.lower()
        if not role_l:
            continue
        if re.search(rf"\b{re.escape(role_l)}\b", q):
            matched.append(role)
    return matched


def role_gated_rows(question: str, rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    roles = infer_query_roles(question, rows)
    if not roles:
        return list(rows), []
    gated = [dict(row) for row in rows if str(row.get("role") or "").strip() in set(roles)]
    return gated or list(rows), roles


def query_snippet(text: str, query: str, max_words: int = 64) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    qterms = set(t for t in base.tokenize(query) if len(t) > 2)
    best_i = 0
    best_score = -1
    window = max_words
    for i in range(0, max(1, len(words) - window + 1), max(1, window // 3)):
        chunk = " ".join(words[i : i + window])
        score = len(qterms & set(base.tokenize(chunk)))
        if score > best_score:
            best_i, best_score = i, score
    prefix = " ".join(words[best_i : best_i + window])
    return prefix


def row_context(rows: Sequence[Dict[str, Any]], meta: Dict[str, Any], window: int) -> str:
    if window <= 0:
        return str(meta.get("raw_content") or "")
    sid = str(meta.get("session_id") or "")
    try:
        turn = int(meta.get("turn_index") or 0)
    except Exception:
        turn = 0
    chunks: List[str] = []
    for row in rows:
        if str(row.get("session_id") or "") != sid:
            continue
        try:
            ti = int(row.get("turn_index") or 0)
        except Exception:
            continue
        if abs(ti - turn) <= window:
            role = str(row.get("role") or "")
            raw = str(row.get("raw_content") or row.get("content") or "")
            chunks.append(f"{role}: {raw}")
    return " ".join(chunks) or str(meta.get("raw_content") or "")


def preserve_numbers_dates(full_text: str, snippet: str) -> str:
    extras: List[str] = []
    patterns = [
        r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b",
        r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|years?|hours?|minutes?)\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\$\s?\d+(?:\.\d+)?\b",
    ]
    for pat in patterns:
        for m in re.findall(pat, full_text, flags=re.I):
            val = m if isinstance(m, str) else " ".join(m)
            if val and val not in snippet and val not in extras:
                extras.append(val)
    if extras:
        return snippet + " [numbers/dates: " + "; ".join(extras[:10]) + "]"
    return snippet


def source_snippet_result(
    ret: ProfileRetrievalResult,
    query: str,
    method_name: str,
    max_words: int = 64,
    rows: Sequence[Dict[str, Any]] | None = None,
    context_window: int = 0,
    keep_numbers: bool = False,
) -> ProfileRetrievalResult:
    facts: List[ProfileFact] = []
    for fact in ret.selected_facts:
        meta = dict(fact.metadata)
        date = str(meta.get("date") or "")
        role = str(meta.get("role") or "")
        raw = row_context(rows or [], meta, context_window) if rows else str(meta.get("raw_content") or fact.content)
        snippet = query_snippet(raw, query, max_words=max_words)
        if keep_numbers:
            snippet = preserve_numbers_dates(raw, snippet)
        content = f"[source={meta.get('row_id', fact.fact_id)} date={date} role={role}] {snippet}"
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
    debug = [dict(x) for x in ret.debug_scores]
    if debug:
        debug[0]["method"] = method_name
        debug[0]["snippet"] = True
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
    facts = base.sorted_evidence_facts(ret)
    lines = []
    for i, fact in enumerate(facts[:16], start=1):
        meta = fact.metadata or {}
        date = str(meta.get("date") or "")
        role = str(meta.get("role") or "")
        text = str(meta.get("display_content") or meta.get("raw_content") or fact.content).replace("\n", " ").strip()
        if meta.get("display_content"):
            lines.append(f"[{i}] {text[:520]}")
        else:
            lines.append(f"[{i}] {date} {role}: {text[:360]}")
    return "\n".join(lines)[:max_chars]


base.evidence_block = evidence_block_display_first
generate_and_judge.__globals__["evidence_block"] = evidence_block_display_first


def action_for_policy(policy: str, example: Any) -> MethodConfig:
    stats = memory_stats(example)
    comp = complexity_bin(example, stats)
    sbin = size_bin(stats)
    if policy == "fixed_compact":
        return MethodConfig("A1-compact", graph_gate="hypermem_full", top_k_facts=12, max_tokens=800, initial_candidates=55, topic_top_k=4, episode_top_k=8, lambda_prop=0.5)
    if policy == "fixed_recall":
        return MethodConfig("A3-recall", graph_gate="hypermem_full", top_k_facts=24, max_tokens=1500, initial_candidates=85, topic_top_k=8, episode_top_k=16, lambda_prop=0.5)
    if comp == "abstention":
        return MethodConfig("D1-dense-compact", graph_gate="qwen_dense", top_k_facts=8, max_tokens=520, initial_candidates=32)
    if comp == "simple" and sbin == "small":
        return MethodConfig("D2-behavior-edge", graph_gate="qwen_hg", top_k_edges=4, top_k_facts=10, max_tokens=680, initial_candidates=40)
    if comp == "multi_count":
        return MethodConfig("D3-full-broad" if sbin == "large" else "D3-full-recall", graph_gate="hypermem_full", top_k_facts=28 if sbin == "large" else 24, max_tokens=1700 if sbin == "large" else 1500, initial_candidates=100 if sbin == "large" else 85, topic_top_k=10 if sbin == "large" else 8, episode_top_k=18 if sbin == "large" else 16, lambda_prop=0.5)
    if comp == "temporal":
        return MethodConfig("D3-full-balanced", graph_gate="hypermem_full", top_k_facts=20, max_tokens=1250, initial_candidates=70, topic_top_k=6, episode_top_k=12, lambda_prop=0.5)
    if comp in {"update", "preference"} and sbin != "large":
        return MethodConfig("D3-full-compact", graph_gate="hypermem_full", top_k_facts=12, max_tokens=800, initial_candidates=55, topic_top_k=4, episode_top_k=8, lambda_prop=0.5)
    return MethodConfig("D3-full-balanced", graph_gate="hypermem_full", top_k_facts=20, max_tokens=1250, initial_candidates=70, topic_top_k=6, episode_top_k=12, lambda_prop=0.5)


def split_examples(examples: Sequence[Any], train_size: int, test_size: int, seed: int) -> List[Any]:
    import random

    rng = random.Random(seed)
    pool = list(examples)
    rng.shuffle(pool)
    train_ids = {ex.qid for ex in pool[:train_size]}
    return [ex for ex in pool if ex.qid not in train_ids][:test_size]


def run_variant(
    name: str,
    example: Any,
    qwen_embed: QwenEmbeddingClient,
    qwen_reranker: QwenRerankerClient | None,
    *,
    better_topics: bool,
    behavior_edges: bool,
    snippet: bool,
    snippet_words: int = 64,
    context_window: int = 0,
    keep_numbers: bool = False,
    role_gate: bool = False,
) -> ProfileRetrievalResult:
    if better_topics:
        enable_better_topics()
    else:
        disable_better_topics()
    rows_for_memory, gated_roles = role_gated_rows(example.question, example.rows) if role_gate else (list(example.rows), [])
    memory = build_memory(rows_for_memory)
    if behavior_edges:
        augment_behavior_edges(memory)
    action = action_for_policy("hybrid", example)
    ret = retrieve_method(example, memory, action, qwen_embed=qwen_embed, qwen_reranker=qwen_reranker)
    ret.channel = name
    if ret.debug_scores:
        ret.debug_scores[0]["method"] = name
        ret.debug_scores[0]["action"] = action.name
        ret.debug_scores[0]["better_topics"] = better_topics
        ret.debug_scores[0]["behavior_edges"] = behavior_edges
        ret.debug_scores[0]["role_gate"] = role_gate
        ret.debug_scores[0]["gated_roles"] = ",".join(gated_roles)
    if snippet:
        ret = source_snippet_result(
            ret,
            example.question,
            name,
            max_words=snippet_words,
            rows=rows_for_memory,
            context_window=context_window,
            keep_numbers=keep_numbers,
        )
    return ret


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=220)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qwen-embedding-url", default="http://localhost:11810/v1/embeddings")
    parser.add_argument("--qwen-reranker-url", default="http://localhost:12810")
    parser.add_argument("--use-qwen-reranker", action="store_true")
    parser.add_argument("--max-llm-judge", type=int, default=0)
    parser.add_argument("--reader-model", default="deepseek-chat")
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--reader-mode", default="temporal")
    parser.add_argument("--variants", default="", help="Comma-separated variant names to run. Empty means all.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(Path(args.data), max_examples=args.max_examples)
    test = split_examples(examples, args.train_size, args.test_size, args.seed)
    qwen_embed = QwenEmbeddingClient(base_url=args.qwen_embedding_url)
    qwen_reranker = QwenRerankerClient(base_url=args.qwen_reranker_url) if args.use_qwen_reranker else None
    cache_path = out_dir / "llm_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    reader = LLMClient(model=args.reader_model) if args.max_llm_judge > 0 else None
    judge = LLMClient(model=args.judge_model) if args.max_llm_judge > 0 else None

    variants = [
        ("HybridDepth", False, False, False, 0, 0, False, False),
        ("HybridDepth+Snippet64", False, False, True, 64, 0, False, False),
        ("HybridDepth+Snippet96", False, False, True, 96, 0, False, False),
        ("HybridDepth+Snippet128", False, False, True, 128, 0, False, False),
        ("HybridDepth+Snippet96Nums", False, False, True, 96, 0, True, False),
        ("HybridDepth+Snippet96Ctx1", False, False, True, 96, 1, False, False),
        ("HybridDepth+Snippet96Ctx1Nums", False, False, True, 96, 1, True, False),
        ("HybridDepth+BetterTopics", True, False, False, 0, 0, False, False),
        ("HybridDepth+BehaviorEdges", False, True, False, 0, 0, False, False),
        ("HybridDepth+RoleGate", False, False, False, 0, 0, False, True),
        ("HybridDepth+RoleGate+Snippet96", False, False, True, 96, 0, False, True),
        ("HybridDepth+BetterTopics+BehaviorEdges", True, True, False, 0, 0, False, False),
        ("HybridDepth+BTBE+Snippet96Ctx1Nums", True, True, True, 96, 1, True, False),
    ]
    wanted = {x.strip() for x in args.variants.split(",") if x.strip()}
    if wanted:
        variants = [item for item in variants if item[0] in wanted]
        if not variants:
            raise SystemExit(f"No matching variants for --variants={args.variants!r}")
    rows: List[Dict[str, Any]] = []
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as trace:
        for qi, ex in enumerate(test, start=1):
            stats = memory_stats(ex)
            for name, better, behavior, snippet, words, ctx, nums, role_gate in variants:
                ret = run_variant(
                    name,
                    ex,
                    qwen_embed,
                    qwen_reranker,
                    better_topics=better,
                    behavior_edges=behavior,
                    snippet=snippet,
                    snippet_words=words or 64,
                    context_window=ctx,
                    keep_numbers=nums,
                    role_gate=role_gate,
                )
                metrics = retrieval_metrics(ex, ret)
                row = {
                    "method": name,
                    "qid": ex.qid,
                    "qtype": ex.qtype,
                    "complexity_bin": complexity_bin(ex, stats),
                    "size_bin": size_bin(stats),
                    "question": ex.question,
                    **metrics,
                    "retrieval_tokens": ret.tokens,
                    "retrieval_ms": ret.debug_scores[0].get("latency_ms", 0.0) if ret.debug_scores else 0.0,
                    "num_facts": len(ret.selected_facts),
                    "action": ret.debug_scores[0].get("action", "") if ret.debug_scores else "",
                    "role_gate": ret.debug_scores[0].get("role_gate", False) if ret.debug_scores else False,
                    "gated_roles": ret.debug_scores[0].get("gated_roles", "") if ret.debug_scores else "",
                }
                if reader is not None and judge is not None and qi <= args.max_llm_judge:
                    key = f"{ex.qid}::{name}::reader={reader.model}::judge={judge.model}::mode={args.reader_mode}::refine_v2"
                    judged = generate_and_judge(reader, judge, ex, ret, cache, key, args.reader_mode)
                    row.update({k: v for k, v in judged.items() if k != "judge_raw"})
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
                rows.append(row)
                trace.write(json.dumps({**row, "evidence": [f.content for f in ret.selected_facts]}, ensure_ascii=False) + "\n")
            print(f"[done] {qi}/{len(test)}", flush=True)

    write_csv(out_dir / "refinement_results.csv", rows)
    write_csv(out_dir / "refinement_summary.csv", summarize(rows))
    compare: List[Dict[str, Any]] = []
    for method in sorted({r["method"] for r in rows}):
        part = [r for r in rows if r["method"] == method]
        compare.append(
            {
                "method": method,
                "n": len(part),
                "llm_acc": (
                    sum(float(r["judge_score"]) for r in part if r.get("judge_score") not in (None, ""))
                    / max(1, sum(1 for r in part if r.get("judge_score") not in (None, "")))
                    if any(r.get("judge_score") not in (None, "") for r in part)
                    else ""
                ),
                "llm_n": sum(1 for r in part if r.get("judge_score") not in (None, "")),
                "fact_hit": sum(float(r["fact_hit"]) for r in part) / len(part),
                "answer_recall": sum(float(r["answer_turn_recall"]) for r in part) / len(part),
                "all_hit": sum(float(r["all_answer_turns_hit"]) for r in part) / len(part),
                "avg_tokens": sum(float(r["retrieval_tokens"]) for r in part) / len(part),
                "avg_ms": sum(float(r["retrieval_ms"]) for r in part) / len(part),
                "p50_ms": statistics.median([float(r["retrieval_ms"]) for r in part]),
            }
        )
    write_csv(out_dir / "refinement_compare.csv", compare)
    print((out_dir / "refinement_compare.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
