"""Probe hyperedge-conditioned dialogue evidence retrieval.

This probe treats hyperedge summaries as conditions, not final evidence. The
final evidence is drawn from source-backed facts and local dialogue/memory rows
around those facts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import profile_centric_hypergraph_eval as base  # noqa: E402
from hypermem.cost_aware_retrieval import retrieve_budget_aware  # noqa: E402
from hypermem.profile_centric_hypergraph import (  # noqa: E402
    HashedEmbeddingModel,
    ProfileCentricHypergraphMemory,
    ProfileFact,
    ProfileRetrievalResult,
    estimate_tokens,
    keyword_overlap,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower().strip())


def clip_words(text: str, max_words: int) -> str:
    words = str(text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def content_keys(text: str) -> List[str]:
    base_text = normalize_text(text)
    keys = [base_text]
    for prefix in ("persona statement:", "dialogue context:", "expected response:", "hyperedge condition:"):
        if base_text.startswith(prefix):
            keys.append(normalize_text(base_text[len(prefix):]))
    return [k for k in dict.fromkeys(keys) if k]


def find_dialogue_id_in_obj(obj: Any, depth: int = 0) -> str:
    if depth > 4:
        return ""
    if isinstance(obj, dict):
        for key in ("dialogue_id", "dialog_id", "conversation_id"):
            if obj.get(key):
                return str(obj[key])
        for val in obj.values():
            found = find_dialogue_id_in_obj(val, depth + 1)
            if found:
                return found
    elif isinstance(obj, list):
        for val in obj:
            found = find_dialogue_id_in_obj(val, depth + 1)
            if found:
                return found
    return ""


def row_sort_key(row: Dict[str, Any]) -> Tuple[int, int, str]:
    return (int(row.get("timestamp") or 0), int(row.get("turn_idx") or 0), str(row.get("source_type") or ""))


def build_dialogue_index(memory_rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_dialogue: Dict[str, List[Dict[str, Any]]] = {}
    for row in memory_rows:
        did = str(row.get("dialogue_id") or "")
        if did:
            by_dialogue.setdefault(did, []).append(row)
    for did in list(by_dialogue):
        by_dialogue[did] = sorted(by_dialogue[did], key=row_sort_key)
    return by_dialogue


def build_content_maps(memory_rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    did_lookup: Dict[str, str] = {}
    row_lookup: Dict[str, Dict[str, Any]] = {}
    for row in memory_rows:
        did = str(row.get("dialogue_id") or "")
        content = str(row.get("content") or "")
        if not did or not content:
            continue
        for key in content_keys(content):
            did_lookup.setdefault(key, did)
            row_lookup.setdefault(key, row)
    return did_lookup, row_lookup


def fact_dialogue_id(fact: ProfileFact, content_lookup: Dict[str, str] | None = None) -> str:
    found = find_dialogue_id_in_obj(fact.metadata or {})
    if found:
        return found
    if content_lookup:
        for key in content_keys(fact.content):
            if key in content_lookup:
                return content_lookup[key]
    return ""


def row_for_fact(fact: ProfileFact, row_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any] | None:
    for key in content_keys(fact.content):
        if key in row_lookup:
            return row_lookup[key]
    return None


def local_dialogue_rows(
    dialogue_index: Dict[str, List[Dict[str, Any]]],
    row_lookup: Dict[str, Dict[str, Any]],
    selected_facts: Sequence[ProfileFact],
    fallback_dialogue_ids: Sequence[str],
    *,
    window: int = 1,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen = set()
    anchor_rows = [row_for_fact(f, row_lookup) for f in selected_facts]
    anchor_rows = [r for r in anchor_rows if r]
    for anchor in anchor_rows:
        did = str(anchor.get("dialogue_id") or "")
        rows = dialogue_index.get(did, [])
        try:
            idx = next(i for i, r in enumerate(rows) if r is anchor or normalize_text(r.get("content")) == normalize_text(anchor.get("content")))
        except StopIteration:
            idx = 0
        lo, hi = max(0, idx - window), min(len(rows), idx + window + 1)
        for row in rows[lo:hi]:
            key = (row.get("dialogue_id"), row.get("timestamp"), row.get("content"))
            if key not in seen:
                seen.add(key); selected.append(row)
    if not selected:
        for did in fallback_dialogue_ids:
            for row in dialogue_index.get(did, [])[:4]:
                key = (row.get("dialogue_id"), row.get("timestamp"), row.get("content"))
                if key not in seen:
                    seen.add(key); selected.append(row)
    return selected


def pack_text_rows(rows: Iterable[Dict[str, Any]], max_tokens: int, max_rows: int) -> List[ProfileFact]:
    out: List[ProfileFact] = []
    tokens = 0
    seen = set()
    for row in rows:
        content = str(row.get("content") or "").strip()
        if not content or content in seen:
            continue
        seen.add(content)
        cost = estimate_tokens(content)
        if out and tokens + cost > max_tokens:
            continue
        out.append(ProfileFact(
            fact_id=f"dialogue_row_{len(out)+1:04d}",
            content=content,
            keywords=[],
            timestamp=float(row.get("timestamp") or 0),
            embedding=[],
            metadata={"source_type": "source_dialogue_row", **row},
        ))
        tokens += cost
        if len(out) >= max_rows or tokens >= max_tokens:
            break
    return out


def top_hyperedges(memory: ProfileCentricHypergraphMemory, query: str, top_k: int) -> List[Tuple[Any, float]]:
    qtype, _, _ = memory.infer_profile_type(query)
    qemb = memory.embedding_model.encode(query)
    scored = []
    for edge in memory.edges.values():
        if edge.status != "active":
            continue
        score, _ = edge.score(qemb, qtype, memory.facts, use_utility=False, weights=memory.weights)
        if score > 0:
            scored.append((edge, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def rank_facts(memory: ProfileCentricHypergraphMemory, query: str, fact_ids: Iterable[str], max_tokens: int, top_k: int) -> List[ProfileFact]:
    qemb = memory.embedding_model.encode(query)
    scored: Dict[str, float] = {}
    for fid in fact_ids:
        fact = memory.facts.get(fid)
        if fact is None:
            continue
        sim = HashedEmbeddingModel.cosine(qemb, fact.embedding)
        lex = keyword_overlap(query, fact.content)
        scored[fid] = max(scored.get(fid, 0.0), 0.72 * sim + 0.28 * lex)
    selected: List[ProfileFact] = []
    tokens = 0
    for fid, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True):
        fact = memory.facts.get(fid)
        if fact is None:
            continue
        cost = estimate_tokens(fact.content)
        if selected and tokens + cost > max_tokens:
            continue
        selected.append(fact)
        tokens += cost
        if len(selected) >= top_k or tokens >= max_tokens:
            break
    return selected


def conditioned_dialogue_retrieve(
    memory: ProfileCentricHypergraphMemory,
    dialogue_index: Dict[str, List[Dict[str, Any]]],
    content_lookup: Dict[str, str],
    row_lookup: Dict[str, Dict[str, Any]],
    query: str,
    *,
    top_k_edges: int,
    top_k_facts: int,
    max_tokens: int,
    include_condition: bool,
    include_facts: bool,
) -> ProfileRetrievalResult:
    edges = top_hyperedges(memory, query, top_k_edges)
    member_fact_ids: List[str] = []
    for edge, _ in edges:
        member_fact_ids.extend(edge.member_fact_ids)
    selected_facts = rank_facts(memory, query, member_fact_ids, max_tokens=max_tokens, top_k=top_k_facts) if include_facts else []

    dialogue_ids = []
    for fact in selected_facts:
        did = fact_dialogue_id(fact, content_lookup)
        if did:
            dialogue_ids.append(did)
    if not dialogue_ids:
        for fid in member_fact_ids[:30]:
            fact = memory.facts.get(fid)
            did = fact_dialogue_id(fact, content_lookup) if fact else ""
            if did:
                dialogue_ids.append(did)
    dialogue_ids = list(dict.fromkeys(dialogue_ids))[:top_k_edges]

    condition_facts: List[ProfileFact] = []
    if include_condition:
        for idx, (edge, score) in enumerate(edges, 1):
            condition_facts.append(ProfileFact(
                fact_id=f"condition_{idx:04d}",
                content=f"Hyperedge condition: {clip_words(edge.summary, 24)}",
                keywords=edge.keywords,
                timestamp=0.0,
                embedding=edge.embedding,
                metadata={"source_type": "hyperedge_condition", "edge_id": edge.edge_id, "score": score},
            ))

    used = estimate_tokens([f.content for f in condition_facts + selected_facts])
    remaining = max(1, max_tokens - used)
    rows = local_dialogue_rows(dialogue_index, row_lookup, selected_facts, dialogue_ids, window=1)
    dialogue_facts = pack_text_rows(rows, max_tokens=remaining, max_rows=6)

    final = condition_facts + selected_facts + dialogue_facts
    return ProfileRetrievalResult(
        query=query,
        channel="conditioned_dialogue_evidence",
        selected_edges=[edge for edge, _ in edges],
        selected_facts=final,
        score=sum(score for _, score in edges) / max(1, len(edges)),
        tokens=estimate_tokens([f.content for f in final]),
        fallback_used=False,
        sufficient=bool(final),
        debug_scores=[{
            "path": "conditioned_dialogue",
            "selected_edges": len(edges),
            "selected_facts": len(selected_facts),
            "selected_dialogues": len(dialogue_ids),
            "dialogue_rows": len(dialogue_facts),
            "condition_tokens": estimate_tokens([f.content for f in condition_facts]),
            "fact_tokens": estimate_tokens([f.content for f in selected_facts]),
            "dialogue_tokens": estimate_tokens([f.content for f in dialogue_facts]),
            "token_budget": max_tokens,
        }],
    )


def retrieve_method(method: str, memory, dialogue_index, content_lookup, row_lookup, question: str, args) -> ProfileRetrievalResult:
    if method == "adaptive_tiny":
        return retrieve_budget_aware(memory, question, top_k_edges=2, top_k_facts=4, max_tokens=110, use_utility=False, top_k_topics=2, top_k_episodes=3, budget_ratio=1.0)
    if method == "condition_dialogue":
        return conditioned_dialogue_retrieve(memory, dialogue_index, content_lookup, row_lookup, question, top_k_edges=2, top_k_facts=4, max_tokens=args.max_tokens, include_condition=True, include_facts=False)
    if method == "condition_fact_dialogue":
        return conditioned_dialogue_retrieve(memory, dialogue_index, content_lookup, row_lookup, question, top_k_edges=2, top_k_facts=4, max_tokens=args.max_tokens, include_condition=True, include_facts=True)
    if method == "fact_dialogue":
        return conditioned_dialogue_retrieve(memory, dialogue_index, content_lookup, row_lookup, question, top_k_edges=2, top_k_facts=4, max_tokens=args.max_tokens, include_condition=False, include_facts=True)
    raise ValueError(method)


def avg(rows, key):
    return sum(float(r.get(key, 0.0)) for r in rows) / max(1, len(rows))


def summarize(rows):
    out = base.summarize(rows)
    out.update({
        "retrieval_ms": round(avg(rows, "retrieval_ms"), 3),
        "num_facts": round(avg(rows, "num_facts"), 3),
        "selected_edges": round(avg(rows, "selected_edges"), 3),
        "selected_dialogues": round(avg(rows, "selected_dialogues"), 3),
        "dialogue_rows": round(avg(rows, "dialogue_rows"), 3),
        "condition_tokens": round(avg(rows, "condition_tokens"), 3),
        "fact_tokens": round(avg(rows, "fact_tokens"), 3),
        "dialogue_tokens": round(avg(rows, "dialogue_tokens"), 3),
    })
    return out


def row_for_fields(method: str, summary: Dict[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    row = {"method": method, **summary}
    return {field: row.get(field, "") for field in fields}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--memory-graph", required=True)
    p.add_argument("--memory-json", required=True)
    p.add_argument("--questions-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-questions", type=int, default=1000)
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--methods", default="adaptive_tiny,condition_dialogue,fact_dialogue,condition_fact_dialogue")
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    memory = ProfileCentricHypergraphMemory.load(args.memory_graph)
    memory_rows = read_jsonl(Path(args.memory_json))
    dialogue_index = build_dialogue_index(memory_rows)
    content_lookup, row_lookup = build_content_maps(memory_rows)
    questions = base.normalize_questions(base.read_json_or_jsonl(Path(args.questions_json)))[: args.max_questions]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows: List[Dict[str, Any]] = []
    trace = (outdir / "conditioned_dialogue_trace.jsonl").open("w", encoding="utf-8")
    for method in methods:
        start = time.time()
        for q in questions:
            t0 = time.time()
            result = retrieve_method(method, memory, dialogue_index, content_lookup, row_lookup, q["question"], args)
            row, _, _, _ = base.row_from_result(method, q, result, update_used=False)
            dbg = result.debug_scores[0] if result.debug_scores else {}
            row.update({
                "retrieval_ms": round((time.time() - t0) * 1000, 3),
                "selected_edges": dbg.get("selected_edges", len(result.selected_edges)),
                "selected_dialogues": dbg.get("selected_dialogues", 0),
                "dialogue_rows": dbg.get("dialogue_rows", 0),
                "condition_tokens": dbg.get("condition_tokens", 0),
                "fact_tokens": dbg.get("fact_tokens", 0),
                "dialogue_tokens": dbg.get("dialogue_tokens", 0),
            })
            rows.append(row)
            trace.write(json.dumps({**row, "gold": q["gold"], "evidence": [f.content for f in result.selected_facts], "debug": result.debug_scores}, ensure_ascii=False) + "\n")
        print(f"[done] {method} avg={(time.time()-start)/max(1,len(questions)):.4f}s/q", flush=True)
    trace.close()

    by: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by.setdefault(row["method"], []).append(row)
    fields = [
        "method", "n", "accuracy", "hit", "recall", "tokens", "reward", "fallback_rate",
        "retrieval_ms", "num_facts", "selected_edges", "selected_dialogues", "dialogue_rows",
        "condition_tokens", "fact_tokens", "dialogue_tokens",
    ]
    summaries = {m: summarize(r) for m, r in by.items()}
    with (outdir / "conditioned_dialogue_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method, summary in summaries.items():
            writer.writerow(row_for_fields(method, summary, fields))
    (outdir / "conditioned_dialogue_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print((outdir / "conditioned_dialogue_summary.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
