"""Global fact retrieval baseline for UP-HyperPool experiments.

This is a lightweight HyperMem-style/global fact retrieval proxy:
for each query, it ranks all fact memories by keyword overlap and returns the
highest-scoring facts under the same evidence token budget used by the profile
pool evaluator.

It consumes the same JSON/JSONL inputs as profile_hyperedge_pool_eval.py and
writes compatible summary/results files:
    global_fact_retrieval_summary.json
    global_fact_retrieval_results.csv
    global_fact_retrieval_trace.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypermem.profile_hyperedge_pool import keyword_overlap, tokenize
from examples.profile_hyperedge_pool_eval import (
    read_json_or_jsonl,
    normalize_memory,
    normalize_questions,
    evidence_hit,
    evidence_recall,
    reward,
)


def estimate_tokens(text: str) -> int:
    # Rough but consistent with the profile-pool demo evaluator's lightweight spirit.
    return max(1, len(tokenize(text)))


def retrieve_global_facts(query: str, memory_rows: Sequence[Dict[str, Any]], max_tokens: int, top_k: int) -> Dict[str, Any]:
    scored = []
    for row in memory_rows:
        content = row.get("content", "")
        keywords = row.get("keywords") or []
        score = keyword_overlap(query, " ".join([content, " ".join(map(str, keywords))]))
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)

    evidence = []
    total_tokens = 0
    for score, row in scored[: max(top_k * 5, top_k)]:
        content = row.get("content", "")
        tok = estimate_tokens(content)
        if evidence and total_tokens + tok > max_tokens:
            continue
        evidence.append({"score": round(score, 6), "content": content, "fact_id": row.get("fact_id", "")})
        total_tokens += tok
        if len(evidence) >= top_k or total_tokens >= max_tokens:
            break

    return {
        "channel": "global_fact_retrieval",
        "evidence": evidence,
        "tokens": total_tokens,
        "evidence_text": "\n".join(item["content"] for item in evidence),
    }


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "hit": 0.0, "recall": 0.0, "tokens": 0.0, "reward": 0.0, "fallback_rate": 0.0}
    return {
        "n": n,
        "hit": round(sum(float(r["hit"]) for r in rows) / n, 6),
        "recall": round(sum(float(r["recall"]) for r in rows) / n, 6),
        "tokens": round(sum(float(r["tokens"]) for r in rows) / n, 3),
        "reward": round(sum(float(r["reward"]) for r in rows) / n, 6),
        "fallback_rate": 0.0,
    }


def run_eval(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    memory_rows = normalize_memory(read_json_or_jsonl(Path(args.memory_json)))
    questions = normalize_questions(read_json_or_jsonl(Path(args.questions_json)))

    rows: List[Dict[str, Any]] = []
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    trace_path = out_dir / "global_fact_retrieval_trace.jsonl"

    with trace_path.open("w", encoding="utf-8") as f:
        for q in questions:
            result = retrieve_global_facts(q["question"], memory_rows, args.max_tokens, args.top_k)
            hit = evidence_hit(result["evidence_text"], q["gold"])
            rec = evidence_recall(result["evidence_text"], q["gold"])
            r = reward(hit, rec, result["tokens"], fallback_used=False)
            row = {
                "qid": q["qid"],
                "category": q["category"],
                "question": q["question"],
                "method": "global_fact_retrieval",
                "channel": result["channel"],
                "hit": hit,
                "recall": round(rec, 6),
                "tokens": result["tokens"],
                "reward": round(r, 6),
                "num_evidence": len(result["evidence"]),
            }
            rows.append(row)
            by_cat.setdefault(q["category"], []).append(row)
            f.write(json.dumps({**row, "gold": q["gold"], "evidence": result["evidence"]}, ensure_ascii=False) + "\n")

    results_path = out_dir / "global_fact_retrieval_results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = summarize(rows)
    summary.update({"method": "global_fact_retrieval", "num_memory_rows": len(memory_rows)})
    summary_path = out_dir / "global_fact_retrieval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    by_cat_rows = []
    for cat, items in by_cat.items():
        by_cat_rows.append({"category": cat, **summarize(items)})
    by_cat_path = out_dir / "global_fact_retrieval_by_category.csv"
    with by_cat_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_cat_rows[0].keys()) if by_cat_rows else [])
        if by_cat_rows:
            writer.writeheader()
            writer.writerows(by_cat_rows)

    print("Global Fact Retrieval Eval")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote:", results_path)
    print("wrote:", by_cat_path)
    print("wrote:", trace_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-json", required=True)
    parser.add_argument("--questions-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
