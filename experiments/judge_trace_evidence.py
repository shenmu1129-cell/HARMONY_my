from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_longmemeval_mini import LLMClient, safe_json  # noqa: E402


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--max-qids", type=int, default=12)
    parser.add_argument("--reader-model", default="deepseek-chat")
    parser.add_argument("--judge-model", default="deepseek-chat")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    rows = [json.loads(line) for line in Path(args.trace).read_text(encoding="utf-8").splitlines() if line.strip()]
    qids: List[str] = []
    for row in rows:
        if row["qid"] not in qids:
            qids.append(row["qid"])
    qids = qids[: args.max_qids]

    cache_path = out_dir / "llm_judge_from_trace_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    reader = LLMClient(model=args.reader_model)
    judge = LLMClient(model=args.judge_model)
    out: List[Dict[str, Any]] = []

    for qid in qids:
        for row in [r for r in rows if r["qid"] == qid and r["method"] in methods]:
            key = f"{qid}::{row['method']}::tracejudge_v1"
            if key in cache:
                judged = dict(cache[key])
            else:
                evidence = "\n".join(f"[{i + 1}] {text[:420]}" for i, text in enumerate(row.get("evidence", [])[:16]))[:4200]
                answer_prompt = (
                    "Answer the long-term memory question using only the retrieved evidence. "
                    "Use date and relative-time reasoning when needed. Output one short phrase. "
                    "If the evidence is genuinely insufficient, say I don't know.\n\n"
                    f"Question: {row['question']}\n\n"
                    f"Retrieved evidence:\n{evidence}\n\n"
                    "Short answer:"
                )
                pred = reader.chat(answer_prompt, max_tokens=160).strip()
                judge_prompt = (
                    "You are a strict but fair evaluator for long-term memory QA. "
                    "Return JSON only: {\"score\":0 or 1,\"reason\":\"short\"}.\n\n"
                    f"Question: {row['question']}\n"
                    f"Gold answer: {row['gold']}\n"
                    f"Predicted answer: {pred}\n"
                )
                raw = judge.chat(judge_prompt, max_tokens=160, json_mode=True)
                data = safe_json(raw)
                score = int(1 if str(data.get("score", "0")).strip().lower() in {"1", "true"} else 0)
                judged = {"answer": pred, "judge_score": score, "judge_reason": str(data.get("reason") or "")[:300]}
                cache[key] = judged
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            out.append(
                {
                    "method": row["method"],
                    "qid": row["qid"],
                    "qtype": row["qtype"],
                    "question": row["question"],
                    "gold": row["gold"],
                    "fact_hit": row["fact_hit"],
                    "answer_turn_recall": row["answer_turn_recall"],
                    "all_answer_turns_hit": row["all_answer_turns_hit"],
                    "retrieval_tokens": row["retrieval_tokens"],
                    "retrieval_ms": row["retrieval_ms"],
                    **judged,
                }
            )
        print(f"[done] {qid}", flush=True)

    write_csv(out_dir / "llm_judge_from_trace_results.csv", out)
    summary: List[Dict[str, Any]] = []
    for method in sorted(methods):
        part = [row for row in out if row["method"] == method]
        if not part:
            continue
        summary.append(
            {
                "method": method,
                "n": len(part),
                "llm_acc": sum(float(row["judge_score"]) for row in part) / len(part),
                "answer_recall": sum(float(row["answer_turn_recall"]) for row in part) / len(part),
                "retrieval_tokens": sum(float(row["retrieval_tokens"]) for row in part) / len(part),
            }
        )
    write_csv(out_dir / "llm_judge_from_trace_summary.csv", summary)
    print((out_dir / "llm_judge_from_trace_summary.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
