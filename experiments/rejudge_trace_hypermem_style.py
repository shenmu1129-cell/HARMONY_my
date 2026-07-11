from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from openai import OpenAI, OpenAIError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from hypermem.prompts.answer_prompts import ANSWER_PROMPT_NEMORI_COT  # noqa: E402


def load_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError("OPENAI_API_KEY is missing")


class OpenAIChat:
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.client = OpenAI(api_key=load_openai_key(), base_url=base_url)

    def chat(self, prompt: str, max_tokens: int = 512, json_mode: bool = False) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except OpenAIError as exc:
                last_exc = exc
                wait_s = min(60, 5 * (2**attempt))
                print(f"[warn] {self.model} failed attempt {attempt + 1}/5: {exc}; retrying in {wait_s}s", flush=True)
                time.sleep(wait_s)
        raise RuntimeError(f"{self.model} failed after retries: {last_exc}") from last_exc


def parse_jsonish(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def category(row: Dict[str, Any]) -> int | None:
    qtype = str(row.get("qtype") or "")
    match = re.search(r"category_(\d+)", qtype)
    return int(match.group(1)) if match else None


def evidence_context(row: Dict[str, Any], max_items: int, max_chars: int) -> str:
    lines = []
    for i, item in enumerate(row.get("evidence", [])[:max_items], start=1):
        lines.append(f"Memory {i}: {item}")
    return "\n".join(lines)[:max_chars]


def generate_hypermem_answer(reader: OpenAIChat, row: Dict[str, Any], max_evidence: int, max_context_chars: int) -> str:
    context = evidence_context(row, max_evidence, max_context_chars)
    prompt = ANSWER_PROMPT_NEMORI_COT.format(context=context, question=row["question"])
    raw = reader.chat(prompt, max_tokens=4096)
    if "FINAL ANSWER:" in raw:
        return raw.split("FINAL ANSWER:", 1)[1].strip()
    return raw.strip()


def judge_hypermem_style(judge: OpenAIChat, row: Dict[str, Any], answer: str) -> Dict[str, Any]:
    prompt = f"""
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {row["question"]}
Gold answer: {row["gold"]}
Generated answer: {answer}

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""
    raw = judge.chat(prompt, max_tokens=160, json_mode=True)
    data = parse_jsonish(raw)
    label = str(data.get("label") or "").strip().upper()
    if not label:
        upper = raw.upper()
        if "CORRECT" in upper and "WRONG" not in upper:
            label = "CORRECT"
        elif "WRONG" in upper and "CORRECT" not in upper:
            label = "WRONG"
    return {"judge_score": 1 if label == "CORRECT" else 0, "judge_label": label or "UNPARSED", "judge_raw": raw}


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = list(rows)
    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("method") or ""), str(row["variant"])), []).append(row)
    out = []
    for (method, variant), part in sorted(groups.items()):
        n = len(part)
        out.append(
            {
                "method": method,
                "variant": variant,
                "n": n,
                "llm_acc": round(sum(float(r["judge_score"]) for r in part) / n, 6) if n else "",
                "avg_tokens": round(sum(float(r.get("retrieval_tokens", 0)) for r in part) / n, 1) if n else "",
                "avg_ms": round(sum(float(r.get("retrieval_ms", 0)) for r in part) / n, 1) if n else "",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reader-model", default="gpt-4.1-mini")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--max-items", type=int, default=0, help="0 means all rows")
    parser.add_argument("--skip-category-5", action="store_true")
    parser.add_argument("--max-evidence", type=int, default=18)
    parser.add_argument("--max-context-chars", type=int, default=9000)
    parser.add_argument("--variants", default="rejudge_existing,regenerate_4_1_mini")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    rows = [json.loads(line) for line in Path(args.trace).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.skip_category_5:
        rows = [row for row in rows if category(row) != 5]
    if args.max_items > 0:
        rows = rows[: args.max_items]

    variants = {v.strip() for v in args.variants.split(",") if v.strip()}
    reader = OpenAIChat(args.reader_model, args.base_url) if "regenerate_4_1_mini" in variants else None
    judge = OpenAIChat(args.judge_model, args.base_url)
    out: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        for variant in sorted(variants):
            key = f"{row['qid']}::{variant}::reader={args.reader_model}::judge={args.judge_model}::hmstyle_v1"
            if key in cache:
                result = dict(cache[key])
            else:
                if variant == "rejudge_existing":
                    answer = str(row.get("answer") or "")
                elif variant == "regenerate_4_1_mini":
                    if reader is None:
                        raise RuntimeError("reader was not initialized")
                    answer = generate_hypermem_answer(reader, row, args.max_evidence, args.max_context_chars)
                else:
                    raise ValueError(f"unknown variant: {variant}")
                judged = judge_hypermem_style(judge, row, answer)
                result = {"answer": answer, **judged}
                cache[key] = result
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            out.append(
                {
                    "method": row.get("method"),
                    "variant": variant,
                    "qid": row["qid"],
                    "qtype": row.get("qtype"),
                    "question": row.get("question"),
                    "gold": row.get("gold"),
                    "retrieval_tokens": row.get("retrieval_tokens"),
                    "retrieval_ms": row.get("retrieval_ms"),
                    **result,
                }
            )
        print(f"[done] {idx}/{len(rows)} {row['qid']}", flush=True)

    jsonl_path = out_dir / "hypermem_style_results.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in out) + "\n", encoding="utf-8")
    write_csv(out_dir / "hypermem_style_results.csv", out)
    summary = summarize(out)
    write_csv(out_dir / "hypermem_style_summary.csv", summary)
    print((out_dir / "hypermem_style_summary.csv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
