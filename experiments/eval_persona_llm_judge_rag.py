from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402

from experiments.eval_persona_advanced_retrieval import (  # noqa: E402
    MethodConfig,
    load_questions,
    read_jsonl,
    retrieve_advanced,
    retrieve_reference,
    row_id_map,
)
from experiments.eval_persona_report_compare import (  # noqa: E402
    BM25Index,
    classic_retrieve,
    graph_retrieve,
    make_source_facts,
)
from experiments.eval_persona_rl_retrieval import (  # noqa: E402
    Arm,
    ThompsonPolicy,
    make_fast_pruned_arms,
    retrieve_arm,
)
from examples import profile_centric_hypergraph_eval as base_eval  # noqa: E402
from hypermem import load_runtime_env  # noqa: E402
from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory, ProfileRetrievalResult, estimate_tokens  # noqa: E402
from hypermem.query_router import route_query  # noqa: E402


def safe_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except Exception:
            return {}
    return {}


class LLMClient:
    def __init__(self, model: str, base_url: str, api_key: str, temperature: float = 0.0) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def chat(self, messages: Sequence[Dict[str, str]], max_tokens: int = 512, json_mode: bool = False) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


def make_llm_client() -> LLMClient:
    load_runtime_env()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")
    return LLMClient(model=model, base_url=base_url, api_key=api_key, temperature=0.0)


def evidence_block(result: ProfileRetrievalResult, max_chars: int = 1800) -> str:
    lines = []
    for i, fact in enumerate(result.selected_facts, start=1):
        lines.append(f"[{i}] {fact.content}")
    text = "\n".join(lines)
    return text[:max_chars]


def generate_answer(llm: LLMClient, question: str, result: ProfileRetrievalResult, cache: Dict[str, Any], key: str) -> str:
    if key in cache:
        return str(cache[key]["answer"])
    prompt = (
        "You are answering a PersonaChat memory question using only the retrieved evidence.\n"
        "If the question asks for the next/expected response, answer with the most likely response text, without the 'Expected response:' prefix.\n"
        "If the question asks for persona/profile information, answer with the stated persona fact.\n"
        "Be concise. Do not mention evidence ids.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{evidence_block(result)}\n\n"
        "Answer:"
    )
    answer = llm.chat([{"role": "user", "content": prompt}], max_tokens=160, json_mode=False).strip()
    cache[key] = {"answer": answer}
    return answer


def judge_answer(llm: LLMClient, question: str, gold: Sequence[str], answer: str, cache: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key in cache:
        return dict(cache[key])
    prompt = (
        "You are a strict but fair evaluator for PersonaChat QA.\n"
        "Judge whether the predicted answer is semantically equivalent to ANY gold answer.\n"
        "For next-response questions, accept paraphrases that preserve the same intent and content, but reject unrelated plausible replies.\n"
        "Return strict JSON only: {\"score\":0 or 1,\"reason\":\"short\"}.\n\n"
        f"Question: {question}\n"
        f"Gold answers: {json.dumps(list(gold), ensure_ascii=False)}\n"
        f"Predicted answer: {answer}\n"
    )
    raw = llm.chat([{"role": "user", "content": prompt}], max_tokens=160, json_mode=True)
    data = safe_json(raw)
    score = int(1 if str(data.get("score", "0")).strip() in {"1", "true", "True"} else 0)
    out = {"score": score, "reason": str(data.get("reason") or "")[:300], "raw": raw}
    cache[key] = out
    return out


def train_thompson_policy(memory, source_rows, questions: Sequence[Dict[str, Any]], train_n: int) -> Tuple[ThompsonPolicy, List[Arm]]:
    arms = make_fast_pruned_arms()
    policy = ThompsonPolicy(len(arms), seed=11)
    for idx, q in enumerate(questions[:train_n], start=1):
        route = route_query(q["question"])
        arm_idx = policy.select(q["question"], route.route, idx, train=True)
        result = retrieve_arm(memory, source_rows, q["question"], arms[arm_idx])
        _, reward, hit, _ = base_eval.row_from_result("train", q, result, update_used=True)
        policy.update(arm_idx, q["question"], route.route, reward, hit)
    return policy, arms


def retrieve_method(
    method: str,
    query: str,
    local_memory,
    llm_memory,
    local_sources,
    llm_sources,
    source_facts,
    bm25,
    policy_pack,
    step: int,
) -> ProfileRetrievalResult:
    if method == "BM25-RAG":
        return classic_retrieve("bm25_source", source_facts, bm25, query, top_k=4, max_tokens=140)
    if method == "RAPTOR-tree":
        return graph_retrieve("topic_episode", llm_memory, llm_sources, query)
    if method == "LightRAG-dual":
        cfg = MethodConfig("LightRAG-dual", response_boost=0.12, persona_boost=0.14, graph_gate="hybrid", top_k_facts=3, max_tokens=110)
        return retrieve_advanced(llm_memory, llm_sources, query, cfg)
    if method == "GRAG-subgraph":
        cfg = MethodConfig("GRAG-subgraph", response_boost=0.30, persona_boost=0.12, graph_gate="edge", top_k_edges=3, top_k_facts=4, max_tokens=140)
        return retrieve_advanced(llm_memory, llm_sources, query, cfg)
    if method == "HG-RL-Thompson":
        policy, arms = policy_pack
        route = route_query(query)
        arm_idx = policy.select(query, route.route, step, train=False)
        return retrieve_arm(llm_memory, llm_sources, query, arms[arm_idx])
    raise ValueError(method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-json", required=True)
    parser.add_argument("--questions-json", required=True)
    parser.add_argument("--local-graph", required=True)
    parser.add_argument("--llm-graph", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=500, help="Start from test split by default.")
    parser.add_argument("--methods", default="BM25-RAG,RAPTOR-tree,LightRAG-dual,GRAG-subgraph,HG-RL-Thompson")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "llm_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    answer_cache = cache.setdefault("answers", {})
    judge_cache = cache.setdefault("judges", {})

    llm = make_llm_client()
    local_memory = ProfileCentricHypergraphMemory.load(args.local_graph)
    llm_memory = ProfileCentricHypergraphMemory.load(args.llm_graph)
    local_sources = row_id_map(read_jsonl(Path(args.memory_json)), local_memory)
    llm_sources = row_id_map(read_jsonl(Path(args.memory_json)), llm_memory)
    source_facts = list({f.fact_id: f for f in local_sources.values()}.values())
    source_facts.sort(key=lambda f: f.timestamp)
    bm25 = BM25Index(source_facts)
    all_questions = load_questions(Path(args.questions_json), max_questions=0)
    eval_questions = all_questions[args.start_index : args.start_index + args.max_questions]
    policy_pack = train_thompson_policy(llm_memory, llm_sources, all_questions, args.train_size)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    rows: List[Dict[str, Any]] = []
    trace_path = out_dir / "llm_judge_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as trace:
        for qi, q in enumerate(eval_questions, start=1):
            for method in methods:
                t0 = time.time()
                result = retrieve_method(
                    method,
                    q["question"],
                    local_memory,
                    llm_memory,
                    local_sources,
                    llm_sources,
                    source_facts,
                    bm25,
                    policy_pack,
                    qi,
                )
                retrieval_ms = (time.time() - t0) * 1000.0
                base_key = f"{q['qid']}::{method}"
                answer = generate_answer(llm, q["question"], result, answer_cache, base_key)
                judged = judge_answer(llm, q["question"], q["gold"], answer, judge_cache, base_key)
                row = {
                    "method": method,
                    "qid": q["qid"],
                    "category": q["category"],
                    "judge_score": judged["score"],
                    "retrieval_tokens": result.tokens,
                    "num_facts": len(result.selected_facts),
                    "retrieval_ms": round(retrieval_ms, 4),
                    "answer": answer,
                    "judge_reason": judged["reason"],
                }
                rows.append(row)
                trace.write(json.dumps({
                    **row,
                    "question": q["question"],
                    "gold": q["gold"],
                    "evidence": [f.content for f in result.selected_facts],
                }, ensure_ascii=False) + "\n")
                trace.flush()
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[done] question {qi}/{len(eval_questions)}", flush=True)

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    summary = []
    for method, part in by_method.items():
        n = len(part)
        summary.append({
            "method": method,
            "n": n,
            "llm_judge_accuracy": round(sum(r["judge_score"] for r in part) / max(1, n), 6),
            "retrieval_tokens": round(sum(float(r["retrieval_tokens"]) for r in part) / max(1, n), 3),
            "num_facts": round(sum(float(r["num_facts"]) for r in part) / max(1, n), 3),
            "retrieval_ms": round(sum(float(r["retrieval_ms"]) for r in part) / max(1, n), 3),
        })
    fields = ["method", "n", "llm_judge_accuracy", "retrieval_tokens", "num_facts", "retrieval_ms"]
    with (out_dir / "llm_judge_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(summary, key=lambda x: x["llm_judge_accuracy"], reverse=True))
    with (out_dir / "llm_judge_results.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print((out_dir / "llm_judge_summary.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
