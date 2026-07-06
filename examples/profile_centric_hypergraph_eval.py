"""End-to-end profile-centric hypergraph memory evaluation.

Pipeline:
    1. construct user-profile hyperedges from memory facts;
    2. build local hashed embeddings for facts, hyperedges and queries;
    3. train hyperedge utility with lightweight reward updates on train QA;
    4. rank profile hyperedges and then rank facts inside selected hyperedges;
    5. report retrieval accuracy/recall/token/reward.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory, ProfileRetrievalResult, keyword_overlap


DEMO_MEMORY = [
    "用户正在研究 LLM memory，重点关注 HyperMem、A-MEM、MeMo、MemEvolve 和 LoCoMo。",
    "用户希望把 memory 方向做成 AAAI 级别创新，而不是简单工程拼接。",
    "用户喜欢审稿人视角：先判断创新性，再判断实验是否能支撑，最后给 Codex prompt。",
    "用户不喜欢空泛鼓励，更希望指出当前方案风险、弱点和可救方向。",
    "用户经常在服务器上运行实验，使用 conda 环境、bash 脚本和 GitHub main 分支。",
    "用户当前主线是用用户画像超边替代 HyperMem 的 Topic-Episode-Fact 主检索路径。",
    "用户认为 embedding 仍然需要，但它负责向量召回和相似度匹配，profile utility 负责个性化价值排序。",
    "用户倾向使用轻量 bandit-style reward update，而不是训练 PPO 或大模型。",
]

DEMO_QUESTIONS = [
    {"qid": "q1", "question": "我现在 memory 方案的核心主线是什么？", "gold": ["用户画像超边", "Topic-Episode-Fact"], "category": "method"},
    {"qid": "q2", "question": "我希望你怎么评价论文创新？", "gold": ["审稿人视角", "创新性", "实验"], "category": "preference"},
    {"qid": "q3", "question": "embedding 在我的方法里还有用吗？", "gold": ["向量", "profile utility"], "category": "method"},
    {"qid": "q4", "question": "我经常让你帮我做哪些工程操作？", "gold": ["服务器", "conda", "GitHub"], "category": "habit"},
    {"qid": "q5", "question": "强化学习版本最好先做哪种？", "gold": ["bandit", "reward update"], "category": "rl"},
]


def read_json_or_jsonl(path: Path) -> List[Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def normalize_memory(rows: Sequence[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if isinstance(row, str):
            out.append({"fact_id": f"fact_{i+1:06d}", "content": row, "timestamp": i + 1})
            continue
        content = row.get("content") or row.get("text") or row.get("fact") or row.get("summary") or ""
        if not content:
            continue
        out.append({
            "fact_id": row.get("fact_id") or row.get("id") or f"fact_{i+1:06d}",
            "content": str(content),
            "keywords": row.get("keywords") or [],
            "timestamp": row.get("timestamp") or row.get("time_index") or i + 1,
            "metadata": row,
        })
    return out


def normalize_questions(rows: Sequence[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if isinstance(row, str):
            out.append({"qid": f"q_{i+1:06d}", "question": row, "gold": [], "category": "unknown"})
            continue
        q = row.get("question") or row.get("query") or row.get("q") or ""
        if not q:
            continue
        gold = row.get("gold") or row.get("evidence") or row.get("gold_evidence") or row.get("answer") or []
        if isinstance(gold, str):
            gold = [gold]
        out.append({
            "qid": row.get("qid") or row.get("id") or f"q_{i+1:06d}",
            "question": str(q),
            "gold": [str(x) for x in gold],
            "category": row.get("category") or row.get("type") or "unknown",
        })
    return out


def evidence_hit(evidence_text: str, gold: Sequence[str]) -> int:
    if not gold:
        return 0
    low = evidence_text.lower()
    for item in gold:
        g = str(item).strip().lower()
        if g and (g in low or keyword_overlap(g, low) >= 0.08):
            return 1
    return 0


def evidence_recall(evidence_text: str, gold: Sequence[str]) -> float:
    if not gold:
        return 0.0
    low = evidence_text.lower()
    matched = 0
    for item in gold:
        g = str(item).strip().lower()
        if g and (g in low or keyword_overlap(g, low) >= 0.08):
            matched += 1
    return matched / max(1, len(gold))


def reward_from_result(hit: int, recall: float, tokens: int, fallback_used: bool) -> float:
    return max(-1.0, min(1.0, recall + 0.20 * hit - 0.12 * tokens / 1000.0 - (0.04 if fallback_used else 0.0)))


def build_memory(memory_rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> ProfileCentricHypergraphMemory:
    memory = ProfileCentricHypergraphMemory(
        user_id="profile_eval_user",
        attach_threshold=args.attach_threshold,
        discovery_threshold=args.discovery_threshold,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
    )
    memory.build_from_rows(memory_rows)
    return memory


def row_from_result(method: str, q: Dict[str, Any], result: ProfileRetrievalResult, update_used: bool) -> Tuple[Dict[str, Any], float, int, float]:
    ev_text = result.evidence_text()
    hit = evidence_hit(ev_text, q["gold"])
    rec = evidence_recall(ev_text, q["gold"])
    reward = reward_from_result(hit, rec, result.tokens, result.fallback_used)
    row = {
        "method": method,
        "qid": q["qid"],
        "category": q["category"],
        "question": q["question"],
        "hit": hit,
        "accuracy": hit,
        "recall": round(rec, 6),
        "tokens": result.tokens,
        "reward": round(reward, 6),
        "fallback_used": int(result.fallback_used),
        "update_used": int(update_used),
        "channel": result.channel,
        "num_edges": len(result.selected_edges),
        "num_facts": len(result.selected_facts),
        "edge_ids": ";".join(edge.edge_id for edge in result.selected_edges),
        "edge_types": ";".join(edge.edge_type.value for edge in result.selected_edges),
    }
    return row, reward, hit, rec


def run_questions(memory: ProfileCentricHypergraphMemory, questions: Sequence[Dict[str, Any]], args: argparse.Namespace, method: str, use_utility: bool, update: bool, trace_file) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for q in questions:
        result = memory.retrieve(
            q["question"],
            top_k_edges=args.top_k_edges,
            top_k_facts=args.top_k_facts,
            max_tokens=args.max_tokens,
            use_utility=use_utility,
            fallback=not args.no_fallback,
            sufficiency_threshold=args.sufficiency_threshold,
        )
        row, reward, hit, _ = row_from_result(method, q, result, update_used=update)
        rows.append(row)
        if update:
            memory.update_from_feedback(result, reward=reward, hit=bool(hit))
        trace_file.write(json.dumps({
            **row,
            "gold": q["gold"],
            "edge_debug": result.debug_scores,
            "evidence": [fact.content for fact in result.selected_facts],
        }, ensure_ascii=False) + "\n")
    return rows


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "hit": 0.0, "recall": 0.0, "tokens": 0.0, "reward": 0.0, "fallback_rate": 0.0}
    return {
        "n": n,
        "accuracy": round(sum(float(row["accuracy"]) for row in rows) / n, 6),
        "hit": round(sum(float(row["hit"]) for row in rows) / n, 6),
        "recall": round(sum(float(row["recall"]) for row in rows) / n, 6),
        "tokens": round(sum(float(row["tokens"]) for row in rows) / n, 3),
        "reward": round(sum(float(row["reward"]) for row in rows) / n, 6),
        "fallback_rate": round(sum(float(row["fallback_used"]) for row in rows) / n, 6),
    }


def split_questions(questions: Sequence[Dict[str, Any]], train_ratio: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not questions:
        return [], []
    if len(questions) == 1:
        return [], list(questions)
    n_train = int(len(questions) * train_ratio)
    n_train = max(1, min(len(questions) - 1, n_train))
    return list(questions[:n_train]), list(questions[n_train:])


def load_inputs(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if args.memory_json:
        memory_rows = normalize_memory(read_json_or_jsonl(Path(args.memory_json)))
    else:
        memory_rows = normalize_memory(DEMO_MEMORY)
    if args.questions_json:
        questions = normalize_questions(read_json_or_jsonl(Path(args.questions_json)))
    else:
        questions = normalize_questions(DEMO_QUESTIONS)
    if args.max_memory and len(memory_rows) > args.max_memory:
        memory_rows = memory_rows[: args.max_memory]
    if args.max_questions and len(questions) > args.max_questions:
        questions = questions[: args.max_questions]
    return memory_rows, questions


def run_eval(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    memory_rows, questions = load_inputs(args)
    train_q, test_q = split_questions(questions, args.train_ratio)

    all_rows: List[Dict[str, Any]] = []
    trace_path = out_dir / "profile_centric_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as trace:
        embedding_memory = build_memory(memory_rows, args)
        embedding_rows = run_questions(embedding_memory, test_q, args, "embedding_only_profile_hg", use_utility=False, update=False, trace_file=trace)
        all_rows.extend(embedding_rows)

        utility_memory = build_memory(memory_rows, args)
        train_rows = run_questions(utility_memory, train_q, args, "reward_utility_train", use_utility=True, update=True, trace_file=trace)
        frozen_rows = run_questions(utility_memory, test_q, args, "reward_utility_frozen_test", use_utility=True, update=False, trace_file=trace)
        all_rows.extend(train_rows)
        all_rows.extend(frozen_rows)

        if args.online_eval:
            online_memory = build_memory(memory_rows, args)
            _ = run_questions(online_memory, train_q, args, "online_warmup_train", use_utility=True, update=True, trace_file=trace)
            online_rows = run_questions(online_memory, test_q, args, "online_predict_then_update_test", use_utility=True, update=True, trace_file=trace)
            all_rows.extend(online_rows)
            online_memory.save(out_dir / "profile_centric_online_memory.json")

    utility_memory.save(out_dir / "profile_centric_trained_memory.json")
    embedding_memory.save(out_dir / "profile_centric_embedding_only_memory.json")

    results_path = out_dir / "profile_centric_results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as f:
        if all_rows:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        by_method.setdefault(row["method"], []).append(row)

    summary = {
        "pipeline": [
            "construct_user_profile_hyperedges",
            "build_local_hashed_embeddings",
            "train_reward_utility_on_train_qa",
            "rank_profile_hyperedges_and_facts",
            "evaluate_accuracy_recall_tokens_reward",
        ],
        "num_memory_rows": len(memory_rows),
        "num_questions": len(questions),
        "num_train_questions": len(train_q),
        "num_test_questions": len(test_q),
        "train_ratio": args.train_ratio,
        "embedding_dim": args.embedding_dim,
        "methods": {method: summarize(rows) for method, rows in by_method.items()},
        "trained_profile": utility_memory.export(),
    }
    summary_path = out_dir / "profile_centric_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_csv_path = out_dir / "profile_centric_summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["method", "n", "accuracy", "hit", "recall", "tokens", "reward", "fallback_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, rows in by_method.items():
            writer.writerow({"method": method, **summarize(rows)})

    print("Profile-Centric Hypergraph Memory Eval")
    print(json.dumps({k: v for k, v in summary.items() if k != "trained_profile"}, ensure_ascii=False, indent=2))
    print("wrote:", results_path)
    print("wrote:", summary_path)
    print("wrote:", summary_csv_path)
    print("wrote:", trace_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-json", type=str, default="")
    parser.add_argument("--questions-json", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/profile_centric_hypergraph")
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--online-eval", action="store_true", help="Also run prequential online test: score each test query before updating.")
    parser.add_argument("--top-k-edges", type=int, default=3)
    parser.add_argument("--top-k-facts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--sufficiency-threshold", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=0.18)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--attach-threshold", type=float, default=0.52)
    parser.add_argument("--discovery-threshold", type=float, default=0.55)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--max-memory", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
