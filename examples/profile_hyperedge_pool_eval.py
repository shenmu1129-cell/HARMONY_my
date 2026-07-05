"""Retrieval-only evaluation for the user-profile hyperedge pool.

The script can run a built-in demo or consume JSON/JSONL files.
It does not call OpenAI, DeepSeek, stage 5/6, or any LLM judge.

Built-in demo:
    python examples/profile_hyperedge_pool_eval.py --output-dir outputs/profile_eval

Custom data:
    python examples/profile_hyperedge_pool_eval.py \
      --memory-json data/memory_facts.jsonl \
      --questions-json data/questions.jsonl \
      --output-dir outputs/profile_eval

Supported memory rows:
    {"content": "...", "fact_id": "optional", "keywords": [...], "timestamp": 1}

Supported question rows:
    {"question": "...", "gold": ["evidence substring"], "category": "optional"}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypermem.profile_hyperedge_pool import MemoryNode, NodeType, UserProfileHyperedgePool, keyword_overlap


DEMO_MEMORY = [
    "用户正在研究 LLM memory，核心对象包括 HyperMem、A-MEM、MeMo、MemEvolve 和 LoCoMo。",
    "用户目标是把 memory 方向做成 AAAI 级别的创新，而不是简单工程拼接。",
    "用户喜欢审稿人视角：先判断创新性，再判断实验是否能支撑，最后给 Codex prompt。",
    "用户不希望被空泛鼓励，更希望指出当前方案的风险、弱点和可救方向。",
    "fixed_400 是强低成本 baseline，global_fact_only_800 命中率高但 token 成本大。",
    "override_logreg 能接近 global_fact_only_800 的 hit，同时显著减少 token，因此说明 query-dependent routing 有价值。",
    "adaptive_controller_v1 已经跑通多步闭环，但规则版 verifier/action policy 引入噪声，reward 不如 override_logreg。",
    "当前新方向是用户画像引导的动态超边池，在 Topic/Episode/Fact 底座之上维护个性化快速通道。",
    "用户画像超边池存 preference、goal、habit、domain knowledge、current state、temporal evolution 等高价值超边。",
    "当 profile fast channel 证据不足时，系统应 fallback 到原始 HyperMem path 或 global fact retrieval。",
    "奖励更新用于维护 profile hyperedge utility：命中和帮助回答则升权，错误、过期或无贡献则降权。",
    "时间处理可以成为创新点，因为长期记忆需要区分过去想法、最新状态和想法演化链。",
]

DEMO_QUESTIONS = [
    {"question": "我现在这个 memory 方案的核心是什么？", "gold": ["用户画像", "动态超边池"], "category": "current_state"},
    {"question": "我通常希望你怎么分析论文？", "gold": ["审稿人视角", "Codex prompt"], "category": "preference"},
    {"question": "为什么 adaptive_controller_v1 还不行？", "gold": ["规则版", "引入噪声"], "category": "experiment"},
    {"question": "如果快速通道证据不足怎么办？", "gold": ["fallback", "原始 HyperMem"], "category": "method"},
    {"question": "强化学习奖励在超边池里干什么？", "gold": ["utility", "升权", "降权"], "category": "rl"},
]


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
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
    out = []
    for i, row in enumerate(rows):
        if isinstance(row, str):
            out.append({"content": row, "fact_id": f"fact_{i+1:05d}", "timestamp": i + 1})
            continue
        content = row.get("content") or row.get("text") or row.get("fact") or row.get("summary") or ""
        if content:
            out.append(
                {
                    "content": str(content),
                    "fact_id": row.get("fact_id") or row.get("id") or f"fact_{i+1:05d}",
                    "keywords": row.get("keywords") or [],
                    "timestamp": row.get("timestamp") or row.get("time_index") or i + 1,
                    "topic_id": row.get("topic_id", ""),
                    "episode_ids": row.get("episode_ids") or ([row["episode_id"]] if row.get("episode_id") else []),
                    "metadata": row,
                }
            )
    return out


def normalize_questions(rows: Sequence[Any]) -> List[Dict[str, Any]]:
    out = []
    for i, row in enumerate(rows):
        if isinstance(row, str):
            out.append({"question": row, "gold": [], "category": "unknown", "qid": f"q_{i+1:05d}"})
            continue
        q = row.get("question") or row.get("query") or row.get("q") or ""
        if not q:
            continue
        gold = row.get("gold") or row.get("evidence") or row.get("gold_evidence") or row.get("answer") or []
        if isinstance(gold, str):
            gold = [gold]
        out.append(
            {
                "qid": row.get("qid") or row.get("id") or f"q_{i+1:05d}",
                "question": str(q),
                "gold": [str(x) for x in gold],
                "category": row.get("category") or row.get("type") or "unknown",
            }
        )
    return out


def evidence_hit(evidence_text: str, gold: Sequence[str]) -> int:
    if not gold:
        return 0
    low = evidence_text.lower()
    for g in gold:
        g = str(g).strip().lower()
        if g and (g in low or keyword_overlap(g, low) >= 0.08):
            return 1
    return 0


def evidence_recall(evidence_text: str, gold: Sequence[str]) -> float:
    if not gold:
        return 0.0
    low = evidence_text.lower()
    matched = 0
    for g in gold:
        g = str(g).strip().lower()
        if g and (g in low or keyword_overlap(g, low) >= 0.08):
            matched += 1
    return matched / max(1, len(gold))


def reward(hit: int, recall: float, tokens: int, fallback_used: bool) -> float:
    return recall + 0.2 * hit - 0.1 * tokens / 1000.0 - (0.03 if fallback_used else 0.0)


def build_pool(memory_rows: Sequence[Dict[str, Any]]) -> UserProfileHyperedgePool:
    pool = UserProfileHyperedgePool(user_id="eval_user")
    for row in normalize_memory(memory_rows):
        pool.add_fact(
            content=row["content"],
            node_id=row["fact_id"],
            keywords=row.get("keywords") or None,
            timestamp=float(row.get("timestamp") or 0),
            topic_id=row.get("topic_id", ""),
            episode_ids=row.get("episode_ids") or [],
            metadata=row.get("metadata") or {},
            promote=True,
        )
    return pool


def fallback_nodes(pool: UserProfileHyperedgePool) -> List[MemoryNode]:
    return list(pool.nodes.values())


def run_eval(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.memory_json:
        memory_rows = read_json_or_jsonl(Path(args.memory_json))
    else:
        memory_rows = DEMO_MEMORY
    if args.questions_json:
        question_rows = read_json_or_jsonl(Path(args.questions_json))
    else:
        question_rows = DEMO_QUESTIONS

    pool = build_pool(memory_rows)
    questions = normalize_questions(question_rows)
    fb_nodes = fallback_nodes(pool)

    trace_path = out_dir / "profile_hyperedge_pool_trace.jsonl"
    rows: List[Dict[str, Any]] = []
    by_cat: Dict[str, List[Dict[str, Any]]] = {}

    with trace_path.open("w", encoding="utf-8") as f:
        for q in questions:
            result = pool.retrieve_fast_channel(
                q["question"],
                top_k_edges=args.top_k_edges,
                max_tokens=args.max_tokens,
                sufficiency_threshold=args.sufficiency_threshold,
                fallback_nodes=fb_nodes if args.enable_fallback else None,
            )
            ev_text = result.evidence_text()
            hit = evidence_hit(ev_text, q["gold"])
            rec = evidence_recall(ev_text, q["gold"])
            r = reward(hit, rec, result.tokens, result.fallback_used)
            edge_ids = [e.edge_id for e in result.hyperedges]
            pool.update_rewards(edge_ids, hit=bool(hit), answer_quality=rec if rec > 0 else 0.2)

            row = {
                "qid": q["qid"],
                "category": q["category"],
                "question": q["question"],
                "channel": result.channel,
                "hit": hit,
                "recall": round(rec, 6),
                "tokens": result.tokens,
                "reward": round(r, 6),
                "fallback_used": int(result.fallback_used),
                "num_edges": len(result.hyperedges),
                "edge_ids": ";".join(edge_ids),
                "edge_types": ";".join(e.edge_type.value for e in result.hyperedges),
            }
            rows.append(row)
            by_cat.setdefault(q["category"], []).append(row)
            f.write(
                json.dumps(
                    {
                        **row,
                        "gold": q["gold"],
                        "hyperedges": [
                            {
                                "edge_id": e.edge_id,
                                "type": e.edge_type.value,
                                "summary": e.summary,
                                "utility_score": e.utility_score,
                                "profile_score": e.profile_score,
                            }
                            for e in result.hyperedges
                        ],
                        "evidence": [n.content for n in result.evidence_nodes],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    results_path = out_dir / "profile_hyperedge_pool_results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = summarize(rows)
    summary_path = out_dir / "profile_hyperedge_pool_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    by_cat_path = out_dir / "profile_hyperedge_pool_by_category.csv"
    cat_rows = []
    for cat, cat_items in by_cat.items():
        s = summarize(cat_items)
        cat_rows.append({"category": cat, **s})
    with by_cat_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cat_rows[0].keys()) if cat_rows else [])
        if cat_rows:
            writer.writeheader()
            writer.writerows(cat_rows)

    pool.save(out_dir / "profile_hyperedge_pool.json")

    print("Profile Hyperedge Pool Eval")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote:", results_path)
    print("wrote:", by_cat_path)
    print("wrote:", trace_path)


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
        "fallback_rate": round(sum(float(r["fallback_used"]) for r in rows) / n, 6),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-json", type=str, default="", help="JSON/JSONL memory facts. If omitted, runs built-in demo.")
    parser.add_argument("--questions-json", type=str, default="", help="JSON/JSONL questions. If omitted, runs built-in demo.")
    parser.add_argument("--output-dir", type=str, default="outputs/profile_hyperedge_pool")
    parser.add_argument("--top-k-edges", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--sufficiency-threshold", type=float, default=0.18)
    parser.add_argument("--enable-fallback", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
