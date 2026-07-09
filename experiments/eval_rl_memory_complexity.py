from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from eval_longmemeval_mini import (
    LLMClient,
    MethodConfig,
    ProfileRetrievalResult,
    QwenEmbeddingClient,
    QwenRerankerClient,
    ThompsonRouter,
    build_memory,
    estimate_tokens,
    generate_and_judge,
    load_examples,
    retrieve_method,
    retrieval_metrics,
    summarize,
    write_csv,
)


def memory_stats(example: Any) -> Dict[str, Any]:
    session_ids = {str(row.get("session_id") or "") for row in example.rows}
    row_tokens = estimate_tokens([str(row.get("text") or row.get("content") or "") for row in example.rows])
    answer_turns = sum(1 for row in example.rows if row.get("has_answer"))
    dates = [str(row.get("date") or row.get("timestamp") or "") for row in example.rows]
    nonempty_dates = [d for d in dates if d]
    return {
        "rows": len(example.rows),
        "sessions": len(session_ids),
        "tokens": row_tokens,
        "answer_turns": answer_turns,
        "has_dates": int(bool(nonempty_dates)),
    }


def size_bin(stats: Dict[str, Any]) -> str:
    rows = int(stats["rows"])
    sessions = int(stats["sessions"])
    toks = int(stats["tokens"])
    if rows <= 80 and sessions <= 4 and toks <= 3500:
        return "small"
    if rows <= 180 and sessions <= 8 and toks <= 9000:
        return "medium"
    return "large"


def complexity_bin(example: Any, stats: Dict[str, Any]) -> str:
    q = example.question.lower()
    qtype = str(example.qtype or "")
    if example.qid.endswith("_abs"):
        return "abstention"
    if "multi" in qtype or re.search(r"\b(how many|total|average|sum|different|more than|less than)\b", q):
        return "multi_count"
    if "temporal" in qtype or re.search(r"\b(before|after|first|last|date|days?|weeks?|months?|when|earlier|later)\b", q):
        return "temporal"
    if "knowledge" in qtype or re.search(r"\b(currently|now|changed|move|working at|recent)\b", q):
        return "update"
    if int(stats["sessions"]) <= 2:
        return "simple"
    return "preference"


def action_space() -> List[MethodConfig]:
    return [
        MethodConfig("D1-dense-compact", graph_gate="qwen_dense", top_k_facts=8, max_tokens=520, initial_candidates=32),
        MethodConfig("D2-edge-compact", graph_gate="qwen_hg", top_k_edges=3, top_k_facts=10, max_tokens=700, initial_candidates=40),
        MethodConfig("D3-full-compact", graph_gate="hypermem_full", top_k_facts=12, max_tokens=800, initial_candidates=55, topic_top_k=4, episode_top_k=8, lambda_prop=0.5),
        MethodConfig("D3-full-balanced", graph_gate="hypermem_full", top_k_facts=20, max_tokens=1250, initial_candidates=70, topic_top_k=6, episode_top_k=12, lambda_prop=0.5),
        MethodConfig("D3-full-recall", graph_gate="hypermem_full", top_k_facts=24, max_tokens=1500, initial_candidates=85, topic_top_k=8, episode_top_k=16, lambda_prop=0.5),
        MethodConfig("D3-full-broad", graph_gate="hypermem_full", top_k_facts=28, max_tokens=1700, initial_candidates=100, topic_top_k=10, episode_top_k=18, lambda_prop=0.5),
    ]


def current_actions() -> List[MethodConfig]:
    return [
        MethodConfig("A1-compact", graph_gate="hypermem_full", top_k_facts=12, max_tokens=800, initial_candidates=55, topic_top_k=4, episode_top_k=8, lambda_prop=0.5),
        MethodConfig("A2-balanced", graph_gate="hypermem_full", top_k_facts=20, max_tokens=1250, initial_candidates=70, topic_top_k=6, episode_top_k=12, lambda_prop=0.5),
        MethodConfig("A3-recall", graph_gate="hypermem_full", top_k_facts=24, max_tokens=1500, initial_candidates=85, topic_top_k=8, episode_top_k=16, lambda_prop=0.5),
        MethodConfig("A4-broad", graph_gate="hypermem_full", top_k_facts=28, max_tokens=1700, initial_candidates=100, topic_top_k=10, episode_top_k=18, lambda_prop=0.5),
    ]


class ContextBandit:
    def __init__(self, arms: Sequence[MethodConfig], mode: str, seed: int = 7, priors: bool = False) -> None:
        self.arms = list(arms)
        self.mode = mode
        self.rng = random.Random(seed)
        self.alpha: Dict[str, List[float]] = {}
        self.beta: Dict[str, List[float]] = {}
        self.priors = priors

    def bucket(self, example: Any) -> str:
        stats = memory_stats(example)
        sbin = size_bin(stats)
        cbin = complexity_bin(example, stats)
        qtype = str(example.qtype or "unknown")
        if self.mode == "size":
            return sbin
        if self.mode == "complexity":
            return cbin
        if self.mode == "size_complexity":
            return f"{sbin}:{cbin}"
        if self.mode == "qtype_size_complexity":
            return f"{qtype}:{sbin}:{cbin}"
        return qtype

    def _ensure(self, bucket: str) -> None:
        if bucket in self.alpha:
            return
        self.alpha[bucket] = [1.0] * len(self.arms)
        self.beta[bucket] = [1.0] * len(self.arms)
        if not self.priors:
            return
        for i, arm in enumerate(self.arms):
            name = arm.name
            if "small" in bucket and ("dense" in name or "edge" in name or "compact" in name):
                self.alpha[bucket][i] += 0.8
            if "multi_count" in bucket and ("recall" in name or "broad" in name):
                self.alpha[bucket][i] += 1.2
            if "temporal" in bucket and ("balanced" in name or "recall" in name):
                self.alpha[bucket][i] += 0.9
            if "large" in bucket and ("full-balanced" in name or "full-recall" in name):
                self.alpha[bucket][i] += 0.8

    def select(self, example: Any, train: bool) -> int:
        bucket = self.bucket(example)
        self._ensure(bucket)
        if train:
            vals = [self.rng.betavariate(a, b) for a, b in zip(self.alpha[bucket], self.beta[bucket])]
        else:
            vals = [a / max(1e-9, a + b) for a, b in zip(self.alpha[bucket], self.beta[bucket])]
        return max(range(len(self.arms)), key=lambda i: (vals[i], -self.arms[i].max_tokens))

    def update(self, example: Any, arm_idx: int, reward: float) -> None:
        bucket = self.bucket(example)
        self._ensure(bucket)
        reward = max(0.0, min(1.0, reward))
        self.alpha[bucket][arm_idx] += reward
        self.beta[bucket][arm_idx] += 1.0 - reward

    def dump(self) -> Dict[str, Any]:
        return {
            bucket: {
                self.arms[i].name: round(self.alpha[bucket][i] / max(1e-9, self.alpha[bucket][i] + self.beta[bucket][i]), 4)
                for i in range(len(self.arms))
            }
            for bucket in sorted(self.alpha)
        }


def reward(example: Any, metrics: Dict[str, Any], ret: ProfileRetrievalResult) -> float:
    stats = memory_stats(example)
    comp = complexity_bin(example, stats)
    latency = float(ret.debug_scores[0].get("latency_ms", 0.0)) if ret.debug_scores else 0.0
    if comp == "multi_count":
        r = 0.20 * metrics["fact_hit"] + 0.45 * metrics["answer_turn_recall"] + 0.35 * metrics["all_answer_turns_hit"]
        token_scale = 2600.0
    elif comp == "temporal":
        r = 0.25 * metrics["fact_hit"] + 0.45 * metrics["answer_turn_recall"] + 0.30 * metrics["all_answer_turns_hit"]
        token_scale = 2200.0
    elif comp == "abstention":
        r = 0.50 * metrics["session_hit"] + 0.25 * (1.0 - min(1.0, metrics["fact_hit"])) + 0.25 * (1.0 - min(1.0, ret.tokens / 900.0))
        token_scale = 1200.0
    else:
        r = 0.45 * metrics["fact_hit"] + 0.35 * metrics["answer_turn_recall"] + 0.20 * metrics["all_answer_turns_hit"]
        token_scale = 1600.0
    r -= min(0.18, ret.tokens / token_scale * 0.10)
    r -= min(0.08, latency / 4000.0 * 0.05)
    return r


def heuristic_action(example: Any, arms: Sequence[MethodConfig]) -> MethodConfig:
    stats = memory_stats(example)
    sbin = size_bin(stats)
    comp = complexity_bin(example, stats)
    by_name = {a.name: a for a in arms}
    if comp == "multi_count":
        return by_name["D3-full-recall"] if sbin != "large" else by_name["D3-full-broad"]
    if comp == "temporal":
        return by_name["D3-full-balanced"]
    if comp in {"simple", "preference", "update"} and sbin == "small":
        return by_name["D2-edge-compact"]
    if comp == "abstention":
        return by_name["D1-dense-compact"]
    if sbin == "large":
        return by_name["D3-full-balanced"]
    return by_name["D3-full-compact"]


def clone_name(ret: ProfileRetrievalResult, name: str) -> ProfileRetrievalResult:
    ret.channel = name
    if ret.debug_scores:
        ret.debug_scores[0]["method"] = name
    return ret


def train_router(router: Any, train: Sequence[Any], qwen_embed: QwenEmbeddingClient, qwen_reranker: QwenRerankerClient | None) -> None:
    for ex in train:
        memory = build_memory(ex.rows)
        arm_idx = router.select(ex, train=True)
        ret = retrieve_method(ex, memory, router.arms[arm_idx], qwen_embed=qwen_embed, qwen_reranker=qwen_reranker)
        m = retrieval_metrics(ex, ret)
        router.update(ex, arm_idx, reward(ex, m, ret))


def split_examples(examples: Sequence[Any], train_size: int, test_size: int, seed: int) -> Tuple[List[Any], List[Any]]:
    rng = random.Random(seed)
    pool = list(examples)
    rng.shuffle(pool)
    train = pool[:train_size]
    train_ids = {ex.qid for ex in train}
    test = [ex for ex in pool if ex.qid not in train_ids][:test_size]
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=220)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-llm-judge", type=int, default=0)
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
    train, test = split_examples(examples, args.train_size, args.test_size, args.seed)
    qwen_embed = QwenEmbeddingClient(base_url=args.qwen_embedding_url)
    qwen_reranker = QwenRerankerClient(base_url=args.qwen_reranker_url) if args.use_qwen_reranker else None

    arms = action_space()
    current = ThompsonRouter(current_actions(), seed=23)
    # Expose arms for common training helper.
    current.arms = current_actions()  # type: ignore[attr-defined]
    routers: Dict[str, Any] = {
        "Current-QTypeBandit-4A": current,
        "RL-MemorySize": ContextBandit(arms, "size", seed=31),
        "RL-QueryComplexity": ContextBandit(arms, "complexity", seed=37),
        "RL-SizeComplexity": ContextBandit(arms, "size_complexity", seed=41),
        "RL-SizeComplexityPrior": ContextBandit(arms, "size_complexity", seed=43, priors=True),
    }

    for name, router in routers.items():
        print(f"[train] {name}", flush=True)
        train_router(router, train, qwen_embed, qwen_reranker)

    cache_path = out_dir / "llm_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    reader = LLMClient(model=args.reader_model) if args.max_llm_judge > 0 else None
    judge = LLMClient(model=args.judge_model) if args.max_llm_judge > 0 else None

    rows: List[Dict[str, Any]] = []
    action_counts: Dict[str, Dict[str, int]] = {name: {} for name in [*routers.keys(), "Heuristic-ComplexityDepth", "Fixed-FullCompact", "Fixed-FullRecall"]}
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as trace:
        for qi, ex in enumerate(test, start=1):
            memory = build_memory(ex.rows)
            stats = memory_stats(ex)
            runs: List[Tuple[str, MethodConfig]] = []
            for name, router in routers.items():
                idx = router.select(ex, train=False)
                action = router.arms[idx]
                runs.append((name, action))
            runs.append(("Heuristic-ComplexityDepth", heuristic_action(ex, arms)))
            runs.append(("Fixed-FullCompact", current_actions()[0]))
            runs.append(("Fixed-FullRecall", current_actions()[2]))

            for method_name, action in runs:
                ret = retrieve_method(ex, memory, action, qwen_embed=qwen_embed, qwen_reranker=qwen_reranker)
                ret = clone_name(ret, method_name)
                m = retrieval_metrics(ex, ret)
                action_counts.setdefault(method_name, {})
                action_counts[method_name][action.name] = action_counts[method_name].get(action.name, 0) + 1
                row = {
                    "method": method_name,
                    "action": action.name,
                    "qid": ex.qid,
                    "qtype": ex.qtype,
                    "complexity_bin": complexity_bin(ex, stats),
                    "size_bin": size_bin(stats),
                    "rows": stats["rows"],
                    "sessions": stats["sessions"],
                    "memory_tokens": stats["tokens"],
                    "question": ex.question,
                    "gold": ex.answer,
                    **m,
                    "retrieval_tokens": ret.tokens,
                    "retrieval_ms": ret.debug_scores[0].get("latency_ms", 0.0) if ret.debug_scores else 0.0,
                    "num_facts": len(ret.selected_facts),
                }
                if reader is not None and judge is not None and qi <= args.max_llm_judge:
                    key = f"{ex.qid}::{method_name}::{action.name}::reader={reader.model}::judge={judge.model}::mode={args.reader_mode}::rlcomplex_v1"
                    judged = generate_and_judge(reader, judge, ex, ret, cache, key, args.reader_mode)
                    row.update({k: v for k, v in judged.items() if k != "judge_raw"})
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
                rows.append(row)
                trace.write(json.dumps({**row, "evidence": [f.content for f in ret.selected_facts]}, ensure_ascii=False) + "\n")
            print(f"[done] {qi}/{len(test)}", flush=True)

    write_csv(out_dir / "rl_complexity_results.csv", rows)
    write_csv(out_dir / "rl_complexity_summary.csv", summarize(rows))

    compare: List[Dict[str, Any]] = []
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    for method, part in sorted(by_method.items()):
        judged = [float(r["judge_score"]) for r in part if "judge_score" in r]
        compare.append(
            {
                "method": method,
                "n": len(part),
                "llm_acc": sum(judged) / len(judged) if judged else "",
                "llm_n": len(judged),
                "fact_hit": sum(float(r["fact_hit"]) for r in part) / len(part),
                "answer_recall": sum(float(r["answer_turn_recall"]) for r in part) / len(part),
                "all_hit": sum(float(r["all_answer_turns_hit"]) for r in part) / len(part),
                "avg_tokens": sum(float(r["retrieval_tokens"]) for r in part) / len(part),
                "avg_ms": sum(float(r["retrieval_ms"]) for r in part) / len(part),
                "p50_ms": statistics.median([float(r["retrieval_ms"]) for r in part]),
            }
        )
    write_csv(out_dir / "rl_complexity_compare.csv", compare)
    (out_dir / "action_counts.json").write_text(json.dumps(action_counts, ensure_ascii=False, indent=2), encoding="utf-8")
    router_dump = {name: router.dump() for name, router in routers.items() if hasattr(router, "dump")}
    (out_dir / "router_dump.json").write_text(json.dumps(router_dump, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "split_info.json").write_text(
        json.dumps({"train": [ex.qid for ex in train], "test": [ex.qid for ex in test]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print((out_dir / "rl_complexity_compare.csv").read_text(encoding="utf-8"))
    print("ACTION_COUNTS", json.dumps(action_counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
