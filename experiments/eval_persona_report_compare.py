from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import profile_centric_hypergraph_eval as base_eval  # noqa: E402
from experiments.eval_persona_advanced_retrieval import (  # noqa: E402
    MethodConfig,
    load_questions,
    read_jsonl,
    retrieve_advanced,
    retrieve_reference,
    row_id_map,
    summarize,
)
from experiments.eval_persona_rl_retrieval import run_policy  # noqa: E402
from hypermem.cost_aware_retrieval import (  # noqa: E402
    retrieve_budget_aware,
    retrieve_topic_episode_only,
)
from hypermem.profile_centric_hypergraph import (  # noqa: E402
    HashedEmbeddingModel,
    ProfileCentricHypergraphMemory,
    ProfileFact,
    ProfileRetrievalResult,
    estimate_tokens,
    keyword_overlap,
    tokenize,
)


def pack_facts(facts: Sequence[Tuple[ProfileFact, float]], *, top_k: int, max_tokens: int) -> List[ProfileFact]:
    out: List[ProfileFact] = []
    tokens = 0
    seen = set()
    for fact, score in facts:
        key = fact.content.lower().strip()
        if key in seen or score <= 0:
            continue
        cost = estimate_tokens(fact.content)
        if out and tokens + cost > max_tokens:
            continue
        out.append(fact)
        seen.add(key)
        tokens += cost
        if len(out) >= top_k or tokens >= max_tokens:
            break
    return out


def wants_response(query: str) -> bool:
    q = query.lower()
    return "next response" in q or "given the persona" in q or "expected response" in q


def wants_persona(query: str) -> bool:
    q = query.lower()
    return "persona/profile" in q or "profile information" in q


def is_response(fact: ProfileFact) -> bool:
    stype = str((fact.metadata or {}).get("source_type") or "").lower()
    return stype == "response" or fact.content.lower().startswith("expected response:")


def is_persona(fact: ProfileFact) -> bool:
    stype = str((fact.metadata or {}).get("source_type") or "").lower()
    return stype == "persona" or fact.content.lower().startswith("persona statement:")


def make_source_facts(memory: ProfileCentricHypergraphMemory, memory_json: Path) -> List[ProfileFact]:
    rows = read_jsonl(memory_json)
    by_alias = row_id_map(rows, memory)
    unique: Dict[str, ProfileFact] = {}
    for fact in by_alias.values():
        unique[fact.fact_id] = fact
    return sorted(unique.values(), key=lambda f: f.timestamp)


class BM25Index:
    def __init__(self, facts: Sequence[ProfileFact], k1: float = 1.5, b: float = 0.75) -> None:
        self.facts = list(facts)
        self.k1 = k1
        self.b = b
        self.doc_terms = [tokenize(f.content) for f in self.facts]
        self.avgdl = sum(len(t) for t in self.doc_terms) / max(1, len(self.doc_terms))
        df: Dict[str, int] = {}
        for terms in self.doc_terms:
            for tok in set(terms):
                df[tok] = df.get(tok, 0) + 1
        n = max(1, len(self.facts))
        self.idf = {tok: math.log(1 + (n - c + 0.5) / (c + 0.5)) for tok, c in df.items()}

    def score(self, query: str, fact_idx: int) -> float:
        q_terms = tokenize(query)
        terms = self.doc_terms[fact_idx]
        if not q_terms or not terms:
            return 0.0
        counts: Dict[str, int] = {}
        for tok in terms:
            counts[tok] = counts.get(tok, 0) + 1
        dl = len(terms)
        score = 0.0
        for tok in q_terms:
            tf = counts.get(tok, 0)
            if tf <= 0:
                continue
            idf = self.idf.get(tok, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1e-6, self.avgdl))
            score += idf * (tf * (self.k1 + 1)) / denom
        return score


def classic_retrieve(method: str, facts: Sequence[ProfileFact], bm25: BM25Index, query: str, *, top_k: int = 4, max_tokens: int = 140) -> ProfileRetrievalResult:
    t0 = time.time()
    rng = random.Random(13)
    if method == "random_source":
        ranked = [(f, 1.0) for f in rng.sample(list(facts), k=min(top_k, len(facts)))]
    elif method == "recency_source":
        ranked = [(f, max(1.0, f.timestamp)) for f in sorted(facts, key=lambda x: x.timestamp, reverse=True)]
    elif method == "bm25_source":
        ranked = [(f, bm25.score(query, i)) for i, f in enumerate(facts)]
    elif method == "dense_hash_source":
        emb = HashedEmbeddingModel(dim=512).encode(query)
        ranked = [(f, HashedEmbeddingModel.cosine(emb, f.embedding)) for f in facts]
    elif method == "hybrid_source":
        emb = HashedEmbeddingModel(dim=512).encode(query)
        ranked = [
            (f, 0.52 * HashedEmbeddingModel.cosine(emb, f.embedding) + 0.34 * keyword_overlap(query, f.content) + 0.14 * bm25.score(query, i))
            for i, f in enumerate(facts)
        ]
    elif method == "response_prior_source":
        emb = HashedEmbeddingModel(dim=512).encode(query)
        ranked = []
        for i, f in enumerate(facts):
            bonus = 0.0
            if wants_response(query) and is_response(f):
                bonus += 0.20
            if wants_persona(query) and is_persona(f):
                bonus += 0.14
            cost = estimate_tokens(f.content)
            score = 0.48 * HashedEmbeddingModel.cosine(emb, f.embedding) + 0.28 * keyword_overlap(query, f.content) + 0.10 * bm25.score(query, i) + bonus
            ranked.append((f, score / (max(1, cost) ** 0.35)))
    else:
        raise ValueError(method)
    ranked.sort(key=lambda x: x[1], reverse=True)
    selected = pack_facts(ranked, top_k=top_k, max_tokens=max_tokens)
    return ProfileRetrievalResult(
        query=query,
        channel=method,
        selected_edges=[],
        selected_facts=selected,
        score=sum(s for _, s in ranked[:top_k]) / max(1, min(top_k, len(ranked))),
        tokens=estimate_tokens([f.content for f in selected]),
        fallback_used=False,
        sufficient=bool(selected),
        debug_scores=[{"method": method, "candidate_facts": len(facts), "latency_ms": round((time.time() - t0) * 1000, 4)}],
    )


def graph_retrieve(method: str, memory: ProfileCentricHypergraphMemory, source_rows: Dict[str, ProfileFact], query: str) -> ProfileRetrievalResult:
    if method == "profile_full":
        return memory.retrieve(query, top_k_edges=3, top_k_facts=8, max_tokens=450, use_utility=False, fallback=True)
    if method == "topic_episode":
        return retrieve_topic_episode_only(memory, query, top_k_facts=8, max_tokens=450, top_k_topics=3, top_k_episodes=6)
    if method == "budget_aware":
        return retrieve_budget_aware(memory, query, top_k_edges=3, top_k_facts=8, max_tokens=450, use_utility=False, top_k_topics=3, top_k_episodes=6, budget_ratio=0.55)
    if method == "adaptive_tiny":
        return retrieve_reference(memory, query)
    if method == "hg_roi_light_k4":
        cfg = MethodConfig("hg_roi_light_k4", response_boost=0.12, persona_boost=0.14, graph_gate="hybrid", top_k_facts=4, max_tokens=140)
        return retrieve_advanced(memory, source_rows, query, cfg)
    if method == "hg_edge_source":
        cfg = MethodConfig("hg_edge_source", response_boost=0.30, persona_boost=0.12, graph_gate="edge", top_k_edges=3, top_k_facts=4, max_tokens=140)
        return retrieve_advanced(memory, source_rows, query, cfg)
    if method == "hg_global_response":
        cfg = MethodConfig("hg_global_response", response_boost=0.40, persona_boost=0.18, graph_gate="global_response", top_k_facts=4, max_tokens=90)
        return retrieve_advanced(memory, source_rows, query, cfg)
    raise ValueError(method)


def evaluate_method(method: str, phase: str, questions: Sequence[Dict[str, Any]], retrieve_fn) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for q in questions:
        t0 = time.time()
        result = retrieve_fn(q["question"])
        row, _, _, _ = base_eval.row_from_result(method, q, result, update_used=False)
        row["phase"] = phase
        row["retrieval_ms"] = round((time.time() - t0) * 1000.0, 4)
        row["candidate_facts"] = result.debug_scores[0].get("candidate_facts", len(result.selected_facts)) if result.debug_scores else len(result.selected_facts)
        rows.append(row)
    return rows


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["phase"]), []).append(row)
    out = []
    for (method, phase), part in sorted(groups.items()):
        out.append({"method": method, "phase": phase, **summarize(part)})
    return out


def run_static_suite(args: argparse.Namespace, graph_path: str, label: str, questions: List[Dict[str, Any]], phases: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    memory = ProfileCentricHypergraphMemory.load(graph_path)
    source_rows = row_id_map(read_jsonl(Path(args.memory_json)), memory)
    source_facts = make_source_facts(memory, Path(args.memory_json))
    bm25 = BM25Index(source_facts)
    rows: List[Dict[str, Any]] = []

    classic_methods = ["random_source", "recency_source", "bm25_source", "dense_hash_source", "hybrid_source", "response_prior_source"]
    graph_methods = ["profile_full", "topic_episode", "budget_aware", "adaptive_tiny", "hg_roi_light_k4", "hg_edge_source", "hg_global_response"]

    for phase, qrows in phases.items():
        for method in classic_methods:
            rows.extend(evaluate_method(f"{label}_{method}", phase, qrows, lambda q, m=method: classic_retrieve(m, source_facts, bm25, q)))
        for method in graph_methods:
            rows.extend(evaluate_method(f"{label}_{method}", phase, qrows, lambda q, m=method: graph_retrieve(m, memory, source_rows, q)))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--memory-json", required=True)
    p.add_argument("--questions-json", required=True)
    p.add_argument("--local-graph", required=True)
    p.add_argument("--llm-graph", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-questions", type=int, default=1000)
    p.add_argument("--split-train", type=int, default=500)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(Path(args.questions_json), max_questions=args.max_questions)
    phases = {
        "all": questions,
        "train": questions[: args.split_train],
        "test": questions[args.split_train :],
    }

    all_rows: List[Dict[str, Any]] = []
    all_rows.extend(run_static_suite(args, args.local_graph, "local", questions, phases))
    all_rows.extend(run_static_suite(args, args.llm_graph, "llm", questions, phases))

    # Report the strongest HG+RL settings already used in the paper-style run.
    for policy, seed in [("thompson", 11), ("ucb1", 11)]:
        name, rows, _, _ = run_policy(
            policy,
            args.llm_graph,
            args.memory_json,
            args.questions_json,
            args.max_questions,
            args.split_train,
            "fast_pruned",
            seed,
        )
        for row in rows:
            row["method"] = f"llm_hg_rl_{policy}_fast_s{seed}"
        all_rows.extend(rows)

    write_rows(out_dir / "report_compare_results.csv", all_rows)
    summary = summarize_rows(all_rows)
    write_rows(out_dir / "report_compare_summary.csv", summary)
    (out_dir / "report_compare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print((out_dir / "report_compare_summary.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
