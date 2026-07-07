from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import profile_centric_hypergraph_eval as base_eval  # noqa: E402
from hypermem.cost_aware_retrieval import retrieve_budget_aware  # noqa: E402
from hypermem.dual_path_retrieval import (  # noqa: E402
    build_topic_episode_indices,
    fact_episode_ids,
    fact_topic_id,
)
from hypermem.profile_centric_hypergraph import (  # noqa: E402
    HashedEmbeddingModel,
    ProfileCentricHypergraphMemory,
    ProfileFact,
    ProfileRetrievalResult,
    estimate_tokens,
    keyword_overlap,
)
from hypermem.query_router import route_query, route_to_dict  # noqa: E402


@dataclass
class MethodConfig:
    name: str
    top_k_edges: int = 2
    top_k_topics: int = 2
    top_k_episodes: int = 3
    top_k_facts: int = 4
    max_tokens: int = 110
    response_boost: float = 0.22
    persona_boost: float = 0.10
    source_preserve: bool = True
    roi_alpha: float = 0.65
    novelty: float = 0.00
    graph_gate: str = "hybrid"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_questions(path: Path, max_questions: int = 0) -> List[Dict[str, Any]]:
    rows = base_eval.normalize_questions(base_eval.read_json_or_jsonl(path))
    return rows[:max_questions] if max_questions else rows


def row_id_map(memory_rows: Sequence[Dict[str, Any]], memory: ProfileCentricHypergraphMemory) -> Dict[str, ProfileFact]:
    out: Dict[str, ProfileFact] = {}
    for idx, row in enumerate(memory_rows, start=1):
        rid = str(row.get("row_id") or f"row_{idx:06d}")
        content = str(row.get("content") or row.get("text") or row.get("fact") or "")
        if not content:
            continue
        fact = ProfileFact(
            fact_id=f"source_{rid}",
            content=content,
            keywords=[str(x) for x in row.get("keywords", [])] if isinstance(row.get("keywords"), list) else [],
            timestamp=float(row.get("timestamp") or idx),
            embedding=memory.embedding_model.encode(content),
            metadata={**row, "source_row_id": rid, "source_preserved": True},
        )
        aliases = {
            rid,
            f"row_{idx:06d}",
            f"imported_{idx:06d}",
            str(row.get("fact_id") or ""),
            str(row.get("id") or ""),
        }
        for alias in aliases:
            if alias:
                out[alias] = fact
    return out


def source_row_ids(fact: ProfileFact) -> List[str]:
    meta = fact.metadata or {}
    ids = []
    for key in ("source_row_ids", "source_rows"):
        val = meta.get(key)
        if isinstance(val, list):
            ids.extend(str(x) for x in val)
    nested = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    fact_meta = nested.get("fact") if isinstance(nested.get("fact"), dict) else {}
    val = fact_meta.get("source_row_ids")
    if isinstance(val, list):
        ids.extend(str(x) for x in val)
    return list(dict.fromkeys(ids))


def is_response_fact(fact: ProfileFact) -> bool:
    text = fact.content.lower().strip()
    source_type = str((fact.metadata or {}).get("source_type") or "").lower()
    return source_type in {"response", "expected_response"} or text.startswith("expected response:")


def is_persona_fact(fact: ProfileFact) -> bool:
    text = fact.content.lower().strip()
    source_type = str((fact.metadata or {}).get("source_type") or "").lower()
    return source_type == "persona" or text.startswith("persona statement:")


def wants_response(query: str) -> bool:
    q = query.lower()
    return "next response" in q or "expected response" in q or "given the persona" in q


def wants_persona(query: str) -> bool:
    q = query.lower()
    return "persona/profile" in q or "persona information" in q or "profile information" in q


def scored_edges(memory: ProfileCentricHypergraphMemory, query: str, top_k: int) -> List[Tuple[Any, float]]:
    qtype, _, _ = memory.infer_profile_type(query)
    qemb = memory.embedding_model.encode(query)
    scored = []
    for edge in memory.edges.values():
        score, _ = edge.score(qemb, qtype, memory.facts, use_utility=False, weights=memory.weights)
        if score > 0:
            scored.append((edge, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def graph_candidate_fact_ids(memory: ProfileCentricHypergraphMemory, query: str, cfg: MethodConfig) -> Dict[str, float]:
    qemb = memory.embedding_model.encode(query)
    candidates: Dict[str, float] = {}

    if cfg.graph_gate in {"hybrid", "edge"}:
        for edge, edge_score in scored_edges(memory, query, cfg.top_k_edges):
            edge_sim = HashedEmbeddingModel.cosine(qemb, edge.embedding)
            for fid in edge.member_fact_ids:
                fact = memory.facts.get(fid)
                if fact is None:
                    continue
                sim = HashedEmbeddingModel.cosine(qemb, fact.embedding)
                score = 0.66 * sim + 0.20 * edge_sim + 0.14 * edge_score
                candidates[fid] = max(candidates.get(fid, 0.0), score)

    if cfg.graph_gate in {"hybrid", "topic"}:
        topics, episodes = build_topic_episode_indices(memory)
        topic_scores = [(topic, HashedEmbeddingModel.cosine(qemb, topic.embedding)) for topic in topics.values()]
        topic_scores.sort(key=lambda x: x[1], reverse=True)
        selected_topics = topic_scores[: cfg.top_k_topics]
        selected_topic_ids = {topic.topic_id for topic, _ in selected_topics}
        episode_scores = []
        for ep in episodes.values():
            if ep.topic_id not in selected_topic_ids:
                continue
            topic_score = next((s for t, s in selected_topics if t.topic_id == ep.topic_id), 0.0)
            ep_sim = HashedEmbeddingModel.cosine(qemb, ep.embedding)
            episode_scores.append((ep, 0.72 * ep_sim + 0.28 * topic_score))
        episode_scores.sort(key=lambda x: x[1], reverse=True)
        for ep, ep_score in episode_scores[: cfg.top_k_episodes]:
            for fid in ep.fact_ids:
                fact = memory.facts.get(fid)
                if fact is None:
                    continue
                sim = HashedEmbeddingModel.cosine(qemb, fact.embedding)
                score = 0.70 * sim + 0.30 * ep_score
                candidates[fid] = max(candidates.get(fid, 0.0), score)

    if not candidates:
        for fact in memory.facts.values():
            candidates[fact.fact_id] = HashedEmbeddingModel.cosine(qemb, fact.embedding)
    return candidates


def materialize_source_facts(
    memory: ProfileCentricHypergraphMemory,
    fact_ids: Iterable[str],
    source_rows: Dict[str, ProfileFact],
    *,
    source_preserve: bool,
) -> Dict[str, ProfileFact]:
    out: Dict[str, ProfileFact] = {}
    for fid in fact_ids:
        fact = memory.facts.get(fid)
        if fact is None:
            continue
        if source_preserve:
            ids = source_row_ids(fact)
            found = False
            for rid in ids:
                src = source_rows.get(rid)
                if src is not None:
                    out[src.fact_id] = src
                    found = True
            if found:
                continue
        out[fact.fact_id] = fact
    return out


def source_type_bonus(fact: ProfileFact, query: str, cfg: MethodConfig, source_weights: Dict[str, float] | None = None) -> float:
    bonus = 0.0
    if wants_response(query) and is_response_fact(fact):
        bonus += cfg.response_boost
    if wants_persona(query) and is_persona_fact(fact):
        bonus += cfg.persona_boost
    if source_weights:
        stype = str((fact.metadata or {}).get("source_type") or ("response" if is_response_fact(fact) else "persona" if is_persona_fact(fact) else "other"))
        bonus += source_weights.get(stype, 0.0)
    return bonus


def pack_ranked(memory: ProfileCentricHypergraphMemory, ranked: Sequence[Tuple[ProfileFact, float]], cfg: MethodConfig) -> List[ProfileFact]:
    selected: List[ProfileFact] = []
    tokens = 0
    used_types = set()
    used_dialogues = set()
    for fact, score in ranked:
        if score <= 0:
            continue
        cost = estimate_tokens(fact.content)
        if selected and tokens + cost > cfg.max_tokens:
            continue
        stype = str((fact.metadata or {}).get("source_type") or "")
        did = str((fact.metadata or {}).get("dialogue_id") or "")
        selected.append(fact)
        tokens += cost
        used_types.add(stype)
        used_dialogues.add(did)
        if len(selected) >= cfg.top_k_facts or tokens >= cfg.max_tokens:
            break
    return selected


def retrieve_advanced(
    memory: ProfileCentricHypergraphMemory,
    source_rows: Dict[str, ProfileFact],
    query: str,
    cfg: MethodConfig,
    source_weights: Dict[str, float] | None = None,
) -> ProfileRetrievalResult:
    qemb = memory.embedding_model.encode(query)
    candidate_scores = graph_candidate_fact_ids(memory, query, cfg)
    candidates = materialize_source_facts(memory, candidate_scores, source_rows, source_preserve=cfg.source_preserve)
    if cfg.graph_gate == "global_response":
        candidates = {f.fact_id: f for f in list(memory.facts.values()) + list(source_rows.values())}

    ranked: List[Tuple[ProfileFact, float]] = []
    seen_content = set()
    for fact in candidates.values():
        key = fact.content.lower().strip()
        if key in seen_content:
            continue
        seen_content.add(key)
        sim = HashedEmbeddingModel.cosine(qemb, fact.embedding)
        lex = keyword_overlap(query, fact.content)
        cost = max(1, estimate_tokens(fact.content))
        prior = source_type_bonus(fact, query, cfg, source_weights)
        base_score = 0.62 * sim + 0.23 * lex + prior - 0.03 * min(1.0, cost / 80.0)
        if "roi" in cfg.name:
            score = base_score / (cost ** cfg.roi_alpha)
        else:
            score = base_score
        ranked.append((fact, score))
    ranked.sort(key=lambda x: x[1], reverse=True)

    if cfg.novelty > 0:
        selected: List[ProfileFact] = []
        selected_tokens = 0
        selected_terms = set()
        for _ in range(cfg.top_k_facts):
            best = None
            best_score = -1.0
            for fact, score in ranked:
                if fact in selected:
                    continue
                terms = set(base_eval.keyword_overlap.__globals__["tokenize"](fact.content)) if False else set()
                overlap_penalty = 0.0
                if selected_terms:
                    fact_terms = set(fact.content.lower().split())
                    overlap_penalty = len(fact_terms & selected_terms) / max(1, len(fact_terms | selected_terms))
                adjusted = score - cfg.novelty * overlap_penalty
                if adjusted > best_score:
                    best_score = adjusted
                    best = fact
            if best is None:
                break
            cost = estimate_tokens(best.content)
            if selected and selected_tokens + cost > cfg.max_tokens:
                break
            selected.append(best)
            selected_tokens += cost
            selected_terms |= set(best.content.lower().split())
        facts = selected
    else:
        facts = pack_ranked(memory, ranked, cfg)

    edges = [edge for edge, _ in scored_edges(memory, query, cfg.top_k_edges)] if cfg.graph_gate in {"hybrid", "edge"} else []
    return ProfileRetrievalResult(
        query=query,
        channel=cfg.name,
        selected_edges=edges,
        selected_facts=facts,
        score=sum(score for _, score in ranked[: max(1, cfg.top_k_facts)]) / max(1, min(len(ranked), cfg.top_k_facts)),
        tokens=estimate_tokens([f.content for f in facts]),
        fallback_used=False,
        sufficient=bool(facts),
        debug_scores=[{
            "method": cfg.name,
            "candidate_facts": len(candidates),
            "top_k_edges": cfg.top_k_edges,
            "top_k_topics": cfg.top_k_topics,
            "top_k_episodes": cfg.top_k_episodes,
            "source_preserve": int(cfg.source_preserve),
            "response_boost": cfg.response_boost,
            "persona_boost": cfg.persona_boost,
        }],
    )


def retrieve_reference(memory: ProfileCentricHypergraphMemory, query: str) -> ProfileRetrievalResult:
    return retrieve_budget_aware(
        memory,
        query,
        top_k_edges=2,
        top_k_facts=4,
        max_tokens=110,
        use_utility=False,
        top_k_topics=2,
        top_k_episodes=3,
        budget_ratio=1.0,
    )


def evaluate_static_method(args_tuple: Tuple[str, Dict[str, Any], str, str, str, int]) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    name, cfg_dict, graph_path, memory_json, questions_json, max_questions = args_tuple
    memory = ProfileCentricHypergraphMemory.load(graph_path)
    memory_rows = read_jsonl(Path(memory_json))
    source_rows = row_id_map(memory_rows, memory)
    questions = load_questions(Path(questions_json), max_questions=max_questions)
    cfg = MethodConfig(name=name, **cfg_dict)
    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    for q in questions:
        t0 = time.time()
        result = retrieve_reference(memory, q["question"]) if name == "adaptive_tiny_ref" else retrieve_advanced(memory, source_rows, q["question"], cfg)
        row, _, _, _ = base_eval.row_from_result(name, q, result, update_used=False)
        route = route_query(q["question"])
        row.update(route_to_dict(route))
        row["retrieval_ms"] = round((time.time() - t0) * 1000.0, 4)
        row["candidate_facts"] = result.debug_scores[0].get("candidate_facts", len(result.selected_facts)) if result.debug_scores else len(result.selected_facts)
        rows.append(row)
        traces.append({**row, "gold": q["gold"], "evidence": [f.content for f in result.selected_facts], "debug": result.debug_scores})
    return name, rows, traces


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out = base_eval.summarize(rows)
    n = max(1, len(rows))
    out.update({
        "retrieval_ms": round(sum(float(r.get("retrieval_ms", 0.0)) for r in rows) / n, 4),
        "num_facts": round(sum(float(r.get("num_facts", 0.0)) for r in rows) / n, 4),
        "candidate_facts": round(sum(float(r.get("candidate_facts", 0.0)) for r in rows) / n, 4),
    })
    return out


def evaluate_bandit(
    graph_path: str,
    memory_json: str,
    questions_json: str,
    max_questions: int,
    out_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    memory = ProfileCentricHypergraphMemory.load(graph_path)
    memory_rows = read_jsonl(Path(memory_json))
    source_rows = row_id_map(memory_rows, memory)
    questions = load_questions(Path(questions_json), max_questions=max_questions)
    arms = [
        MethodConfig("bandit_arm_ref"),
        MethodConfig("bandit_arm_source_response", response_boost=0.28, graph_gate="hybrid"),
        MethodConfig("bandit_arm_roi", response_boost=0.24, graph_gate="hybrid", top_k_facts=4, max_tokens=110),
        MethodConfig("bandit_arm_global_response_roi", response_boost=0.35, graph_gate="global_response", top_k_facts=4, max_tokens=90),
        MethodConfig("bandit_arm_wide_source", response_boost=0.22, top_k_edges=3, top_k_topics=3, top_k_episodes=5, top_k_facts=6, max_tokens=160),
    ]
    stats: Dict[str, Dict[str, float]] = {
        route: {arm.name + "_n": 0.0 for arm in arms} | {arm.name + "_r": 0.0 for arm in arms}
        for route in ["mixed", "behavioral", "episodic"]
    }
    rng = random.Random(7)
    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    source_weights: Dict[str, float] = {"response": 0.0, "persona": 0.0, "dialogue_context": 0.0}
    for idx, q in enumerate(questions, start=1):
        route = route_query(q["question"]).route
        total = sum(stats[route][arm.name + "_n"] for arm in arms) + 1.0
        if idx <= len(arms) or rng.random() < 0.04:
            arm = arms[(idx - 1) % len(arms)]
        else:
            def ucb(a: MethodConfig) -> float:
                n = stats[route][a.name + "_n"]
                r = stats[route][a.name + "_r"]
                mean = r / max(1.0, n)
                return mean + 0.35 * math.sqrt(math.log(total + 1.0) / max(1.0, n))
            arm = max(arms, key=ucb)
        t0 = time.time()
        if arm.name == "bandit_arm_ref":
            result = retrieve_reference(memory, q["question"])
        else:
            result = retrieve_advanced(memory, source_rows, q["question"], arm, source_weights=source_weights)
        row, reward, hit, _ = base_eval.row_from_result("rl_ucb_router", q, result, update_used=True)
        stats[route][arm.name + "_n"] += 1.0
        stats[route][arm.name + "_r"] += reward
        for f in result.selected_facts:
            stype = str((f.metadata or {}).get("source_type") or "")
            if stype:
                source_weights[stype] = max(-0.15, min(0.25, source_weights.get(stype, 0.0) + (0.015 if hit else -0.004)))
        row.update(route_to_dict(route_query(q["question"])))
        row["chosen_arm"] = arm.name
        row["retrieval_ms"] = round((time.time() - t0) * 1000.0, 4)
        row["candidate_facts"] = result.debug_scores[0].get("candidate_facts", len(result.selected_facts)) if result.debug_scores else len(result.selected_facts)
        rows.append(row)
        traces.append({**row, "gold": q["gold"], "evidence": [f.content for f in result.selected_facts], "debug": result.debug_scores})
    (out_dir / "rl_bandit_policy.json").write_text(json.dumps({"stats": stats, "source_weights": source_weights}, indent=2), encoding="utf-8")
    return rows, traces


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-graph", required=True)
    parser.add_argument("--memory-json", required=True)
    parser.add_argument("--questions-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-questions", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs: List[Tuple[str, Dict[str, Any]]] = [
        ("adaptive_tiny_ref", {}),
        ("source_response_gate", {"response_boost": 0.24, "persona_boost": 0.12, "graph_gate": "hybrid"}),
        ("source_response_gate_wide", {"response_boost": 0.22, "persona_boost": 0.12, "top_k_edges": 3, "top_k_topics": 3, "top_k_episodes": 5, "top_k_facts": 6, "max_tokens": 160}),
        ("response_prior_roi", {"response_boost": 0.28, "persona_boost": 0.12, "graph_gate": "hybrid"}),
        ("response_prior_roi_strong", {"response_boost": 0.38, "persona_boost": 0.16, "graph_gate": "hybrid"}),
        ("global_response_roi", {"response_boost": 0.40, "persona_boost": 0.18, "graph_gate": "global_response", "top_k_facts": 4, "max_tokens": 90}),
        ("novelty_roi", {"response_boost": 0.28, "persona_boost": 0.12, "novelty": 0.10, "graph_gate": "hybrid"}),
        ("topic_source_response", {"response_boost": 0.30, "persona_boost": 0.12, "graph_gate": "topic", "top_k_topics": 3, "top_k_episodes": 6}),
        ("edge_source_response", {"response_boost": 0.30, "persona_boost": 0.12, "graph_gate": "edge", "top_k_edges": 3}),
    ]

    all_rows: List[Dict[str, Any]] = []
    all_traces: List[Dict[str, Any]] = []
    jobs = [
        (name, cfg, args.memory_graph, args.memory_json, args.questions_json, args.max_questions)
        for name, cfg in configs
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(evaluate_static_method, job) for job in jobs]
        for fut in as_completed(futures):
            name, rows, traces = fut.result()
            print(f"[done] {name} n={len(rows)} acc={summarize(rows)['accuracy']} ms={summarize(rows)['retrieval_ms']}", flush=True)
            all_rows.extend(rows)
            all_traces.extend(traces)

    bandit_rows, bandit_traces = evaluate_bandit(args.memory_graph, args.memory_json, args.questions_json, args.max_questions, out_dir)
    print(f"[done] rl_ucb_router n={len(bandit_rows)} acc={summarize(bandit_rows)['accuracy']} ms={summarize(bandit_rows)['retrieval_ms']}", flush=True)
    all_rows.extend(bandit_rows)
    all_traces.extend(bandit_traces)

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        by_method.setdefault(row["method"], []).append(row)
    summary_rows = [{"method": method, **summarize(rows)} for method, rows in sorted(by_method.items())]
    write_rows(out_dir / "advanced_results.csv", all_rows)
    write_rows(out_dir / "advanced_summary.csv", summary_rows)
    with (out_dir / "advanced_trace.jsonl").open("w", encoding="utf-8") as f:
        for row in all_traces:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "advanced_summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print((out_dir / "advanced_summary.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
