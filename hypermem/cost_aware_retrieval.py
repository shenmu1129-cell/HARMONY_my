"""Cost-aware adaptive retrieval for profile-hypergraph memory.

This module adds three lightweight retrieval strategies on top of an already
built Topic-Episode-Fact + behavioral-hyperedge memory graph:

1. query_adaptive: choose profile, topic-episode, or compact dual path from the
   query route instead of always using both paths.
2. progressive: retrieve summaries/edges first and expand only a small number of
   representative facts.
3. budget: rank candidate facts by evidence value per token and pack under a
   token budget.

The implementation intentionally reuses the saved graph; it does not rebuild
memory or call any external LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from hypermem.dual_path_retrieval import (
    CandidateEvidence,
    EpisodeRecord,
    TopicRecord,
    build_topic_episode_indices,
    fact_episode_ids,
    fact_topic_id,
    retrieve_dual_path,
)
from hypermem.profile_centric_hypergraph import (
    HashedEmbeddingModel,
    ProfileCentricHypergraphMemory,
    ProfileFact,
    ProfileHyperedgeUnit,
    ProfileRetrievalResult,
    estimate_tokens,
)
from hypermem.query_router import route_query


@dataclass
class _ScoredEdge:
    edge: ProfileHyperedgeUnit
    score: float
    parts: Dict[str, float]


def _pack_by_score(
    memory: ProfileCentricHypergraphMemory,
    ranked: Iterable[Tuple[str, float]],
    *,
    top_k: int,
    max_tokens: int,
) -> List[ProfileFact]:
    selected: List[ProfileFact] = []
    seen = set()
    tokens = 0
    for fid, score in ranked:
        if fid in seen or score <= 0:
            continue
        fact = memory.facts.get(fid)
        if fact is None:
            continue
        tok = estimate_tokens(fact.content)
        if selected and tokens + tok > max_tokens:
            continue
        selected.append(fact)
        seen.add(fid)
        tokens += tok
        if len(selected) >= top_k or tokens >= max_tokens:
            break
    return selected


def _score_edges(
    memory: ProfileCentricHypergraphMemory,
    query_embedding: Sequence[float],
    query: str,
    *,
    top_k_edges: int,
    use_utility: bool,
) -> Tuple[List[_ScoredEdge], float]:
    query_type, _, _ = memory.infer_profile_type(query)
    scored: List[_ScoredEdge] = []
    for edge in memory.edges.values():
        score, parts = edge.score(
            query_embedding=query_embedding,
            query_type=query_type,
            facts=memory.facts,
            use_utility=use_utility,
            weights=memory.weights,
        )
        if score > 0:
            scored.append(_ScoredEdge(edge=edge, score=score, parts=parts))
    scored.sort(key=lambda item: item.score, reverse=True)
    selected = scored[:top_k_edges]
    edge_score = sum(item.score for item in selected) / max(1, len(selected))
    return selected, edge_score


def _candidate_from_edges(
    memory: ProfileCentricHypergraphMemory,
    query_embedding: Sequence[float],
    selected_edges: Sequence[_ScoredEdge],
) -> Dict[str, CandidateEvidence]:
    candidates: Dict[str, CandidateEvidence] = {}
    for item in selected_edges:
        edge = item.edge
        edge_sim = HashedEmbeddingModel.cosine(query_embedding, edge.embedding)
        for fid in edge.member_fact_ids:
            fact = memory.facts.get(fid)
            if fact is None:
                continue
            fact_sim = HashedEmbeddingModel.cosine(query_embedding, fact.embedding)
            score = 0.64 * fact_sim + 0.24 * edge_sim + 0.12 * item.score
            cand = candidates.setdefault(fid, CandidateEvidence(fact_id=fid))
            cand.profile_score = max(cand.profile_score, score)
            cand.score = max(cand.score, score)
            cand.profile_edge_ids.append(edge.edge_id)
            cand.source_paths.append("profile")
            cand.topic_ids.append(fact_topic_id(fact))
            cand.episode_ids.extend(fact_episode_ids(fact))
    return candidates


def _score_topic_episode_candidates(
    memory: ProfileCentricHypergraphMemory,
    query_embedding: Sequence[float],
    *,
    top_k_topics: int,
    top_k_episodes: int,
) -> Tuple[Dict[str, CandidateEvidence], List[Tuple[TopicRecord, float]], List[Tuple[EpisodeRecord, float]]]:
    topics, episodes = build_topic_episode_indices(memory)
    scored_topics = [
        (topic, HashedEmbeddingModel.cosine(query_embedding, topic.embedding))
        for topic in topics.values()
    ]
    scored_topics.sort(key=lambda item: item[1], reverse=True)
    selected_topics = scored_topics[:top_k_topics]
    topic_score_map = {topic.topic_id: score for topic, score in selected_topics}

    candidate_episodes: List[Tuple[EpisodeRecord, float]] = []
    for topic, topic_score in selected_topics:
        for eid in topic.episode_ids:
            ep = episodes.get(eid)
            if ep is None:
                continue
            ep_sim = HashedEmbeddingModel.cosine(query_embedding, ep.embedding)
            score = 0.70 * ep_sim + 0.30 * topic_score
            candidate_episodes.append((ep, score))
    candidate_episodes.sort(key=lambda item: item[1], reverse=True)
    selected_episodes = candidate_episodes[:top_k_episodes]

    candidates: Dict[str, CandidateEvidence] = {}
    for ep, ep_score in selected_episodes:
        topic_score = topic_score_map.get(ep.topic_id, 0.0)
        for fid in ep.fact_ids:
            fact = memory.facts.get(fid)
            if fact is None:
                continue
            fact_sim = HashedEmbeddingModel.cosine(query_embedding, fact.embedding)
            score = 0.62 * fact_sim + 0.25 * ep_score + 0.13 * topic_score
            cand = candidates.setdefault(fid, CandidateEvidence(fact_id=fid))
            cand.topic_score = max(cand.topic_score, topic_score)
            cand.episode_score = max(cand.episode_score, score)
            cand.score = max(cand.score, score)
            cand.topic_ids.append(ep.topic_id)
            cand.episode_ids.append(ep.episode_id)
            cand.source_paths.append("topic_episode")
    return candidates, selected_topics, selected_episodes


def retrieve_topic_episode_only(
    memory: ProfileCentricHypergraphMemory,
    query: str,
    *,
    top_k_facts: int = 8,
    max_tokens: int = 450,
    top_k_topics: int = 3,
    top_k_episodes: int = 6,
) -> ProfileRetrievalResult:
    query_embedding = memory.embedding_model.encode(query)
    candidates, selected_topics, selected_episodes = _score_topic_episode_candidates(
        memory,
        query_embedding,
        top_k_topics=top_k_topics,
        top_k_episodes=top_k_episodes,
    )
    ranked = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
    facts = _pack_by_score(memory, [(cand.fact_id, cand.score) for cand in ranked], top_k=top_k_facts, max_tokens=max_tokens)
    fact_score = {cand.fact_id: cand.score for cand in ranked}
    return ProfileRetrievalResult(
        query=query,
        channel="topic_episode_only",
        selected_edges=[],
        selected_facts=facts,
        score=sum(fact_score.get(f.fact_id, 0.0) for f in facts) / max(1, len(facts)),
        tokens=estimate_tokens([fact.content for fact in facts]),
        fallback_used=False,
        sufficient=bool(facts),
        debug_scores=[
            {
                "path": "topic",
                "topic_id": topic.topic_id,
                "score": round(score, 6),
                "title": topic.title,
                "episodes": len(topic.episode_ids),
                "facts": len(topic.fact_ids),
            }
            for topic, score in selected_topics
        ]
        + [
            {
                "path": "episode",
                "episode_id": ep.episode_id,
                "topic_id": ep.topic_id,
                "score": round(score, 6),
                "title": ep.title,
                "facts": len(ep.fact_ids),
            }
            for ep, score in selected_episodes
        ]
        + [
            {
                "path": "topic_episode_fact",
                "fact_id": fact.fact_id,
                "score": round(fact_score.get(fact.fact_id, 0.0), 6),
            }
            for fact in facts
        ],
    )


def retrieve_progressive(
    memory: ProfileCentricHypergraphMemory,
    query: str,
    *,
    top_k_edges: int = 3,
    top_k_facts: int = 8,
    max_tokens: int = 450,
    use_utility: bool = True,
    top_k_topics: int = 3,
    top_k_episodes: int = 6,
    representative_facts_per_edge: int = 2,
    expansion_ratio: float = 0.55,
) -> ProfileRetrievalResult:
    route = route_query(query)
    query_embedding = memory.embedding_model.encode(query)
    fact_budget = max(1, int(top_k_facts * expansion_ratio))
    token_budget = max(32, int(max_tokens * expansion_ratio))

    selected_edges: List[_ScoredEdge] = []
    candidates: Dict[str, CandidateEvidence] = {}
    selected_topics: List[Tuple[TopicRecord, float]] = []
    selected_episodes: List[Tuple[EpisodeRecord, float]] = []

    if route.route in {"behavioral", "mixed"}:
        selected_edges, _ = _score_edges(
            memory,
            query_embedding,
            query,
            top_k_edges=max(1, min(top_k_edges, 2)),
            use_utility=use_utility,
        )
        edge_candidates = _candidate_from_edges(memory, query_embedding, selected_edges)
        allowed_ids = set()
        for edge_item in selected_edges:
            edge_fact_ids = [
                (fid, edge_candidates.get(fid, CandidateEvidence(fid)).score)
                for fid in edge_item.edge.member_fact_ids
                if fid in edge_candidates
            ]
            edge_fact_ids.sort(key=lambda item: item[1], reverse=True)
            for fid, _ in edge_fact_ids[:representative_facts_per_edge]:
                allowed_ids.add(fid)
        candidates.update({fid: cand for fid, cand in edge_candidates.items() if fid in allowed_ids})

    if route.route in {"episodic", "mixed"}:
        topic_candidates, selected_topics, selected_episodes = _score_topic_episode_candidates(
            memory,
            query_embedding,
            top_k_topics=max(1, min(top_k_topics, 2)),
            top_k_episodes=max(1, min(top_k_episodes, 4)),
        )
        for fid, cand in topic_candidates.items():
            if fid not in candidates or cand.score > candidates[fid].score:
                candidates[fid] = cand

    if not candidates:
        topic_candidates, selected_topics, selected_episodes = _score_topic_episode_candidates(
            memory,
            query_embedding,
            top_k_topics=max(1, min(top_k_topics, 2)),
            top_k_episodes=max(1, min(top_k_episodes, 4)),
        )
        candidates.update(topic_candidates)

    ranked = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
    facts = _pack_by_score(memory, [(cand.fact_id, cand.score) for cand in ranked], top_k=fact_budget, max_tokens=token_budget)
    selected_edge_objs = [item.edge for item in selected_edges]
    fact_score = {cand.fact_id: cand for cand in ranked}
    for edge in selected_edge_objs:
        edge.access_count += 1
    return ProfileRetrievalResult(
        query=query,
        channel=f"progressive_{route.route}",
        selected_edges=selected_edge_objs,
        selected_facts=facts,
        score=sum(fact_score.get(f.fact_id, CandidateEvidence(f.fact_id)).score for f in facts) / max(1, len(facts)),
        tokens=estimate_tokens([fact.content for fact in facts]),
        fallback_used=False,
        sufficient=bool(facts),
        debug_scores=[
            {
                "path": "progressive_route",
                "route": route.route,
                "confidence": round(route.confidence, 6),
                "expanded_edges": len(selected_edge_objs),
                "expanded_topics": len(selected_topics),
                "expanded_episodes": len(selected_episodes),
                "candidate_facts": len(candidates),
                "selected_facts": len(facts),
                "token_budget": token_budget,
            }
        ]
        + [
            {
                "path": "progressive_fact",
                "fact_id": fact.fact_id,
                "score": round(fact_score.get(fact.fact_id, CandidateEvidence(fact.fact_id)).score, 6),
                "source_paths": fact_score.get(fact.fact_id, CandidateEvidence(fact.fact_id)).source_paths,
            }
            for fact in facts
        ],
    )


def retrieve_budget_aware(
    memory: ProfileCentricHypergraphMemory,
    query: str,
    *,
    top_k_edges: int = 3,
    top_k_facts: int = 8,
    max_tokens: int = 450,
    use_utility: bool = True,
    top_k_topics: int = 3,
    top_k_episodes: int = 6,
    budget_ratio: float = 0.65,
) -> ProfileRetrievalResult:
    query_embedding = memory.embedding_model.encode(query)
    token_budget = max(32, int(max_tokens * budget_ratio))

    selected_edges, _ = _score_edges(
        memory,
        query_embedding,
        query,
        top_k_edges=top_k_edges,
        use_utility=use_utility,
    )
    profile_candidates = _candidate_from_edges(memory, query_embedding, selected_edges)
    topic_candidates, selected_topics, selected_episodes = _score_topic_episode_candidates(
        memory,
        query_embedding,
        top_k_topics=top_k_topics,
        top_k_episodes=top_k_episodes,
    )

    candidates: Dict[str, CandidateEvidence] = {}
    for source in (profile_candidates, topic_candidates):
        for fid, cand in source.items():
            dst = candidates.setdefault(fid, CandidateEvidence(fact_id=fid))
            dst.profile_score = max(dst.profile_score, cand.profile_score)
            dst.topic_score = max(dst.topic_score, cand.topic_score)
            dst.episode_score = max(dst.episode_score, cand.episode_score)
            dst.score = max(dst.score, cand.score)
            dst.profile_edge_ids.extend(cand.profile_edge_ids)
            dst.topic_ids.extend(cand.topic_ids)
            dst.episode_ids.extend(cand.episode_ids)
            dst.source_paths.extend(cand.source_paths)

    ranked: List[Tuple[str, float]] = []
    for fid, cand in candidates.items():
        fact = memory.facts.get(fid)
        if fact is None:
            continue
        cost = max(1, estimate_tokens(fact.content))
        diversity_bonus = 0.04 * len(set(cand.source_paths))
        structural_bonus = 0.03 if cand.profile_score > 0 and cand.episode_score > 0 else 0.0
        value = cand.score + diversity_bonus + structural_bonus
        roi = value / (cost ** 0.65)
        ranked.append((fid, roi))
    ranked.sort(key=lambda item: item[1], reverse=True)
    facts = _pack_by_score(memory, ranked, top_k=top_k_facts, max_tokens=token_budget)

    fact_roi = dict(ranked)
    selected_edge_objs = [item.edge for item in selected_edges]
    for edge in selected_edge_objs:
        edge.access_count += 1
    return ProfileRetrievalResult(
        query=query,
        channel="budget_aware_value_per_token",
        selected_edges=selected_edge_objs,
        selected_facts=facts,
        score=sum(fact_roi.get(f.fact_id, 0.0) for f in facts) / max(1, len(facts)),
        tokens=estimate_tokens([fact.content for fact in facts]),
        fallback_used=False,
        sufficient=bool(facts),
        debug_scores=[
            {
                "path": "budget_controller",
                "token_budget": token_budget,
                "candidate_facts": len(candidates),
                "selected_facts": len(facts),
                "expanded_edges": len(selected_edge_objs),
                "expanded_topics": len(selected_topics),
                "expanded_episodes": len(selected_episodes),
            }
        ]
        + [
            {
                "path": "budget_fact",
                "fact_id": fact.fact_id,
                "roi": round(fact_roi.get(fact.fact_id, 0.0), 6),
                "tokens": estimate_tokens(fact.content),
            }
            for fact in facts
        ],
    )


def retrieve_query_adaptive(
    memory: ProfileCentricHypergraphMemory,
    query: str,
    *,
    top_k_edges: int = 3,
    top_k_facts: int = 8,
    max_tokens: int = 450,
    use_utility: bool = True,
    fallback: bool = True,
    sufficiency_threshold: float = 0.10,
    top_k_topics: int = 3,
    top_k_episodes: int = 6,
) -> ProfileRetrievalResult:
    route = route_query(query)
    if route.route == "episodic":
        result = retrieve_topic_episode_only(
            memory,
            query,
            top_k_facts=top_k_facts,
            max_tokens=max_tokens,
            top_k_topics=top_k_topics,
            top_k_episodes=top_k_episodes,
        )
    elif route.route == "behavioral":
        result = memory.retrieve(
            query,
            top_k_edges=top_k_edges,
            top_k_facts=top_k_facts,
            max_tokens=max_tokens,
            use_utility=use_utility,
            fallback=fallback,
            sufficiency_threshold=sufficiency_threshold,
        )
    else:
        result = retrieve_dual_path(
            memory,
            query,
            top_k_edges=max(1, min(top_k_edges, 2)),
            top_k_facts=top_k_facts,
            max_tokens=max_tokens,
            use_utility=use_utility,
            fallback=fallback,
            sufficiency_threshold=sufficiency_threshold,
            top_k_topics=max(1, min(top_k_topics, 2)),
            top_k_episodes=max(1, min(top_k_episodes, 4)),
        )
        result.channel = "query_adaptive_mixed_compact_dual_path"
    result.debug_scores.insert(
        0,
        {
            "path": "query_adaptive_router",
            "route": route.route,
            "confidence": round(route.confidence, 6),
            "matched_rules": route.matched_rules,
            "channel": result.channel,
        },
    )
    return result


def retrieve_cost_aware(
    memory: ProfileCentricHypergraphMemory,
    query: str,
    *,
    strategy: str,
    top_k_edges: int = 3,
    top_k_facts: int = 8,
    max_tokens: int = 450,
    use_utility: bool = True,
    fallback: bool = True,
    sufficiency_threshold: float = 0.10,
    top_k_topics: int = 3,
    top_k_episodes: int = 6,
    budget_ratio: float = 0.65,
    expansion_ratio: float = 0.55,
    representative_facts_per_edge: int = 2,
) -> ProfileRetrievalResult:
    if strategy == "query_adaptive":
        return retrieve_query_adaptive(
            memory,
            query,
            top_k_edges=top_k_edges,
            top_k_facts=top_k_facts,
            max_tokens=max_tokens,
            use_utility=use_utility,
            fallback=fallback,
            sufficiency_threshold=sufficiency_threshold,
            top_k_topics=top_k_topics,
            top_k_episodes=top_k_episodes,
        )
    if strategy == "progressive":
        return retrieve_progressive(
            memory,
            query,
            top_k_edges=top_k_edges,
            top_k_facts=top_k_facts,
            max_tokens=max_tokens,
            use_utility=use_utility,
            top_k_topics=top_k_topics,
            top_k_episodes=top_k_episodes,
            representative_facts_per_edge=representative_facts_per_edge,
            expansion_ratio=expansion_ratio,
        )
    if strategy == "budget":
        return retrieve_budget_aware(
            memory,
            query,
            top_k_edges=top_k_edges,
            top_k_facts=top_k_facts,
            max_tokens=max_tokens,
            use_utility=use_utility,
            top_k_topics=top_k_topics,
            top_k_episodes=top_k_episodes,
            budget_ratio=budget_ratio,
        )
    raise ValueError(f"unknown cost-aware strategy: {strategy}")
