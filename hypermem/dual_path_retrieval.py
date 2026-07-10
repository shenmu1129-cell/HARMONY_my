"""Dual-path hypergraph evidence alignment retrieval.

This module does not rebuild memory. It works on an already saved
ProfileCentricHypergraphMemory graph and derives topic/episode indices from the
fact metadata emitted by the Topic-Episode-Fact hierarchy builder.

Retrieval paths:
    A. query -> behavioral profile hyperedges -> member facts -> source episodes
    B. query -> topics -> episodes -> facts

The final evidence set is selected by fact-level fusion with an alignment bonus
when the same topic/episode/fact is supported by both paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from hypermem.profile_centric_hypergraph import (
    HashedEmbeddingModel,
    ProfileCentricHypergraphMemory,
    ProfileFact,
    ProfileHyperedgeUnit,
    ProfileRetrievalResult,
    estimate_tokens,
)


@dataclass
class TopicRecord:
    topic_id: str
    title: str = ""
    summary: str = ""
    fact_ids: List[str] = field(default_factory=list)
    episode_ids: List[str] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)

    def text(self) -> str:
        return " ".join([self.title, self.summary]).strip()


@dataclass
class EpisodeRecord:
    episode_id: str
    topic_id: str = ""
    title: str = ""
    summary: str = ""
    fact_ids: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    embedding: List[float] = field(default_factory=list)

    def text(self) -> str:
        return " ".join([self.title, self.summary]).strip()


@dataclass
class CandidateEvidence:
    fact_id: str
    score: float = 0.0
    profile_score: float = 0.0
    topic_score: float = 0.0
    episode_score: float = 0.0
    alignment_score: float = 0.0
    profile_edge_ids: List[str] = field(default_factory=list)
    topic_ids: List[str] = field(default_factory=list)
    episode_ids: List[str] = field(default_factory=list)
    source_paths: List[str] = field(default_factory=list)


def _fact_meta(fact: ProfileFact) -> Dict[str, Any]:
    return fact.metadata or {}


def _clean_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x) for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def fact_topic_id(fact: ProfileFact) -> str:
    meta = _fact_meta(fact)
    if meta.get("topic_id"):
        return str(meta.get("topic_id"))
    nested = meta.get("metadata") or {}
    topic = nested.get("topic") if isinstance(nested, dict) else {}
    if isinstance(topic, dict) and topic.get("topic_id"):
        return str(topic.get("topic_id"))
    return "unknown_topic"


def fact_topic_title(fact: ProfileFact) -> str:
    meta = _fact_meta(fact)
    if meta.get("topic_title"):
        return str(meta.get("topic_title"))
    nested = meta.get("metadata") or {}
    topic = nested.get("topic") if isinstance(nested, dict) else {}
    return str(topic.get("title") or "") if isinstance(topic, dict) else ""


def fact_topic_summary(fact: ProfileFact) -> str:
    meta = _fact_meta(fact)
    if meta.get("topic_summary"):
        return str(meta.get("topic_summary"))
    nested = meta.get("metadata") or {}
    topic = nested.get("topic") if isinstance(nested, dict) else {}
    return str(topic.get("summary") or "") if isinstance(topic, dict) else ""


def fact_episode_ids(fact: ProfileFact) -> List[str]:
    meta = _fact_meta(fact)
    out = []
    if meta.get("episode_id"):
        out.append(str(meta.get("episode_id")))
    nested = meta.get("metadata") or {}
    if isinstance(nested, dict):
        out.extend(_clean_list(nested.get("all_episode_ids")))
        fact_row = nested.get("fact") if isinstance(nested.get("fact"), dict) else {}
        out.extend(_clean_list(fact_row.get("episode_ids")))
        episode = nested.get("episode") if isinstance(nested.get("episode"), dict) else {}
        if episode.get("episode_id"):
            out.append(str(episode.get("episode_id")))
    return list(dict.fromkeys(x for x in out if x)) or ["unknown_episode"]


def fact_episode_title(fact: ProfileFact) -> str:
    meta = _fact_meta(fact)
    if meta.get("episode_title"):
        return str(meta.get("episode_title"))
    nested = meta.get("metadata") or {}
    episode = nested.get("episode") if isinstance(nested, dict) else {}
    return str(episode.get("title") or "") if isinstance(episode, dict) else ""


def fact_episode_summary(fact: ProfileFact) -> str:
    meta = _fact_meta(fact)
    if meta.get("episode_summary"):
        return str(meta.get("episode_summary"))
    nested = meta.get("metadata") or {}
    episode = nested.get("episode") if isinstance(nested, dict) else {}
    return str(episode.get("summary") or episode.get("content") or "") if isinstance(episode, dict) else ""


def build_topic_episode_indices(memory: ProfileCentricHypergraphMemory) -> Tuple[Dict[str, TopicRecord], Dict[str, EpisodeRecord]]:
    topics: Dict[str, TopicRecord] = {}
    episodes: Dict[str, EpisodeRecord] = {}
    for fact in memory.facts.values():
        tid = fact_topic_id(fact)
        topic = topics.setdefault(
            tid,
            TopicRecord(topic_id=tid, title=fact_topic_title(fact), summary=fact_topic_summary(fact)),
        )
        if fact.fact_id not in topic.fact_ids:
            topic.fact_ids.append(fact.fact_id)
        for eid in fact_episode_ids(fact):
            ep = episodes.setdefault(
                eid,
                EpisodeRecord(
                    episode_id=eid,
                    topic_id=tid,
                    title=fact_episode_title(fact),
                    summary=fact_episode_summary(fact),
                    timestamp=fact.timestamp,
                ),
            )
            if fact.fact_id not in ep.fact_ids:
                ep.fact_ids.append(fact.fact_id)
            if eid not in topic.episode_ids:
                topic.episode_ids.append(eid)
    for topic in topics.values():
        sample_facts = " ".join(memory.facts[fid].content for fid in topic.fact_ids[:8] if fid in memory.facts)
        topic.embedding = memory.embedding_model.encode(" ".join([topic.text(), sample_facts]))
    for ep in episodes.values():
        sample_facts = " ".join(memory.facts[fid].content for fid in ep.fact_ids[:8] if fid in memory.facts)
        ep.embedding = memory.embedding_model.encode(" ".join([ep.text(), sample_facts]))
    return topics, episodes


def _pack_candidates(
    memory: ProfileCentricHypergraphMemory,
    ranked: Iterable[CandidateEvidence],
    *,
    top_k: int,
    max_tokens: int,
) -> List[ProfileFact]:
    selected: List[ProfileFact] = []
    seen = set()
    tokens = 0
    for cand in ranked:
        if cand.fact_id in seen or cand.score <= 0:
            continue
        fact = memory.facts.get(cand.fact_id)
        if fact is None:
            continue
        tok = estimate_tokens(fact.content)
        if selected and tokens + tok > max_tokens:
            continue
        selected.append(fact)
        seen.add(cand.fact_id)
        tokens += tok
        if len(selected) >= top_k or tokens >= max_tokens:
            break
    return selected


def retrieve_dual_path(
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
    profile_weight: float = 0.38,
    topic_weight: float = 0.32,
    episode_weight: float = 0.18,
    alignment_weight: float = 0.12,
) -> ProfileRetrievalResult:
    query_type, _, _ = memory.infer_profile_type(query)
    query_embedding = memory.embedding_model.encode(query)

    # Path A: behavioral profile hyperedge -> member facts -> source episodes.
    scored_edges: List[Tuple[ProfileHyperedgeUnit, float, Dict[str, float]]] = []
    for edge in memory.edges.values():
        score, parts = edge.score(
            query_embedding=query_embedding,
            query_type=query_type,
            facts=memory.facts,
            use_utility=use_utility,
            weights=memory.weights,
        )
        if score > 0:
            scored_edges.append((edge, score, parts))
    scored_edges.sort(key=lambda item: item[1], reverse=True)
    selected = scored_edges[:top_k_edges]
    selected_edges = [edge for edge, _, _ in selected]
    edge_score_map = {edge.edge_id: score for edge, score, _ in selected}
    edge_sim_map = {edge.edge_id: HashedEmbeddingModel.cosine(query_embedding, edge.embedding) for edge in selected_edges}
    edge_score = sum(edge_score_map.values()) / max(1, len(edge_score_map))

    candidates: Dict[str, CandidateEvidence] = {}
    profile_episode_ids = set()
    profile_topic_ids = set()
    for edge in selected_edges:
        edge_score_value = edge_score_map.get(edge.edge_id, 0.0)
        edge_sim = edge_sim_map.get(edge.edge_id, 0.0)
        for fid in edge.member_fact_ids:
            fact = memory.facts.get(fid)
            if fact is None:
                continue
            fact_sim = HashedEmbeddingModel.cosine(query_embedding, fact.embedding)
            score = 0.62 * fact_sim + 0.24 * edge_sim + 0.14 * edge_score_value
            cand = candidates.setdefault(fid, CandidateEvidence(fact_id=fid))
            cand.profile_score = max(cand.profile_score, score)
            cand.profile_edge_ids.append(edge.edge_id)
            cand.source_paths.append("profile")
            eps = fact_episode_ids(fact)
            tid = fact_topic_id(fact)
            cand.episode_ids.extend(eps)
            cand.topic_ids.append(tid)
            profile_episode_ids.update(eps)
            profile_topic_ids.add(tid)

    # Path B: topic -> episode -> facts.
    topics, episodes = build_topic_episode_indices(memory)
    scored_topics = [
        (topic, HashedEmbeddingModel.cosine(query_embedding, topic.embedding))
        for topic in topics.values()
    ]
    scored_topics.sort(key=lambda item: item[1], reverse=True)
    selected_topics = scored_topics[:top_k_topics]
    selected_topic_ids = {topic.topic_id for topic, _ in selected_topics}
    topic_score_map = {topic.topic_id: score for topic, score in selected_topics}

    candidate_episodes: List[Tuple[EpisodeRecord, float]] = []
    for topic, topic_score_value in selected_topics:
        for eid in topic.episode_ids:
            ep = episodes.get(eid)
            if ep is None:
                continue
            ep_sim = HashedEmbeddingModel.cosine(query_embedding, ep.embedding)
            score = 0.72 * ep_sim + 0.28 * topic_score_value
            candidate_episodes.append((ep, score))
    candidate_episodes.sort(key=lambda item: item[1], reverse=True)
    selected_episodes = candidate_episodes[:top_k_episodes]
    selected_episode_ids = {ep.episode_id for ep, _ in selected_episodes}
    episode_score_map = {ep.episode_id: score for ep, score in selected_episodes}

    for ep, ep_score in selected_episodes:
        topic_score_value = topic_score_map.get(ep.topic_id, 0.0)
        for fid in ep.fact_ids:
            fact = memory.facts.get(fid)
            if fact is None:
                continue
            fact_sim = HashedEmbeddingModel.cosine(query_embedding, fact.embedding)
            score = 0.60 * fact_sim + 0.25 * ep_score + 0.15 * topic_score_value
            cand = candidates.setdefault(fid, CandidateEvidence(fact_id=fid))
            cand.topic_score = max(cand.topic_score, topic_score_value)
            cand.episode_score = max(cand.episode_score, score)
            cand.episode_ids.append(ep.episode_id)
            cand.topic_ids.append(ep.topic_id)
            cand.source_paths.append("topic_episode")

    # Fusion with episode/topic alignment.
    for cand in candidates.values():
        cand.profile_edge_ids = list(dict.fromkeys(cand.profile_edge_ids))
        cand.topic_ids = list(dict.fromkeys(x for x in cand.topic_ids if x))
        cand.episode_ids = list(dict.fromkeys(x for x in cand.episode_ids if x))
        cand.source_paths = list(dict.fromkeys(cand.source_paths))
        episode_overlap = bool(set(cand.episode_ids) & profile_episode_ids & selected_episode_ids)
        topic_overlap = bool(set(cand.topic_ids) & profile_topic_ids & selected_topic_ids)
        dual_source = "profile" in cand.source_paths and "topic_episode" in cand.source_paths
        cand.alignment_score = (0.55 if episode_overlap else 0.0) + (0.25 if topic_overlap else 0.0) + (0.20 if dual_source else 0.0)
        cand.score = (
            profile_weight * cand.profile_score
            + topic_weight * cand.topic_score
            + episode_weight * cand.episode_score
            + alignment_weight * cand.alignment_score
        )

    ranked = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
    facts = _pack_candidates(memory, ranked, top_k=top_k_facts, max_tokens=max_tokens)
    fallback_used = False
    if (not facts or edge_score < sufficiency_threshold) and fallback:
        fallback_used = True
        facts = memory.global_fact_retrieval(query_embedding, top_k=top_k_facts, max_tokens=max_tokens)

    for edge in selected_edges:
        edge.access_count += 1

    fact_score_map = {cand.fact_id: cand for cand in ranked[: max(20, top_k_facts * 3)]}
    return ProfileRetrievalResult(
        query=query,
        channel="dual_path_profile_topic_episode_alignment" if not fallback_used else "dual_path_profile_topic_episode_alignment+fallback",
        selected_edges=selected_edges,
        selected_facts=facts,
        score=edge_score,
        tokens=estimate_tokens([fact.content for fact in facts]),
        fallback_used=fallback_used,
        sufficient=bool(facts),
        debug_scores=[
            {
                "path": "profile_edge",
                "edge_id": edge.edge_id,
                "edge_type": edge.edge_type.value,
                "score": round(score, 6),
                **parts,
                "summary": edge.summary,
                "members": len(edge.member_fact_ids),
            }
            for edge, score, parts in selected
        ]
        + [
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
            for ep, score in selected_episodes[:top_k_episodes]
        ]
        + [
            {
                "path": "fused_fact",
                "fact_id": fact.fact_id,
                "score": round(fact_score_map.get(fact.fact_id, CandidateEvidence(fact.fact_id)).score, 6),
                "profile_score": round(fact_score_map.get(fact.fact_id, CandidateEvidence(fact.fact_id)).profile_score, 6),
                "topic_score": round(fact_score_map.get(fact.fact_id, CandidateEvidence(fact.fact_id)).topic_score, 6),
                "episode_score": round(fact_score_map.get(fact.fact_id, CandidateEvidence(fact.fact_id)).episode_score, 6),
                "alignment_score": round(fact_score_map.get(fact.fact_id, CandidateEvidence(fact.fact_id)).alignment_score, 6),
            }
            for fact in facts
        ],
    )
