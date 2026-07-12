"""Extractive summary hypernodes for Topic-Episode-Fact memory.

A summary hypernode is not a free-form generated summary. It is an extractive,
provenance-preserving compression unit built from existing memory/fact text.

Design constraints:
- Fact-level summary hypernodes are always allowed because facts are source-backed.
- Topic/Episode-level summary hypernodes are optional and are created only when
  they have member facts/source evidence.
- Summary text is assembled from member fact/source snippets, not hallucinated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from hypermem.profile_centric_hypergraph import (
    HashedEmbeddingModel,
    ProfileCentricHypergraphMemory,
    ProfileFact,
    ProfileRetrievalResult,
    estimate_tokens,
    keyword_overlap,
    tokenize,
)


@dataclass
class SummaryHypernode:
    summary_id: str
    level: str
    summary: str
    member_ids: List[str]
    source_fact_ids: List[str]
    source_row_ids: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)
    token_cost: int = 0
    coherence: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)


def _dedupe(items: Iterable[Any]) -> List[str]:
    return list(dict.fromkeys(str(x) for x in items if str(x).strip()))


def _clip_words(text: str, max_tokens: int) -> str:
    toks = tokenize(text)
    if len(toks) <= max_tokens:
        return str(text).strip()
    # Tokenizer loses punctuation, but this is acceptable for an extractive compact view.
    return " ".join(toks[:max_tokens]).strip()


def _source_ids(fact: ProfileFact) -> List[str]:
    meta = fact.metadata or {}
    out: List[str] = []
    out.extend(_as_list(meta.get("source_row_ids")))
    nested = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    if nested:
        out.extend(_as_list(nested.get("source_row_ids")))
        row = nested.get("fact") if isinstance(nested.get("fact"), dict) else {}
        out.extend(_as_list(row.get("source_row_ids")))
    return _dedupe(out)


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x) for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _fact_topic_id(fact: ProfileFact) -> str:
    meta = fact.metadata or {}
    if meta.get("topic_id"):
        return str(meta.get("topic_id"))
    nested = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    topic = nested.get("topic") if isinstance(nested.get("topic"), dict) else {}
    if topic.get("topic_id"):
        return str(topic.get("topic_id"))
    return "unknown_topic"


def _fact_episode_ids(fact: ProfileFact) -> List[str]:
    meta = fact.metadata or {}
    out: List[str] = []
    if meta.get("episode_id"):
        out.append(str(meta.get("episode_id")))
    nested = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    if nested:
        out.extend(_as_list(nested.get("all_episode_ids")))
        row = nested.get("fact") if isinstance(nested.get("fact"), dict) else {}
        out.extend(_as_list(row.get("episode_ids")))
        ep = nested.get("episode") if isinstance(nested.get("episode"), dict) else {}
        if ep.get("episode_id"):
            out.append(str(ep.get("episode_id")))
    return _dedupe(out) or ["unknown_episode"]


def _keywords(memory: ProfileCentricHypergraphMemory, facts: Sequence[ProfileFact], summary: str) -> List[str]:
    return memory.extract_keywords(" ".join([summary] + [f.content for f in facts]), max_keywords=12)


def _coherence(facts: Sequence[ProfileFact]) -> float:
    if len(facts) <= 1:
        return 0.55
    sims: List[float] = []
    for i, left in enumerate(facts):
        for right in facts[i + 1 : min(len(facts), i + 9)]:
            sims.append(HashedEmbeddingModel.cosine(left.embedding, right.embedding))
    return sum(sims) / len(sims) if sims else 0.55


def _centroid_embedding(memory: ProfileCentricHypergraphMemory, summary: str, facts: Sequence[ProfileFact]) -> List[float]:
    vectors = [f.embedding for f in facts if f.embedding]
    vectors.append(memory.embedding_model.encode(summary))
    dim = memory.embedding_model.dim
    out = [0.0] * dim
    for vec in vectors:
        for i, value in enumerate(vec[:dim]):
            out[i] += value
    norm = math.sqrt(sum(x * x for x in out))
    return [x / norm for x in out] if norm else out


def _extractive_summary(facts: Sequence[ProfileFact], *, max_snippets: int = 3, max_tokens: int = 48) -> str:
    """Build a compact summary strictly from member fact text."""
    if not facts:
        return ""
    # Prefer short, information-dense snippets; keep original content snippets.
    ranked = sorted(facts, key=lambda f: (estimate_tokens(f.content), f.fact_id))
    snippets: List[str] = []
    for fact in ranked[:max_snippets]:
        snippet = _clip_words(fact.content.strip(), max(8, max_tokens // max(1, max_snippets)))
        if snippet:
            snippets.append(snippet)
    text = " ; ".join(snippets)
    return _clip_words(text, max_tokens)


def _make_node(
    memory: ProfileCentricHypergraphMemory,
    *,
    summary_id: str,
    level: str,
    member_ids: Sequence[str],
    facts: Sequence[ProfileFact],
    provenance: Dict[str, Any],
    max_summary_tokens: int,
) -> SummaryHypernode | None:
    member_ids = _dedupe(member_ids)
    facts = [f for f in facts if f is not None and f.content.strip()]
    if not member_ids or not facts:
        return None
    summary = _extractive_summary(facts, max_tokens=max_summary_tokens)
    if not summary:
        return None
    source_fact_ids = _dedupe(f.fact_id for f in facts)
    source_row_ids = _dedupe(x for fact in facts for x in _source_ids(fact))
    return SummaryHypernode(
        summary_id=summary_id,
        level=level,
        summary=summary,
        member_ids=list(member_ids),
        source_fact_ids=source_fact_ids,
        source_row_ids=source_row_ids,
        keywords=_keywords(memory, facts, summary),
        embedding=_centroid_embedding(memory, summary, facts),
        token_cost=estimate_tokens(summary),
        coherence=round(_coherence(facts), 6),
        provenance={
            "mode": "extractive_from_member_facts",
            "no_llm_generation": True,
            "summary_is_not_free_form": True,
            **provenance,
        },
    )


def build_summary_hypernodes(
    memory: ProfileCentricHypergraphMemory,
    *,
    max_summary_tokens: int = 48,
    fact_group_size: int = 4,
    max_nodes_per_level: int = 200,
) -> List[SummaryHypernode]:
    """Build extractive summary hypernodes from an already materialized graph."""
    nodes: List[SummaryHypernode] = []
    facts = list(memory.facts.values())

    # Fact-level summaries are guaranteed because every fact is source-backed.
    # Group by topic/episode where possible, then split into small chunks.
    grouped: Dict[Tuple[str, str], List[ProfileFact]] = {}
    for fact in facts:
        topic_id = _fact_topic_id(fact)
        ep_ids = _fact_episode_ids(fact)
        ep_id = ep_ids[0] if ep_ids else "unknown_episode"
        grouped.setdefault((topic_id, ep_id), []).append(fact)
    fact_node_count = 0
    for (topic_id, ep_id), group in sorted(grouped.items(), key=lambda item: item[0]):
        group = sorted(group, key=lambda f: (f.timestamp, f.fact_id))
        for start in range(0, len(group), max(1, fact_group_size)):
            chunk = group[start : start + max(1, fact_group_size)]
            if not chunk:
                continue
            fact_node_count += 1
            node = _make_node(
                memory,
                summary_id=f"summary_fact_{fact_node_count:06d}",
                level="fact",
                member_ids=[f.fact_id for f in chunk],
                facts=chunk,
                provenance={"topic_id": topic_id, "episode_id": ep_id, "chunk_start": start},
                max_summary_tokens=max_summary_tokens,
            )
            if node:
                nodes.append(node)
            if fact_node_count >= max_nodes_per_level:
                break
        if fact_node_count >= max_nodes_per_level:
            break

    # Episode-level summary hypernodes are optional: only if an episode has facts.
    by_episode: Dict[str, List[ProfileFact]] = {}
    for fact in facts:
        for eid in _fact_episode_ids(fact):
            by_episode.setdefault(eid, []).append(fact)
    ep_count = 0
    for eid, group in sorted(by_episode.items()):
        if eid == "unknown_episode" or not group:
            continue
        ep_count += 1
        node = _make_node(
            memory,
            summary_id=f"summary_episode_{ep_count:06d}",
            level="episode",
            member_ids=[eid],
            facts=group[: max(1, fact_group_size * 2)],
            provenance={"episode_id": eid},
            max_summary_tokens=max_summary_tokens,
        )
        if node:
            nodes.append(node)
        if ep_count >= max_nodes_per_level:
            break

    # Topic-level summary hypernodes are optional: only if a topic has facts.
    by_topic: Dict[str, List[ProfileFact]] = {}
    for fact in facts:
        by_topic.setdefault(_fact_topic_id(fact), []).append(fact)
    topic_count = 0
    for tid, group in sorted(by_topic.items()):
        if tid == "unknown_topic" or not group:
            continue
        topic_count += 1
        node = _make_node(
            memory,
            summary_id=f"summary_topic_{topic_count:06d}",
            level="topic",
            member_ids=[tid],
            facts=group[: max(1, fact_group_size * 3)],
            provenance={"topic_id": tid},
            max_summary_tokens=max_summary_tokens,
        )
        if node:
            nodes.append(node)
        if topic_count >= max_nodes_per_level:
            break

    return nodes


def save_summary_hypernodes(nodes: Sequence[SummaryHypernode], path: str | Path) -> None:
    data = {
        "schema": "extractive_summary_hypernodes_v1",
        "description": "Summary hypernodes are extractive and provenance-preserving; they are assembled from member fact/source text.",
        "num_nodes": len(nodes),
        "nodes": [asdict(node) for node in nodes],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_summary_hypernodes(path: str | Path) -> List[SummaryHypernode]:
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [SummaryHypernode(**row) for row in data.get("nodes", [])]


def _pack_fact_ids(
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


def retrieve_summary_hypernodes(
    memory: ProfileCentricHypergraphMemory,
    query: str,
    nodes: Sequence[SummaryHypernode],
    *,
    mode: str = "summary_first",
    top_k_summaries: int = 3,
    top_k_facts: int = 6,
    max_tokens: int = 180,
    expand_ratio: float = 0.45,
) -> ProfileRetrievalResult:
    query_emb = memory.embedding_model.encode(query)
    scored: List[Tuple[SummaryHypernode, float]] = []
    for node in nodes:
        sim = HashedEmbeddingModel.cosine(query_emb, node.embedding)
        lex = keyword_overlap(query, " ".join([node.summary] + node.keywords))
        cost_penalty = min(1.0, node.token_cost / max(1.0, float(max_tokens)))
        level_bonus = {"fact": 0.04, "episode": 0.02, "topic": 0.01}.get(node.level, 0.0)
        score = 0.70 * sim + 0.20 * lex + 0.06 * node.coherence + level_bonus - 0.05 * cost_penalty
        if score > 0:
            scored.append((node, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    selected = scored[:top_k_summaries]

    synthetic: List[ProfileFact] = []
    summary_tokens = 0
    for node, score in selected:
        if summary_tokens + node.token_cost > max_tokens and synthetic:
            continue
        synthetic.append(
            ProfileFact(
                fact_id=node.summary_id,
                content=node.summary,
                keywords=node.keywords,
                timestamp=0.0,
                embedding=node.embedding,
                metadata={
                    "source_type": "summary_hypernode",
                    "summary_level": node.level,
                    "member_ids": node.member_ids,
                    "source_fact_ids": node.source_fact_ids,
                    "source_row_ids": node.source_row_ids,
                    "provenance": node.provenance,
                    "score": score,
                },
            )
        )
        summary_tokens += node.token_cost

    selected_facts = list(synthetic)
    expanded_facts: List[ProfileFact] = []
    if mode in {"summary_adaptive", "summary_expand"}:
        fact_budget = max(1, int(top_k_facts * expand_ratio))
        token_budget = max(16, max_tokens - estimate_tokens([f.content for f in selected_facts]))
        candidates: Dict[str, float] = {}
        for node, node_score in selected:
            for fid in node.source_fact_ids:
                fact = memory.facts.get(fid)
                if fact is None:
                    continue
                fact_sim = HashedEmbeddingModel.cosine(query_emb, fact.embedding)
                candidates[fid] = max(candidates.get(fid, 0.0), 0.72 * fact_sim + 0.28 * node_score)
        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        expanded_facts = _pack_fact_ids(memory, ranked, top_k=fact_budget, max_tokens=token_budget)
        selected_facts.extend(expanded_facts)

    debug = [
        {
            "path": "summary_hypernode_controller",
            "mode": mode,
            "candidate_summaries": len(nodes),
            "selected_summaries": len(selected),
            "expanded_facts": len(expanded_facts),
            "summary_tokens": estimate_tokens([f.content for f in synthetic]),
            "max_tokens": max_tokens,
            "extractive_only": True,
        }
    ]
    debug.extend(
        {
            "path": "summary_hypernode",
            "summary_id": node.summary_id,
            "level": node.level,
            "score": round(score, 6),
            "members": len(node.member_ids),
            "source_facts": len(node.source_fact_ids),
            "tokens": node.token_cost,
            "provenance": node.provenance,
        }
        for node, score in selected
    )
    return ProfileRetrievalResult(
        query=query,
        channel=mode,
        selected_edges=[],
        selected_facts=selected_facts,
        score=sum(score for _, score in selected) / max(1, len(selected)),
        tokens=estimate_tokens([fact.content for fact in selected_facts]),
        fallback_used=False,
        sufficient=bool(selected_facts),
        debug_scores=debug,
    )
