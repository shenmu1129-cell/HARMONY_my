"""Summary-gated fact retrieval.

Summary hypernodes are used only as a candidate gate. Their text is not returned
as final evidence. Final evidence is selected from source facts under the top
summary hypernodes, which keeps provenance and avoids summary+fact token bloat.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from hypermem.profile_centric_hypergraph import (
    HashedEmbeddingModel,
    ProfileCentricHypergraphMemory,
    ProfileFact,
    ProfileRetrievalResult,
    estimate_tokens,
    keyword_overlap,
)
from hypermem.summary_hypernodes import SummaryHypernode


def _pack(memory: ProfileCentricHypergraphMemory, ranked: Sequence[Tuple[str, float]], *, top_k: int, max_tokens: int) -> List[ProfileFact]:
    facts: List[ProfileFact] = []
    seen = set()
    tokens = 0
    for fid, score in ranked:
        if fid in seen or score <= 0:
            continue
        fact = memory.facts.get(fid)
        if fact is None:
            continue
        cost = estimate_tokens(fact.content)
        if facts and tokens + cost > max_tokens:
            continue
        facts.append(fact)
        seen.add(fid)
        tokens += cost
        if len(facts) >= top_k or tokens >= max_tokens:
            break
    return facts


def retrieve_summary_gate(
    memory: ProfileCentricHypergraphMemory,
    query: str,
    summaries: Sequence[SummaryHypernode],
    *,
    top_k_summaries: int = 2,
    top_k_facts: int = 4,
    max_tokens: int = 110,
    include_one_summary_hint: bool = False,
) -> ProfileRetrievalResult:
    qemb = memory.embedding_model.encode(query)
    scored_summaries: List[Tuple[SummaryHypernode, float]] = []
    for node in summaries:
        sim = HashedEmbeddingModel.cosine(qemb, node.embedding)
        lex = keyword_overlap(query, " ".join([node.summary] + node.keywords))
        cost_penalty = min(1.0, node.token_cost / max(1, max_tokens))
        level_bonus = {"fact": 0.04, "episode": 0.02, "topic": 0.01}.get(node.level, 0.0)
        score = 0.72 * sim + 0.22 * lex + level_bonus - 0.06 * cost_penalty
        if score > 0:
            scored_summaries.append((node, score))
    scored_summaries.sort(key=lambda x: x[1], reverse=True)
    selected = scored_summaries[:top_k_summaries]

    candidates: Dict[str, float] = {}
    for node, sscore in selected:
        for fid in node.source_fact_ids:
            fact = memory.facts.get(fid)
            if fact is None:
                continue
            fsim = HashedEmbeddingModel.cosine(qemb, fact.embedding)
            flex = keyword_overlap(query, fact.content)
            score = 0.64 * fsim + 0.20 * flex + 0.16 * sscore
            candidates[fid] = max(candidates.get(fid, 0.0), score)
    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    facts = _pack(memory, ranked, top_k=top_k_facts, max_tokens=max_tokens)

    hint_facts: List[ProfileFact] = []
    if include_one_summary_hint and selected:
        node, score = selected[0]
        remaining = max(1, max_tokens - estimate_tokens([f.content for f in facts]))
        if node.token_cost <= remaining:
            hint_facts.append(ProfileFact(
                fact_id=node.summary_id,
                content=node.summary,
                keywords=node.keywords,
                timestamp=0.0,
                embedding=node.embedding,
                metadata={"source_type": "summary_gate_hint", "source_fact_ids": node.source_fact_ids, "score": score},
            ))
    final_facts = hint_facts + facts
    return ProfileRetrievalResult(
        query=query,
        channel="summary_gate_fact_only" if not include_one_summary_hint else "summary_gate_with_hint",
        selected_edges=[],
        selected_facts=final_facts,
        score=sum(s for _, s in selected) / max(1, len(selected)),
        tokens=estimate_tokens([f.content for f in final_facts]),
        fallback_used=False,
        sufficient=bool(final_facts),
        debug_scores=[{
            "path": "summary_gate",
            "candidate_summaries": len(summaries),
            "selected_summaries": len(selected),
            "candidate_facts": len(candidates),
            "expanded_facts": len(facts),
            "summary_as_final_evidence": bool(include_one_summary_hint),
            "token_budget": max_tokens,
        }] + [
            {"path": "summary_gate_node", "summary_id": n.summary_id, "level": n.level, "score": round(s, 6), "source_facts": len(n.source_fact_ids), "tokens": n.token_cost}
            for n, s in selected
        ],
    )
