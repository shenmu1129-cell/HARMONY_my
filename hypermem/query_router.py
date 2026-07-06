"""Query routing for hybrid behavioral-profile memory.

The router separates three retrieval intents:
- behavioral: long-term preferences, habits, values, stable roles, recurring patterns;
- episodic: concrete details, dates, places, one-off events;
- mixed: queries that need both a long-term profile and specific evidence.

This module is intentionally lightweight and deterministic for experiments. It can
be replaced by an LLM router or a distilled classifier later without changing the
retrieval/evaluation interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class QueryRoute:
    route: str
    confidence: float
    matched_rules: List[str]
    update_behavioral_edges: bool
    use_behavioral_edges: bool
    use_episodic_fallback: bool


BEHAVIORAL_TERMS = [
    "usually", "often", "tend", "tends", "recurring", "habit", "routine", "pattern",
    "preference", "prefer", "like", "likes", "interest", "value", "values", "care about",
    "long-term", "support system", "personality", "goal", "goals", "aspiration", "identity",
    "what kind", "what are the user's", "what does the user", "how does the user usually",
]

EPISODIC_TERMS = [
    "when", "where", "who", "which", "what did", "what was", "last", "yesterday",
    "two weekends", "last week", "last month", "specific", "date", "place", "event",
    "agency", "photo", "picture", "trip", "weekend", "recently", "what happened",
]

MIXED_TERMS = [
    "relate", "related", "connection", "connect", "how does", "why", "based on", "evidence",
    "examples", "what shows", "what suggests", "tell me about",
]


def route_query(query: str) -> QueryRoute:
    q = (query or "").lower()
    behavioral_hits = [term for term in BEHAVIORAL_TERMS if term in q]
    episodic_hits = [term for term in EPISODIC_TERMS if term in q]
    mixed_hits = [term for term in MIXED_TERMS if term in q]

    # Mixed queries need both high-level profile patterns and concrete facts.
    if (behavioral_hits and episodic_hits) or (mixed_hits and (behavioral_hits or episodic_hits)):
        hits = mixed_hits + behavioral_hits + episodic_hits
        return QueryRoute(
            route="mixed",
            confidence=min(0.95, 0.55 + 0.06 * len(hits)),
            matched_rules=hits,
            update_behavioral_edges=True,
            use_behavioral_edges=True,
            use_episodic_fallback=True,
        )

    if behavioral_hits and len(behavioral_hits) >= len(episodic_hits):
        return QueryRoute(
            route="behavioral",
            confidence=min(0.95, 0.55 + 0.08 * len(behavioral_hits)),
            matched_rules=behavioral_hits,
            update_behavioral_edges=True,
            use_behavioral_edges=True,
            use_episodic_fallback=True,
        )

    if episodic_hits:
        return QueryRoute(
            route="episodic",
            confidence=min(0.95, 0.55 + 0.08 * len(episodic_hits)),
            matched_rules=episodic_hits,
            update_behavioral_edges=False,
            use_behavioral_edges=False,
            use_episodic_fallback=True,
        )

    # Conservative default: use both for retrieval, but do not punish behavioral
    # hyperedges too aggressively unless the query has profile-like evidence.
    return QueryRoute(
        route="mixed",
        confidence=0.40,
        matched_rules=[],
        update_behavioral_edges=True,
        use_behavioral_edges=True,
        use_episodic_fallback=True,
    )


def route_to_dict(route: QueryRoute) -> Dict[str, object]:
    return {
        "route": route.route,
        "confidence": round(route.confidence, 6),
        "matched_rules": route.matched_rules,
        "update_behavioral_edges": int(route.update_behavioral_edges),
        "use_behavioral_edges": int(route.use_behavioral_edges),
        "use_episodic_fallback": int(route.use_episodic_fallback),
    }
