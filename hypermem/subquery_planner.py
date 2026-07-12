"""Lightweight subquery planning for cost-aware long-term memory retrieval.

The planner is deliberately model-free in the first HARMONY implementation.
It turns clearly compositional questions into at most three information goals,
records the only two dependency relations that the scheduler needs, and exposes
a small route vocabulary.  Retrieval backends remain responsible for executing
the routes and for source-preserving evidence fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Sequence


ROUTE_GLOBAL_COMPACT = "global_compact"
ROUTE_ROLE_SHORTCUT = "role_shortcut"
ROUTE_HIERARCHICAL = "hierarchical"
ROUTE_ROLE_TEMPORAL = "role_temporal"


TEMPORAL_TERMS = (
    "before", "after", "later", "earlier", "then", "when", "timeline",
    "changed", "change", "previously", "formerly", "first", "last",
)
CAUSAL_TERMS = ("why", "because", "reason", "caused", "led to", "made")
COMPOSITION_TERMS = (" and ", " but ", " while ", " then ", ";", ":")
STRUCTURAL_COMPOSITION_RE = re.compile(
    r"\b(?:and|but|while)\s+(?:why|how|what|when|where|who|did|does|was|were|is|are|then|later)\b"
)


@dataclass(frozen=True)
class Subquery:
    """One bounded information goal in a retrieval plan."""

    subquery_id: str
    text: str
    intent: str
    depends_on: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    decomposed: bool
    confidence: float
    subqueries: List[Subquery]
    relations: List[Dict[str, str]]
    matched_signals: List[str]


@dataclass(frozen=True)
class RouteCandidate:
    subquery_id: str
    route: str
    expected_evidence: float
    token_cost: int
    search_cost: int


def _contains_any(query: str, terms: Iterable[str]) -> List[str]:
    lowered = query.lower()
    return [term for term in terms if term in lowered]


def infer_named_roles(query: str, available_roles: Sequence[str]) -> List[str]:
    """Return role labels explicitly named in the query, without guessing."""
    lowered = (query or "").lower()
    return [
        role for role in available_roles
        if role and re.search(rf"\b{re.escape(role.lower())}\b", lowered)
    ]


def decompose_query(query: str, max_subqueries: int = 3) -> QueryPlan:
    """Produce a conservative, deterministic plan for a user query.

    We only decompose when temporal/causal relations co-occur with a composition
    marker.  This preserves atomic fact-query latency and avoids inventing
    unsupported subquestions for short queries.
    """
    original = " ".join((query or "").split())
    lowered = original.lower()
    temporal = _contains_any(lowered, TEMPORAL_TERMS)
    causal = _contains_any(lowered, CAUSAL_TERMS)
    composition = _contains_any(lowered, COMPOSITION_TERMS)
    structural_composition = bool(STRUCTURAL_COMPOSITION_RE.search(lowered) or ";" in lowered)
    signals = [f"temporal:{item}" for item in temporal]
    signals += [f"causal:{item}" for item in causal]
    signals += [f"composition:{item.strip()}" for item in composition]

    should_decompose = bool(
        original
        and (
            (temporal and causal)
            # A structural marker such as "..., and how does ..." already
            # expresses two answerable information goals.  Requiring a
            # temporal or causal keyword here missed genuine multi-hop LoCoMo
            # questions that ask for a challenge and its response.
            or structural_composition
        )
        and len(original.split()) >= 7
    )
    if not should_decompose:
        intent = "atomic_causal" if causal else "atomic_temporal" if temporal else "atomic"
        return QueryPlan(
            original_query=original,
            decomposed=False,
            confidence=0.90 if original else 0.0,
            subqueries=[Subquery("q1", original, intent)],
            relations=[],
            matched_signals=signals,
        )

    tasks: List[Subquery] = []
    relations: List[Dict[str, str]] = []
    if temporal:
        tasks.append(Subquery(
            "q1",
            f"Find the time or event anchor needed to answer: {original}",
            "temporal_anchor",
        ))
    if causal:
        tasks.append(Subquery(
            f"q{len(tasks) + 1}",
            f"Find the cause, change, or explanatory evidence for: {original}",
            "causal_evidence",
            depends_on=["q1"] if tasks else [],
        ))
    if not tasks and structural_composition:
        match = STRUCTURAL_COMPOSITION_RE.search(lowered)
        first_goal = original[: match.start()].strip(" ,;:") if match else original
        tasks.append(Subquery(
            "q1",
            f"Find evidence for the first information goal: {first_goal}",
            "context_evidence",
        ))

    tasks.append(Subquery(
        f"q{len(tasks) + 1}",
        original,
        "answer_evidence",
        depends_on=[task.subquery_id for task in tasks],
    ))
    tasks = tasks[:max(1, max_subqueries)]
    valid_ids = {task.subquery_id for task in tasks}
    tasks = [
        Subquery(task.subquery_id, task.text, task.intent, [dep for dep in task.depends_on if dep in valid_ids])
        for task in tasks
    ]
    for task in tasks:
        for parent in task.depends_on:
            relations.append({"from": parent, "to": task.subquery_id, "type": "dependent"})
    for left in tasks:
        for right in tasks:
            if left.subquery_id < right.subquery_id and not left.depends_on and not right.depends_on:
                relations.append({"from": left.subquery_id, "to": right.subquery_id, "type": "parallel"})
    return QueryPlan(
        original_query=original,
        decomposed=True,
        confidence=min(0.95, 0.58 + 0.08 * len(signals)),
        subqueries=tasks,
        relations=relations,
        matched_signals=signals,
    )


def candidate_routes(
    task: Subquery,
    *,
    has_role_hint: bool,
    max_candidates: int = 3,
) -> List[RouteCandidate]:
    """Generate only plausible routes for a subquery, not the full cross product."""
    intent = task.intent
    # token_cost is the expected *incremental* evidence cost used by the
    # scheduler.  The backend's larger per-route cap is only a candidate pool;
    # final source evidence is packed once under the Reader budget.
    routes = [RouteCandidate(task.subquery_id, ROUTE_GLOBAL_COMPACT, 0.54, 300, 55)]
    if intent in {"temporal_anchor", "causal_evidence", "answer_evidence", "atomic_temporal", "atomic_causal"}:
        routes.append(RouteCandidate(task.subquery_id, ROUTE_HIERARCHICAL, 0.68, 420, 70))
    if has_role_hint:
        routes.append(RouteCandidate(task.subquery_id, ROUTE_ROLE_SHORTCUT, 0.72, 360, 70))
        if intent == "temporal_anchor":
            routes.append(RouteCandidate(task.subquery_id, ROUTE_ROLE_TEMPORAL, 0.82, 480, 90))
    routes.sort(key=lambda item: (-item.expected_evidence, item.token_cost, item.route))
    return routes[:max_candidates]


def lexical_overlap(left: str, right: str) -> float:
    left_terms = {term for term in re.findall(r"[a-z0-9]+", left.lower()) if len(term) > 2}
    right_terms = {term for term in re.findall(r"[a-z0-9]+", right.lower()) if len(term) > 2}
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)
