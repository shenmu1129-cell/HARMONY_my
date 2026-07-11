from hypermem.subquery_planner import (
    ROUTE_HIERARCHICAL,
    ROUTE_ROLE_SHORTCUT,
    ROUTE_ROLE_TEMPORAL,
    candidate_routes,
    decompose_query,
)


def test_atomic_question_stays_atomic():
    plan = decompose_query("What is my pet's name?")
    assert not plan.decomposed
    assert len(plan.subqueries) == 1
    assert plan.subqueries[0].intent == "atomic"


def test_temporal_causal_question_has_dependencies_and_role_routes():
    plan = decompose_query(
        "Alice moved before she changed her diet, and why did her food preference change later?"
    )
    assert plan.decomposed
    assert 2 <= len(plan.subqueries) <= 3
    assert any(relation["type"] == "dependent" for relation in plan.relations)
    routes = candidate_routes(plan.subqueries[0], has_role_hint=True)
    route_names = {route.route for route in routes}
    assert ROUTE_HIERARCHICAL in route_names
    assert ROUTE_ROLE_SHORTCUT in route_names or ROUTE_ROLE_TEMPORAL in route_names


def test_atomic_causal_question_keeps_a_hierarchical_fallback_route():
    plan = decompose_query("Why did Maria join the military?")
    assert not plan.decomposed
    assert plan.subqueries[0].intent == "atomic_causal"
    route_names = {route.route for route in candidate_routes(plan.subqueries[0], has_role_hint=False)}
    assert ROUTE_HIERARCHICAL in route_names
