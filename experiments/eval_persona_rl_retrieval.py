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
from typing import Any, Dict, List, Sequence, Tuple

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
from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory  # noqa: E402
from hypermem.query_router import route_query, route_to_dict  # noqa: E402

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


@dataclass
class Arm:
    name: str
    cfg: MethodConfig | None
    reference: bool = False


def make_core_arms() -> List[Arm]:
    return [
        Arm("adaptive_tiny_ref", None, reference=True),
        Arm("hybrid_roi_light_k4", MethodConfig("hybrid_roi_light_k4", response_boost=0.12, persona_boost=0.14, graph_gate="hybrid", top_k_facts=4, max_tokens=140)),
        Arm("hybrid_roi_mid_k4", MethodConfig("hybrid_roi_mid_k4", response_boost=0.28, persona_boost=0.14, graph_gate="hybrid", top_k_facts=4, max_tokens=140)),
        Arm("hybrid_roi_light_k3", MethodConfig("hybrid_roi_light_k3", response_boost=0.12, persona_boost=0.14, graph_gate="hybrid", top_k_facts=3, max_tokens=110)),
        Arm("hybrid_roi_light_k5", MethodConfig("hybrid_roi_light_k5", response_boost=0.12, persona_boost=0.14, graph_gate="hybrid", top_k_facts=5, max_tokens=160)),
        Arm("global_response_k4", MethodConfig("global_response_k4", response_boost=0.40, persona_boost=0.18, graph_gate="global_response", top_k_facts=4, max_tokens=90)),
        Arm("global_response_k2", MethodConfig("global_response_k2", response_boost=0.40, persona_boost=0.18, graph_gate="global_response", top_k_facts=2, max_tokens=70)),
        Arm("edge_source_response", MethodConfig("edge_source_response", response_boost=0.30, persona_boost=0.12, graph_gate="edge", top_k_edges=3, top_k_facts=4, max_tokens=140)),
        Arm("topic_source_response", MethodConfig("topic_source_response", response_boost=0.30, persona_boost=0.12, graph_gate="topic", top_k_topics=3, top_k_episodes=6, top_k_facts=4, max_tokens=140)),
    ]


def make_dynamic_k_arms() -> List[Arm]:
    arms: List[Arm] = []
    for k, mtok in [(2, 70), (3, 110), (4, 140), (5, 160), (6, 180)]:
        arms.append(Arm(f"dyn_hybrid_k{k}", MethodConfig(f"dyn_hybrid_k{k}", response_boost=0.12, persona_boost=0.14, graph_gate="hybrid", top_k_facts=k, max_tokens=mtok)))
    return arms


def make_llm_pruned_arms() -> List[Arm]:
    return [
        Arm("hybrid_roi_light_k3", MethodConfig("hybrid_roi_light_k3", response_boost=0.12, persona_boost=0.14, graph_gate="hybrid", top_k_facts=3, max_tokens=110)),
        Arm("hybrid_roi_light_k4", MethodConfig("hybrid_roi_light_k4", response_boost=0.12, persona_boost=0.14, graph_gate="hybrid", top_k_facts=4, max_tokens=140)),
        Arm("edge_source_response", MethodConfig("edge_source_response", response_boost=0.30, persona_boost=0.12, graph_gate="edge", top_k_edges=3, top_k_facts=4, max_tokens=140)),
        Arm("global_response_k2", MethodConfig("global_response_k2", response_boost=0.40, persona_boost=0.18, graph_gate="global_response", top_k_facts=2, max_tokens=70)),
        Arm("global_response_k4", MethodConfig("global_response_k4", response_boost=0.40, persona_boost=0.18, graph_gate="global_response", top_k_facts=4, max_tokens=90)),
        Arm("adaptive_tiny_ref", None, reference=True),
    ]


def make_fast_pruned_arms() -> List[Arm]:
    return [
        Arm("hybrid_roi_light_k3", MethodConfig("hybrid_roi_light_k3", response_boost=0.12, persona_boost=0.14, graph_gate="hybrid", top_k_facts=3, max_tokens=110)),
        Arm("edge_source_response", MethodConfig("edge_source_response", response_boost=0.30, persona_boost=0.12, graph_gate="edge", top_k_edges=3, top_k_facts=4, max_tokens=140)),
        Arm("global_response_k2", MethodConfig("global_response_k2", response_boost=0.40, persona_boost=0.18, graph_gate="global_response", top_k_facts=2, max_tokens=70)),
        Arm("adaptive_tiny_ref", None, reference=True),
    ]


def make_arms(arm_set: str) -> List[Arm]:
    if arm_set == "core":
        return make_core_arms()
    if arm_set == "dynamic_k":
        return make_dynamic_k_arms()
    if arm_set == "llm_pruned":
        return make_llm_pruned_arms()
    if arm_set == "fast_pruned":
        return make_fast_pruned_arms()
    raise ValueError(f"unknown arm_set={arm_set}")


def context_features(question: str) -> List[float]:
    route = route_query(question)
    q = question.lower()
    toks = q.split()
    return [
        1.0,
        1.0 if route.route == "mixed" else 0.0,
        1.0 if route.route == "behavioral" else 0.0,
        1.0 if route.route == "episodic" else 0.0,
        1.0 if "next response" in q or "given the persona" in q else 0.0,
        1.0 if "persona/profile" in q or "profile information" in q else 0.0,
        min(1.0, len(toks) / 80.0),
        min(1.0, route.confidence),
    ]


def retrieve_arm(memory, source_rows, question: str, arm: Arm):
    if arm.reference:
        return retrieve_reference(memory, question)
    assert arm.cfg is not None
    return retrieve_advanced(memory, source_rows, question, arm.cfg)


def reward01(reward: float) -> float:
    return max(0.0, min(1.0, (reward + 1.0) / 2.0))


class Policy:
    def select(self, question: str, route: str, step: int, train: bool) -> int:
        raise NotImplementedError

    def update(self, arm_idx: int, question: str, route: str, reward: float, hit: int) -> None:
        raise NotImplementedError

    def state(self) -> Dict[str, Any]:
        return {}


class EpsilonGreedyPolicy(Policy):
    def __init__(self, n_arms: int, seed: int = 7, eps0: float = 0.28) -> None:
        self.n = n_arms
        self.rng = random.Random(seed)
        self.eps0 = eps0
        self.counts: Dict[str, List[int]] = {}
        self.rewards: Dict[str, List[float]] = {}

    def _ensure(self, route: str) -> None:
        self.counts.setdefault(route, [0] * self.n)
        self.rewards.setdefault(route, [0.0] * self.n)

    def select(self, question: str, route: str, step: int, train: bool) -> int:
        self._ensure(route)
        eps = self.eps0 / math.sqrt(max(1, step))
        if train and self.rng.random() < eps:
            return self.rng.randrange(self.n)
        means = [self.rewards[route][i] / max(1, self.counts[route][i]) for i in range(self.n)]
        return max(range(self.n), key=lambda i: (means[i], -i))

    def update(self, arm_idx: int, question: str, route: str, reward: float, hit: int) -> None:
        self._ensure(route)
        self.counts[route][arm_idx] += 1
        self.rewards[route][arm_idx] += reward

    def state(self) -> Dict[str, Any]:
        return {"counts": self.counts, "rewards": self.rewards}


class UCBPolicy(EpsilonGreedyPolicy):
    def __init__(self, n_arms: int, c: float = 0.45, seed: int = 7) -> None:
        super().__init__(n_arms, seed=seed)
        self.c = c

    def select(self, question: str, route: str, step: int, train: bool) -> int:
        self._ensure(route)
        for i, c in enumerate(self.counts[route]):
            if c == 0:
                return i
        total = sum(self.counts[route]) + 1
        scores = []
        for i in range(self.n):
            mean = self.rewards[route][i] / max(1, self.counts[route][i])
            bonus = self.c * math.sqrt(math.log(total + 1) / max(1, self.counts[route][i]))
            scores.append(mean + bonus)
        return max(range(self.n), key=lambda i: (scores[i], -i))


class ThompsonPolicy(Policy):
    def __init__(self, n_arms: int, seed: int = 11) -> None:
        self.n = n_arms
        self.rng = random.Random(seed)
        self.alpha: Dict[str, List[float]] = {}
        self.beta: Dict[str, List[float]] = {}

    def _ensure(self, route: str) -> None:
        self.alpha.setdefault(route, [1.0] * self.n)
        self.beta.setdefault(route, [1.0] * self.n)

    def select(self, question: str, route: str, step: int, train: bool) -> int:
        self._ensure(route)
        if not train:
            means = [self.alpha[route][i] / (self.alpha[route][i] + self.beta[route][i]) for i in range(self.n)]
            return max(range(self.n), key=lambda i: (means[i], -i))
        samples = [self.rng.betavariate(self.alpha[route][i], self.beta[route][i]) for i in range(self.n)]
        return max(range(self.n), key=lambda i: (samples[i], -i))

    def update(self, arm_idx: int, question: str, route: str, reward: float, hit: int) -> None:
        self._ensure(route)
        self.alpha[route][arm_idx] += reward01(reward)
        self.beta[route][arm_idx] += 1.0 - reward01(reward)

    def state(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "beta": self.beta}


class EXP3Policy(Policy):
    def __init__(self, n_arms: int, gamma: float = 0.07, seed: int = 13) -> None:
        self.n = n_arms
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.weights: Dict[str, List[float]] = {}
        self.last_probs: Dict[str, List[float]] = {}

    def _ensure(self, route: str) -> None:
        self.weights.setdefault(route, [1.0] * self.n)

    def _probs(self, route: str) -> List[float]:
        weights = self.weights[route]
        total = sum(weights)
        return [(1 - self.gamma) * w / total + self.gamma / self.n for w in weights]

    def select(self, question: str, route: str, step: int, train: bool) -> int:
        self._ensure(route)
        probs = self._probs(route)
        self.last_probs[route] = probs
        if not train:
            return max(range(self.n), key=lambda i: (probs[i], -i))
        x = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if x <= acc:
                return i
        return self.n - 1

    def update(self, arm_idx: int, question: str, route: str, reward: float, hit: int) -> None:
        self._ensure(route)
        probs = self.last_probs.get(route) or self._probs(route)
        xhat = reward01(reward) / max(1e-6, probs[arm_idx])
        self.weights[route][arm_idx] *= math.exp(self.gamma * xhat / self.n)

    def state(self) -> Dict[str, Any]:
        return {"weights": self.weights}


class SoftmaxPGPolicy(EpsilonGreedyPolicy):
    def __init__(self, n_arms: int, lr: float = 0.12, seed: int = 17) -> None:
        super().__init__(n_arms, seed=seed)
        self.lr = lr
        self.pref: Dict[str, List[float]] = {}
        self.baseline: Dict[str, float] = {}
        self.last_probs: Dict[str, List[float]] = {}
        self.last_arm: Dict[str, int] = {}

    def _ensure(self, route: str) -> None:
        super()._ensure(route)
        self.pref.setdefault(route, [0.0] * self.n)
        self.baseline.setdefault(route, 0.5)

    def _probs(self, route: str) -> List[float]:
        vals = self.pref[route]
        m = max(vals)
        exps = [math.exp(v - m) for v in vals]
        s = sum(exps)
        return [e / s for e in exps]

    def select(self, question: str, route: str, step: int, train: bool) -> int:
        self._ensure(route)
        probs = self._probs(route)
        self.last_probs[route] = probs
        if not train:
            return max(range(self.n), key=lambda i: (probs[i], -i))
        x = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if x <= acc:
                self.last_arm[route] = i
                return i
        self.last_arm[route] = self.n - 1
        return self.n - 1

    def update(self, arm_idx: int, question: str, route: str, reward: float, hit: int) -> None:
        self._ensure(route)
        r = reward01(reward)
        adv = r - self.baseline[route]
        self.baseline[route] = 0.95 * self.baseline[route] + 0.05 * r
        probs = self.last_probs.get(route) or self._probs(route)
        for i in range(self.n):
            grad = (1.0 if i == arm_idx else 0.0) - probs[i]
            self.pref[route][i] += self.lr * adv * grad
        self.counts[route][arm_idx] += 1
        self.rewards[route][arm_idx] += reward

    def state(self) -> Dict[str, Any]:
        return {"pref": self.pref, "baseline": self.baseline, "counts": self.counts, "rewards": self.rewards}


class LinUCBPolicy(Policy):
    def __init__(self, n_arms: int, alpha: float = 0.55) -> None:
        if np is None:
            raise RuntimeError("numpy is required for LinUCBPolicy")
        self.n = n_arms
        self.alpha = alpha
        self.d = 8
        self.A = [np.eye(self.d) for _ in range(n_arms)]
        self.b = [np.zeros((self.d, 1)) for _ in range(n_arms)]

    def select(self, question: str, route: str, step: int, train: bool) -> int:
        x = np.array(context_features(question)).reshape((self.d, 1))
        vals = []
        for i in range(self.n):
            inv = np.linalg.inv(self.A[i])
            theta = inv @ self.b[i]
            mean = float((theta.T @ x).item())
            uncertainty = float((x.T @ inv @ x).item())
            p = mean + self.alpha * math.sqrt(max(0.0, uncertainty))
            vals.append(p)
        return max(range(self.n), key=lambda i: (vals[i], -i))

    def update(self, arm_idx: int, question: str, route: str, reward: float, hit: int) -> None:
        x = np.array(context_features(question)).reshape((self.d, 1))
        self.A[arm_idx] += x @ x.T
        self.b[arm_idx] += reward01(reward) * x

    def state(self) -> Dict[str, Any]:
        return {"alpha": self.alpha}


def policy_factory(
    policy_name: str,
    n_arms: int,
    *,
    seed: int = 7,
    eps0: float = 0.28,
    ucb_c: float = 0.45,
    exp3_gamma: float = 0.07,
    linucb_alpha: float = 0.55,
) -> Policy:
    if policy_name == "epsilon_greedy":
        return EpsilonGreedyPolicy(n_arms, seed=seed, eps0=eps0)
    if policy_name == "ucb1":
        return UCBPolicy(n_arms, c=ucb_c, seed=seed)
    if policy_name == "thompson":
        return ThompsonPolicy(n_arms, seed=seed)
    if policy_name == "exp3":
        return EXP3Policy(n_arms, gamma=exp3_gamma, seed=seed)
    if policy_name == "softmax_pg":
        return SoftmaxPGPolicy(n_arms, seed=seed)
    if policy_name == "linucb":
        return LinUCBPolicy(n_arms, alpha=linucb_alpha)
    if policy_name == "dynamic_k_ucb":
        return UCBPolicy(n_arms, c=ucb_c, seed=seed)
    raise ValueError(policy_name)


def run_policy(
    policy_name: str,
    graph_path: str,
    memory_json: str,
    questions_json: str,
    max_questions: int,
    split_train: int,
    arm_set: str = "core",
    seed: int = 7,
    eps0: float = 0.28,
    ucb_c: float = 0.45,
    exp3_gamma: float = 0.07,
    linucb_alpha: float = 0.55,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    memory = ProfileCentricHypergraphMemory.load(graph_path)
    source_rows = row_id_map(read_jsonl(Path(memory_json)), memory)
    questions = load_questions(Path(questions_json), max_questions=max_questions)
    arms = make_dynamic_k_arms() if policy_name == "dynamic_k_ucb" and arm_set == "core" else make_arms(arm_set)
    policy = policy_factory(
        policy_name,
        len(arms),
        seed=seed,
        eps0=eps0,
        ucb_c=ucb_c,
        exp3_gamma=exp3_gamma,
        linucb_alpha=linucb_alpha,
    )
    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    for idx, q in enumerate(questions, start=1):
        route = route_query(q["question"])
        train = idx <= split_train if split_train > 0 else True
        arm_idx = policy.select(q["question"], route.route, idx, train=train)
        arm = arms[arm_idx]
        t0 = time.time()
        result = retrieve_arm(memory, source_rows, q["question"], arm)
        method_label = f"{policy_name}_{arm_set}_s{seed}"
        row, reward, hit, _ = base_eval.row_from_result(method_label, q, result, update_used=train)
        if train:
            policy.update(arm_idx, q["question"], route.route, reward, hit)
        row.update(route_to_dict(route))
        row["chosen_arm"] = arm.name
        row["phase"] = "train" if train and split_train > 0 else "test" if split_train > 0 else "online"
        row["retrieval_ms"] = round((time.time() - t0) * 1000.0, 4)
        row["candidate_facts"] = result.debug_scores[0].get("candidate_facts", len(result.selected_facts)) if result.debug_scores else len(result.selected_facts)
        rows.append(row)
        traces.append({**row, "gold": q["gold"], "evidence": [f.content for f in result.selected_facts], "debug": result.debug_scores})
    state = {
        "policy": policy_name,
        "arm_set": arm_set,
        "seed": seed,
        "arms": [arm.name for arm in arms],
        "state": policy.state(),
        "split_train": split_train,
    }
    return f"{policy_name}_{arm_set}_s{seed}", rows, traces, state


def run_oracle(
    graph_path: str,
    memory_json: str,
    questions_json: str,
    max_questions: int,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    memory = ProfileCentricHypergraphMemory.load(graph_path)
    source_rows = row_id_map(read_jsonl(Path(memory_json)), memory)
    questions = load_questions(Path(questions_json), max_questions=max_questions)
    arms = make_core_arms()
    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    for q in questions:
        best = None
        best_tuple = (-999.0, -999.0)
        best_arm = None
        for arm in arms:
            result = retrieve_arm(memory, source_rows, q["question"], arm)
            row, reward, hit, rec = base_eval.row_from_result("oracle_arm_upper_bound", q, result, update_used=False)
            score = (hit, reward)
            if score > best_tuple:
                best_tuple = score
                best = (result, row, reward, hit)
                best_arm = arm
        assert best is not None and best_arm is not None
        result, row, _, _ = best
        row["chosen_arm"] = best_arm.name
        row["phase"] = "oracle"
        row["retrieval_ms"] = 0.0
        row["candidate_facts"] = result.debug_scores[0].get("candidate_facts", len(result.selected_facts)) if result.debug_scores else len(result.selected_facts)
        rows.append(row)
        traces.append({**row, "gold": q["gold"], "evidence": [f.content for f in result.selected_facts], "debug": result.debug_scores})
    return "oracle_arm_upper_bound", rows, traces, {"arms": [a.name for a in arms]}


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


def summarize_by_phase(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], str(row.get("phase", "all"))), []).append(row)
    out = []
    for (method, phase), part in sorted(grouped.items()):
        out.append({"method": method, "phase": phase, **summarize(part)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-graph", required=True)
    parser.add_argument("--memory-json", required=True)
    parser.add_argument("--questions-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-questions", type=int, default=1000)
    parser.add_argument("--split-train", type=int, default=500)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--policies", default="epsilon_greedy,ucb1,thompson,exp3,softmax_pg,linucb,dynamic_k_ucb")
    parser.add_argument("--arm-set", default="core", choices=["core", "dynamic_k", "llm_pruned", "fast_pruned"])
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--eps0", type=float, default=0.28)
    parser.add_argument("--ucb-c", type=float, default=0.45)
    parser.add_argument("--exp3-gamma", type=float, default=0.07)
    parser.add_argument("--linucb-alpha", type=float, default=0.55)
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    jobs = []
    for name in policies:
        for seed in seeds:
            jobs.append((name, args.split_train, seed))
            jobs.append((name + "_online", 0, seed))

    all_rows: List[Dict[str, Any]] = []
    all_traces: List[Dict[str, Any]] = []
    states: Dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = []
        for name, split, seed in jobs:
            base_name = name.replace("_online", "")
            futs.append(pool.submit(
                run_policy,
                base_name,
                args.memory_graph,
                args.memory_json,
                args.questions_json,
                args.max_questions,
                split,
                args.arm_set,
                seed,
                args.eps0,
                args.ucb_c,
                args.exp3_gamma,
                args.linucb_alpha,
            ))
        if args.oracle:
            futs.append(pool.submit(run_oracle, args.memory_graph, args.memory_json, args.questions_json, args.max_questions))
        for fut in as_completed(futs):
            name, rows, traces, state = fut.result()
            label = name if not rows else f"{name}:{rows[0].get('phase')}"
            print(f"[done] {label} n={len(rows)}", flush=True)
            all_rows.extend(rows)
            all_traces.extend(traces)
            states[label] = state

    summary = summarize_by_phase(all_rows)
    write_rows(out_dir / "rl_results.csv", all_rows)
    write_rows(out_dir / "rl_summary.csv", summary)
    with (out_dir / "rl_trace.jsonl").open("w", encoding="utf-8") as f:
        for row in all_traces:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "rl_policy_states.json").write_text(json.dumps(states, ensure_ascii=False, indent=2), encoding="utf-8")
    print((out_dir / "rl_summary.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
