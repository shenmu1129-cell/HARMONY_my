"""Evaluate QA retrieval using a previously built profile graph.

This avoids rebuilding an LLM-induced graph for every evaluation variant.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import profile_centric_hypergraph_eval as base_eval  # noqa: E402
from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory  # noqa: E402


def build_memory_from_saved_graph(memory_rows: Sequence[Dict[str, Any]], args: argparse.Namespace, label: str) -> ProfileCentricHypergraphMemory:
    print(f"[loaded-graph] load graph for {label}: {args.memory_graph}", flush=True)
    return ProfileCentricHypergraphMemory.load(args.memory_graph)


def load_only_questions(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    questions = base_eval.normalize_questions(base_eval.read_json_or_jsonl(Path(args.questions_json)))
    if args.max_questions and len(questions) > args.max_questions:
        questions = questions[: args.max_questions]
    return [], questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-graph", required=True)
    parser.add_argument("--questions-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--online-eval", action="store_true")
    parser.add_argument("--top-k-edges", type=int, default=3)
    parser.add_argument("--top-k-facts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--sufficiency-threshold", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=0.18)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--attach-threshold", type=float, default=0.52)
    parser.add_argument("--discovery-threshold", type=float, default=0.55)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--max-questions", type=int, default=0)
    args = parser.parse_args()
    args.memory_json = args.memory_graph
    args.max_memory = 0
    args.construction_mode = "loaded_llm_graph"
    args.batch_size = 0
    args.canonical_threshold = 0.0
    args.max_edge_facts = 0
    args.min_feature_support = 1
    args.consolidate_every = 0
    args.max_auto_edge_pairs = 0
    return args


if __name__ == "__main__":
    base_eval.build_memory = build_memory_from_saved_graph
    base_eval.load_inputs = load_only_questions
    base_eval.run_eval(parse_args())
