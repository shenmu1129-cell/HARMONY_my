#!/usr/bin/env bash
set -euo pipefail

GRAPH=${GRAPH:-outputs/persona_chat_cost_aware/persona_chat/graph_50/behavioral_hybrid_graph.json}
MEMORY=${MEMORY:-outputs/persona_chat_cost_aware/persona_chat/data/memory_facts.jsonl}
QUESTIONS=${QUESTIONS:-outputs/persona_chat_cost_aware/persona_chat/data/questions.jsonl}
OUT=${OUT:-outputs/persona_chat_cost_aware/persona_chat/conditioned_dialogue_probe}
MAXQ=${MAXQ:-1000}
MAX_TOKENS=${MAX_TOKENS:-160}
METHODS=${METHODS:-adaptive_tiny,condition_dialogue,fact_dialogue,condition_fact_dialogue}

python -m py_compile examples/probe_conditioned_dialogue_evidence.py

python examples/probe_conditioned_dialogue_evidence.py \
  --memory-graph "${GRAPH}" \
  --memory-json "${MEMORY}" \
  --questions-json "${QUESTIONS}" \
  --output-dir "${OUT}" \
  --max-questions "${MAXQ}" \
  --max-tokens "${MAX_TOKENS}" \
  --methods "${METHODS}"
