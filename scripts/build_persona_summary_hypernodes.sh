#!/usr/bin/env bash
set -euo pipefail

GRAPH=${GRAPH:-outputs/persona_chat_cost_aware/persona_chat/graph_50/behavioral_hybrid_graph.json}
OUT=${OUT:-outputs/persona_chat_cost_aware/persona_chat/graph_50/summary_hypernodes.json}
MAX_SUMMARY_TOKENS=${MAX_SUMMARY_TOKENS:-48}
FACT_GROUP_SIZE=${FACT_GROUP_SIZE:-4}

python -m py_compile hypermem/summary_hypernodes.py
python -m py_compile examples/build_summary_hypernodes.py

python examples/build_summary_hypernodes.py \
  --memory-graph "${GRAPH}" \
  --output "${OUT}" \
  --max-summary-tokens "${MAX_SUMMARY_TOKENS}" \
  --fact-group-size "${FACT_GROUP_SIZE}"
