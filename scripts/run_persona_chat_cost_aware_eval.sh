#!/usr/bin/env bash
set -euo pipefail

# Persona-Chat only smoke test for cost-aware hypergraph retrieval.
# This wrapper prevents accidentally running ConvAI2 or MSC.

export DATASET_ROOTS="${DATASET_ROOTS:-/home/sutongtong/wwt/dataset/Persona-Chat}"
export MAX_MEMORY="${MAX_MEMORY:-50}"
export MAX_QUESTIONS="${MAX_QUESTIONS:-1000}"
export USE_LLM_HIERARCHY="${USE_LLM_HIERARCHY:-0}"
export OUT_ROOT="${OUT_ROOT:-outputs/persona_chat_cost_aware}"
export METHODS="${METHODS:-profile_full,topic_episode,progressive,budget,adaptive_budget,adaptive_tiny}"

case "${DATASET_ROOTS}" in
  *ConvAI2*|*msc*|*MSC*)
    echo "ERROR: run_persona_chat_cost_aware_eval.sh is Persona-Chat only."
    echo "DATASET_ROOTS=${DATASET_ROOTS}"
    exit 2
    ;;
esac

bash scripts/run_parlai_cost_aware_eval.sh
