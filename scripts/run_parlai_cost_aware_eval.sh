#!/usr/bin/env bash
set -euo pipefail

# Run the cost-aware hypergraph retrieval test on local ParlAI datasets.
# Default is a light Persona-Chat smoke test: 50 memory rows + 1000 questions.
#
# Usage:
#   bash scripts/run_parlai_cost_aware_eval.sh
#
# Useful env controls:
#   OUT_ROOT=outputs/parlai_cost_aware
#   MAX_MEMORY=50
#   MAX_QUESTIONS=1000
#   USE_LLM_HIERARCHY=0       # 0 = fast fallback hierarchy, 1 = DeepSeek hierarchy
#   USE_LLM_BEHAVIOR=0        # 0 = local deterministic hyperedge induction, 1 = DeepSeek behavioral induction
#   DATASET_ROOTS="/path/Persona-Chat /path/ConvAI2 /path/msc"
#   METHODS="profile_full,topic_episode,progressive,budget,adaptive_budget,adaptive_tiny"
#   NO_PROGRESS=0

OUT_ROOT=${OUT_ROOT:-outputs/parlai_cost_aware}
MAX_MEMORY=${MAX_MEMORY:-50}
MAX_QUESTIONS=${MAX_QUESTIONS:-1000}
USE_LLM_HIERARCHY=${USE_LLM_HIERARCHY:-0}
USE_LLM_BEHAVIOR=${USE_LLM_BEHAVIOR:-0}
NO_PROGRESS=${NO_PROGRESS:-0}
METHODS=${METHODS:-profile_full,topic_episode,progressive,budget,adaptive_budget,adaptive_tiny}
TOP_K_EDGES=${TOP_K_EDGES:-3}
TOP_K_FACTS=${TOP_K_FACTS:-8}
TOP_K_TOPICS=${TOP_K_TOPICS:-3}
TOP_K_EPISODES=${TOP_K_EPISODES:-6}
MAX_TOKENS=${MAX_TOKENS:-450}
BUDGET_RATIO=${BUDGET_RATIO:-0.55}
EXPANSION_RATIO=${EXPANSION_RATIO:-0.45}
TINY_BUDGET_TOKENS=${TINY_BUDGET_TOKENS:-110}
REP_FACTS_PER_EDGE=${REP_FACTS_PER_EDGE:-2}
HIERARCHY_BATCH_SIZE=${HIERARCHY_BATCH_SIZE:-10}
BEHAVIOR_BATCH_SIZE=${BEHAVIOR_BATCH_SIZE:-25}
CANONICAL_THRESHOLD=${CANONICAL_THRESHOLD:-0.72}
CONSOLIDATE_EVERY=${CONSOLIDATE_EVERY:-2}
MAX_EDGE_FACTS=${MAX_EDGE_FACTS:-80}

if [[ -z "${DATASET_ROOTS:-}" ]]; then
  DATASET_ROOTS="/home/sutongtong/wwt/dataset/Persona-Chat"
fi

mkdir -p "${OUT_ROOT}/logs"
LOG_FILE="${OUT_ROOT}/logs/parlai_cost_aware_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

PROGRESS_ARG="--show-progress"
EVAL_PROGRESS_ARG=""
if [[ "${NO_PROGRESS}" == "1" ]]; then
  PROGRESS_ARG=""
  EVAL_PROGRESS_ARG="--no-progress"
fi

HIERARCHY_ARGS=""
if [[ "${USE_LLM_HIERARCHY}" != "1" ]]; then
  HIERARCHY_ARGS="--no-llm-hierarchy"
fi

BEHAVIOR_ARGS=""
if [[ "${USE_LLM_BEHAVIOR}" != "1" ]]; then
  BEHAVIOR_ARGS="--no-llm-behavior"
fi

echo "=============================================================================="
echo "ParlAI Cost-aware Hypergraph Evaluation"
echo "=============================================================================="
echo "out_root          : ${OUT_ROOT}"
echo "max_memory        : ${MAX_MEMORY}"
echo "max_questions     : ${MAX_QUESTIONS}"
echo "use_llm_hierarchy : ${USE_LLM_HIERARCHY}"
echo "use_llm_behavior  : ${USE_LLM_BEHAVIOR}"
echo "methods           : ${METHODS}"
echo "dataset_roots     : ${DATASET_ROOTS}"
echo "git_commit        : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python            : $(python --version)"
echo "log_file          : ${LOG_FILE}"
echo "=============================================================================="

python -m py_compile examples/prepare_parlai_memory_data.py
python -m py_compile examples/build_behavioral_hybrid_memory.py
python -m py_compile examples/eval_behavioral_profile_graph.py
python -m py_compile hypermem/cost_aware_retrieval.py
bash -n scripts/run_parlai_cost_aware_eval.sh

SUMMARY_FILES=()
for ROOT in ${DATASET_ROOTS}; do
  if [[ ! -d "${ROOT}" ]]; then
    echo "WARNING: skip missing dataset root: ${ROOT}"
    continue
  fi
  NAME=$(basename "${ROOT}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
  DS_OUT="${OUT_ROOT}/${NAME}"
  DATA_OUT="${DS_OUT}/data"
  GRAPH_OUT="${DS_OUT}/graph_50"
  EVAL_OUT="${DS_OUT}/cost_aware_eval"
  mkdir -p "${DS_OUT}"

  echo ""
  echo "=============================================================================="
  echo "DATASET: ${NAME}"
  echo "root   : ${ROOT}"
  echo "=============================================================================="

  echo "[1/3] Prepare ParlAI data -> ${DATA_OUT}"
  python examples/prepare_parlai_memory_data.py \
    --dataset-root "${ROOT}" \
    --out-dir "${DATA_OUT}" \
    --max-memory "${MAX_MEMORY}" \
    --max-questions "${MAX_QUESTIONS}" \
    ${PROGRESS_ARG}

  echo "[2/3] Build 50-memory behavioral hypergraph -> ${GRAPH_OUT}"
  python examples/build_behavioral_hybrid_memory.py \
    --memory-json "${DATA_OUT}/memory_facts.jsonl" \
    --output-dir "${GRAPH_OUT}" \
    --max-memory "${MAX_MEMORY}" \
    --hierarchy-batch-size "${HIERARCHY_BATCH_SIZE}" \
    --behavior-batch-size "${BEHAVIOR_BATCH_SIZE}" \
    --canonical-threshold "${CANONICAL_THRESHOLD}" \
    --consolidate-every "${CONSOLIDATE_EVERY}" \
    --llm-consolidation-rounds 0 \
    --max-edge-facts "${MAX_EDGE_FACTS}" \
    ${HIERARCHY_ARGS} \
    ${BEHAVIOR_ARGS} \
    ${EVAL_PROGRESS_ARG}

  echo "[3/3] Cost-aware eval on generated questions -> ${EVAL_OUT}"
  python examples/eval_behavioral_profile_graph.py \
    --memory-graph "${GRAPH_OUT}/behavioral_hybrid_graph.json" \
    --questions-json "${DATA_OUT}/questions.jsonl" \
    --output-dir "${EVAL_OUT}" \
    --eval-scope all \
    --methods "${METHODS}" \
    --top-k-edges "${TOP_K_EDGES}" \
    --top-k-facts "${TOP_K_FACTS}" \
    --top-k-topics "${TOP_K_TOPICS}" \
    --top-k-episodes "${TOP_K_EPISODES}" \
    --max-tokens "${MAX_TOKENS}" \
    --budget-ratio "${BUDGET_RATIO}" \
    --expansion-ratio "${EXPANSION_RATIO}" \
    --tiny-budget-tokens "${TINY_BUDGET_TOKENS}" \
    --representative-facts-per-edge "${REP_FACTS_PER_EDGE}" \
    ${EVAL_PROGRESS_ARG}

  SUMMARY_FILES+=("${EVAL_OUT}/cost_aware_summary.csv")
  echo "Summary: ${EVAL_OUT}/cost_aware_summary.csv"
  cat "${EVAL_OUT}/cost_aware_summary.csv"
done

COMBINED="${OUT_ROOT}/combined_cost_aware_summary.csv"
SUMMARY_LIST="${OUT_ROOT}/summary_files.txt"
printf "%s\n" "${SUMMARY_FILES[@]}" > "${SUMMARY_LIST}"
python - <<PY
from pathlib import Path
import csv
summary_list = Path("${SUMMARY_LIST}")
summary_files = [Path(x.strip()) for x in summary_list.read_text(encoding="utf-8").splitlines() if x.strip()]
out = Path("${COMBINED}")
rows = []
for path in summary_files:
    if not path.exists():
        continue
    dataset = path.parents[1].name
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"dataset": dataset, **row})
if rows:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("combined summary:", out)
    print(out.read_text(encoding="utf-8"))
else:
    print("WARNING: no summary rows collected")
PY

echo "=============================================================================="
echo "Done"
echo "Log file        : ${LOG_FILE}"
echo "Combined summary: ${COMBINED}"
echo "=============================================================================="
