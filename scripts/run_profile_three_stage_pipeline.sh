#!/usr/bin/env bash
set -euo pipefail

# Three-stage one-command pipeline for UP-HyperPool experiments.
#
# This script performs:
#   Stage 1: prepare/convert profile-eval data from a source directory, plus a small subset.
#   Stage 2: run retrieval-only profile experiments on the subset.
#   Stage 3: run retrieval-only profile experiments on the full converted data.
#
# It writes three top-level result files:
#   <OUT_ROOT>/01_data_report.json
#   <OUT_ROOT>/02_subset_retrieval_summary.csv
#   <OUT_ROOT>/03_full_retrieval_summary.csv
#
# Optional QA note:
#   A final answer-generation QA stage is not included yet because this repository currently
#   does not contain a full QA generation/evaluation script. This runner writes a QA status
#   note at <OUT_ROOT>/04_qa_status.txt so the absence is explicit rather than silent.
#
# Usage:
#   bash scripts/run_profile_three_stage_pipeline.sh SOURCE_DIR OUT_ROOT
#
# Example:
#   bash scripts/run_profile_three_stage_pipeline.sh \
#     /home/sutongtong/wwt/code \
#     outputs/profile_three_stage
#
# If SOURCE_DIR has no recognizable JSON/JSONL facts/questions, the script stops.
# For a smoke test only, append --demo:
#   bash scripts/run_profile_three_stage_pipeline.sh /home/sutongtong/wwt/code outputs/profile_three_stage_demo --demo

SOURCE_DIR=${1:-}
OUT_ROOT=${2:-outputs/profile_three_stage}
MODE=${3:-}

if [[ -z "${SOURCE_DIR}" ]]; then
  echo "ERROR: missing SOURCE_DIR." >&2
  echo "Usage: bash scripts/run_profile_three_stage_pipeline.sh SOURCE_DIR OUT_ROOT [--demo]" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/logs"
MASTER_LOG="${OUT_ROOT}/logs/three_stage_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=============================================================================="
echo "UP-HyperPool Three-stage Experiment Pipeline"
echo "=============================================================================="
echo "source_dir : ${SOURCE_DIR}"
echo "out_root   : ${OUT_ROOT}"
echo "mode       : ${MODE:-scan}"
echo "started_at : $(date '+%Y-%m-%d %H:%M:%S')"
echo "git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python     : $(python --version)"
echo "=============================================================================="

echo "[0] Syntax checks"
python -m py_compile hypermem/profile_hyperedge_pool.py
python -m py_compile examples/profile_hyperedge_pool_eval.py
python -m py_compile examples/prepare_profile_eval_data.py
python -m py_compile examples/collect_profile_eval_results.py

DATA_FULL="${OUT_ROOT}/data_full"
DATA_SUBSET="${OUT_ROOT}/data_subset"
FULL_MEMORY="${DATA_FULL}/locomo_memory_facts.jsonl"
FULL_QUESTIONS="${DATA_FULL}/locomo_questions.jsonl"
SUBSET_MEMORY="${DATA_SUBSET}/locomo_memory_facts.jsonl"
SUBSET_QUESTIONS="${DATA_SUBSET}/locomo_questions.jsonl"

mkdir -p "${DATA_FULL}" "${DATA_SUBSET}"

echo "=============================================================================="
echo "[1] Stage 1: prepare converted data"
echo "=============================================================================="
if [[ "${MODE}" == "--demo" ]]; then
  echo "Preparing built-in demo data. This is NOT a formal LoCoMo result."
  python examples/prepare_profile_eval_data.py --demo --out-dir "${DATA_FULL}"
  cp "${FULL_MEMORY}" "${SUBSET_MEMORY}"
  cp "${FULL_QUESTIONS}" "${SUBSET_QUESTIONS}"
  cp "${DATA_FULL}/profile_eval_data_report.json" "${DATA_SUBSET}/profile_eval_data_report.json"
else
  if [[ ! -d "${SOURCE_DIR}" ]]; then
    echo "ERROR: source directory not found: ${SOURCE_DIR}" >&2
    exit 2
  fi

  echo "Preparing full converted data from source directory."
  python examples/prepare_profile_eval_data.py \
    --source-dir "${SOURCE_DIR}" \
    --out-dir "${DATA_FULL}" \
    --max-memory 1000000 \
    --max-questions 1000000

  echo "Preparing subset converted data from source directory."
  python examples/prepare_profile_eval_data.py \
    --source-dir "${SOURCE_DIR}" \
    --out-dir "${DATA_SUBSET}" \
    --max-memory 1000 \
    --max-questions 200
fi

FULL_MEMORY_N=$(wc -l < "${FULL_MEMORY}" | tr -d ' ')
FULL_QUESTION_N=$(wc -l < "${FULL_QUESTIONS}" | tr -d ' ')
SUBSET_MEMORY_N=$(wc -l < "${SUBSET_MEMORY}" | tr -d ' ')
SUBSET_QUESTION_N=$(wc -l < "${SUBSET_QUESTIONS}" | tr -d ' ')

python - <<PY
import json
from pathlib import Path
out = Path("${OUT_ROOT}/01_data_report.json")
report = {
  "source_dir": "${SOURCE_DIR}",
  "mode": "${MODE:-scan}",
  "full_memory_rows": int("${FULL_MEMORY_N}"),
  "full_question_rows": int("${FULL_QUESTION_N}"),
  "subset_memory_rows": int("${SUBSET_MEMORY_N}"),
  "subset_question_rows": int("${SUBSET_QUESTION_N}"),
  "full_memory_path": "${FULL_MEMORY}",
  "full_questions_path": "${FULL_QUESTIONS}",
  "subset_memory_path": "${SUBSET_MEMORY}",
  "subset_questions_path": "${SUBSET_QUESTIONS}",
}
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

if [[ "${FULL_MEMORY_N}" -eq 0 || "${FULL_QUESTION_N}" -eq 0 ]]; then
  echo "ERROR: full converted data is empty." >&2
  echo "Inspect: ${DATA_FULL}/profile_eval_data_report.json" >&2
  echo "The source directory likely does not contain recognizable JSON/JSONL LoCoMo/HyperMem data." >&2
  exit 3
fi

if [[ "${SUBSET_MEMORY_N}" -eq 0 || "${SUBSET_QUESTION_N}" -eq 0 ]]; then
  echo "ERROR: subset converted data is empty." >&2
  echo "Inspect: ${DATA_SUBSET}/profile_eval_data_report.json" >&2
  exit 3
fi

echo "=============================================================================="
echo "[2] Stage 2: subset retrieval-only experiment"
echo "=============================================================================="
bash scripts/run_profile_formal_eval.sh \
  "${SUBSET_MEMORY}" \
  "${SUBSET_QUESTIONS}" \
  "${OUT_ROOT}/02_subset_retrieval"

cp "${OUT_ROOT}/02_subset_retrieval/summary.csv" "${OUT_ROOT}/02_subset_retrieval_summary.csv"
cp "${OUT_ROOT}/02_subset_retrieval/summary.json" "${OUT_ROOT}/02_subset_retrieval_summary.json"

echo "=============================================================================="
echo "[3] Stage 3: full retrieval-only experiment"
echo "=============================================================================="
bash scripts/run_profile_formal_eval.sh \
  "${FULL_MEMORY}" \
  "${FULL_QUESTIONS}" \
  "${OUT_ROOT}/03_full_retrieval"

cp "${OUT_ROOT}/03_full_retrieval/summary.csv" "${OUT_ROOT}/03_full_retrieval_summary.csv"
cp "${OUT_ROOT}/03_full_retrieval/summary.json" "${OUT_ROOT}/03_full_retrieval_summary.json"

echo "=============================================================================="
echo "[4] QA generation/evaluation status"
echo "=============================================================================="
QA_STATUS="${OUT_ROOT}/04_qa_status.txt"
cat > "${QA_STATUS}" <<'EOF'
Full QA generation/evaluation was not executed.
Reason: this repository currently contains retrieval-only profile hyperedge evaluation scripts, but no full answer-generation QA evaluator that computes EM/F1/ROUGE-L/Contains/Answer Recall.

Completed stages:
1. Data conversion/preparation
2. Subset retrieval-only profile evaluation
3. Full retrieval-only profile evaluation

Next implementation step:
Add a QA evaluator that reads retrieval traces, prompts an LLM with the selected evidence, generates answers, and computes final answer metrics.
EOF
cat "${QA_STATUS}"

echo "=============================================================================="
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Master log: ${MASTER_LOG}"
echo "Core result files:"
echo "  ${OUT_ROOT}/01_data_report.json"
echo "  ${OUT_ROOT}/02_subset_retrieval_summary.csv"
echo "  ${OUT_ROOT}/03_full_retrieval_summary.csv"
echo "  ${OUT_ROOT}/04_qa_status.txt"
echo "=============================================================================="
