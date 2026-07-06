#!/usr/bin/env bash
set -euo pipefail

# Run the new profile-centric hypergraph memory flow.
#
# Flow:
#   1. prepare/convert memory facts and QA
#   2. optionally keep only a data fraction, e.g. 0.5 for half data
#   3. build User Profile Hyperedge -> Fact memory
#   4. train hyperedge utility on train QA
#   5. evaluate embedding-only, frozen utility, and online predict-then-update
#
# Usage:
#   bash scripts/run_profile_centric_hypergraph.sh SOURCE_DIR OUT_ROOT [DATA_FRACTION] [TRAIN_RATIO]
#
# Example:
#   bash scripts/run_profile_centric_hypergraph.sh /home/sutongtong/wwt/code outputs/profile_centric_hg 0.5 0.5

SOURCE_DIR=${1:-/home/sutongtong/wwt/code}
OUT_ROOT=${2:-outputs/profile_centric_hg}
DATA_FRACTION=${3:-0.5}
TRAIN_RATIO=${4:-0.5}

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/data_full" "${OUT_ROOT}/data_used"
LOG_FILE="${OUT_ROOT}/logs/profile_centric_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

PIPELINE_START=$(date +%s)
STAGE_START=0

progress_bar() {
  local current=$1
  local total=$2
  local label=$3
  local width=28
  local filled=$(( current * width / total ))
  local empty=$(( width - filled ))
  local bar=""
  local i
  for ((i=0; i<filled; i++)); do bar+="#"; done
  for ((i=0; i<empty; i++)); do bar+="-"; done
  printf '\n[%s] [%s] %s/%s\n' "${label}" "${bar}" "${current}" "${total}"
}

stage_begin() {
  local idx=$1
  local total=$2
  local msg=$3
  STAGE_START=$(date +%s)
  echo ""
  echo "=============================================================================="
  progress_bar "${idx}" "${total}" "stage"
  echo "START stage ${idx}/${total}: ${msg}"
  echo "started_at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=============================================================================="
}

stage_end() {
  local idx=$1
  local total=$2
  local msg=$3
  local now elapsed
  now=$(date +%s)
  elapsed=$(( now - STAGE_START ))
  progress_bar "${idx}" "${total}" "done "
  echo "DONE stage ${idx}/${total}: ${msg} | elapsed=${elapsed}s"
}

echo "=============================================================================="
echo "Profile-Centric Hypergraph Memory Runner"
echo "=============================================================================="
echo "source_dir   : ${SOURCE_DIR}"
echo "out_root     : ${OUT_ROOT}"
echo "data_fraction: ${DATA_FRACTION}"
echo "train_ratio  : ${TRAIN_RATIO}"
echo "started_at   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "git_commit   : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python       : $(python --version)"
echo "=============================================================================="

python -m py_compile hypermem/profile_centric_hypergraph.py
python -m py_compile examples/profile_centric_hypergraph_eval.py
python -m py_compile examples/prepare_profile_eval_data.py

TOTAL_STAGES=3

stage_begin 1 "${TOTAL_STAGES}" "Prepare full converted data"
python examples/prepare_profile_eval_data.py \
  --source-dir "${SOURCE_DIR}" \
  --out-dir "${OUT_ROOT}/data_full" \
  --max-memory 1000000 \
  --max-questions 1000000
stage_end 1 "${TOTAL_STAGES}" "Prepare full converted data"

FULL_MEMORY="${OUT_ROOT}/data_full/locomo_memory_facts.jsonl"
FULL_QUESTIONS="${OUT_ROOT}/data_full/locomo_questions.jsonl"
USED_MEMORY="${OUT_ROOT}/data_used/locomo_memory_facts.jsonl"
USED_QUESTIONS="${OUT_ROOT}/data_used/locomo_questions.jsonl"

stage_begin 2 "${TOTAL_STAGES}" "Select data fraction"
python - <<PY
from pathlib import Path
import json
fraction = float("${DATA_FRACTION}")
fraction = min(1.0, max(0.01, fraction))
full_memory = Path("${FULL_MEMORY}")
full_questions = Path("${FULL_QUESTIONS}")
used_memory = Path("${USED_MEMORY}")
used_questions = Path("${USED_QUESTIONS}")
mem_lines = full_memory.read_text(encoding="utf-8").splitlines()
q_lines = full_questions.read_text(encoding="utf-8").splitlines()
mem_n = max(1, int(len(mem_lines) * fraction)) if mem_lines else 0
q_n = max(1, int(len(q_lines) * fraction)) if q_lines else 0
used_memory.write_text("\n".join(mem_lines[:mem_n]) + ("\n" if mem_n else ""), encoding="utf-8")
used_questions.write_text("\n".join(q_lines[:q_n]) + ("\n" if q_n else ""), encoding="utf-8")
report = {
  "source_dir": "${SOURCE_DIR}",
  "data_fraction": fraction,
  "full_memory_rows": len(mem_lines),
  "full_question_rows": len(q_lines),
  "used_memory_rows": mem_n,
  "used_question_rows": q_n,
  "used_memory_path": str(used_memory),
  "used_questions_path": str(used_questions),
  "train_ratio": float("${TRAIN_RATIO}"),
}
out = Path("${OUT_ROOT}/data_report.json")
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
stage_end 2 "${TOTAL_STAGES}" "Select data fraction"

stage_begin 3 "${TOTAL_STAGES}" "Train utility and evaluate profile-centric retrieval"
python examples/profile_centric_hypergraph_eval.py \
  --memory-json "${USED_MEMORY}" \
  --questions-json "${USED_QUESTIONS}" \
  --train-ratio "${TRAIN_RATIO}" \
  --online-eval \
  --output-dir "${OUT_ROOT}/eval"
stage_end 3 "${TOTAL_STAGES}" "Train utility and evaluate profile-centric retrieval"

PIPELINE_END=$(date +%s)
PIPELINE_ELAPSED=$(( PIPELINE_END - PIPELINE_START ))

echo "=============================================================================="
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total elapsed: ${PIPELINE_ELAPSED}s"
echo "Log file   : ${LOG_FILE}"
echo "Data report: ${OUT_ROOT}/data_report.json"
echo "Summary CSV: ${OUT_ROOT}/eval/profile_centric_summary.csv"
echo "Summary JSON: ${OUT_ROOT}/eval/profile_centric_summary.json"
echo "=============================================================================="
cat "${OUT_ROOT}/eval/profile_centric_summary.csv"
