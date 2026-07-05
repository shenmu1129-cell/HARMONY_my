#!/usr/bin/env bash
set -euo pipefail

# Half-data formal comparison runner.
#
# Purpose:
#   Use exactly half of the converted dataset, then compare:
#     1. FULL HyperMem baseline
#     2. RL/reward-based profile method
#
# This script deliberately does NOT fake either method.
# You must provide the actual commands through environment variables:
#   FULL_HYPERMEM_CMD : command that runs the complete HyperMem baseline
#   RL_PROFILE_CMD    : command that runs your RL/reward-based profile method
#
# The commands may use the following placeholders:
#   {MEMORY_JSON}      -> half-data memory facts JSONL
#   {QUESTIONS_JSON}   -> half-data questions JSONL
#   {OUT_DIR}          -> method-specific output directory
#   {SUMMARY_JSON}     -> method-specific summary JSON target path
#   {SUMMARY_CSV}      -> method-specific summary CSV target path
#
# Example:
#   FULL_HYPERMEM_CMD='python examples/hypermem_full_eval.py --memory-json {MEMORY_JSON} --questions-json {QUESTIONS_JSON} --output-dir {OUT_DIR}' \
#   RL_PROFILE_CMD='python examples/rl_profile_eval.py --memory-json {MEMORY_JSON} --questions-json {QUESTIONS_JSON} --output-dir {OUT_DIR}' \
#   bash scripts/run_halfdata_fullhypermem_vs_rlprofile.sh /home/sutongtong/wwt/code outputs/half_fullhypermem_vs_rl
#
# Outputs:
#   <OUT_ROOT>/data_half/locomo_memory_facts.jsonl
#   <OUT_ROOT>/data_half/locomo_questions.jsonl
#   <OUT_ROOT>/data_report.json
#   <OUT_ROOT>/full_hypermem/
#   <OUT_ROOT>/rl_profile/
#   <OUT_ROOT>/comparison_manifest.json
#   <OUT_ROOT>/logs/*.log

SOURCE_DIR=${1:-/home/sutongtong/wwt/code}
OUT_ROOT=${2:-outputs/half_fullhypermem_vs_rl}

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/data_full" "${OUT_ROOT}/data_half" "${OUT_ROOT}/full_hypermem" "${OUT_ROOT}/rl_profile"
LOG_FILE="${OUT_ROOT}/logs/half_compare_$(date +%Y%m%d_%H%M%S).log"
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

run_template_cmd() {
  local template=$1
  local memory_json=$2
  local questions_json=$3
  local out_dir=$4
  local summary_json=$5
  local summary_csv=$6

  local cmd="${template}"
  cmd="${cmd//\{MEMORY_JSON\}/${memory_json}}"
  cmd="${cmd//\{QUESTIONS_JSON\}/${questions_json}}"
  cmd="${cmd//\{OUT_DIR\}/${out_dir}}"
  cmd="${cmd//\{SUMMARY_JSON\}/${summary_json}}"
  cmd="${cmd//\{SUMMARY_CSV\}/${summary_csv}}"

  echo "COMMAND: ${cmd}"
  bash -lc "${cmd}"
}

echo "=============================================================================="
echo "Half-data Full-HyperMem vs RL-Profile Comparison"
echo "=============================================================================="
echo "source_dir : ${SOURCE_DIR}"
echo "out_root   : ${OUT_ROOT}"
echo "started_at : $(date '+%Y-%m-%d %H:%M:%S')"
echo "git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python     : $(python --version)"
echo "=============================================================================="

if [[ -z "${FULL_HYPERMEM_CMD:-}" ]]; then
  echo "ERROR: FULL_HYPERMEM_CMD is not set." >&2
  echo "This script requires the real complete HyperMem evaluation command." >&2
  echo "Do not use global_fact_retrieval if you need full HyperMem." >&2
  exit 10
fi

if [[ -z "${RL_PROFILE_CMD:-}" ]]; then
  echo "ERROR: RL_PROFILE_CMD is not set." >&2
  echo "This script requires the real RL/reward-based profile evaluation command." >&2
  echo "Do not use the plain hybrid profile pool if you need the RL version." >&2
  exit 11
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "ERROR: source directory not found: ${SOURCE_DIR}" >&2
  exit 2
fi

python -m py_compile examples/prepare_profile_eval_data.py

TOTAL_STAGES=4

stage_begin 1 "${TOTAL_STAGES}" "Convert source data and keep exactly half"
python examples/prepare_profile_eval_data.py \
  --source-dir "${SOURCE_DIR}" \
  --out-dir "${OUT_ROOT}/data_full" \
  --max-memory 1000000 \
  --max-questions 1000000

FULL_MEMORY="${OUT_ROOT}/data_full/locomo_memory_facts.jsonl"
FULL_QUESTIONS="${OUT_ROOT}/data_full/locomo_questions.jsonl"
HALF_MEMORY="${OUT_ROOT}/data_half/locomo_memory_facts.jsonl"
HALF_QUESTIONS="${OUT_ROOT}/data_half/locomo_questions.jsonl"

FULL_MEMORY_N=$(wc -l < "${FULL_MEMORY}" | tr -d ' ')
FULL_QUESTION_N=$(wc -l < "${FULL_QUESTIONS}" | tr -d ' ')
HALF_MEMORY_N=$(( (FULL_MEMORY_N + 1) / 2 ))
HALF_QUESTION_N=$(( (FULL_QUESTION_N + 1) / 2 ))

head -n "${HALF_MEMORY_N}" "${FULL_MEMORY}" > "${HALF_MEMORY}"
head -n "${HALF_QUESTION_N}" "${FULL_QUESTIONS}" > "${HALF_QUESTIONS}"

python - <<PY
import json
from pathlib import Path
report = {
  "source_dir": "${SOURCE_DIR}",
  "full_memory_rows": int("${FULL_MEMORY_N}"),
  "full_question_rows": int("${FULL_QUESTION_N}"),
  "half_memory_rows": int("${HALF_MEMORY_N}"),
  "half_question_rows": int("${HALF_QUESTION_N}"),
  "half_memory_path": "${HALF_MEMORY}",
  "half_questions_path": "${HALF_QUESTIONS}",
}
out = Path("${OUT_ROOT}/data_report.json")
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

if [[ "${HALF_MEMORY_N}" -eq 0 || "${HALF_QUESTION_N}" -eq 0 ]]; then
  echo "ERROR: empty half-data files." >&2
  exit 3
fi
stage_end 1 "${TOTAL_STAGES}" "Convert source data and keep exactly half"

stage_begin 2 "${TOTAL_STAGES}" "Run COMPLETE HyperMem baseline on half data"
run_template_cmd \
  "${FULL_HYPERMEM_CMD}" \
  "${HALF_MEMORY}" \
  "${HALF_QUESTIONS}" \
  "${OUT_ROOT}/full_hypermem" \
  "${OUT_ROOT}/full_hypermem/summary.json" \
  "${OUT_ROOT}/full_hypermem/summary.csv"
stage_end 2 "${TOTAL_STAGES}" "Run COMPLETE HyperMem baseline on half data"

stage_begin 3 "${TOTAL_STAGES}" "Run RL/reward-based profile method on half data"
run_template_cmd \
  "${RL_PROFILE_CMD}" \
  "${HALF_MEMORY}" \
  "${HALF_QUESTIONS}" \
  "${OUT_ROOT}/rl_profile" \
  "${OUT_ROOT}/rl_profile/summary.json" \
  "${OUT_ROOT}/rl_profile/summary.csv"
stage_end 3 "${TOTAL_STAGES}" "Run RL/reward-based profile method on half data"

stage_begin 4 "${TOTAL_STAGES}" "Write comparison manifest"
python - <<PY
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
manifest = {
  "data_report": str(root / "data_report.json"),
  "full_hypermem_dir": str(root / "full_hypermem"),
  "rl_profile_dir": str(root / "rl_profile"),
  "full_hypermem_cmd": "${FULL_HYPERMEM_CMD}",
  "rl_profile_cmd": "${RL_PROFILE_CMD}",
  "log_file": "${LOG_FILE}",
  "note": "This manifest records the actual commands used. The script does not replace full HyperMem or RL-profile implementations; it calls the commands supplied by the user.",
}
out = root / "comparison_manifest.json"
out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY
stage_end 4 "${TOTAL_STAGES}" "Write comparison manifest"

PIPELINE_END=$(date +%s)
PIPELINE_ELAPSED=$(( PIPELINE_END - PIPELINE_START ))

echo "=============================================================================="
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total elapsed: ${PIPELINE_ELAPSED}s"
echo "Log file: ${LOG_FILE}"
echo "Data report: ${OUT_ROOT}/data_report.json"
echo "Manifest: ${OUT_ROOT}/comparison_manifest.json"
echo "=============================================================================="
