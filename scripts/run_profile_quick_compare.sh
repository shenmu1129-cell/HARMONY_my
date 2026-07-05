#!/usr/bin/env bash
set -euo pipefail

# Quick, low-cost comparison runner for UP-HyperPool.
#
# Goal:
#   Quickly check whether the proposed hybrid profile fast channel is useful
#   against simple HyperMem-style/global retrieval proxies, without running
#   16 threshold/no-fallback configurations.
#
# It runs only three settings:
#   1. rule_thr014       : seed-rule profile typing baseline
#   2. hybrid_thr014     : proposed hybrid profile discovery
#   3. hybrid_nofallback : profile fast channel only, token-saving ablation
#
# Outputs:
#   <OUT_ROOT>/quick_compare.log
#   <OUT_ROOT>/quick_compare_summary.csv
#   <OUT_ROOT>/quick_compare_summary.json
#
# Usage:
#   bash scripts/run_profile_quick_compare.sh SOURCE_DIR OUT_ROOT [MAX_MEMORY] [MAX_QUESTIONS]
#
# Example:
#   bash scripts/run_profile_quick_compare.sh /home/sutongtong/wwt/code outputs/profile_quick_compare 300 50

SOURCE_DIR=${1:-/home/sutongtong/wwt/code}
OUT_ROOT=${2:-outputs/profile_quick_compare}
MAX_MEMORY=${3:-300}
MAX_QUESTIONS=${4:-50}
THRESHOLD=${5:-0.14}

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/data"
LOG_FILE="${OUT_ROOT}/logs/quick_compare_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================================================="
echo "UP-HyperPool Quick Comparison"
echo "=============================================================================="
echo "source_dir   : ${SOURCE_DIR}"
echo "out_root     : ${OUT_ROOT}"
echo "max_memory   : ${MAX_MEMORY}"
echo "max_questions: ${MAX_QUESTIONS}"
echo "threshold    : ${THRESHOLD}"
echo "started_at   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "git_commit   : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python       : $(python --version)"
echo "=============================================================================="

python -m py_compile hypermem/profile_hyperedge_pool.py
python -m py_compile examples/profile_hyperedge_pool_eval.py
python -m py_compile examples/prepare_profile_eval_data.py
python -m py_compile examples/collect_profile_eval_results.py

echo "[1/4] Preparing a small real-data subset"
python examples/prepare_profile_eval_data.py \
  --source-dir "${SOURCE_DIR}" \
  --out-dir "${OUT_ROOT}/data" \
  --max-memory "${MAX_MEMORY}" \
  --max-questions "${MAX_QUESTIONS}"

MEMORY_JSON="${OUT_ROOT}/data/locomo_memory_facts.jsonl"
QUESTIONS_JSON="${OUT_ROOT}/data/locomo_questions.jsonl"
MEMORY_N=$(wc -l < "${MEMORY_JSON}" | tr -d ' ')
QUESTION_N=$(wc -l < "${QUESTIONS_JSON}" | tr -d ' ')

echo "prepared memory rows   : ${MEMORY_N}"
echo "prepared question rows : ${QUESTION_N}"

if [[ "${MEMORY_N}" -eq 0 || "${QUESTION_N}" -eq 0 ]]; then
  echo "ERROR: no usable memory/questions extracted." >&2
  echo "Inspect ${OUT_ROOT}/data/profile_eval_data_report.json" >&2
  exit 3
fi

echo "[2/4] Rule baseline"
python examples/profile_hyperedge_pool_eval.py \
  --memory-json "${MEMORY_JSON}" \
  --questions-json "${QUESTIONS_JSON}" \
  --profile-typing-mode rule \
  --sufficiency-threshold "${THRESHOLD}" \
  --output-dir "${OUT_ROOT}/rule_thr${THRESHOLD}"

echo "[3/4] Hybrid main method"
python examples/profile_hyperedge_pool_eval.py \
  --memory-json "${MEMORY_JSON}" \
  --questions-json "${QUESTIONS_JSON}" \
  --profile-typing-mode hybrid \
  --sufficiency-threshold "${THRESHOLD}" \
  --output-dir "${OUT_ROOT}/hybrid_thr${THRESHOLD}"

echo "[4/4] Hybrid fast-channel only, no fallback"
python examples/profile_hyperedge_pool_eval.py \
  --memory-json "${MEMORY_JSON}" \
  --questions-json "${QUESTIONS_JSON}" \
  --profile-typing-mode hybrid \
  --sufficiency-threshold "${THRESHOLD}" \
  --no-fallback \
  --output-dir "${OUT_ROOT}/hybrid_thr${THRESHOLD}_nofallback"

python examples/collect_profile_eval_results.py \
  --root "${OUT_ROOT}" \
  --out-prefix "${OUT_ROOT}/quick_compare_summary"

echo "=============================================================================="
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log file    : ${LOG_FILE}"
echo "Summary CSV : ${OUT_ROOT}/quick_compare_summary.csv"
echo "Summary JSON: ${OUT_ROOT}/quick_compare_summary.json"
echo "=============================================================================="
cat "${OUT_ROOT}/quick_compare_summary.csv"
