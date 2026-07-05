#!/usr/bin/env bash
set -euo pipefail

# Formal retrieval-only evaluation runner for UP-HyperPool.
# It requires prepared memory/questions JSONL files and writes all logs/results.
#
# Usage:
#   bash scripts/run_profile_formal_eval.sh \
#     data/locomo_memory_facts.jsonl \
#     data/locomo_questions.jsonl \
#     outputs/formal_profile_eval
#
# Outputs:
#   <OUT_ROOT>/logs/*.log
#   <OUT_ROOT>/rule_default/...
#   <OUT_ROOT>/unsupervised_default/...
#   <OUT_ROOT>/hybrid_thr014/...
#   <OUT_ROOT>/hybrid_thr014_nofallback/...
#   <OUT_ROOT>/sweep/hybrid_thr_*/...
#   <OUT_ROOT>/summary.csv
#   <OUT_ROOT>/summary.json

MEMORY_JSON=${1:-}
QUESTIONS_JSON=${2:-}
OUT_ROOT=${3:-outputs/formal_profile_eval}

if [[ -z "${MEMORY_JSON}" || -z "${QUESTIONS_JSON}" ]]; then
  echo "ERROR: missing input files." >&2
  echo "Usage:" >&2
  echo "  bash scripts/run_profile_formal_eval.sh data/locomo_memory_facts.jsonl data/locomo_questions.jsonl outputs/formal_profile_eval" >&2
  exit 2
fi

if [[ ! -f "${MEMORY_JSON}" ]]; then
  echo "ERROR: memory file not found: ${MEMORY_JSON}" >&2
  exit 2
fi

if [[ ! -f "${QUESTIONS_JSON}" ]]; then
  echo "ERROR: questions file not found: ${QUESTIONS_JSON}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/logs"
MASTER_LOG="${OUT_ROOT}/logs/formal_eval_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=============================================================================="
echo "UP-HyperPool Formal Retrieval-only Evaluation"
echo "=============================================================================="
echo "memory_json   : ${MEMORY_JSON}"
echo "questions_json: ${QUESTIONS_JSON}"
echo "out_root      : ${OUT_ROOT}"
echo "started_at    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "git_commit    : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python        : $(python --version)"
echo "=============================================================================="

echo "[0] Syntax checks"
python -m py_compile hypermem/profile_hyperedge_pool.py
python -m py_compile examples/profile_hyperedge_pool_eval.py
python -m py_compile examples/collect_profile_eval_results.py

echo "[1] Rule baseline"
python examples/profile_hyperedge_pool_eval.py \
  --memory-json "${MEMORY_JSON}" \
  --questions-json "${QUESTIONS_JSON}" \
  --profile-typing-mode rule \
  --sufficiency-threshold 0.14 \
  --output-dir "${OUT_ROOT}/rule_thr014"

echo "[2] Unsupervised baseline"
python examples/profile_hyperedge_pool_eval.py \
  --memory-json "${MEMORY_JSON}" \
  --questions-json "${QUESTIONS_JSON}" \
  --profile-typing-mode unsupervised \
  --sufficiency-threshold 0.14 \
  --output-dir "${OUT_ROOT}/unsupervised_thr014"

echo "[3] Hybrid main method, threshold=0.14"
python examples/profile_hyperedge_pool_eval.py \
  --memory-json "${MEMORY_JSON}" \
  --questions-json "${QUESTIONS_JSON}" \
  --profile-typing-mode hybrid \
  --sufficiency-threshold 0.14 \
  --output-dir "${OUT_ROOT}/hybrid_thr014"

echo "[4] Hybrid no-fallback ablation, threshold=0.14"
python examples/profile_hyperedge_pool_eval.py \
  --memory-json "${MEMORY_JSON}" \
  --questions-json "${QUESTIONS_JSON}" \
  --profile-typing-mode hybrid \
  --sufficiency-threshold 0.14 \
  --no-fallback \
  --output-dir "${OUT_ROOT}/hybrid_thr014_nofallback"

echo "[5] Hybrid threshold sweep"
for t in 0.10 0.12 0.14 0.15 0.16 0.18; do
  echo "--- sweep threshold=${t} ---"
  python examples/profile_hyperedge_pool_eval.py \
    --memory-json "${MEMORY_JSON}" \
    --questions-json "${QUESTIONS_JSON}" \
    --profile-typing-mode hybrid \
    --sufficiency-threshold "${t}" \
    --output-dir "${OUT_ROOT}/sweep/hybrid_thr_${t}"
done

echo "[6] Hybrid threshold sweep without fallback"
for t in 0.10 0.12 0.14 0.15 0.16 0.18; do
  echo "--- sweep threshold=${t} no-fallback ---"
  python examples/profile_hyperedge_pool_eval.py \
    --memory-json "${MEMORY_JSON}" \
    --questions-json "${QUESTIONS_JSON}" \
    --profile-typing-mode hybrid \
    --sufficiency-threshold "${t}" \
    --no-fallback \
    --output-dir "${OUT_ROOT}/sweep_nofallback/hybrid_thr_${t}"
done

echo "[7] Collect summaries"
python examples/collect_profile_eval_results.py \
  --root "${OUT_ROOT}" \
  --out-prefix "${OUT_ROOT}/summary"

echo "=============================================================================="
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Master log : ${MASTER_LOG}"
echo "Summary CSV: ${OUT_ROOT}/summary.csv"
echo "Summary JSON: ${OUT_ROOT}/summary.json"
echo "=============================================================================="
