#!/usr/bin/env bash
set -euo pipefail

# Full-data minimal comparison for UP-HyperPool.
#
# This is the recommended first formal run when the goal is to prove the method
# against a HyperMem-style/global retrieval baseline without wasting time on many
# ablations.
#
# It runs on FULL converted data, not a subset.
#
# Configurations:
#   1. global_fact_retrieval: HyperMem-style/global fact retrieval proxy
#   2. hybrid_profile_pool : proposed user-profile-guided hyperedge pool
#
# Outputs:
#   <OUT_ROOT>/data_report.json
#   <OUT_ROOT>/full_minimal_compare_summary.csv
#   <OUT_ROOT>/full_minimal_compare_summary.json
#   <OUT_ROOT>/logs/*.log
#
# Usage:
#   bash scripts/run_profile_full_minimal_compare.sh SOURCE_DIR OUT_ROOT [THRESHOLD]
#
# Example:
#   bash scripts/run_profile_full_minimal_compare.sh /home/sutongtong/wwt/code outputs/profile_full_minimal_compare 0.14

SOURCE_DIR=${1:-/home/sutongtong/wwt/code}
OUT_ROOT=${2:-outputs/profile_full_minimal_compare}
THRESHOLD=${3:-0.14}
MAX_TOKENS=${4:-450}
GLOBAL_TOP_K=${5:-8}

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/data"
LOG_FILE="${OUT_ROOT}/logs/full_minimal_compare_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================================================="
echo "UP-HyperPool Full-data Minimal Comparison"
echo "=============================================================================="
echo "source_dir  : ${SOURCE_DIR}"
echo "out_root    : ${OUT_ROOT}"
echo "threshold   : ${THRESHOLD}"
echo "max_tokens  : ${MAX_TOKENS}"
echo "global_top_k: ${GLOBAL_TOP_K}"
echo "started_at  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "git_commit  : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python      : $(python --version)"
echo "=============================================================================="

python -m py_compile hypermem/profile_hyperedge_pool.py
python -m py_compile examples/profile_hyperedge_pool_eval.py
python -m py_compile examples/global_fact_retrieval_eval.py
python -m py_compile examples/prepare_profile_eval_data.py

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "ERROR: source directory not found: ${SOURCE_DIR}" >&2
  exit 2
fi

echo "[1/4] Preparing FULL converted data"
python examples/prepare_profile_eval_data.py \
  --source-dir "${SOURCE_DIR}" \
  --out-dir "${OUT_ROOT}/data" \
  --max-memory 1000000 \
  --max-questions 1000000

MEMORY_JSON="${OUT_ROOT}/data/locomo_memory_facts.jsonl"
QUESTIONS_JSON="${OUT_ROOT}/data/locomo_questions.jsonl"
MEMORY_N=$(wc -l < "${MEMORY_JSON}" | tr -d ' ')
QUESTION_N=$(wc -l < "${QUESTIONS_JSON}" | tr -d ' ')

python - <<PY
import json
from pathlib import Path
report = {
  "source_dir": "${SOURCE_DIR}",
  "memory_rows": int("${MEMORY_N}"),
  "question_rows": int("${QUESTION_N}"),
  "memory_path": "${MEMORY_JSON}",
  "questions_path": "${QUESTIONS_JSON}",
  "threshold": float("${THRESHOLD}"),
  "max_tokens": int("${MAX_TOKENS}"),
  "global_top_k": int("${GLOBAL_TOP_K}"),
}
out = Path("${OUT_ROOT}/data_report.json")
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

if [[ "${MEMORY_N}" -eq 0 || "${QUESTION_N}" -eq 0 ]]; then
  echo "ERROR: no usable full data extracted." >&2
  echo "Inspect ${OUT_ROOT}/data/profile_eval_data_report.json" >&2
  exit 3
fi

echo "[2/4] Running HyperMem-style/global fact retrieval baseline on FULL data"
python examples/global_fact_retrieval_eval.py \
  --memory-json "${MEMORY_JSON}" \
  --questions-json "${QUESTIONS_JSON}" \
  --max-tokens "${MAX_TOKENS}" \
  --top-k "${GLOBAL_TOP_K}" \
  --output-dir "${OUT_ROOT}/global_fact_retrieval"

echo "[3/4] Running proposed hybrid profile hyperedge pool on FULL data"
python examples/profile_hyperedge_pool_eval.py \
  --memory-json "${MEMORY_JSON}" \
  --questions-json "${QUESTIONS_JSON}" \
  --profile-typing-mode hybrid \
  --sufficiency-threshold "${THRESHOLD}" \
  --max-tokens "${MAX_TOKENS}" \
  --output-dir "${OUT_ROOT}/hybrid_profile_pool"

echo "[4/4] Writing comparison summary"
python - <<PY
import csv, json
from pathlib import Path
root = Path("${OUT_ROOT}")
base = json.loads((root / "global_fact_retrieval" / "global_fact_retrieval_summary.json").read_text(encoding="utf-8"))
hyb = json.loads((root / "hybrid_profile_pool" / "profile_hyperedge_pool_summary.json").read_text(encoding="utf-8"))
rows = []
rows.append({
    "method": "global_fact_retrieval_proxy",
    "n": base.get("n"),
    "hit": base.get("hit"),
    "recall": base.get("recall"),
    "tokens": base.get("tokens"),
    "reward": base.get("reward"),
    "fallback_rate": base.get("fallback_rate", 0.0),
    "fast_channel_rate": "",
    "num_edges": "",
    "active_edges": "",
    "discovery_buffer_size": "",
    "edge_type_counts": "",
})
rows.append({
    "method": "hybrid_profile_pool",
    "n": hyb.get("n"),
    "hit": hyb.get("hit"),
    "recall": hyb.get("recall"),
    "tokens": hyb.get("tokens"),
    "reward": hyb.get("reward"),
    "fallback_rate": hyb.get("fallback_rate"),
    "fast_channel_rate": round(1.0 - float(hyb.get("fallback_rate", 0.0)), 6),
    "num_edges": hyb.get("num_edges"),
    "active_edges": hyb.get("active_edges"),
    "discovery_buffer_size": hyb.get("discovery_buffer_size"),
    "edge_type_counts": json.dumps(hyb.get("edge_type_counts", {}), ensure_ascii=False),
})
fields = list(rows[0].keys())
with (root / "full_minimal_compare_summary.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
(root / "full_minimal_compare_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", root / "full_minimal_compare_summary.csv")
print("wrote", root / "full_minimal_compare_summary.json")
PY

echo "=============================================================================="
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log file    : ${LOG_FILE}"
echo "Data report : ${OUT_ROOT}/data_report.json"
echo "Summary CSV : ${OUT_ROOT}/full_minimal_compare_summary.csv"
echo "Summary JSON: ${OUT_ROOT}/full_minimal_compare_summary.json"
echo "=============================================================================="
cat "${OUT_ROOT}/full_minimal_compare_summary.csv"
