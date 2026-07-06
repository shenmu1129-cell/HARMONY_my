#!/usr/bin/env bash
set -euo pipefail

# One-command pipeline for the single retained method.
# Usage:
#   bash scripts/run_profile_centric_hypergraph.sh SOURCE_DIR OUT_ROOT [DATA_FRACTION] [TRAIN_RATIO]
#
# Useful controls:
#   CONSTRUCTION_MODE=batch|online_attach   default: batch
#   BATCH_SIZE=200                          facts per feature-induction batch
#   CANONICAL_THRESHOLD=0.62                feature canonicalization threshold
#   MIN_FEATURE_SUPPORT=1                   minimum facts for a batch feature
#   CONSOLIDATE_EVERY=5                     consolidate feature bank every N batches
#   MAX_EDGE_FACTS=600                      split oversized feature hyperedges
#   ONLINE_EVAL=0|1                         disable/enable online prequential eval
#   NO_PROGRESS=0|1                         disable progress bars when needed

SOURCE_DIR=${1:-DEMO}
OUT_ROOT=${2:-outputs/profile_centric_hg}
DATA_FRACTION=${3:-1.0}
TRAIN_RATIO=${4:-0.5}
ONLINE_EVAL=${ONLINE_EVAL:-1}
MAX_AUTO_EDGE_PAIRS=${MAX_AUTO_EDGE_PAIRS:-0}
NO_PROGRESS=${NO_PROGRESS:-0}
CONSTRUCTION_MODE=${CONSTRUCTION_MODE:-batch}
BATCH_SIZE=${BATCH_SIZE:-200}
CANONICAL_THRESHOLD=${CANONICAL_THRESHOLD:-0.62}
MIN_FEATURE_SUPPORT=${MIN_FEATURE_SUPPORT:-1}
CONSOLIDATE_EVERY=${CONSOLIDATE_EVERY:-5}
MAX_EDGE_FACTS=${MAX_EDGE_FACTS:-600}

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/data_full" "${OUT_ROOT}/data_used"
LOG_FILE="${OUT_ROOT}/logs/profile_centric_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

PIPELINE_START=$(date +%s)
STAGE_START=0

stage_begin() {
  local idx=$1
  local total=$2
  local msg=$3
  STAGE_START=$(date +%s)
  echo ""
  echo "=============================================================================="
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
  echo "DONE stage ${idx}/${total}: ${msg} | elapsed=${elapsed}s"
}

echo "=============================================================================="
echo "Profile-Centric Hypergraph Memory Runner"
echo "=============================================================================="
echo "source_dir          : ${SOURCE_DIR}"
echo "out_root            : ${OUT_ROOT}"
echo "data_fraction       : ${DATA_FRACTION}"
echo "train_ratio         : ${TRAIN_RATIO}"
echo "online_eval         : ${ONLINE_EVAL}"
echo "construction_mode   : ${CONSTRUCTION_MODE}"
echo "batch_size          : ${BATCH_SIZE}"
echo "canonical_threshold : ${CANONICAL_THRESHOLD}"
echo "min_feature_support : ${MIN_FEATURE_SUPPORT}"
echo "consolidate_every   : ${CONSOLIDATE_EVERY}"
echo "max_edge_facts      : ${MAX_EDGE_FACTS}"
echo "max_auto_edge_pairs : ${MAX_AUTO_EDGE_PAIRS}"
echo "no_progress         : ${NO_PROGRESS}"
echo "started_at          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "git_commit          : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python              : $(python --version)"
echo "=============================================================================="

python -m py_compile hypermem/profile_centric_hypergraph.py
python -m py_compile hypermem/batch_profile_builder.py
python -m py_compile examples/profile_centric_hypergraph_eval.py
python -m py_compile examples/prepare_profile_centric_data.py

TOTAL_STAGES=3
PROGRESS_ARG=--show-progress
if [[ "${NO_PROGRESS}" == "1" ]]; then
  PROGRESS_ARG=""
fi

stage_begin 1 "${TOTAL_STAGES}" "Prepare memory facts and QA"
if [[ "${SOURCE_DIR}" == "DEMO" ]]; then
  python examples/prepare_profile_centric_data.py \
    --demo \
    --out-dir "${OUT_ROOT}/data_full" \
    ${PROGRESS_ARG}
else
  python examples/prepare_profile_centric_data.py \
    --source-dir "${SOURCE_DIR}" \
    --out-dir "${OUT_ROOT}/data_full" \
    --max-memory 1000000 \
    --max-questions 1000000 \
    ${PROGRESS_ARG}
fi
stage_end 1 "${TOTAL_STAGES}" "Prepare memory facts and QA"

FULL_MEMORY="${OUT_ROOT}/data_full/memory_facts.jsonl"
FULL_QUESTIONS="${OUT_ROOT}/data_full/questions.jsonl"
USED_MEMORY="${OUT_ROOT}/data_used/memory_facts.jsonl"
USED_QUESTIONS="${OUT_ROOT}/data_used/questions.jsonl"

stage_begin 2 "${TOTAL_STAGES}" "Select data fraction"
python - <<PY
from pathlib import Path
import json
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

show_progress = "${NO_PROGRESS}" != "1" and tqdm is not None
steps = ["read full files", "compute fraction", "write used files", "write report"]
bar = tqdm(steps, desc="[stage2] select fraction", dynamic_ncols=True) if show_progress else steps

fraction = float("${DATA_FRACTION}")
fraction = min(1.0, max(0.01, fraction))
full_memory = Path("${FULL_MEMORY}")
full_questions = Path("${FULL_QUESTIONS}")
used_memory = Path("${USED_MEMORY}")
used_questions = Path("${USED_QUESTIONS}")

for step in bar:
    if step == "read full files":
        mem_lines = full_memory.read_text(encoding="utf-8").splitlines()
        q_lines = full_questions.read_text(encoding="utf-8").splitlines()
    elif step == "compute fraction":
        mem_n = max(1, int(len(mem_lines) * fraction)) if mem_lines else 0
        q_n = max(1, int(len(q_lines) * fraction)) if q_lines else 0
    elif step == "write used files":
        used_memory.write_text("\n".join(mem_lines[:mem_n]) + ("\n" if mem_n else ""), encoding="utf-8")
        used_questions.write_text("\n".join(q_lines[:q_n]) + ("\n" if q_n else ""), encoding="utf-8")
    elif step == "write report":
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
          "online_eval": int("${ONLINE_EVAL}"),
          "construction_mode": "${CONSTRUCTION_MODE}",
          "batch_size": int("${BATCH_SIZE}"),
          "canonical_threshold": float("${CANONICAL_THRESHOLD}"),
          "min_feature_support": int("${MIN_FEATURE_SUPPORT}"),
          "consolidate_every": int("${CONSOLIDATE_EVERY}"),
          "max_edge_facts": int("${MAX_EDGE_FACTS}"),
          "max_auto_edge_pairs": int("${MAX_AUTO_EDGE_PAIRS}"),
        }
        out = Path("${OUT_ROOT}/data_report.json")
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
PY
stage_end 2 "${TOTAL_STAGES}" "Select data fraction"

stage_begin 3 "${TOTAL_STAGES}" "Build profile hypergraph, train reward utility, embed/rank/retrieve, evaluate accuracy"
EVAL_ARGS=(
  --memory-json "${USED_MEMORY}"
  --questions-json "${USED_QUESTIONS}"
  --train-ratio "${TRAIN_RATIO}"
  --construction-mode "${CONSTRUCTION_MODE}"
  --batch-size "${BATCH_SIZE}"
  --canonical-threshold "${CANONICAL_THRESHOLD}"
  --min-feature-support "${MIN_FEATURE_SUPPORT}"
  --consolidate-every "${CONSOLIDATE_EVERY}"
  --max-edge-facts "${MAX_EDGE_FACTS}"
  --max-auto-edge-pairs "${MAX_AUTO_EDGE_PAIRS}"
  --output-dir "${OUT_ROOT}/eval"
)
if [[ "${ONLINE_EVAL}" == "1" ]]; then
  EVAL_ARGS+=(--online-eval)
fi
if [[ "${NO_PROGRESS}" == "1" ]]; then
  EVAL_ARGS+=(--no-progress)
fi
python examples/profile_centric_hypergraph_eval.py "${EVAL_ARGS[@]}"
stage_end 3 "${TOTAL_STAGES}" "Build profile hypergraph, train reward utility, embed/rank/retrieve, evaluate accuracy"

PIPELINE_END=$(date +%s)
PIPELINE_ELAPSED=$(( PIPELINE_END - PIPELINE_START ))

echo "=============================================================================="
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total elapsed: ${PIPELINE_ELAPSED}s"
echo "Log file    : ${LOG_FILE}"
echo "Data report : ${OUT_ROOT}/data_report.json"
echo "Summary CSV : ${OUT_ROOT}/eval/profile_centric_summary.csv"
echo "Summary JSON: ${OUT_ROOT}/eval/profile_centric_summary.json"
echo "=============================================================================="
cat "${OUT_ROOT}/eval/profile_centric_summary.csv"
