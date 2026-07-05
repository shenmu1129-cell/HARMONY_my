#!/usr/bin/env bash
set -euo pipefail

# Run the semi-automatic user-profile hyperedge pool demo and retrieval-only eval.
# This script does not call OpenAI / DeepSeek / LLM judge and does not run stage 5/6.

OUT_ROOT=${1:-outputs/profile_eval}

echo "[1/4] Running semi-automatic profile hyperedge pool demo..."
python examples/user_profile_hyperedge_demo.py

echo "[2/4] Running rule profile typing eval..."
python examples/profile_hyperedge_pool_eval.py \
  --profile-typing-mode rule \
  --output-dir "${OUT_ROOT}/rule"

echo "[3/4] Running unsupervised profile discovery eval..."
python examples/profile_hyperedge_pool_eval.py \
  --profile-typing-mode unsupervised \
  --output-dir "${OUT_ROOT}/unsupervised"

echo "[4/4] Running hybrid profile discovery eval..."
python examples/profile_hyperedge_pool_eval.py \
  --profile-typing-mode hybrid \
  --output-dir "${OUT_ROOT}/hybrid"

echo "Done. Outputs written under ${OUT_ROOT}"
