#!/usr/bin/env bash
set -euo pipefail

# Run the user-profile guided hyperedge pool demo and retrieval-only eval.
# This script does not call OpenAI / DeepSeek / LLM judge and does not run stage 5/6.

OUT_DIR=${1:-outputs/profile_eval}

echo "[1/2] Running profile hyperedge pool demo..."
python examples/user_profile_hyperedge_demo.py

echo "[2/2] Running profile hyperedge pool retrieval-only eval..."
python examples/profile_hyperedge_pool_eval.py --output-dir "${OUT_DIR}"

echo "Done. Outputs written to ${OUT_DIR}"
