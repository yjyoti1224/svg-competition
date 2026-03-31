#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Submit to Kaggle
# Usage: bash submit.sh [optional message]
# ──────────────────────────────────────────────────────────────
set -euo pipefail

COMP="dl-spring-2026-svg-generation"
SUBMISSION="output/submission.csv"
MESSAGE="${1:-Qwen2.5-Coder-7B QLoRA submission}"

if [ ! -f "${SUBMISSION}" ]; then
    echo "ERROR: ${SUBMISSION} not found. Run inference first."
    exit 1
fi

# Quick sanity check
ROWS=$(wc -l < "${SUBMISSION}")
echo "Submission file: ${SUBMISSION}"
echo "Rows (including header): ${ROWS}"

if [ "${ROWS}" -lt 1001 ]; then
    echo "WARNING: Expected 1001 rows (1 header + 1000 test), got ${ROWS}"
fi

echo ""
echo "Submitting to ${COMP} ..."
kaggle competitions submit -c "${COMP}" -f "${SUBMISSION}" -m "${MESSAGE}"

echo ""
echo "Check leaderboard: https://www.kaggle.com/competitions/${COMP}/leaderboard"
