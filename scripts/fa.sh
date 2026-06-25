#!/usr/bin/env bash
# Failure attribution on Who&When.

set -euo pipefail

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

PAIR="${PAIR:-qwen3.5-9b}"
DATA_DIR="${DATA_DIR:-data/who_and_when}"
OUT_DIR="${OUT_DIR:-results/fa}"
DEVICE="${DEVICE:-cuda:0}"

[ -d "${DATA_DIR}" ] || python -m pa.data --scenario fa
mkdir -p "${OUT_DIR}"

python -m runners.fa \
  --data-dir "${DATA_DIR}" \
  --out "${OUT_DIR}/${PAIR}.json" \
  --policy-pair "${PAIR}" \
  --splits Hand-Crafted Algorithm-Generated \
  --methods progress_advantage self_certainty policy_logprob \
  --token-aggs sum \
  --device-map "${DEVICE}"
