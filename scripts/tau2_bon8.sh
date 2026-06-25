#!/usr/bin/env bash
# Best-of-8 selection on tau2-Airline.
# Cached rollouts must already be at data/tau2/bon8/${PAIR}/trial_<i>/<domain>.json.

set -euo pipefail

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

PAIR="${PAIR:-qwen3.5-9b}"
DOMAIN="${DOMAIN:-airline}"
METHOD="${METHOD:-progress_advantage}"
TRIALS_DIR="${TRIALS_DIR:-data/tau2/bon8/${PAIR}}"
OUT_DIR="${OUT_DIR:-results/bon8}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUT_DIR}"

python -m runners.tau2_bon \
  --trials-dir "${TRIALS_DIR}" \
  --out "${OUT_DIR}/${PAIR}_${DOMAIN}_${METHOD}.json" \
  --policy-pair "${PAIR}" \
  --n-trials 8 \
  --domain "${DOMAIN}" \
  --method "${METHOD}" \
  --token-agg mean \
  --step-agg min \
  --device-map "${DEVICE}"
