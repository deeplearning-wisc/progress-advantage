#!/usr/bin/env bash
# Best-of-8 selection on WebShop.
# Rollouts are fetched on demand to data/webshop/bon8/${PAIR}/trial_<i>/webshop.json.

set -euo pipefail

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

PAIR="${PAIR:-qwen3.5-9b}"
METHOD="${METHOD:-progress_advantage}"
TRIALS_DIR="${TRIALS_DIR:-data/webshop/bon8/${PAIR}}"
OUT_DIR="${OUT_DIR:-results/bon8}"
DEVICE="${DEVICE:-cuda:0}"

# Per-backbone tuned token/step aggregation for the best-of-8 results
# (override with TOKEN_AGG / STEP_AGG).
case "${PAIR}" in
  gemma4-4b)  DEF_TOK=max;  DEF_STEP=min  ;;
  qwen3.5-9b) DEF_TOK=min;  DEF_STEP=last ;;
  *)          DEF_TOK=mean; DEF_STEP=mean ;;
esac
TOKEN_AGG="${TOKEN_AGG:-${DEF_TOK}}"
STEP_AGG="${STEP_AGG:-${DEF_STEP}}"

[ -d "${TRIALS_DIR}" ] || python -m pa.data --scenario tts
mkdir -p "${OUT_DIR}"

python -m runners.tau2_bon \
  --trials-dir "${TRIALS_DIR}" \
  --out "${OUT_DIR}/webshop_${PAIR}_${METHOD}.json" \
  --policy-pair "${PAIR}" \
  --n-trials 8 \
  --domain webshop \
  --method "${METHOD}" \
  --token-agg "${TOKEN_AGG}" \
  --step-agg "${STEP_AGG}" \
  --device-map "${DEVICE}"
