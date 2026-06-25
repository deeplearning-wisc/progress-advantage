#!/usr/bin/env bash
# Uncertainty quantification on tau2-bench.
# Each pair scores its own greedy-decoding trajectories.

set -euo pipefail

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

PAIR="${PAIR:-qwen3.5-9b}"
TRAJ_DIR="${TRAJ_DIR:-data/tau2/greedy/${PAIR}}"
OUT_DIR="${OUT_DIR:-results/uq_onpolicy}"
DEVICE="${DEVICE:-cuda:0}"

# Best (token_agg, step_agg) combination per backbone and domain.
case "${PAIR}" in
  gemma4-4b)   AIR_TOK=max AIR_STEP=min  RET_TOK=min RET_STEP=last ;;
  qwen3.5-9b)  AIR_TOK=max AIR_STEP=mean RET_TOK=max RET_STEP=last ;;
  *) echo "no tuned aggregation for PAIR=${PAIR} (use gemma4-4b|qwen3.5-9b)"; exit 1 ;;
esac

[ -d "${TRAJ_DIR}" ] || python -m pa.data --scenario uq
mkdir -p "${OUT_DIR}"

python -m runners.tau2_uq \
  --traj-dir "${TRAJ_DIR}" \
  --out "${OUT_DIR}/${PAIR}_airline.json" \
  --policy-pair "${PAIR}" \
  --domains airline \
  --token-aggs "${AIR_TOK}" \
  --step-aggs "${AIR_STEP}" \
  --device-map "${DEVICE}"

python -m runners.tau2_uq \
  --traj-dir "${TRAJ_DIR}" \
  --out "${OUT_DIR}/${PAIR}_retail.json" \
  --policy-pair "${PAIR}" \
  --domains retail \
  --token-aggs "${RET_TOK}" \
  --step-aggs "${RET_STEP}" \
  --device-map "${DEVICE}"