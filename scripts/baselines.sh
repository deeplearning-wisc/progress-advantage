#!/usr/bin/env bash
# Off-the-shelf reward-model baselines.
#   ThinkPRM (process RM) -> uq | bon | fa     WildReward (outcome RM) -> uq | bon
#
#   TASK=uq  MODEL=wildreward PAIR=qwen3.5-9b                      bash scripts/baselines.sh
#   TASK=bon MODEL=wildreward PAIR=gemma4-4b                       bash scripts/baselines.sh
#   TASK=fa  MODEL=thinkprm   MODEL_ID=launch/ThinkPRM-14B         bash scripts/baselines.sh

set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

TASK="${TASK:-fa}"
MODEL="${MODEL:-thinkprm}"
MODEL_ID="${MODEL_ID:-}"          # ThinkPRM: launch/ThinkPRM-7B (default) or -14B
PAIR="${PAIR:-qwen3.5-9b}"        # which backbone's trajectories to score (uq/bon)
OUT_DIR="${OUT_DIR:-results/baselines}"
mkdir -p "${OUT_DIR}"

ID_ARGS=(); [ -n "${MODEL_ID}" ] && ID_ARGS=(--model-id "${MODEL_ID}")
TAG="${MODEL}"; [ -n "${MODEL_ID}" ] && TAG="$(basename "${MODEL_ID}")"

case "${TASK}" in
  uq)
    [ -d "data/tau2/greedy/${PAIR}" ] || python -m pa.data --scenario uq
    python -m runners.baselines --task uq --model "${MODEL}" "${ID_ARGS[@]}" \
      --traj-dir "data/tau2/greedy/${PAIR}" --domains airline retail \
      --out "${OUT_DIR}/uq_${PAIR}_${TAG}.json" ;;
  bon)
    [ -d "data/webshop/bon8/${PAIR}" ] || python -m pa.data --scenario tts
    python -m runners.baselines --task bon --model "${MODEL}" "${ID_ARGS[@]}" \
      --trials-dir "data/webshop/bon8/${PAIR}" --domain webshop --n-trials 8 \
      --out "${OUT_DIR}/bon_webshop_${PAIR}_${TAG}.json" ;;
  fa)
    [ -d "data/who_and_when" ] || python -m pa.data --scenario fa
    python -m runners.baselines --task fa --model "${MODEL}" "${ID_ARGS[@]}" \
      --data-dir data/who_and_when --splits Hand-Crafted Algorithm-Generated \
      --out "${OUT_DIR}/fa_${TAG}.json" ;;
  *) echo "unknown TASK=${TASK} (use uq|bon|fa)"; exit 1 ;;
esac
