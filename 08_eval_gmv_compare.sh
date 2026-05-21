#!/bin/bash
# Compare GMV metrics between base model and GRPO-trained model.
#
# Usage:
#   bash 08_eval_gmv_compare.sh [--base MODEL_PATH] [--grpo MODEL_PATH] [--eval-data PARQUET] [--samples N]
#
# Output: outputs/results/gmv_base.json, outputs/results/gmv_grpo.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/08_eval_gmv.py"
DEFAULT_EVAL_DATA="${SCRIPT_DIR}/outputs/rl_data_enriched/eval_gmv.parquet"

BASE_MODEL="${BASE_MODEL:-/scratch/dyvm6xra/dyvm6xrauser45/fred/models--OpenOneRec--OneRec-1.7B-pretrain/snapshots/db455d0bdcf4b5e0b42f30c45d65260a49656a7f}"
GRPO_MODEL="${GRPO_MODEL:-/home/dyvm6xra/dyvm6xrauser45/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B-pro/snapshots/5dc1b097ab8194f48f14730e5400a276a22f4ca1}"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"
MAX_SAMPLES=-1
NUM_BEAMS=16
K_VALUES="1,5,10,16"
DEVICE="cuda"

RESULT_DIR="${SCRIPT_DIR}/outputs/results"
mkdir -p "${RESULT_DIR}"

BASE_JSON="${RESULT_DIR}/gmv_base.json"
GRPO_JSON="${RESULT_DIR}/gmv_grpo.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base) BASE_MODEL="$2"; shift 2 ;;
        --grpo) GRPO_MODEL="$2"; shift 2 ;;
        --eval-data) EVAL_DATA="$2"; shift 2 ;;
        --samples) MAX_SAMPLES="$2"; shift 2 ;;
        --num-beams) NUM_BEAMS="$2"; shift 2 ;;
        --k-values) K_VALUES="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash 08_eval_gmv_compare.sh [--base MODEL] [--grpo MODEL] [--eval-data DATA] [--samples N]"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

MAX_SAMPLES_ARG=""
if [[ "${MAX_SAMPLES}" -gt 0 ]]; then
    MAX_SAMPLES_ARG="--max-samples ${MAX_SAMPLES}"
fi

echo "==============================================="
echo "  GMV Eval Comparison"
echo "==============================================="
echo "  Base model:  ${BASE_MODEL}"
echo "  GRPO model:  ${GRPO_MODEL}"
echo "  Eval data:   ${EVAL_DATA}"
echo "  Max samples: ${MAX_SAMPLES}"
echo "  Num beams:   ${NUM_BEAMS}"
echo "  K values:    ${K_VALUES}"
echo "==============================================="

# Evaluate base model
if [[ -f "${BASE_JSON}" ]]; then
    echo "[base] Results already exist at ${BASE_JSON}, skipping."
else
    echo ""
    echo "=== Evaluating BASE model ==="
    python3 "${EVAL_SCRIPT}" \
        --model-path "${BASE_MODEL}" \
        --eval-data "${EVAL_DATA}" \
        --output-json "${BASE_JSON}" \
        --num-beams "${NUM_BEAMS}" \
        --k-values "${K_VALUES}" \
        --device "${DEVICE}" \
        ${MAX_SAMPLES_ARG}
fi

# Evaluate GRPO model
if [[ -f "${GRPO_JSON}" ]]; then
    echo "[grpo] Results already exist at ${GRPO_JSON}, skipping."
else
    echo ""
    echo "=== Evaluating GRPO model ==="
    python3 "${EVAL_SCRIPT}" \
        --model-path "${GRPO_MODEL}" \
        --eval-data "${EVAL_DATA}" \
        --output-json "${GRPO_JSON}" \
        --num-beams "${NUM_BEAMS}" \
        --k-values "${K_VALUES}" \
        --device "${DEVICE}" \
        ${MAX_SAMPLES_ARG}
fi

# Compare
echo ""
python3 "${EVAL_SCRIPT}" --compare "${BASE_JSON}" "${GRPO_JSON}"
