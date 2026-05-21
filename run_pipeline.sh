#!/bin/bash
# Master pipeline: GMV-targeting GRPO data curation
#
# Stages:
#   1. Signal extraction (price lookup, engagement scores, CoT samples)
#   2. Dataset assembly (ranked product, purchase pred, CoT reasoning)
#   3. Merge & split (train/test/eval_gmv)
#
# Usage:
#   bash run_pipeline.sh [--skip-stage1] [--skip-stage2] [--skip-stage3]
#
# Output in: outputs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"

# Source dataset paths
SOURCE_DATA="/home/dyvm6xra/dyvm6xrauser45/.cache/huggingface/hub/datasets--OpenOneRec--OpenOneRec-RecIF/snapshots/8f7cf2ee0b949e955a87a708d02024687be232c8/onerec_bench_release.parquet"
VIDEO_PID2SID="/home/dyvm6xra/dyvm6xrauser45/.cache/huggingface/hub/datasets--OpenOneRec--OpenOneRec-RecIF/snapshots/8f7cf2ee0b949e955a87a708d02024687be232c8/video_ad_pid2sid.parquet"
PRODUCT_PID2SID="/home/dyvm6xra/dyvm6xrauser45/.cache/huggingface/hub/datasets--OpenOneRec--OpenOneRec-RecIF/snapshots/8f7cf2ee0b949e955a87a708d02024687be232c8/product_pid2sid.parquet"

SKIP_S1=false SKIP_S2=false SKIP_S3=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-stage1) SKIP_S1=true; shift ;;
        --skip-stage2) SKIP_S2=true; shift ;;
        --skip-stage3) SKIP_S3=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"

echo "==============================================="
echo "  GMV Data Curation Pipeline"
echo "==============================================="
echo "  Output: ${OUTPUT_DIR}"
echo "==============================================="

# ============================================================
# Stage 1: Signal Extraction
# ============================================================
if ! ${SKIP_S1}; then
    echo ""
    echo "===== STAGE 1: Signal Extraction ====="

    echo ""
    echo "[1/3] Building price lookup..."
    python3 "${SCRIPT_DIR}/01_build_price_lookup.py" \
        --input "${SOURCE_DATA}" \
        --product_pid2sid "${PRODUCT_PID2SID}" \
        --output_dir "${OUTPUT_DIR}"

    echo ""
    echo "[2/3] Computing engagement scores..."
    python3 "${SCRIPT_DIR}/02_compute_engagement_scores.py" \
        --input "${SOURCE_DATA}" \
        --output_dir "${OUTPUT_DIR}"

    echo ""
    echo "[3/3] Extracting CoT samples..."
    python3 "${SCRIPT_DIR}/03_extract_cot_samples.py" \
        --input "${SOURCE_DATA}" \
        --price_stats "${OUTPUT_DIR}/price_stats.json" \
        --output_dir "${OUTPUT_DIR}"

    echo ""
    echo "Stage 1 complete."
else
    echo "Skipping Stage 1."
fi

# ============================================================
# Stage 2: Dataset Assembly
# ============================================================
if ! ${SKIP_S2}; then
    echo ""
    echo "===== STAGE 2: Dataset Assembly ====="

    echo ""
    echo "[4/6] Building ranked product task..."
    python3 "${SCRIPT_DIR}/04_build_ranked_product_task.py" \
        --input "${SOURCE_DATA}" \
        --video_pid2sid "${VIDEO_PID2SID}" \
        --product_pid2sid "${PRODUCT_PID2SID}" \
        --item_prices "${OUTPUT_DIR}/item_price_lookup.parquet" \
        --cat_prices "${OUTPUT_DIR}/category_price_lookup.parquet" \
        --price_stats "${OUTPUT_DIR}/price_stats.json" \
        --engagement "${OUTPUT_DIR}/user_item_engagement.parquet" \
        --output_dir "${OUTPUT_DIR}"

    echo ""
    echo "[5/6] Building purchase prediction task..."
    python3 "${SCRIPT_DIR}/05_build_purchase_prediction_task.py" \
        --input "${SOURCE_DATA}" \
        --video_pid2sid "${VIDEO_PID2SID}" \
        --product_pid2sid "${PRODUCT_PID2SID}" \
        --item_prices "${OUTPUT_DIR}/item_price_lookup.parquet" \
        --cat_prices "${OUTPUT_DIR}/category_price_lookup.parquet" \
        --price_stats "${OUTPUT_DIR}/price_stats.json" \
        --output_dir "${OUTPUT_DIR}"

    echo ""
    echo "[6/6] Building CoT reasoning task..."
    python3 "${SCRIPT_DIR}/06_build_cot_reasoning_task.py" \
        --input "${SOURCE_DATA}" \
        --item_prices "${OUTPUT_DIR}/item_price_lookup.parquet" \
        --cat_prices "${OUTPUT_DIR}/category_price_lookup.parquet" \
        --price_stats "${OUTPUT_DIR}/price_stats.json" \
        --output_dir "${OUTPUT_DIR}"

    echo ""
    echo "Stage 2 complete."
else
    echo "Skipping Stage 2."
fi

# ============================================================
# Stage 3: Merge & Split
# ============================================================
if ! ${SKIP_S3}; then
    echo ""
    echo "===== STAGE 3: Merge & Split ====="

    RL_DATA_DIR="${OUTPUT_DIR}/rl_data_enriched"
    mkdir -p "${RL_DATA_DIR}"

    python3 "${SCRIPT_DIR}/07_merge_and_split.py" \
        --ranked_product "${OUTPUT_DIR}/ranked_product.parquet" \
        --purchase_pred "${OUTPUT_DIR}/purchase_pred.parquet" \
        --cot_reason "${OUTPUT_DIR}/cot_reason.parquet" \
        --output_dir "${RL_DATA_DIR}"

    echo ""
    echo "Stage 3 complete."
else
    echo "Skipping Stage 3."
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "==============================================="
echo "  Pipeline Complete!"
echo "==============================================="
echo "  Output files:"
for f in "${OUTPUT_DIR}"/*.parquet "${OUTPUT_DIR}"/*.json "${RL_DATA_DIR:-${OUTPUT_DIR}/rl_data_enriched}"/*.parquet 2>/dev/null; do
    if [[ -f "$f" ]]; then
        size=$(du -h "$f" | cut -f1)
        echo "    $(basename "$f")  (${size})"
    fi
done
echo ""
echo "  GMV eval data: ${RL_DATA_DIR:-${OUTPUT_DIR}/rl_data_enriched}/eval_gmv.parquet"
echo "  Train data:    ${RL_DATA_DIR:-${OUTPUT_DIR}/rl_data_enriched}/train.parquet"
echo "  Test data:     ${RL_DATA_DIR:-${OUTPUT_DIR}/rl_data_enriched}/test.parquet"
echo "==============================================="
