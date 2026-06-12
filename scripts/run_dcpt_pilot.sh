#!/bin/bash
# Run degradation-consistency pilot experiments.
# Usage: bash scripts/run_dcpt_pilot.sh [GPU_ID] [SETTING] [SEED] [METHODS] [MAX_TRAIN]
# Example: bash scripts/run_dcpt_pilot.sh 0 multi 42 "patchltd single_lora" 3000
#
# Optional:
#   EXTRA_EVAL=1 bash scripts/run_dcpt_pilot.sh 0 multi 42 "patchltd single_lora" 3000

set -e

GPU=${1:-0}
SETTING=${2:-"multi"}
SEED=${3:-42}
METHODS_ARG=${4:-"patchltd single_lora"}
MAX_TRAIN=${5:-3000}
EXTRA_EVAL=${EXTRA_EVAL:-0}

export CUDA_VISIBLE_DEVICES=$GPU

for METHOD in $METHODS_ARG; do
    echo ">>> DCPT | Seed=$SEED | Method=$METHOD | Setting=$SETTING"

    EXTRA_ARGS=()
    if [ "$EXTRA_EVAL" = "1" ]; then
        EXTRA_ARGS+=(--extra_degradation_eval)
    fi

    if [ "$SETTING" = "multi" ]; then
        python train.py \
            --data_dir ./data/GenImage \
            --method "$METHOD" \
            --multi_gen "sd14,biggan,adm" \
            --max_train "$MAX_TRAIN" \
            --seed "$SEED" \
            --degradation_consistency \
            --lambda_feat 0.1 \
            --lambda_pred 0.5 \
            --degradations "jpeg,blur,downsample" \
            "${EXTRA_ARGS[@]}"
        SRC="outputs/${METHOD}_dcpt_sd14+biggan+adm"
    else
        python train.py \
            --data_dir ./data/GenImage \
            --train_gen sd14 \
            --method "$METHOD" \
            --max_train "$MAX_TRAIN" \
            --seed "$SEED" \
            --degradation_consistency \
            --lambda_feat 0.1 \
            --lambda_pred 0.5 \
            --degradations "jpeg,blur,downsample" \
            "${EXTRA_ARGS[@]}"
        SRC="outputs/${METHOD}_dcpt_sd14"
    fi

    if [ -f "$SRC/results.txt" ]; then
        cp "$SRC/results.txt" "results/${METHOD}_dcpt_${SETTING}_seed${SEED}.txt"
    fi

    echo "<<< Done DCPT | Seed=$SEED | Method=$METHOD | Setting=$SETTING"
done
