#!/bin/bash
# Run all experiments for the paper with 3 seeds
# Usage: bash scripts/run_all.sh [GPU_ID] [SETTING]
# SETTING: "single" or "multi"

GPU=${1:-0}
SETTING=${2:-"single"}

export CUDA_VISIBLE_DEVICES=$GPU

METHODS=("single_lora" "patchltd" "cls_ltd" "patchltd_meanpool")
SEEDS=(42 123 456)

for SEED in "${SEEDS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        echo ">>> Seed=$SEED | Method=$METHOD | Setting=$SETTING"
        if [ "$SETTING" = "multi" ]; then
            python train.py \
                --data_dir ./data/GenImage \
                --method "$METHOD" \
                --multi_gen "sd14,biggan,adm" \
                --max_train 3000 \
                --seed "$SEED"
            SRC="outputs/${METHOD}_sd14+biggan+adm"
        else
            python train.py \
                --data_dir ./data/GenImage \
                --train_gen sd14 \
                --method "$METHOD" \
                --max_train 5000 \
                --seed "$SEED"
            SRC="outputs/${METHOD}_sd14"
        fi
        # Save with seed suffix
        if [ -f "$SRC/results.txt" ]; then
            cp "$SRC/results.txt" "results/${METHOD}_${SETTING}_seed${SEED}.txt"
        fi
        echo "<<< Done"
    done
done
echo "All experiments complete!"
