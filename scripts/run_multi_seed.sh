#!/bin/bash
# Run experiments with multiple seeds for statistical significance
# Usage: bash scripts/run_multi_seed.sh [GPU_ID] [METHOD] [TRAIN_GEN] [MAX_TRAIN]

GPU=${1:-0}
METHOD=${2:-"patchltd"}
TRAIN_GEN=${3:-"sd14"}
MAX_TRAIN=${4:-5000}
MULTI_GEN=${5:-""}  # e.g. "sd14,biggan,adm"

export CUDA_VISIBLE_DEVICES=$GPU

for SEED in 42 123 456; do
    echo ">>> Seed=$SEED | Method=$METHOD | Train=$TRAIN_GEN"

    if [ -n "$MULTI_GEN" ]; then
        python train.py \
            --data_dir ./data/GenImage \
            --method "$METHOD" \
            --multi_gen "$MULTI_GEN" \
            --max_train "$MAX_TRAIN"
    else
        python train.py \
            --data_dir ./data/GenImage \
            --train_gen "$TRAIN_GEN" \
            --method "$METHOD" \
            --max_train "$MAX_TRAIN"
    fi

    # Rename output with seed
    SRC="outputs/${METHOD}_${TRAIN_GEN}"
    if [ -n "$MULTI_GEN" ]; then
        SRC="outputs/${METHOD}_$(echo $MULTI_GEN | tr ',' '+')"
    fi
    DST="${SRC}_seed${SEED}"
    if [ -d "$SRC" ]; then
        cp -r "$SRC" "$DST"
        cp "$DST/results.txt" "results/$(basename $DST).txt" 2>/dev/null
    fi

    echo "<<< Done: Seed=$SEED"
done
echo "All seeds complete!"
