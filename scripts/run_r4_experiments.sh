#!/bin/bash
# R4 reviewer experiments: more seeds + ViT-L/14 + no-aug multi-seed
# Usage: bash scripts/run_r4_experiments.sh [PHASE]
# PHASE 1: extra seeds for main experiments (n=5→10)
# PHASE 2: no-aug multi-seed (n=1→5)
# PHASE 3: ViT-L/14 backbone
# Run all phases: bash scripts/run_r4_experiments.sh all

set -e
PHASE=${1:-all}
DATA="./data/GenImage"
MULTI_GEN="sd14,biggan,adm"
MAX_TRAIN=3000

run_one() {
    local GPU=$1 METHOD=$2 SEED=$3 EXTRA=$4
    echo "[GPU$GPU] $METHOD seed=$SEED $EXTRA"
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --data_dir "$DATA" --method "$METHOD" \
        --multi_gen "$MULTI_GEN" --max_train $MAX_TRAIN \
        --seed "$SEED" $EXTRA
}

copy_result() {
    local PATTERN=$1 DEST=$2
    for d in outputs/${PATTERN}/results.txt; do
        if [ -f "$d" ]; then cp "$d" "$DEST" && echo "  -> $DEST"; fi
    done
}

# =============================================
# PHASE 1: 5 more seeds for multi-gen (n=5→10)
# Seeds: 2024, 3000, 4000, 5000, 6000
# 4 methods × 5 seeds = 20 tasks
# =============================================
if [[ "$PHASE" == "1" || "$PHASE" == "all" ]]; then
echo "===== PHASE 1: Extra seeds (multi-gen) ====="
for SEED in 2024 3000 4000 5000 6000; do
    run_one 0 single_lora    $SEED "" &
    run_one 1 patchltd       $SEED "" &
    run_one 2 cls_ltd        $SEED "" &
    run_one 3 patchltd_meanpool $SEED "" &
    wait
    for M in single_lora patchltd cls_ltd patchltd_meanpool; do
        copy_result "${M}_sd14+biggan+adm_s${SEED}" "results/${M}_multi_seed${SEED}.txt"
    done
done
echo "===== PHASE 1 DONE ====="
fi

# =============================================
# PHASE 2: no-aug with 4 more seeds (n=1→5)
# Seeds: 789, 1024, 2024, 3000
# 4 methods × 4 seeds = 16 tasks
# =============================================
if [[ "$PHASE" == "2" || "$PHASE" == "all" ]]; then
echo "===== PHASE 2: No-aug multi-seed ====="
for SEED in 789 1024 2024 3000; do
    run_one 0 single_lora    $SEED "--no_jpeg_aug" &
    run_one 1 patchltd       $SEED "--no_jpeg_aug" &
    run_one 2 cls_ltd        $SEED "--no_jpeg_aug" &
    run_one 3 patchltd_meanpool $SEED "--no_jpeg_aug" &
    wait
    for M in single_lora patchltd cls_ltd patchltd_meanpool; do
        copy_result "${M}_sd14+biggan+adm_noaug_s${SEED}" "results/${M}_noaug_multi_seed${SEED}.txt"
    done
done
echo "===== PHASE 2 DONE ====="
fi

# =============================================
# PHASE 3: ViT-L/14 backbone
# PatchLTD + SingleLoRA × 3 seeds
# ViT-L/14 has 24 blocks → layers [5,11,17,23]
# =============================================
if [[ "$PHASE" == "3" || "$PHASE" == "all" ]]; then
echo "===== PHASE 3: ViT-L/14 ====="
CLIP_ARGS="--clip_model ViT-L-14 --clip_pretrained laion2b_s32b_b82k --selected_layers 5,11,17,23"
for SEED in 42 123 456; do
    run_one 0 patchltd    $SEED "$CLIP_ARGS" &
    run_one 1 single_lora $SEED "$CLIP_ARGS" &
    wait
    for M in patchltd single_lora; do
        copy_result "${M}_sd14+biggan+adm_ViT-L-14_L511723_s${SEED}" "results/${M}_vitl14_multi_seed${SEED}.txt"
    done
done
echo "===== PHASE 3 DONE ====="
fi

echo "===== ALL PHASES COMPLETE ====="
