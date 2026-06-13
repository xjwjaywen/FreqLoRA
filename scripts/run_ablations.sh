#!/bin/bash
# Run ablation experiments: layer selection + LoRA rank
# Usage: bash scripts/run_ablations.sh [GPU_ID]

set -e
GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

SEED=42
DATA="./data/GenImage"
MULTI_GEN="sd14,biggan,adm"
MAX_TRAIN=3000

echo "=========================================="
echo "Ablation 1: Layer Selection"
echo "=========================================="

# Default: 2,5,8,11 (layers 3,6,9,12 in 1-indexed)
# ViT-B/16 has 12 blocks (0-11)

LAYER_CONFIGS=(
    "1,4,7,10"     # early-mid
    "2,5,8,11"     # default (mid, evenly spaced)
    "3,6,9,11"     # mid-late
    "0,3,7,11"     # wide spread
    "5,7,9,11"     # late layers only
)

for LAYERS in "${LAYER_CONFIGS[@]}"; do
    TAG=$(echo "$LAYERS" | tr ',' '')
    echo ">>> Layers=$LAYERS"
    python train.py \
        --data_dir "$DATA" \
        --method patchltd \
        --multi_gen "$MULTI_GEN" \
        --max_train "$MAX_TRAIN" \
        --seed "$SEED" \
        --selected_layers "$LAYERS"

    SRC="outputs/patchltd_${MULTI_GEN//,/+}_L${TAG}"
    if [ -f "$SRC/results.txt" ]; then
        cp "$SRC/results.txt" "results/ablation_layers_${TAG}.txt"
    fi
    echo "<<< Done Layers=$LAYERS"
done

echo ""
echo "=========================================="
echo "Ablation 2: LoRA Rank"
echo "=========================================="

for RANK in 4 8 16; do
    echo ">>> Rank=$RANK"
    python train.py \
        --data_dir "$DATA" \
        --method patchltd \
        --multi_gen "$MULTI_GEN" \
        --max_train "$MAX_TRAIN" \
        --seed "$SEED" \
        --lora_rank "$RANK"

    SRC="outputs/patchltd_${MULTI_GEN//,/+}_r${RANK}"
    # rank=8 is default, output dir won't have _r8 suffix
    if [ "$RANK" = "8" ]; then
        SRC="outputs/patchltd_${MULTI_GEN//,/+}"
    fi
    if [ -f "$SRC/results.txt" ]; then
        cp "$SRC/results.txt" "results/ablation_rank_${RANK}.txt"
    fi
    echo "<<< Done Rank=$RANK"
done

echo ""
echo "=========================================="
echo "All ablations complete!"
echo "=========================================="

echo ""
echo "=== Layer Selection Results ==="
for f in results/ablation_layers_*.txt; do
    LAYERS=$(basename "$f" .txt | sed 's/ablation_layers_//')
    Q30=$(grep "Q=30:" "$f" | head -1 | grep -oP 'acc=\K[0-9.]+')
    CLEAN=$(grep "best_val_acc:" "$f" | grep -oP '[0-9.]+')
    echo "  Layers=$LAYERS  clean=$CLEAN  Q30=$Q30"
done

echo ""
echo "=== LoRA Rank Results ==="
for f in results/ablation_rank_*.txt; do
    RANK=$(basename "$f" .txt | sed 's/ablation_rank_//')
    Q30=$(grep "Q=30:" "$f" | head -1 | grep -oP 'acc=\K[0-9.]+')
    CLEAN=$(grep "best_val_acc:" "$f" | grep -oP '[0-9.]+')
    echo "  Rank=$RANK  clean=$CLEAN  Q30=$Q30"
done
