#!/bin/bash
# Run all ablation experiments
# Usage: bash scripts/run_all.sh [GPU_ID] [DATA_DIR] [TRAIN_GEN]

GPU=${1:-0}
DATA=${2:-"./data/GenImage"}
TRAIN_GEN=${3:-"stable_diffusion_v_1_4"}

export CUDA_VISIBLE_DEVICES=$GPU

METHODS=(
    "clip_linear"   # Baseline: frozen CLIP + linear
    "clip_lora"     # Ablation: CLIP + LoRA (no freq)
    "clip_freq"     # Ablation: CLIP + freq (no LoRA)
    "freqlora"      # Full method
)

for method in "${METHODS[@]}"; do
    echo ">>> Running: $method"
    python train.py \
        --data_dir "$DATA" \
        --train_gen "$TRAIN_GEN" \
        --method "$method" \
        --device cuda
    echo "<<< Done: $method"
    echo ""
done

echo "All experiments complete!"
