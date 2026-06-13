#!/bin/bash
# Run 2 additional seeds (789, 1024) for all main experiments + ablations
# Usage: bash scripts/run_extra_seeds.sh [GPU_START]
# With 4 GPUs: assigns tasks round-robin to GPU 0-3

set -e
GPU_START=${1:-0}
NUM_GPUS=4
DATA="./data/GenImage"

declare -a TASKS=()

# --- Main experiments: 2 new seeds × 4 methods × 2 settings = 16 tasks ---
for SEED in 789 1024; do
  for METHOD in single_lora patchltd cls_ltd patchltd_meanpool; do
    # Single-gen
    TASKS+=("$SEED|$METHOD|single|sd14|5000|")
    # Multi-gen
    TASKS+=("$SEED|$METHOD|multi|sd14,biggan,adm|3000|")
  done
done

# --- Layer ablation: 2 new seeds × 5 configs = 10 tasks ---
for SEED in 789 1024; do
  for LAYERS in "1,4,7,10" "2,5,8,11" "3,6,9,11" "0,3,7,11" "5,7,9,11"; do
    TASKS+=("$SEED|patchltd|multi|sd14,biggan,adm|3000|$LAYERS")
  done
done

# --- LoRA rank ablation: 2 new seeds × 2 ranks = 4 tasks ---
for SEED in 789 1024; do
  for RANK in 4 16; do
    TASKS+=("$SEED|patchltd|multi_r${RANK}|sd14,biggan,adm|3000|")
  done
done

TOTAL=${#TASKS[@]}
echo "Total tasks: $TOTAL"
echo "GPUs: $GPU_START to $((GPU_START + NUM_GPUS - 1))"

run_task() {
    local TASK=$1
    local GPU=$2
    IFS='|' read -r SEED METHOD SETTING GENS MAX_TRAIN LAYERS <<< "$TASK"

    local EXTRA_ARGS=""
    local RESULT_NAME=""

    if [[ "$SETTING" == "single" ]]; then
        EXTRA_ARGS="--train_gen $GENS"
        RESULT_NAME="results/${METHOD}_single_seed${SEED}.txt"
        local SRC_DIR="outputs/${METHOD}_${GENS}"
    elif [[ "$SETTING" == multi_r* ]]; then
        local RANK=${SETTING#multi_r}
        EXTRA_ARGS="--multi_gen $GENS --lora_rank $RANK"
        RESULT_NAME="results/ablation_rank_${RANK}_seed${SEED}.txt"
        local GEN_TAG=$(echo "$GENS" | tr ',' '+')
        local SRC_DIR="outputs/${METHOD}_${GEN_TAG}_r${RANK}"
    elif [[ -n "$LAYERS" ]]; then
        EXTRA_ARGS="--multi_gen $GENS --selected_layers $LAYERS"
        local LTAG=$(echo "$LAYERS" | tr -d ',')
        RESULT_NAME="results/ablation_layers_${LTAG}_seed${SEED}.txt"
        local GEN_TAG=$(echo "$GENS" | tr ',' '+')
        local SRC_DIR="outputs/${METHOD}_${GEN_TAG}_L${LTAG}"
    else
        EXTRA_ARGS="--multi_gen $GENS"
        local GEN_TAG=$(echo "$GENS" | tr ',' '+')
        RESULT_NAME="results/${METHOD}_multi_seed${SEED}.txt"
        local SRC_DIR="outputs/${METHOD}_${GEN_TAG}"
    fi

    echo "[GPU$GPU] SEED=$SEED METHOD=$METHOD SETTING=$SETTING"
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --data_dir "$DATA" \
        --method "$METHOD" \
        --max_train "$MAX_TRAIN" \
        --seed "$SEED" \
        $EXTRA_ARGS

    if [ -f "$SRC_DIR/results.txt" ]; then
        cp "$SRC_DIR/results.txt" "$RESULT_NAME"
        echo "[GPU$GPU] Saved: $RESULT_NAME"
    else
        echo "[GPU$GPU] WARNING: $SRC_DIR/results.txt not found"
    fi
}

# Distribute tasks across GPUs
for i in "${!TASKS[@]}"; do
    GPU=$(( (i % NUM_GPUS) + GPU_START ))
    run_task "${TASKS[$i]}" "$GPU" &

    # Wait for batch to complete every NUM_GPUS tasks
    if (( (i + 1) % NUM_GPUS == 0 )); then
        wait
        echo "=== Batch $((i / NUM_GPUS + 1)) complete ==="
    fi
done
wait
echo "=== ALL $TOTAL TASKS COMPLETE ==="
