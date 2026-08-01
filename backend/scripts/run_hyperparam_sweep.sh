#!/bin/bash
# Targeted hyperparameter sweep. Each run trains for 1 epoch and is evaluated
# on the full Spider dev set without self-correction, so results are comparable to the
# epoch-1 baseline (greedy + post-processing).

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

RUN () {
    tag=$1
    shift
    echo "=== Starting $tag ==="
    WANDB_NAME=$tag python training/train_qlora.py \
        --output_dir models/qlora-adapter-$tag \
        --num_epochs 1 \
        "$@" 2>&1 | tee logs/hyperparam_${tag}_train.log
    python scripts/run_baseline.py \
        --adapter_path models/qlora-adapter-$tag \
        --mode zero \
        --output outputs/qlora_${tag}.json 2>&1 | tee logs/hyperparam_${tag}_eval.log
    echo "=== Finished $tag ==="
}

RUN rank-8 --lora_r 8 --lora_alpha 16
RUN rank-32 --lora_r 32 --lora_alpha 64
RUN lr-5e5 --learning_rate 5e-5
RUN lr-2e4 --learning_rate 2e-4
RUN dropout-0.1 --lora_dropout 0.1
RUN mlp-adapters --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

echo "HYPERPARAM_DONE" > logs/hyperparam_done
