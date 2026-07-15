#!/bin/bash
# Real GRPO online training script

# Switch to the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Set PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}/qwen-vl-finetune:${PYTHONPATH}"

# Disable DeepSpeed custom optimizers
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_AIO=0
export DS_BUILD_SPARSE_ATTN=0

# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
# Clean up unused GPU memory fragments
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

# NCCL optimization - GRPO sampling requires a longer timeout
export NCCL_TIMEOUT=7200  # 2-hour timeout (the sampling stage is slow)
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

echo "============================================================================"
echo "GRPO online training - real Group Relative Policy Optimization"
echo "============================================================================"
echo ""
echo "How it works:"
echo "1. Sample K responses for each prompt (online generation)"
echo "2. Use the reward model to score these K responses"
echo "3. Compute the within-group relative advantage (advantage = reward - group_mean)"
echo "4. Optimize the policy based on advantages (maximize the probability of high-reward responses)"
echo ""
echo "Difference from DPO:"
echo "- DPO: uses pre-prepared chosen/rejected pairs (offline)"
echo "- GRPO: samples and scores in real time during training (online)"
echo "============================================================================"
echo ""

# ============================================================================
# Configuration parameters
# ============================================================================

INPUT_MODEL="/path/to/finetuned_model"
TRAINING_DATA="grpo_training_data.json"  # GRPO-format data (contains only prompts)
OUTPUT_DIR="/path/to/checkpoints/Qwen-grpo-online"

# GRPO-specific parameters
NUM_SAMPLES_PER_PROMPT=3    # Number of responses sampled per prompt (increase to reduce variance)
TEMPERATURE=0.7             # Sampling temperature
TOP_P=0.9                   # nucleus sampling
MAX_NEW_TOKENS=9120          # Maximum generation length (large enough for the model to naturally generate a complete CIF)
GRPO_BETA=0.1              # KL penalty coefficient

# Training parameters
LEARNING_RATE=5e-7  # Lower the learning rate to prevent training collapse
NUM_EPOCHS=2
BATCH_SIZE=1
GRADIENT_ACCUMULATION=4  # Increase gradient accumulation to reduce GPU memory pressure (optimization: 2→4)

# Hardware configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC_PER_NODE=4
MASTER_PORT=29501

# DeepSpeed configuration
DEEPSPEED_CONFIG="qwen-vl-finetune/scripts/deepspeed_config_grpo.json"
LOG_FILE="grpo_online_training.log"

# ============================================================================
# Step 1: Convert data format
# ============================================================================

if [ ! -f "$TRAINING_DATA" ]; then
    echo "Converting data format: DPO → GRPO ..."
    python3 convert_to_grpo_format.py test_with_gt.json "$TRAINING_DATA"
    echo "✓ Data conversion completed"
    echo ""
fi

# ============================================================================
# Step 2: Check files
# ============================================================================

if [ ! -d "$INPUT_MODEL" ]; then
    echo "Error: model path does not exist: $INPUT_MODEL"
    exit 1
fi

if [ ! -f "$TRAINING_DATA" ]; then
    echo "Error: training data does not exist: $TRAINING_DATA"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Step 3: Start GRPO training
# ============================================================================

echo "Starting GRPO online training..."
echo "Log file: $LOG_FILE"
echo "Using $NPROC_PER_NODE GPUs for distributed training"
echo ""

# Use torchrun to launch multi-GPU training
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    qwen-vl-finetune/qwenvl/train/train_grpo_online.py \
    --model_name_or_path "$INPUT_MODEL" \
    --data_path "$TRAINING_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --grpo_beta $GRPO_BETA \
    --num_samples_per_prompt $NUM_SAMPLES_PER_PROMPT \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --max_new_tokens $MAX_NEW_TOKENS \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --save_steps 10 \
    --save_total_limit 3 \
    --bf16 True \
    --tf32 True \
    --model_max_length 256 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --remove_unused_columns False \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --report_to "tensorboard" \
    --ddp_backend "nccl" \
    --ddp_find_unused_parameters False \
    --ddp_timeout 7200 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================================"
echo "Training completed!"
echo "Model saved at: $OUTPUT_DIR"
echo "============================================================================"



