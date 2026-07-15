#!/bin/bash
# GRPO online training script - LoRA version
# Use LoRA for parameter-efficient fine-tuning, greatly reducing GPU memory requirements and training cost

# Switch to the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Set PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}/qwen-vl-finetune:${PYTHONPATH}"

# Disable DeepSpeed custom optimizers and JIT compilation
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_AIO=0
export DS_BUILD_SPARSE_ATTN=0
export DS_SKIP_CUDA_CHECK=1  # Skip CUDA version check

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
echo "GRPO online training - LoRA version"
echo "============================================================================"
echo ""
echo "LoRA advantages:"
echo "- Trains only a small number of adapter parameters (~1-5% of model parameters)"
echo "- Significantly reduces GPU memory requirements (can use ZeRO-2 instead of ZeRO-3)"
echo "- Faster training speed and more stable convergence"
echo "- Can save multiple task-specific LoRA adapters"
echo ""
echo "How GRPO works:"
echo "1. Sample K responses for each prompt (online generation)"
echo "2. Use the reward model to score these K responses"
echo "3. Compute the within-group relative advantage (advantage = reward - group_mean)"
echo "4. Optimize the policy based on advantages (maximize the probability of high-reward responses)"
echo "============================================================================"
echo ""

# ============================================================================
# Configuration parameters
# ============================================================================

INPUT_MODEL="/path/to/finetuned_model"
TRAINING_DATA="grpo_training_data.json"  # GRPO-format data (contains only prompts)
OUTPUT_DIR="/path/to/checkpoints/Qwen-grpo-online-lora"

# LoRA configuration parameters
USE_LORA=True
LORA_R=64                   # LoRA rank, controls the number of adapter parameters
LORA_ALPHA=16               # LoRA alpha, usually set to 1/4 to 1/2 of the rank
LORA_DROPOUT=0.05           # LoRA dropout, prevents overfitting
TUNE_MM_VISION=True         # Whether to fine-tune the vision encoder
TUNE_MM_MLP=True            # Whether to fine-tune the MLP projection layer
TUNE_MM_LLM=False           # Whether to fine-tune the LLM (LoRA automatically handles the LLM part)

# GRPO-specific parameters
NUM_SAMPLES_PER_PROMPT=3    # Number of responses sampled per prompt
TEMPERATURE=0.7             # Sampling temperature
TOP_P=0.9                   # nucleus sampling
MAX_NEW_TOKENS=20         ######### Maximum generation length
GRPO_BETA=0.1              # KL penalty coefficient

# Training parameters (LoRA can use a higher learning rate)
LEARNING_RATE=1e-4          # LoRA learning rate (higher than full-parameter fine-tuning)
MM_PROJECTOR_LR=1e-4        # MLP projection layer learning rate
VISION_TOWER_LR=1e-5        # Vision encoder learning rate (smaller)
NUM_EPOCHS=2
BATCH_SIZE=1                # Batch size per GPU
GRADIENT_ACCUMULATION=4     # Gradient accumulation steps

# Hardware configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC_PER_NODE=4
MASTER_PORT=29502

# DeepSpeed configuration (uses ZeRO-2 + CPU offload; LoRA does not need ZeRO-3)
DEEPSPEED_CONFIG="qwen-vl-finetune/scripts/deepspeed_config_grpo_lora.json"
LOG_FILE="grpo_online_training_lora.log"

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
# Step 3: Start GRPO LoRA training
# ============================================================================

echo "Starting GRPO online training (LoRA version)..."
echo "Log file: $LOG_FILE"
echo "Using $NPROC_PER_NODE GPUs for distributed training"
echo ""
echo "LoRA configuration:"
echo "  - Rank: $LORA_R"
echo "  - Alpha: $LORA_ALPHA"
echo "  - Dropout: $LORA_DROPOUT"
echo "  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj"
echo ""

# Use torchrun to launch multi-GPU training
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    qwen-vl-finetune/qwenvl/train/train_grpo_online_lora.py \
    --model_name_or_path "$INPUT_MODEL" \
    --data_path "$TRAINING_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --use_lora $USE_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --tune_mm_llm $TUNE_MM_LLM \
    --tune_mm_vision $TUNE_MM_VISION \
    --tune_mm_mlp $TUNE_MM_MLP \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --vision_tower_lr $VISION_TOWER_LR \
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
    --deepspeed "$DEEPSPEED_CONFIG" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================================"
echo "Training completed!"
echo "LoRA adapter saved at: $OUTPUT_DIR"
echo ""
echo "Merge LoRA weights into the base model:"
echo "python Q_merge_lora_weights.py \\"
echo "    --base_model $INPUT_MODEL \\"
echo "    --lora_adapter $OUTPUT_DIR \\"
echo "    --output_path ${OUTPUT_DIR}_merged"
echo "============================================================================"
