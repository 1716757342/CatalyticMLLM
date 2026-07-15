#!/bin/bash
# Stage 3: IRFT (Iterative Reward Fine-Tuning) training script
# GP-GRPO: R_step3 = ω₁·R_step2 + ω₂·R_energy
# Based on Stage 2 GRPO training, introduce the energy reward and ExemplarPool iterative mechanism

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
export DS_SKIP_CUDA_CHECK=1

# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

# NCCL optimization (Stage 3 energy prediction inference takes longer, so use a longer timeout)
export NCCL_TIMEOUT=10800   # 3 hours
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

echo "============================================================================"
echo "Stage 3: IRFT (Iterative Reward Fine-Tuning)"
echo "GP-GRPO: R_step3 = ω₁·R_step2 + ω₂·R_energy"
echo "============================================================================"
echo ""
echo "New mechanisms in Stage 3:"
echo "  1. ExemplarPool: maintain a high-quality CIF example pool (random sampling, not similarity retrieval)"
echo "  2. Energy reward: use full multimodal inference to predict energy, R_energy = exp(-λ·|E_pred - E_target|)"
echo "  3. Composite reward: R_step3 = ω₁·R_step2 + ω₂·R_energy"
echo "  4. Iterative generation: starting from round 2, take one exemplar 3D structure from the pool as additional input"
echo "============================================================================"
echo ""

# ============================================================================
# Configuration parameters
# ============================================================================

# Input model: LoRA adapter trained in Stage 2 (or the merged model)
INPUT_MODEL="/path/to/checkpoint"

TRAINING_DATA="grpo_training_data.json"   # Same data format as Stage 2
OUTPUT_DIR="/path/to/CatalyticMLLM-V1/checkpoints/Qwen-irft-stage3-lora"

# ExemplarPool path (automatically loaded when resuming from a checkpoint)
EXEMPLAR_POOL_PATH="${OUTPUT_DIR}/exemplar_pool.json"

# LoRA configuration (keep consistent with Stage 2)
USE_LORA=True
LORA_R=64
LORA_ALPHA=16
LORA_DROPOUT=0.05
TUNE_MM_VISION=True
TUNE_MM_MLP=True
TUNE_MM_LLM=False

# GRPO sampling parameters
NUM_SAMPLES_PER_PROMPT=3    # Generate K candidates per prompt
TEMPERATURE=0.7
TOP_P=0.9
MAX_NEW_TOKENS=10          # CIF generation length (longer than Stage 2)
GRPO_BETA=0.1

# Stage 3 reward weights
STRUCTURE_REWARD_WEIGHT=0.7  # ω₁: structure quality reward weight
ENERGY_REWARD_WEIGHT=0.3     # ω₂: energy reward weight
ENERGY_LAMBDA=1.0            # λ: energy reward decay coefficient
EXEMPLAR_POOL_SIZE=50        # Maximum ExemplarPool capacity
MAX_ENERGY_PRED_TOKENS=64    # Maximum number of generated tokens for energy prediction

# Training parameters
LEARNING_RATE=5e-5           # Stage 3 learning rate (slightly smaller than Stage 2 for fine adjustment)
MM_PROJECTOR_LR=5e-5
VISION_TOWER_LR=5e-6
NUM_EPOCHS=2
BATCH_SIZE=1
GRADIENT_ACCUMULATION=4

# Hardware configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC_PER_NODE=4
MASTER_PORT=29503             # Use a different port from Stage 2 to avoid conflicts

# DeepSpeed configuration (reuse the ZeRO-2 configuration from Stage 2)
DEEPSPEED_CONFIG="qwen-vl-finetune/scripts/deepspeed_config_grpo_lora.json"
LOG_FILE="irft_stage3_training.log"

# ============================================================================
# Check files
# ============================================================================

if [ ! -d "$INPUT_MODEL" ]; then
    echo "Error: input model path does not exist: $INPUT_MODEL"
    echo "Please finish Stage 2 training first, or change INPUT_MODEL to the correct path"
    exit 1
fi

if [ ! -f "$TRAINING_DATA" ]; then
    echo "Error: training data does not exist: $TRAINING_DATA"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Start Stage 3 IRFT training
# ============================================================================

echo "Starting Stage 3 IRFT training..."
echo "Log file: $LOG_FILE"
echo "Using $NPROC_PER_NODE GPUs"
echo ""
echo "Stage 3 parameters:"
echo "  - Structure reward weight ω₁: $STRUCTURE_REWARD_WEIGHT"
echo "  - Energy reward weight ω₂: $ENERGY_REWARD_WEIGHT"
echo "  - Energy decay coefficient λ:  $ENERGY_LAMBDA"
echo "  - ExemplarPool capacity: $EXEMPLAR_POOL_SIZE"
echo "  - Candidates per prompt K: $NUM_SAMPLES_PER_PROMPT"
echo ""

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    qwen-vl-finetune/qwenvl/train/train_irft_stage3.py \
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
    --structure_reward_weight $STRUCTURE_REWARD_WEIGHT \
    --energy_reward_weight $ENERGY_REWARD_WEIGHT \
    --energy_lambda $ENERGY_LAMBDA \
    --exemplar_pool_size $EXEMPLAR_POOL_SIZE \
    --exemplar_pool_path "$EXEMPLAR_POOL_PATH" \
    --max_energy_pred_tokens $MAX_ENERGY_PRED_TOKENS \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --save_steps 10 \
    --save_total_limit 3 \
    --bf16 True \
    --tf32 True \
    --model_max_length 4096 \
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
    --ddp_timeout 10800 \
    --deepspeed "$DEEPSPEED_CONFIG" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================================"
echo "Stage 3 training completed!"
echo "LoRA adapter saved at: $OUTPUT_DIR"
echo "ExemplarPool saved at: $EXEMPLAR_POOL_PATH"
echo ""
echo "The three-stage training workflow is complete:"
echo "  Stage 1: SFT supervised fine-tuning"
echo "  Stage 2: GRPO structure-quality reinforcement fine-tuning  (grpo_online_train_lora.sh)"
echo "  Stage 3: IRFT energy + structure composite reward  (irft_stage3_train.sh)  ← current"
echo "============================================================================"
