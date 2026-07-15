#!/bin/bash

### Complete QwenVL training launch script using the specified GPUs ###

echo "========= Script check: data_flatten is set to False ========="
echo "========= GPU check: using 8 GPUs ========="
echo "========= Text-only training ========="

# ======================
# Path and dataset configuration (recommended to define here)
# ======================
# Optimization 1: define path and dataset variables here
# Modify these paths according to your actual environment
MODEL_PATH="/path/to/base_model"
OUTPUT_DIR="/path/to/output_dir"
CACHE_DIR="./cache"
DATASETS="MOLECULE_RELAXED_ENERGY_TUSNS_24K%100"

# ======================
# Distributed training configuration
# ======================
# Specify GPUs 0, 1, 2, 3, 4, 5. This tells CUDA programs that only these GPUs are visible,
# and torchrun will renumber them as 0, 1, 2, 3, 4, 5
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NPROC_PER_NODE=8 # The number of processes must match the number of GPUs in CUDA_VISIBLE_DEVICES

MASTER_ADDR="10.3.2.7"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts

# ======================
# Performance recommendation
# ======================
# Optimization 2: recommendation about CUDA_LAUNCH_BLOCKING
# This parameter is used for CUDA error debugging; it makes the program execute synchronously and significantly slows training.
# Enable it only when encountering CUDA errors such as 'device-side assert triggered'.
# For production training, comment out or remove this line for best performance.
# export CUDA_LAUNCH_BLOCKING=1

# ======================
# Run training
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwen-vl-finetune/qwenvl/train/train_qwen.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm True \
         --tune_mm_vision False \
         --tune_mm_mlp True \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 True \
         --per_device_train_batch_size 1 \
         --gradient_accumulation_steps 1 \
         --learning_rate 1e-5 \
         --mm_projector_lr 1e-4 \
         --vision_tower_lr 1e-5 \
         --optim adamw_torch \
         --model_max_length 4096 \
         --data_flatten False \
         --data_packing False \
         --max_pixels 451584 \
         --min_pixels 12544 \
         --base_interval 2 \
         --video_max_frames 8 \
         --video_min_frames 3 \
         --video_max_frame_pixels 1304576 \
         --video_min_frame_pixels 200704 \
         --num_train_epochs 8 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type "cosine" \
         --weight_decay 0.01 \
         --logging_steps 1 \
         --save_steps 1000 \
         --save_total_limit 3 \
         --max_grad_norm 1.0 \
         --deepspeed qwen-vl-finetune/scripts/zero3.json