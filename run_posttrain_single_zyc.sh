#!/bin/bash
# ZYC Single-View Post-Training Script (2 GPU Setup)
# 仅训练前置摄像头视角

set -e

# 配置
DATASET_ROOT="/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud"
OUTPUT_ROOT="/mnt/zihanw/cosmos-transfer2.5/outputs"
NUM_GPUS=2
MASTER_PORT=12341

# 设置输出目录
export IMAGINAIRE_OUTPUT_ROOT=$OUTPUT_ROOT

echo "======================================"
echo "ZYC Single-View Post-Training (2 GPUs)"
echo "======================================"
echo "Dataset: $DATASET_ROOT"
echo "Output: $OUTPUT_ROOT"
echo "Num GPUs: $NUM_GPUS"
echo "Camera: Front Wide (120 FOV) only"
echo "======================================"
echo ""

# 运行训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m scripts.train \
    --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
    -- experiment=transfer2_auto_multiview_post_train_example \
    data_train=example_singleview_train_data_control_input_hdmap_single_zyc \
    job.wandb_mode=disabled \
    job.name=zyc_pointcloud2rgb_single \
    job.group=zyc_posttrain_single \
    dataloader_train.dataset.dataset_dir=$DATASET_ROOT \
    model_parallel.context_parallel_size=1 \
    trainer.max_iter=500 \
    trainer.validation_iter=100 \
    checkpoint.save_iter=100 \
    trainer.logging_iter=10 \
    optimizer.lr=1e-5

echo ""
echo "======================================"
echo "✅ Training completed!"
echo "Checkpoints saved to:"
echo "$OUTPUT_ROOT/cosmos_transfer_v2p5/zyc_posttrain_single/zyc_pointcloud2rgb_single/checkpoints"
echo "======================================"
