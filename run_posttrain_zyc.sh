#!/bin/bash
# ZYC Post-Training Script
# 用彩色点云投影训练 multiview 模型

set -e

# 配置
DATASET_ROOT="/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud"
OUTPUT_ROOT="/mnt/zihanw/cosmos-transfer2.5/outputs"
NUM_GPUS=2
MASTER_PORT=12341

# 设置输出目录
export IMAGINAIRE_OUTPUT_ROOT=$OUTPUT_ROOT

echo "======================================"
echo "ZYC Post-Training"
echo "======================================"
echo "Dataset: $DATASET_ROOT"
echo "Output: $OUTPUT_ROOT"
echo "Num GPUs: $NUM_GPUS"
echo "======================================"
echo ""

# 运行训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m scripts.train \
    --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
    -- experiment=transfer2_auto_multiview_post_train_example \
    job.wandb_mode=disabled \
    job.name=zyc_pointcloud2rgb \
    job.group=zyc_posttrain \
    dataloader_train.dataset.dataset_dir=$DATASET_ROOT \
    trainer.max_iter=500 \
    trainer.validation_iter=100 \
    trainer.snapshot_save_iter=100 \
    trainer.logging_iter=10 \
    gen_opt.lr=1e-5

echo ""
echo "======================================"
echo "✅ Training completed!"
echo "Checkpoints saved to:"
echo "$OUTPUT_ROOT/cosmos_transfer_v2p5/zyc_posttrain/zyc_pointcloud2rgb/checkpoints"
echo "======================================"
