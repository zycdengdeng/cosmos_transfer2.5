#!/usr/bin/env python3
"""
ZYC Post-Training Script for Point Cloud to RGB
使用方法：
    torchrun --nproc_per_node=2 --master_port=12341 \
        -m scripts.train_zyc_posttrain \
        --output_dir /mnt/zihanw/cosmos-transfer2.5/outputs/posttrain_zyc
"""

import os
import sys
from pathlib import Path

# 设置环境变量
os.environ.setdefault("IMAGINAIRE_OUTPUT_ROOT", "/mnt/zihanw/cosmos-transfer2.5/outputs")

# 数据配置
DATASET_ROOT = Path("/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud")
TRAIN_SAMPLES = ["002", "004", "006", "008"]  # 训练样本
VAL_SAMPLES = ["009"]  # 验证样本

# 训练参数
CONFIG = {
    "experiment": "zyc_posttrain_rgbcloud",
    "job": {
        "project": "cosmos_transfer_v2p5",
        "group": "zyc_posttrain",
        "name": "rgbcloud_pointcloud2rgb",
        "wandb_mode": "disabled",
    },
    "data_train": {
        "root": str(DATASET_ROOT),
        "train_samples": TRAIN_SAMPLES,
        "n_views": 7,
    },
    "data_val": {
        "root": str(DATASET_ROOT),
        "val_samples": VAL_SAMPLES,
        "n_views": 7,
    },
    "trainer": {
        "max_iter": 500,  # 训练步数（小数据集，不需要太多）
        "validation_iter": 100,  # 每 100 步验证一次
        "snapshot_save_iter": 100,  # 每 100 步保存一次
        "logging_iter": 10,  # 每 10 步打印日志
    },
    "gen_opt": {
        "lr": 1e-5,  # 学习率（post-training 用小学习率）
    },
}

def main():
    # 运行训练
    cmd = f"""
torchrun --nproc_per_node=2 --master_port=12341 \
    -m scripts.train \
    --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
    -- experiment=transfer2_auto_multiview_post_train_example \
    job.wandb_mode=disabled \
    job.name=rgbcloud_pointcloud2rgb \
    data_train.root={DATASET_ROOT} \
    trainer.max_iter=500 \
    trainer.validation_iter=100 \
    trainer.snapshot_save_iter=100
"""

    print("="*60)
    print("ZYC Post-Training Starting...")
    print("="*60)
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Train samples: {TRAIN_SAMPLES}")
    print(f"Val samples: {VAL_SAMPLES}")
    print("="*60)

    os.system(cmd)

if __name__ == "__main__":
    main()
