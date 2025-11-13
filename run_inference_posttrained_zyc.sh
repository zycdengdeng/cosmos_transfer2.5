#!/bin/bash
# ZYC Post-Trained Model Inference Script
# 使用微调后的模型进行推理

set -e

# 配置
OUTPUT_ROOT="/mnt/zihanw/cosmos-transfer2.5/outputs"
CHECKPOINT_DIR="$OUTPUT_ROOT/cosmos_transfer_v2p5/zyc_posttrain/zyc_pointcloud2rgb/checkpoints"
NUM_GPUS=2
MASTER_PORT=12342

echo "======================================"
echo "ZYC Post-Trained Model Inference"
echo "======================================"

# Step 1: 获取最新的 checkpoint
if [ ! -f "$CHECKPOINT_DIR/latest_checkpoint.txt" ]; then
    echo "❌ Error: No checkpoint found at $CHECKPOINT_DIR"
    echo "Please run post-training first: bash run_posttrain_zyc.sh"
    exit 1
fi

CHECKPOINT_ITER=$(cat $CHECKPOINT_DIR/latest_checkpoint.txt)
CHECKPOINT_PATH="$CHECKPOINT_DIR/$CHECKPOINT_ITER"

echo "Latest checkpoint: $CHECKPOINT_ITER"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo ""

# Step 2: 转换 checkpoint（如果还没转换）
if [ ! -f "$CHECKPOINT_PATH/model_ema_bf16.pt" ]; then
    echo "Converting DCP checkpoint to PyTorch format..."
    python scripts/convert_distcp_to_pt.py \
        $CHECKPOINT_PATH/model \
        $CHECKPOINT_PATH
    echo "✅ Conversion completed"
    echo ""
fi

# Step 3: 生成推理配置（sample 009）
cat > /tmp/zyc_posttrain_inference.json <<EOF
{
    "name": "zyc_009_posttrained",
    "prompt_path": "/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud/captions/ftheta_camera_front_wide_120fov/009.json",
    "front_wide": {
        "control_path": "/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud/control_input_hdmap_bbox/ftheta_camera_front_wide_120fov/009.mp4"
    },
    "cross_left": {
        "control_path": "/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud/control_input_hdmap_bbox/ftheta_camera_cross_left_120fov/009.mp4"
    },
    "cross_right": {
        "control_path": "/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud/control_input_hdmap_bbox/ftheta_camera_cross_right_120fov/009.mp4"
    },
    "front_tele": {
        "control_path": "/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud/control_input_hdmap_bbox/ftheta_camera_front_tele_30fov/009.mp4"
    },
    "rear_left": {
        "control_path": "/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud/control_input_hdmap_bbox/ftheta_camera_rear_left_70fov/009.mp4"
    },
    "rear_right": {
        "control_path": "/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud/control_input_hdmap_bbox/ftheta_camera_rear_right_70fov/009.mp4"
    },
    "rear": {
        "control_path": "/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud/control_input_hdmap_bbox/ftheta_camera_rear_tele_30fov/009.mp4"
    }
}
EOF

echo "Generated inference config: /tmp/zyc_posttrain_inference.json"
echo ""

# Step 4: 运行推理
echo "Running inference with post-trained model..."
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m examples.multiview \
    -i /tmp/zyc_posttrain_inference.json \
    -o outputs/posttrained_zyc_009 \
    --checkpoint_path $CHECKPOINT_PATH/model_ema_bf16.pt \
    --experiment transfer2_auto_multiview_post_train_example

echo ""
echo "======================================"
echo "✅ Inference completed!"
echo "Results saved to: outputs/posttrained_zyc_009/"
echo "======================================"
