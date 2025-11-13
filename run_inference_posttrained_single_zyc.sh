#!/bin/bash
# ZYC Single-View Post-Trained Model Inference Script
# 使用微调后的单视角模型进行推理（测试样本 009）

set -e

# 配置
OUTPUT_ROOT="/mnt/zihanw/cosmos-transfer2.5/outputs"
CHECKPOINT_DIR="$OUTPUT_ROOT/cosmos_transfer_v2p5/zyc_posttrain_single/zyc_pointcloud2rgb_single/checkpoints"
DATASET_ROOT="/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud"
NUM_GPUS=2
MASTER_PORT=12342

echo "======================================"
echo "ZYC Single-View Post-Trained Inference"
echo "======================================"

# Step 1: 获取最新的 checkpoint
if [ ! -f "$CHECKPOINT_DIR/latest_checkpoint.txt" ]; then
    echo "❌ Error: No checkpoint found at $CHECKPOINT_DIR"
    echo "Please run post-training first: bash run_posttrain_single_zyc.sh"
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

# Step 3: 生成推理配置（sample 009 - 只有前置摄像头）
INFERENCE_CONFIG="/tmp/zyc_posttrain_single_inference_009.jsonl"

# JSONL 格式：每行一个紧凑的 JSON 对象
cat > $INFERENCE_CONFIG <<'EOFCONFIG'
{"name": "009", "prompt": "A realistic driving scene at an urban intersection with multiple lanes, traffic lights, road markings, and surrounding buildings. The scene captures a typical city road environment with clear visibility and detailed urban infrastructure.", "guidance": 3, "num_conditional_frames": 0, "num_steps": 35, "resolution": "720", "seed": 2025, "depth": {"control_path": "/mnt/zihanw/cosmos-transfer2.5/data_prepa/ftheta_camera_front_wide_120fov/depth/009_90frames_1280x720.mp4", "control_weight": 0.8}, "vis": {"control_path": "/mnt/zihanw/cosmos-transfer2.5/data_prepa/ftheta_camera_front_wide_120fov/color/009_90frames_1280x720.mp4", "control_weight": 0.2}}
EOFCONFIG

echo "Generated inference config: $INFERENCE_CONFIG"
echo ""

# Step 4: 运行推理
echo "Running inference with post-trained model on sample 009..."
echo "Checkpoint: $CHECKPOINT_PATH/model_ema_bf16.pt"
echo "Output directory: /mnt/zihanw/cosmos-transfer2.5/outputs/posttrained_single_zyc_009"
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m examples.inference_zyc \
    -i $INFERENCE_CONFIG \
    -o /mnt/zihanw/cosmos-transfer2.5/outputs/posttrained_single_zyc_009 \
    --checkpoint-path $CHECKPOINT_PATH/model_ema_bf16.pt

echo ""
echo "======================================"
echo "✅ Inference completed!"
echo "Results saved to:"
echo "/mnt/zihanw/cosmos-transfer2.5/outputs/posttrained_single_zyc_009/"
echo ""
echo "Check generated video:"
echo "ls -lh /mnt/zihanw/cosmos-transfer2.5/outputs/posttrained_single_zyc_009/"
echo "======================================"
