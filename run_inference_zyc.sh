#!/bin/bash
# ZYC 推理脚本
# 使用方法：
#   bash run_inference_zyc.sh          # 单视角测试（只推理 FW）
#   bash run_inference_zyc.sh all      # 所有7个视角

set -e

# 配置
NUM_GPUS=2
MASTER_PORT=12342
OUTPUT_DIR="outputs/zyc_inference"

# 根据参数选择配置文件
if [ "$1" = "all" ]; then
    echo "🚀 Running inference for all 7 views..."
    INPUT_FILE="assets/zyc_multicontrol.jsonl"
else
    echo "🚀 Running test inference for single view (FW)..."
    INPUT_FILE="assets/zyc_test_single.jsonl"
fi

# 检查文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file $INPUT_FILE not found!"
    exit 1
fi

# 打印配置
echo "="
echo "Configuration:"
echo "  Input file: $INPUT_FILE"
echo "  Output dir: $OUTPUT_DIR"
echo "  Num GPUs: $NUM_GPUS"
echo "  Master port: $MASTER_PORT"
echo "="
echo ""

# 运行推理
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m examples.inference_zyc \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_DIR"

echo ""
echo "✅ Inference completed! Check results in: $OUTPUT_DIR"
