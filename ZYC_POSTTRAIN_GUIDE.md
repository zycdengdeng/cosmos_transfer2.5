# ZYC Post-Training 完整指南

## 数据准备

### ✅ 你的数据结构（已完成）
```
/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud/
├── captions/                          # 文本描述
│   └── ftheta_camera_front_wide_120fov/
│       ├── 002.json
│       ├── 004.json
│       ├── 006.json
│       ├── 008.json
│       └── 009.json
├── control_input_hdmap_bbox/          # 点云投影（控制信号）
│   ├── ftheta_camera_front_wide_120fov/
│   │   ├── 002.mp4
│   │   └── ...
│   └── ...（7个视角）
└── videos/                            # 真实 RGB（ground truth）
    ├── ftheta_camera_front_wide_120fov/
    │   ├── 002.mp4
    │   └── ...
    └── ...（7个视角）
```

### 样本划分
- **训练集**: 002, 004, 006, 008 (4 个样本)
- **测试集**: 009 (1 个样本)

---

## 完整流程

### **Step 1: 验证数据准备**

确认所有 caption 文件已准备好：
```bash
ls /mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud/captions/ftheta_camera_front_wide_120fov/
# 应该看到: 002.json, 004.json, 006.json, 008.json, 009.json
```

**Caption 格式示例**（已准备好 ✅）：
```json
{
    "caption": "A driving scene captured from a vehicle's front wide camera, The car is driving through the intersection.",
    "sequence_id": "004",
    "camera": "ftheta_camera_front_wide_120fov"
}
```

---

### **Step 2: 运行 Post-Training**

```bash
# 给脚本执行权限
chmod +x run_posttrain_zyc.sh

# 运行训练（需要 2 个 GPU）
bash run_posttrain_zyc.sh
```

**训练参数**：
- GPU 数量: 2
- 训练步数: 500
- 学习率: 1e-5（小学习率用于微调）
- 验证间隔: 每 100 步
- 保存间隔: 每 100 步

**训练时间**：
- 约 1-2 小时（取决于 GPU）

**输出位置**：
```
/mnt/zihanw/cosmos-transfer2.5/outputs/
└── cosmos_transfer_v2p5/
    └── zyc_posttrain/
        └── zyc_pointcloud2rgb/
            ├── checkpoints/
            │   ├── latest_checkpoint.txt
            │   ├── iter_00000100/
            │   ├── iter_00000200/
            │   └── ...
            └── logs/
```

---

### **Step 3: 转换 Checkpoint（自动完成）**

推理脚本会自动转换，也可以手动转换：

```bash
# 获取最新 checkpoint 路径
CHECKPOINT_DIR="/mnt/zihanw/cosmos-transfer2.5/outputs/cosmos_transfer_v2p5/zyc_posttrain/zyc_pointcloud2rgb/checkpoints"
CHECKPOINT_ITER=$(cat $CHECKPOINT_DIR/latest_checkpoint.txt)

# 转换
python scripts/convert_distcp_to_pt.py \
    $CHECKPOINT_DIR/$CHECKPOINT_ITER/model \
    $CHECKPOINT_DIR/$CHECKPOINT_ITER
```

**生成文件**：
- `model_ema_bf16.pt` ← 推理使用

---

### **Step 4: 用微调模型推理（测试集 009）**

```bash
# 给脚本执行权限
chmod +x run_inference_posttrained_zyc.sh

# 运行推理
bash run_inference_posttrained_zyc.sh
```

**输出**：
```
outputs/posttrained_zyc_009/
└── zyc_009_posttrained/
    ├── zyc_009_posttrained_front_wide.mp4
    ├── zyc_009_posttrained_cross_left.mp4
    ├── zyc_009_posttrained_cross_right.mp4
    ├── zyc_009_posttrained_front_tele.mp4
    ├── zyc_009_posttrained_rear_left.mp4
    ├── zyc_009_posttrained_rear_right.mp4
    └── zyc_009_posttrained_rear.mp4
```

---

## 对比实验

### **对比 1: Pre-trained vs Post-trained**

**使用预训练模型推理 009**：
```bash
# 先创建 009 的单视角配置
cat > assets/zyc_009_pretrained.json <<EOF
{
    "name": "zyc_009_pretrained",
    "prompt": "A realistic urban driving scene at a busy intersection with clear sunny weather...",
    "video_path": "/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud/control_input_hdmap_bbox/ftheta_camera_front_wide_120fov/009.mp4",
    "guidance": 3,
    "num_conditional_frames": 0,
    "depth": {
        "control_path": "/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud/control_input_hdmap_bbox/ftheta_camera_front_wide_120fov/009.mp4",
        "control_weight": 1.0
    }
}
EOF

# 用预训练模型推理
torchrun --nproc_per_node=2 -m examples.inference_zyc \
    -i assets/zyc_009_pretrained.json \
    -o outputs/pretrained_zyc_009
```

**对比**：
- Pre-trained: 通用模型，可能对你的点云格式不够适配
- Post-trained: 专门学习了你的点云→RGB 映射，应该更准确

---

## 监控训练

### **查看日志**
```bash
tail -f /mnt/zihanw/cosmos-transfer2.5/outputs/cosmos_transfer_v2p5/zyc_posttrain/zyc_pointcloud2rgb/logs/train.log
```

### **关键指标**
- `loss`: 越低越好（通常从 0.1 降到 0.01 左右）
- `val_loss`: 验证集损失（应该逐渐下降）
- 如果 `val_loss` 不降反升 → 过拟合（减少训练步数）

---

## 调整训练参数

### **如果训练太慢**：
```bash
# 修改 run_posttrain_zyc.sh
trainer.max_iter=300  # 从 500 降到 300
```

### **如果数据更多**：
```bash
trainer.max_iter=1000  # 增加到 1000
trainer.snapshot_save_iter=200  # 每 200 步保存
```

### **如果过拟合**：
```bash
gen_opt.lr=5e-6  # 降低学习率
trainer.max_iter=300  # 减少步数
```

---

## 故障排除

### **Error: No checkpoint found**
- 原因: 训练还没完成或失败
- 解决: 检查训练日志，重新运行训练

### **CUDA out of memory**
- 解决: 减少 batch size 或用更少的视角训练
```bash
# 修改配置只训练部分视角
data_train.n_views=3  # 从 7 降到 3
```

### **训练损失不降**
- 原因: 学习率太小或数据有问题
- 解决: 增加学习率或检查数据质量
```bash
gen_opt.lr=2e-5  # 从 1e-5 提高到 2e-5
```

---

## 最佳实践

1. **先小规模测试**：
   - 用 2 个样本快速训练 100 步
   - 验证流程能跑通

2. **对比质量**：
   - 保存预训练模型的结果
   - 保存微调模型的结果
   - 对比 009 的生成质量

3. **增量训练**：
   - 如果效果不够好，可以继续训练
   - 从最新 checkpoint 恢复，再训练 500 步

4. **多次实验**：
   - 尝试不同的学习率（1e-5, 5e-6, 2e-5）
   - 尝试不同的训练步数（300, 500, 1000）

---

## 预期结果

**Post-Training 后应该看到**：
- ✅ 生成的视频更符合你的点云投影风格
- ✅ 细节更准确（车道线、红绿灯、标志）
- ✅ 颜色更接近真实场景
- ✅ 时序更连贯

**如果效果不明显**：
- 可能需要更多训练数据（>10 个样本）
- 可能需要更长训练时间（1000+ 步）
- 可能需要调整学习率

---

## 快速命令总结

```bash
# 1. 训练（captions 已准备好 ✅）
bash run_posttrain_zyc.sh

# 2. 推理（微调模型）
bash run_inference_posttrained_zyc.sh

# 3. 对比（预训练模型）
torchrun --nproc_per_node=2 -m examples.inference_zyc \
    -i assets/zyc_009_pretrained.json \
    -o outputs/pretrained_zyc_009
```

---

祝训练成功！🚀
