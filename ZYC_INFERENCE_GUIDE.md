# ZYC Cosmos Transfer2 推理指南

## 📋 推理流程详解

### 核心问题解答

**Q: 推理是怎么推的？真正的 input 是什么？**

**A: 推理流程如下**：

```
1. 加载预训练 checkpoint (depth/edge/seg/vis 模型)
   ↓
2. 读取你提供的 control 视频（depth.mp4, color.mp4）
   ↓
3. 读取 text prompt（描述你想生成的场景）
   ↓
4. 从纯噪声（或 video_path）开始，根据 control 信号生成 RGB 视频
   ↓
5. 输出生成的真实 RGB 视频
```

**关键点**：
- ✅ **不需要真实的驾驶场景 RGB 视频作为 input**（如果 `num_conditional_frames=0`）
- ✅ `video_path` 主要用于提取 FPS、分辨率等元信息
- ✅ 真正的控制信号来自 `control_path`（你的 depth 和 color 视频）
- ✅ checkpoint 已经包含了预训练的生成能力，不需要从 input 提取特征

---

## 🎯 你的数据如何使用

### 数据结构
```
/mnt/zihanw/车路协同投影工作/mutiCPU加速乌鸡变投影/Transfer_mp4视频生成/002/
├── color/          # 点云彩色投影 → 用作 vis (blur) control
│   ├── FL_color.mp4
│   ├── FN_color.mp4
│   ├── FR_color.mp4
│   ├── FW_color.mp4
│   ├── RL_color.mp4
│   ├── RN_color.mp4
│   └── RR_color.mp4
└── depth/          # 深度信息 → 用作 depth control
    ├── FL_depth.mp4
    ├── FN_depth.mp4
    ├── FR_depth.mp4
    ├── FW_depth.mp4
    ├── RL_depth.mp4
    ├── RN_depth.mp4
    └── RR_depth.mp4
```

### 配置文件解释

**示例配置** (`assets/zyc_test_single.jsonl`):

```json
{
    "name": "zyc_test_FW",
    "prompt": "A realistic driving scene on a city street with clear weather and good visibility",
    "video_path": "/mnt/.../color/FW_color.mp4",
    "guidance": 3,
    "num_conditional_frames": 0,
    "depth": {
        "control_path": "/mnt/.../depth/FW_depth.mp4",
        "control_weight": 1.0
    }
}
```

**参数说明**：

| 参数 | 作用 | 你的设置 |
|------|------|---------|
| `name` | 输出文件名 | `zyc_test_FW` |
| `prompt` | 文本描述，指导生成风格 | 真实街景 |
| `video_path` | 元信息来源（FPS、分辨率） | 你的 color 视频 |
| `guidance` | 控制强度（1-7） | 3（推荐） |
| `num_conditional_frames` | 首帧条件数量 | **0（从噪声生成）** |
| `depth.control_path` | 深度控制视频 | 你的 depth 视频 |
| `depth.control_weight` | 深度控制权重（0-1） | 1.0（强控制） |

**关键参数解释**：

1. **`num_conditional_frames = 0`**：
   - 从**纯噪声**开始生成
   - 不需要真实的 RGB input
   - 完全根据 control 信号生成新内容

2. **`video_path` 的作用**：
   - 提取 FPS（保证输出视频帧率一致）
   - 提取分辨率信息
   - 如果某个 control 没提供 path，会从这里自动计算
   - **但不参与生成过程**（因为 `num_conditional_frames=0`）

3. **为什么可以用 `color` 作为 `video_path`**：
   - 你的 color 和 depth 视频同步，FPS、分辨率相同
   - 用 color 视频提取这些元信息
   - 同时 color 视频也作为 vis 的 control（如果需要）

---

## 🚀 使用方法

### 方法 1：使用提供的脚本（推荐）

#### Step 1: 单视角测试
```bash
# 先测试单个视角（FW）
bash run_inference_zyc.sh

# 输出位置：outputs/zyc_inference/zyc_test_FW/zyc_test_FW.mp4
```

#### Step 2: 所有视角推理
```bash
# 推理所有7个视角
bash run_inference_zyc.sh all

# 输出位置：outputs/zyc_inference/{zyc_FL, zyc_FN, ...}/
```

---

### 方法 2：手动运行

#### 单视角 + 单控制（depth only）
```bash
torchrun --nproc_per_node=2 --master_port=12342 \
  -m examples.inference_zyc \
  -i assets/zyc_test_single.jsonl \
  -o outputs/zyc_depth_only \
  --setup.model depth
```

#### 单视角 + 多控制（depth + vis）
修改 `assets/zyc_test_single.jsonl`，添加 vis 控制：
```json
{
    "name": "zyc_test_FW_multicontrol",
    "prompt": "A realistic driving scene...",
    "video_path": "/mnt/.../color/FW_color.mp4",
    "guidance": 3,
    "num_conditional_frames": 0,
    "depth": {
        "control_path": "/mnt/.../depth/FW_depth.mp4",
        "control_weight": 0.8
    },
    "vis": {
        "control_path": "/mnt/.../color/FW_color.mp4",
        "control_weight": 0.5
    }
}
```

然后运行：
```bash
torchrun --nproc_per_node=2 --master_port=12342 \
  -m examples.inference_zyc \
  -i assets/zyc_test_single.jsonl \
  -o outputs/zyc_multicontrol \
  --setup.model depth  # 会自动加载多控制模型
```

---

## 🔧 高级配置

### 调整控制强度

```json
"depth": {
    "control_path": "...",
    "control_weight": 0.8  // 0.0-1.0，越大控制越强
}
```

- `0.0`: 完全忽略该控制
- `0.5`: 中等强度（生成更自由）
- `1.0`: 强控制（严格遵循控制信号）

### 调整生成质量

```json
{
    "guidance": 3,           // 1-7，越大越符合 prompt
    "num_steps": 35,         // 推理步数，越多质量越高（但更慢）
    "resolution": "720",     // 分辨率：480/720/1080
    "seed": 2025            // 随机种子，固定可复现
}
```

### 使用 spatiotemporal mask

如果你的点云投影是稀疏的（有空白区域），可以添加 mask：

```json
"depth": {
    "control_path": "/mnt/.../depth/FW_depth.mp4",
    "mask_path": "/mnt/.../masks/FW_mask.mp4",  // 二值mask，白色=控制，黑色=自由生成
    "control_weight": 1.0
}
```

---

## 📊 预期输出

### 输出结构
```
outputs/zyc_inference/
├── zyc_test_FW/
│   ├── zyc_test_FW.mp4         # 生成的 RGB 视频
│   ├── zyc_test_FW_control_depth.mp4  # 深度控制可视化
│   ├── zyc_test_FW.json        # 推理参数
│   └── zyc_test_FW.txt         # 使用的 prompt
├── zyc_FL/
│   └── ...
└── config.yaml                  # 模型配置
```

### 生成质量因素

1. **Control 质量**：
   - 你的 depth/color 视频越准确，生成越好
   - 稀疏点云可能导致生成不稳定

2. **Prompt 质量**：
   - 详细描述场景（天气、光照、物体）
   - 示例：`"A realistic urban driving scene during daytime with clear sky, showing cars, pedestrians, and buildings with sharp details"`

3. **Control weight**：
   - 点云稀疏 → 降低 weight（0.5-0.7）
   - 点云密集 → 提高 weight（0.8-1.0）

---

## ❓ 常见问题

### Q1: 我没有真实的 RGB 视频，能推理吗？
**A**: 可以！设置 `num_conditional_frames=0`，从纯噪声生成。

### Q2: `video_path` 必须是真实 RGB 吗？
**A**: 不需要。可以是：
- 你的 color 视频（点云投影）
- 你的 depth 视频
- 任何同步的视频（只要 FPS/分辨率对得上）

### Q3: 如何提高生成质量？
**A**:
1. 调整 `control_weight`（0.5-1.0）
2. 增加 `num_steps`（35 → 50）
3. 提高 `guidance`（3 → 5）
4. 优化 prompt 描述

### Q4: 如何加速推理？
**A**:
1. 增加 GPU 数量（`--nproc_per_node=4`）
2. 减少步数（`num_steps=25`）
3. 降低分辨率（`resolution="480"`）

### Q5: 内存不够怎么办？
**A**:
```bash
# 添加这些参数
--setup.offload_guardrail_models True  # 卸载 guardrail
--setup.compile_tokenizer none         # 禁用编译
```

---

## 🎓 核心概念总结

### Checkpoint 的作用
- 包含预训练的**生成网络**（DiT）
- 包含预训练的**控制编码器**（ControlNet）
- **不需要从 input 提取特征**，直接根据 control 生成

### 推理过程（简化）
```python
# 伪代码
noise = torch.randn(shape)              # 1. 初始化噪声
control_features = encode(depth_video)   # 2. 编码控制信号
text_features = encode(prompt)           # 3. 编码文本

for step in range(num_steps):
    noise = denoise(
        noise,
        control_features,                # 根据 depth 指导
        text_features                    # 根据 prompt 指导
    )

rgb_video = vae.decode(noise)            # 4. 解码为 RGB
```

### Control vs Input
- **Control**（control_path）：指导生成的**结构/布局**
- **Input**（video_path）：提供**首帧内容**（如果 num_conditional_frames > 0）
- 你的任务：只需要 control，不需要 input

---

## 📞 下一步

1. **测试单个视角**：
   ```bash
   bash run_inference_zyc.sh
   ```

2. **检查输出质量**

3. **调整参数**（control_weight, guidance）

4. **批量处理所有视角**：
   ```bash
   bash run_inference_zyc.sh all
   ```

5. **如果质量不好**：
   - 考虑微调模型（使用你的数据）
   - 调整 prompt
   - 尝试不同的 control 组合

祝推理顺利！🎉
