# 🚀 ZYC Cosmos Transfer2 快速开始

## 核心问题解答

### ❓ 推理是怎么推的？

```
你的数据 → [预训练 Checkpoint] → 生成的 RGB 视频
  ↓
depth.mp4 (控制结构)
color.mp4 (控制颜色/纹理)
prompt.txt (控制风格)
```

**关键点**：
- ✅ **不需要真实的驾驶场景 RGB 视频**
- ✅ Checkpoint 已经训练好，直接用你的 control 生成
- ✅ `video_path` 只是提供 FPS/分辨率，不参与生成

### ❓ 真正的 input 是什么？

**真正的 input 是**：
1. **Control 视频**（必需）：
   - `depth.mp4` - 你的深度视频
   - `color.mp4` - 你的彩色点云投影（可选）

2. **Text prompt**（必需）：
   - 描述想生成的场景风格

3. **Noise**（自动生成）：
   - 随机噪声，作为生成起点

**不是 input**：
- ❌ 真实的 RGB 驾驶视频（你没有也不需要）
- ❌ `video_path`（只是元信息来源）

---

## 🎯 一分钟快速测试

### Step 1: 生成配置文件

```bash
cd /home/user/cosmos_transfer2.5
python generate_zyc_config.py
```

输出：
```
✅ Generated: assets/zyc_test_single.jsonl       # 单视角测试
✅ Generated: assets/zyc_all_depth.jsonl         # 7视角，depth only
✅ Generated: assets/zyc_multicontrol.jsonl      # 7视角，depth+vis
✅ Generated: assets/zyc_high_control.jsonl      # 高控制强度
✅ Generated: assets/zyc_low_control.jsonl       # 低控制强度
```

### Step 2: 运行推理

**方法 A: 使用脚本（推荐）**
```bash
# 单视角测试
bash run_inference_zyc.sh

# 所有7个视角
bash run_inference_zyc.sh all
```

**方法 B: 手动命令**
```bash
# 单视角
torchrun --nproc_per_node=2 --master_port=12342 \
  -m examples.inference_zyc \
  -i assets/zyc_test_single.jsonl \
  -o outputs/zyc_test \
  --setup.model depth

# 多视角
torchrun --nproc_per_node=2 --master_port=12342 \
  -m examples.inference_zyc \
  -i assets/zyc_all_depth.jsonl \
  -o outputs/zyc_all \
  --setup.model depth
```

### Step 3: 查看结果

```bash
ls outputs/zyc_test/zyc_test_FW/

# 输出：
# zyc_test_FW.mp4              ← 生成的 RGB 视频
# zyc_test_FW_control_depth.mp4 ← 深度控制可视化
# zyc_test_FW.json             ← 推理参数
# zyc_test_FW.txt              ← 使用的 prompt
```

---

## 📊 配置文件说明

### 配置示例

```json
{
    "name": "zyc_FW",
    "prompt": "A realistic driving scene...",
    "video_path": "/mnt/.../color/FW_color.mp4",
    "guidance": 3,
    "num_conditional_frames": 0,
    "depth": {
        "control_path": "/mnt/.../depth/FW_depth.mp4",
        "control_weight": 1.0
    }
}
```

### 关键参数

| 参数 | 作用 | 推荐值 |
|------|------|-------|
| `num_conditional_frames` | 首帧条件数 | **0** (从噪声生成) |
| `depth.control_weight` | 深度控制强度 | 0.8-1.0 (密集点云)<br>0.5-0.7 (稀疏点云) |
| `vis.control_weight` | 颜色控制强度 | 0.3-0.5 |
| `guidance` | Prompt 引导强度 | 3-5 |
| `num_steps` | 推理步数 | 35 (质量)<br>25 (速度) |

---

## 🔧 常见调整

### 场景 1: 点云很稀疏，生成不稳定

**解决方案：降低 control_weight**
```json
"depth": {"control_weight": 0.5},  // 从 1.0 降到 0.5
"vis": {"control_weight": 0.3}     // 从 0.5 降到 0.3
```

使用配置：`assets/zyc_low_control.jsonl`

### 场景 2: 想要更真实的生成

**解决方案：提高 guidance 和 steps**
```json
"guidance": 5,      // 从 3 提高到 5
"num_steps": 50     // 从 35 提高到 50
```

手动修改配置文件，或调整 `generate_zyc_config.py` 中的 `DEFAULT_CONFIG`。

### 场景 3: 内存不够

**解决方案：添加优化参数**
```bash
torchrun --nproc_per_node=2 -m examples.inference_zyc \
  -i assets/zyc_test_single.jsonl \
  -o outputs/test \
  --setup.model depth \
  --setup.offload_guardrail_models True \
  --setup.compile_tokenizer none
```

或降低分辨率：
```json
"resolution": "480"  // 从 720 降到 480
```

### 场景 4: 加速推理

**方法 1: 增加 GPU**
```bash
# 从 2 卡提高到 4 卡
torchrun --nproc_per_node=4 ...
```

**方法 2: 减少步数**
```json
"num_steps": 25  // 从 35 降到 25
```

---

## 📁 文件结构

```
cosmos_transfer2.5/
├── assets/
│   ├── zyc_test_single.jsonl       ← 单视角测试配置
│   ├── zyc_all_depth.jsonl         ← 7视角 depth 配置
│   ├── zyc_multicontrol.jsonl      ← 7视角 depth+vis 配置
│   ├── zyc_high_control.jsonl      ← 高控制强度
│   └── zyc_low_control.jsonl       ← 低控制强度
├── examples/
│   └── inference_zyc.py            ← 推理脚本（ZYC版本）
├── outputs/                         ← 输出目录
│   └── zyc_test/
│       └── zyc_test_FW/
│           ├── zyc_test_FW.mp4     ← 生成的视频
│           └── ...
├── run_inference_zyc.sh            ← 一键运行脚本
├── generate_zyc_config.py          ← 配置生成工具
├── ZYC_QUICK_START.md              ← 本文档（快速开始）
└── ZYC_INFERENCE_GUIDE.md          ← 详细指南
```

---

## ✅ 检查清单

运行前确认：

- [ ] 你的数据路径正确：`/mnt/zihanw/车路协同投影工作/.../002/`
- [ ] 视频文件存在：`FL_color.mp4`, `FL_depth.mp4` 等
- [ ] 视频可以播放（检查编码格式）
- [ ] 有足够的 GPU 内存（推荐 2x A100 40GB 或以上）
- [ ] 已安装 Cosmos Transfer2 环境

---

## 🎓 核心理解

### Checkpoint 做什么？
```
Checkpoint = 预训练的生成模型
           ├─ DiT (生成网络): 学会了"如何生成真实视频"
           └─ ControlNet (控制编码器): 学会了"如何理解 depth/edge/color"
```

### 推理过程
```
1. 初始化噪声 [B, C, T, H, W]
   ↓
2. ControlNet 编码你的 depth/color 视频
   ↓
3. DiT 根据 control + prompt 逐步去噪
   ↓
4. VAE 解码成 RGB 视频
```

### 为什么不需要真实 RGB？
- Checkpoint 已经在**大规模真实视频**上训练过
- 学会了：给定 depth/prompt → 生成真实 RGB 的能力
- 你只需要提供新的 depth（你的点云投影）
- 模型会自动生成对应的 RGB

---

## 🐛 常见错误

### Error 1: `FileNotFoundError: /mnt/.../FW_color.mp4`
**原因**：路径不对或文件不存在
**解决**：
```bash
# 检查文件
ls /mnt/zihanw/车路协同投影工作/mutiCPU加速乌鸡变投影/Transfer_mp4视频生成/002/color/

# 修改 generate_zyc_config.py 中的 BASE_DIR
```

### Error 2: `CUDA out of memory`
**解决**：
```bash
# 方法 1: 降低分辨率
# 修改配置文件：resolution: "480"

# 方法 2: 减少视角数量
# 只推理 1-2 个视角

# 方法 3: 卸载 guardrail
--setup.offload_guardrail_models True
```

### Error 3: `RuntimeError: NCCL error`
**原因**：多 GPU 通信问题
**解决**：
```bash
# 检查 GPU
nvidia-smi

# 使用单卡测试
torchrun --nproc_per_node=1 ...
```

---

## 📞 下一步

1. ✅ **现在就试试**：
   ```bash
   python generate_zyc_config.py
   bash run_inference_zyc.sh
   ```

2. 📊 **查看结果质量**

3. 🔧 **根据质量调整参数**

4. 📖 **阅读详细指南**：`ZYC_INFERENCE_GUIDE.md`

5. 🎓 **如果效果不好，考虑微调**（联系我）

祝使用顺利！有问题随时问 🎉
