#!/usr/bin/env python3
"""
生成 ZYC 推理配置文件
用于批量创建多视角的推理配置
"""

import json
from pathlib import Path

# 配置
BASE_DIR = Path("/mnt/zihanw/cosmos-transfer2.5/data_prepa")
COLOR_DIR = BASE_DIR / "color"
DEPTH_DIR = BASE_DIR / "depth"

# 文件名模板
FILE_SUFFIX = "_90frames_1280x720.mp4"

# 视角映射（你的命名 → Cosmos 标准命名）
VIEW_MAPPING = {
    "FW": "front_wide",      # Front Wide
    "FL": "front_left",      # Front Left (cross_left in Cosmos)
    "FR": "front_right",     # Front Right (cross_right in Cosmos)
    "FN": "front_narrow",    # Front Narrow (front_tele in Cosmos)
    "RL": "rear_left",       # Rear Left
    "RR": "rear_right",      # Rear Right
    "RN": "rear_narrow",     # Rear Narrow (rear in Cosmos)
}

# 默认配置
DEFAULT_CONFIG = {
    "prompt": "A realistic driving scene on a city street with clear weather, showing cars, buildings, and pedestrians with sharp details and good visibility",
    "guidance": 3,
    "num_conditional_frames": 0,  # 从纯噪声生成
    "num_steps": 35,
    "resolution": "720",
    "seed": 2025,
}

def generate_single_view_config(view_name: str, depth_weight: float = 1.0, vis_weight: float = 0.0):
    """生成单个视角的配置"""
    config = {
        "name": f"zyc_{view_name}",
        **DEFAULT_CONFIG,
        "video_path": str(COLOR_DIR / f"{view_name}_color{FILE_SUFFIX}"),
        "depth": {
            "control_path": str(DEPTH_DIR / f"{view_name}_depth{FILE_SUFFIX}"),
            "control_weight": depth_weight,
        }
    }

    # 如果使用 vis control
    if vis_weight > 0:
        config["vis"] = {
            "control_path": str(COLOR_DIR / f"{view_name}_color{FILE_SUFFIX}"),
            "control_weight": vis_weight,
        }

    return config

def generate_jsonl(output_file: str, views: list[str], depth_weight: float = 1.0, vis_weight: float = 0.0):
    """生成 JSONL 配置文件"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for view in views:
            config = generate_single_view_config(view, depth_weight, vis_weight)
            f.write(json.dumps(config, ensure_ascii=False) + '\n')

    print(f"✅ Generated: {output_path}")
    print(f"   Views: {views}")
    print(f"   Depth weight: {depth_weight}, Vis weight: {vis_weight}")

def main():
    """生成各种配置文件"""

    # 1. 单视角测试（depth only）
    generate_jsonl(
        "assets/zyc_test_single.jsonl",
        views=["FW"],
        depth_weight=1.0,
        vis_weight=0.0
    )

    # 2. 所有视角（depth only）
    generate_jsonl(
        "assets/zyc_all_depth.jsonl",
        views=list(VIEW_MAPPING.keys()),
        depth_weight=1.0,
        vis_weight=0.0
    )

    # 3. 所有视角（depth + vis）
    generate_jsonl(
        "assets/zyc_multicontrol.jsonl",
        views=list(VIEW_MAPPING.keys()),
        depth_weight=0.8,
        vis_weight=0.5
    )

    # 4. 高控制强度（适合密集点云）
    generate_jsonl(
        "assets/zyc_high_control.jsonl",
        views=list(VIEW_MAPPING.keys()),
        depth_weight=1.0,
        vis_weight=0.8
    )

    # 5. 低控制强度（适合稀疏点云）
    generate_jsonl(
        "assets/zyc_low_control.jsonl",
        views=list(VIEW_MAPPING.keys()),
        depth_weight=0.5,
        vis_weight=0.3
    )

    print("\n" + "="*50)
    print("📝 配置文件生成完成！")
    print("="*50)
    print("\n使用方法：")
    print("  1. 单视角测试：")
    print("     torchrun --nproc_per_node=2 -m examples.inference_zyc \\")
    print("       -i assets/zyc_test_single.jsonl -o outputs/test")
    print("")
    print("  2. 所有视角（depth only）：")
    print("     torchrun --nproc_per_node=2 -m examples.inference_zyc \\")
    print("       -i assets/zyc_all_depth.jsonl -o outputs/all_depth")
    print("")
    print("  3. 多控制（depth + vis）：")
    print("     torchrun --nproc_per_node=2 -m examples.inference_zyc \\")
    print("       -i assets/zyc_multicontrol.jsonl -o outputs/multicontrol")

if __name__ == "__main__":
    main()
