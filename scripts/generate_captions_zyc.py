#!/usr/bin/env python3
"""
为所有视角生成 caption JSON 文件
"""

import json
from pathlib import Path

# 配置
CAPTION_DIR = Path("/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud/captions")
SAMPLE_IDS = ["002", "004", "006", "008", "009"]

# 所有视角
CAMERAS = [
    "ftheta_camera_front_wide_120fov",
    "ftheta_camera_cross_left_120fov",
    "ftheta_camera_cross_right_120fov",
    "ftheta_camera_front_tele_30fov",
    "ftheta_camera_rear_left_70fov",
    "ftheta_camera_rear_right_70fov",
    "ftheta_camera_rear_tele_30fov",
]

# 每个视角的 caption 模板
CAPTION_TEMPLATES = {
    "ftheta_camera_front_wide_120fov": "A realistic urban driving scene at a busy intersection captured from the front wide camera with clear sunny weather, showing well-defined white lane markings, functioning traffic lights, clear road signs, multiple cars, pedestrians, and surrounding buildings with sharp details and excellent visibility",
    "ftheta_camera_cross_left_120fov": "A realistic urban driving scene at an intersection captured from the left side camera, showing cross traffic, lane markings, traffic signals, vehicles, and pedestrians with clear visibility",
    "ftheta_camera_cross_right_120fov": "A realistic urban driving scene at an intersection captured from the right side camera, showing cross traffic, lane markings, traffic signals, vehicles, and pedestrians with clear visibility",
    "ftheta_camera_front_tele_30fov": "A realistic urban driving scene captured from the front telephoto camera, showing distant road details, traffic lights, road signs, and vehicles ahead with sharp focus",
    "ftheta_camera_rear_left_70fov": "A realistic urban driving scene captured from the rear left camera, showing traffic behind and to the left, lane markings, and surrounding environment",
    "ftheta_camera_rear_right_70fov": "A realistic urban driving scene captured from the rear right camera, showing traffic behind and to the right, lane markings, and surrounding environment",
    "ftheta_camera_rear_tele_30fov": "A realistic urban driving scene captured from the rear telephoto camera, showing distant vehicles and road conditions behind with clear details",
}

def main():
    for camera in CAMERAS:
        camera_dir = CAPTION_DIR / camera
        camera_dir.mkdir(parents=True, exist_ok=True)

        for sample_id in SAMPLE_IDS:
            caption_file = camera_dir / f"{sample_id}.json"

            caption_data = {
                "caption": CAPTION_TEMPLATES[camera],
                "sequence_id": sample_id,
                "camera": camera
            }

            with open(caption_file, 'w') as f:
                json.dump(caption_data, f, indent=4)

            print(f"✅ Created: {caption_file}")

    print("\n" + "="*50)
    print("All caption files generated!")
    print("="*50)

if __name__ == "__main__":
    main()
