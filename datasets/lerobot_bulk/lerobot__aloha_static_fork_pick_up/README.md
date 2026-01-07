---
task_categories:
- robotics
tags:
- LeRobot
- aloha
---
This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

[meta/info.json](meta/info.json)
```json
{
    "codebase_version": "v2.0",
    "data_path": "data/train-{episode_index:05d}-of-{total_episodes:05d}.parquet",
    "robot_type": "aloha",
    "total_episodes": 100,
    "total_frames": 60000,
    "total_tasks": 1,
    "fps": 50,
    "splits": {
        "train": "0:100"
    },
    "keys": [
        "observation.state",
        "observation.effort",
        "action"
    ],
    "video_keys": [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_low",
        "observation.images.cam_right_wrist"
    ],
    "image_keys": [],
    "shapes": {
        "observation.state": 14,
        "observation.effort": 14,
        "action": 14,
        "observation.images.cam_high": {
            "width": 640,
            "height": 480,
            "channels": 3
        },
        "observation.images.cam_left_wrist": {
            "width": 640,
            "height": 480,
            "channels": 3
        },
        "observation.images.cam_low": {
            "width": 640,
            "height": 480,
            "channels": 3
        },
        "observation.images.cam_right_wrist": {
            "width": 640,
            "height": 480,
            "channels": 3
        }
    },
    "names": {
        "observation.state": [
            "left_waist",
            "left_shoulder",
            "left_elbow",
            "left_forearm_roll",
            "left_wrist_angle",
            "left_wrist_rotate",
            "left_gripper",
            "right_waist",
            "right_shoulder",
            "right_elbow",
            "right_forearm_roll",
            "right_wrist_angle",
            "right_wrist_rotate",
            "right_gripper"
        ],
        "action": [
            "left_waist",
            "left_shoulder",
            "left_elbow",
            "left_forearm_roll",
            "left_wrist_angle",
            "left_wrist_rotate",
            "left_gripper",
            "right_waist",
            "right_shoulder",
            "right_elbow",
            "right_forearm_roll",
            "right_wrist_angle",
            "right_wrist_rotate",
            "right_gripper"
        ],
        "observation.effort": [
            "left_waist",
            "left_shoulder",
            "left_elbow",
            "left_forearm_roll",
            "left_wrist_angle",
            "left_wrist_rotate",
            "left_gripper",
            "right_waist",
            "right_shoulder",
            "right_elbow",
            "right_forearm_roll",
            "right_wrist_angle",
            "right_wrist_rotate",
            "right_gripper"
        ]
    },
    "videos": {
        "videos_path": "videos/{video_key}_episode_{episode_index:06d}.mp4",
        "observation.images.cam_high": {
            "video.fps": 50.0,
            "video.width": 640,
            "video.height": 480,
            "video.channels": 3,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": false,
            "has_audio": false
        },
        "observation.images.cam_left_wrist": {
            "video.fps": 50.0,
            "video.width": 640,
            "video.height": 480,
            "video.channels": 3,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": false,
            "has_audio": false
        },
        "observation.images.cam_low": {
            "video.fps": 50.0,
            "video.width": 640,
            "video.height": 480,
            "video.channels": 3,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": false,
            "has_audio": false
        },
        "observation.images.cam_right_wrist": {
            "video.fps": 50.0,
            "video.width": 640,
            "video.height": 480,
            "video.channels": 3,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": false,
            "has_audio": false
        }
    }
}
```