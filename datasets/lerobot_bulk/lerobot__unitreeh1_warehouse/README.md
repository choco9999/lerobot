---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description



- **Homepage:** [More Information Needed]
- **Paper:** [More Information Needed]
- **License:** apache-2.0

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{
    "codebase_version": "v2.0",
    "robot_type": "unknown",
    "total_episodes": 24,
    "total_frames": 11275,
    "total_tasks": 1,
    "total_videos": 48,
    "total_chunks": 1,
    "chunks_size": 1000,
    "fps": 50,
    "splits": {
        "train": "0:24"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "observation.images.cam_left": {
            "dtype": "video",
            "shape": [
                360,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ],
            "video_info": {
                "video.fps": 50.0,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "observation.images.cam_right": {
            "dtype": "video",
            "shape": [
                360,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ],
            "video_info": {
                "video.fps": 50.0,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                19
            ],
            "names": {
                "motors": [
                    "motor_0",
                    "motor_1",
                    "motor_2",
                    "motor_3",
                    "motor_4",
                    "motor_5",
                    "motor_6",
                    "motor_7",
                    "motor_8",
                    "motor_9",
                    "motor_10",
                    "motor_11",
                    "motor_12",
                    "motor_13",
                    "motor_14",
                    "motor_15",
                    "motor_16",
                    "motor_17",
                    "motor_18"
                ]
            }
        },
        "action": {
            "dtype": "float32",
            "shape": [
                40
            ],
            "names": {
                "motors": [
                    "motor_0",
                    "motor_1",
                    "motor_2",
                    "motor_3",
                    "motor_4",
                    "motor_5",
                    "motor_6",
                    "motor_7",
                    "motor_8",
                    "motor_9",
                    "motor_10",
                    "motor_11",
                    "motor_12",
                    "motor_13",
                    "motor_14",
                    "motor_15",
                    "motor_16",
                    "motor_17",
                    "motor_18",
                    "motor_19",
                    "motor_20",
                    "motor_21",
                    "motor_22",
                    "motor_23",
                    "motor_24",
                    "motor_25",
                    "motor_26",
                    "motor_27",
                    "motor_28",
                    "motor_29",
                    "motor_30",
                    "motor_31",
                    "motor_32",
                    "motor_33",
                    "motor_34",
                    "motor_35",
                    "motor_36",
                    "motor_37",
                    "motor_38",
                    "motor_39"
                ]
            }
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "next.done": {
            "dtype": "bool",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        }
    }
}
```


## Citation

**BibTeX:**

```bibtex
[More Information Needed]
```