---
license: mit
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
- **License:** mit

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{
    "codebase_version": "v2.0",
    "robot_type": "unknown",
    "total_episodes": 150,
    "total_frames": 3970,
    "total_tasks": 8,
    "total_videos": 150,
    "total_chunks": 1,
    "chunks_size": 1000,
    "fps": 5,
    "splits": {
        "train": "0:150"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "observation.images.image": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ],
            "video_info": {
                "video.fps": 5.0,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "language_instruction": {
            "dtype": "string",
            "shape": [
                1
            ],
            "names": null
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                21
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
                    "motor_20"
                ]
            }
        },
        "action": {
            "dtype": "float32",
            "shape": [
                8
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
                    "motor_7"
                ]
            }
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
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
        "next.reward": {
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
@ARTICLE{ucsd_kitchens,
    author = {Ge Yan, Kris Wu, and Xiaolong Wang},
    title = {{ucsd kitchens Dataset}},
    year = {2023},
    month = {August}
}
```