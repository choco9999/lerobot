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
    "total_episodes": 240,
    "total_frames": 32708,
    "total_tasks": 3,
    "total_videos": 240,
    "total_chunks": 1,
    "chunks_size": 1000,
    "fps": 5,
    "splits": {
        "train": "0:240"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "observation.images.image": {
            "dtype": "video",
            "shape": [
                128,
                128,
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
                7
            ],
            "names": {
                "motors": [
                    "motor_0",
                    "motor_1",
                    "motor_2",
                    "motor_3",
                    "motor_4",
                    "motor_5",
                    "motor_6"
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
@misc{oh2023pr2utokyodatasets,
    author={Jihoon Oh and Naoaki Kanazawa and Kento Kawaharazuka},
    title={X-Embodiment U-Tokyo PR2 Datasets},
    year={2023},
    url={https://github.com/ojh6404/rlds_dataset_builder},
}
```