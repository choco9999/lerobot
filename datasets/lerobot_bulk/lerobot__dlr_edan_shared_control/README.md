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
- **Paper:** https://ieeexplore.ieee.org/document/9341156
- **License:** mit

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{
    "codebase_version": "v2.0",
    "robot_type": "unknown",
    "total_episodes": 104,
    "total_frames": 8928,
    "total_tasks": 14,
    "total_videos": 104,
    "total_chunks": 1,
    "chunks_size": 1000,
    "fps": 5,
    "splits": {
        "train": "0:104"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "observation.images.image": {
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
@inproceedings{vogel_edan_2020,
    title = {EDAN - an EMG-Controlled Daily Assistant to Help People with Physical Disabilities},
    language = {en},
    booktitle = {2020 {IEEE}/{RSJ} {International} {Conference} on {Intelligent} {Robots} and {Systems} ({IROS})},
    author = {Vogel, Jörn and Hagengruber, Annette and Iskandar, Maged and Quere, Gabriel and Leipscher, Ulrike and Bustamante, Samuel and Dietrich, Alexander and Hoeppner, Hannes and Leidner, Daniel and Albu-Schäffer, Alin},
    year = {2020}
}
@inproceedings{quere_shared_2020,
    address = {Paris, France},
    title = {Shared {Control} {Templates} for {Assistive} {Robotics}},
    language = {en},
    booktitle = {2020 {IEEE} {International} {Conference} on {Robotics} and {Automation} ({ICRA})},
    author = {Quere, Gabriel and Hagengruber, Annette and Iskandar, Maged and Bustamante, Samuel and Leidner, Daniel and Stulp, Freek and Vogel, Joern},
    year = {2020},
    pages = {7},
}
```