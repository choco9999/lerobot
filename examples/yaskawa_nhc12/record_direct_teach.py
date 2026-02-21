#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Direct Teach Data Collection for YASKAWA NHC12

This script enables data collection using direct teach mode (gravity compensation).
The robot can be manually moved to demonstrate tasks, and the movements are recorded
along with camera observations.

Usage:
    python record_direct_teach.py

Controls during recording:
    - Physically move the robot to demonstrate the task
    - Press 'Enter' to start/stop episode recording
    - Press 'q' to quit
"""

import time

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.robots.yaskawa_nhc12 import YaskawaNHC12Config, YaskawaNHC12Robot
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ========================================
# Configuration - Update these parameters
# ========================================

# Robot Configuration
ROBOT_IP = "192.168.1.31"  # TODO: Update to your robot's IP address
ROBOT_PORT = 10040  # TODO: Update if using different port

# Dataset Configuration
HF_REPO_ID = "<your_hf_username>/<dataset_name>"  # TODO: Update with your HuggingFace username
TASK_DESCRIPTION = "Pick and place object"  # TODO: Describe your task
NUM_EPISODES = 50  # Number of episodes to record
FPS = 30  # Recording frequency (Hz)
EPISODE_TIME_SEC = 60  # Maximum episode duration in seconds

# Camera Configuration (RealSense)
# TODO: Update camera serial numbers or use index
CAMERA_CONFIGS = {
    "wrist": RealSenseCameraConfig(
        serial_number=None,  # Or specify serial like "123456789"
        width=640,
        height=480,
        fps=FPS,
    ),
    "front": RealSenseCameraConfig(
        serial_number=None,
        width=640,
        height=480,
        fps=FPS,
    ),
}

# Joint Configuration
# TODO: Update based on your robot's specifications
JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]

JOINT_LIMITS = {
    "joint_1": (-170.0, 170.0),
    "joint_2": (-130.0, 90.0),
    "joint_3": (-170.0, 90.0),
    "joint_4": (-200.0, 200.0),
    "joint_5": (-135.0, 135.0),
    "joint_6": (-360.0, 360.0),
}


def record_episode(
    robot: YaskawaNHC12Robot,
    dataset: LeRobotDataset,
    episode_idx: int,
    fps: int,
    max_duration_s: float,
) -> bool:
    """
    Record a single episode using direct teach.

    Args:
        robot: Connected YASKAWA robot
        dataset: LeRobot dataset for saving data
        episode_idx: Current episode number
        fps: Recording frequency in Hz
        max_duration_s: Maximum episode duration in seconds

    Returns:
        True if episode was recorded successfully, False if user wants to re-record
    """
    log_say(f"Episode {episode_idx + 1}/{NUM_EPISODES}")
    log_say("Physically move the robot to demonstrate the task")
    log_say("Press Enter when ready to start recording...")
    input()

    log_say("Recording started! Move the robot to perform the task...")

    frame_count = 0
    max_frames = int(fps * max_duration_s)
    start_time = time.time()

    try:
        while frame_count < max_frames:
            frame_start = time.time()

            # Get observation (joint positions + camera images)
            observation = robot.get_observation()

            # In direct teach mode, the action is the same as observation
            # (we record where the robot is, not where we command it to go)
            action = {
                key: value
                for key, value in observation.items()
                if key in robot.action_features
            }

            # Add frame to dataset
            dataset.add_frame(observation, action)

            # Visualize data if rerun is initialized
            try:
                log_rerun_data(observation, action, frame_count)
            except Exception:
                pass  # Continue even if visualization fails

            frame_count += 1

            # Status update every second
            if frame_count % fps == 0:
                elapsed = time.time() - start_time
                log_say(f"Recorded {frame_count} frames ({elapsed:.1f}s)")

            # Sleep to maintain FPS
            elapsed_frame = time.time() - frame_start
            sleep_time = (1.0 / fps) - elapsed_frame
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        log_say("Recording interrupted by user")

    total_time = time.time() - start_time
    log_say(f"Episode recorded: {frame_count} frames in {total_time:.1f}s")

    # Ask if user wants to keep or re-record
    log_say("Keep this episode? (y/n/q to quit)")
    response = input().strip().lower()

    if response == 'q':
        return None  # Signal to quit
    elif response == 'n':
        log_say("Discarding episode...")
        dataset.clear_episode_buffer()
        return False  # Re-record
    else:
        log_say("Saving episode...")
        dataset.save_episode()
        return True  # Episode saved


def main():
    """Main data collection loop."""
    log_say("YASKAWA NHC12 Direct Teach Data Collection")
    log_say("=" * 50)

    # Create robot configuration
    robot_config = YaskawaNHC12Config(
        ip_address=ROBOT_IP,
        port=ROBOT_PORT,
        num_joints=len(JOINT_NAMES),
        joint_names=JOINT_NAMES,
        joint_limits_deg=JOINT_LIMITS,
        cameras=CAMERA_CONFIGS,
        enable_direct_teach=True,  # Enable direct teach mode
        use_degrees=True,
    )

    # Initialize robot
    log_say(f"Connecting to YASKAWA NHC12 at {ROBOT_IP}:{ROBOT_PORT}...")
    robot = YaskawaNHC12Robot(robot_config)
    robot.connect()

    if not robot.is_connected:
        raise RuntimeError("Failed to connect to robot")

    log_say("Robot connected successfully")

    # Create dataset
    log_say(f"Creating dataset: {HF_REPO_ID}")
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=robot.observation_features,  # Use robot's observation features
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Initialize visualization
    init_rerun(session_name="yaskawa_nhc12_direct_teach")

    try:
        episode_idx = 0
        while episode_idx < NUM_EPISODES:
            log_say("")
            log_say("-" * 50)

            # Record episode
            result = record_episode(
                robot=robot,
                dataset=dataset,
                episode_idx=episode_idx,
                fps=FPS,
                max_duration_s=EPISODE_TIME_SEC,
            )

            if result is None:
                # User wants to quit
                break
            elif result is True:
                # Episode saved successfully
                episode_idx += 1
            # If result is False, re-record (don't increment episode_idx)

        log_say("")
        log_say("=" * 50)
        log_say(f"Data collection complete! Recorded {episode_idx} episodes")

        # Finalize dataset
        log_say("Finalizing dataset...")
        dataset.finalize()

        # Optionally push to HuggingFace Hub
        log_say("Push dataset to HuggingFace Hub? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            log_say("Pushing to hub...")
            dataset.push_to_hub()
            log_say(f"Dataset uploaded: https://huggingface.co/datasets/{HF_REPO_ID}")

    except Exception as e:
        log_say(f"Error during data collection: {e}")
        raise

    finally:
        # Disconnect robot
        log_say("Disconnecting robot...")
        robot.disconnect()
        log_say("Done!")


if __name__ == "__main__":
    main()
