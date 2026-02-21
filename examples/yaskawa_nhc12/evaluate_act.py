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
Evaluate trained ACT policy on YASKAWA NHC12 robot.

This script loads a trained ACT model and runs it on the real robot.

Usage:
    python evaluate_act.py --policy_path <hf_username>/<model_name>
"""

import argparse
import time

import torch

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.policies.factory import make_policy
from lerobot.robots.yaskawa_nhc12 import YaskawaNHC12Config, YaskawaNHC12Robot
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ========================================
# Configuration
# ========================================

# Robot Configuration
ROBOT_IP = "192.168.1.31"
ROBOT_PORT = 10040

# Evaluation Settings
NUM_EPISODES = 10  # Number of episodes to evaluate
MAX_STEPS_PER_EPISODE = 300  # Maximum steps per episode
FPS = 30  # Control frequency (Hz)

# Camera Configuration
CAMERA_CONFIGS = {
    "wrist": RealSenseCameraConfig(
        serial_number=None,
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


def evaluate_episode(
    robot: YaskawaNHC12Robot,
    policy: torch.nn.Module,
    episode_idx: int,
    max_steps: int,
    fps: int,
) -> dict:
    """
    Run one evaluation episode.

    Args:
        robot: Connected YASKAWA robot
        policy: Trained ACT policy
        episode_idx: Current episode number
        max_steps: Maximum number of steps
        fps: Control frequency in Hz

    Returns:
        Dictionary with episode statistics
    """
    log_say(f"Episode {episode_idx + 1}/{NUM_EPISODES}")
    log_say("Position the robot at the starting pose and press Enter...")
    input()

    log_say("Starting episode...")

    step_count = 0
    start_time = time.time()
    episode_rewards = []

    try:
        while step_count < max_steps:
            step_start = time.time()

            # Get observation from robot
            observation = robot.get_observation()

            # Get action from policy
            with torch.no_grad():
                action = policy.select_action(observation)

            # Send action to robot
            robot.send_action(action)

            # Log data for visualization
            try:
                log_rerun_data(observation, action, step_count)
            except Exception:
                pass

            step_count += 1

            # Status update
            if step_count % (fps * 5) == 0:  # Every 5 seconds
                elapsed = time.time() - start_time
                log_say(f"Step {step_count}/{max_steps} ({elapsed:.1f}s)")

            # Maintain control frequency
            elapsed_step = time.time() - step_start
            sleep_time = (1.0 / fps) - elapsed_step
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        log_say("Episode interrupted by user")

    total_time = time.time() - start_time
    log_say(f"Episode completed: {step_count} steps in {total_time:.1f}s")

    return {
        "episode_idx": episode_idx,
        "num_steps": step_count,
        "duration": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACT policy on YASKAWA NHC12")
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="Path to trained policy (e.g., <hf_username>/<model_name>)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the policy on (cuda/cpu)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=NUM_EPISODES,
        help="Number of episodes to evaluate",
    )
    args = parser.parse_args()

    log_say("YASKAWA NHC12 ACT Policy Evaluation")
    log_say("=" * 50)

    # Create robot configuration
    robot_config = YaskawaNHC12Config(
        ip_address=ROBOT_IP,
        port=ROBOT_PORT,
        num_joints=len(JOINT_NAMES),
        joint_names=JOINT_NAMES,
        joint_limits_deg=JOINT_LIMITS,
        cameras=CAMERA_CONFIGS,
        enable_direct_teach=False,  # Disable direct teach for automatic control
        use_degrees=True,
    )

    # Initialize robot
    log_say(f"Connecting to YASKAWA NHC12 at {ROBOT_IP}:{ROBOT_PORT}...")
    robot = YaskawaNHC12Robot(robot_config)
    robot.connect()

    if not robot.is_connected:
        raise RuntimeError("Failed to connect to robot")

    log_say("Robot connected successfully")

    # Load policy
    log_say(f"Loading policy from {args.policy_path}...")
    policy = make_policy(
        policy_path=args.policy_path,
        device=args.device,
    )
    log_say("Policy loaded successfully")

    # Initialize visualization
    init_rerun(session_name="yaskawa_nhc12_evaluation")

    try:
        # Run evaluation episodes
        episode_stats = []

        for episode_idx in range(args.num_episodes):
            log_say("")
            log_say("-" * 50)

            stats = evaluate_episode(
                robot=robot,
                policy=policy,
                episode_idx=episode_idx,
                max_steps=MAX_STEPS_PER_EPISODE,
                fps=FPS,
            )
            episode_stats.append(stats)

        # Print summary
        log_say("")
        log_say("=" * 50)
        log_say("Evaluation Summary")
        log_say("-" * 50)

        total_steps = sum(s["num_steps"] for s in episode_stats)
        total_time = sum(s["duration"] for s in episode_stats)
        avg_steps = total_steps / len(episode_stats)
        avg_time = total_time / len(episode_stats)

        log_say(f"Total episodes: {len(episode_stats)}")
        log_say(f"Total steps: {total_steps}")
        log_say(f"Total time: {total_time:.1f}s")
        log_say(f"Average steps per episode: {avg_steps:.1f}")
        log_say(f"Average time per episode: {avg_time:.1f}s")

        # Ask user for success rate
        log_say("")
        log_say("Please manually assess the success rate:")
        log_say(f"How many episodes were successful? (0-{len(episode_stats)})")
        try:
            success_count = int(input().strip())
            success_rate = (success_count / len(episode_stats)) * 100
            log_say(f"Success rate: {success_rate:.1f}% ({success_count}/{len(episode_stats)})")
        except ValueError:
            log_say("Invalid input, skipping success rate calculation")

    except Exception as e:
        log_say(f"Error during evaluation: {e}")
        raise

    finally:
        # Disconnect robot
        log_say("Disconnecting robot...")
        robot.disconnect()
        log_say("Done!")


if __name__ == "__main__":
    main()
