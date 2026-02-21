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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@dataclass
class YaskawaNHC12Config(RobotConfig):
    """
    Configuration class for YASKAWA NHC12 robot controller.

    The YASKAWA NHC12 is a compact robot controller that communicates via TCP/IP.
    This configuration enables integration with LeRobot for data collection and policy deployment.
    """

    # Robot IP address for Ethernet/TCP-IP communication
    ip_address: str = "192.168.1.31"

    # Port number for TCP/IP communication (default: 10040 for YASKAWA controllers)
    port: int = 10040

    # Number of joints in the robot (typically 6 for industrial arms)
    # TODO: Update this based on your specific YASKAWA robot model
    num_joints: int = 6

    # Joint names in order (customize based on your robot)
    # TODO: Update these to match your robot's joint naming convention
    joint_names: list[str] = field(default_factory=lambda: [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
    ])

    # Camera configurations (RealSense cameras supported)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Connection timeout in seconds
    connection_timeout: float = 5.0

    # Communication cycle time in seconds (how often to read/write data)
    cycle_time: float = 0.008  # 8ms = 125Hz

    # Enable direct teach mode (gravity compensation)
    enable_direct_teach: bool = True

    # Joint limits in degrees (min, max for each joint)
    # TODO: Update these based on your robot's specifications
    joint_limits_deg: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "joint_1": (-170.0, 170.0),
        "joint_2": (-130.0, 90.0),
        "joint_3": (-170.0, 90.0),
        "joint_4": (-200.0, 200.0),
        "joint_5": (-135.0, 135.0),
        "joint_6": (-360.0, 360.0),
    })

    # Maximum joint velocity in degrees per second
    # TODO: Update based on your robot's safe operating parameters
    max_joint_velocity_deg_s: float = 100.0

    # Use degrees instead of radians for joint angles
    use_degrees: bool = True

    # Safety parameters
    enable_safety_limits: bool = True
    emergency_stop_on_limit: bool = True

    def __post_init__(self):
        super().__post_init__()

        # Validate joint names match number of joints
        if len(self.joint_names) != self.num_joints:
            raise ValueError(
                f"Number of joint names ({len(self.joint_names)}) must match "
                f"num_joints ({self.num_joints})"
            )

        # Validate joint limits are provided for all joints
        if self.enable_safety_limits and len(self.joint_limits_deg) != self.num_joints:
            raise ValueError(
                f"Joint limits must be provided for all {self.num_joints} joints. "
                f"Got {len(self.joint_limits_deg)} limits."
            )
