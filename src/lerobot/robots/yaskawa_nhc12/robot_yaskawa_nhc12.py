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

from __future__ import annotations

import logging
import socket
import time
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot

from .config_yaskawa_nhc12 import YaskawaNHC12Config

logger = logging.getLogger(__name__)


class YaskawaNHC12Robot(Robot):
    """
    YASKAWA NHC12 Robot Controller Integration for LeRobot.

    This class provides a LeRobot-compatible interface for YASKAWA NHC12 robot controllers.
    It supports:
    - TCP/IP communication with the controller
    - Direct teach mode (gravity compensation) for data collection
    - Joint position reading and control
    - Integration with RealSense cameras

    Note: This implementation requires the YASKAWA NHC12 controller to be configured
    for TCP/IP communication. Please refer to your controller's manual for:
    - Network configuration
    - Communication protocol specification
    - Command formats for position reading/writing
    - Direct teach mode activation commands
    """

    config_class = YaskawaNHC12Config
    name = "yaskawa_nhc12"

    def __init__(self, config: YaskawaNHC12Config):
        super().__init__(config)
        self.config = config

        # TCP/IP socket for communication
        self._socket: socket.socket | None = None
        self._connected = False

        # Current joint positions (in degrees or radians based on config)
        self._current_joint_positions = np.zeros(config.num_joints)

        # Direct teach mode state
        self._direct_teach_enabled = False

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

        logger.info(f"Initialized YASKAWA NHC12 robot with {config.num_joints} joints")

    @property
    def observation_features(self) -> dict[str, Any]:
        """Define the observation space (joint positions + camera images)."""
        features = {}

        # Add joint position features
        for joint_name in self.config.joint_names:
            features[f"{joint_name}.pos"] = float

        # Add camera features
        for cam_name, cam in self.cameras.items():
            features[cam_name] = (cam.height, cam.width, 3)

        return features

    @property
    def action_features(self) -> dict[str, type]:
        """Define the action space (target joint positions)."""
        return {f"{joint_name}.pos": float for joint_name in self.config.joint_names}

    @property
    def is_connected(self) -> bool:
        """Check if the robot is connected."""
        return self._connected and self._socket is not None

    def connect(self, calibrate: bool = False) -> None:
        """
        Connect to the YASKAWA NHC12 controller via TCP/IP.

        Args:
            calibrate: Not used for this robot (always calibrated).
        """
        try:
            # Create TCP socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.config.connection_timeout)

            # Connect to controller
            logger.info(f"Connecting to YASKAWA NHC12 at {self.config.ip_address}:{self.config.port}")
            self._socket.connect((self.config.ip_address, self.config.port))

            self._connected = True
            logger.info("Successfully connected to YASKAWA NHC12")

            # Connect cameras
            for cam_name, cam in self.cameras.items():
                logger.info(f"Connecting camera: {cam_name}")
                cam.connect()

            # Configure the robot
            self.configure()

        except socket.timeout:
            raise ConnectionError(
                f"Connection timeout: Could not connect to YASKAWA NHC12 at "
                f"{self.config.ip_address}:{self.config.port}. "
                f"Please check the IP address and ensure the controller is powered on."
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to YASKAWA NHC12: {e}")

    def configure(self) -> None:
        """
        Configure the robot controller after connection.

        This includes:
        - Setting control mode
        - Enabling direct teach mode if configured
        - Setting safety parameters
        """
        if not self.is_connected:
            raise RuntimeError("Robot is not connected. Call connect() first.")

        # TODO: Send configuration commands to the controller
        # Example commands (update based on YASKAWA protocol):
        # - Set control mode to position control
        # - Enable servo power
        # - Set cycle time
        # - Configure safety limits

        logger.info("Configuring YASKAWA NHC12...")

        # Enable direct teach mode if configured
        if self.config.enable_direct_teach:
            self._enable_direct_teach_mode()

        logger.info("Configuration complete")

    def _enable_direct_teach_mode(self) -> None:
        """
        Enable direct teach mode (gravity compensation).

        TODO: Implement the specific command sequence for your YASKAWA controller.
        This typically involves:
        1. Sending a command to enable gravity compensation
        2. Reducing motor stiffness to allow manual movement
        3. Enabling force/torque sensing if available
        """
        logger.info("Enabling direct teach mode...")

        # TODO: Send direct teach enable command
        # Example (update based on actual protocol):
        # command = "TEACH_MODE ON\r\n"
        # self._send_command(command)

        self._direct_teach_enabled = True
        logger.info("Direct teach mode enabled")

    def _disable_direct_teach_mode(self) -> None:
        """Disable direct teach mode and return to normal control."""
        if not self._direct_teach_enabled:
            return

        logger.info("Disabling direct teach mode...")

        # TODO: Send direct teach disable command
        # Example:
        # command = "TEACH_MODE OFF\r\n"
        # self._send_command(command)

        self._direct_teach_enabled = False
        logger.info("Direct teach mode disabled")

    def _send_command(self, command: str | bytes) -> str:
        """
        Send a command to the YASKAWA controller and receive response.

        Args:
            command: Command string or bytes to send

        Returns:
            Response from the controller

        TODO: Implement based on YASKAWA NHC12 protocol specification.
        """
        if not self.is_connected:
            raise RuntimeError("Robot is not connected")

        try:
            # Convert string to bytes if needed
            if isinstance(command, str):
                command = command.encode('utf-8')

            # Send command
            self._socket.sendall(command)

            # Receive response
            response = self._socket.recv(4096)
            return response.decode('utf-8')

        except Exception as e:
            logger.error(f"Communication error: {e}")
            raise

    def _read_joint_positions(self) -> np.ndarray:
        """
        Read current joint positions from the controller.

        Returns:
            Array of joint positions (in degrees or radians based on config)

        TODO: Implement based on YASKAWA NHC12 protocol.
        Example command: "READ_JOINT_POS\r\n"
        Expected response format: "J1=10.5,J2=20.3,J3=30.1,J4=40.2,J5=50.0,J6=60.1\r\n"
        """
        # TODO: Send read position command and parse response
        # command = "READ_JOINT_POS\r\n"
        # response = self._send_command(command)
        # positions = self._parse_joint_positions(response)
        # return positions

        # Placeholder: return last known positions
        return self._current_joint_positions

    def _write_joint_positions(self, positions: np.ndarray) -> None:
        """
        Send target joint positions to the controller.

        Args:
            positions: Target joint positions (in degrees or radians based on config)

        TODO: Implement based on YASKAWA NHC12 protocol.
        Example command: "MOVE_JOINT J1=10.5,J2=20.3,J3=30.1,J4=40.2,J5=50.0,J6=60.1\r\n"
        """
        if len(positions) != self.config.num_joints:
            raise ValueError(
                f"Expected {self.config.num_joints} joint positions, got {len(positions)}"
            )

        # Check safety limits if enabled
        if self.config.enable_safety_limits:
            self._check_safety_limits(positions)

        # TODO: Format and send move command
        # Example:
        # joint_str = ",".join([f"J{i+1}={pos:.2f}" for i, pos in enumerate(positions)])
        # command = f"MOVE_JOINT {joint_str}\r\n"
        # self._send_command(command)

        logger.debug(f"Sent joint positions: {positions}")

    def _check_safety_limits(self, positions: np.ndarray) -> None:
        """
        Check if joint positions are within safety limits.

        Args:
            positions: Joint positions to check

        Raises:
            ValueError: If any joint position exceeds limits
        """
        for i, (joint_name, pos) in enumerate(zip(self.config.joint_names, positions)):
            if joint_name in self.config.joint_limits_deg:
                min_pos, max_pos = self.config.joint_limits_deg[joint_name]
                if pos < min_pos or pos > max_pos:
                    error_msg = (
                        f"Joint {joint_name} position {pos:.2f} exceeds limits "
                        f"[{min_pos:.2f}, {max_pos:.2f}]"
                    )
                    logger.error(error_msg)
                    if self.config.emergency_stop_on_limit:
                        raise ValueError(error_msg)
                    else:
                        logger.warning(f"{error_msg} - Clamping to limits")
                        positions[i] = np.clip(pos, min_pos, max_pos)

    @property
    def is_calibrated(self) -> bool:
        """YASKAWA robots are always calibrated by the controller."""
        return True

    def calibrate(self) -> None:
        """Not applicable for YASKAWA robots (always calibrated)."""
        pass

    def get_observation(self) -> RobotObservation:
        """
        Get current observation from the robot (joint positions + camera images).

        Returns:
            RobotObservation dictionary with joint positions and camera images
        """
        if not self.is_connected:
            raise RuntimeError("Robot is not connected")

        observation = {}

        # Read joint positions
        self._current_joint_positions = self._read_joint_positions()

        # Add joint positions to observation
        for joint_name, pos in zip(self.config.joint_names, self._current_joint_positions):
            observation[f"{joint_name}.pos"] = float(pos)

        # Capture camera images
        for cam_name, cam in self.cameras.items():
            observation[cam_name] = cam.get_image()

        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Send action command to the robot.

        Args:
            action: Dictionary of target joint positions

        Returns:
            The action that was actually sent (potentially clipped)
        """
        if not self.is_connected:
            raise RuntimeError("Robot is not connected")

        # Extract joint positions from action
        target_positions = np.array([
            action[f"{joint_name}.pos"]
            for joint_name in self.config.joint_names
        ])

        # If in direct teach mode, don't send commands (robot is manually controlled)
        if self._direct_teach_enabled:
            logger.debug("Direct teach mode active - not sending position commands")
            return action

        # Send positions to robot
        self._write_joint_positions(target_positions)

        return action

    def disconnect(self) -> None:
        """Disconnect from the robot and clean up resources."""
        logger.info("Disconnecting from YASKAWA NHC12...")

        # Disable direct teach mode if enabled
        if self._direct_teach_enabled:
            self._disable_direct_teach_mode()

        # Disconnect cameras
        for cam_name, cam in self.cameras.items():
            try:
                cam.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting camera {cam_name}: {e}")

        # Close socket connection
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception as e:
                logger.warning(f"Error closing socket: {e}")
            finally:
                self._socket = None

        self._connected = False
        logger.info("Disconnected from YASKAWA NHC12")
