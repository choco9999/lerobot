#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import os
import sys
import time
from queue import Queue

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_so101_leader import SO101LeaderConfig

logger = logging.getLogger(__name__)

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard as pynput_keyboard
except Exception as e:  # noqa: BLE001
    logger.info(f"pynput not available for SO101Leader events: {e}")
    pynput_keyboard = None
    PYNPUT_AVAILABLE = False


class SO101Leader(Teleoperator):
    """
    SO-101 Leader Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = SO101LeaderConfig
    name = "so101_leader"

    def __init__(self, config: SO101LeaderConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        self._event_queue: Queue[str] = Queue()
        self._keyboard_listener = None
        self._intervention_active = False
        self._pending_terminate_episode = False
        self._pending_success = False
        self._pending_rerecord_episode = False
        self._space_pressed = False
        self._last_action: dict[str, float] | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()

        if PYNPUT_AVAILABLE and pynput_keyboard is not None:
            self._keyboard_listener = pynput_keyboard.Listener(
                on_press=self._on_key_press, on_release=self._on_key_release
            )
            self._keyboard_listener.start()

        logger.info(f"{self} connected.")

    def _on_key_press(self, key) -> None:
        if not PYNPUT_AVAILABLE or pynput_keyboard is None:
            return

        try:
            if key == pynput_keyboard.Key.space:
                # Debounce key repeat while space is held down.
                if not self._space_pressed:
                    self._space_pressed = True
                    self._event_queue.put("space")
            elif key == pynput_keyboard.Key.esc:
                self._event_queue.put("esc")
            elif hasattr(key, "char") and key.char is not None:
                if key.char in {"s", "r", "q"}:
                    self._event_queue.put(key.char)
        except Exception:  # noqa: BLE001
            return

    def _on_key_release(self, key) -> None:
        if not PYNPUT_AVAILABLE or pynput_keyboard is None:
            return
        try:
            if key == pynput_keyboard.Key.space:
                self._space_pressed = False
        except Exception:  # noqa: BLE001
            return

    def get_teleop_events(self) -> dict:
        if not (PYNPUT_AVAILABLE and pynput_keyboard is not None and self._keyboard_listener is not None):
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        while not self._event_queue.empty():
            key = self._event_queue.get_nowait()
            if key == "space":
                self._intervention_active = not self._intervention_active
            elif key == "s":
                self._pending_success = True
            elif key == "r":
                self._pending_terminate_episode = True
                self._pending_rerecord_episode = True
            elif key in {"q", "esc"}:
                self._pending_terminate_episode = True

        terminate_episode = self._pending_terminate_episode
        success = self._pending_success
        rerecord_episode = self._pending_rerecord_episode

        self._pending_terminate_episode = False
        self._pending_success = False
        self._pending_rerecord_episode = False

        return {
            TeleopEvents.IS_INTERVENTION: self._intervention_active,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        try:
            action_raw = self.bus.sync_read("Present_Position", num_retry=2)
        except ConnectionError as e:
            if self._last_action is not None:
                logger.warning("%s failed to read action (%s); using last action.", self, e)
                return dict(self._last_action)
            raise

        action = {f"{motor}.pos": val for motor, val in action_raw.items()}
        self._last_action = dict(action)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
