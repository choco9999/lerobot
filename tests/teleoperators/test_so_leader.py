#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from unittest.mock import MagicMock, patch

import pytest

from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig


@pytest.fixture
def leader(tmp_path):
    bus_mock = MagicMock(name="FeetechBusMock")
    bus_mock.is_connected = False

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        motors_order: list[str] = list(bus_mock.motors)

        bus_mock.sync_read.return_value = {motor: idx for idx, motor in enumerate(motors_order, 1)}
        bus_mock.write.return_value = None
        bus_mock.disable_torque.return_value = None
        bus_mock.is_calibrated = True
        return bus_mock

    def _connect():
        bus_mock.is_connected = True

    def _disconnect():
        bus_mock.is_connected = False

    bus_mock.connect.side_effect = _connect
    bus_mock.disconnect.side_effect = _disconnect

    with (
        patch(
            "lerobot.teleoperators.so_leader.so_leader.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(SO100Leader, "configure", lambda self: None),
    ):
        teleop = SO100Leader(
            SO100LeaderConfig(
                id="test_so100_leader",
                calibration_dir=tmp_path,
                port="/dev/null",
            )
        )
        yield teleop
        if teleop.is_connected:
            teleop.disconnect()


def test_calibrate_records_wrist_roll_range(leader):
    leader.bus.set_half_turn_homings.return_value = {
        motor: idx * 100 for idx, motor in enumerate(leader.bus.motors, 1)
    }
    range_mins = {motor: idx * 10 for idx, motor in enumerate(leader.bus.motors, 1)}
    range_maxes = {motor: value + 50 for motor, value in range_mins.items()}
    leader.bus.record_ranges_of_motion.return_value = (range_mins, range_maxes)

    with (
        patch("builtins.input", return_value=""),
        patch.object(SO100Leader, "_save_calibration", autospec=True),
    ):
        leader.calibrate()

    leader.bus.record_ranges_of_motion.assert_called_once_with()
    assert leader.calibration["wrist_roll"].range_min == range_mins["wrist_roll"]
    assert leader.calibration["wrist_roll"].range_max == range_maxes["wrist_roll"]
