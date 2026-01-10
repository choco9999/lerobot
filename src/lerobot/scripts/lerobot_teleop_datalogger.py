"""
Teleoperate while logging joint tracking + motor diagnostics to CSV.

Example:

```bash
python -m src.lerobot.scripts.lerobot_teleop_datalogger \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_awesome_leader_arm \
  --joint=shoulder_lift \
  --fps=60 \
  --duration_s=30
```
"""

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    so_leader,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


def _safe_call(fn, *args, **kwargs) -> tuple[Any, str]:
    try:
        return fn(*args, **kwargs), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


@dataclass
class TeleopDataloggerConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig

    # Control loop frequency (Hz)
    fps: int = 60
    # Stop after this many seconds (None = run until Ctrl+C)
    duration_s: float | None = None

    # Joint name to focus diagnostics on (SO-101 axis 2 = "shoulder_lift")
    joint: str = "shoulder_lift"
    # Also log all joints' leader/follower positions
    include_all_joints: bool = False

    # Read follower positions at this rate (Hz). Use 0 to disable.
    pos_hz: int = 30
    # Read extra diagnostic registers at this rate (Hz)
    diagnostics_hz: int = 10
    # Try re-reading when a read fails
    read_num_retry: int = 2

    # Output files
    output_dir: Path = Path("outputs/teleop_logs")
    output_name: str | None = None


def _read_static_registers(robot: Robot, joint: str, read_num_retry: int) -> dict[str, Any]:
    if not hasattr(robot, "bus"):
        return {}

    bus = robot.bus
    regs_raw = [
        "Torque_Enable",
        "Operating_Mode",
        "P_Coefficient",
        "I_Coefficient",
        "D_Coefficient",
        "Acceleration",
        "Min_Position_Limit",
        "Max_Position_Limit",
        "Max_Torque_Limit",
        "Torque_Limit",
        "Protection_Current",
        "Overload_Torque",
        "Phase",
    ]

    out: dict[str, Any] = {}
    for reg in regs_raw:
        val, err = _safe_call(bus.read, reg, joint, normalize=False, num_retry=read_num_retry)
        out[reg] = {"value": val, "error": err} if err else {"value": val}
    return out


def teleop_datalogger_loop(
    *,
    teleop: Teleoperator,
    robot: Robot,
    cfg: TeleopDataloggerConfig,
    csv_path: Path,
    meta_path: Path,
) -> None:
    teleop_action_processor, robot_action_processor, _ = make_default_processors()

    motors: list[str]
    if hasattr(robot, "bus") and hasattr(robot.bus, "motors"):
        motors = list(robot.bus.motors)
    else:
        motors = sorted({k.removesuffix(".pos") for k in robot.action_features if k.endswith(".pos")})

    if cfg.joint not in motors:
        raise ValueError(f"Unknown joint '{cfg.joint}'. Known joints: {motors}")

    log_motors = motors if cfg.include_all_joints else [cfg.joint]

    fieldnames = [
        "t_s",
        "loop_dt_ms",
        "teleop_dt_ms",
        "send_dt_ms",
        "pos_dt_ms",
        "diag_dt_ms",
        "pos_sampled",
        "diag_sampled",
        "teleop_error",
        "send_error",
        "pos_error",
        "diag_error",
    ]
    for motor in log_motors:
        fieldnames.extend(
            [
                f"leader.{motor}.pos",
                f"sent.{motor}.pos",
                f"follower.{motor}.pos",
                f"error.{motor}.pos",
            ]
        )

    # Focus joint diagnostics (raw)
    diag_fields = [
        f"follower.{cfg.joint}.pos_raw",
        f"follower.{cfg.joint}.goal_raw",
        f"follower.{cfg.joint}.torque_enable_raw",
        f"follower.{cfg.joint}.operating_mode_raw",
        f"follower.{cfg.joint}.velocity_raw",
        f"follower.{cfg.joint}.moving_raw",
        f"follower.{cfg.joint}.load_raw",
        f"follower.{cfg.joint}.current_raw",
        f"follower.{cfg.joint}.voltage_raw",
        f"follower.{cfg.joint}.temperature_raw",
        f"follower.{cfg.joint}.status_raw",
    ]
    fieldnames.extend(diag_fields)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "csv_path": str(csv_path),
        "config": asdict(cfg),
        "robot_action_features": sorted(robot.action_features),
        "teleop_action_features": sorted(teleop.action_features),
        "static_registers": _read_static_registers(robot, cfg.joint, cfg.read_num_retry),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=_json_default))

    last_action: RobotAction | None = None
    last_positions: dict[str, float] = {}
    last_pos_time = 0.0
    last_diag: dict[str, Any] = {}
    last_diag_time = 0.0

    start_t = time.perf_counter()

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        row_i = 0
        while True:
            loop_t0 = time.perf_counter()
            t_s = loop_t0 - start_t

            row: dict[str, Any] = {
                "t_s": t_s,
                "teleop_error": "",
                "send_error": "",
                "pos_error": "",
                "diag_error": "",
                "pos_sampled": False,
                "diag_sampled": False,
            }

            # 1) Teleop action
            action_t0 = time.perf_counter()
            action, teleop_err = _safe_call(teleop.get_action)
            action_t1 = time.perf_counter()
            row["teleop_dt_ms"] = (action_t1 - action_t0) * 1e3
            if teleop_err:
                row["teleop_error"] = teleop_err
                action = dict(last_action) if last_action is not None else {}
            else:
                last_action = dict(action)

            # 2) Map + send to robot
            proc_t0 = time.perf_counter()
            teleop_action = teleop_action_processor((action, None))  # observation not needed by default pipeline
            robot_action_to_send = robot_action_processor((teleop_action, None))
            proc_t1 = time.perf_counter()
            _ = proc_t1  # placeholder for future timing breakdown if needed

            send_t0 = time.perf_counter()
            sent_action, send_err = _safe_call(robot.send_action, robot_action_to_send)
            send_t1 = time.perf_counter()
            row["send_dt_ms"] = (send_t1 - send_t0) * 1e3
            if send_err:
                row["send_error"] = send_err
                sent_action = {}

            # 3) Read follower positions (normalized)
            pos_t0 = time.perf_counter()
            positions: dict[str, float] = {}
            pos_err = ""
            if cfg.pos_hz > 0 and hasattr(robot, "bus") and (t_s - last_pos_time) >= (1.0 / cfg.pos_hz):
                positions, pos_err = _safe_call(
                    robot.bus.sync_read,
                    "Present_Position",
                    log_motors,
                    normalize=True,
                    num_retry=cfg.read_num_retry,
                )
                last_pos_time = t_s
                row["pos_sampled"] = True
            pos_t1 = time.perf_counter()
            row["pos_dt_ms"] = (pos_t1 - pos_t0) * 1e3
            if pos_err:
                row["pos_error"] = pos_err
                positions = dict(last_positions)
            else:
                if positions:
                    last_positions = dict(positions)

            # 4) Diagnostics (raw, lower rate)
            diag_t0 = time.perf_counter()
            if cfg.diagnostics_hz > 0 and (t_s - last_diag_time) >= (1.0 / cfg.diagnostics_hz):
                diag: dict[str, Any] = {}
                if hasattr(robot, "bus"):
                    bus = robot.bus
                    # Raw present/goal positions to see saturation or stuck-at-limit.
                    diag[f"follower.{cfg.joint}.pos_raw"], err = _safe_call(
                        bus.read, "Present_Position", cfg.joint, normalize=False, num_retry=cfg.read_num_retry
                    )
                    if err:
                        row["diag_error"] = err
                    diag[f"follower.{cfg.joint}.goal_raw"], err = _safe_call(
                        bus.read, "Goal_Position", cfg.joint, normalize=False, num_retry=cfg.read_num_retry
                    )
                    if err and not row["diag_error"]:
                        row["diag_error"] = err

                    # Dynamic status registers.
                    for reg, key in [
                        ("Torque_Enable", "torque_enable_raw"),
                        ("Operating_Mode", "operating_mode_raw"),
                        ("Present_Velocity", "velocity_raw"),
                        ("Moving", "moving_raw"),
                        ("Present_Load", "load_raw"),
                        ("Present_Current", "current_raw"),
                        ("Present_Voltage", "voltage_raw"),
                        ("Present_Temperature", "temperature_raw"),
                        ("Status", "status_raw"),
                    ]:
                        diag_key = f"follower.{cfg.joint}.{key}"
                        diag[diag_key], err = _safe_call(
                            bus.read, reg, cfg.joint, normalize=False, num_retry=cfg.read_num_retry
                        )
                        if err and not row["diag_error"]:
                            row["diag_error"] = err

                last_diag = dict(diag)
                last_diag_time = t_s
                row["diag_sampled"] = True
            diag_t1 = time.perf_counter()
            row["diag_dt_ms"] = (diag_t1 - diag_t0) * 1e3

            # 5) Flatten signals into row
            for motor in log_motors:
                leader_val = action.get(f"{motor}.pos")
                if leader_val is not None:
                    row[f"leader.{motor}.pos"] = leader_val
                sent_val = sent_action.get(f"{motor}.pos")
                if sent_val is not None:
                    row[f"sent.{motor}.pos"] = sent_val
                follower_val = positions.get(motor)
                if follower_val is not None:
                    row[f"follower.{motor}.pos"] = follower_val
                if (leader_val is not None) and (follower_val is not None):
                    row[f"error.{motor}.pos"] = leader_val - follower_val

            row.update(last_diag)

            loop_t1 = time.perf_counter()
            row["loop_dt_ms"] = (loop_t1 - loop_t0) * 1e3

            writer.writerow(row)
            row_i += 1
            if row_i % 50 == 0:
                f.flush()

            if cfg.duration_s is not None and t_s >= cfg.duration_s:
                return

            precise_sleep(1 / cfg.fps - (time.perf_counter() - loop_t0))


@parser.wrap()
def teleop_datalogger(cfg: TeleopDataloggerConfig):
    init_logging()
    register_third_party_plugins()

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = cfg.output_name or f"teleop_log_{cfg.robot.type}_{cfg.teleop.type}_{timestamp}.csv"
    csv_path = (cfg.output_dir / out_name).resolve()
    meta_path = csv_path.with_suffix(".meta.json")

    logger.info("Writing CSV to %s", csv_path)
    logger.info("Writing metadata to %s", meta_path)

    teleop.connect()
    robot.connect()

    try:
        teleop_datalogger_loop(
            teleop=teleop,
            robot=robot,
            cfg=cfg,
            csv_path=csv_path,
            meta_path=meta_path,
        )
    except KeyboardInterrupt:
        pass
    finally:
        try:
            teleop.disconnect()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to disconnect teleop cleanly.")
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to disconnect robot cleanly.")


def main():
    teleop_datalogger()


if __name__ == "__main__":
    main()
