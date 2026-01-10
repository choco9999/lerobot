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
"""
Actor server runner for distributed HILSerl robot policy training.

This script implements the actor component of the distributed HILSerl architecture.
It executes the policy in the robot environment, collects experience,
and sends transitions to the learner server for policy updates.

Examples of usage:

- Start an actor server for real robot training with human-in-the-loop intervention:
```bash
python -m lerobot.rl.actor --config_path src/lerobot/configs/train_config_hilserl_so101.json
```

**NOTE**: The actor server requires a running learner server to connect to. Ensure the learner
server is started before launching the actor.

**NOTE**: Human intervention is key to HILSerl training. Press the upper right trigger button on the
gamepad to take control of the robot during training. Initially intervene frequently, then gradually
reduce interventions as the policy improves.

**WORKFLOW**:
1. Determine robot workspace bounds using `lerobot-find-joint-limits`
2. Record demonstrations with `gym_manipulator.py` in record mode
3. Process the dataset and determine camera crops with `crop_dataset_roi.py`
4. Start the learner server with the training configuration
5. Start this actor server with the same configuration
6. Use human interventions to guide policy learning

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import sys
import time
import json
from functools import lru_cache
from queue import Empty, Full, Queue as ThreadQueue

import grpc
import torch
from torch import nn
from torch.multiprocessing import Queue as MpQueue

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.processor import TransitionKey
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.queue import get_last_item_from_queue
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import (
    bytes_to_state_dict,
    grpc_channel_options,
    python_object_to_bytes,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    transitions_to_bytes,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import (
    Transition,
    move_state_dict_to_device,
    move_transition_to_device,
)
from lerobot.utils.constants import OBS_IMAGE, OBS_STATE
from lerobot.utils.utils import (
    TimerManager,
    get_safe_torch_device,
    init_logging,
)

from .gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)

# Main entry point


@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()
    display_pid = False
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_{cfg.job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Actor logging initialized, writing to {log_file}")

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    learner_client, grpc_channel = learner_service_client(
        host=cfg.policy.actor_learner_config.learner_host,
        port=cfg.policy.actor_learner_config.learner_port,
    )

    logging.info("[ACTOR] Establishing connection with Learner")
    if not establish_learner_connection(learner_client, shutdown_event):
        logging.error("[ACTOR] Failed to establish connection with Learner")
        return

    if not use_threads(cfg):
        # If we use multithreading, we can reuse the channel
        grpc_channel.close()
        grpc_channel = None

    logging.info("[ACTOR] Connection with Learner established")

    is_threaded = use_threads(cfg)
    if is_threaded:
        parameters_queue = ThreadQueue(maxsize=4)
        # Each item can hold a chunk of large image transitions; keep this small to avoid OOM / jitter.
        transitions_queue = ThreadQueue(maxsize=4)
        interactions_queue = ThreadQueue(maxsize=32)
    else:
        parameters_queue = MpQueue(maxsize=4)
        transitions_queue = MpQueue(maxsize=4)
        interactions_queue = MpQueue(maxsize=32)

    concurrency_entity = None
    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from multiprocessing import Process

        concurrency_entity = Process

    receive_policy_process = concurrency_entity(
        target=receive_policy,
        args=(cfg, parameters_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process = concurrency_entity(
        target=send_transitions,
        args=(cfg, transitions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    interactions_process = concurrency_entity(
        target=send_interactions,
        args=(cfg, interactions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process.start()
    interactions_process.start()
    receive_policy_process.start()

    act_with_policy(
        cfg=cfg,
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        transitions_queue=transitions_queue,
        interactions_queue=interactions_queue,
    )
    logging.info("[ACTOR] Policy process joined")

    logging.info("[ACTOR] Closing queues")
    for q in (transitions_queue, interactions_queue, parameters_queue):
        close = getattr(q, "close", None)
        if callable(close):
            close()

    transitions_process.join()
    logging.info("[ACTOR] Transitions process joined")
    interactions_process.join()
    logging.info("[ACTOR] Interactions process joined")
    receive_policy_process.join()
    logging.info("[ACTOR] Receive policy process joined")

    logging.info("[ACTOR] join queues")
    for q in (transitions_queue, interactions_queue, parameters_queue):
        cancel_join_thread = getattr(q, "cancel_join_thread", None)
        if callable(cancel_join_thread):
            cancel_join_thread()

    logging.info("[ACTOR] queues closed")


# Core algorithm functions


def act_with_policy(
    cfg: TrainRLServerPipelineConfig,
    shutdown_event: any,  # Event,
    parameters_queue: any,
    transitions_queue: any,
    interactions_queue: any,
):
    """
    Executes policy interaction within the environment.

    This function rolls out the policy in the environment, collecting interaction data and pushing it to a queue for streaming to the learner.
    Once an episode is completed, updated network parameters received from the learner are retrieved from a queue and loaded into the network.

    Args:
        cfg: Configuration settings for the interaction process.
        shutdown_event: Event to check if the process should shutdown.
        parameters_queue: Queue to receive updated network parameters from the learner.
        transitions_queue: Queue to send transitions to the learner.
        interactions_queue: Queue to send interactions to the learner.
    """
    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_policy_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor policy process logging initialized")

    action_chunk_log_f = None
    last_logged_chunk_id: int | None = None
    if getattr(cfg.policy, "act_log_action_chunk", False):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        action_chunk_log_path = os.path.join(log_dir, f"act_action_chunks_{os.getpid()}.jsonl")
        action_chunk_log_f = open(action_chunk_log_path, "a", encoding="utf-8")
        logging.info(f"[ACTOR] ACT action-chunk logging enabled: {action_chunk_log_path}")

    logging.info("make_env online")

    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    try:
        action_min = None
        action_max = None
        if getattr(cfg.policy, "dataset_stats", None) is not None and isinstance(cfg.policy.dataset_stats, dict):
            action_stats = cfg.policy.dataset_stats.get("action", {})
            if isinstance(action_stats, dict):
                action_min = action_stats.get("min")
                action_max = action_stats.get("max")

        processor_device = cfg.policy.device
        keep_images_as_uint8 = False
        if getattr(cfg.env, "name", None) == "real_robot":
            # Keep all environment-side tensors on CPU and keep images as uint8 to reduce
            # control-loop jitter. We convert/move only the subset needed for policy inference.
            processor_device = "cpu"
            keep_images_as_uint8 = True

        env_processor, action_processor = make_processors(
            online_env,
            teleop_device,
            cfg.env,
            processor_device,
            action_min=action_min,
            action_max=action_max,
            keep_images_as_uint8=keep_images_as_uint8,
        )

        set_seed(cfg.seed)
        device = get_safe_torch_device(cfg.policy.device, log=True)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        logging.info("make_policy")

        ### Instantiate the policy in both the actor and learner processes
        ### To avoid sending a SACPolicy object through the port, we create a policy instance
        ### on both sides, the learner sends the updated parameters every n steps to update the actor's parameters
        policy: SACPolicy = make_policy(
            cfg=cfg.policy,
            env_cfg=cfg.env,
        )

        # Keep SAC(-ACT) inference consistent with the real-robot action clamp by propagating
        # `env.robot.max_relative_target` into the policy config (if not already set).
        try:
            env_robot = getattr(cfg.env, "robot", None)
            env_max_relative_target = (
                getattr(env_robot, "max_relative_target", None) if env_robot is not None else None
            )
            if (
                env_max_relative_target is not None
                and getattr(getattr(cfg.env, "processor", None), "inverse_kinematics", None) is None
                and getattr(getattr(policy, "config", None), "max_relative_target", None) is None
            ):
                policy.config.max_relative_target = float(env_max_relative_target)
                logging.info(
                    "[ACTOR] policy.max_relative_target set from env.robot.max_relative_target=%s",
                    policy.config.max_relative_target,
                )
        except Exception:
            logging.debug("[ACTOR] Failed to propagate env.robot.max_relative_target into policy config.")
        policy = policy.eval()
        assert isinstance(policy, nn.Module)

        # On real robots, avoid executing a random/uninitialized policy before the learner has
        # streamed its initial parameters.
        initial_params_timeout_s = 30.0 if getattr(cfg.env, "name", None) == "real_robot" else 5.0
        wait_for_initial_policy_parameters(
            policy=policy,
            parameters_queue=parameters_queue,
            device=device,
            timeout_s=initial_params_timeout_s,
        )

        obs, info = online_env.reset()
        env_processor.reset()
        action_processor.reset()
        policy.reset()

        # Process initial observation
        transition = create_transition(observation=obs, info=info)
        transition = env_processor(transition)

        # NOTE: For the moment we will solely handle the case of a single environment
        sum_reward_episode = 0
        pending_transitions: list[Transition] = []
        episode_intervention = False
        # Add counters for intervention rate calculation
        episode_intervention_steps = 0
        episode_total_steps = 0
        last_intervention_state = False
        prev_info = transition.get(TransitionKey.INFO, {})
        if isinstance(prev_info, dict):
            last_intervention_state = bool(
                prev_info.get(TeleopEvents.IS_INTERVENTION, False)
                or prev_info.get(TeleopEvents.IS_INTERVENTION.value, False)
            )

        policy_timer = TimerManager("Policy inference", log=False)
        last_policy_ms: float | None = None
        last_act_queue_len: int | None = None
        last_telemetry_log_s = time.monotonic()
        telemetry_interval_s = 1.0
        telemetry_prev_state_t: torch.Tensor | None = None
        telemetry_prev_robot_action_t: torch.Tensor | None = None
        action_dim = None
        if cfg.policy is not None:
            try:
                action_ft = cfg.policy.action_feature
                if action_ft is not None and action_ft.shape:
                    action_dim = int(action_ft.shape[0])
            except Exception:
                action_dim = None
        if action_dim is None:
            raise ValueError("Could not infer action dimension from policy config.")
        neutral_action = torch.zeros(action_dim, dtype=torch.float32)
        action_min_t = (
            torch.tensor(action_min, dtype=torch.float32, device="cpu")
            if action_min is not None and action_max is not None
            else None
        )
        action_max_t = (
            torch.tensor(action_max, dtype=torch.float32, device="cpu")
            if action_min is not None and action_max is not None
            else None
        )

        def _hold_action_from_current_state() -> torch.Tensor:
            """Return a safe 'do nothing' action in policy (normalized) space.

            - For joint-space absolute control (no IK) with known action bounds, this maps the current
              `observation.state` back into SAC's [-1, 1] action space so that unnormalization yields
              a true hold-position command.
            - Otherwise, falls back to zeros (appropriate for delta-action control).
            """
            if action_min_t is None or action_max_t is None:
                return neutral_action
            if cfg.env.processor.inverse_kinematics is not None:
                return neutral_action

            obs = transition.get(TransitionKey.OBSERVATION, {})
            if not isinstance(obs, dict):
                return neutral_action
            state = obs.get(OBS_STATE)
            if not isinstance(state, torch.Tensor):
                return neutral_action

            state_t = state
            if state_t.ndim == 1:
                state_t = state_t.unsqueeze(0)
            state_t = state_t.to(device="cpu", dtype=torch.float32)

            if state_t.shape[-1] != action_min_t.numel():
                return neutral_action

            denom = (action_max_t - action_min_t).clamp(min=1e-6)
            hold = 2.0 * (state_t - action_min_t) / denom - 1.0
            return hold.clamp(min=-1.0, max=1.0).squeeze(0)

        def _can_skip_observation_for_action_queue(policy: SACPolicy) -> bool:
            cfg_obj = getattr(policy, "config", None)
            if not getattr(cfg_obj, "act_use_action_queue", False):
                return False
            actor_obj = getattr(policy, "actor", None)
            act_policy = getattr(actor_obj, "act_policy", None)
            if act_policy is None:
                return False
            if getattr(getattr(act_policy, "config", None), "temporal_ensemble_coeff", None) is not None:
                return False
            q = getattr(act_policy, "_action_queue", None)
            try:
                return q is not None and len(q) > 0
            except Exception:
                return False

        def _prepare_observation_for_policy(observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            prepared: dict[str, torch.Tensor] = {}
            for key, value in observation.items():
                if not isinstance(value, torch.Tensor):
                    prepared[key] = value
                    continue
                if isinstance(key, str) and key.startswith(OBS_IMAGE) and value.dtype == torch.uint8:
                    value = value.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
                elif value.is_floating_point():
                    value = value.to(device=device, dtype=torch.float32, non_blocking=True)
                else:
                    value = value.to(device=device, non_blocking=True)
                prepared[key] = value
            return prepared

        for interaction_step in range(cfg.policy.online_steps):
            start_time = time.perf_counter()
            if shutdown_event.is_set():
                logging.info("[ACTOR] Shutting down act_with_policy")
                return

            observation = {
                k: v
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if k in cfg.policy.input_features
            }

            # If we are currently in a human intervention phase, prioritize control-loop responsiveness
            # by skipping policy inference (which can be slow) and letting the action processor override
            # the action with teleop commands.
            prev_info = transition.get(TransitionKey.INFO, {})
            is_prev_intervention = False
            if isinstance(prev_info, dict):
                is_prev_intervention = bool(
                    prev_info.get(TeleopEvents.IS_INTERVENTION, False)
                    or prev_info.get(TeleopEvents.IS_INTERVENTION.value, False)
                )

            policy_fps: float | None = None
            if is_prev_intervention:
                # Keep the robot stable even if the user just toggled intervention OFF.
                # There is a 1-step lag between reading teleop events (in the action processor) and
                # this loop's `is_prev_intervention` flag, so using a literal zero action here can
                # command an unintended absolute joint target.
                action = _hold_action_from_current_state()
            else:
                # Time policy inference and check if it meets FPS requirement
                with policy_timer:
                    if _can_skip_observation_for_action_queue(policy):
                        # Pass the (CPU) observation so the policy can use it for background ACT prefetch,
                        # while still avoiding expensive device transfers in the control loop.
                        action = policy.select_action(batch=observation)
                    else:
                        action = policy.select_action(batch=_prepare_observation_for_policy(observation))
                # Keep env-side control on CPU to avoid GPU sync/jitter.
                if isinstance(action, torch.Tensor) and action.device.type != "cpu":
                    action = action.detach().to("cpu")
                policy_fps = policy_timer.fps_last
                last_policy_ms = policy_timer.last * 1000.0
                try:
                    actor_obj = getattr(policy, "actor", None)
                    act_policy = getattr(actor_obj, "act_policy", None)
                    q = getattr(act_policy, "_action_queue", None)
                    last_act_queue_len = len(q) if q is not None else None
                except Exception:
                    last_act_queue_len = None

            if not is_prev_intervention:
                if action_chunk_log_f is not None:
                    act_policy = None
                    if hasattr(policy, "actor") and hasattr(getattr(policy, "actor"), "act_policy"):
                        act_policy = getattr(policy.actor, "act_policy")
                    elif hasattr(policy, "predict_action_chunk"):
                        act_policy = policy

                    chunk = getattr(act_policy, "_last_action_chunk", None) if act_policy is not None else None
                    chunk_id = getattr(act_policy, "_last_action_chunk_id", None) if act_policy is not None else None
                    if (
                        isinstance(chunk, torch.Tensor)
                        and isinstance(chunk_id, int)
                        and chunk_id != last_logged_chunk_id
                    ):
                        chunk_cpu = chunk.detach().to("cpu")
                        record = {
                            "time": time.time(),
                            "interaction_step": interaction_step,
                            "chunk_id": chunk_id,
                            "chunk_shape": list(chunk_cpu.shape),
                            "actions": chunk_cpu.tolist(),
                        }
                        action_chunk_log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        action_chunk_log_f.flush()
                        last_logged_chunk_id = chunk_id

                if policy_fps is not None:
                    log_policy_frequency_issue(
                        policy_fps=policy_fps, cfg=cfg, interaction_step=interaction_step
                    )

            try:
                new_transition = step_env_and_process_transition(
                    env=online_env,
                    transition=transition,
                    action=action,
                    env_processor=env_processor,
                    action_processor=action_processor,
                )
            except Exception:
                logging.exception("[ACTOR] Environment step failed; shutting down actor.")
                shutdown_event.set()
                return

            # Teleop action is the action that was executed in the environment
            # It is either the action from the teleop device or the action from the policy
            executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]
            executed_robot_action = new_transition.get(TransitionKey.ACTION)

            reward = new_transition[TransitionKey.REWARD]
            done = new_transition.get(TransitionKey.DONE, False)
            truncated = new_transition.get(TransitionKey.TRUNCATED, False)

            sum_reward_episode += float(reward)
            episode_total_steps += 1

            # Check for intervention from transition info
            intervention_info = new_transition[TransitionKey.INFO]
            if intervention_info.get(TeleopEvents.IS_INTERVENTION, False):
                episode_intervention = True
                episode_intervention_steps += 1

            complementary_info = {
                "discrete_penalty": torch.tensor(
                    [new_transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)]
                ),
                TeleopEvents.IS_INTERVENTION.value: bool(
                    new_transition[TransitionKey.INFO].get(TeleopEvents.IS_INTERVENTION, False)
                ),
            }
            # Create transition for learner (convert to old format)
            pending_transitions.append(
                Transition(
                    state=observation,
                    action=executed_action,
                    reward=reward,
                    # The replay buffer uses `optimize_memory=True` and derives next_state from the next transition.
                    # Avoid sending next_state over gRPC to drastically reduce memory / bandwidth usage.
                    next_state={},
                    done=done,
                    truncated=truncated,
                    complementary_info=complementary_info,
                )
            )

            # Update transition for next iteration
            transition = new_transition
            current_intervention_state = False
            current_info = transition.get(TransitionKey.INFO, {})
            if isinstance(current_info, dict):
                current_intervention_state = bool(
                    current_info.get(TeleopEvents.IS_INTERVENTION, False)
                    or current_info.get(TeleopEvents.IS_INTERVENTION.value, False)
                )
            if current_intervention_state != last_intervention_state:
                logging.info(
                    "[ACTOR] Human intervention toggled %s",
                    "ON" if current_intervention_state else "OFF",
                )
                # Clear ACT/policy internal queues so we don't execute stale actions across the boundary.
                try:
                    policy.reset()
                except Exception:
                    logging.exception("[ACTOR] Failed to reset policy on intervention toggle.")
                last_intervention_state = current_intervention_state

            # Lightweight telemetry to debug "robot hardly moves" reports.
            now_s = time.monotonic()
            interval_s = now_s - last_telemetry_log_s
            if interval_s >= telemetry_interval_s:
                last_telemetry_log_s = now_s
                info = new_transition.get(TransitionKey.INFO, {})
                action_source = None
                is_intervention = current_intervention_state
                if isinstance(info, dict):
                    action_source = info.get("_intervention_action_source")

                state = new_transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
                robot_action = executed_robot_action
                policy_action = executed_action

                def _as_1d(t: torch.Tensor | None) -> torch.Tensor | None:
                    if not isinstance(t, torch.Tensor):
                        return None
                    if t.ndim > 1 and t.shape[0] == 1:
                        t = t.squeeze(0)
                    return t.detach().to("cpu", dtype=torch.float32)

                state_t = _as_1d(state if isinstance(state, torch.Tensor) else None)
                robot_action_t = _as_1d(robot_action if isinstance(robot_action, torch.Tensor) else None)
                policy_action_t = _as_1d(policy_action if isinstance(policy_action, torch.Tensor) else None)

                delta_max = None
                if state_t is not None and robot_action_t is not None and state_t.shape == robot_action_t.shape:
                    delta_max = float((robot_action_t - state_t).abs().max().item())

                move_max = None
                if (
                    telemetry_prev_state_t is not None
                    and state_t is not None
                    and telemetry_prev_state_t.shape == state_t.shape
                ):
                    move_max = float((state_t - telemetry_prev_state_t).abs().max().item())
                telemetry_prev_state_t = state_t

                cmd_change_max = None
                if (
                    telemetry_prev_robot_action_t is not None
                    and robot_action_t is not None
                    and telemetry_prev_robot_action_t.shape == robot_action_t.shape
                ):
                    cmd_change_max = float((robot_action_t - telemetry_prev_robot_action_t).abs().max().item())
                telemetry_prev_robot_action_t = robot_action_t

                def _max_abs(t: torch.Tensor | None) -> float | None:
                    if t is None:
                        return None
                    return float(t.abs().max().item())

                def _fmt_vec(t: torch.Tensor | None, *, decimals: int = 1) -> str:
                    if t is None:
                        return "n/a"
                    vals = t.tolist()
                    if not isinstance(vals, list):
                        vals = [vals]
                    fmt = f"{{:.{decimals}f}}"
                    return "[" + ", ".join(fmt.format(float(v)) for v in vals) + "]"

                clamp_flag = None
                clamp_desired = None
                clamp_applied = None
                clamp_amount = None
                if isinstance(info, dict):
                    clamp_flag = info.get("_max_relative_target_clamped")
                    clamp_desired = info.get("_max_relative_target_desired_delta_max")
                    clamp_applied = info.get("_max_relative_target_applied_delta_max")
                    clamp_amount = info.get("_max_relative_target_clamp_amount_max")

                logging.info(
                    "[ACTOR] step=%d mode=%s src=%s |a|_norm=%s |a|_robot=%s |Δ|=%s move=%s cmdΔ=%s clamp=%s/%s/%s/%s state=%s cmd=%s act_q=%s policy_ms=%s",
                    interaction_step,
                    "HIL" if is_intervention else "POLICY",
                    action_source if action_source is not None else "n/a",
                    f"{_max_abs(policy_action_t):.3f}" if policy_action_t is not None else "n/a",
                    f"{_max_abs(robot_action_t):.1f}" if robot_action_t is not None else "n/a",
                    f"{delta_max:.1f}" if delta_max is not None else "n/a",
                    f"{move_max:.1f}" if move_max is not None else "n/a",
                    f"{cmd_change_max:.1f}" if cmd_change_max is not None else "n/a",
                    "Y" if clamp_flag else "N" if clamp_flag is not None else "n/a",
                    f"{float(clamp_desired):.1f}" if clamp_desired is not None else "n/a",
                    f"{float(clamp_applied):.1f}" if clamp_applied is not None else "n/a",
                    f"{float(clamp_amount):.1f}" if clamp_amount is not None else "n/a",
                    _fmt_vec(state_t, decimals=1),
                    _fmt_vec(robot_action_t, decimals=1),
                    last_act_queue_len if last_act_queue_len is not None else "n/a",
                    f"{last_policy_ms:.0f}" if last_policy_ms is not None else "n/a",
                )

            # Flush transitions frequently to avoid building huge in-memory buffers which can get the
            # process killed by the OOM killer when serializing large episodes with images.
            max_transitions_per_send = 10
            if not (done or truncated) and len(pending_transitions) >= max_transitions_per_send:
                push_transitions_to_transport_queue(
                    transitions=pending_transitions,
                    transitions_queue=transitions_queue,
                    serialize_before_put=not use_threads(cfg),
                )
                pending_transitions = []

            if done or truncated:
                pre_label_episode_reward = float(sum_reward_episode)
                logging.info(
                    "[ACTOR] Global step %d: Episode ended (done=%s truncated=%s). Pre-label episodic reward: %.6f",
                    interaction_step,
                    bool(done),
                    bool(truncated),
                    pre_label_episode_reward,
                )

                episode_success: bool | None = None
                if (
                    cfg.env.processor.reset is not None
                    and getattr(cfg.env.processor.reset, "prompt_success_failure", False)
                ):
                    # Keep the terminal transition local until the user provides the outcome label.
                    terminal_transition: Transition | None = None
                    if pending_transitions:
                        terminal_transition = pending_transitions.pop()

                    if pending_transitions:
                        push_transitions_to_transport_queue(
                            transitions=pending_transitions,
                            transitions_queue=transitions_queue,
                            serialize_before_put=not use_threads(cfg),
                        )
                        pending_transitions = []

                    try:
                        episode_success = prompt_episode_success_failure(teleop_device=teleop_device)
                    except KeyboardInterrupt:
                        logging.info("[ACTOR] Episode outcome prompt interrupted; shutting down actor.")
                        shutdown_event.set()
                        return

                    # Overwrite the terminal transition reward/done flags based on the label.
                    # This is useful when episodes end via time limits (truncation) and the
                    # outcome must be provided by the user.
                    if terminal_transition is not None:
                        previous_reward = float(terminal_transition["reward"])
                        terminal_transition["reward"] = float(episode_success)
                        terminal_transition["done"] = True
                        terminal_transition["truncated"] = False
                        pending_transitions.append(terminal_transition)
                        sum_reward_episode += float(episode_success) - previous_reward

                if episode_success is None:
                    logging.info(
                        "[ACTOR] Global step %d: Episodic reward (final): %.6f",
                        interaction_step,
                        float(sum_reward_episode),
                    )
                else:
                    logging.info(
                        "[ACTOR] Global step %d: Episode label=%d -> episodic reward (final): %.6f (was %.6f)",
                        interaction_step,
                        int(episode_success),
                        float(sum_reward_episode),
                        pre_label_episode_reward,
                    )

                update_policy_parameters(policy=policy, parameters_queue=parameters_queue, device=device)

                if pending_transitions:
                    push_transitions_to_transport_queue(
                        transitions=pending_transitions,
                        transitions_queue=transitions_queue,
                        serialize_before_put=not use_threads(cfg),
                    )
                    pending_transitions = []

                stats = get_frequency_stats(policy_timer)
                policy_timer.reset()

                # Calculate intervention rate
                intervention_rate = 0.0
                if episode_total_steps > 0:
                    intervention_rate = episode_intervention_steps / episode_total_steps

                # Send episodic reward to the learner
                interactions_queue.put(
                    python_object_to_bytes(
                        {
                            "Episodic reward": sum_reward_episode,
                            "Interaction step": interaction_step,
                            "Episode intervention": int(episode_intervention),
                            "Intervention rate": intervention_rate,
                            **(
                                {"Episode success": int(episode_success)}
                                if episode_success is not None
                                else {}
                            ),
                            **stats,
                        }
                    )
                )

                # Reset intervention counters and environment
                sum_reward_episode = 0.0
                episode_intervention = False
                episode_intervention_steps = 0
                episode_total_steps = 0

                # Reset environment and processors
                obs, info = online_env.reset()
                env_processor.reset()
                action_processor.reset()
                policy.reset()

                # Process initial observation
                transition = create_transition(observation=obs, info=info)
                transition = env_processor(transition)
                prev_info = transition.get(TransitionKey.INFO, {})
                last_intervention_state = False
                if isinstance(prev_info, dict):
                    last_intervention_state = bool(
                        prev_info.get(TeleopEvents.IS_INTERVENTION, False)
                        or prev_info.get(TeleopEvents.IS_INTERVENTION.value, False)
                    )

            if cfg.env.fps is not None:
                dt_time = time.perf_counter() - start_time
                target_dt = 1 / cfg.env.fps
                if not (done or truncated) and dt_time > 2 * target_dt:
                    now_s = time.monotonic()
                    last_slow_log_s = getattr(act_with_policy, "_last_slow_step_log_s", 0.0)
                    if now_s - last_slow_log_s >= 1.0:
                        qsize = None
                        try:
                            qsize = transitions_queue.qsize()
                        except Exception:
                            qsize = None
                        info = transition.get(TransitionKey.INFO, {})
                        timing = ""
                        if isinstance(info, dict) and "_timing_total_ms" in info:
                            timing = (
                                f" breakdown(action_proc={info.get('_timing_action_processor_ms', 0.0):.0f}ms,"
                                f" env_step={info.get('_timing_env_step_ms', 0.0):.0f}ms,"
                                f" env_proc={info.get('_timing_env_processor_ms', 0.0):.0f}ms)"
                            )
                        logging.warning(
                            "[ACTOR] Slow control step %.0f ms (target %.0f ms). transitions_queue=%s policy=%.0fms act_q=%s%s",
                            dt_time * 1000.0,
                            target_dt * 1000.0,
                            qsize if qsize is not None else "n/a",
                            last_policy_ms if last_policy_ms is not None else 0.0,
                            last_act_queue_len if last_act_queue_len is not None else "n/a",
                            timing,
                        )
                        setattr(act_with_policy, "_last_slow_step_log_s", now_s)
                precise_sleep(max(1 / cfg.env.fps - dt_time, 0.0))
    finally:
        if action_chunk_log_f is not None:
            try:
                action_chunk_log_f.close()
            except Exception:
                logging.exception("[ACTOR] Failed to close action chunk log file cleanly.")

        if teleop_device is not None:
            try:
                if getattr(teleop_device, "is_connected", False):
                    teleop_device.disconnect()
            except Exception:
                logging.exception("[ACTOR] Failed to disconnect teleop device cleanly.")

        try:
            if hasattr(online_env, "close"):
                online_env.close()
        except Exception:
            logging.exception("[ACTOR] Failed to close environment cleanly.")


#  Communication Functions - Group all gRPC/messaging functions


def establish_learner_connection(
    stub: services_pb2_grpc.LearnerServiceStub,
    shutdown_event: any,
    attempts: int = 30,
):
    """Establish a connection with the learner.

    Args:
        stub (services_pb2_grpc.LearnerServiceStub): The stub to use for the connection.
        shutdown_event (Event): The event to check if the connection should be established.
        attempts (int): The number of attempts to establish the connection.
    Returns:
        bool: True if the connection is established, False otherwise.
    """
    for _ in range(attempts):
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down establish_learner_connection")
            return False

        # Force a connection attempt and check state
        try:
            logging.info("[ACTOR] Send ready message to Learner")
            if stub.Ready(services_pb2.Empty()) == services_pb2.Empty():
                return True
        except grpc.RpcError as e:
            logging.error(f"[ACTOR] Waiting for Learner to be ready... {e}")
            time.sleep(2)
    return False


@lru_cache(maxsize=1)
def learner_service_client(
    host: str = "127.0.0.1",
    port: int = 50051,
) -> tuple[services_pb2_grpc.LearnerServiceStub, grpc.Channel]:
    """
    Returns a client for the learner service.

    GRPC uses HTTP/2, which is a binary protocol and multiplexes requests over a single connection.
    So we need to create only one client and reuse it.
    """

    channel = grpc.insecure_channel(
        f"{host}:{port}",
        grpc_channel_options(),
    )
    stub = services_pb2_grpc.LearnerServiceStub(channel)
    logging.info("[ACTOR] Learner service client created")
    return stub, channel


def receive_policy(
    cfg: TrainRLServerPipelineConfig,
    parameters_queue: any,
    shutdown_event: any,
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
):
    """Receive parameters from the learner.

    Args:
        cfg (TrainRLServerPipelineConfig): The configuration for the actor.
        parameters_queue (Queue): The queue to receive the parameters.
        shutdown_event (Event): The event to check if the process should shutdown.
    """
    logging.info("[ACTOR] Start receiving parameters from the Learner")
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_receive_policy_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor receive policy process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        iterator = learner_client.StreamParameters(services_pb2.Empty())
        receive_bytes_in_chunks(
            iterator,
            parameters_queue,
            shutdown_event,
            log_prefix="[ACTOR] parameters",
        )

    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Received policy loop stopped")


def send_transitions(
    cfg: TrainRLServerPipelineConfig,
    transitions_queue: any,
    shutdown_event: any,
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    Sends transitions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - Transition Data:
        - A batch of transitions (observation, action, reward, next observation) is collected.
        - Transitions are moved to the CPU and serialized using PyTorch.
        - The serialized data is wrapped in a `services_pb2.Transition` message and sent to the learner.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_transitions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor transitions process logging initialized")

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendTransitions(
            transitions_stream(
                shutdown_event, transitions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming transitions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Transitions process stopped")


def send_interactions(
    cfg: TrainRLServerPipelineConfig,
    interactions_queue: any,
    shutdown_event: any,
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    Sends interactions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - Interaction Messages:
        - Contains useful statistics about episodic rewards and policy timings.
        - The message is serialized using `pickle` and sent to the learner.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_interactions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor interactions process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendInteractions(
            interactions_stream(
                shutdown_event, interactions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming interactions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Interactions process stopped")


def _compress_image_tensor_to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.is_floating_point():
        return (tensor * 255.0).round().clamp(0, 255).to(torch.uint8)
    return tensor.to(torch.uint8)


def _serialize_transitions_chunk(transitions: list[Transition]) -> bytes:
    """Convert a chunk of transitions to bytes, compressing image tensors to uint8 first.

    Note: We compress images *before* moving to CPU so that CUDA->CPU transfers are smaller.
    """
    transition_to_send: list[Transition] = []

    for transition in transitions:
        tr: Transition = transition

        # Compress observation images.
        state = tr.get("state", {})
        if isinstance(state, dict) and state:
            new_state = dict(state)
            for key, value in state.items():
                if isinstance(key, str) and key.startswith(OBS_IMAGE) and isinstance(value, torch.Tensor):
                    new_state[key] = _compress_image_tensor_to_uint8(value)
            tr = dict(tr)
            tr["state"] = new_state

        # Compress next_state images if present.
        next_state = tr.get("next_state", {})
        if isinstance(next_state, dict) and next_state:
            new_next_state = dict(next_state)
            for key, value in next_state.items():
                if isinstance(key, str) and key.startswith(OBS_IMAGE) and isinstance(value, torch.Tensor):
                    new_next_state[key] = _compress_image_tensor_to_uint8(value)
            tr = dict(tr)
            tr["next_state"] = new_next_state

        tr_cpu = move_transition_to_device(transition=tr, device="cpu")
        transition_to_send.append(tr_cpu)

    return transitions_to_bytes(transition_to_send)


def transitions_stream(shutdown_event: any, transitions_queue: any, timeout: float) -> services_pb2.Empty:  # type: ignore
    while not shutdown_event.is_set():
        try:
            message = transitions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Transition queue is empty")
            continue

        try:
            if isinstance(message, (list, tuple)):
                message = _serialize_transitions_chunk(list(message))
            if not isinstance(message, (bytes, bytearray)):
                logging.warning(
                    "[ACTOR] Unexpected transitions queue message type: %s", type(message).__name__
                )
                continue
        except Exception:  # noqa: BLE001
            logging.exception("[ACTOR] Failed to serialize transitions message; dropping it.")
            continue

        yield from send_bytes_in_chunks(
            bytes(message), services_pb2.Transition, log_prefix="[ACTOR] Send transitions"
        )

    return services_pb2.Empty()


def interactions_stream(
    shutdown_event: any,
    interactions_queue: any,
    timeout: float,  # type: ignore
) -> services_pb2.Empty:
    while not shutdown_event.is_set():
        try:
            message = interactions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Interaction queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message,
            services_pb2.InteractionMessage,
            log_prefix="[ACTOR] Send interactions",
        )

    return services_pb2.Empty()


#  Policy functions


def update_policy_parameters(policy: SACPolicy, parameters_queue: any, device):
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
    if bytes_state_dict is not None:
        _load_policy_parameters_from_bytes(policy=policy, bytes_state_dict=bytes_state_dict, device=device)


def _load_policy_parameters_from_bytes(policy: SACPolicy, bytes_state_dict: bytes, device) -> None:
    logging.info("[ACTOR] Load new parameters from Learner.")
    state_dicts = bytes_to_state_dict(bytes_state_dict)

    # TODO: check encoder parameter synchronization possible issues:
    # 1. When shared_encoder=True, we're loading stale encoder params from actor's state_dict
    #    instead of the updated encoder params from critic (which is optimized separately)
    # 2. When freeze_vision_encoder=True, we waste bandwidth sending/loading frozen params
    # 3. Need to handle encoder params correctly for both actor and discrete_critic
    # Potential fixes:
    # - Send critic's encoder state when shared_encoder=True
    # - Skip encoder params entirely when freeze_vision_encoder=True
    # - Ensure discrete_critic gets correct encoder state (currently uses encoder_critic)

    # Load actor state dict
    actor_state_dict = move_state_dict_to_device(state_dicts["policy"], device=device)
    policy.actor.load_state_dict(actor_state_dict)

    # Load discrete critic if present
    if hasattr(policy, "discrete_critic") and "discrete_critic" in state_dicts:
        discrete_critic_state_dict = move_state_dict_to_device(state_dicts["discrete_critic"], device=device)
        policy.discrete_critic.load_state_dict(discrete_critic_state_dict)
        logging.info("[ACTOR] Loaded discrete critic parameters from Learner.")


def wait_for_initial_policy_parameters(
    policy: SACPolicy,
    parameters_queue: any,
    *,
    device,
    timeout_s: float = 30.0,
) -> None:
    """Best-effort wait for the first learner parameters before moving the robot."""
    deadline_s = time.monotonic() + max(timeout_s, 0.0)
    while time.monotonic() < deadline_s:
        remaining = max(0.0, deadline_s - time.monotonic())
        bytes_state_dict = get_last_item_from_queue(
            parameters_queue, block=True, timeout=min(0.5, remaining)
        )
        if bytes_state_dict is None:
            continue
        _load_policy_parameters_from_bytes(policy=policy, bytes_state_dict=bytes_state_dict, device=device)
        return

    logging.warning(
        "[ACTOR] Timed out waiting for initial parameters from Learner (%.1fs); "
        "continuing with local policy weights.",
        timeout_s,
    )


#  Utilities functions


def push_transitions_to_transport_queue(
    transitions: list[Transition],
    transitions_queue: any,
    *,
    max_transitions_per_message: int = 10,
    serialize_before_put: bool = True,
) -> None:
    """Enqueue transitions to be sent to the learner.

    When `serialize_before_put=True` (default), transitions are serialized to bytes in this
    function. When False, raw transition chunks are enqueued and serialization is done in
    the transitions sender thread/process.
    """
    if not transitions:
        return

    now_s = time.monotonic()
    last_log_s = getattr(push_transitions_to_transport_queue, "_last_full_log_s", 0.0)

    for start in range(0, len(transitions), max_transitions_per_message):
        chunk = transitions[start : start + max_transitions_per_message]
        try:
            payload = _serialize_transitions_chunk(chunk) if serialize_before_put else chunk
            transitions_queue.put(payload, block=False)
        except Full:
            if now_s - last_log_s >= 1.0:
                logging.warning(
                    "[ACTOR] Transitions queue full; dropping %d transitions.", len(chunk)
                )
                last_log_s = now_s
                setattr(push_transitions_to_transport_queue, "_last_full_log_s", last_log_s)
        except Exception:  # noqa: BLE001
            logging.exception("[ACTOR] Failed to enqueue transitions; dropping them.")


def get_frequency_stats(timer: TimerManager) -> dict[str, float]:
    """Get the frequency statistics of the policy.

    Args:
        timer (TimerManager): The timer with collected metrics.

    Returns:
        dict[str, float]: The frequency statistics of the policy.
    """
    stats = {}
    if timer.count > 1:
        avg_fps = timer.fps_avg
        p90_fps = timer.fps_percentile(90)
        logging.debug(f"[ACTOR] Average policy frame rate: {avg_fps}")
        logging.debug(f"[ACTOR] Policy frame rate 90th percentile: {p90_fps}")
        stats = {
            "Policy frequency [Hz]": avg_fps,
            "Policy frequency 90th-p [Hz]": p90_fps,
        }
    return stats


def log_policy_frequency_issue(policy_fps: float, cfg: TrainRLServerPipelineConfig, interaction_step: int):
    if policy_fps < cfg.env.fps:
        logging.warning(
            f"[ACTOR] Policy FPS {policy_fps:.1f} below required {cfg.env.fps} at step {interaction_step}"
        )


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.actor == "threads"


def _read_single_keypress() -> str:
    """Read a single keypress from stdin (best-effort, cross-platform)."""
    if os.name == "nt":
        import msvcrt  # noqa: PLC0415

        ch = msvcrt.getch()
        # Handle special keys (arrows, function keys) which come as a prefix byte.
        if ch in (b"\x00", b"\xe0"):
            ch = msvcrt.getch()
        return ch.decode(errors="ignore")

    import termios  # noqa: PLC0415
    import tty  # noqa: PLC0415

    fd = sys.stdin.fileno()
    if not sys.stdin.isatty():
        return sys.stdin.read(1)

    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _flush_teleop_events(teleop_device) -> None:
    getter = getattr(teleop_device, "get_teleop_events", None)
    if not callable(getter):
        return
    try:
        # Consume any queued key events so they don't leak into the next episode.
        getter()
    except Exception:
        logging.debug("[ACTOR] Failed to flush teleop events.", exc_info=True)


def prompt_episode_success_failure(teleop_device) -> bool:
    """Block until the user labels the episode as success ('s') or failure (Esc)."""
    print("\nEpisode finished. Press 's' for SUCCESS, or Esc for FAILURE.", flush=True)

    while True:
        if sys.stdin.isatty():
            key = _read_single_keypress()
        else:
            key = input("Type 's' for success or 'esc' for failure: ").strip()

        if key in {"s", "S"}:
            _flush_teleop_events(teleop_device)
            print("Marked as SUCCESS.", flush=True)
            return True
        if key == "\x1b" or str(key).strip().lower() in {"esc", "escape"}:
            _flush_teleop_events(teleop_device)
            print("Marked as FAILURE.", flush=True)
            return False

        print("Invalid input. Press 's' or Esc.", flush=True)


if __name__ == "__main__":
    actor_cli()
