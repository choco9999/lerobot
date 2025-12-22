#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
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
import math
import time
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents

from .core import EnvTransition, PolicyAction, TransitionKey
from .pipeline import (
    ComplementaryDataProcessorStep,
    InfoProcessorStep,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    TruncatedProcessorStep,
)

GRIPPER_KEY = "gripper"
DISCRETE_PENALTY_KEY = "discrete_penalty"
TELEOP_ACTION_KEY = "teleop_action"

_MINMAX_EPS = 1e-8
logger = logging.getLogger(__name__)


def _squeeze_batch_dim_if_present(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim > 1 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def _minmax_normalize(x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    denom = max_val - min_val
    denom = torch.where(denom == 0, torch.full_like(denom, _MINMAX_EPS), denom)
    return 2 * (x - min_val) / denom - 1


def _minmax_unnormalize(x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    denom = max_val - min_val
    denom = torch.where(denom == 0, torch.full_like(denom, _MINMAX_EPS), denom)
    return (x + 1) / 2 * denom + min_val


@runtime_checkable
class HasTeleopEvents(Protocol):
    """
    Minimal protocol for objects that provide teleoperation events.

    This protocol defines the `get_teleop_events()` method, allowing processor
    steps to interact with teleoperators that support event-based controls
    (like episode termination or success flagging) without needing to know the
    teleoperator's specific class.
    """

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the teleoperator.

        Returns:
            A dictionary containing control events such as:
            - `is_intervention`: bool - Whether the human is currently intervening.
            - `terminate_episode`: bool - Whether to terminate the current episode.
            - `success`: bool - Whether the episode was successful.
            - `rerecord_episode`: bool - Whether to rerecord the episode.
        """
        ...


# Type variable constrained to Teleoperator subclasses that also implement events
TeleopWithEvents = TypeVar("TeleopWithEvents", bound=Teleoperator)


def _check_teleop_with_events(teleop: Teleoperator) -> None:
    """
    Runtime check that a teleoperator implements the `HasTeleopEvents` protocol.

    Args:
        teleop: The teleoperator instance to check.

    Raises:
        TypeError: If the teleoperator does not have a `get_teleop_events` method.
    """
    if not isinstance(teleop, HasTeleopEvents):
        raise TypeError(
            f"Teleoperator {type(teleop).__name__} must implement get_teleop_events() method. "
            f"Compatible teleoperators: GamepadTeleop, KeyboardEndEffectorTeleop"
        )


@ProcessorStepRegistry.register("add_teleop_action_as_complementary_data")
@dataclass
class AddTeleopActionAsComplimentaryDataStep(ComplementaryDataProcessorStep):
    """
    Adds the raw action from a teleoperator to the transition's complementary data.

    This is useful for human-in-the-loop scenarios where the human's input needs to
    be available to downstream processors, for example, to override a policy's action
    during an intervention.

    Attributes:
        teleop_device: The teleoperator instance to get the action from.
    """

    teleop_device: Teleoperator

    def complementary_data(self, complementary_data: dict) -> dict:
        """
        Retrieves the teleoperator's action and adds it to the complementary data.

        Args:
            complementary_data: The incoming complementary data dictionary.

        Returns:
            A new dictionary with the teleoperator action added under the
            `teleop_action` key.
        """
        new_complementary_data = dict(complementary_data)
        info = {}
        if hasattr(self, "_current_transition"):
            maybe_info = self._current_transition.get(TransitionKey.INFO, {})
            if isinstance(maybe_info, dict):
                info = maybe_info

        has_intervention_flag = (
            TeleopEvents.IS_INTERVENTION in info or TeleopEvents.IS_INTERVENTION.value in info
        )
        is_intervention = True
        if has_intervention_flag:
            is_intervention = bool(
                info.get(TeleopEvents.IS_INTERVENTION, False)
                or info.get(TeleopEvents.IS_INTERVENTION.value, False)
            )

        if not is_intervention:
            new_complementary_data[TELEOP_ACTION_KEY] = None
            return new_complementary_data

        try:
            new_complementary_data[TELEOP_ACTION_KEY] = self.teleop_device.get_action()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to read teleop action (%s); intervention will fall back to holding the current pose.",
                e,
            )
            new_complementary_data[TELEOP_ACTION_KEY] = None
        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("add_teleop_action_as_info")
@dataclass
class AddTeleopEventsAsInfoStep(InfoProcessorStep):
    """
    Adds teleoperator control events (e.g., terminate, success) to the transition's info.

    This step extracts control events from teleoperators that support event-based
    interaction, making these signals available to other parts of the system.

    Attributes:
        teleop_device: An instance of a teleoperator that implements the
                       `HasTeleopEvents` protocol.
    """

    teleop_device: TeleopWithEvents

    def __post_init__(self):
        """Validates that the provided teleoperator supports events after initialization."""
        _check_teleop_with_events(self.teleop_device)

    def info(self, info: dict) -> dict:
        """
        Retrieves teleoperator events and updates the info dictionary.

        Args:
            info: The incoming info dictionary.

        Returns:
            A new dictionary including the teleoperator events.
        """
        new_info = dict(info)

        teleop_events = self.teleop_device.get_teleop_events()
        new_info.update(teleop_events)
        return new_info

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("image_crop_resize_processor")
@dataclass
class ImageCropResizeProcessorStep(ObservationProcessorStep):
    """
    Crops and/or resizes image observations.

    This step iterates through all image keys in an observation dictionary and applies
    the specified transformations. It handles device placement, moving tensors to the
    CPU if necessary for operations not supported on certain accelerators like MPS.

    Attributes:
        crop_params_dict: A dictionary mapping image keys to cropping parameters
                          (top, left, height, width).
        resize_size: A tuple (height, width) to resize all images to.
    """

    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None

    def observation(self, observation: dict) -> dict:
        """
        Applies cropping and resizing to all images in the observation dictionary.

        Args:
            observation: The observation dictionary, potentially containing image tensors.

        Returns:
            A new observation dictionary with transformed images.
        """
        if self.resize_size is None and not self.crop_params_dict:
            return observation

        new_observation = dict(observation)

        # Process all image keys in the observation
        for key in observation:
            if "image" not in key:
                continue

            image = observation[key]
            device = image.device
            # NOTE (maractingi): No mps kernel for crop and resize, so we need to move to cpu
            if device.type == "mps":
                image = image.cpu()
            # Crop if crop params are provided for this key
            if self.crop_params_dict is not None and key in self.crop_params_dict:
                crop_params = self.crop_params_dict[key]
                image = F.crop(image, *crop_params)
            if self.resize_size is not None:
                image = F.resize(image, self.resize_size)
                image = image.clamp(0.0, 1.0)
            new_observation[key] = image.to(device)

        return new_observation

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary with the crop parameters and resize dimensions.
        """
        return {
            "crop_params_dict": self.crop_params_dict,
            "resize_size": self.resize_size,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the image feature shapes in the policy features dictionary if resizing is applied.

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary with new image shapes.
        """
        if self.resize_size is None:
            return features
        for key in features[PipelineFeatureType.OBSERVATION]:
            if "image" in key:
                nb_channel = features[PipelineFeatureType.OBSERVATION][key].shape[0]
                features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                    type=features[PipelineFeatureType.OBSERVATION][key].type,
                    shape=(nb_channel, *self.resize_size),
                )
        return features


@dataclass
@ProcessorStepRegistry.register("time_limit_processor")
class TimeLimitProcessorStep(TruncatedProcessorStep):
    """
    Tracks episode steps and enforces a time limit by truncating the episode.

    Attributes:
        max_episode_steps: The maximum number of steps allowed per episode.
        current_step: The current step count for the active episode.
    """

    max_episode_steps: int
    current_step: int = 0

    def truncated(self, truncated: bool) -> bool:
        """
        Increments the step counter and sets the truncated flag if the time limit is reached.

        Args:
            truncated: The incoming truncated flag.

        Returns:
            True if the episode step limit is reached, otherwise the incoming value.
        """
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
        # TODO (steven): missing an else truncated = False?
        return truncated

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the `max_episode_steps`.
        """
        return {
            "max_episode_steps": self.max_episode_steps,
        }

    def reset(self) -> None:
        """Resets the step counter, typically called at the start of a new episode."""
        self.current_step = 0

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("gripper_penalty_processor")
class GripperPenaltyProcessorStep(ComplementaryDataProcessorStep):
    """
    Applies a penalty for inefficient gripper usage.

    This step penalizes actions that attempt to close an already closed gripper or
    open an already open one, based on position thresholds.

    Attributes:
        penalty: The negative reward value to apply.
        max_gripper_pos: The maximum position value for the gripper, used for normalization.
    """

    penalty: float = -0.01
    max_gripper_pos: float = 30.0

    def complementary_data(self, complementary_data: dict) -> dict:
        """
        Calculates the gripper penalty and adds it to the complementary data.

        Args:
            complementary_data: The incoming complementary data, which should contain
                                raw joint positions.

        Returns:
            A new complementary data dictionary with the `discrete_penalty` key added.
        """
        action = self.transition.get(TransitionKey.ACTION)

        raw_joint_positions = complementary_data.get("raw_joint_positions")
        if raw_joint_positions is None:
            return complementary_data

        current_gripper_pos = raw_joint_positions.get(GRIPPER_KEY, None)
        if current_gripper_pos is None:
            return complementary_data

        # Gripper action is a PolicyAction at this stage
        gripper_action = action[-1].item()
        gripper_action_normalized = gripper_action / self.max_gripper_pos

        # Normalize gripper state and action
        gripper_state_normalized = current_gripper_pos / self.max_gripper_pos

        # Calculate penalty boolean as in original
        gripper_penalty_bool = (gripper_state_normalized < 0.5 and gripper_action_normalized > 0.5) or (
            gripper_state_normalized > 0.75 and gripper_action_normalized < 0.5
        )

        gripper_penalty = self.penalty * int(gripper_penalty_bool)

        # Create new complementary data with penalty info
        new_complementary_data = dict(complementary_data)
        new_complementary_data[DISCRETE_PENALTY_KEY] = gripper_penalty

        return new_complementary_data

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the penalty value and max gripper position.
        """
        return {
            "penalty": self.penalty,
            "max_gripper_pos": self.max_gripper_pos,
        }

    def reset(self) -> None:
        """Resets the processor's internal state."""
        pass

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("intervention_action_processor")
class InterventionActionProcessorStep(ProcessorStep):
    """
    Handles human intervention, overriding policy actions and managing episode termination.

    When an intervention is detected (via teleoperator events in the `info` dict),
    this step replaces the policy's action with the human's teleoperated action.
    It also processes signals to terminate the episode or flag success.

    Attributes:
        use_gripper: Whether to include the gripper in the teleoperated action.
        terminate_on_success: If True, automatically sets the `done` flag when a
                              `success` event is received.
    """

    use_gripper: bool = False
    terminate_on_success: bool = True
    action_min: list[float] | None = None
    action_max: list[float] | None = None
    normalize_teleop_action: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Processes the transition to handle interventions.

        Args:
            transition: The incoming environment transition.

        Returns:
            The modified transition, potentially with an overridden action, updated
            reward, and termination status.
        """
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        # Get intervention signals from complementary data
        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        teleop_action = complementary_data.get(TELEOP_ACTION_KEY)
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)
        terminate_episode = info.get(TeleopEvents.TERMINATE_EPISODE, False)
        success = info.get(TeleopEvents.SUCCESS, False)
        rerecord_episode = info.get(TeleopEvents.RERECORD_EPISODE, False)

        new_transition = transition.copy()

        # Override action if intervention is active. If the teleop action cannot be read, fall back
        # to holding the current pose (derived from the raw joint positions injected into the
        # transition observation by the control loop).
        action_source = "policy"
        if is_intervention:
            if teleop_action is None:
                obs = transition.get(TransitionKey.OBSERVATION, {})
                if isinstance(obs, dict):
                    hold_action = {
                        k: v for k, v in obs.items() if isinstance(k, str) and k.endswith(".pos")
                    }
                    teleop_action = hold_action if hold_action else None
                action_source = "hold" if teleop_action is not None else "none"
            else:
                action_source = "teleop"

        if is_intervention and teleop_action is not None:
            teleop_action_tensor: torch.Tensor
            if isinstance(teleop_action, torch.Tensor):
                teleop_action_tensor = teleop_action.to(device=action.device, dtype=action.dtype)
            elif isinstance(teleop_action, np.ndarray):
                teleop_action_tensor = torch.from_numpy(teleop_action).to(
                    device=action.device, dtype=action.dtype
                )
            elif isinstance(teleop_action, dict):
                if {"delta_x", "delta_y", "delta_z"}.issubset(teleop_action):
                    # End-effector delta control.
                    action_list = [
                        teleop_action.get("delta_x", 0.0),
                        teleop_action.get("delta_y", 0.0),
                        teleop_action.get("delta_z", 0.0),
                    ]
                    if self.use_gripper:
                        action_list.append(teleop_action.get(GRIPPER_KEY, 1.0))
                else:
                    # Joint-space control: expect keys like "{joint}.pos".
                    obs = transition.get(TransitionKey.OBSERVATION, {})
                    obs_keys = []
                    if isinstance(obs, dict):
                        obs_keys = [
                            k
                            for k in obs
                            if isinstance(k, str) and k.endswith(".pos") and k in teleop_action
                        ]

                    joint_keys = obs_keys if obs_keys else sorted(
                        [k for k in teleop_action if isinstance(k, str) and k.endswith(".pos")]
                    )
                    action_list = [teleop_action[k] for k in joint_keys]

                teleop_action_tensor = torch.tensor(action_list, dtype=action.dtype, device=action.device)
            else:
                teleop_action_tensor = torch.tensor(teleop_action, dtype=action.dtype, device=action.device)

            teleop_action_tensor = _squeeze_batch_dim_if_present(teleop_action_tensor)

            if self.normalize_teleop_action and self.action_min is not None and self.action_max is not None:
                action_min = torch.tensor(self.action_min, dtype=action.dtype, device=action.device)
                action_max = torch.tensor(self.action_max, dtype=action.dtype, device=action.device)
                if teleop_action_tensor.numel() != action_min.numel():
                    raise ValueError(
                        f"Teleop action dim mismatch: got {teleop_action_tensor.numel()} values, "
                        f"but action_min/action_max have {action_min.numel()} values."
                    )
                teleop_action_tensor = _minmax_normalize(teleop_action_tensor, action_min, action_max).clamp(
                    -1.0, 1.0
                )

            new_transition[TransitionKey.ACTION] = teleop_action_tensor

        # Handle episode termination
        new_transition[TransitionKey.DONE] = bool(terminate_episode) or (
            self.terminate_on_success and success
        )
        new_transition[TransitionKey.REWARD] = float(success)

        # Update info with intervention metadata
        info = new_transition.get(TransitionKey.INFO, {})
        if isinstance(info, dict):
            info["_intervention_action_source"] = action_source
        info[TeleopEvents.IS_INTERVENTION] = is_intervention
        info[TeleopEvents.RERECORD_EPISODE] = rerecord_episode
        info[TeleopEvents.SUCCESS] = success
        new_transition[TransitionKey.INFO] = info

        # Update complementary data with teleop action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data[TELEOP_ACTION_KEY] = new_transition.get(TransitionKey.ACTION)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the step's configuration attributes.
        """
        return {
            "use_gripper": self.use_gripper,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("reward_classifier_processor")
class RewardClassifierProcessorStep(ProcessorStep):
    """
    Applies a pretrained reward classifier to image observations to predict success.

    This step uses a model to determine if the current state is successful, updating
    the reward and potentially terminating the episode.

    Attributes:
        pretrained_path: Path to the pretrained reward classifier model.
        device: The device to run the classifier on.
        success_threshold: The probability threshold to consider a prediction as successful.
        success_reward: The reward value to assign on success.
        terminate_on_success: If True, terminates the episode upon successful classification.
        reward_classifier: The loaded classifier model instance.
    """

    pretrained_path: str | None = None
    device: str = "cpu"
    success_threshold: float = 0.5
    success_reward: float = 1.0
    terminate_on_success: bool = True

    reward_classifier: Any = None

    def __post_init__(self):
        """Initializes the reward classifier model after the dataclass is created."""
        if self.pretrained_path is not None:
            from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

            self.reward_classifier = Classifier.from_pretrained(self.pretrained_path)
            self.reward_classifier.to(self.device)
            self.reward_classifier.eval()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Processes a transition, applying the reward classifier to its image observations.

        Args:
            transition: The incoming environment transition.

        Returns:
            The modified transition with an updated reward and done flag based on the
            classifier's prediction.
        """
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or self.reward_classifier is None:
            return new_transition

        # Extract images from observation
        images = {key: value for key, value in observation.items() if "image" in key}

        if not images:
            return new_transition

        # Run reward classifier
        start_time = time.perf_counter()
        with torch.inference_mode():
            success = self.reward_classifier.predict_reward(images, threshold=self.success_threshold)

        classifier_frequency = 1 / (time.perf_counter() - start_time)

        # Calculate reward and termination
        reward = new_transition.get(TransitionKey.REWARD, 0.0)
        terminated = new_transition.get(TransitionKey.DONE, False)

        if math.isclose(success, 1, abs_tol=1e-2):
            reward = self.success_reward
            if self.terminate_on_success:
                terminated = True

        # Update transition
        new_transition[TransitionKey.REWARD] = reward
        new_transition[TransitionKey.DONE] = terminated

        # Update info with classifier frequency
        info = new_transition.get(TransitionKey.INFO, {})
        info["reward_classifier_frequency"] = classifier_frequency
        new_transition[TransitionKey.INFO] = info

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the step's configuration attributes.
        """
        return {
            "device": self.device,
            "success_threshold": self.success_threshold,
            "success_reward": self.success_reward,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("finalize_hil_action_processor")
class FinalizeHILActionProcessorStep(ProcessorStep):
    """
    Finalize the action field for env stepping and training.

    - Squeezes a singleton batch dimension from the action if present.
    - Sets `complementary_data['teleop_action']` to the (possibly overridden) action tensor.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            action = _squeeze_batch_dim_if_present(action)
            new_transition[TransitionKey.ACTION] = action

        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        complementary_data[TELEOP_ACTION_KEY] = action
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("minmax_unnormalize_action_processor")
class MinMaxUnnormalizeActionProcessorStep(ProcessorStep):
    """Unnormalize a policy action from [-1, 1] to [min, max] using per-dimension bounds."""

    action_min: list[float] | None = None
    action_max: list[float] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        if self.action_min is None or self.action_max is None:
            return new_transition

        action = new_transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            return new_transition

        action = _squeeze_batch_dim_if_present(action)

        action_min = torch.tensor(self.action_min, dtype=action.dtype, device=action.device)
        action_max = torch.tensor(self.action_max, dtype=action.dtype, device=action.device)
        if action.numel() != action_min.numel():
            raise ValueError(
                f"Action dim mismatch: got {action.numel()} values, but action_min/action_max have {action_min.numel()} values."
            )

        new_transition[TransitionKey.ACTION] = _minmax_unnormalize(action, action_min, action_max)
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
