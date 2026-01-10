#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
# All rights reserved.
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
import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from typing import Literal

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.distributions import MultivariateNormal, TanhTransform, Transform, TransformedDistribution

from lerobot.configs.types import NormalizationMode
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.configuration_sac import SACActConfig, SACConfig, is_image_feature
from lerobot.policies.utils import get_device_from_parameters
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import (
    ACTION,
    OBS_ENV_STATE,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

DISCRETE_DIMENSION_INDEX = -1  # Gripper is always the last dimension


class SACPolicy(
    PreTrainedPolicy,
):
    config_class = SACConfig
    name = "sac"

    def __init__(
        self,
        config: SACConfig | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Determine action dimension and initialize all components
        continuous_action_dim = config.output_features[ACTION].shape[0]
        self._init_encoders()
        self._init_critics(continuous_action_dim)
        self._init_actor(continuous_action_dim)
        self._init_temperature()

    def get_optim_params(self) -> dict:
        optim_params = {
            "actor": [
                p
                for n, p in self.actor.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
            "critic": self.critic_ensemble.parameters(),
            "temperature": self.log_alpha,
        }
        if self.config.num_discrete_actions is not None:
            optim_params["discrete_critic"] = self.discrete_critic.parameters()
        return optim_params

    def reset(self):
        """Reset the policy"""
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("SACPolicy does not support action chunking. It returns single actions!")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""

        observations_features = None
        if self.shared_encoder and self.actor.encoder.has_images:
            observations_features = self.actor.encoder.get_cached_image_features(batch)

        actions, _, _ = self.actor(batch, observations_features)
        actions, _ = self._project_actions_to_max_relative_target(observations=batch, actions=actions)

        if self.config.num_discrete_actions is not None:
            discrete_action_value = self.discrete_critic(batch, observations_features)
            discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)
            actions = torch.cat([actions, discrete_action], dim=-1)

        return actions

    def critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through a critic network ensemble

        Args:
            observations: Dictionary of observations
            actions: Action tensor
            use_target: If True, use target critics, otherwise use ensemble critics

        Returns:
            Tensor of Q-values from all critics
        """

        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = critics(observations, actions, observation_features)
        return q_values

    def discrete_critic_forward(
        self, observations, use_target=False, observation_features=None
    ) -> torch.Tensor:
        """Forward pass through a discrete critic network

        Args:
            observations: Dictionary of observations
            use_target: If True, use target critics, otherwise use ensemble critics
            observation_features: Optional pre-computed observation features to avoid recomputing encoder output

        Returns:
            Tensor of Q-values from the discrete critic network
        """
        discrete_critic = self.discrete_critic_target if use_target else self.discrete_critic
        q_values = discrete_critic(observations, observation_features)
        return q_values

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "critic", "temperature", "discrete_critic"] = "critic",
    ) -> dict[str, Tensor]:
        """Compute the loss for the given model

        Args:
            batch: Dictionary containing:
                - action: Action tensor
                - reward: Reward tensor
                - state: Observations tensor dict
                - next_state: Next observations tensor dict
                - done: Done mask tensor
                - observation_feature: Optional pre-computed observation features
                - next_observation_feature: Optional pre-computed next observation features
            model: Which model to compute the loss for ("actor", "critic", "discrete_critic", or "temperature")

        Returns:
            The computed loss tensor
        """
        # Extract common components from batch
        actions: Tensor = batch[ACTION]
        observations: dict[str, Tensor] = batch["state"]
        observation_features: Tensor = batch.get("observation_feature")

        if model == "critic":
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            next_observation_features: Tensor = batch.get("next_observation_feature")

            loss_critic = self.compute_loss_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
            )

            return {"loss_critic": loss_critic}

        if model == "discrete_critic" and self.config.num_discrete_actions is not None:
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            next_observation_features: Tensor = batch.get("next_observation_feature")
            complementary_info = batch.get("complementary_info")
            loss_discrete_critic = self.compute_loss_discrete_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
                complementary_info=complementary_info,
            )
            return {"loss_discrete_critic": loss_discrete_critic}
        if model == "actor":
            complementary_info = batch.get("complementary_info")
            actor_bc_only = batch.get("actor_bc_only", False)
            if isinstance(actor_bc_only, torch.Tensor):
                actor_bc_only = bool(actor_bc_only.item())
            return {
                "loss_actor": self.compute_loss_actor(
                    observations=observations,
                    observation_features=observation_features,
                    actions=actions,
                    complementary_info=complementary_info,
                    bc_only=bool(actor_bc_only),
                )
            }

        if model == "temperature":
            return {
                "loss_temperature": self.compute_loss_temperature(
                    observations=observations,
                    observation_features=observation_features,
                )
            }

        raise ValueError(f"Unknown model type: {model}")

    def update_target_networks(self):
        """Update target networks with exponential moving average"""
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic_ensemble.parameters(),
            strict=True,
        ):
            target_param.data.copy_(
                param.data * self.config.critic_target_update_weight
                + target_param.data * (1.0 - self.config.critic_target_update_weight)
            )
        if self.config.num_discrete_actions is not None:
            for target_param, param in zip(
                self.discrete_critic_target.parameters(),
                self.discrete_critic.parameters(),
                strict=True,
            ):
                target_param.data.copy_(
                    param.data * self.config.critic_target_update_weight
                    + target_param.data * (1.0 - self.config.critic_target_update_weight)
                )

    def update_temperature(self):
        self.temperature = self.log_alpha.exp().item()

    def _project_actions_to_max_relative_target(
        self, observations: dict[str, Tensor], actions: Tensor
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        """Project normalized actions to respect `config.max_relative_target` (if configured).

        The real-robot control loop clamps *absolute joint targets* to be at most
        `max_relative_target` away from the current joint positions. During training,
        the critic is fit on executed (clamped) actions, but the actor may otherwise
        optimize Q-values for unreachable (pre-clamp) actions, leading to policy drift.

        This helper mirrors the robot-side clamp in the SAC loss computation so that:
        - critic targets use reachable next actions
        - actor gradients optimize Q(s, a_executed)
        """
        max_relative_target = getattr(self.config, "max_relative_target", None)
        if max_relative_target is None:
            return actions, None

        if not isinstance(actions, Tensor) or actions.ndim != 2:
            return actions, None

        state = observations.get(OBS_STATE)
        if not isinstance(state, Tensor):
            return actions, None
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[0] != actions.shape[0]:
            return actions, None

        action_dim = int(actions.shape[-1])
        if state.shape[-1] < action_dim:
            return actions, None

        dataset_stats = getattr(self.config, "dataset_stats", None)
        if not isinstance(dataset_stats, dict):
            return actions, None

        action_stats = dataset_stats.get(ACTION, {})
        if not isinstance(action_stats, dict):
            return actions, None

        if "min" not in action_stats or "max" not in action_stats:
            return actions, None

        try:
            action_min = torch.as_tensor(
                action_stats["min"], device=actions.device, dtype=actions.dtype
            ).view(1, -1)
            action_max = torch.as_tensor(
                action_stats["max"], device=actions.device, dtype=actions.dtype
            ).view(1, -1)
        except Exception:
            return actions, None

        if action_min.numel() != action_dim or action_max.numel() != action_dim:
            return actions, None

        present = state[:, :action_dim].to(device=actions.device, dtype=actions.dtype)

        denom = (action_max - action_min).clamp(min=1e-6)
        action_robot = 0.5 * (actions + 1.0) * denom + action_min

        try:
            max_diff_val = float(max_relative_target)
        except Exception:
            return actions, None
        if max_diff_val < 0:
            return actions, None

        max_diff = torch.full_like(action_robot, max_diff_val)
        diff = action_robot - present
        safe_diff = torch.minimum(diff, max_diff)
        safe_diff = torch.maximum(safe_diff, -max_diff)
        clamped_action_robot = present + safe_diff
        clamped_action_robot = torch.minimum(torch.maximum(clamped_action_robot, action_min), action_max)

        clamped_actions = 2.0 * (clamped_action_robot - action_min) / denom - 1.0
        clamped_actions = clamped_actions.clamp(min=-1.0, max=1.0)

        # Lightweight stats (stay as tensors to let the caller decide when to sync).
        action_delta_max = diff.abs().max(dim=-1)[0]
        clamp_amount_max = (action_robot - clamped_action_robot).abs().max(dim=-1)[0]
        violation = (actions - clamped_actions).abs().max(dim=-1)[0]
        clamped_mask = violation > 1e-6

        stats = {
            "max_relative_target_delta_max_mean": action_delta_max.mean(),
            "max_relative_target_clamp_amount_max_mean": clamp_amount_max.mean(),
            "max_relative_target_violation_max_mean": violation.mean(),
            "max_relative_target_clamp_frac": clamped_mask.float().mean(),
        }
        return clamped_actions, stats

    def compute_loss_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
    ) -> Tensor:
        with torch.no_grad():
            next_action_preds, next_log_probs, _ = self.actor(next_observations, next_observation_features)
            next_action_preds, proj_stats = self._project_actions_to_max_relative_target(
                observations=next_observations,
                actions=next_action_preds,
            )
            if proj_stats is not None:
                # Detached scalars for optional logging by the learner.
                self._last_max_relative_target_stats_critic = {k: v.detach() for k, v in proj_stats.items()}

            # 2- compute q targets
            q_targets = self.critic_forward(
                observations=next_observations,
                actions=next_action_preds,
                use_target=True,
                observation_features=next_observation_features,
            )

            # subsample critics to prevent overfitting if use high UTD (update to date)
            # TODO: Get indices before forward pass to avoid unnecessary computation
            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)
                indices = indices[: self.config.num_subsample_critics]
                q_targets = q_targets[indices]

            # critics subsample size
            min_q, _ = q_targets.min(dim=0)  # Get values from min operation
            if self.config.use_backup_entropy:
                min_q = min_q - (self.temperature * next_log_probs)

            td_target = rewards + (1 - done) * self.config.discount * min_q

        # 3- compute predicted qs
        if self.config.num_discrete_actions is not None:
            # NOTE: We only want to keep the continuous action part
            # In the buffer we have the full action space (continuous + discrete)
            # We need to split them before concatenating them in the critic forward
            actions: Tensor = actions[:, :DISCRETE_DIMENSION_INDEX]
        q_preds = self.critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
            observation_features=observation_features,
        )

        # 4- Calculate loss
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        # You compute the mean loss of the batch for each critic and then to compute the final loss you sum them up
        critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            ).mean(dim=1)
        ).sum()
        return critics_loss

    def compute_loss_discrete_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features=None,
        next_observation_features=None,
        complementary_info=None,
    ):
        # NOTE: We only want to keep the discrete action part
        # In the buffer we have the full action space (continuous + discrete)
        # We need to split them before concatenating them in the critic forward
        actions_discrete: Tensor = actions[:, DISCRETE_DIMENSION_INDEX:].clone()
        actions_discrete = torch.round(actions_discrete)
        actions_discrete = actions_discrete.long()

        discrete_penalties: Tensor | None = None
        if complementary_info is not None:
            discrete_penalties: Tensor | None = complementary_info.get("discrete_penalty")

        with torch.no_grad():
            # For DQN, select actions using online network, evaluate with target network
            next_discrete_qs = self.discrete_critic_forward(
                next_observations, use_target=False, observation_features=next_observation_features
            )
            best_next_discrete_action = torch.argmax(next_discrete_qs, dim=-1, keepdim=True)

            # Get target Q-values from target network
            target_next_discrete_qs = self.discrete_critic_forward(
                observations=next_observations,
                use_target=True,
                observation_features=next_observation_features,
            )

            # Use gather to select Q-values for best actions
            target_next_discrete_q = torch.gather(
                target_next_discrete_qs, dim=1, index=best_next_discrete_action
            ).squeeze(-1)

            # Compute target Q-value with Bellman equation
            rewards_discrete = rewards
            if discrete_penalties is not None:
                rewards_discrete = rewards + discrete_penalties
            target_discrete_q = rewards_discrete + (1 - done) * self.config.discount * target_next_discrete_q

        # Get predicted Q-values for current observations
        predicted_discrete_qs = self.discrete_critic_forward(
            observations=observations, use_target=False, observation_features=observation_features
        )

        # Use gather to select Q-values for taken actions
        predicted_discrete_q = torch.gather(predicted_discrete_qs, dim=1, index=actions_discrete).squeeze(-1)

        # Compute MSE loss between predicted and target Q-values
        discrete_critic_loss = F.mse_loss(input=predicted_discrete_q, target=target_discrete_q)
        return discrete_critic_loss

    def compute_loss_temperature(self, observations, observation_features: Tensor | None = None) -> Tensor:
        """Compute the temperature loss"""
        # calculate temperature loss
        with torch.no_grad():
            _, log_probs, _ = self.actor(observations, observation_features)
        temperature_loss = (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()
        return temperature_loss

    def compute_loss_actor(
        self,
        observations,
        observation_features: Tensor | None = None,
        actions: Tensor | None = None,
        complementary_info: dict | None = None,
        *,
        bc_only: bool = False,
    ) -> Tensor:
        actions_pi, log_probs, actor_aux = self.actor(observations, observation_features)
        try:
            residual_stats = getattr(self.actor, "_last_residual_stats", None)
            if isinstance(residual_stats, dict):
                self._last_act_residual_stats = {k: v.detach() for k, v in residual_stats.items()}
        except Exception:
            pass
        bc_weight = float(getattr(self.config, "intervention_bc_loss_weight", 0.0) or 0.0)
        bc_only_weight = float(
            getattr(self.config, "intervention_bc_loss_weight_bc_only", bc_weight) or bc_weight
        )
        bc_loss: Tensor | None = None
        bc_enabled = (bc_weight > 0.0) or (bc_only and bc_only_weight > 0.0)
        if bc_enabled and isinstance(actions, Tensor) and isinstance(complementary_info, dict):
            is_intervention = complementary_info.get(TeleopEvents.IS_INTERVENTION.value)
            if is_intervention is None:
                is_intervention = complementary_info.get(TeleopEvents.IS_INTERVENTION)
            if isinstance(is_intervention, Tensor) and is_intervention.numel() == actions.shape[0]:
                mask = is_intervention.view(-1) > 0.5
                if bool(mask.any().item()):
                    # Use the deterministic policy action (mode) for BC regularization:
                    # - SAC: tanh(mean)
                    # - SAC-ACT: ACT prior action (already in [-1, 1] space)
                    deterministic_action = actor_aux
                    if not hasattr(self.actor, "prior_action"):
                        deterministic_action = torch.tanh(deterministic_action)
                    if isinstance(deterministic_action, Tensor) and deterministic_action.ndim == 2:
                        action_dim = int(deterministic_action.shape[-1])
                        target = actions[:, :action_dim] if actions.shape[-1] >= action_dim else actions
                        target = target.to(
                            device=deterministic_action.device, dtype=deterministic_action.dtype
                        )
                        deterministic_action, _ = self._project_actions_to_max_relative_target(
                            observations=observations,
                            actions=deterministic_action,
                        )
                        bc_loss = F.mse_loss(deterministic_action[mask], target[mask])
                        self._last_intervention_bc_loss = bc_loss.detach()

        if bc_only:
            if bc_loss is None:
                return torch.zeros((), device=actions_pi.device)
            return bc_only_weight * bc_loss

        actions_pi_projected, stats = self._project_actions_to_max_relative_target(
            observations=observations,
            actions=actions_pi,
        )
        if stats is not None:
            # Detached scalars for optional logging by the learner.
            self._last_max_relative_target_stats_actor = {k: v.detach() for k, v in stats.items()}

        q_preds = self.critic_forward(
            observations=observations,
            actions=actions_pi_projected,
            use_target=False,
            observation_features=observation_features,
        )
        min_q_preds = q_preds.min(dim=0)[0]

        actor_loss = ((self.temperature * log_probs) - min_q_preds).mean()

        penalty_weight = float(getattr(self.config, "max_relative_target_violation_penalty", 0.0) or 0.0)
        if penalty_weight > 0.0 and stats is not None:
            # Encourage the policy to stay within the reachable set (in normalized action space).
            actor_loss = actor_loss + penalty_weight * stats["max_relative_target_violation_max_mean"]

        if bc_loss is not None and bc_weight > 0.0:
            actor_loss = actor_loss + bc_weight * bc_loss

        return actor_loss

    def _init_encoders(self):
        """Initialize shared or separate encoders for actor and critic."""
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = SACObservationEncoder(self.config)
        self.encoder_actor = (
            self.encoder_critic if self.shared_encoder else SACObservationEncoder(self.config)
        )

    def _init_critics(self, continuous_action_dim):
        """Build critic ensemble, targets, and optional discrete critic."""
        heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(encoder=self.encoder_critic, ensemble=heads)
        target_heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(encoder=self.encoder_critic, ensemble=target_heads)
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

        if self.config.num_discrete_actions is not None:
            self._init_discrete_critics()

    def _init_discrete_critics(self):
        """Build discrete discrete critic ensemble and target networks."""
        self.discrete_critic = DiscreteCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )
        self.discrete_critic_target = DiscreteCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )

        # TODO: (maractingi, azouitine) Compile the discrete critic
        self.discrete_critic_target.load_state_dict(self.discrete_critic.state_dict())

    def _init_actor(self, continuous_action_dim):
        """Initialize policy actor network and default target entropy."""
        # NOTE: The actor select only the continuous action part
        self.actor = Policy(
            encoder=self.encoder_actor,
            network=MLP(input_dim=self.encoder_actor.output_dim, **asdict(self.config.actor_network_kwargs)),
            action_dim=continuous_action_dim,
            encoder_is_shared=self.shared_encoder,
            **asdict(self.config.policy_kwargs),
        )

        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            dim = continuous_action_dim + (1 if self.config.num_discrete_actions is not None else 0)
            self.target_entropy = -np.prod(dim) / 2

    def _init_temperature(self):
        """Set up temperature parameter and initial log_alpha."""
        temp_init = self.config.temperature_init
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)]))
        self.temperature = self.log_alpha.exp().item()


class SACObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.config = config
        self._init_image_layers()
        self._init_state_layers()
        self._compute_output_dim()

    def _init_image_layers(self) -> None:
        self.image_keys = [k for k in self.config.input_features if is_image_feature(k)]
        self.has_images = bool(self.image_keys)
        if not self.has_images:
            return

        if self.config.vision_encoder_name is not None:
            self.image_encoder = PretrainedImageEncoder(self.config)
        else:
            self.image_encoder = DefaultImageEncoder(self.config)

        if self.config.freeze_vision_encoder:
            freeze_image_encoder(self.image_encoder)

        dummy = torch.zeros(1, *self.config.input_features[self.image_keys[0]].shape)
        with torch.no_grad():
            _, channels, height, width = self.image_encoder(dummy).shape

        self.spatial_embeddings = nn.ModuleDict()
        self.post_encoders = nn.ModuleDict()

        for key in self.image_keys:
            name = key.replace(".", "_")
            self.spatial_embeddings[name] = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channels,
                num_features=self.config.image_embedding_pooling_dim,
            )
            self.post_encoders[name] = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(
                    in_features=channels * self.config.image_embedding_pooling_dim,
                    out_features=self.config.latent_dim,
                ),
                nn.LayerNorm(normalized_shape=self.config.latent_dim),
                nn.Tanh(),
            )

    def _init_state_layers(self) -> None:
        self.has_env = OBS_ENV_STATE in self.config.input_features
        self.has_state = OBS_STATE in self.config.input_features
        if self.has_env:
            dim = self.config.input_features[OBS_ENV_STATE].shape[0]
            self.env_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )
        if self.has_state:
            dim = self.config.input_features[OBS_STATE].shape[0]
            self.state_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )

    def _compute_output_dim(self) -> None:
        out = 0
        if self.has_images:
            out += len(self.image_keys) * self.config.latent_dim
        if self.has_env:
            out += self.config.latent_dim
        if self.has_state:
            out += self.config.latent_dim
        self._out_dim = out

    def forward(
        self, obs: dict[str, Tensor], cache: dict[str, Tensor] | None = None, detach: bool = False
    ) -> Tensor:
        parts = []
        if self.has_images:
            if cache is None:
                cache = self.get_cached_image_features(obs)
            parts.append(self._encode_images(cache, detach))
        if self.has_env:
            parts.append(self.env_encoder(obs[OBS_ENV_STATE]))
        if self.has_state:
            parts.append(self.state_encoder(obs[OBS_STATE]))
        if parts:
            return torch.cat(parts, dim=-1)

        raise ValueError(
            "No parts to concatenate, you should have at least one image or environment state or state"
        )

    def get_cached_image_features(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Extract and optionally cache image features from observations.

        This function processes image observations through the vision encoder once and returns
        the resulting features.
        When the image encoder is shared between actor and critics AND frozen, these features can be safely cached and
        reused across policy components (actor, critic, discrete_critic), avoiding redundant forward passes.

        Performance impact:
        - The vision encoder forward pass is typically the main computational bottleneck during training and inference
        - Caching these features can provide 2-4x speedup in training and inference

        Usage patterns:
        - Called in select_action()
        - Called in learner.py's get_observation_features() to pre-compute features for all policy components
        - Called internally by forward()

        Args:
            obs: Dictionary of observation tensors containing image keys

        Returns:
            Dictionary mapping image keys to their corresponding encoded features
        """
        batched = torch.cat([obs[k] for k in self.image_keys], dim=0)
        out = self.image_encoder(batched)
        chunks = torch.chunk(out, len(self.image_keys), dim=0)
        return dict(zip(self.image_keys, chunks, strict=False))

    def _encode_images(self, cache: dict[str, Tensor], detach: bool) -> Tensor:
        """Encode image features from cached observations.

        This function takes pre-encoded image features from the cache and applies spatial embeddings and post-encoders.
        It also supports detaching the encoded features if specified.

        Args:
            cache (dict[str, Tensor]): The cached image features.
            detach (bool): Usually when the encoder is shared between actor and critics,
            we want to detach the encoded features on the policy side to avoid backprop through the encoder.
            More detail here `https://cdn.aaai.org/ojs/17276/17276-13-20770-1-2-20210518.pdf`

        Returns:
            Tensor: The encoded image features.
        """
        feats = []
        for k, feat in cache.items():
            safe_key = k.replace(".", "_")
            x = self.spatial_embeddings[safe_key](feat)
            x = self.post_encoders[safe_key](x)
            if detach:
                x = x.detach()
            feats.append(x)
        return torch.cat(feats, dim=-1)

    @property
    def output_dim(self) -> int:
        return self._out_dim


class MLP(nn.Module):
    """Multi-layer perceptron builder.

    Dynamically constructs a sequence of layers based on `hidden_dims`:
      1) Linear (in_dim -> out_dim)
      2) Optional Dropout if `dropout_rate` > 0 and (not final layer or `activate_final`)
      3) LayerNorm on the output features
      4) Activation (standard for intermediate layers, `final_activation` for last layer if `activate_final`)

    Arguments:
        input_dim (int): Size of input feature dimension.
        hidden_dims (list[int]): Sizes for each hidden layer.
        activations (Callable or str): Activation to apply between layers.
        activate_final (bool): Whether to apply activation at the final layer.
        dropout_rate (Optional[float]): Dropout probability applied before normalization and activation.
        final_activation (Optional[Callable or str]): Activation for the final layer when `activate_final` is True.

    For each layer, `in_dim` is updated to the previous `out_dim`. All constructed modules are
    stored in `self.net` as an `nn.Sequential` container.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        total = len(hidden_dims)

        for idx, out_dim in enumerate(hidden_dims):
            # 1) linear transform
            layers.append(nn.Linear(in_dim, out_dim))

            is_last = idx == total - 1
            # 2-4) optionally add dropout, normalization, and activation
            if not is_last or activate_final:
                if dropout_rate and dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.LayerNorm(out_dim))
                act_cls = final_activation if is_last and final_activation else activations
                act = act_cls if isinstance(act_cls, nn.Module) else getattr(nn, act_cls)()
                layers.append(act)

            in_dim = out_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.net(x))


class CriticEnsemble(nn.Module):
    """
    CriticEnsemble wraps multiple CriticHead modules into an ensemble.

    Args:
        encoder (SACObservationEncoder): encoder for observations.
        ensemble (List[CriticHead]): list of critic heads.
        init_final (float | None): optional initializer scale for final layers.

    Forward returns a tensor of shape (num_critics, batch_size) containing Q-values.
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        ensemble: list[CriticHead],
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.init_final = init_final
        self.critics = nn.ModuleList(ensemble)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        # Move each tensor in observations to device
        observations = {k: v.to(device) for k, v in observations.items()}

        obs_enc = self.encoder(observations, cache=observation_features)

        inputs = torch.cat([obs_enc, actions], dim=-1)

        # Loop through critics and collect outputs
        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))

        # Stack outputs to match expected shape [num_critics, batch_size]
        q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)
        return q_values


class DiscreteCritic(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 3,
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim

        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )

        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=self.output_dim)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(
        self, observations: torch.Tensor, observation_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        observations = {k: v.to(device) for k, v in observations.items()}
        obs_enc = self.encoder(observations, cache=observation_features)
        return self.output_layer(self.net(obs_enc))


class Policy(nn.Module):
    def __init__(
        self,
        encoder: SACObservationEncoder,
        network: nn.Module,
        action_dim: int,
        std_min: float = -5,
        std_max: float = 2,
        fixed_std: torch.Tensor | None = None,
        init_final: float | None = None,
        use_tanh_squash: bool = False,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: SACObservationEncoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max
        self.fixed_std = fixed_std
        self.use_tanh_squash = use_tanh_squash
        self.encoder_is_shared = encoder_is_shared

        # Find the last Linear layer's output dimension
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break
        # Mean layer
        self.mean_layer = nn.Linear(out_features, action_dim)
        if init_final is not None:
            nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.mean_layer.weight)

        # Standard deviation layer or parameter
        if fixed_std is None:
            self.std_layer = nn.Linear(out_features, action_dim)
            if init_final is not None:
                nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
                nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
            else:
                orthogonal_init()(self.std_layer.weight)

    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # We detach the encoder if it is shared to avoid backprop through it
        # This is important to avoid the encoder to be updated through the policy
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # Get network outputs
        outputs = self.network(obs_enc)
        means = self.mean_layer(outputs)

        # Compute standard deviations
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            std = torch.exp(log_std)  # Match JAX "exp"
            std = torch.clamp(std, self.std_min, self.std_max)  # Match JAX default clip
        else:
            std = self.fixed_std.expand_as(means)

        # Build transformed distribution
        dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)

        # Sample actions (reparameterized)
        actions = dist.rsample()

        # Compute log_probs
        log_probs = dist.log_prob(actions)

        return actions, log_probs, means

    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get encoded features from observations"""
        device = get_device_from_parameters(self)
        observations = observations.to(device)
        if self.encoder is not None:
            with torch.inference_mode():
                return self.encoder(observations)
        return observations


class DefaultImageEncoder(nn.Module):
    def __init__(self, config: SACConfig):
        super().__init__()
        image_key = next(key for key in config.input_features if is_image_feature(key))
        self.image_enc_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=config.input_features[image_key].shape[0],
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=7,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=5,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.image_enc_layers(x)
        return x


def freeze_image_encoder(image_encoder: nn.Module):
    """Freeze all parameters in the encoder"""
    for param in image_encoder.parameters():
        param.requires_grad = False


class PretrainedImageEncoder(nn.Module):
    def __init__(self, config: SACConfig):
        super().__init__()

        self.image_enc_layers, self.image_enc_out_shape = self._load_pretrained_vision_encoder(config)

    def _load_pretrained_vision_encoder(self, config: SACConfig):
        """Set up CNN encoder"""
        from transformers import AutoModel

        self.image_enc_layers = AutoModel.from_pretrained(config.vision_encoder_name, trust_remote_code=True)

        if hasattr(self.image_enc_layers.config, "hidden_sizes"):
            self.image_enc_out_shape = self.image_enc_layers.config.hidden_sizes[-1]  # Last channel dimension
        elif hasattr(self.image_enc_layers, "fc"):
            self.image_enc_out_shape = self.image_enc_layers.fc.in_features
        else:
            raise ValueError("Unsupported vision encoder architecture, make sure you are using a CNN")
        return self.image_enc_layers, self.image_enc_out_shape

    def forward(self, x):
        enc_feat = self.image_enc_layers(x).last_hidden_state
        return enc_feat


def orthogonal_init():
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        PyTorch implementation of learned spatial embeddings

        Args:
            height: Spatial height of input features
            width: Spatial width of input features
            channel: Number of input channels
            num_features: Number of output embedding dimensions
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))

        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        """
        Forward pass for spatial embedding

        Args:
            features: Input tensor of shape [B, C, H, W] where B is batch size,
                     C is number of channels, H is height, and W is width
        Returns:
            Output tensor of shape [B, C*F] where F is the number of features
        """

        features_expanded = features.unsqueeze(-1)  # [B, C, H, W, 1]
        kernel_expanded = self.kernel.unsqueeze(0)  # [1, C, H, W, F]

        # Element-wise multiplication and spatial reduction
        output = (features_expanded * kernel_expanded).sum(dim=(2, 3))  # Sum over H,W dimensions

        # Reshape to combine channel and feature dimensions
        output = output.view(output.size(0), -1)  # [B, C*F]

        return output


class RescaleFromTanh(Transform):
    def __init__(self, low: float = -1, high: float = 1):
        super().__init__()

        self.low = low

        self.high = high

    def _call(self, x):
        # Rescale from (-1, 1) to (low, high)

        return 0.5 * (x + 1.0) * (self.high - self.low) + self.low

    def _inverse(self, y):
        # Rescale from (low, high) back to (-1, 1)

        return 2.0 * (y - self.low) / (self.high - self.low) - 1.0

    def log_abs_det_jacobian(self, x, y):
        # log|d(rescale)/dx| = sum(log(0.5 * (high - low)))

        scale = 0.5 * (self.high - self.low)

        return torch.sum(torch.log(scale), dim=-1)


class TanhMultivariateNormalDiag(TransformedDistribution):
    def __init__(self, loc, scale_diag, low=None, high=None):
        base_dist = MultivariateNormal(loc, torch.diag_embed(scale_diag))

        transforms = [TanhTransform(cache_size=1)]

        if low is not None and high is not None:
            low = torch.as_tensor(low)

            high = torch.as_tensor(high)

            transforms.insert(0, RescaleFromTanh(low, high))

        super().__init__(base_dist, transforms)

    def mode(self):
        # Mode is mean of base distribution, passed through transforms

        x = self.base_dist.mean

        for transform in self.transforms:
            x = transform(x)

        return x

    def stddev(self):
        std = self.base_dist.stddev

        x = std

        for transform in self.transforms:
            x = transform(x)

        return x


class _ACTGaussianActor(nn.Module):
    """Stochastic tanh-Gaussian actor backed by a pretrained ACT model (mean provider)."""

    def __init__(
        self,
        *,
        act_policy: nn.Module,
        act_image_features: list[str],
        action_dim: int,
        action_index: int = 0,
        init_std: float = 0.2,
        std_min: float = 1e-5,
        std_max: float | None = None,
        preprocessor: object | None = None,
        postprocessor: object | None = None,
        action_min: list[float] | None = None,
        action_max: list[float] | None = None,
        act_action_norm_mode: NormalizationMode | None = None,
        prefetch_action_queue: bool = False,
        prefetch_min_queue_len: int = 10,
        freeze_act_policy: bool = False,
        residual_network: nn.Module | None = None,
        residual_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.act_policy = act_policy
        self.act_image_features = list(act_image_features)
        self.action_index = int(action_index)
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.act_action_norm_mode = act_action_norm_mode
        self._prefetch_action_queue = bool(prefetch_action_queue)
        self._prefetch_min_queue_len = max(0, int(prefetch_min_queue_len))
        self._freeze_act_policy = bool(freeze_act_policy)
        self.residual_network = residual_network
        self.residual_scale = float(residual_scale)
        self._last_residual_stats: dict[str, Tensor] | None = None

        self._prefetch_generation: int = 0
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_lock = threading.Lock()
        self._act_queue_lock = threading.Lock()
        self._act_forward_lock = threading.Lock()

        if self._freeze_act_policy and isinstance(self.act_policy, nn.Module):
            for param in self.act_policy.parameters():
                param.requires_grad = False

        if (action_min is None) != (action_max is None):
            raise ValueError("action_min and action_max must be provided together (or both be None).")
        if action_min is not None and action_max is not None:
            if len(action_min) != action_dim or len(action_max) != action_dim:
                raise ValueError(
                    f"Action dim mismatch for action_min/action_max: expected {action_dim} values, "
                    f"got min={len(action_min)} max={len(action_max)}."
                )
            self.register_buffer(
                "_action_min",
                torch.tensor(action_min, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "_action_max",
                torch.tensor(action_max, dtype=torch.float32),
                persistent=False,
            )
        else:
            self._action_min = None  # type: ignore[assignment]
            self._action_max = None  # type: ignore[assignment]

        try:
            std_min_val = float(std_min)
        except Exception:
            std_min_val = 1e-5
        if std_min_val <= 0:
            raise ValueError(f"std_min must be > 0, got {std_min_val}")
        std_max_val: float | None
        try:
            std_max_val = None if std_max is None else float(std_max)
        except Exception:
            std_max_val = None
        if std_max_val is not None and std_max_val <= 0:
            std_max_val = None
        if std_max_val is not None and std_max_val < std_min_val:
            std_max_val = std_min_val
        self._std_min = std_min_val
        self._std_max = std_max_val

        if init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {init_std}")
        init_std_val = float(init_std)
        if self._std_max is not None:
            init_std_val = min(init_std_val, self._std_max)
        init_std_val = max(init_std_val, self._std_min)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_std_val), dtype=torch.float32))

    def _compute_residual(self, observations: dict[str, Tensor], *, batch: int) -> Tensor:
        if self.residual_network is None or self.residual_scale <= 0.0:
            self._last_residual_stats = {
                "act_residual_mean_abs": torch.tensor(0.0, device=self.log_std.device),
                "act_residual_max_abs": torch.tensor(0.0, device=self.log_std.device),
            }
            return torch.zeros(
                (batch, self.log_std.shape[0]), dtype=torch.float32, device=self.log_std.device
            )

        state = observations.get(OBS_STATE)
        if not isinstance(state, Tensor):
            return torch.zeros(
                (batch, self.log_std.shape[0]), dtype=torch.float32, device=self.log_std.device
            )
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2:
            return torch.zeros(
                (batch, self.log_std.shape[0]), dtype=torch.float32, device=self.log_std.device
            )

        action_dim = int(self.log_std.shape[0])
        if state.shape[-1] < action_dim:
            return torch.zeros(
                (batch, self.log_std.shape[0]), dtype=torch.float32, device=self.log_std.device
            )

        state_in = state[:, :action_dim].to(
            device=self.log_std.device, dtype=torch.float32, non_blocking=True
        )
        # Normalize state to roughly [-1, 1] when action bounds are known (common for joint-space absolute targets).
        if self._action_min is not None and self._action_max is not None:
            try:
                action_min = self._action_min.to(device=state_in.device, dtype=state_in.dtype)
                action_max = self._action_max.to(device=state_in.device, dtype=state_in.dtype)
                if action_min.numel() == action_dim and action_max.numel() == action_dim:
                    denom = (action_max - action_min).clamp(min=1e-6)
                    state_in = 2.0 * (state_in - action_min) / denom - 1.0
            except Exception:
                pass
        residual_raw = self.residual_network(state_in)
        if not isinstance(residual_raw, Tensor) or residual_raw.shape != (
            state_in.shape[0],
            self.log_std.shape[0],
        ):
            self._last_residual_stats = {
                "act_residual_mean_abs": torch.tensor(0.0, device=self.log_std.device),
                "act_residual_max_abs": torch.tensor(0.0, device=self.log_std.device),
            }
            return torch.zeros(
                (batch, self.log_std.shape[0]), dtype=torch.float32, device=self.log_std.device
            )
        residual = torch.tanh(residual_raw) * float(self.residual_scale)
        try:
            self._last_residual_stats = {
                "act_residual_mean_abs": residual.abs().mean().detach(),
                "act_residual_max_abs": residual.abs().max().detach(),
            }
        except Exception:
            self._last_residual_stats = None
        return residual

    def _apply_residual(self, base_action: Tensor, observations: dict[str, Tensor]) -> Tensor:
        if self.residual_network is None or self.residual_scale <= 0.0:
            return base_action
        if not isinstance(base_action, Tensor):
            return base_action
        if base_action.ndim == 1:
            base_action = base_action.unsqueeze(0)
        if base_action.ndim != 2:
            return base_action

        batch = int(base_action.shape[0])
        residual = self._compute_residual(observations, batch=batch)
        if residual.shape != base_action.shape:
            return base_action
        residual = residual.to(device=base_action.device, dtype=base_action.dtype, non_blocking=True)
        return (base_action + residual).clamp(min=-1.0, max=1.0)

    def train(self, mode: bool = True):
        # Keep ACT in eval mode even when fine-tuning with RL:
        # - avoids the VAE training-only assertion requiring ACTION in batch
        # - keeps dropout disabled for stability
        super().train(mode)
        if isinstance(self.act_policy, nn.Module):
            self.act_policy.eval()
        return self

    def reset(self) -> None:
        reset_fn = getattr(self.act_policy, "reset", None)
        if callable(reset_fn):
            reset_fn()
        with self._prefetch_lock:
            self._prefetch_generation += 1
            self._prefetch_thread = None
        self._ensure_act_action_queue_capacity()

    def _get_act_action_queue(self) -> deque | None:
        q = getattr(self.act_policy, "_action_queue", None)
        if isinstance(q, deque):
            return q
        return None

    def _act_n_action_steps(self) -> int:
        cfg = getattr(self.act_policy, "config", None)
        n_action_steps = getattr(cfg, "n_action_steps", None)
        try:
            n_action_steps_int = int(n_action_steps)
        except Exception:
            n_action_steps_int = 0
        return max(0, n_action_steps_int)

    def _ensure_act_action_queue_capacity(self) -> None:
        if not self._prefetch_action_queue:
            return
        q = self._get_act_action_queue()
        if q is None:
            return
        n_action_steps = self._act_n_action_steps()
        if n_action_steps <= 0:
            return
        desired = n_action_steps * 2
        if q.maxlen is not None and q.maxlen >= desired:
            return
        self.act_policy._action_queue = deque(q, maxlen=desired)

    def _prefetch_inflight(self) -> bool:
        t = self._prefetch_thread
        return t is not None and t.is_alive()

    def _maybe_start_prefetch(self, observations: dict[str, Tensor]) -> None:
        if not self._prefetch_action_queue or self._prefetch_min_queue_len <= 0:
            return
        if not isinstance(observations, dict) or not observations:
            return
        if self._prefetch_inflight():
            return

        with self._prefetch_lock:
            if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
                return
            generation = self._prefetch_generation
            obs_snapshot = dict(observations)
            t = threading.Thread(
                target=self._prefetch_worker,
                args=(obs_snapshot, generation),
                daemon=True,
            )
            self._prefetch_thread = t
            t.start()

    def _prefetch_worker(self, observations: dict[str, Tensor], generation: int) -> None:
        try:
            actions_cpu = self._compute_prior_action_chunk(observations)
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).exception("ACT prefetch failed; continuing without prefetch.")
            return

        with self._prefetch_lock:
            if generation != self._prefetch_generation:
                return

        q = self._get_act_action_queue()
        if q is None:
            return

        with self._act_queue_lock:
            if generation != self._prefetch_generation:
                return
            q.extend(actions_cpu.transpose(0, 1))

    def _prepare_act_inputs(self, observations: dict[str, Tensor]) -> dict[str, Tensor]:
        device = get_device_from_parameters(self.act_policy)
        prepared: dict[str, Tensor] = {}
        for key, value in observations.items():
            if not isinstance(value, Tensor):
                prepared[key] = value
                continue

            is_image = False
            if isinstance(key, str):
                is_image = (
                    key in self.act_image_features
                    or key == OBS_IMAGE
                    or key.startswith(f"{OBS_IMAGES}.")
                    or key.startswith(f"{OBS_IMAGE}.")
                )

            if is_image:
                if value.dtype == torch.uint8:
                    value = value.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
                else:
                    value = value.to(device=device, dtype=torch.float32, non_blocking=True)
            elif value.is_floating_point():
                value = value.to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                value = value.to(device=device, non_blocking=True)

            prepared[key] = value
        return prepared

    def _compute_prior_action_chunk(self, observations: dict[str, Tensor]) -> Tensor:
        """Compute an ACT action chunk mapped into SAC's normalized [-1, 1] space (CPU tensor)."""
        batch = self._prepare_act_inputs(observations)
        batch = self._act_batch(batch)

        act_model = getattr(self.act_policy, "model", None)
        if not isinstance(act_model, nn.Module):
            raise TypeError("ACT policy is missing a `.model` nn.Module.")

        with self._act_forward_lock:
            if self._freeze_act_policy:
                with torch.no_grad():
                    actions_seq, _ = act_model(batch)
            else:
                actions_seq, _ = act_model(batch)

        try:
            self.act_policy._last_action_chunk = actions_seq
            prev_id = getattr(self.act_policy, "_last_action_chunk_id", 0)
            self.act_policy._last_action_chunk_id = int(prev_id) + 1
        except Exception:
            pass

        if actions_seq.ndim != 3:
            raise ValueError(
                f"Expected ACT to return (B, S, D) actions, got shape {tuple(actions_seq.shape)}"
            )

        n_action_steps = self._act_n_action_steps()
        if n_action_steps <= 0:
            n_action_steps = actions_seq.shape[1]
        n_action_steps = min(n_action_steps, actions_seq.shape[1])

        actions = actions_seq[:, :n_action_steps]

        if self.postprocessor is not None:
            b, s, d = actions.shape
            flat = actions.reshape(b * s, d)
            flat = self._postprocess_action(flat)
            flat = self._minmax_normalize_action(flat)
            actions = flat.reshape(b, s, d)

        return actions.clamp(min=-1.0, max=1.0).detach().to("cpu")

    def _hold_action_from_state(self, observations: dict[str, Tensor]) -> Tensor:
        state = observations.get(OBS_STATE)
        if not isinstance(state, Tensor):
            raise KeyError("Missing observation.state; cannot compute hold action.")
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.to(dtype=torch.float32, device="cpu")

        if self._action_min is None or self._action_max is None:
            # Best-effort: keep still in normalized space (may not correspond to hold for absolute targets).
            return torch.zeros_like(state, dtype=torch.float32, device="cpu")

        action_min = self._action_min.to(device=state.device, dtype=state.dtype)
        action_max = self._action_max.to(device=state.device, dtype=state.dtype)
        denom = (action_max - action_min).clamp(min=1e-6)
        hold = 2.0 * (state - action_min) / denom - 1.0
        return hold.clamp(min=-1.0, max=1.0)

    def _act_batch(self, observations: dict[str, Tensor]) -> dict[str, Tensor]:
        batch: dict[str, Tensor] = observations
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)

        if self.act_image_features:
            batch = dict(batch)
            missing = [key for key in self.act_image_features if key not in batch]
            if missing:
                available_images = sorted(
                    key
                    for key in batch.keys()
                    if key == OBS_IMAGE or key.startswith(f"{OBS_IMAGES}.") or key.startswith(f"{OBS_IMAGE}.")
                )
                raise KeyError(
                    "Missing image features required by the ACT checkpoint: "
                    f"{missing}. Available image keys in the current batch: {available_images}. "
                    "Make sure your environment/camera names and `policy.input_features` match the ACT "
                    "`act_pretrained_path` config.json."
                )

            batch[OBS_IMAGES] = [batch[key] for key in self.act_image_features]
        return batch

    def _predict_act_action(self, batch: dict[str, Tensor], *, use_action_queue: bool) -> Tensor:
        """Return a single ACT action (B, D) in ACT output space."""
        if use_action_queue:
            select_action = getattr(self.act_policy, "select_action", None)
            if callable(select_action):
                action = select_action(batch)
                if not isinstance(action, torch.Tensor):
                    raise TypeError(
                        f"ACT policy select_action() must return a torch.Tensor, got {type(action)}"
                    )
                return action

        act_model = getattr(self.act_policy, "model", None)
        if not isinstance(act_model, nn.Module):
            raise TypeError("ACT policy is missing a `.model` nn.Module.")

        if self._freeze_act_policy:
            with torch.no_grad():
                actions_seq, _ = act_model(batch)
        else:
            actions_seq, _ = act_model(batch)
        # Best-effort cache of the latest predicted chunk for logging/debugging. This is useful when
        # `use_action_queue` is False (otherwise ACTPolicy.select_action manages chunking internally).
        try:
            self.act_policy._last_action_chunk = actions_seq
            prev_id = getattr(self.act_policy, "_last_action_chunk_id", 0)
            self.act_policy._last_action_chunk_id = int(prev_id) + 1
        except Exception:
            pass
        if actions_seq.ndim != 3:
            raise ValueError(
                f"Expected ACT to return (B, S, D) actions, got shape {tuple(actions_seq.shape)}"
            )

        if not (0 <= self.action_index < actions_seq.shape[1]):
            raise ValueError(
                f"act_action_index={self.action_index} out of range for chunk_size={actions_seq.shape[1]}"
            )
        return actions_seq[:, self.action_index]

    def _postprocess_action(self, action: Tensor) -> Tensor:
        if self.postprocessor is None:
            return action

        # The processor pipeline expects a batch-like dict with an `action` key.
        processed = self.postprocessor({ACTION: action})
        if not isinstance(processed, dict) or ACTION not in processed:
            raise TypeError("ACT postprocessor must return a dict containing an 'action' key.")
        processed_action = processed[ACTION]
        if not isinstance(processed_action, torch.Tensor):
            raise TypeError(f"ACT postprocessor returned non-tensor action: {type(processed_action)}")
        return processed_action

    def _minmax_normalize_action(self, action: Tensor) -> Tensor:
        if self._action_min is None or self._action_max is None:
            raise ValueError(
                "action_min/action_max are required to map ACT postprocessed actions back into SAC's [-1, 1] space."
            )
        action_min = self._action_min.to(device=action.device, dtype=action.dtype)
        action_max = self._action_max.to(device=action.device, dtype=action.dtype)
        denom = (action_max - action_min).clamp(min=1e-6)
        return 2.0 * (action - action_min) / denom - 1.0

    def prior_action(self, observations: dict[str, Tensor], *, use_action_queue: bool) -> Tensor:
        """Compute the ACT-based prior action in SAC's normalized [-1, 1] action space."""
        temporal_coeff = getattr(getattr(self.act_policy, "config", None), "temporal_ensemble_coeff", None)
        q = self._get_act_action_queue() if use_action_queue and temporal_coeff is None else None

        # Queue-based ACT inference path. We store ACT actions already mapped into SAC's normalized
        # [-1, 1] space so that per-step control stays on CPU and avoids periodic GPU stalls.
        if use_action_queue and q is not None:
            self._ensure_act_action_queue_capacity()

            act_action: Tensor | None = None
            remaining = 0
            with self._act_queue_lock:
                if len(q) > 0:
                    act_action = q.popleft()
                    remaining = len(q)

            if act_action is None:
                # If a background prefetch is in flight, keep the robot stable by commanding a
                # "hold current joint positions" action in normalized space.
                if self._prefetch_inflight():
                    try:
                        return self._hold_action_from_state(observations)
                    except Exception:
                        pass

                try:
                    actions_cpu = self._compute_prior_action_chunk(observations)
                    with self._act_queue_lock:
                        q.extend(actions_cpu.transpose(0, 1))
                        if len(q) > 0:
                            act_action = q.popleft()
                            remaining = len(q)
                except Exception:  # noqa: BLE001
                    logging.getLogger(__name__).exception(
                        "ACT action-queue refill failed; falling back to hold action."
                    )
                    try:
                        return self._hold_action_from_state(observations)
                    except Exception:
                        batch = 1
                        state = observations.get(OBS_STATE)
                        if isinstance(state, Tensor) and state.ndim >= 1:
                            batch = int(state.shape[0]) if state.ndim > 1 else 1
                        return torch.zeros((batch, self.log_std.shape[0]), dtype=torch.float32)

            if remaining <= self._prefetch_min_queue_len:
                self._maybe_start_prefetch(observations)

            act_action = act_action.clamp(min=-1.0, max=1.0)
            return self._apply_residual(act_action, observations)

        # Fallback: single-step ACT inference (may be expensive, but used for training and/or
        # when ACT temporal ensembling is enabled).
        batch = self._act_batch(observations)
        act_action = self._predict_act_action(batch, use_action_queue=use_action_queue)

        # If a postprocessor exists, it typically unnormalizes ACT actions into "env action" units.
        # Map back into SAC's normalized [-1, 1] space using the configured min/max bounds.
        if self.postprocessor is not None:
            act_action = self._postprocess_action(act_action)
            act_action = self._minmax_normalize_action(act_action)
        else:
            # Without a postprocessor we assume ACT already outputs in SAC-normalized space (common when
            # ACTION normalization during ACT training used MIN_MAX).
            pass

        act_action = act_action.clamp(min=-1.0, max=1.0)
        return self._apply_residual(act_action, observations)

    def _sample_from_prior(self, prior_action: Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if prior_action.device != self.log_std.device:
            prior_action = prior_action.to(device=self.log_std.device)

        eps = 1e-6
        prior_action_safe = prior_action.clamp(min=-1.0 + eps, max=1.0 - eps)
        means = torch.atanh(prior_action_safe)

        std = torch.exp(self.log_std)
        if self._std_max is not None:
            std = std.clamp(min=self._std_min, max=self._std_max)
        else:
            std = std.clamp(min=self._std_min)
        std = std.expand_as(means)
        dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)
        actions = dist.rsample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def forward(
        self,
        observations: dict[str, Tensor],
        observation_features: Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del observation_features

        prior_action = self.prior_action(observations, use_action_queue=False)
        actions, log_probs = self._sample_from_prior(prior_action)
        return actions, log_probs, prior_action


class SACActPolicy(SACPolicy):
    """SAC policy that uses a pretrained ACT model as its actor (fine-tuned via SAC)."""

    config_class = SACActConfig
    name = "sac_act"

    def __init__(self, config: SACActConfig | None = None):
        super().__init__(config)

    def _init_actor(self, continuous_action_dim: int):
        cfg: SACActConfig = self.config  # type: ignore[assignment]
        if cfg.act_pretrained_path is None:
            raise ValueError("`policy.act_pretrained_path` is required when `policy.type` is 'sac_act'.")

        # Load ACT policy on the same device as the SAC policy.
        # Import locally to avoid heavyweight imports unless this policy type is used.
        from lerobot.configs.policies import PreTrainedConfig as _PreTrainedConfig
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.processor.device_processor import DeviceProcessorStep
        from lerobot.processor.pipeline import DataProcessorPipeline

        act_cfg = _PreTrainedConfig.from_pretrained(cfg.act_pretrained_path)
        if act_cfg.type != "act":
            raise ValueError(
                f"`policy.act_pretrained_path` must point to an ACT policy, but got type={act_cfg.type!r}."
            )
        act_cfg.device = cfg.device
        act_policy: ACTPolicy = ACTPolicy.from_pretrained(cfg.act_pretrained_path, config=act_cfg)
        act_policy.to(cfg.device)
        act_policy.eval()

        act_action_norm_mode = None
        if isinstance(act_policy.config, ACTConfig) and isinstance(
            act_policy.config.normalization_mapping, dict
        ):
            act_action_norm_mode = act_policy.config.normalization_mapping.get("ACTION")

        # Optional ACT preprocessor (normalization) to match the distribution seen during ACT training.
        preprocessor = None
        if cfg.act_use_preprocessor:
            preprocessor_config = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
            try:
                preprocessor = DataProcessorPipeline.from_pretrained(
                    cfg.act_pretrained_path, config_filename=preprocessor_config
                )
                # Ensure preprocessor device matches current run device (best-effort).
                for step in getattr(preprocessor, "steps", []):
                    if isinstance(step, DeviceProcessorStep):
                        step.device = cfg.device
                        step.__post_init__()
            except Exception:
                # If the ACT checkpoint doesn't contain a preprocessor, proceed without it.
                preprocessor = None

        # Optional ACT postprocessor (action unnormalization). Needed when ACT was trained with
        # ACTION normalization modes like MEAN_STD. If unavailable, we fall back to treating ACT
        # outputs as already in SAC-normalized space (common for MIN_MAX).
        postprocessor = None
        if getattr(cfg, "act_use_postprocessor", True):
            postprocessor_config = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
            try:
                postprocessor = DataProcessorPipeline.from_pretrained(
                    cfg.act_pretrained_path, config_filename=postprocessor_config
                )
                for step in getattr(postprocessor, "steps", []):
                    if isinstance(step, DeviceProcessorStep):
                        step.device = cfg.device
                        step.__post_init__()
            except FileNotFoundError:
                postprocessor = None
            if (
                postprocessor is None
                and act_action_norm_mode is not None
                and act_action_norm_mode != NormalizationMode.MIN_MAX
            ):
                raise FileNotFoundError(
                    f"{postprocessor_config} not found in {cfg.act_pretrained_path}. "
                    "This ACT checkpoint appears to use an ACTION normalization mode that requires a postprocessor "
                    f"(got {act_action_norm_mode!r}). Point `policy.act_pretrained_path` to a full ACT model export "
                    "that includes the policy processors, or set `policy.act_use_postprocessor=false` if you know "
                    "the ACT outputs are already in SAC-normalized [-1, 1] space."
                )

        act_image_features = []
        if getattr(act_policy.config, "image_features", None):
            # ACTConfig.image_features returns list[str]
            act_image_features = list(act_policy.config.image_features)

        # Basic dimension sanity check.
        act_action_dim = act_policy.config.action_feature.shape[0]
        if act_action_dim != continuous_action_dim:
            raise ValueError(
                f"Action dim mismatch between SAC config and ACT checkpoint: "
                f"SAC expects {continuous_action_dim}, ACT provides {act_action_dim}."
            )

        # If we apply an ACT postprocessor, we must map back to SAC's [-1, 1] space using min/max stats.
        action_min = None
        action_max = None
        action_stats = (cfg.dataset_stats or {}).get(ACTION, {})
        if isinstance(action_stats, dict) and "min" in action_stats and "max" in action_stats:
            action_min = action_stats["min"]
            action_max = action_stats["max"]
        elif postprocessor is not None:
            raise ValueError(
                "When using an ACT postprocessor, you must provide `policy.dataset_stats.action.min/max` "
                "so SAC-ACT can map ACT actions back into SAC's [-1, 1] space."
            )

        residual_network = None
        residual_scale = float(getattr(cfg, "act_residual_scale", 0.0) or 0.0)
        if residual_scale > 0.0:
            # Residual policy head: small MLP on robot state only.
            state_ft = (cfg.input_features or {}).get(OBS_STATE)
            state_dim = None
            if state_ft is not None and getattr(state_ft, "shape", None):
                try:
                    state_dim = int(state_ft.shape[0])
                except Exception:
                    state_dim = None
            if state_dim is None:
                raise ValueError(
                    "act_residual_scale>0 requires `observation.state` in policy.input_features."
                )

            hidden_dims = list(getattr(cfg, "act_residual_hidden_dims", [64, 64]))
            layers: list[nn.Module] = []
            prev = state_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev, int(h)))
                layers.append(nn.ReLU())
                prev = int(h)
            out = nn.Linear(prev, continuous_action_dim)
            # Start from the pretrained ACT behavior (zero residual).
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
            layers.append(out)
            residual_network = nn.Sequential(*layers)

        self.actor = _ACTGaussianActor(
            act_policy=act_policy,
            act_image_features=act_image_features,
            action_dim=continuous_action_dim,
            action_index=cfg.act_action_index,
            init_std=cfg.act_init_std,
            std_min=getattr(getattr(cfg, "policy_kwargs", None), "std_min", 1e-5),
            std_max=getattr(getattr(cfg, "policy_kwargs", None), "std_max", None),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            action_min=action_min,
            action_max=action_max,
            act_action_norm_mode=act_action_norm_mode,
            prefetch_action_queue=getattr(cfg, "act_prefetch_action_queue", False),
            prefetch_min_queue_len=getattr(cfg, "act_prefetch_min_queue_len", 10),
            freeze_act_policy=bool(getattr(cfg, "act_freeze_backbone", False)),
            residual_network=residual_network,
            residual_scale=residual_scale,
        )

        # Keep SAC target entropy logic.
        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            dim = continuous_action_dim + (1 if self.config.num_discrete_actions is not None else 0)
            self.target_entropy = -np.prod(dim) / 2

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation.

        If `policy.act_deterministic_inference` is True, returns the ACT prior (distribution mode).
        Otherwise, samples a tanh-Gaussian action centered on the ACT prior (SAC-style).
        """
        cfg: SACActConfig = self.config  # type: ignore[assignment]
        prior_action = self.actor.prior_action(
            batch, use_action_queue=getattr(cfg, "act_use_action_queue", False)
        )
        if getattr(cfg, "act_deterministic_inference", False):
            actions = prior_action
        else:
            actions, _ = self.actor._sample_from_prior(prior_action)

        actions, _ = self._project_actions_to_max_relative_target(observations=batch, actions=actions)

        if self.config.num_discrete_actions is not None:
            discrete_action_value = self.discrete_critic(batch, None)
            discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)
            actions = torch.cat([actions, discrete_action], dim=-1)

        return actions

    def reset(self) -> None:
        reset_fn = getattr(self.actor, "reset", None)
        if callable(reset_fn):
            reset_fn()
