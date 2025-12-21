# !/usr/bin/env python

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.sac.configuration_sac import SACActConfig
from lerobot.policies.sac.modeling_sac import SACActPolicy
from lerobot.processor.device_processor import DeviceProcessorStep
from lerobot.processor.normalize_processor import UnnormalizerProcessorStep
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


@pytest.mark.parametrize("batch_size,state_dim,env_dim,action_dim", [(2, 6, 3, 6), (1, 10, 4, 3)])
def test_sac_act_policy_select_action(tmp_path, batch_size: int, state_dim: int, env_dim: int, action_dim: int):
    # 1) Create and save a minimal ACT checkpoint (state + env_state only, no images).
    act_cfg = ACTConfig(
        # minimal transformer sizes for test speed
        chunk_size=2,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(env_dim,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        device="cpu",
    )
    act_cfg.validate_features()
    act_policy = ACTPolicy(config=act_cfg)
    act_policy.save_pretrained(tmp_path)

    # 2) Instantiate SACActPolicy that loads the ACT checkpoint as actor init.
    sac_act_cfg = SACActConfig(
        act_pretrained_path=tmp_path,
        act_use_preprocessor=False,
        act_use_postprocessor=False,
        act_action_index=0,
        act_init_std=0.5,
        device="cpu",
        use_torch_compile=False,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(env_dim,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        dataset_stats={
            OBS_STATE: {"min": [0.0] * state_dim, "max": [1.0] * state_dim},
            OBS_ENV_STATE: {"min": [0.0] * env_dim, "max": [1.0] * env_dim},
            ACTION: {"min": [0.0] * action_dim, "max": [1.0] * action_dim},
        },
    )
    sac_act_cfg.validate_features()

    policy = SACActPolicy(config=sac_act_cfg)
    policy.eval()

    obs = {
        OBS_STATE: torch.randn(batch_size, state_dim),
        OBS_ENV_STATE: torch.randn(batch_size, env_dim),
    }

    with torch.no_grad():
        action = policy.select_action(obs)

    assert action.shape == (batch_size, action_dim)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_sac_act_policy_losses_run(tmp_path):
    batch_size = 2
    state_dim = 6
    env_dim = 3
    action_dim = 6

    act_cfg = ACTConfig(
        chunk_size=2,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(env_dim,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        device="cpu",
    )
    act_cfg.validate_features()
    ACTPolicy(config=act_cfg).save_pretrained(tmp_path)

    sac_act_cfg = SACActConfig(
        act_pretrained_path=tmp_path,
        act_use_preprocessor=False,
        act_use_postprocessor=False,
        device="cpu",
        use_torch_compile=False,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(env_dim,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        dataset_stats={
            OBS_STATE: {"min": [0.0] * state_dim, "max": [1.0] * state_dim},
            OBS_ENV_STATE: {"min": [0.0] * env_dim, "max": [1.0] * env_dim},
            ACTION: {"min": [0.0] * action_dim, "max": [1.0] * action_dim},
        },
    )
    sac_act_cfg.validate_features()
    policy = SACActPolicy(config=sac_act_cfg)

    batch = {
        ACTION: torch.randn(batch_size, action_dim),
        "reward": torch.randn(batch_size),
        "state": {OBS_STATE: torch.randn(batch_size, state_dim), OBS_ENV_STATE: torch.randn(batch_size, env_dim)},
        "next_state": {
            OBS_STATE: torch.randn(batch_size, state_dim),
            OBS_ENV_STATE: torch.randn(batch_size, env_dim),
        },
        "done": torch.zeros(batch_size),
    }

    # Forward passes should run without raising.
    loss_critic = policy.forward(batch, model="critic")["loss_critic"]
    loss_actor = policy.forward(batch, model="actor")["loss_actor"]
    loss_temp = policy.forward(batch, model="temperature")["loss_temperature"]

    assert loss_critic.shape == ()
    assert loss_actor.shape == ()
    assert loss_temp.shape == ()


def test_sac_act_policy_uses_postprocessor_for_prior(tmp_path):
    batch_size = 2
    state_dim = 6
    env_dim = 3
    action_dim = 6

    act_cfg = ACTConfig(
        chunk_size=2,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(env_dim,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        device="cpu",
    )
    act_cfg.validate_features()
    act_policy = ACTPolicy(config=act_cfg)
    act_policy.save_pretrained(tmp_path)

    # Save a minimal ACT postprocessor that performs MEAN_STD unnormalization for ACTION.
    mean = [1.0] * action_dim
    std = [2.0] * action_dim
    unnormalizer = UnnormalizerProcessorStep(
        features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
        stats={ACTION: {"mean": mean, "std": std}},
        device="cpu",
    )
    postprocessor = DataProcessorPipeline(
        steps=[unnormalizer, DeviceProcessorStep(device="cpu")],
        name="policy_postprocessor",
    )
    postprocessor.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")

    action_min = [-100.0] * action_dim
    action_max = [100.0] * action_dim
    sac_act_cfg = SACActConfig(
        act_pretrained_path=tmp_path,
        act_use_preprocessor=False,
        act_use_postprocessor=True,
        act_deterministic_inference=True,
        act_action_index=0,
        act_init_std=0.5,
        device="cpu",
        use_torch_compile=False,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(env_dim,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        dataset_stats={
            OBS_STATE: {"min": [0.0] * state_dim, "max": [1.0] * state_dim},
            OBS_ENV_STATE: {"min": [0.0] * env_dim, "max": [1.0] * env_dim},
            ACTION: {"min": action_min, "max": action_max},
        },
    )
    sac_act_cfg.validate_features()
    policy = SACActPolicy(config=sac_act_cfg).eval()

    obs = {
        OBS_STATE: torch.randn(batch_size, state_dim),
        OBS_ENV_STATE: torch.randn(batch_size, env_dim),
    }

    with torch.no_grad():
        deterministic_action = policy.select_action(obs)

    # Expected: raw ACT action -> MEAN_STD unnormalize -> min-max normalize back into [-1, 1].
    with torch.no_grad():
        raw_actions = act_policy.predict_action_chunk(obs)[:, 0]
        unnormalized = raw_actions * torch.tensor(std) + torch.tensor(mean)
        denom = torch.tensor(action_max) - torch.tensor(action_min)
        expected = 2.0 * (unnormalized - torch.tensor(action_min)) / denom - 1.0
        expected = expected.clamp(min=-1.0, max=1.0)

    assert torch.allclose(deterministic_action, expected, atol=1e-5)
