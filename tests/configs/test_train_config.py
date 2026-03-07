from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


def test_rabc_pi05_defaults_to_pretrained(monkeypatch):
    monkeypatch.setattr("lerobot.configs.parser.get_path_arg", lambda _name: None)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="dummy/dataset"),
        policy=PI05Config(push_to_hub=False),
        use_rabc=True,
    )

    cfg.validate()

    assert cfg.policy is not None
    assert cfg.policy.pretrained_path == Path("lerobot/pi05_base")


def test_rabc_pi05_keeps_explicit_pretrained_path(monkeypatch):
    monkeypatch.setattr("lerobot.configs.parser.get_path_arg", lambda _name: None)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="dummy/dataset"),
        policy=PI05Config(push_to_hub=False, pretrained_path=Path("custom/pi05_checkpoint")),
        use_rabc=True,
    )

    cfg.validate()

    assert cfg.policy is not None
    assert cfg.policy.pretrained_path == Path("custom/pi05_checkpoint")
