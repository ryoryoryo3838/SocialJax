"""Config helpers for training entrypoints."""
from __future__ import annotations

from typing import Any, Dict

import os

from omegaconf import DictConfig, OmegaConf


def _set_nested(config: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = config
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def build_config(cfg: DictConfig) -> Dict[str, Any]:
    config = OmegaConf.to_container(cfg, resolve=True)
    algorithm_cfg = dict(config.get("algorithm", {}))
    env_cfg = dict(config.get("env", {}))

    algorithm_cfg["ENV_NAME"] = env_cfg.get("env_name")
    algorithm_cfg["ENV_KWARGS"] = env_cfg.get("env_kwargs", {})
    algorithm_cfg["ENV_ID"] = env_cfg.get("env_name")
    algorithm_cfg["ALGORITHM"] = algorithm_cfg.get("name")
    wandb_cfg = dict(config.get("wandb", {}))
    env_wandb = dict(env_cfg.get("wandb", {}))
    wandb_cfg.update(env_wandb)
    algorithm_cfg["WANDB"] = wandb_cfg

    seed = cfg.get("seed")
    if seed is not None:
        algorithm_cfg["SEED"] = seed

    independent_policy = cfg.get("independent_policy")
    if independent_policy is not None:
        algorithm_cfg["PARAMETER_SHARING"] = not independent_policy

    independent_reward = cfg.get("independent_reward")
    if independent_reward is not None:
        _set_nested(algorithm_cfg, "ENV_KWARGS.shared_rewards", not independent_reward)

    encoder_type = str(algorithm_cfg.get("ENCODER_TYPE", "cnn")).lower()
    if encoder_type in ("cnn", "transformer"):
        _set_nested(algorithm_cfg, "ENV_KWARGS.cnn", True)
    elif encoder_type == "mlp":
        _set_nested(algorithm_cfg, "ENV_KWARGS.cnn", False)

    ckpt_dir = algorithm_cfg.get("CHECKPOINT_DIR")
    if ckpt_dir and not os.path.isabs(ckpt_dir):
        algorithm_cfg["CHECKPOINT_DIR"] = os.path.abspath(ckpt_dir)

    return algorithm_cfg
