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
    # Convert Hydra's DictConfig to a regular Python dictionary (including resolving interpolations)
    config = OmegaConf.to_container(cfg, resolve=True)
    algorithm_cfg = dict(config.get("algorithm", {}))
    model_cfg = config.get("model") or {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    encoder_cfg = dict(model_cfg.get("encoder", {}) or {})
    decoder_cfg = dict(model_cfg.get("decoder", {}) or {})
    # merge config in the order of encoder→decoder→algorithm
    if encoder_cfg or decoder_cfg:
        merged = {}
        merged.update(encoder_cfg)
        merged.update(decoder_cfg)
        merged.update(algorithm_cfg)
        algorithm_cfg = merged
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
    if independent_policy is None:
        independent_policy = algorithm_cfg.get("independent_policy")
    if independent_policy is not None:
        algorithm_cfg["PARAMETER_SHARING"] = not independent_policy

    independent_reward = cfg.get("independent_reward")
    if independent_reward is None:
        independent_reward = algorithm_cfg.get("independent_reward")
    if independent_reward is not None:
        _set_nested(algorithm_cfg, "ENV_KWARGS.shared_rewards", not independent_reward)

    encoder_type = str(algorithm_cfg.get("ENCODER_TYPE", "cnn")).lower()
    cnn_override = {"cnn": True, "transformer": True, "mlp": False}.get(encoder_type)
    if cnn_override is not None:
        _set_nested(algorithm_cfg, "ENV_KWARGS.cnn", cnn_override)

    ckpt_dir = algorithm_cfg.get("CHECKPOINT_DIR")
    # Change relative path to absolute path
    if ckpt_dir and not os.path.isabs(ckpt_dir):
        algorithm_cfg["CHECKPOINT_DIR"] = os.path.abspath(ckpt_dir)

    return algorithm_cfg
