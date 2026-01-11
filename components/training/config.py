"""Config helpers for training entrypoints."""
from __future__ import annotations

from typing import Any, Dict

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

    if cfg.seed is not None:
        algorithm_cfg["SEED"] = cfg.seed

    if cfg.independent_policy is not None:
        algorithm_cfg["PARAMETER_SHARING"] = not cfg.independent_policy

    if cfg.independent_reward is not None:
        _set_nested(algorithm_cfg, "ENV_KWARGS.shared_rewards", not cfg.independent_reward)

    return algorithm_cfg
