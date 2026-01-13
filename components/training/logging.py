"""Logging utilities for training."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import jax.numpy as jnp
from loguru import logger


def init_wandb(config: Dict[str, Any]) -> Optional[Any]:
    wandb_cfg = config.get("WANDB", {})
    if not wandb_cfg.get("enabled", False):
        return None

    import wandb

    project = wandb_cfg.get("project", "socialjax")
    entity = wandb_cfg.get("entity")
    name = wandb_cfg.get("name")
    group = wandb_cfg.get("group")
    tags = wandb_cfg.get("tags")
    mode = wandb_cfg.get("mode", "online")

    if name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f"run/{timestamp}"

    try:
        wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            tags=tags,
            mode=mode,
            config=config,
        )
    except Exception as exc:
        logger.warning("W&B init failed, continuing without W&B: {}", exc)
        return None
    return wandb


def log_metrics(metrics: Dict[str, Any], wandb: Optional[Any]) -> None:
    if wandb is not None:
        wandb.log(metrics, step=int(metrics.get("env_step", 0)))

    update_step = metrics.get("update_step")
    env_step = metrics.get("env_step")
    reward = metrics.get("train/reward_mean")
    parts = []
    if update_step is not None:
        parts.append(f"update={int(update_step)}")
    if env_step is not None:
        parts.append(f"env_step={int(env_step)}")
    if reward is not None:
        parts.append(f"reward_mean={float(reward):.4f}")
    if not parts:
        return
    logger.info(" | ".join(parts))


def update_info_stats(stats: Dict[str, Dict[str, float]], info: Dict[str, Any]) -> None:
    for key, value in info.items():
        try:
            mean_val = float(jnp.mean(value))
        except Exception:
            continue
        entry = stats.setdefault(key, {"sum": 0.0, "count": 0.0})
        entry["sum"] += mean_val
        entry["count"] += 1.0


def finalize_info_stats(stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in stats.items():
        if value["count"] == 0:
            continue
        metrics[f"env/{key}"] = value["sum"] / value["count"]
    return metrics
