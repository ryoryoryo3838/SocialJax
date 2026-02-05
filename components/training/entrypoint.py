"""Shared entrypoint helpers for training/evaluation."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import warnings

from loguru import logger
from omegaconf import DictConfig, OmegaConf

_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"


def configure_console_logging() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    logger.remove()
    logger.add(sys.stderr, level="INFO", format=_LOG_FORMAT)
    warnings.filterwarnings(
        "ignore",
        message=r"scatter inputs have incompatible types:.*",
        category=FutureWarning,
    )
    import logging

    logging.getLogger("jax._src.xla_bridge").addFilter(
        lambda record: "Unable to initialize backend 'tpu'" not in record.getMessage()
    )


def add_file_logger(path: str | Path) -> None:
    logger.add(str(path), level="INFO", format=_LOG_FORMAT, mode="w")


def save_hydra_config(cfg: DictConfig, path: str | Path) -> None:
    OmegaConf.save(cfg, str(path), resolve=True)
