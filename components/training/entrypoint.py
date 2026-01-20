"""
Shared entrypoint helpers for training/evaluation.


loguru is a Python logging library designed to be simpler and more user-friendly than the standard logging module. Here's a high-level overview of how it works:

The logger is a global singleton that can manage multiple "handlers" (output destinations)
You add output destinations using logger.add(...), such as stdout/stderr/files
Existing handlers can be removed with logger.remove()
Each handler can be configured with settings like log level, format, and rotation
When you call methods like logger.info(...), the log message is sent to all handlers that meet the specified criteria

In this code snippet, we're controlling the behavior to "remove all existing handlers and output to stderr and a file."
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import warnings

from loguru import logger
from omegaconf import DictConfig, OmegaConf

_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"


def configure_console_logging() -> None:
    # Suppresses low-level TensorFlow logging
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # Remove to prevent duplicate logs
    logger.remove()
    # Add a handler to output INFO or higher level logs to the Python standard error stream in the _LOG_FORMAT format
    logger.add(sys.stderr, level="INFO", format=_LOG_FORMAT)
    # Ignore future type error warnings
    warnings.filterwarnings(
        "ignore",
        message=r"scatter inputs have incompatible types:.*",
        category=FutureWarning,
    )

# make train.log
def add_file_logger(path: str | Path) -> None:
    logger.add(str(path), level="INFO", format=_LOG_FORMAT, mode="w")

# save hydra config
def save_hydra_config(cfg: DictConfig, path: str | Path) -> None:
    OmegaConf.save(cfg, str(path), resolve=True)
