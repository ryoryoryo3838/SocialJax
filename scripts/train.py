"""Hydra entrypoint for reusable SocialJax training runs."""
from __future__ import annotations

import os
from datetime import datetime
import sys
import warnings

from loguru import logger
import hydra
from omegaconf import DictConfig, OmegaConf

from components.training.config import build_config

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)
warnings.filterwarnings(
    "ignore",
    message=r"scatter inputs have incompatible types:.*",
    category=FutureWarning,
)


_ALGO_NAMES = {"ippo", "mappo", "svo"}


def _load_algorithms():
    from components.algorithms import ippo, mappo, svo

    return {
        "ippo": ippo.make_train,
        "mappo": mappo.make_train,
        "svo": svo.make_train,
    }


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    mem_fraction = cfg.get("xla_python_client_mem_fraction")
    if mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)

    config = build_config(cfg)
    algo_name = cfg.algorithm.name
    if algo_name not in _ALGO_NAMES:
        raise ValueError(f"Unknown algorithm '{algo_name}'.")

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("runs", run_id)
    config["RUN_DIR"] = run_dir
    ckpt_dir = config.get("CHECKPOINT_DIR")
    if ckpt_dir:
        if os.path.isabs(ckpt_dir):
            try:
                ckpt_dir = os.path.relpath(ckpt_dir, os.getcwd())
            except ValueError:
                ckpt_dir = os.path.basename(ckpt_dir)
        config["CHECKPOINT_DIR"] = os.path.join(run_dir, ckpt_dir)

    os.makedirs(run_dir, exist_ok=True)
    logger.add(
        os.path.join(run_dir, "train.log"),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        mode="w",
    )

    if cfg.dry_run:
        OmegaConf.save(cfg, os.path.join(run_dir, "hydra.yaml"), resolve=True)
        return
    import jax

    algo_map = _load_algorithms()
    train_fn = algo_map[algo_name](config)
    train_fn(jax.random.PRNGKey(config["SEED"]))
    OmegaConf.save(cfg, os.path.join(run_dir, "hydra.yaml"), resolve=True)


if __name__ == "__main__":
    main()
