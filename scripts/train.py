"""Hydra entrypoint for reusable SocialJax training runs."""
from __future__ import annotations

import os
from datetime import datetime
import hydra
from omegaconf import DictConfig

from components.training.config import build_config
from components.training.entrypoint import (
    add_file_logger,
    configure_console_logging,
    save_hydra_config,
)

# Initializing loguru
configure_console_logging()


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
    # Reduce GPU usage
    mem_fraction = cfg.get("xla_python_client_mem_fraction")
    if mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
    # organize hydra settings for training
    config = build_config(cfg)
    algo_name = cfg.algorithm.name
    if algo_name not in _ALGO_NAMES:
        raise ValueError(f"Unknown algorithm '{algo_name}'.")
    # location of outputs
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("runs", run_id)
    config["RUN_DIR"] = run_dir
    wandb_cfg = config.setdefault("WANDB", {})
    if wandb_cfg.get("name") is None:
        # Keep W&B run names aligned with runs/<run_id>.
        wandb_cfg["name"] = run_id
    ckpt_dir = config.get("CHECKPOINT_DIR")
    if ckpt_dir:
        if os.path.isabs(ckpt_dir):
            try:
                ckpt_dir = os.path.relpath(ckpt_dir, os.getcwd())
            except ValueError:
                ckpt_dir = os.path.basename(ckpt_dir)
        config["CHECKPOINT_DIR"] = os.path.join(run_dir, ckpt_dir)

    os.makedirs(run_dir, exist_ok=True)
    # make train.log
    add_file_logger(os.path.join(run_dir, "train.log"))

    # hydra config test
    if cfg.dry_run:
        save_hydra_config(cfg, os.path.join(run_dir, "hydra.yaml"))
        return
    
    # imported here to reduce GPU usage
    import jax

    algo_map = _load_algorithms()
    # get train()
    train_fn = algo_map[algo_name](config)
    # start learning
    train_fn(jax.random.PRNGKey(config["SEED"]))
    # save hydra config in the learning directory
    save_hydra_config(cfg, os.path.join(run_dir, "hydra.yaml"))


if __name__ == "__main__":
    main()
