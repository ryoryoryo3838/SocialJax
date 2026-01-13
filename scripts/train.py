"""Hydra entrypoint for reusable SocialJax training runs."""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

import jax

from components.algorithms import ippo, mappo, svo
from components.training.config import build_config

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)


_ALGO_MAP = {
    "ippo": ippo.make_train,
    "mappo": mappo.make_train,
    "svo": svo.make_train,
}


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    config = build_config(cfg)
    algo_name = cfg.algorithm.name
    if algo_name not in _ALGO_MAP:
        raise ValueError(f"Unknown algorithm '{algo_name}'.")

    train_fn = _ALGO_MAP[algo_name](config)
    if cfg.dry_run:
        return
    train_fn(jax.random.PRNGKey(config["SEED"]))


if __name__ == "__main__":
    main()
