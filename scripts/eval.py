"""Evaluate trained policies and optionally render GIFs."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List
import sys
import warnings

from loguru import logger
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

import socialjax
from components.algorithms.networks import Actor, ActorCritic, Critic, EncoderConfig
from components.training.checkpoint import agent_checkpoint_dir, load_checkpoint
from components.training.config import build_config
from components.training.utils import build_world_state, flatten_obs, unflatten_actions

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


def _build_encoder_cfg(config: Dict) -> EncoderConfig:
    return EncoderConfig(
        activation=config.get("ACTIVATION", "relu"),
        mlp_sizes=tuple(config.get("MLP_HIDDEN_SIZES", (64, 64))),
        cnn_channels=tuple(config.get("CNN_CHANNELS", (32, 32, 32))),
        cnn_kernel_sizes=tuple(config.get("CNN_KERNEL_SIZES", ((5, 5), (3, 3), (3, 3)))),
        cnn_dense_size=int(config.get("CNN_DENSE_SIZE", 64)),
        encoder_type=config.get("ENCODER_TYPE", "cnn"),
        transformer_patch_size=int(config.get("TRANSFORMER_PATCH_SIZE", 4)),
        transformer_layers=int(config.get("TRANSFORMER_LAYERS", 2)),
        transformer_heads=int(config.get("TRANSFORMER_HEADS", 4)),
        transformer_mlp_dim=int(config.get("TRANSFORMER_MLP_DIM", 128)),
        transformer_embed_dim=int(config.get("TRANSFORMER_EMBED_DIM", 64)),
    )


def _select_action(dist, rng, deterministic: bool):
    if deterministic:
        return jnp.argmax(dist.probs, axis=-1)
    return dist.sample(seed=rng)


def _save_gif(frames: List[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def _resolve_output_dir(output_dir: str | None, ckpt_dir: str) -> Path:
    if output_dir:
        base = Path(output_dir)
        if base.is_absolute():
            return base
        return Path(ckpt_dir) / base
    return Path(ckpt_dir) / "evaluation"


def _has_agent_checkpoints(ckpt_dir: str) -> bool:
    return Path(agent_checkpoint_dir(ckpt_dir, 0)).exists()


def _load_agent_payloads(
    ckpt_dir: str,
    step: int | None,
    num_agents: int,
    key: str,
    target: Any,
) -> List:
    payloads = []
    for agent_idx in range(num_agents):
        payload = load_checkpoint(
            agent_checkpoint_dir(ckpt_dir, agent_idx),
            step,
            target={key: target},
        )
        payloads.append(payload[key])
    return payloads


@hydra.main(version_base=None, config_path="config", config_name="eval")
def main(cfg: DictConfig) -> None:
    config = build_config(cfg)
    algorithm = cfg.algorithm.name
    env = socialjax.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))

    ckpt_dir = cfg.checkpoint_dir or config.get("CHECKPOINT_DIR")
    if ckpt_dir is None:
        raise ValueError("checkpoint_dir is required for evaluation")
    output_root = _resolve_output_dir(cfg.output_dir, ckpt_dir)
    eval_run_dir = Path(ckpt_dir).parent / "evaluation"
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, eval_run_dir / "hydra.yaml", resolve=True)

    encoder_cfg = _build_encoder_cfg(config)
    num_agents = env.num_agents
    rng = jax.random.PRNGKey(0)
    parameter_sharing = bool(config.get("PARAMETER_SHARING", True))
    init_obs = jnp.zeros((1, *(env.observation_space()[0]).shape))

    if algorithm == "mappo":
        actor_net = Actor(env.action_space().n, encoder_cfg)
        critic_net = Critic(encoder_cfg)
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        actor_params = actor_net.init(actor_rng, init_obs)

        world_shape = build_world_state(
            jnp.zeros((1, num_agents, *env.observation_space()[0].shape))
        ).shape[1:]
        critic_init = jnp.zeros((1, *world_shape))
        critic_params = critic_net.init(critic_rng, critic_init)

        if not parameter_sharing:
            raise ValueError("MAPPO uses shared actor policies; set independent_policy=false.")
        target = {"actor_params": actor_params, "critic_params": critic_params}
        payload = load_checkpoint(ckpt_dir, cfg.checkpoint_step, target=target)
        params = payload["actor_params"]
        network = actor_net
    else:
        network = ActorCritic(env.action_space().n, encoder_cfg)
        rng, init_rng = jax.random.split(rng)
        base_params = network.init(init_rng, init_obs)
        if parameter_sharing:
            target = {"params": base_params}
            payload = load_checkpoint(ckpt_dir, cfg.checkpoint_step, target=target)
            params = payload["params"]
        else:
            if _has_agent_checkpoints(ckpt_dir):
                params = _load_agent_payloads(
                    ckpt_dir,
                    cfg.checkpoint_step,
                    num_agents,
                    "params",
                    base_params,
                )
            else:
                target = {"params": [base_params] * num_agents}
                payload = load_checkpoint(ckpt_dir, cfg.checkpoint_step, target=target)
                params = payload["params"]

    for episode in range(cfg.num_episodes):
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng)
        frames = []
        for step in range(cfg.max_steps):
            if cfg.render:
                frame = np.array(env.render(env_state))
                frames.append(frame)

            if isinstance(params, list):
                obs_batch = [obs[i][None, ...] for i in range(num_agents)]
                actions = []
                for i in range(num_agents):
                    rng, action_rng = jax.random.split(rng)
                    if algorithm == "mappo":
                        dist = network.apply(params[i], obs_batch[i])
                    else:
                        dist, _ = network.apply(params[i], obs_batch[i])
                    action = _select_action(dist, action_rng, cfg.deterministic)
                    actions.append(jnp.squeeze(action, axis=0))
                env_actions = actions
            else:
                obs_batch = flatten_obs(obs[None, ...])
                if algorithm == "mappo":
                    dist = network.apply(params, obs_batch)
                else:
                    dist, _ = network.apply(params, obs_batch)
                rng, action_rng = jax.random.split(rng)
                action = _select_action(dist, action_rng, cfg.deterministic)
                env_actions = unflatten_actions(action, 1, num_agents)

            rng, step_rng = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(step_rng, env_state, env_actions)
            if done["__all__"]:
                break

        if cfg.render and frames:
            output_dir = output_root
            gif_path = output_dir / f"episode_{episode}.gif"
            _save_gif(frames, gif_path, cfg.gif_fps)


if __name__ == "__main__":
    main()
