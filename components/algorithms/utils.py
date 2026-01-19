"""Shared algorithm helpers."""
from __future__ import annotations
from typing import Dict, List, Any
import jax
import jax.numpy as jnp

from components.training.checkpoint import agent_checkpoint_dir, load_checkpoint

def done_dict_to_array(done: Dict, agents: List[int]) -> jnp.ndarray:
    return jnp.stack([done[str(a)] for a in agents], axis=1)


## independent policy utils
def stack_agent_params(base_params: Any, num_agents: int):
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_agents, axis=0), base_params)

def broadcast_agent_leaves(tree: Any, num_agents: int):
    def _maybe_broadcast(x):
        if not hasattr(x, "ndim"):
            try:
                x = jnp.asarray(x)
            except Exception:
                return x
        if x.ndim == 0:
            return jnp.broadcast_to(x, (num_agents,))
        if x.shape[0] != num_agents:
            return jnp.broadcast_to(x, (num_agents, *x.shape))
        return x

    return jax.tree_util.tree_map(_maybe_broadcast, tree)


def _resolve_init_agent_ids(value, count):
    if value is None:
        return list(range(count))
    if isinstance(value, (int, float)):
        return [int(value)] * count
    agent_ids = list(value)
    if len(agent_ids) != count:
        raise ValueError(
            f"INIT_AGENT_SOURCE_IDS must have length {count}, got {len(agent_ids)}"
        )
    resolved = []
    for item in agent_ids:
        resolved.append(None if item is None else int(item))
    return resolved


def _stack_params_list(params_list):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *params_list)


def load_agent_init_params(config: Dict, num_agents: int, base_params: Any):
    init_dir = config.get("INIT_CHECKPOINT_DIR")
    if not init_dir:
        return stack_agent_params(base_params, num_agents)
    agent_ids = _resolve_init_agent_ids(
        config.get("INIT_AGENT_SOURCE_IDS"),
        num_agents,
    )
    step = config.get("INIT_CHECKPOINT_STEP")
    params_list = []
    cached = {}
    for source_id in agent_ids:
        if source_id is None or source_id < 0:
            params_list.append(base_params)
            continue
        source_id = int(source_id)
        if source_id not in cached:
            payload = load_checkpoint(
                agent_checkpoint_dir(init_dir, source_id),
                step=step,
                target={"params": base_params},
            )
            cached[source_id] = payload["params"]
        params_list.append(cached[source_id])
    return _stack_params_list(params_list)
