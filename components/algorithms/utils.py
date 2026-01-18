"""Shared algorithm helpers."""
from __future__ import annotations
from typing import Dict, List
import jax
import jax.numpy as jnp
from __future__ import annotations
from typing import Any

def done_dict_to_array(done: Dict, agents: List[int]) -> jnp.ndarray:
    return jnp.stack([done[str(a)] for a in agents], axis=1)

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