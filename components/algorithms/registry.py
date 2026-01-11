"""Registry for algorithm entrypoints."""
from typing import Callable, Dict

from . import ippo, mappo, svo


REGISTRY: Dict[str, Callable] = {
    "ippo": ippo.make_train,
    "mappo": mappo.make_train,
    "svo": svo.make_train,
}


def get_algorithm(name: str) -> Callable:
    if name not in REGISTRY:
        raise KeyError(f"Unknown algorithm '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]
