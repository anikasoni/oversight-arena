"""
Oversight Arena — OpenEnv environment for scalable oversight research.
"""
from .models import OverseerAction, OversightObservation, OversightState
from .env_openenv import OversightArenaOpenEnv

__all__ = [
    "OversightArenaEnv",
    "OversightArenaOpenEnv",
    "OverseerAction",
    "OversightObservation",
    "OversightState",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "OversightArenaEnv":
        from .client import OversightArenaEnv
        return OversightArenaEnv
    raise AttributeError(f"module 'oversight_arena' has no attribute {name!r}")
