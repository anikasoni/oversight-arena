from __future__ import annotations
from env.oversight_arena import OversightArena, EpisodeConfig


class OversightArenaEnv:
    """OpenEnv-compatible wrapper around OversightArena.

    Exposes reset(), step(), state() interface per openenv-core spec.
    """

    def __init__(self, config: EpisodeConfig | None = None) -> None:
        self._arena = OversightArena(config)
        self._current_state = None

    def reset(self) -> dict:
        """Reset environment and return initial observation."""
        self._current_state = self._arena.reset()
        return {
            "observation": self._arena._get_observation(self._current_state),
            "state": self._current_state,
        }

    def step(self, action: dict) -> dict:
        """Take one step. action must follow PRD 3.3 JSON schema."""
        state, reward, done, info = self._arena.step(action)
        self._current_state = state
        return {
            "observation": self._arena._get_observation(state),
            "reward": reward,
            "done": done,
            "info": info,
            "state": state,
        }

    def state(self) -> dict:
        """Return current environment state per openenv-core spec."""
        if self._current_state is None:
            raise RuntimeError("Call reset() before state()")
        return {"state": self._current_state}