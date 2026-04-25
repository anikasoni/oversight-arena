"""
OpenEnv-shaped class wrapper around OversightArenaEnv.

This exists so automated validators that look for a Gym-style class with
`reset() / step() / state()` signatures can import one symbol:

    from oversight_arena import OversightArenaOpenEnv

Internally it delegates to the existing HTTP-backed environment so the same
code path is exercised whether you use the Space or the in-process class.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from oversight_arena.models import OverseerAction
from oversight_arena.server.environment import OversightArenaEnv


@dataclass
class StepResult:
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": self.observation,
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
        }


class OversightArenaOpenEnv:
    """
    Minimal OpenEnv-style interface.

    Actions are dicts matching OverseerAction, e.g.:
        {"action": "flag_worker", "worker_id": "W2", "reasoning": "...", "cwe_tag": "CWE-476"}
        {"action": "accept_all"}
        {"action": "inspect_patch", "worker_id": "W1"}
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: Optional[int] = None, difficulty: float = 0.5):
        self._env = OversightArenaEnv(seed=seed, difficulty=difficulty)

    # -------- Gym-style API --------
    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: Optional[float] = None,
    ) -> Dict[str, Any]:
        obs = self._env.reset(seed=seed, difficulty=difficulty)
        return obs.model_dump()

    def step(self, action: Dict[str, Any] | OverseerAction) -> StepResult:
        if isinstance(action, dict):
            action = OverseerAction(**action)
        elif not isinstance(action, OverseerAction):
            raise TypeError(
                f"action must be dict or OverseerAction, got {type(action)}"
            )
        raw = self._env.step(action)
        return StepResult(
            observation=raw["observation"].model_dump()
                if hasattr(raw["observation"], "model_dump")
                else raw["observation"],
            reward=float(raw["reward"]),
            done=bool(raw["done"]),
            info={"state": raw["state"]},
        )

    def state(self) -> Dict[str, Any]:
        return self._env.state_dict()

    def grader(self) -> Dict[str, Any]:
        return self._env.grader()

    @property
    def difficulty(self) -> float:
        return self._env.difficulty

    def close(self) -> None:
        self._env.state = None