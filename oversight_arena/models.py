"""
Oversight Arena data models.

OpenEnv's Action / Observation / State are PYDANTIC BaseModel classes.
We must use Pydantic syntax (not @dataclass) when openenv-core is installed,
otherwise @dataclass strips inherited Pydantic fields and OversightState()
breaks with "missing required positional arguments".

When openenv-core is NOT installed, we fall back to plain dataclasses for
local testing.
"""
from __future__ import annotations
from typing import List

try:
    from openenv.core.env_server import Action, Observation, State
    _OPENENV = True
except ImportError:
    _OPENENV = False


if _OPENENV:
    # ---- Pydantic v2 path (openenv-core installed) ----
    from pydantic import Field

    class OverseerAction(Action):
        action: str = "inspect_patch"
        worker_id: str = ""
        reasoning: str = ""
        cwe_tag: str = ""

    class OversightObservation(Observation):
        # done: bool and reward: Optional[float] are INHERITED from Observation
        turn: int = 0
        workers: List[str] = Field(default_factory=list)
        focused_patch_diff: str = ""
        message: str = ""

    class OversightState(State):
        # episode_id and step_count are INHERITED from State
        max_turns: int = 8
        done: bool = False
        workers: List[str] = Field(default_factory=list)
        malicious_workers: List[str] = Field(default_factory=list)
        flagged_workers: List[str] = Field(default_factory=list)
        rejected_workers: List[str] = Field(default_factory=list)
        action_history: List[str] = Field(default_factory=list)
        cumulative_reward: float = 0.0
        difficulty: float = 0.5

else:
    # ---- Plain dataclass fallback (no openenv-core) ----
    from dataclasses import dataclass, field

    @dataclass
    class OverseerAction:
        action: str = "inspect_patch"
        worker_id: str = ""
        reasoning: str = ""
        cwe_tag: str = ""

    @dataclass
    class OversightObservation:
        turn: int = 0
        workers: List[str] = field(default_factory=list)
        focused_patch_diff: str = ""
        message: str = ""
        reward: float = 0.0
        done: bool = False

    @dataclass
    class OversightState:
        episode_id: str = ""
        step_count: int = 0
        max_turns: int = 8
        done: bool = False
        workers: List[str] = field(default_factory=list)
        malicious_workers: List[str] = field(default_factory=list)
        flagged_workers: List[str] = field(default_factory=list)
        rejected_workers: List[str] = field(default_factory=list)
        action_history: List[str] = field(default_factory=list)
        cumulative_reward: float = 0.0
        difficulty: float = 0.5