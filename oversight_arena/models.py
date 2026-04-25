"""
Pydantic / OpenEnv data models for Oversight Arena.

OpenEnv expects three concrete classes derived from its base Action / Observation
/ State. We fall back to plain Pydantic if openenv-core is not installed (so tests
and CI can run without the framework).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from openenv.core.types import Action as _BaseAction
    from openenv.core.types import Observation as _BaseObservation
    from openenv.core.types import State as _BaseState
except ImportError:
    from pydantic import BaseModel as _PydBase

    class _BaseAction(_PydBase):  # type: ignore
        pass

    class _BaseObservation(_PydBase):  # type: ignore
        pass

    class _BaseState(_PydBase):  # type: ignore
        pass


# --------------------------------------------------------------------------- #
# Action                                                                       #
# --------------------------------------------------------------------------- #

class OverseerAction(_BaseAction):
    """One overseer decision per step."""

    action: str = "inspect_patch"            # inspect_patch | flag_worker | reject_patch | accept_all | request_resubmit
    worker_id: Optional[str] = None          # "W1" | "W2" | "W3" | None
    reasoning: str = ""
    cwe_tag: str = ""
    suspicion_scores: Optional[Dict[str, float]] = None


# --------------------------------------------------------------------------- #
# Observation                                                                  #
# --------------------------------------------------------------------------- #

class OversightObservation(_BaseObservation):
    """What the overseer sees on each step."""

    turn: int = 0
    workers: List[str] = []
    focused_patch_diff: str = ""
    message: str = ""
    reward: float = 0.0
    done: bool = False


# --------------------------------------------------------------------------- #
# State (server's internal record; partial fields in observation)              #
# --------------------------------------------------------------------------- #

class OversightState(_BaseState):
    """Server-side state. Visible to the grader, NOT to the model."""

    episode_id: Optional[str] = None
    step_count: int = 0
    max_turns: int = 8
    done: bool = False

    workers: List[str] = []
    malicious_workers: List[str] = []     # ground truth, hidden from model
    flagged_workers: List[str] = []
    rejected_workers: List[str] = []
    first_flag_turn: Dict[str, int] = {}
    action_history: List[str] = []
    suspicion_log: List[Dict[str, Any]] = []

    cumulative_reward: float = 0.0
    terminal_reward: float = 0.0
    terminal_breakdown: Dict[str, Any] = {}
    difficulty: float = 0.5