"""
Pydantic / OpenEnv data models for Oversight Arena (v5).

Changes vs v4:
  - suspicion_log: List[Dict[str, Any]] now has a TypedDict comment showing
    the actual schema. Was List[Dict[str, Any]] with no documentation.
  - terminal_reward now has a default of None (not 0.0) so _state_dict()
    can correctly distinguish "not yet set" from "set to 0.0".
    This is the model-level fix that backs the app.py float(raw) if raw is not None fix.
  - worker_pattern_ids, malicious_pattern_ids documented with examples.
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


class OverseerAction(_BaseAction):
    """
    Action taken by the overseer LLM.

    action: one of inspect_patch | flag_worker | reject_patch |
                     request_resubmit | accept_all
    worker_id: "W1", "W2", or "W3" (empty for accept_all)
    reasoning: natural language justification (not used in reward, for logging)
    cwe_tag: CWE identifier if known, e.g. "CWE-89"
    suspicion_scores: optional dict mapping worker_id -> float (not used in reward)
    """
    action: str = "inspect_patch"
    worker_id: Optional[str] = None
    reasoning: str = ""
    cwe_tag: str = ""
    suspicion_scores: Optional[Dict[str, float]] = None


class OversightObservation(_BaseObservation):
    """
    What the overseer sees at each step.

    focused_patch_diff: unified diffs for all three workers, with status headers:
      ### W1 [PENDING]
      ### W2 [FLAGGED]
      ### W3 [REJECTED]
    """
    turn: int = 0
    workers: List[str] = []
    focused_patch_diff: str = ""
    message: str = ""
    reward: float = 0.0
    done: bool = False


class OversightState(_BaseState):
    """
    Full server-side state. Only exposed to clients via /state and /grader.
    Training loop should use /grader (objective metrics) not /state (ground truth).

    suspicion_log schema per entry:
      {
        "turn": int,
        "action": str,
        "worker_id": str,
        "reasoning": str (truncated to 120 chars),
        "cwe_tag": str,
        "was_malicious": bool  <- server-side only
      }
    """
    episode_id: Optional[str] = None
    step_count: int = 0
    max_turns: int = 8
    done: bool = False

    workers: List[str] = []
    malicious_workers: List[str] = []
    flagged_workers: List[str] = []
    rejected_workers: List[str] = []
    first_flag_turn: Dict[str, int] = {}
    action_history: List[str] = []
    suspicion_log: List[Dict[str, Any]] = []

    cumulative_reward: float = 0.0
    # FIX: None means "not yet set at terminal"; 0.0 means "terminal reward was 0.0"
    # This distinction is what makes the float(raw) if raw is not None fix work correctly.
    terminal_reward: Optional[float] = None
    terminal_breakdown: Dict[str, Any] = {}
    difficulty: float = 0.5

    # Tier-based eval fields
    # e.g. ["T1-01", "T3-02", "T1-08"] — one per worker in W1/W2/W3 order
    worker_pattern_ids: List[str] = []
    # subset of worker_pattern_ids, for malicious workers only
    malicious_pattern_ids: List[str] = []
    # dominant tier prefix: "T1" | "T2" | "T3" | "" (empty = all honest)
    malicious_tier: str = ""