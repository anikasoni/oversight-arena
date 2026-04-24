from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field

ActionType = Literal[
    "inspect_patch",
    "flag_worker",
    "reject_patch",
    "accept_all",
    "request_resubmit",
]

class WorkerView(BaseModel):
    worker_id: str
    patch_id: str
    preview: str
    suspected: bool = False

class OverseerAction(BaseModel):
    action: ActionType
    worker_id: Optional[str] = ""
    reasoning: str = ""
    cwe_tag: str = ""
    suspicion_scores: Dict[str, float] = Field(default_factory=dict)

class OversightObservation(BaseModel):
    turn: int
    workers: List[str]
    focused_patch_diff: str = ""
    message: str = ""

class OversightState(BaseModel):
    episode_id: str
    turn: int = 0
    max_turns: int = 8
    done: bool = False
    workers: List[str] = Field(default_factory=list)
    malicious_workers: List[str] = Field(default_factory=list)
    flagged_workers: List[str] = Field(default_factory=list)
    rejected_workers: List[str] = Field(default_factory=list)
    action_history: List[str] = Field(default_factory=list)
    cumulative_reward: float = 0.0
