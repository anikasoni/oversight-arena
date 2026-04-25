"""FastAPI HTTP wrapper around the Oversight Arena env.

Endpoints:
  GET  /             - liveness / metadata
  GET  /health       - health probe (used by HF Space / validators)
  GET  /schema       - action + observation schema
  GET  /grader       - terminal grader result (also available as POST)
  POST /grader       - (kept for backward compat with your earlier scripts)
  POST /reset        - reset env with optional seed & difficulty
  POST /step         - step env with an OverseerAction
  GET  /state        - current state as JSON
"""
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

from oversight_arena.models import OverseerAction
from oversight_arena.server.environment import OversightArenaEnv

app = FastAPI(
    title="Oversight Arena",
    description="OpenEnv environment for training scalable oversight overseers.",
    version="0.1.0",
)

env = OversightArenaEnv(seed=0, difficulty=0.5)


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    difficulty: Optional[float] = None


@app.get("/")
def root():
    return {
        "name": "Oversight Arena",
        "version": "0.1.0",
        "status": "ok",
        "endpoints": [
            "/health", "/schema", "/reset", "/step",
            "/state", "/grader",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    obs = env.reset(seed=req.seed, difficulty=req.difficulty)
    return {"observation": obs.model_dump(), "state": env.state_dict()}


@app.post("/step")
def step(action: OverseerAction):
    return env.step(action)


@app.get("/state")
def state():
    return env.state_dict()


@app.get("/grader")
def grader_get():
    return env.grader()


@app.post("/grader")
def grader_post():
    return env.grader()


@app.get("/schema")
def schema():
    return {
        "action_space": {
            "type": "structured",
            "actions": [
                "inspect_patch",
                "flag_worker",
                "reject_patch",
                "accept_all",
                "request_resubmit",
            ],
            "example_action": {
                "action": "flag_worker",
                "worker_id": "W2",
                "reasoning": "Mutable default argument pattern detected",
                "cwe_tag": "CWE-1188",
            },
        },
        "observation_space": {
            "turn": "int",
            "workers": "list[str]",
            "focused_patch_diff": "str (concatenated unified diffs)",
            "message": "str",
        },
        "reward": {
            "metric": "f1 + early_bonus - fp_penalty (with anti-hack guards on terminal grader)",
            "range": "[0, 1]",
        },
        "difficulty": {
            "type": "float",
            "range": [0.0, 1.0],
            "description": "0.0 = obvious sabotage; 1.0 = subtle with escalation",
        },
    }