"""
Version-safe HTTP client for Oversight Arena.

This avoids importing StepResult from openenv.core because openenv-core 0.2.x
does not export StepResult from that location on this machine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

import requests

from .models import OverseerAction


@dataclass
class StepResult:
    observation: Any
    reward: float
    done: bool
    info: Dict[str, Any]


def _to_payload(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    raise TypeError(f"Cannot serialize action of type {type(obj)}")


def _to_namespace(x: Any) -> Any:
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_to_namespace(v) for v in x]
    return x


class _SyncOversightArenaEnv:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self) -> None:
        self.session.close()

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = self.session.post(url, json=payload, timeout=self.timeout)

        # Some OpenEnv wrappers expect {"action": ...}; others expect direct action fields.
        if r.status_code in (400, 422) and path == "/step":
            r = self.session.post(url, json={"action": payload}, timeout=self.timeout)

        r.raise_for_status()
        return r.json()

    def _get(self, path: str) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}{path}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _result(self, payload: Dict[str, Any]) -> StepResult:
        obs = payload.get("observation", payload)
        reward = payload.get("reward", obs.get("reward", 0.0) if isinstance(obs, dict) else 0.0)
        done = payload.get("done", obs.get("done", False) if isinstance(obs, dict) else False)

        info = payload.get("info", {})
        if "state" in payload:
            info = {**info, "state": payload["state"]}

        return StepResult(
            observation=_to_namespace(obs),
            reward=float(reward or 0.0),
            done=bool(done),
            info=info,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: Optional[float] = None,
        **kwargs: Any,
    ) -> StepResult:
        payload: Dict[str, Any] = dict(kwargs)
        if seed is not None:
            payload["seed"] = seed
        if difficulty is not None:
            payload["difficulty"] = difficulty
        return self._result(self._post("/reset", payload))

    def step(self, action: Dict[str, Any] | OverseerAction) -> StepResult:
        return self._result(self._post("/step", _to_payload(action)))

    def state(self) -> Dict[str, Any]:
        return self._get("/state")

    def grader(self) -> Dict[str, Any]:
        return self._get("/grader")


class OversightArenaEnv:
    """
    Simple sync-compatible client.

    Usage:
        with OversightArenaEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset()
            result = env.step(OverseerAction(action="flag_worker", worker_id="W2"))
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0, **_: Any):
        self.base_url = base_url
        self.timeout = timeout
        self._sync = _SyncOversightArenaEnv(base_url=base_url, timeout=timeout)

    def sync(self) -> _SyncOversightArenaEnv:
        return _SyncOversightArenaEnv(base_url=self.base_url, timeout=self.timeout)

    def reset(self, *args: Any, **kwargs: Any) -> StepResult:
        return self._sync.reset(*args, **kwargs)

    def step(self, action: Dict[str, Any] | OverseerAction) -> StepResult:
        return self._sync.step(action)

    def state(self) -> Dict[str, Any]:
        return self._sync.state()

    def grader(self) -> Dict[str, Any]:
        return self._sync.grader()

    def close(self) -> None:
        self._sync.close()
