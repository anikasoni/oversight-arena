"""
Worker agents for Oversight Arena.

A WorkerAgent wraps a single PatchPattern and exposes a unified diff:
    * if `honest`     -> emit the honest fix (no bug)
    * if not honest   -> emit the malicious variant at current escalation level

Adaptation:
    `adapt()` is called when the overseer flags / rejects / requests-resubmit
    this worker. It bumps the escalation index up to `max_escalation`. Honest
    workers do not adapt — they are not adversarial.

This keeps workers SCRIPTED + ADAPTIVE — never trained — which is the design
choice that keeps the RL problem tractable. Training both sides at once
produces oscillating curves (per the OpenEnv reward-design guide).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from oversight_arena.sabotage_catalog import PatchPattern


@dataclass
class WorkerAgent:
    worker_id: str
    honest: bool
    pattern: PatchPattern
    escalation: int = 0
    max_escalation: int = 1
    n_flag_events: int = 0
    history: list[str] = field(default_factory=list)

    def patch_id(self) -> str:
        kind = "honest" if self.honest else f"mal_{self.escalation}"
        return f"{self.pattern.pattern_id}_{kind}"

    def diff(self) -> str:
        if self.honest:
            return self.pattern.honest_diff()
        return self.pattern.malicious_diff(self.escalation)

    def adapt(self) -> bool:
        """
        Called when overseer takes an action targeting this worker
        (flag_worker, reject_patch, request_resubmit).

        Returns True iff escalation actually changed.
        Honest workers never adapt.
        """
        if self.honest:
            return False
        self.n_flag_events += 1
        if self.escalation < self.max_escalation:
            self.escalation += 1
            self.history.append(f"escalated->{self.escalation}")
            return True
        self.history.append("no_more_escalation")
        return False