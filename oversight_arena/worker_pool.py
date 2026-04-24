from dataclasses import dataclass
from oversight_arena.sabotage_catalog import PatchPattern

@dataclass
class WorkerAgent:
    worker_id: str
    honest: bool
    pattern: PatchPattern
    escalation: int = 0
    max_escalation: int = 1

    def patch_id(self):
        return f"{self.pattern.pattern_id}_{'honest' if self.honest else 'malicious'}_{self.escalation}"

    def diff(self):
        return self.pattern.honest_diff() if self.honest else self.pattern.malicious_diff(self.escalation)

    def adapt(self):
        if not self.honest and self.escalation < self.max_escalation:
            self.escalation += 1
