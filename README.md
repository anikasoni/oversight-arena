# Oversight Arena

Scalable Oversight Arena is a multi-agent AI safety benchmark where overseer agents detect misaligned worker agents submitting subtle vulnerable code patches.

## Current Features

- 3 worker agents per episode
- 1 adaptive malicious worker
- CVE/CWE-style sabotage patterns
- Objective F1-based reward
- FastAPI endpoints: `/reset`, `/step`, `/state`, `/grader`, `/schema`
- Scripted Gen 0 baseline

## Run

```bash
uvicorn oversight_arena.server.app:app --host 0.0.0.0 --port 8000 --reload
## Gen 0 Result

Three-overseer panel baseline:

{
  "final_reward": 0.925,
  "success": true
}

Key behavior: majority vote detected the malicious worker, then stopped to avoid false positives.
