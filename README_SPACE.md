---
title: Oversight Arena
emoji: 🛡️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: OpenEnv RL environment for training scalable-oversight overseers
---

# 🛡️ Oversight Arena — Live Environment

Multi-agent AI safety benchmark. An overseer agent reviews code patches from
3 worker agents per episode; malicious workers submit subtle CWE-style bugs
(null deref, mutable default arg, off-by-one, etc.). The overseer must flag
them without over-flagging honest workers.

> 📦 **Code, training scripts, and writeup:** https://github.com/anikasoni/oversight-arena

## Try it

Once the Space is running, open `/docs` for interactive Swagger UI.

Quick probes:

```bash
SPACE=https://<your-username>-oversight-arena.hf.space

curl $SPACE/health
curl $SPACE/schema
curl -X POST $SPACE/reset -H 'content-type: application/json' \
    -d '{"seed": 42, "difficulty": 0.5}'
curl -X POST $SPACE/step  -H 'content-type: application/json' \
    -d '{"action": "flag_worker", "worker_id": "W2", "reasoning": "mutable default"}'
curl $SPACE/grader
```

## Endpoints

| Method | Path       | Purpose                          |
|--------|------------|----------------------------------|
| GET    | `/`        | metadata                         |
| GET    | `/health`  | health probe                     |
| GET    | `/schema`  | action & observation schema      |
| POST   | `/reset`   | start episode (seed, difficulty) |
| POST   | `/step`    | take an overseer action          |
| GET    | `/state`   | current JSON state               |
| GET    | `/grader`  | terminal reward with anti-hack guards |

Built for the Meta × PyTorch OpenEnv Hackathon 2026.