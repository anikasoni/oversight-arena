from fastapi import FastAPI
from oversight_arena.models import OverseerAction
from oversight_arena.server.environment import OversightArenaEnv

app = FastAPI(title="Oversight Arena")

env = OversightArenaEnv()

@app.get("/")
def root():
    return {"name": "Oversight Arena", "status": "ok"}

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: OverseerAction):
    return env.step(action)

@app.get("/state")
def state():
    return env.state

@app.post("/grader")
def grader():
    return env.grader()

@app.get("/schema")
def schema():
    return {
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
            "reasoning": "Potential subtle vulnerability",
            "cwe_tag": "CWE-476",
        },
    }
