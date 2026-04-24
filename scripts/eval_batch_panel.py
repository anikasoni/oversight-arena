import json
import requests
from statistics import mean
from oversight_arena.overseer_panel import OverseerPanel

BASE = "http://localhost:8000"
N = 20

results = []

for i in range(N):
    panel = OverseerPanel()

    obs = requests.post(f"{BASE}/reset").json()

    for _ in range(5):
        decision = panel.act(obs)
        result = requests.post(f"{BASE}/step", json=decision["final_action"]).json()
        obs = result["observation"]

        if result["done"]:
            break

    grade = requests.post(f"{BASE}/grader").json()
    results.append(grade)

success_rate = mean([1 if r["success"] else 0 for r in results])
avg_reward = mean([r["final_reward"] for r in results])

summary = {
    "experiment": "Batch Gen 0 Three-Overseer Panel Evaluation",
    "episodes": N,
    "average_reward": round(avg_reward, 4),
    "success_rate": round(success_rate, 4),
    "results": results,
}

print(json.dumps(summary, indent=2))

with open("results/batch_gen0_panel_eval.json", "w") as f:
    json.dump(summary, f, indent=2)
