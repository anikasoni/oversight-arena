import requests
from oversight_arena.overseer_panel import OverseerPanel

BASE = "http://localhost:8000"

panel = OverseerPanel()

print("Resetting environment...")
obs = requests.post(f"{BASE}/reset").json()
print(obs)

for step in range(1, 5):
    print(f"\n===== PANEL TURN {step} =====")

    decision = panel.act(obs)
    print("Individual votes:")
    for vote in decision["individual_votes"]:
        print(vote)

    print("\nFinal majority action:")
    print(decision["final_action"])

    result = requests.post(f"{BASE}/step", json=decision["final_action"]).json()
    print("\nEnv result:")
    print(result)

    obs = result["observation"]

    if result["done"]:
        break

print("\nFinal grader:")
print(requests.post(f"{BASE}/grader").json())
