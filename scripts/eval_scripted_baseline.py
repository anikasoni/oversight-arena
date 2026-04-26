import requests

BASE = "http://localhost:8000"

print("Resetting env...")
r = requests.post(f"{BASE}/reset")
print(r.json())

print("\nFlagging W2 as scripted baseline...")
action = {
    "action": "flag_worker",
    "worker_id": "W2",
    "reasoning": "Scripted baseline suspects W2",
    "cwe_tag": "CWE-476",
}
r = requests.post(f"{BASE}/step", json=action)
print(r.json())

print("\nGrader...")
r = requests.post(f"{BASE}/grader")
print(r.json())
