import json
import matplotlib.pyplot as plt

# Load results
with open("results/gen0_baseline.json") as f:
    panel = json.load(f)

with open("results/single_overseer_ablation.json") as f:
    single = json.load(f)

# Extract values
labels = ["Single-Precision", "Single-Recall", "Single-Speed", "Panel"]

success = [
    0.0,
    single["single_overseers"]["recall"]["success_rate"],
    single["single_overseers"]["speed"]["success_rate"],
    panel["success_rate"],
]

reward = [
    0.0,
    single["single_overseers"]["recall"]["average_reward"],
    single["single_overseers"]["speed"]["average_reward"],
    panel["average_reward"],
]

x = range(len(labels))

plt.figure()
plt.bar(x, success)
plt.xticks(x, labels, rotation=20)
plt.title("Success Rate Comparison")
plt.tight_layout()
plt.savefig("results/success_plot.png")

plt.figure()
plt.bar(x, reward)
plt.xticks(x, labels, rotation=20)
plt.title("Average Reward Comparison")
plt.tight_layout()
plt.savefig("results/reward_plot.png")

print("Plots saved in results/")
