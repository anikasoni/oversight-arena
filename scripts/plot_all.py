"""
Consolidated plotting — generates every PNG the README embeds.

Run after training + LLM eval:
    python scripts/plot_all.py

Produces (all in results/):
    headline_bar.png            — top-of-README comparison bar chart
    eval_comparison.png         — baseline vs trained reward by difficulty
    ablation_reward_hacking.png — if eval_no_guards.csv exists
    success_plot.png            — scripted panel ablation (back-compat)
    reward_plot.png             — scripted panel ablation (back-compat)
"""
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")


def _csv(p: Path) -> List[dict]:
    if not p.exists(): return []
    with open(p) as f: return list(csv.DictReader(f))

def _json(p: Path) -> dict:
    if not p.exists(): return {}
    with open(p) as f: return json.load(f)

def _mean(rows, col): return float(np.mean([float(r[col]) for r in rows])) if rows else 0.0


# ── 1. Headline bar ──────────────────────────────────────────────────────────
def plot_headline():
    base    = _csv(RESULTS / "eval_baseline.csv")
    trained = _csv(RESULTS / "eval_grpo.csv")
    panel   = _json(RESULTS / "batch_gen0_panel_eval.json")

    labels, vals, colors = [], [], []
    if base:
        labels.append("Baseline LLM");       vals.append(_mean(base, "success_rate"))
        colors.append("#95A5A6")
    if panel:
        labels.append("Scripted 3-panel");   vals.append(float(panel.get("success_rate", 0)))
        colors.append("#3498DB")
    if trained:
        labels.append("GRPO (ours)");        vals.append(_mean(trained, "success_rate"))
        colors.append("#C0392B")

    if not labels:
        print("⚠️  No eval data for headline_bar. Run eval scripts first."); return

    fig, ax = plt.subplots(figsize=(6.4, 4))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Mean success rate (held-out)")
    ax.set_title("GRPO-trained overseer vs baselines")
    ax.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(RESULTS / "headline_bar.png", dpi=140, bbox_inches="tight"); plt.close()
    print("✅ headline_bar.png")


# ── 2. Reward vs difficulty ───────────────────────────────────────────────────
def plot_comparison():
    base = _csv(RESULTS / "eval_baseline.csv")
    tr   = _csv(RESULTS / "eval_grpo.csv")
    if not base or not tr:
        print("⚠️  Need eval_baseline.csv + eval_grpo.csv for comparison plot."); return

    xs = [float(r["difficulty"]) for r in base]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.errorbar(xs, [float(r["mean_reward"]) for r in base],
                yerr=[float(r["std_reward"]) for r in base],
                marker="o", capsize=4, color="#7F8C8D", linewidth=1.8,
                label="Baseline (untrained)")
    ax.errorbar(xs, [float(r["mean_reward"]) for r in tr],
                yerr=[float(r["std_reward"]) for r in tr],
                marker="s", capsize=4, color="#C0392B", linewidth=1.8,
                label="GRPO (ours)")
    ax.set_xlabel("Attack difficulty"); ax.set_ylabel("Mean episode reward")
    ax.set_title("Held-out reward vs attack difficulty")
    ax.grid(True, alpha=0.3); ax.legend(); ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(RESULTS / "eval_comparison.png", dpi=140, bbox_inches="tight"); plt.close()
    print("✅ eval_comparison.png")


# ── 3. Reward-hacking ablation ────────────────────────────────────────────────
def plot_ablation():
    base = _csv(RESULTS / "eval_baseline.csv")
    tr   = _csv(RESULTS / "eval_grpo.csv")
    ng   = _csv(RESULTS / "eval_no_guards.csv")
    if not (base and tr and ng):
        print("⚠️  Ablation needs eval_no_guards.csv too — skipping."); return

    xs = [float(r["difficulty"]) for r in base]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.errorbar(xs, [float(r["mean_reward"]) for r in base],
                yerr=[float(r["std_reward"]) for r in base],
                marker="o", capsize=4, color="#7F8C8D", linewidth=1.8, label="Baseline")
    ax.errorbar(xs, [float(r["mean_reward"]) for r in ng],
                yerr=[float(r["std_reward"]) for r in ng],
                marker="x", capsize=4, color="#F39C12", linewidth=1.8,
                label="GRPO — no anti-hack guards")
    ax.errorbar(xs, [float(r["mean_reward"]) for r in tr],
                yerr=[float(r["std_reward"]) for r in tr],
                marker="s", capsize=4, color="#C0392B", linewidth=1.8,
                label="GRPO + guards (ours)")
    ax.set_xlabel("Attack difficulty"); ax.set_ylabel("Mean episode reward")
    ax.set_title("Anti-hacking guards keep held-out quality up")
    ax.grid(True, alpha=0.3); ax.legend(); ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(RESULTS / "ablation_reward_hacking.png", dpi=140, bbox_inches="tight")
    plt.close(); print("✅ ablation_reward_hacking.png")


# ── 4. Back-compat scripted-panel plots ───────────────────────────────────────
def plot_scripted_ablation():
    single = _json(RESULTS / "single_overseer_eval.json")
    panel  = _json(RESULTS / "batch_gen0_panel_eval.json")
    if not (single and panel): return

    ov = single.get("single_overseers", {})
    labels = ["Precision", "Recall", "Speed", "Panel"]
    succ = [
        float(ov.get("precision",{}).get("success_rate", 0)),
        float(ov.get("recall",   {}).get("success_rate", 0)),
        float(ov.get("speed",    {}).get("success_rate", 0)),
        float(panel.get("success_rate", 0)),
    ]
    rew = [
        float(ov.get("precision",{}).get("average_reward", 0)),
        float(ov.get("recall",   {}).get("average_reward", 0)),
        float(ov.get("speed",    {}).get("average_reward", 0)),
        float(panel.get("average_reward", 0)),
    ]
    colors = ["#E74C3C", "#F1C40F", "#2ECC71", "#3498DB"]

    for fname, vals, title in [
        ("success_plot.png", succ, "Success Rate — scripted overseers"),
        ("reward_plot.png",  rew,  "Average Reward — scripted overseers"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, vals, color=colors, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}",
                    ha="center", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.1); ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(RESULTS / fname, dpi=140, bbox_inches="tight"); plt.close()
        print(f"✅ {fname}")


def main():
    RESULTS.mkdir(exist_ok=True)
    plot_headline()
    plot_comparison()
    plot_ablation()
    plot_scripted_ablation()

    for name in ("loss_curve.png", "reward_curve.png"):
        if not (RESULTS / name).exists():
            print(f"⚠️  {name} missing — run scripts/train_grpo.py to generate it.")


if __name__ == "__main__":
    main()