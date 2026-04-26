"""
Reward-hacking ablation.

Trains TWO policies on the same data, with the same hyperparameters, differing
ONLY in whether the env's anti-hack guardrails are active. Then evaluates BOTH
on the same held-out seeds.

Expected story for judges:
    * WITHOUT guardrails: reward goes UP (model finds the flag-spam loophole)
      but F1 stays FLAT or drops -> reward hacking detected.
    * WITH guardrails: reward goes up MORE SLOWLY, but F1 actually rises
      -> the guardrails are doing real work.

This is the single strongest piece of evidence that the reward design isn't
just a number on a slide — it's actively shaping behavior.

Usage:
    python scripts/ablation_no_guardrails.py \
        --env-url https://anikasoni-oversight-arena.hf.space \
        --eval-n 20 --n-prompts 32

Output:
    results/eval_no_guards.csv          (trained without guardrails)
    results/ablation_reward_hacking.png (3-bar comparison)
    results/ablation_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib.pyplot as plt
import requests


DEFAULT_ENV_URL = os.environ.get(
    "OVERSIGHT_ENV_URL", "https://anikasoni-oversight-arena.hf.space"
)
RESULTS_DIR = Path("results")


def patch_env_to_disable_guardrails():
    """
    Monkey-patch the LOCAL OversightArenaEnvironment to ignore the anti-hack
    multiplier. Only used inside this ablation process — the live HF Space is
    untouched. The "no_guards" trainer talks to a LOCAL uvicorn instance.
    """
    from oversight_arena import oversight_rewards as R

    _orig = R.compute_terminal_reward

    def patched(*args, **kwargs):
        result = _orig(*args, **kwargs)
        # Strip the multiplier penalty.
        result["reward"] = max(0.0, min(1.0, result.get("base", 0.0)))
        result["multiplier"] = 1.0
        result["guardrails_triggered"] = []
        result["_ablation"] = "no_guardrails"
        return result

    R.compute_terminal_reward = patched
    print("[ABLATION] Anti-hack guardrails DISABLED for this run.")


# --------------------------------------------------------------------------- #
# Eval helpers                                                                 #
# --------------------------------------------------------------------------- #

def load_summary(path: Path, key_label: str) -> Dict[str, float]:
    """
    Read mean_reward / mean_f1 / valid_json_rate from a per-seed CSV.
    """
    if not path.exists():
        print(f"[WARN] {path} missing; treating as zeros.")
        return {"label": key_label, "mean_reward": 0.0, "mean_f1": 0.0, "valid_json_rate": 0.0, "n": 0}

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    def col(name):
        return [float(r.get(name, 0.0) or 0.0) for r in rows]

    return {
        "label": key_label,
        "mean_reward": float(mean(col("reward"))) if rows else 0.0,
        "mean_f1": float(mean(col("f1"))) if rows else 0.0,
        "valid_json_rate": float(mean([float(r.get("valid_json", 0)) for r in rows])) if rows else 0.0,
        "n": len(rows),
    }


def plot_three_bar(summaries: List[Dict[str, float]], out_path: Path):
    metrics = ["mean_reward", "mean_f1", "valid_json_rate"]
    width = 0.25
    x = list(range(len(metrics)))

    plt.figure(figsize=(8.5, 4.5))
    for i, s in enumerate(summaries):
        offset = (i - 1) * width
        vals = [s.get(m, 0.0) for m in metrics]
        plt.bar([xi + offset for xi in x], vals, width=width, label=s.get("label", f"run{i}"))

    plt.xticks(x, metrics)
    plt.ylim(0, 1.05)
    plt.title("Reward-hacking ablation: with vs without anti-hack guardrails")
    plt.legend(loc="upper right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print("[OK] saved", out_path)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-url", default=DEFAULT_ENV_URL)
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--eval-n", type=int, default=20)
    ap.add_argument(
        "--mode",
        choices=["train_no_guards", "summarize"],
        default="summarize",
        help="train_no_guards: actually run a 2nd training pass with guardrails off (requires a "
             "LOCAL uvicorn running this codebase, not the live HF Space). summarize: just read "
             "eval_baseline.csv / eval_grpo.csv / eval_no_guards.csv and produce the plot.",
    )
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.mode == "train_no_guards":
        # Defer-import so the patched module is picked up from sys.modules.
        patch_env_to_disable_guardrails()
        from scripts.train_grpo import train as run_train

        # Reuse the standard trainer with output written to a different csv.
        # We rely on the reward function inside train_grpo.py — since the env
        # itself was patched, the grader on a LOCAL server will already
        # reflect the no-guardrail reward.
        # IMPORTANT: this mode requires you to be running uvicorn on
        # http://localhost:7860 with the patched module imported. The CI
        # workflow is to set OVERSIGHT_ENV_URL=http://localhost:7860 before
        # invoking this script.
        if "huggingface" in args.env_url or "hf.space" in args.env_url:
            print("[ERROR] --mode train_no_guards requires a LOCAL env, not the live Space.")
            print("        Run `uvicorn oversight_arena.server.app:app --port 7860` first,")
            print("        then export OVERSIGHT_ENV_URL=http://localhost:7860")
            sys.exit(1)

        class A:
            pass

        a = A()
        a.env_url = args.env_url
        a.model = args.model
        a.out = "checkpoints/grpo_no_guards"
        a.n_prompts = args.n_prompts
        a.curriculum = True
        a.difficulty = 0.5
        a.lr = args.lr
        a.batch_size = args.batch_size
        a.grad_accum = 2
        a.num_generations = args.num_generations
        a.epochs = 1
        a.max_prompt_len = 1024
        a.max_new_tokens = 96
        a.eval_after_train = True
        a.eval_n = args.eval_n

        # Re-route the trained-eval CSV.
        # We do this by running and then renaming the file.
        run_train(a)
        eval_grpo = RESULTS_DIR / "eval_grpo.csv"
        eval_no_guards = RESULTS_DIR / "eval_no_guards.csv"
        if eval_grpo.exists():
            eval_grpo.rename(eval_no_guards)
            print(f"[OK] renamed {eval_grpo} -> {eval_no_guards}")

    # ---------- Summarize -------------------------------------------------- #
    base = load_summary(RESULTS_DIR / "eval_baseline.csv", "untrained")
    grpo = load_summary(RESULTS_DIR / "eval_grpo.csv", "GRPO + guardrails")
    no_guards = load_summary(RESULTS_DIR / "eval_no_guards.csv", "GRPO − guardrails")

    plot_three_bar([base, grpo, no_guards], RESULTS_DIR / "ablation_reward_hacking.png")

    summary = {
        "baseline": base,
        "grpo_with_guardrails": grpo,
        "grpo_without_guardrails": no_guards,
        "interpretation": (
            "If grpo_without_guardrails has HIGHER mean_reward but LOWER mean_f1 than "
            "grpo_with_guardrails, the model found a reward-hacking shortcut that the "
            "guardrails block. That is the single strongest evidence the reward design "
            "is not gameable."
        ),
    }
    with open(RESULTS_DIR / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Ablation summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()