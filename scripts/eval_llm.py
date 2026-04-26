"""
Evaluate an LLM overseer checkpoint on held-out seeds across difficulty levels.

Usage:
    # Baseline (no adapter)
    python scripts/eval_llm.py --ckpt '' --label baseline --n-episodes 20

    # Your trained LoRA
    python scripts/eval_llm.py --ckpt checkpoints/grpo --label grpo --n-episodes 20

Outputs:
    results/eval_<label>.csv         — per-difficulty aggregates
    results/eval_<label>_episodes.json — per-episode detail
    results/eval_<label>.png          — reward-vs-difficulty plot
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from oversight_arena import OversightArenaOpenEnv
from scripts.train_grpo import (
    SYSTEM_PROMPT,
    parse_completion,
    build_prompt,
)


BASE_MODEL_DEFAULT = "unsloth/Qwen2.5-1.5B-Instruct"


def load_model(ckpt: str, base: str = BASE_MODEL_DEFAULT):
    from unsloth import FastLanguageModel
    target = ckpt if ckpt else base
    print(f"Loading model: {target}")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=target,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tok


def run_episode(model, tokenizer, difficulty: float, seed: int,
                max_turns: int = 5) -> Dict[str, Any]:
    env = OversightArenaOpenEnv(seed=seed, difficulty=difficulty)
    obs = env.reset(seed=seed, difficulty=difficulty)
    total_reward = 0.0
    format_violations = 0
    for t in range(max_turns):
        prompt = build_prompt(obs, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,     # deterministic for held-out eval
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        parsed = parse_completion(text)
        if parsed is None:
            format_violations += 1
            parsed = {"action": "accept_all"}
        step = env.step(parsed)
        total_reward += step.reward
        obs = step.observation
        if step.done:
            break

    grade = env.grader()
    return {
        "seed": seed,
        "difficulty": difficulty,
        "format_violations": format_violations,
        "turns": t + 1,
        "final_reward": grade["final_reward"],
        "success": grade["success"],
        "malicious": grade["malicious_workers"],
        "flagged": grade["flagged_workers"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="LoRA adapter dir; empty = baseline")
    ap.add_argument("--base", default=BASE_MODEL_DEFAULT)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--difficulties", default="0.2,0.4,0.6,0.8")
    args = ap.parse_args()

    results_dir = Path(args.out_dir)
    results_dir.mkdir(exist_ok=True)
    diffs = [float(x) for x in args.difficulties.split(",")]

    model, tok = load_model(args.ckpt, args.base)

    all_eps: List[Dict[str, Any]] = []
    agg: List[Dict[str, Any]] = []

    for d in diffs:
        print(f"\n=== Difficulty {d} ===")
        eps: List[Dict[str, Any]] = []
        for i in range(args.n_episodes):
            seed = 100_000 + int(d * 10_000) + i
            ep = run_episode(model, tok, d, seed)
            eps.append(ep); all_eps.append(ep)
            print(f"  seed={seed}  reward={ep['final_reward']:.3f}  "
                  f"success={ep['success']}  fmt_viol={ep['format_violations']}")
        rewards = [e["final_reward"] for e in eps]
        successes = [1.0 if e["success"] else 0.0 for e in eps]
        agg.append({
            "difficulty": d,
            "n_episodes": len(eps),
            "mean_reward": float(np.mean(rewards)),
            "std_reward":  float(np.std(rewards)),
            "success_rate": float(np.mean(successes)),
            "mean_format_violations": float(np.mean([e["format_violations"] for e in eps])),
        })

    # Write CSV
    csv_path = results_dir / f"eval_{args.label}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg[0].keys())
        w.writeheader()
        w.writerows(agg)
    print(f"✅ Wrote {csv_path}")

    # Write per-episode JSON
    with open(results_dir / f"eval_{args.label}_episodes.json", "w") as f:
        json.dump(all_eps, f, indent=2)

    # Plot
    import matplotlib.pyplot as plt
    xs = [r["difficulty"] for r in agg]
    ys = [r["mean_reward"] for r in agg]
    es = [r["std_reward"] for r in agg]
    plt.figure(figsize=(6.2, 4))
    plt.errorbar(xs, ys, yerr=es, marker="o", capsize=4,
                 color="#C0392B", linewidth=2)
    plt.xlabel("Attack difficulty"); plt.ylabel("Mean terminal reward")
    plt.title(f"Held-out eval: {args.label}")
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(results_dir / f"eval_{args.label}.png",
                dpi=140, bbox_inches="tight")
    print(f"✅ Wrote {results_dir / f'eval_{args.label}.png'}")


if __name__ == "__main__":
    main()