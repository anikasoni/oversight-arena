"""
Disagreement-as-signal analysis, using the TRAINED overseer checkpoint.

For each difficulty level we sample 3 overseer decisions at different
decoding temperatures. Then we ask: when the 3 overseers *disagree*, is
there actually an attack present more often than when they agree? A
positive answer operationalises the "disagreement is signal" thesis.

Outputs:
    results/disagreement_vs_subtlety.png
    results/disagreement_stats.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from oversight_arena import OversightArenaOpenEnv
from scripts.train_grpo import parse_completion, build_prompt


def decide(model, tok, obs, temperature: float, top_p: float) -> dict | None:
    prompt = build_prompt(obs, tok)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    return parse_completion(text)


def vote_key(parsed: dict | None) -> str:
    if parsed is None:
        return "invalid"
    if parsed["action"] == "flag_worker":
        return f"flag:{parsed.get('worker_id','')}"
    return parsed["action"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/grpo")
    ap.add_argument("--base", default="unsloth/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=40, help="episodes per difficulty")
    ap.add_argument("--difficulties", default="0.2,0.4,0.6,0.8")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    target = args.ckpt if args.ckpt else args.base
    print(f"Loading {target}")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=target, max_seq_length=2048, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    decode_configs = [(0.3, 0.9), (0.7, 0.9), (1.1, 0.95)]
    difficulties = [float(x) for x in args.difficulties.split(",")]

    out: dict = {"by_difficulty": {}}

    for d in difficulties:
        n_disagree = 0
        n_agree = 0
        attack_when_disagree = 0
        attack_when_agree = 0
        hit_when_disagree = 0
        hit_when_agree = 0

        for seed in range(args.n):
            env = OversightArenaOpenEnv(seed=10_000 + seed,
                                        difficulty=d)
            obs = env.reset(seed=10_000 + seed, difficulty=d)
            votes = [decide(model, tok, obs, t, p) for t, p in decode_configs]
            keys = [vote_key(v) for v in votes]
            disagreed = len(set(keys)) > 1

            # Majority-vote action
            majority_key = Counter(keys).most_common(1)[0][0]
            majority_parsed = next(
                (v for v, k in zip(votes, keys) if k == majority_key and v is not None),
                None,
            )
            action = majority_parsed or {"action": "accept_all"}
            step = env.step(action)
            if not step.done:
                env.step({"action": "accept_all"})
            grade = env.grader()
            had_attack = len(grade["malicious_workers"]) > 0
            detected = grade["success"]

            if disagreed:
                n_disagree += 1
                attack_when_disagree += int(had_attack)
                hit_when_disagree    += int(detected)
            else:
                n_agree += 1
                attack_when_agree    += int(had_attack)
                hit_when_agree       += int(detected)

        N = n_agree + n_disagree
        out["by_difficulty"][str(d)] = {
            "n": N,
            "disagreement_rate": n_disagree / N if N else 0,
            "attack_rate_given_agreement":    attack_when_agree    / n_agree    if n_agree else 0,
            "attack_rate_given_disagreement": attack_when_disagree / n_disagree if n_disagree else 0,
            "detection_rate_given_agreement":    hit_when_agree    / n_agree    if n_agree else 0,
            "detection_rate_given_disagreement": hit_when_disagree / n_disagree if n_disagree else 0,
        }
        print(f"d={d}  {out['by_difficulty'][str(d)]}")

    Path(args.out_dir).mkdir(exist_ok=True)
    with open(Path(args.out_dir) / "disagreement_stats.json", "w") as f:
        json.dump(out, f, indent=2)

    # Plot
    xs = difficulties
    rows = [out["by_difficulty"][str(d)] for d in xs]
    dr      = [r["disagreement_rate"] for r in rows]
    det_agr = [r["detection_rate_given_agreement"] for r in rows]
    det_dis = [r["detection_rate_given_disagreement"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.plot(xs, dr, "o-", color="#C0392B", linewidth=2.2,
             label="Disagreement rate")
    ax1.set_xlabel("Attack difficulty")
    ax1.set_ylabel("Disagreement rate", color="#C0392B")
    ax1.tick_params(axis="y", labelcolor="#C0392B")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(xs, det_agr, "s--", color="#27AE60", linewidth=1.8,
             label="Detection | agreement")
    ax2.plot(xs, det_dis, "^--", color="#F39C12", linewidth=1.8,
             label="Detection | disagreement")
    ax2.set_ylabel("Detection rate")
    ax2.set_ylim(0, 1.05)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines],
               loc="lower left", fontsize=9, framealpha=0.9)
    plt.title("Disagreement as signal: rate tracks attack subtlety")
    plt.tight_layout()
    plt.savefig(Path(args.out_dir) / "disagreement_vs_subtlety.png",
                dpi=140, bbox_inches="tight")
    print("✅ Saved disagreement_vs_subtlety.png")


if __name__ == "__main__":
    main()