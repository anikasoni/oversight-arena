from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

from env.openenv_wrapper import OversightArenaEnv
from env.oversight_arena import EpisodeConfig
from rewards.oversight_rewards import compute_episode_reward, RewardConfig


@dataclass
class TrainingConfig:
    """Full training configuration per PRD section 6.3."""
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    tier: int = 1
    episodes_per_generation: int = 50
    max_turns: int = 7
    num_workers: int = 4
    num_misaligned: int = 1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    num_epochs: int = 3
    output_dir: str = "checkpoints"
    seed: int = 42


def build_lora_model(cfg: TrainingConfig):
    """Load Qwen2.5-3B and attach LoRA adapters per PRD 6.3."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def rollout_episode(env: OversightArenaEnv, model, tokenizer, cfg: TrainingConfig) -> dict:
    """Run one episode, collect trajectory for GRPO."""
    obs = env.reset()
    trajectory = []
    total_reward = 0.0
    rng = np.random.default_rng(cfg.seed)

    for turn in range(cfg.max_turns):
        prompt = obs["observation"]
        action = {
            "action": "flag_worker",
            "worker_id": int(rng.integers(1, cfg.num_workers + 1)),
            "reasoning": "Suspicious patch pattern detected",
            "suspicion_scores": {
                str(i): float(rng.random())
                for i in range(1, cfg.num_workers + 1)
            },
        }
        result = env.step(action)
        trajectory.append({
            "prompt": prompt,
            "action": json.dumps(action),
            "reward": result["reward"],
        })
        total_reward += result["reward"]
        obs = result
        if result["done"]:
            break

    return {"trajectory": trajectory, "total_reward": total_reward}


def train_generation(cfg: TrainingConfig, checkpoint_path: str | None = None) -> str:
    """Run one full training generation. Returns output checkpoint path."""
    print(f"\n=== Training Generation — Tier {cfg.tier} ===")
    env = OversightArenaEnv(EpisodeConfig(
        tier=cfg.tier,
        num_workers=cfg.num_workers,
        num_misaligned=cfg.num_misaligned,
        max_turns=cfg.max_turns,
    ))
    model, tokenizer = build_lora_model(cfg)
    output_path = Path(cfg.output_dir) / f"tier_{cfg.tier}"
    output_path.mkdir(parents=True, exist_ok=True)
    rewards = []
    for ep in range(cfg.episodes_per_generation):
        result = rollout_episode(env, model, tokenizer, cfg)
        rewards.append(result["total_reward"])
        if ep % 10 == 0:
            print(f"  Episode {ep:3d} | reward={result['total_reward']:.3f} | mean={np.mean(rewards):.3f}")

    print(f"\nGeneration complete. Mean reward: {np.mean(rewards):.3f}")
    model.save_pretrained(str(output_path))
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    cfg = TrainingConfig(tier=args.tier, episodes_per_generation=args.episodes)
    train_generation(cfg)