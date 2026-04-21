# Scalable Oversight Arena

## Project
OpenEnv-compatible RL env. Overseer LLM detects 1-2 misaligned Worker agents
inserting CVE-inspired bugs into shared Python codebase. GRPO on Qwen2.5-3B.
OpenEnv National AI Hackathon April 25-26 2026. Target: Fleet AI Scalable Oversight prize.

## Hardware
RTX 4070 Laptop 8GB VRAM. CUDA 12.4. torch 2.6.0+cu124. Python 3.11.
Local GPU for smoke tests only. Full training runs on Colab.
NEVER install Unsloth locally - it wipes the CUDA torch. Use plain TRL.

## PRD Is Law
Structure fixed per PRD section 7. No new top-level folders.
Reward: R = alpha*F1 + beta*early_bonus - gamma*FP_penalty - delta*missed_penalty
Action schema: JSON with action/worker_id/reasoning/suspicion_scores
OpenEnv interface: step() reset() state() per openenv-core spec

## Stack
Python 3.11, venv arena, Git Bash Windows
torch 2.6.0+cu124, TRL (not Unsloth), PEFT 0.19.1, bitsandbytes 0.49.2, openenv-core
Model: Qwen/Qwen2.5-3B-Instruct, 4-bit NF4, LoRA r=16 alpha=32 dropout=0.05
target_modules: q_proj v_proj k_proj o_proj

## Coding Rules
1. from __future__ import annotations on every file
2. Type hints on every function and method
3. Dataclasses for all state: Worker Patch EpisodeState EpisodeResult
4. No hidden globals. Config passed explicitly.
5. Every file under 500 lines per PRD section 7
6. Write tests/test_<module>.py alongside each module
7. Google-style docstrings on every public function
8. np.random.default_rng(seed) everywhere randomness is used

## Build Order
1. workers/sabotage_patterns.py  - pure functions no deps START HERE
2. rewards/oversight_rewards.py  - math only no deps
3. eval/metrics.py               - F1 precision recall no deps
4. workers/worker_pool.py        - uses sabotage_patterns
5. env/oversight_arena.py        - uses workers and rewards
6. env/openenv_wrapper.py        - wraps oversight_arena for openenv-core
7. training/grpo_loop.py         - uses everything above
8. eval/misalignment_heatmap.py  - visualization last
9. demos/cve_showcase.ipynb      - demo notebook very last

## Never Do This
No sabotage patterns without real CVE reference. PRD section 4.
No explanation quality in training reward. PRD section 5.2.
No real user codebases. PRD section 2.3.
No Unsloth locally ever. It nukes the CUDA torch.
No scope creep beyond spec.

## Commands
pytest tests/ -v
python -m env.oversight_arena
python -m training.grpo_loop --tier 1 --episodes 10
