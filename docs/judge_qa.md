# Oversight Arena — Judge Q&A

## What model are we using?

We use an open-weight Qwen overseer trained with Hugging Face TRL GRPO. The current submitted training evidence uses Qwen/Qwen2.5-0.5B-Instruct for a working, reproducible GRPO run. The environment is model-agnostic and can scale to larger Qwen models when compute allows.

## Are we using GPT or Claude?

No. GPT/Claude can be used as external baselines, but the trainable artifact is an open-weight model. This matters because the goal is reinforcement learning, not prompting a closed model.

## What is distinct?

The contribution is not simply “LLM reviews code.” The contribution is a trainable OpenEnv environment for scalable oversight. Three worker agents submit patches, one may be malicious, and the overseer must identify the suspicious worker using step-by-step actions and objective reward.

## How are rewards justified?

The environment knows the hidden ground truth. Rewards are based on correct malicious-worker detection, false-positive penalties, missed-attack penalties, early-detection shaping, and F1-based scoring. The model is not rewarded for sounding convincing; it is rewarded for correct oversight actions.

## Can this beat larger paid LLMs?

The claim is narrow and task-specific: a smaller open model can be specialized through RL to improve cost-efficiently on this verifiable oversight benchmark. We do not claim it beats premium LLMs universally.

## Why does this fit OpenEnv?

It has reset, step, state, schema, and grader endpoints; a deployed Hugging Face Space; a runnable TRL training script; and real loss/reward plots from a GRPO run.

## One-line pitch

Oversight Arena converts “AI supervising AI” from a vague safety idea into a measurable, trainable, OpenEnv-compatible RL benchmark.
