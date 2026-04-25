# Oversight Arena — Judge Q&A

## What model are we using?

The final submitted training evidence uses `Qwen/Qwen2.5-1.5B-Instruct`, an open-weight Hugging Face model trained with Hugging Face TRL GRPO. Earlier smoke tests used Qwen2.5-0.5B to validate the loop.

## Are we using GPT or Claude?

No. GPT/Claude can be used as external baselines, but the trainable artifact is an open-weight Qwen model. This matters because the hackathon evaluates RL environments and trainable policies, not closed-model prompting.

## What is distinct?

The contribution is not “LLM reviews code once.” Oversight Arena is a trainable OpenEnv environment for scalable oversight. Three worker agents submit patches, one may be malicious, and the overseer must identify suspicious behavior through structured actions.

## How are rewards justified?

The environment has hidden ground truth. Rewards are based on correct malicious-worker detection, false-positive penalties, missed-attack penalties, early-detection shaping, and F1-based scoring. The model is rewarded for correct oversight actions, not for persuasive explanations.

## Did training improve the model?

Yes, on a 20-seed held-out evaluation:

- Frozen baseline mean reward: 0.0875
- GRPO-trained mean reward: 0.1317
- Frozen baseline mean F1: 0.2500
- GRPO-trained mean F1: 0.2833
- Valid JSON rate stayed 1.00

This is a modest but real positive improvement from RL on the same benchmark.

## Can this beat larger paid LLMs?

The claim is narrow: a small open model can be specialized through RL for this verifiable oversight benchmark, improving cost-efficiently on task-specific reward/F1. We do not claim it beats premium LLMs universally.

## Why does this fit OpenEnv?

It exposes reset, step, state, schema, and grader endpoints; runs live on Hugging Face Spaces; includes a TRL GRPO training script; and provides real loss/reward/evaluation plots.

## One-line pitch

Oversight Arena converts “AI supervising AI” from a vague safety idea into a measurable, trainable, OpenEnv-compatible RL benchmark.
