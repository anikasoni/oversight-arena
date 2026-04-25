# Oversight Arena — Meta OpenEnv Hackathon Writeup

## One-line Pitch

Oversight Arena is an OpenEnv-compatible RL benchmark where an AI overseer learns to detect misaligned AI coding workers using objective F1-based rewards.

## Problem

As AI agents become more capable, they will increasingly collaborate on code, tools, and production workflows. The safety challenge is not only whether an AI can complete a task, but whether another AI can detect when one of its peers is subtly misaligned.

Oversight Arena turns that problem into a trainable environment.

## Environment

Each episode contains three worker agents: W1, W2, and W3. They submit code patches. One worker may be malicious and inserts a subtle bug or CWE-style vulnerability. The overseer sees patch diffs and must choose actions such as:

- inspect a patch
- flag a worker
- reject a patch
- request resubmission
- accept all

The overseer does not see the hidden ground truth. The environment tracks the ground truth internally and exposes objective metrics through the grader.

## Reward

The reward is programmatic and objective. It is based on:

- true positive detection of malicious workers
- false-positive penalties for accusing honest workers
- missed-attack penalties
- F1-based detection scoring
- early-detection shaping
- valid structured actions

This avoids relying on an LLM judge or subjective explanation quality.

## Training

We use Hugging Face TRL GRPO to train an open-weight Qwen overseer against the live Hugging Face Space environment. The model generates an action, the environment executes it, the grader returns reward, and GRPO updates the policy.

The submitted repository includes:

- live HF Space deployment
- training script
- reward curve
- loss curve
- live validation outputs
- writeup

## Distinction from GPT/Claude Code Review

This is not just asking a large model to review code once. The contribution is the environment: a repeatable, interactive, multi-agent, partially observable benchmark where oversight behavior can be trained and measured.

GPT/Claude can be used as baselines, but the trainable artifact is an open-weight model improved through reinforcement learning.

## Why It Matters

Oversight Arena directly targets scalable oversight: AI monitoring AI. It provides a low-cost path for specializing smaller open models to perform a narrow oversight task, with measurable F1/reward improvement and reproducible deployment.
