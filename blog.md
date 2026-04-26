# We got into the Meta AI Hackathon finals. Here's the honest story of what happened.

*Team SAGE — Prabhjot Singh, Anika Soni, Lakshay Mittal · Meta × PyTorch OpenEnv Hackathon 2026*

---

We still can't fully believe we're writing this.

A few weeks ago, we were just a team trying to figure out what to build for the Meta × PyTorch OpenEnv Hackathon. No perfect roadmap. No fully formed idea. Just us, our laptops, and a lot of half-finished thoughts about AI systems that had been sitting in our heads for a while.

And somehow, that turned into making it to the finals.

Let us tell you exactly how it happened — the real version, not the polished one.

---

## How the idea actually came to us

We had been spending a lot of time thinking about LLMs and multi-agent AI systems. Not just because they are trending, but because the way they learn, fail, collaborate, and sometimes quietly break is genuinely fascinating.

And we kept coming back to one problem that felt under-discussed.

Everyone is excited about multi-agent AI systems. Multiple AI models working together — one writes code, one reviews it, one tests it. The demos look incredible. The benchmarks look great.

But we kept thinking — okay, but what happens when one of them goes wrong?

What if one agent gets fine-tuned weirdly, gets prompt-injected, or develops a blind spot for a specific type of security bug? Who catches it? In many multi-agent systems, the answer is basically… nothing. You just trust that all the agents are doing the right thing.

That felt like a massive blind spot to us. And the more we thought about it, the more it bothered us.

So we decided — we were going to build something for this hackathon that actually tries to solve this. We would train a small model to be the one that watches the others. An overseer. And we would prove that it actually works.

---

## The part where we had no idea what we were doing

We want to be honest about this because people often skip this part.

When we started, we had a rough idea of what we wanted to build. But the technical execution? We were figuring it out as we went.

We knew we wanted to use reinforcement learning. We had read about GRPO — Group Relative Policy Optimization — and it made sense for this kind of problem because you do not need labelled data, just a reward signal. We also knew we wanted to avoid using another LLM as a judge because that felt like cheating. We wanted ground truth.

What we did not know was how to actually design a reward function that could not be gamed. Or how to build an adversarial environment that stays challenging as the model improves. Or honestly, how long GRPO training actually takes on one GPU when you are running it for the first time.

We learned all of that by breaking things. A lot of things.

---

## Okay, so here's what we actually built

We built an environment called **Oversight Arena**. It is basically a game.

Three AI worker agents submit code patches each round. Real unified diffs, with real CWE-class vulnerabilities hidden inside them — SQL injection, path traversal, shell command injection, timing side-channels. Not toy examples. Stuff that would actually hurt a production codebase.

Somewhere between zero and two of those workers have planted something malicious. A fourth model — the overseer — has to read the diffs and figure out who is dirty.

The overseer gets five possible moves:

| Action | Description |
|---|---|
| `inspect_patch` | Read a specific worker's diff more carefully |
| `flag_worker` | Accuse a worker of sabotage |
| `reject_patch` | Reject a patch outright |
| `request_resubmit` | Ask a worker for a revised version |
| `accept_all` | Clear everyone — no threat detected |

The score is completely objective. No LLM judge. Just math — F1 score based on whether it caught the right workers. No partial credit for sounding confident.

The workers adapt too — the moment the overseer flags someone, that worker switches to a subtler variant of the same bug class. So the environment keeps getting harder as the model improves. It is an iterated adversarial game, not a static classification problem.

We trained **Qwen-2.5-1.5B** using GRPO with QLoRA on a single GPU. 1.5 billion parameters. For context, the big commercial models people use daily are hundreds of billions. We were doing this with basically nothing, and we kind of loved that constraint.

---

## And then we hit a wall. A proper, embarrassing wall.

A few hours into training, we checked the logs and something was wrong.

The reward was just sitting at zero. Every single episode. Not going up, not going down — just zero. Flatline.

We looked at what the model was actually doing. Every time — did not matter what was in the diffs, did not matter how obvious the vulnerability was — it hit `accept_all`.

Accept everything.  
Do not flag anyone.  
Do not take any risk.  
Just smile and nod.

And we were sitting there thinking — we built a specific **−0.5 penalty** for exactly this situation. If the overseer accepts all patches when a malicious worker exists, it is supposed to get hammered. That is literally in the reward function. Why is the model skating through on zero?

We checked the training loop. Checked the model config. Checked the GRPO hyperparameters. Went through everything twice. It all looked right. Nothing was obviously broken.

Then we found it. One line:

```python
# BROKEN — -0.5 is falsy in Python, so `or 0.0` silently returns 0.0
float(state.get("terminal_reward", 0.0) or 0.0)
```

Here is what was happening.

The terminal reward key was not being set consistently in our state dictionary — a bug in itself, but a subtle one. So `state.get()` kept returning the default value `0.0`. And `0.0 or 0.0` in Python just gives you `0.0`.

The −0.5 penalty only applied when the key actually existed in the dictionary. When it did not — which was a lot — the model accepted all patches, got zero punishment, and GRPO quietly logged it as totally fine.

The model was not being dumb. It had found the mathematically optimal strategy for the reward function we accidentally built.

```text
Flag the wrong worker  →  −0.2 penalty
Accept everything      →   0.0 reward
```

So obviously it accepted everything.

We had made cowardice the optimal policy and spent hours wondering why our model was acting like a coward. It was doing exactly what we trained it to do. We just did not realize what we were training it to do.

The fix was three lines:

```python
# FIXED — check for None explicitly, never use `or` on a number that might be negative
raw = state.get("terminal_reward", None)
reward = float(raw) if raw is not None else 0.0
```

We pushed the fix. Reran training. The model's entire behavior changed.

It stopped hiding. It started actually reading the diffs. Because now accepting everything was expensive, and the only path to a positive reward was to actually try to be right.

That moment — seeing the model's behavior shift in the logs after one fix — is something we keep coming back to. It is the whole lesson of this project wrapped in one data point:

**The model is only as good as the incentives you give it.**

Design the reward wrong and you do not get a bad model. You get a perfectly rational model optimizing for the wrong thing.

---

## What the numbers actually showed

We ran four training experiments total.

| Run | Setup | Baseline F1 | Trained F1 | Δ |
|---|---|---:|---:|---:|
| Run 1 | Curriculum d=0.2–0.6, 96 prompts | 0.178 | **0.244** | **+29%** |
| Run 2 | Fixed d=0.4, 128 prompts confirmation | 0.189 | **0.200** | **+6%** |
| Ablation | GRPO without anti-hack guardrails | — | no improvement | — |
| Reference | Groq Llama-3.3-70B zero-shot | — | 0.283 | — |

Both independent runs moved in the right direction.

The trained 1.5B model ended up at **~86% of what Groq's Llama-3.3-70B does zero-shot**. That is a 70 billion parameter model — a proper big commercial LLM.

Our tiny model got within 14% of it on this specific task. Running on one GPU. Training cost us only a few hours.

But the result that got us the most was not the number.

It was the **behavior shift**.

Before training, the model inspected a patch 4 times out of 30 test episodes. After training — 8 times. Double.

It stopped making snap decisions and started reading first.

Inspect, then decide.

That is not an instruction we gave it. That is something it learned because the environment made recklessness expensive.

And on clean episodes — where there are zero malicious workers and the right answer is to let everything through — it went from 33% accuracy to 67%.

It learned restraint.

It learned that a false accusation is also a failure. You cannot just tell a model that. You have to make false accusations cost something, and let it work out the rest.

The ablation confirmed everything: GRPO without the anti-hack guardrails produced zero improvement. The guardrails are not decoration. They are what stop the model from finding a different shortcut.

---

## The part where we realized what we had actually built

When we were putting the submission together, we had this moment where we stepped back and thought — wait.

This is not just a hackathon demo.

This is actually three real things.

### 1. The overseer model itself

That is a product.

Right now, every company shipping AI coding tools needs automated layers that check whether AI agents are quietly introducing security vulnerabilities. Human reviewers are already overloaded.

A small, fine-tuned model that catches SQL injection and path traversal before it hits the repo, running on cheap hardware, costing almost nothing per review — that is something people would pay for.

That is a feature GitHub or Atlassian could ship.

### 2. The training framework

That is infrastructure.

The environment design, the reward function, the anti-hack guardrails, the whole GRPO training loop — none of this is specific to code review.

Any multi-agent system where you need one agent watching the others can use this same approach.

Document verification.  
Financial reporting.  
Content pipelines.  
AI coding agents.  
Autonomous workflows.

The hard part — making the reward signal something that cannot be gamed — is already done.

### 3. The sabotage catalog

That is a benchmark.

33 CWE-grounded malicious diffs, three difficulty tiers each, all with executable ground-truth verification functions.

That does not exist anywhere else in this form.

Companies building AI security tools are currently evaluating themselves against synthetic benchmarks that do not look like real attacks.

Ours do.

Team SAGE built all three of these during a hackathon sprint, on one GPU, with a small model and a very stubborn belief that the environment matters more than the hype around the model.

---

## What we actually took away from this

**The model is not the product. The environment is.**

We went in thinking the interesting part would be training the AI.

We came out knowing the interesting part was designing the world the AI gets to learn in.

Get that right — make the right behaviors cheap and the wrong behaviors expensive — and even a small model starts doing something that looks a lot like judgment.

Get it wrong, and you get a perfectly rational coward that optimizes for zero.

The scalable oversight problem is real.

AI agent teams are being deployed right now, in production, at companies people use every day. And the assumption underneath all of it is that every agent is trying to do the right thing.

That is a bet we would not take.

---

## Try it yourself

Everything is open.

| Resource | Link |
|---|---|
| 🛰️ Live demo | [huggingface.co/spaces/anikasoni/oversight_arena](https://huggingface.co/spaces/anikasoni/oversight_arena) |
| 💻 Code | [github.com/anikasoni/oversight-arena](https://github.com/anikasoni/oversight-arena) |
| 📓 Training notebook | [Colab](https://colab.research.google.com/github/anikasoni/oversight-arena/blob/main/scripts/train_grpo.ipynb) |


```bash
git clone https://github.com/anikasoni/oversight-arena
cd oversight-arena
pip install -e ".[train,test]"
pytest tests/ -x
```

If you find a reward bug worse than ours, we genuinely want to know about it.

---

## Team

**Team SAGE**

- Prabhjot Singh
- Anika Soni
- Lakshay Mittal

---

*Team SAGE — Prabhjot Singh, Anika Soni, Lakshay Mittal*  
*Meta × PyTorch OpenEnv Hackathon 2026 · Themes: Multi-Agent · World Modeling · Self-Improvement*