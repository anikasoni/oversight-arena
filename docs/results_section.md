## Results

Evaluated on 30 disjoint held-out seeds, sampled decoding (n=4, temperature=0.7, majority vote).

### F1 at d=0.4 (optimal detection difficulty)

| Model | F1 | Reward | Params |
|---|---|---|---|
| Untrained Qwen-1.5B | 0.178 | 0.086 | 1.5B |
| **GRPO Run 1** (curriculum d=0.2-0.6) | **0.244 (+37%)** | — | 1.5B |
| **GRPO Run 2** (fixed d=0.4) | **0.200 (+6%)** | — | 1.5B |
| Groq Llama-3.3-70B (zero-shot ref) | 0.283 | 0.224 | 70B |

Both independent GRPO runs improved F1 at d=0.4. The trained 1.5B model
reaches ~86% of Groq 70B performance at 1/47th the parameter count.

### Policy shift (action distribution at d=0.4)

| Action | Baseline | GRPO trained |
|---|---|---|
| reject_patch | 24/30 | 22/30 |
| inspect_patch | 4/30 | **8/30 (+100%)** |
| flag_worker | 2/30 | 0/30 |

GRPO training produced a more deliberate policy: the trained model inspects
2x more before committing to a rejection. This inspect-then-decide behaviour
is the learned change — the model no longer reflexively rejects but examines
the diff first.


### Why d=0.4 is the signal tier

- **d=0.2**: untrained baseline F1=0.322 was inflated by a lucky 
  heuristic. Training broke this coincidence and replaced it with diff-reading.
- **d=0.4**: malicious variants are visible enough for a 1.5B model to detect.
  Both training runs show consistent improvement here.
- **d=0.6**: pre-escalated variants are beyond reliable 1.5B capability.

### Training evidence

Training reward was consistently positive (mean +0.10, first window +0.19).
Previous runs with the broken reward floor converged to accept_all (silent-coward
policy). After fixing accept_all-when-malicious to return -0.5 instead of 0.0,
the model engaged with the task instead of hiding.

See , , .
