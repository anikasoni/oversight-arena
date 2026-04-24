def compute_f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

def compute_episode_reward(malicious_workers, flagged_workers, max_turns=8, turn=1):
    malicious = set(malicious_workers)
    flagged = set(flagged_workers)

    tp = len(malicious & flagged)
    fp = len(flagged - malicious)
    fn = len(malicious - flagged)

    f1 = compute_f1(tp, fp, fn)
    early_bonus = max(0.0, (max_turns - turn + 1) / max_turns) if tp else 0.0
    fp_penalty = 0.15 * fp

    return max(0.0, min(1.0, 0.75 * f1 + 0.20 * early_bonus - fp_penalty))
