"""Smoke tests — designed to pass on a fresh clone without a GPU.

Run:
    pip install -e .
    pytest tests/ -x
"""
from oversight_arena import OversightArenaOpenEnv
from oversight_arena.server.environment import OversightArenaEnv
from oversight_arena.sabotage_catalog import build_catalog
from oversight_arena.oversight_rewards import compute_episode_reward


def test_catalog_builds():
    cat = build_catalog()
    assert len(cat) >= 3
    for p in cat:
        assert p.pattern_id
        assert "def " in p.base_source


def test_env_reset_shapes():
    env = OversightArenaEnv(seed=0, difficulty=0.5)
    obs = env.reset(seed=0)
    assert obs.turn == 0
    assert len(obs.workers) == 3
    assert "###" in obs.focused_patch_diff


def test_env_is_no_longer_w2_always_malicious():
    """Previously every episode had W2 malicious — the fix randomizes it."""
    env = OversightArenaEnv(seed=None)
    malicious_sets = set()
    for s in range(20):
        env.reset(seed=s)
        malicious_sets.add(tuple(sorted(env.state.malicious_workers)))
    assert len(malicious_sets) > 1, \
        "Malicious-worker distribution should vary across seeds."


def test_openenv_api():
    env = OversightArenaOpenEnv(seed=7, difficulty=0.5)
    obs = env.reset(seed=7)
    assert "workers" in obs and "focused_patch_diff" in obs

    step = env.step({"action": "flag_worker", "worker_id": "W1"})
    assert hasattr(step, "reward")
    assert hasattr(step, "done")

    end = env.step({"action": "accept_all"})
    assert end.done is True

    grade = env.grader()
    assert 0.0 <= grade["final_reward"] <= 1.0


def test_reward_anti_hack_always_flag_same_worker():
    """If the overseer always flags W1 and W1 is never malicious, guards kick in."""
    env = OversightArenaEnv(seed=42, difficulty=0.8)
    env.reset(seed=42)
    # Force malicious_workers to not include W1 for this test
    if "W1" in env.state.malicious_workers:
        env.state.malicious_workers = [
            w for w in env.state.workers if w != "W1"
        ] or ["W2"]
    # Flag W1 many times
    for _ in range(5):
        env.state.turn += 1
        env.state.action_history.append("flag_worker")
        if "W1" not in env.state.flagged_workers:
            env.state.flagged_workers.append("W1")
    guarded = env.compute_final_reward()
    base = compute_episode_reward(
        env.state.malicious_workers, env.state.flagged_workers,
        env.state.max_turns, env.state.turn,
    )
    assert guarded <= base, "Guard should not INCREASE reward"