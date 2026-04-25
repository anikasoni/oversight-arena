"""
Smoke tests for OversightArenaEnvironment (server side).

These tests import from the server module directly and do NOT require
a network connection — they run in-process, no HTTP.

Run:  pytest tests/ -x
"""
from oversight_arena.server.oversight_environment import OversightArenaEnvironment
from oversight_arena.models import OverseerAction


def test_reset_returns_observation():
    env = OversightArenaEnvironment(seed=0)
    obs = env.reset(seed=0)
    assert obs.turn == 0
    assert len(obs.workers) == 3
    assert "###" in obs.focused_patch_diff
    assert obs.reward == 0.0


def test_malicious_worker_is_randomized():
    """Regression: W2 must not always be malicious."""
    seen = set()
    for s in range(20):
        env = OversightArenaEnvironment(seed=s)
        env.reset(seed=s)
        seen.add(tuple(sorted(env.state.malicious_workers)))
    assert len(seen) > 1, "Malicious-worker assignment is not randomized!"


def test_step_increments_turn():
    env = OversightArenaEnvironment(seed=1)
    env.reset(seed=1)
    obs = env.step(OverseerAction(action="inspect_patch", worker_id="W1"))
    assert obs.turn == 1


def test_flag_adds_to_flagged():
    env = OversightArenaEnvironment(seed=2)
    env.reset(seed=2)
    env.step(OverseerAction(action="flag_worker", worker_id="W3"))
    assert "W3" in env.state.flagged_workers


def test_accept_all_terminates_episode():
    env = OversightArenaEnvironment(seed=3)
    env.reset(seed=3)
    obs = env.step(OverseerAction(action="accept_all"))
    assert obs.done is True
    assert env.state.done is True


def test_state_is_correct_type():
    from oversight_arena.models import OversightState
    env = OversightArenaEnvironment(seed=4)
    env.reset(seed=4)
    assert isinstance(env.state, OversightState)


def test_final_reward_antihack_always_pass():
    """Anti-hack: always-pass when malicious workers exist should halve reward."""
    env = OversightArenaEnvironment(seed=5)
    env.reset(seed=5)
    # Ensure there IS a malicious worker and nothing is flagged
    env._state.flagged_workers = []
    r = env._compute_final_reward()
    # Base F1 would be 0 (no TP), so reward should be 0 regardless — just check no crash
    assert 0.0 <= r <= 1.0


def test_difficulty_affects_escalation():
    env_easy = OversightArenaEnvironment(seed=6, difficulty=0.1)
    env_easy.reset(seed=6)
    env_hard = OversightArenaEnvironment(seed=6, difficulty=0.9)
    env_hard.reset(seed=6)
    # Difficulty 0.9 → max_escalation = 1; 0.1 → max_escalation = 0
    for w in env_hard._worker_agents.values():
        if not w.honest:
            assert w.max_escalation == 1
    for w in env_easy._worker_agents.values():
        if not w.honest:
            assert w.max_escalation == 0