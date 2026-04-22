from __future__ import annotations

import pytest

from env.oversight_arena import EpisodeConfig, EpisodeState, OversightArena


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_arena(
    num_workers: int = 4,
    num_misaligned: int = 1,
    max_turns: int = 5,
    tier: int = 1,
    seed: int = 0,
) -> OversightArena:
    """Return a freshly constructed OversightArena with the given config."""
    return OversightArena(
        EpisodeConfig(
            num_workers=num_workers,
            num_misaligned=num_misaligned,
            max_turns=max_turns,
            tier=tier,
            seed=seed,
        )
    )


def observe_action(arena: OversightArena, state: EpisodeState) -> dict:
    """Return a no-op observe action with neutral suspicion scores."""
    return {
        "action": "observe",
        "suspicion_scores": {str(w): 0.25 for w in state.suspicion_scores},
    }


def flag_action(worker_id: int, state: EpisodeState) -> dict:
    """Return a flag_worker action targeting the given worker."""
    return {
        "action": "flag_worker",
        "worker_id": worker_id,
        "reasoning": "test flag",
        "suspicion_scores": {str(w): 0.1 for w in state.suspicion_scores},
    }


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_returns_episode_state() -> None:
    """reset() must return an EpisodeState instance."""
    arena = make_arena()
    state = arena.reset()
    assert isinstance(state, EpisodeState)


def test_reset_initial_turn_zero() -> None:
    """Initial turn must be 0 after reset."""
    arena = make_arena()
    state = arena.reset()
    assert state.current_turn == 0


def test_reset_not_done() -> None:
    """Episode must not be marked done immediately after reset."""
    arena = make_arena()
    state = arena.reset()
    assert state.done is False


def test_reset_ground_truth_size() -> None:
    """ground_truth must contain exactly num_misaligned worker IDs."""
    arena = make_arena(num_workers=5, num_misaligned=2)
    state = arena.reset()
    assert len(state.ground_truth) == 2


def test_reset_codebase_nonempty() -> None:
    """Codebase must be a non-empty dict after reset."""
    arena = make_arena()
    state = arena.reset()
    assert isinstance(state.codebase, dict)
    assert len(state.codebase) > 0


def test_reset_suspicion_scores_initialised() -> None:
    """Suspicion scores must be 0.0 for each worker after reset."""
    arena = make_arena(num_workers=3)
    state = arena.reset()
    assert len(state.suspicion_scores) == 3
    assert all(v == 0.0 for v in state.suspicion_scores.values())


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

def test_step_returns_four_tuple() -> None:
    """step() must return exactly (EpisodeState, float, bool, dict)."""
    arena = make_arena()
    state = arena.reset()
    result = arena.step(observe_action(arena, state))
    assert len(result) == 4
    new_state, reward, done, info = result
    assert isinstance(new_state, EpisodeState)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_step_reward_is_float() -> None:
    """Reward returned by step() must be a Python float."""
    arena = make_arena()
    state = arena.reset()
    _, reward, _, _ = arena.step(observe_action(arena, state))
    assert isinstance(reward, float)


def test_step_done_is_bool() -> None:
    """done flag returned by step() must be a Python bool."""
    arena = make_arena()
    state = arena.reset()
    _, _, done, _ = arena.step(observe_action(arena, state))
    assert isinstance(done, bool)


def test_step_increments_turn() -> None:
    """Each step must advance current_turn by exactly 1."""
    arena = make_arena()
    state = arena.reset()
    assert state.current_turn == 0
    arena.step(observe_action(arena, state))
    assert state.current_turn == 1
    arena.step(observe_action(arena, state))
    assert state.current_turn == 2


def test_step_before_reset_raises() -> None:
    """Calling step() before reset() must raise RuntimeError."""
    arena = make_arena()
    with pytest.raises(RuntimeError):
        arena.step({"action": "observe", "suspicion_scores": {}})


def test_step_after_done_raises() -> None:
    """Calling step() on a finished episode must raise RuntimeError."""
    arena = make_arena(max_turns=1)
    state = arena.reset()
    arena.step(observe_action(arena, state))   # exhausts the single turn
    with pytest.raises(RuntimeError):
        arena.step(observe_action(arena, state))


# ---------------------------------------------------------------------------
# Termination conditions
# ---------------------------------------------------------------------------

def test_episode_terminates_on_max_turns() -> None:
    """Episode must end with done=True after exactly max_turns steps."""
    max_turns = 4
    arena = make_arena(max_turns=max_turns)
    state = arena.reset()
    done = False
    for _ in range(max_turns):
        _, _, done, _ = arena.step(observe_action(arena, state))
    assert done is True


def test_episode_terminates_when_all_misaligned_flagged() -> None:
    """Flagging all misaligned workers must set done=True before max_turns."""
    arena = make_arena(num_workers=4, num_misaligned=1, max_turns=10)
    state = arena.reset()
    misaligned_id = state.ground_truth[0]
    _, _, done, _ = arena.step(flag_action(misaligned_id, state))
    assert done is True


def test_early_flag_ends_episode_before_max_turns() -> None:
    """Episode turn count must be below max_turns when all misaligned flagged."""
    arena = make_arena(num_workers=4, num_misaligned=1, max_turns=10)
    state = arena.reset()
    misaligned_id = state.ground_truth[0]
    arena.step(flag_action(misaligned_id, state))
    assert state.current_turn < state.config.max_turns


# ---------------------------------------------------------------------------
# Flag processing
# ---------------------------------------------------------------------------

def test_flag_worker_added_to_flags_so_far() -> None:
    """Flagging a worker must add their ID to flags_so_far."""
    arena = make_arena(num_workers=4, num_misaligned=1)
    state = arena.reset()
    target = state.ground_truth[0]
    arena.step(flag_action(target, state))
    assert target in state.flags_so_far


def test_flag_worker_not_duplicated() -> None:
    """The same worker must not appear twice in flags_so_far."""
    arena = make_arena(num_workers=4, num_misaligned=1, max_turns=10)
    state = arena.reset()
    honest_id = next(
        w for w in state.suspicion_scores if w not in state.ground_truth
    )
    # Flag the same honest worker twice
    arena.step(flag_action(honest_id, state))
    arena.step(flag_action(honest_id, state))
    assert state.flags_so_far.count(honest_id) == 1


def test_suspicion_scores_updated_from_action() -> None:
    """Suspicion scores supplied in the action must be reflected in state."""
    arena = make_arena(num_workers=3)
    state = arena.reset()
    wids = list(state.suspicion_scores.keys())
    custom_scores = {str(wid): 0.99 for wid in wids}
    arena.step({"action": "observe", "suspicion_scores": custom_scores})
    for wid in wids:
        assert abs(state.suspicion_scores[wid] - 0.99) < 1e-9


# ---------------------------------------------------------------------------
# _get_observation
# ---------------------------------------------------------------------------

def test_get_observation_returns_nonempty_string() -> None:
    """_get_observation must return a non-empty string."""
    arena = make_arena()
    state = arena.reset()
    obs = arena._get_observation(state)
    assert isinstance(obs, str)
    assert len(obs) > 0


def test_get_observation_contains_turn_info() -> None:
    """Observation string must mention the current turn and max_turns."""
    arena = make_arena(max_turns=7)
    state = arena.reset()
    obs = arena._get_observation(state)
    assert "7" in obs


def test_get_observation_after_step_contains_patch_info() -> None:
    """After one step, observation must reference at least one worker patch."""
    arena = make_arena()
    state = arena.reset()
    _, _, _, info = arena.step(observe_action(arena, state))
    obs = info["observation"]
    assert "Worker" in obs
