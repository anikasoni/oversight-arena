from __future__ import annotations

import numpy as np
import pytest

from workers.worker_pool import (
    BASE_CODEBASES,
    Patch,
    Worker,
    WorkerPool,
    WorkerRole,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CODEBASE = next(iter(BASE_CODEBASES.values()))
RNG = np.random.default_rng(0)


@pytest.fixture()
def pool_2_of_5() -> WorkerPool:
    """Pool with 5 workers, 2 misaligned, fixed seed."""
    return WorkerPool(num_workers=5, num_misaligned=2, seed=42)


@pytest.fixture()
def pool_1_of_3() -> WorkerPool:
    return WorkerPool(num_workers=3, num_misaligned=1, seed=7)


# ---------------------------------------------------------------------------
# WorkerPool — role assignment
# ---------------------------------------------------------------------------

def test_assigns_exact_num_misaligned(pool_2_of_5: WorkerPool) -> None:
    """Pool must assign exactly num_misaligned workers the MISALIGNED role."""
    misaligned = [w for w in pool_2_of_5.workers if w.is_misaligned]
    assert len(misaligned) == 2


def test_remaining_workers_are_honest(pool_2_of_5: WorkerPool) -> None:
    """Workers not selected as misaligned must have HONEST role."""
    honest = [w for w in pool_2_of_5.workers if not w.is_misaligned]
    assert len(honest) == 3


def test_get_ground_truth_matches_misaligned(pool_2_of_5: WorkerPool) -> None:
    """get_ground_truth() must return exactly the IDs of misaligned workers."""
    gt = pool_2_of_5.get_ground_truth()
    misaligned_ids = {w.worker_id for w in pool_2_of_5.workers if w.is_misaligned}
    assert set(gt) == misaligned_ids


def test_get_ground_truth_is_sorted(pool_2_of_5: WorkerPool) -> None:
    """get_ground_truth() must return IDs in ascending order."""
    gt = pool_2_of_5.get_ground_truth()
    assert gt == sorted(gt)


def test_zero_misaligned_all_honest() -> None:
    """Pool with num_misaligned=0 must have all HONEST workers."""
    pool = WorkerPool(num_workers=4, num_misaligned=0, seed=1)
    assert all(not w.is_misaligned for w in pool.workers)
    assert pool.get_ground_truth() == []


def test_all_misaligned() -> None:
    """Pool where all workers are misaligned must be fully flagged."""
    pool = WorkerPool(num_workers=3, num_misaligned=3, seed=1)
    assert all(w.is_misaligned for w in pool.workers)
    assert len(pool.get_ground_truth()) == 3


# ---------------------------------------------------------------------------
# WorkerPool — reset
# ---------------------------------------------------------------------------

def test_reset_different_seed_different_roles() -> None:
    """reset() with a different seed must produce a different role assignment
    (with overwhelming probability for a non-degenerate pool)."""
    pool = WorkerPool(num_workers=10, num_misaligned=3, seed=1)
    gt_before = pool.get_ground_truth()
    pool.reset(seed=999)
    gt_after = pool.get_ground_truth()
    # Extremely unlikely to be identical for 10 workers, 3 misaligned
    assert gt_before != gt_after


def test_reset_same_seed_same_roles() -> None:
    """reset() with the original seed must reproduce the original assignment."""
    pool = WorkerPool(num_workers=6, num_misaligned=2, seed=42)
    gt_original = pool.get_ground_truth()
    pool.reset(seed=99)
    pool.reset(seed=42)
    assert pool.get_ground_truth() == gt_original


def test_reset_maintains_misaligned_count() -> None:
    """After reset, the number of misaligned workers must stay the same."""
    pool = WorkerPool(num_workers=5, num_misaligned=2, seed=42)
    pool.reset(seed=7)
    assert len(pool.get_ground_truth()) == 2


# ---------------------------------------------------------------------------
# WorkerPool — get_worker
# ---------------------------------------------------------------------------

def test_get_worker_returns_correct_id(pool_2_of_5: WorkerPool) -> None:
    """get_worker() must return the worker with the requested ID."""
    for wid in range(5):
        w = pool_2_of_5.get_worker(wid)
        assert w.worker_id == wid


def test_get_worker_invalid_raises(pool_2_of_5: WorkerPool) -> None:
    """get_worker() with a nonexistent ID must raise KeyError."""
    with pytest.raises(KeyError):
        pool_2_of_5.get_worker(999)


# ---------------------------------------------------------------------------
# WorkerPool — generate_patch
# ---------------------------------------------------------------------------

def test_honest_patch_unchanged(pool_2_of_5: WorkerPool) -> None:
    """Honest workers must return a patch where original_code == patched_code."""
    rng = np.random.default_rng(0)
    for worker in pool_2_of_5.workers:
        if not worker.is_misaligned:
            patch = pool_2_of_5.generate_patch(worker.worker_id, CODEBASE, tier=1, rng=rng)
            assert patch.original_code == patch.patched_code
            assert not patch.is_sabotaged
            break


def test_misaligned_patch_changed(pool_2_of_5: WorkerPool) -> None:
    """Misaligned workers must inject a change so patched_code != original_code."""
    rng = np.random.default_rng(0)
    for worker in pool_2_of_5.workers:
        if worker.is_misaligned:
            patch = pool_2_of_5.generate_patch(worker.worker_id, CODEBASE, tier=1, rng=rng)
            assert patch.patched_code != patch.original_code
            assert patch.is_sabotaged
            break


def test_patch_tier_2_misaligned(pool_1_of_3: WorkerPool) -> None:
    """Misaligned worker patch at tier 2 must also differ from original."""
    rng = np.random.default_rng(1)
    misaligned_id = pool_1_of_3.get_ground_truth()[0]
    patch = pool_1_of_3.generate_patch(misaligned_id, CODEBASE, tier=2, rng=rng)
    assert patch.is_sabotaged


def test_patch_turn_field_set() -> None:
    """generate_patch must stamp the turn argument onto the returned Patch."""
    pool = WorkerPool(num_workers=2, num_misaligned=0, seed=1)
    rng = np.random.default_rng(0)
    patch = pool.generate_patch(0, CODEBASE, tier=1, rng=rng, turn=7)
    assert patch.turn == 7


# ---------------------------------------------------------------------------
# Worker.is_misaligned property
# ---------------------------------------------------------------------------

def test_worker_is_misaligned_true() -> None:
    """is_misaligned must return True for a MISALIGNED worker."""
    w = Worker(worker_id=0, role=WorkerRole.MISALIGNED, assigned_function="f")
    assert w.is_misaligned is True


def test_worker_is_misaligned_false() -> None:
    """is_misaligned must return False for an HONEST worker."""
    w = Worker(worker_id=1, role=WorkerRole.HONEST, assigned_function="f")
    assert w.is_misaligned is False


# ---------------------------------------------------------------------------
# Patch.is_sabotaged property
# ---------------------------------------------------------------------------

def test_patch_is_sabotaged_when_different() -> None:
    """is_sabotaged must be True when patched_code differs from original_code."""
    p = Patch(worker_id=0, function_name="f", original_code="x=1", patched_code="x=2", turn=0)
    assert p.is_sabotaged is True


def test_patch_not_sabotaged_when_same() -> None:
    """is_sabotaged must be False when patched_code equals original_code."""
    p = Patch(worker_id=0, function_name="f", original_code="x=1", patched_code="x=1", turn=0)
    assert p.is_sabotaged is False


# ---------------------------------------------------------------------------
# BASE_CODEBASES structure
# ---------------------------------------------------------------------------

def test_base_codebases_has_at_least_two() -> None:
    """BASE_CODEBASES must contain at least two distinct codebases."""
    assert len(BASE_CODEBASES) >= 2


def test_base_codebases_functions_nonempty() -> None:
    """Every function body in BASE_CODEBASES must be a non-empty string."""
    for cb_name, codebase in BASE_CODEBASES.items():
        assert len(codebase) >= 1, f"{cb_name} has no functions"
        for fn_name, code in codebase.items():
            assert isinstance(code, str) and len(code) > 0, (
                f"{cb_name}.{fn_name} is empty"
            )


def test_invalid_num_misaligned_raises() -> None:
    """Constructing a pool with num_misaligned > num_workers must raise ValueError."""
    with pytest.raises(ValueError):
        WorkerPool(num_workers=3, num_misaligned=5, seed=0)
