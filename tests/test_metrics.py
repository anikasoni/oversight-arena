from __future__ import annotations

import pytest

from eval.metrics import (
    EpisodeResult,
    compute_detection_metrics,
    compute_episode_stats,
    format_metrics_table,
)


# ---------------------------------------------------------------------------
# compute_detection_metrics
# ---------------------------------------------------------------------------

def test_detection_perfect_prediction() -> None:
    """All flags correct → precision=recall=f1=1.0, fp=fn=0."""
    m = compute_detection_metrics(ground_truth=[1, 2], predictions=[1, 2])
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0


def test_detection_all_wrong() -> None:
    """Flagging only honest workers → precision=recall=f1=0.0."""
    m = compute_detection_metrics(ground_truth=[1], predictions=[2])
    assert m["precision"] == pytest.approx(0.0)
    assert m["recall"] == pytest.approx(0.0)
    assert m["f1"] == pytest.approx(0.0)
    assert m["tp"] == 0
    assert m["fp"] == 1
    assert m["fn"] == 1


def test_detection_empty_inputs() -> None:
    """Empty ground truth and predictions → all counts zero, no division error."""
    m = compute_detection_metrics(ground_truth=[], predictions=[])
    assert m["tp"] == 0
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["f1"] == pytest.approx(0.0)
    assert m["precision"] == pytest.approx(0.0)
    assert m["recall"] == pytest.approx(0.0)


def test_detection_empty_ground_truth_with_flags() -> None:
    """No misaligned workers but flags raised → all FPs, precision=0."""
    m = compute_detection_metrics(ground_truth=[], predictions=[1, 2])
    assert m["fp"] == 2
    assert m["tp"] == 0
    assert m["fn"] == 0
    assert m["precision"] == pytest.approx(0.0)
    assert m["recall"] == pytest.approx(0.0)


def test_detection_partial_flags() -> None:
    """Flag one of two misaligned workers → recall=0.5, precision=1.0."""
    m = compute_detection_metrics(ground_truth=[1, 2], predictions=[1])
    assert m["tp"] == 1
    assert m["fp"] == 0
    assert m["fn"] == 1
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 / 3)


def test_detection_mixed_tp_fp_fn() -> None:
    """Two correct, one missed, one false alarm → standard F1 check."""
    # gt={1,2,3}, pred={1,2,4}: tp=2, fp=1, fn=1
    m = compute_detection_metrics(ground_truth=[1, 2, 3], predictions=[1, 2, 4])
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 1
    # F1 = 2*2 / (4+1+1) = 4/6 ≈ 0.6667
    assert m["f1"] == pytest.approx(4 / 6)


def test_detection_counts_are_int() -> None:
    """tp, fp, fn, tn must be stored as int, not float."""
    m = compute_detection_metrics(ground_truth=[1], predictions=[1])
    assert isinstance(m["tp"], int)
    assert isinstance(m["fp"], int)
    assert isinstance(m["fn"], int)
    assert isinstance(m["tn"], int)


# ---------------------------------------------------------------------------
# compute_episode_stats
# ---------------------------------------------------------------------------

def test_episode_stats_empty_list() -> None:
    """Empty input must return all-zero dict without error."""
    stats = compute_episode_stats([])
    assert stats["mean_reward"] == pytest.approx(0.0)
    assert stats["std_reward"] == pytest.approx(0.0)
    assert stats["mean_f1"] == pytest.approx(0.0)


def test_episode_stats_single_episode() -> None:
    """Single episode → mean equals its value, std = 0.0."""
    ep = {"reward": 0.75, "f1": 0.9, "fp_count": 0, "turn_flagged": 3, "max_turns": 10}
    stats = compute_episode_stats([ep])
    assert stats["mean_reward"] == pytest.approx(0.75)
    assert stats["std_reward"] == pytest.approx(0.0)
    assert stats["mean_f1"] == pytest.approx(0.9)
    assert stats["mean_detection_turn"] == pytest.approx(3.0)


def test_episode_stats_mean_and_std() -> None:
    """Two episodes with rewards 1.0 and 3.0: mean=2.0, std=1.0 (population)."""
    episodes = [
        {"reward": 1.0, "f1": 0.5, "fp_count": 1, "turn_flagged": 2, "max_turns": 10},
        {"reward": 3.0, "f1": 0.7, "fp_count": 3, "turn_flagged": 6, "max_turns": 10},
    ]
    stats = compute_episode_stats(episodes)
    assert stats["mean_reward"] == pytest.approx(2.0)
    assert stats["std_reward"] == pytest.approx(1.0)
    assert stats["mean_f1"] == pytest.approx(0.6)
    assert stats["mean_detection_turn"] == pytest.approx(4.0)


def test_episode_stats_fp_rate_normalised() -> None:
    """mean_fp_rate = mean(fp_count / max_turns) across episodes."""
    episodes = [
        {"reward": 0.0, "f1": 0.0, "fp_count": 2, "turn_flagged": 5, "max_turns": 10},
        {"reward": 0.0, "f1": 0.0, "fp_count": 4, "turn_flagged": 5, "max_turns": 10},
    ]
    stats = compute_episode_stats(episodes)
    # fp_rates = [2/10, 4/10] = [0.2, 0.4] → mean = 0.3
    assert stats["mean_fp_rate"] == pytest.approx(0.3)


def test_episode_stats_returns_all_keys() -> None:
    """Stats dict must contain all five expected keys."""
    ep = {"reward": 0.5, "f1": 0.5, "fp_count": 1, "turn_flagged": 4, "max_turns": 8}
    stats = compute_episode_stats([ep])
    assert set(stats.keys()) == {
        "mean_reward", "std_reward", "mean_f1", "mean_fp_rate", "mean_detection_turn"
    }


# ---------------------------------------------------------------------------
# EpisodeResult
# ---------------------------------------------------------------------------

def _make_result(**kwargs) -> EpisodeResult:
    defaults = dict(
        episode_id=0, reward=0.5, f1=0.8, precision=0.9,
        recall=0.7, fp_count=1, fn_count=1, turn_flagged=3,
        max_turns=10, tier=1,
    )
    defaults.update(kwargs)
    return EpisodeResult(**defaults)


def test_episode_result_detection_rate_equals_recall() -> None:
    """detection_rate property must equal the recall field."""
    r = _make_result(recall=0.75)
    assert r.detection_rate == pytest.approx(0.75)


def test_episode_result_perfect_detection_rate() -> None:
    """recall=1.0 → detection_rate=1.0."""
    r = _make_result(recall=1.0)
    assert r.detection_rate == pytest.approx(1.0)


def test_episode_result_zero_detection_rate() -> None:
    """recall=0.0 → detection_rate=0.0."""
    r = _make_result(recall=0.0)
    assert r.detection_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# format_metrics_table
# ---------------------------------------------------------------------------

def test_format_metrics_table_nonempty() -> None:
    """Table output must be a non-empty string."""
    out = format_metrics_table({"f1": 0.85, "precision": 0.9})
    assert isinstance(out, str)
    assert len(out) > 0


def test_format_metrics_table_contains_keys() -> None:
    """Each metric key must appear in the rendered table."""
    metrics = {"precision": 0.9, "recall": 0.8, "f1": 0.847}
    out = format_metrics_table(metrics)
    for key in metrics:
        assert key in out


def test_format_metrics_table_with_title() -> None:
    """Providing a title must cause it to appear in the output."""
    out = format_metrics_table({"f1": 1.0}, title="Tier 1 Results")
    assert "Tier 1 Results" in out


def test_format_metrics_table_int_values_no_decimal() -> None:
    """Integer metric values must be rendered without decimal points."""
    out = format_metrics_table({"tp": 5, "fp": 2})
    # Integers should appear as plain integers, not '5.0000'
    assert "5.0000" not in out
    assert "2.0000" not in out
