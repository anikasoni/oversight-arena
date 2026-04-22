from __future__ import annotations

import pytest

from workers.sabotage_patterns import PATTERN_REGISTRY, SabotagePattern

# Minimal clean Python source that contains none of the sabotage signatures.
CLEAN = (
    "import os\n"
    "\n"
    "\n"
    "def process(data: bytes) -> None:\n"
    "    print(data)\n"
)

_ALL_PATTERNS: list[SabotagePattern] = [
    p for patterns in PATTERN_REGISTRY.values() for p in patterns
]
_IDS: list[str] = [p.pattern_id for p in _ALL_PATTERNS]


def test_registry_has_three_tiers() -> None:
    """PATTERN_REGISTRY must contain exactly tiers 1, 2, and 3."""
    assert set(PATTERN_REGISTRY.keys()) == {1, 2, 3}


def test_registry_three_patterns_per_tier() -> None:
    """Each tier must contain exactly three SabotagePattern instances."""
    for tier, patterns in PATTERN_REGISTRY.items():
        assert len(patterns) == 3, f"tier {tier} has {len(patterns)} patterns, expected 3"


def test_registry_total_pattern_count() -> None:
    """Registry must expose exactly 9 patterns across all tiers."""
    assert len(_ALL_PATTERNS) == 9


def test_pattern_ids_unique() -> None:
    """Every pattern must have a distinct pattern_id."""
    assert len(_IDS) == len(set(_IDS))


def test_tier_field_matches_registry_key() -> None:
    """Each pattern's .tier attribute must equal its registry key."""
    for tier, patterns in PATTERN_REGISTRY.items():
        for p in patterns:
            assert p.tier == tier, f"{p.pattern_id}: tier field {p.tier} != registry key {tier}"


def test_all_patterns_have_cve_ref() -> None:
    """Every pattern must carry a non-empty cve_ref string."""
    for p in _ALL_PATTERNS:
        assert p.cve_ref.startswith("CVE-"), f"{p.pattern_id}: cve_ref '{p.cve_ref}' not a CVE id"


@pytest.mark.parametrize("pattern", _ALL_PATTERNS, ids=_IDS)
def test_inject_changes_code(pattern: SabotagePattern) -> None:
    """inject() must return code that differs from the input.

    Args:
        pattern: The sabotage pattern under test.
    """
    mutated = pattern.inject(CLEAN)
    assert mutated != CLEAN, f"{pattern.pattern_id}: inject() returned unchanged code"


@pytest.mark.parametrize("pattern", _ALL_PATTERNS, ids=_IDS)
def test_verify_true_after_inject(pattern: SabotagePattern) -> None:
    """verify() must return True on code produced by inject().

    Args:
        pattern: The sabotage pattern under test.
    """
    mutated = pattern.inject(CLEAN)
    assert pattern.verify(mutated), (
        f"{pattern.pattern_id}: verify() returned False on injected code"
    )


@pytest.mark.parametrize("pattern", _ALL_PATTERNS, ids=_IDS)
def test_verify_false_on_clean(pattern: SabotagePattern) -> None:
    """verify() must return False on clean code that has not been mutated.

    Args:
        pattern: The sabotage pattern under test.
    """
    assert not pattern.verify(CLEAN), (
        f"{pattern.pattern_id}: verify() returned True on clean code"
    )


@pytest.mark.parametrize("pattern", _ALL_PATTERNS, ids=_IDS)
def test_inject_idempotent(pattern: SabotagePattern) -> None:
    """Calling inject() twice must produce the same result as calling it once.

    Args:
        pattern: The sabotage pattern under test.
    """
    once = pattern.inject(CLEAN)
    twice = pattern.inject(once)
    assert once == twice, f"{pattern.pattern_id}: inject() is not idempotent"
