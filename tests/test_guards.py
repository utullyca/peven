"""Test reusable guard helpers."""

from peven.petri.guards import score_at_least
from peven.petri.schema import Token
from peven.petri.types import JudgeOutput


def test_score_at_least_matches_score():
    guard = score_at_least(0.7)
    assert guard([JudgeOutput(score=0.75)]) is True
    assert guard([JudgeOutput(score=0.65)]) is False


def test_score_at_least_skips_non_scored_tokens():
    guard = score_at_least(0.5)
    assert guard([Token(), JudgeOutput(score=0.6)]) is True
    assert guard([Token()]) is False


def test_score_at_least_rejects_multiple_scores():
    guard = score_at_least(0.5)
    try:
        guard([JudgeOutput(score=0.4), JudgeOutput(score=0.9)])
    except ValueError as exc:
        assert "exactly one score-bearing token" in str(exc)
    else:
        raise AssertionError("expected ValueError for multiple score-bearing tokens")
