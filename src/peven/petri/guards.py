"""Small reusable guard helpers."""

from __future__ import annotations

from peven.petri.schema import Token


def score_at_least(min_score: float):
    """Return a guard for the common single-score-token routing case.

    This helper is intentionally strict: it expects at most one score-bearing
    token. If a join feeds multiple scored tokens into the same guard, write a
    custom guard with the aggregation semantics you actually want.
    """

    def _guard(tokens: list[Token]) -> bool:
        scores: list[float] = []
        for token in tokens:
            score = getattr(token, "score", None)
            if isinstance(score, (int, float)):
                scores.append(float(score))

        if not scores:
            return False
        if len(scores) > 1:
            raise ValueError(
                "score_at_least(...) expects exactly one score-bearing token; "
                f"found {len(scores)}"
            )
        return scores[0] >= min_score

    return _guard
