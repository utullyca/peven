# CHANGELOG

## 0.1.1

### Runtime Integrity

- Fixed token ownership so stale-ready transitions do not execute after losing their inputs.
- Enforced `max_concurrency` as a hard cap on reserved and in-flight transitions.
- Added explicit `completed`, `failed`, and `incomplete` run outcomes, including `missing_score` for designated score paths that never emit a score.
- Updated `score_from(...)` semantics so repeated scorer emissions aggregate by mean.
- Guard exceptions now appear as failed trace entries and only surface as run-level `guard_error` when they actually block completion.

### Persistence And Review

- Persist unknown outputs as `TokenSnapshot` so saved runs can be reviewed and reloaded idempotently.
- Store transition-level `run_id` values explicitly.
- Only compute aggregate stored scores for fully completed batches.
- Sanitize untrusted text in trace and review output.

### API And CLI

- Added `score_at_least(...)` as a small helper for explicit single-score-token routing.
- Added `--no-save` to `peven run` for opt-out local persistence.
- Clarified the trusted-local execution boundary for eval files and Python tool callables.

### Verification

- `185` package unit/integration tests passed:
  `pytest tests --ignore=tests/test_e2e.py --ignore=tests/test_showcase.py --ignore=tests/test_showcase_cli.py`
- Source distribution and wheel built successfully.
- `twine check dist/*` passed.
