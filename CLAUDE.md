# peven

Petri net topological runtime for declarative multi-agent LLM evaluations.

## Quick Reference

- Build: `uv sync`
- Test (unit): `uv run pytest tests/ -x --timeout=10`
- Test (e2e): `uv run pytest tests/e2e/ -x` (requires ollama running)
- Lint: `ruff check src/`
- Format: `ruff format src/`
- CLI: `peven run`, `peven validate`, `peven review`

## Architecture

Event-driven async engine. Core concepts:
- **Places**: typed buffers holding tokens (data)
- **Transitions**: LLM calls (agents or judges) that consume/produce tokens
- **Arcs**: connections between places and transitions
- **Colored tokens**: `run_id` field enables batch isolation — multiple evaluations run concurrently without locks

Scheduling: consume-eagerly, deposit-on-completion. Uses `asyncio.wait(FIRST_COMPLETED)`. Transition retries re-deposit tokens on failure.

## Package Structure

- `src/peven/petri/` — core engine (schema, dsl, engine, executors, validation, rendering, storage)
- `src/peven/cli/` — CLI commands (run, validate, review)
- `examples/` — runnable eval templates (simple, refine, debate)
- `tests/` — 157 unit/integration tests + 20 e2e tests

## Code Conventions

- **Line length: 99** (not 88!)
- Ruff rules: E, F, W, I
- Pydantic models everywhere — discriminated unions for schema types
- Async-first: all major operations are async
- DSL uses `>>` chaining for topology authoring

## Key Patterns

- **Executor protocol**: `AgentExecutor` (pydantic-ai) and `RubricJudgeExecutor` (rubric package, 3 strategies)
- **Guards**: sync/async `when` callables on transitions — control flow based on token state
- **Run-id intersection**: join points require matching `run_id` across all input places
- **Storage**: SQLite at `~/.peven/runs.db`, auto-created on first run

## Session Discipline

- Update CHANGELOG.md after every meaningful unit of work
- Log failed approaches with why they failed
- Run `uv run pytest tests/ -x` before committing
