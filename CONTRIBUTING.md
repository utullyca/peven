# Contributing

## Setup

```bash
git clone git@github.com:utullyca/peven.git
cd peven
uv sync
uv run pre-commit install
```

## Tests

```bash
# Unit + integration (no external deps)
uv run pytest tests/ --ignore=tests/test_e2e.py -v

# Coverage
uv run pytest tests/ --ignore=tests/test_e2e.py --cov=src/peven --cov-report=term-missing
```

### E2E tests with live models

Peven works with any model supported by pydantic-ai (OpenAI, Anthropic, Google, etc.). The e2e test suite uses [ollama](https://ollama.com) as a free, local option — ollama exposes an OpenAI-compatible API, so pydantic-ai connects to it through its OpenAI provider gateway.

To run e2e tests:

```bash
# Install ollama (https://ollama.com), then:
ollama pull qwen2.5:0.5b
ollama serve  # if not already running

# Run e2e tests
uv run pytest tests/test_e2e.py -v
```

E2E tests are skipped automatically if ollama isn't running.

## Lint + Format

Pre-commit hooks run `ruff check` and `ruff format` automatically on every commit. To run manually:

```bash
uv run ruff check src/
uv run ruff format src/
```

## Structure

```
src/peven/
  petri/      # Petri net engine (schema, engine, executors, DSL, render)
  cli/        # CLI commands (peven run, peven validate)
tests/        # All tests
examples/     # Runnable eval files
```

## Pull Requests

- Pre-commit hooks handle lint and formatting automatically
- All tests must pass
- Keep PRs focused — one feature or fix per PR
