# Roadmap

## v0.1

**Engine**
- Async event-driven execution with `asyncio.wait(FIRST_COMPLETED)` scheduling
- Colored tokens (`run_id`) for batch isolation — single engine, N concurrent runs
- Consume-eagerly / deposit-on-completion lifecycle (no locks)
- Semaphore-based concurrency control
- Transition-level retries

**Executors**
- Pluggable executor protocol
- Agent executor (pydantic-ai) — tool support, model_settings passthrough
- Rubric judge executor — per_criterion, oneshot, rubric_as_judge strategies

**Guards**
- Sync and async `when` guards on transitions
- LLM judge gates (async guard that calls a model)

**DSL**
- Python DSL with `>>` chaining — compiles to Pydantic IR
- Bipartite enforcement (place >> transition >> place)

**Persistence**
- SQLite store (`~/.peven/runs.db`) with normalized schema: runs → results → transitions
- Automatic persistence on every `peven run`

**Rendering**
- Rich results table, trace trees, topology view with cycle detection
- Per-criterion rubric breakdown in trace view (MET/UNMET, reasons, weights)

**CLI**
- `peven run` — execute + persist + render
- `peven validate` — validate + show topology
- `peven review <id>` / `peven review all` — query stored runs

**Tests**
- 157 unit/integration tests, 20 live e2e tests (ollama qwen2.5:0.5b)

---

## v0.2

**Arcs**
- Read arcs (test arcs) — check for a token without consuming it, enabling broadcast from a single place to multiple transitions
- Weighted arc DSL — `place.arc(transition, weight=N)` for transitions that need N tokens from the same place (e.g., ensemble voting gates that collect N judge scores before firing)

**Execution**
- Subgraph execution (nested nets)
- Budget enforcement (USD spend caps, per-node cost tracking)

**Data**
- Dataset versioning (CSV, JSON, Parquet, HuggingFace import/publish)

**Analysis**
- Replay from saved runs (extract subgraph, re-execute)
- Counterfactual branching (fork from decision point, change one variable)

---

## v0.3

**Execution**
- Sandbox/environment execution (Docker containers, file I/O between nodes)

**Judges**
- Code execution judge (run code, compare output)
- Env check judge (score from environment state)

**Registries**
- Rubric registry (versioned documents, activate/archive)
- Model registry (pricing, capabilities)

**Tools**
- `@peven_tool()` decorator with auto-discovery and token compaction

---

## v0.4+

**Analysis**
- Run comparison (side-by-side deltas: cost, score, output)
- Topology diff (compare execution shapes across runs)
- Ablation studies (baseline vs variants)
- Trace extraction with filtering (per-node, per-tool, per-child-run)

**Performance**
- Partial-copy marking in consume/deposit — only copy places being modified instead of the full marking

**Observability**
- Live streaming / watch mode

**CLI**
- Full expansion (results, dataset, rubric, model, secret, docs, doctor)
