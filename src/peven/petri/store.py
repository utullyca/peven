"""SQLite persistence for run results.

Schema:
    runs        — one per `peven run` invocation
    results     — one per RunResult (1 for single, N for batch)
    transitions — one per TransitionResult (the execution trace)
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from peven.petri.types import (
    GenerateOutput,
    JudgeOutput,
    RunResult,
    RunSummary,
    StoredRun,
    TokenSnapshot,
    TransitionResult,
)


_DB_DIR = Path.home() / ".peven"
_DB_PATH = _DB_DIR / "runs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    file TEXT,
    status TEXT NOT NULL,
    score REAL,
    error TEXT,
    result_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    token_run_id TEXT,
    status TEXT NOT NULL,
    terminal_reason TEXT,
    score REAL,
    error TEXT,
    seq INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS transitions (
    id TEXT PRIMARY KEY,
    result_id TEXT NOT NULL REFERENCES results(id),
    transition_run_id TEXT,
    transition_id TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    output_type TEXT,
    output_module TEXT,
    output_json TEXT,
    seq INTEGER NOT NULL
);
"""


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Additive schema migrations for older local databases."""
    columns = {
        "results": {"terminal_reason": "TEXT"},
        "transitions": {"output_module": "TEXT", "transition_run_id": "TEXT"},
    }
    for table, table_columns in columns.items():
        existing = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for column_name, column_type in table_columns.items():
            if column_name in existing:
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_type}")


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or _DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _ensure_columns(conn)
    return conn


def _snapshot_output(output):
    """Convert unknown token outputs into inert snapshots before persistence."""
    if output is None:
        return None
    if isinstance(output, (GenerateOutput, JudgeOutput, TokenSnapshot)):
        return output

    return TokenSnapshot(
        run_id=output.run_id,
        type_name=f"{type(output).__module__}.{type(output).__name__}",
        payload=output.model_dump(mode="json"),
    )


def _serialize_output(output) -> tuple[str | None, str | None, str | None]:
    """Return (type, module, json) for a transition output."""
    output = _snapshot_output(output)
    if output is None:
        return None, None, None

    return type(output).__name__, type(output).__module__, output.model_dump_json()


def _aggregate_status(results: list[RunResult]) -> str:
    """Collapse per-run statuses into a stored run status."""
    failed = any(r.status == "failed" for r in results)
    incomplete = any(r.status == "incomplete" for r in results)
    if failed:
        return "failed"
    if incomplete:
        return "incomplete"
    return "completed"


def _aggregate_score(results: list[RunResult]) -> float | None:
    """Store an aggregate score only when every child result completed."""
    if not results:
        return None
    if any(r.status != "completed" for r in results):
        return None

    scores = [r.score for r in results]
    if any(score is None for score in scores):
        return None
    return sum(scores) / len(scores)


def _aggregate_error(results: list[RunResult], status: str) -> str | None:
    """Summarize aggregate errors without picking an arbitrary child error."""
    if not results:
        return None
    if len(results) == 1:
        return results[0].error
    if status == "completed":
        return None

    failed = sum(1 for r in results if r.status == "failed")
    incomplete = sum(1 for r in results if r.status == "incomplete")
    parts: list[str] = []
    if failed:
        parts.append(f"{failed} failed")
    if incomplete:
        parts.append(f"{incomplete} incomplete")
    summary = ", ".join(parts) if parts else status
    return f"{summary}; inspect child results for details"


def save(
    results: list[RunResult],
    file: str | None = None,
    db_path: Path | None = None,
) -> str:
    """Persist run results. Returns the run ID."""
    run_id = uuid.uuid4().hex[:12]
    now = datetime.now(UTC).isoformat()
    status = _aggregate_status(results)
    score = _aggregate_score(results)
    error = _aggregate_error(results, status)

    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO runs (id, timestamp, file, status, score, error, result_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                now,
                file,
                status,
                score,
                error,
                len(results),
            ),
        )

        for i, r in enumerate(results):
            result_id = uuid.uuid4().hex[:12]
            conn.execute(
                "INSERT INTO results (id, run_id, token_run_id, status, terminal_reason, score, error, seq) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (result_id, run_id, r.run_id, r.status, r.terminal_reason, r.score, r.error, i),
            )

            for j, t in enumerate(r.trace):
                output_type, output_module, output_json = _serialize_output(t.output)

                conn.execute(
                    "INSERT INTO transitions "
                    "(id, result_id, transition_run_id, transition_id, status, error, "
                    "output_type, output_module, output_json, seq) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        uuid.uuid4().hex[:12],
                        result_id,
                        t.run_id,
                        t.transition_id,
                        t.status,
                        t.error,
                        output_type,
                        output_module,
                        output_json,
                        j,
                    ),
                )

        conn.commit()
    finally:
        conn.close()

    return run_id


def get(run_id: str, db_path: Path | None = None) -> StoredRun | None:
    """Fetch a run by ID with all results and transitions."""
    conn = _connect(db_path)
    try:
        run_row = conn.execute(
            "SELECT id, timestamp, file, status, score, error, result_count "
            "FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()

        if run_row is None:
            return None

        result_rows = conn.execute(
            "SELECT id, token_run_id, status, terminal_reason, score, error "
            "FROM results WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()

        results = []
        for rr in result_rows:
            transition_rows = conn.execute(
                "SELECT transition_run_id, transition_id, status, error, "
                "output_type, output_module, output_json "
                "FROM transitions WHERE result_id = ? ORDER BY seq",
                (rr["id"],),
            ).fetchall()

            trace = []
            for tr in transition_rows:
                output = None
                if tr["output_json"]:
                    data = json.loads(tr["output_json"])
                    if tr["output_type"] == "GenerateOutput":
                        output = GenerateOutput(**data)
                    elif tr["output_type"] == "JudgeOutput":
                        output = JudgeOutput(**data)
                    elif tr["output_type"] == "TokenSnapshot":
                        output = TokenSnapshot(**data)
                    else:
                        type_name = tr["output_type"] or "Token"
                        if tr["output_module"] and tr["output_module"] != "builtins":
                            type_name = f"{tr['output_module']}.{type_name}"
                        output = TokenSnapshot(
                            run_id=data.get("run_id"),
                            type_name=type_name,
                            payload=data,
                        )
                trace.append(
                    TransitionResult(
                        transition_id=tr["transition_id"],
                        status=tr["status"],
                        error=tr["error"],
                        run_id=tr["transition_run_id"] or rr["token_run_id"],
                        output=output,
                    )
                )

            results.append(
                RunResult(
                    run_id=rr["token_run_id"],
                    status=rr["status"],
                    terminal_reason=rr["terminal_reason"],
                    score=rr["score"],
                    error=rr["error"],
                    trace=trace,
                )
            )
    finally:
        conn.close()

    return StoredRun(
        id=run_row["id"],
        timestamp=run_row["timestamp"],
        file=run_row["file"],
        status=run_row["status"],
        score=run_row["score"],
        error=run_row["error"],
        result_count=run_row["result_count"],
        results=results,
    )


def list_runs(limit: int | None = 20, db_path: Path | None = None) -> list[RunSummary]:
    """List recent runs, newest first. Pass limit=None for all."""
    conn = _connect(db_path)
    try:
        if limit is None:
            rows = conn.execute(
                "SELECT id, timestamp, file, status, score, result_count "
                "FROM runs ORDER BY timestamp DESC",
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, timestamp, file, status, score, result_count "
                "FROM runs ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
    finally:
        conn.close()

    return [
        RunSummary(
            id=r["id"],
            timestamp=r["timestamp"],
            file=r["file"],
            status=r["status"],
            score=r["score"],
            result_count=r["result_count"],
        )
        for r in rows
    ]
