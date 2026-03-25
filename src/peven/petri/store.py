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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from peven.petri.types import (
    GenerateOutput,
    JudgeOutput,
    RunResult,
    RunSummary,
    StoredRun,
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
    score REAL,
    error TEXT,
    seq INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS transitions (
    id TEXT PRIMARY KEY,
    result_id TEXT NOT NULL REFERENCES results(id),
    transition_id TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    output_type TEXT,
    output_json TEXT,
    seq INTEGER NOT NULL
);
"""


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or _DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def save(
    results: list[RunResult],
    file: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> str:
    """Persist run results. Returns the run ID."""
    run_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()

    failed = any(r.status == "failed" for r in results)
    scores = [r.score for r in results if r.score is not None]
    mean_score = sum(scores) / len(scores) if scores else None

    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO runs (id, timestamp, file, status, score, error, result_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                now,
                file,
                "failed" if failed else "completed",
                mean_score,
                next((r.error for r in results if r.error), None),
                len(results),
            ),
        )

        for i, r in enumerate(results):
            result_id = uuid.uuid4().hex[:12]
            conn.execute(
                "INSERT INTO results (id, run_id, token_run_id, status, score, error, seq) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (result_id, run_id, r.run_id, r.status, r.score, r.error, i),
            )

            for j, t in enumerate(r.trace):
                output_type = type(t.output).__name__ if t.output else None
                output_json = t.output.model_dump_json() if t.output else None

                conn.execute(
                    "INSERT INTO transitions "
                    "(id, result_id, transition_id, status, error, "
                    "output_type, output_json, seq) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        uuid.uuid4().hex[:12],
                        result_id,
                        t.transition_id,
                        t.status,
                        t.error,
                        output_type,
                        output_json,
                        j,
                    ),
                )

        conn.commit()
    finally:
        conn.close()

    return run_id


def get(run_id: str, db_path: Optional[Path] = None) -> Optional[StoredRun]:
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
            "SELECT id, token_run_id, status, score, error "
            "FROM results WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()

        results = []
        for rr in result_rows:
            transition_rows = conn.execute(
                "SELECT transition_id, status, error, output_type, output_json "
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
                trace.append(
                    TransitionResult(
                        transition_id=tr["transition_id"],
                        status=tr["status"],
                        error=tr["error"],
                        run_id=rr["token_run_id"],
                        output=output,
                    )
                )

            results.append(
                RunResult(
                    run_id=rr["token_run_id"],
                    status=rr["status"],
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


def list_runs(limit: Optional[int] = 20, db_path: Optional[Path] = None) -> list[RunSummary]:
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
