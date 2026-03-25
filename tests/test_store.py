"""Test SQLite persistence: save, get, list, roundtrip, edge cases."""

from __future__ import annotations

import pytest

from peven.petri.store import get, list_runs, save
from peven.petri.types import GenerateOutput, JudgeOutput, RunResult, TransitionResult

# -- save + get roundtrip ------------------------------------------------------


def test_save_and_get_single(tmp_path):
    """Single result with full trace survives roundtrip."""
    db = tmp_path / "test.db"
    results = [
        RunResult(
            status="completed",
            score=0.9,
            trace=[
                TransitionResult(
                    transition_id="gen",
                    status="completed",
                    output=GenerateOutput(text="hello world"),
                ),
                TransitionResult(
                    transition_id="jdg",
                    status="completed",
                    output=JudgeOutput(score=0.9),
                ),
            ],
        )
    ]

    run_id = save(results, file="eval.py", db_path=db)
    data = get(run_id, db_path=db)

    assert data is not None
    assert data.id == run_id
    assert data.status == "completed"
    assert data.score == 0.9
    assert data.file == "eval.py"
    assert data.result_count == 1

    r = data.results[0]
    assert len(r.trace) == 2
    assert r.trace[0].transition_id == "gen"
    assert isinstance(r.trace[0].output, GenerateOutput)
    assert r.trace[0].output.text == "hello world"
    assert r.trace[1].transition_id == "jdg"
    assert isinstance(r.trace[1].output, JudgeOutput)
    assert r.trace[1].output.score == 0.9


def test_save_and_get_batch(tmp_path):
    """Batch results stored as separate rows in results table."""
    db = tmp_path / "test.db"
    results = [
        RunResult(run_id="a", status="completed", score=0.8, trace=[]),
        RunResult(run_id="b", status="failed", error="boom", trace=[]),
        RunResult(run_id="c", status="completed", score=0.6, trace=[]),
    ]

    run_id = save(results, db_path=db)
    data = get(run_id, db_path=db)

    assert data.result_count == 3
    assert data.status == "failed"  # one failed → run is failed
    assert data.score == pytest.approx(0.7)  # mean of 0.8 and 0.6
    assert data.error == "boom"
    assert len(data.results) == 3

    # Verify order preserved
    assert data.results[0].run_id == "a"
    assert data.results[1].run_id == "b"
    assert data.results[2].run_id == "c"


def test_save_preserves_token_run_ids(tmp_path):
    """Colored token run_ids survive roundtrip."""
    db = tmp_path / "test.db"
    results = [
        RunResult(
            run_id="abc123",
            status="completed",
            trace=[
                TransitionResult(
                    transition_id="gen",
                    status="completed",
                    run_id="abc123",
                    output=GenerateOutput(text="ok"),
                ),
            ],
        )
    ]

    run_id = save(results, db_path=db)
    data = get(run_id, db_path=db)
    assert data.results[0].run_id == "abc123"
    assert data.results[0].trace[0].run_id == "abc123"


# -- edge cases: empty and None ------------------------------------------------


def test_save_empty_results(tmp_path):
    """No results produces a valid run with count 0."""
    db = tmp_path / "test.db"
    run_id = save([], db_path=db)
    data = get(run_id, db_path=db)

    assert data.result_count == 0
    assert data.status == "completed"
    assert data.score is None
    assert data.error is None
    assert data.results == []


def test_save_no_score(tmp_path):
    """Results with no score → run score is None."""
    db = tmp_path / "test.db"
    results = [
        RunResult(
            status="completed",
            trace=[
                TransitionResult(transition_id="gen", status="completed"),
            ],
        ),
    ]

    run_id = save(results, db_path=db)
    data = get(run_id, db_path=db)
    assert data.score is None


def test_save_no_file(tmp_path):
    """File is optional."""
    db = tmp_path / "test.db"
    run_id = save([RunResult(status="completed", trace=[])], db_path=db)
    data = get(run_id, db_path=db)
    assert data.file is None


def test_save_transition_no_output(tmp_path):
    """Transition with no output roundtrips correctly."""
    db = tmp_path / "test.db"
    results = [
        RunResult(
            status="completed",
            trace=[
                TransitionResult(transition_id="t", status="completed", output=None),
            ],
        )
    ]

    run_id = save(results, db_path=db)
    data = get(run_id, db_path=db)
    assert data.results[0].trace[0].output is None


def test_save_transition_with_error(tmp_path):
    """Failed transition error string survives roundtrip."""
    db = tmp_path / "test.db"
    results = [
        RunResult(
            status="failed",
            error="executor crashed",
            trace=[
                TransitionResult(
                    transition_id="gen",
                    status="failed",
                    error="timeout after 30s",
                ),
            ],
        )
    ]

    run_id = save(results, db_path=db)
    data = get(run_id, db_path=db)
    assert data.error == "executor crashed"
    assert data.results[0].trace[0].error == "timeout after 30s"
    assert data.results[0].trace[0].status == "failed"


# -- get not found -------------------------------------------------------------


def test_get_not_found(tmp_path):
    db = tmp_path / "test.db"
    assert get("nonexistent", db_path=db) is None


def test_get_not_found_empty_db(tmp_path):
    """Fresh db with no runs returns None."""
    db = tmp_path / "empty.db"
    assert get("anything", db_path=db) is None


# -- list_runs -----------------------------------------------------------------


def test_list_runs_newest_first(tmp_path):
    db = tmp_path / "test.db"
    save([RunResult(status="completed", trace=[])], file="first.py", db_path=db)
    save([RunResult(status="completed", trace=[])], file="second.py", db_path=db)
    save([RunResult(status="completed", trace=[])], file="third.py", db_path=db)

    runs = list_runs(db_path=db)
    assert len(runs) == 3
    assert runs[0].file == "third.py"
    assert runs[2].file == "first.py"


def test_list_runs_respects_limit(tmp_path):
    db = tmp_path / "test.db"
    for i in range(10):
        save([RunResult(status="completed", trace=[])], file=f"{i}.py", db_path=db)

    runs = list_runs(limit=3, db_path=db)
    assert len(runs) == 3


def test_list_runs_empty(tmp_path):
    db = tmp_path / "test.db"
    runs = list_runs(db_path=db)
    assert runs == []


def test_list_runs_shows_summary(tmp_path):
    """List includes status, score, result_count."""
    db = tmp_path / "test.db"
    save(
        [
            RunResult(run_id="a", status="completed", score=0.9, trace=[]),
            RunResult(run_id="b", status="completed", score=0.7, trace=[]),
        ],
        file="batch.py",
        db_path=db,
    )

    [run] = list_runs(db_path=db)
    assert run.status == "completed"
    assert run.score == pytest.approx(0.8)
    assert run.result_count == 2
    assert run.file == "batch.py"
    assert run.id is not None
    assert run.timestamp is not None


# -- multiple saves don't collide ----------------------------------------------


def test_unique_run_ids(tmp_path):
    """Each save produces a unique run ID."""
    db = tmp_path / "test.db"
    ids = set()
    for _ in range(20):
        rid = save([RunResult(status="completed", trace=[])], db_path=db)
        ids.add(rid)
    assert len(ids) == 20


# -- batch with mixed statuses -------------------------------------------------


def test_batch_all_completed(tmp_path):
    """All completed → run status is completed."""
    db = tmp_path / "test.db"
    results = [
        RunResult(run_id="a", status="completed", score=1.0, trace=[]),
        RunResult(run_id="b", status="completed", score=0.5, trace=[]),
    ]
    run_id = save(results, db_path=db)
    data = get(run_id, db_path=db)
    assert data.status == "completed"


def test_batch_all_failed(tmp_path):
    """All failed → run status is failed, score is None."""
    db = tmp_path / "test.db"
    results = [
        RunResult(run_id="a", status="failed", error="e1", trace=[]),
        RunResult(run_id="b", status="failed", error="e2", trace=[]),
    ]
    run_id = save(results, db_path=db)
    data = get(run_id, db_path=db)
    assert data.status == "failed"
    assert data.score is None
    assert data.error == "e1"  # first error


def test_batch_mixed_status(tmp_path):
    """One failed → run is failed, score is mean of completed only."""
    db = tmp_path / "test.db"
    results = [
        RunResult(run_id="a", status="completed", score=1.0, trace=[]),
        RunResult(run_id="b", status="failed", error="oops", trace=[]),
    ]
    run_id = save(results, db_path=db)
    data = get(run_id, db_path=db)
    assert data.status == "failed"
    assert data.score == pytest.approx(1.0)  # only the completed one


# -- complex trace roundtrip ---------------------------------------------------


def test_multi_transition_trace_roundtrip(tmp_path):
    """Long trace with mixed output types survives roundtrip in order."""
    db = tmp_path / "test.db"
    results = [
        RunResult(
            status="completed",
            score=0.75,
            trace=[
                TransitionResult(
                    transition_id="gen1",
                    status="completed",
                    output=GenerateOutput(text="first"),
                ),
                TransitionResult(
                    transition_id="gen2",
                    status="completed",
                    output=GenerateOutput(text="second"),
                ),
                TransitionResult(
                    transition_id="jdg",
                    status="completed",
                    output=JudgeOutput(score=0.75),
                ),
                TransitionResult(
                    transition_id="done",
                    status="completed",
                    output=None,
                ),
            ],
        )
    ]

    run_id = save(results, db_path=db)
    data = get(run_id, db_path=db)
    trace = data.results[0].trace

    assert len(trace) == 4
    assert trace[0].transition_id == "gen1"
    assert trace[0].output.text == "first"
    assert trace[1].transition_id == "gen2"
    assert trace[1].output.text == "second"
    assert trace[2].transition_id == "jdg"
    assert trace[2].output.score == 0.75
    assert trace[3].transition_id == "done"
    assert trace[3].output is None


def test_judge_output_score_zero_counted_in_mean(tmp_path):
    """score=0.0 is a valid score and should be included in mean calculation."""
    db = tmp_path / "test.db"
    results = [
        RunResult(status="completed", score=0.0, trace=[]),
        RunResult(status="completed", score=1.0, trace=[]),
    ]
    run_id = save(results, db_path=db)
    data = get(run_id, db_path=db)
    assert data.score == pytest.approx(0.5), "score=0.0 should be counted in mean"


def test_judge_output_full_roundtrip(tmp_path):
    """All JudgeOutput fields survive save/get roundtrip."""
    from rubric import CriterionReport

    db = tmp_path / "test.db"
    judge_output = JudgeOutput(
        score=0.85,
        raw_score=0.85,
        llm_raw_score=85.0,
        report=[CriterionReport(weight=1.0, requirement="clear", verdict="MET", reason="good")],
    )
    results = [
        RunResult(
            status="completed",
            score=0.85,
            trace=[TransitionResult(transition_id="jdg", status="completed", output=judge_output)],
        )
    ]
    run_id = save(results, db_path=db)
    data = get(run_id, db_path=db)

    restored = data.results[0].trace[0].output
    assert isinstance(restored, JudgeOutput)
    assert restored.score == 0.85, "score should survive roundtrip"
    assert restored.raw_score == 0.85, "raw_score should survive roundtrip"
    assert restored.llm_raw_score == 85.0, "llm_raw_score should survive roundtrip"
    assert len(restored.report) == 1, "report should survive roundtrip"
    assert restored.report[0].requirement == "clear"
