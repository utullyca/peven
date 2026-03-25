"""Test CLI: loader, run, validate, review commands."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from peven.cli.loader import LoadError, load
from peven.cli.main import app
from peven.petri.store import save

runner = CliRunner()


# -- Helpers -------------------------------------------------------------------


def _write_eval(tmp_path: Path, code: str) -> Path:
    p = tmp_path / "eval.py"
    p.write_text(textwrap.dedent(code))
    return p


# -- loader --------------------------------------------------------------------


def test_load_missing_file():
    with pytest.raises(LoadError, match="not found"):
        load("/nonexistent/eval.py")


def test_load_not_python(tmp_path):
    p = tmp_path / "eval.txt"
    p.write_text("hello")
    with pytest.raises(LoadError, match=".py"):
        load(str(p))


def test_load_no_net(tmp_path):
    p = _write_eval(tmp_path, "x = 42\n")
    with pytest.raises(LoadError, match="No 'net'"):
        load(str(p))


def test_load_net_wrong_type(tmp_path):
    p = _write_eval(tmp_path, "net = 'not a net'\n")
    with pytest.raises(LoadError, match="must be a Net"):
        load(str(p))


def test_load_rows_without_place(tmp_path):
    p = _write_eval(
        tmp_path,
        """\
        from peven.petri.dsl import NetBuilder, agent
        from peven.petri.schema import Token
        n = NetBuilder()
        p = n.place("in")
        o = n.place("out")
        t = n.transition("t", agent(model="test", prompt="{text}"))
        p >> t >> o
        p.token(Token())
        net = n.build()
        rows = [Token()]
    """,
    )
    with pytest.raises(LoadError, match="'rows' requires"):
        load(str(p))


def test_load_valid(tmp_path):
    p = _write_eval(
        tmp_path,
        """\
        from peven.petri.dsl import NetBuilder, agent
        from peven.petri.schema import Token
        n = NetBuilder()
        inp = n.place("in")
        out = n.place("out")
        t = n.transition("t", agent(model="test", prompt="{text}"))
        inp >> t >> out
        inp.token(Token())
        net = n.build()
    """,
    )
    net, rows, place = load(str(p))
    assert len(net.places) == 2
    assert rows is None
    assert place is None


def test_load_syntax_error(tmp_path):
    p = _write_eval(tmp_path, "def broken(\n")
    with pytest.raises(LoadError, match="Error executing"):
        load(str(p))


def test_load_spec_none(tmp_path):
    """Covers the spec-is-None guard in loader."""
    p = _write_eval(tmp_path, "net = 1\n")
    with patch("peven.cli.loader.importlib.util.spec_from_file_location", return_value=None):
        with pytest.raises(LoadError, match="Cannot load module"):
            load(str(p))


# -- validate command ----------------------------------------------------------


def test_validate_valid():
    result = runner.invoke(app, ["validate", "examples/simple.py"])
    assert result.exit_code == 0
    assert "Valid" in result.output
    assert "prompt" in result.output
    assert "gen" in result.output


def test_validate_missing_file():
    result = runner.invoke(app, ["validate", "/nonexistent.py"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_validate_invalid_net(tmp_path):
    """Validate catches structural errors."""
    p = _write_eval(
        tmp_path,
        """\
        from peven.petri.schema import Arc, Marking, Net, Place, Transition
        net = Net(
            places=[Place(id="a"), Place(id="a")],
            transitions=[Transition(id="t", executor="agent")],
            arcs=[Arc(source="a", target="t")],
            initial_marking=Marking(tokens={}),
        )
    """,
    )
    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code == 1
    assert "Validation failed" in result.output


# -- run command ---------------------------------------------------------------


def test_run_with_mock(tmp_path):
    db = tmp_path / "test.db"
    mock_result = MagicMock()
    mock_result.output = "hello"
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_result)

    with (
        patch("peven.petri.executors.PydanticAgent", return_value=mock_agent),
        patch("peven.cli.commands.run.save", wraps=lambda r, **kw: save(r, db_path=db, **kw)),
    ):
        result = runner.invoke(app, ["run", "examples/simple.py"])

    assert result.exit_code == 0
    assert "Run " in result.output


def test_run_missing_file():
    result = runner.invoke(app, ["run", "/nonexistent.py"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_run_engine_exception():
    """Engine exception is caught and shows friendly error."""
    with patch(
        "peven.cli.commands.run.engine_run",
        side_effect=RuntimeError("boom"),
    ):
        result = runner.invoke(app, ["run", "examples/simple.py"])

    assert result.exit_code == 1
    assert "Execution failed" in result.output


def test_run_with_trace(tmp_path):
    """--trace flag renders execution trace."""
    db = tmp_path / "test.db"
    mock_result = MagicMock()
    mock_result.output = "hello"
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_result)

    with (
        patch("peven.petri.executors.PydanticAgent", return_value=mock_agent),
        patch("peven.cli.commands.run.save", wraps=lambda r, **kw: save(r, db_path=db, **kw)),
    ):
        result = runner.invoke(app, ["run", "examples/simple.py", "--trace"])

    assert result.exit_code == 0
    assert "gen" in result.output  # transition name appears in trace


# -- review command ------------------------------------------------------------


def test_review_not_found():
    result = runner.invoke(app, ["review", "nonexistent"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_review_all_empty():
    result = runner.invoke(app, ["review", "all"])
    assert result.exit_code == 0


def test_review_all_with_data(tmp_path):
    """'review all' renders a table when runs exist."""
    from peven.petri.types import GenerateOutput, RunResult, TransitionResult

    db = tmp_path / "test.db"
    results = [
        RunResult(
            status="completed",
            score=0.85,
            trace=[
                TransitionResult(
                    transition_id="gen", status="completed", output=GenerateOutput(text="hi")
                )
            ],
        )
    ]
    run_id = save(results, file="eval.py", db_path=db)

    with patch(
        "peven.cli.commands.review.list_runs",
        wraps=lambda **kw: __import__("peven.petri.store", fromlist=["list_runs"]).list_runs(
            db_path=db, **kw
        ),
    ):
        result = runner.invoke(app, ["review", "all"])

    assert result.exit_code == 0
    assert run_id in result.output


def test_review_show_run(tmp_path):
    """'review <id>' shows run details."""
    from peven.petri.types import GenerateOutput, RunResult, TransitionResult

    db = tmp_path / "test.db"
    results = [
        RunResult(
            status="completed",
            score=0.9,
            trace=[
                TransitionResult(
                    transition_id="gen", status="completed", output=GenerateOutput(text="hello")
                )
            ],
        )
    ]
    run_id = save(results, file="eval.py", db_path=db)

    with patch(
        "peven.cli.commands.review.get",
        wraps=lambda rid, **kw: __import__("peven.petri.store", fromlist=["get"]).get(
            rid, db_path=db
        ),
    ):
        result = runner.invoke(app, ["review", run_id])

    assert result.exit_code == 0
    assert run_id in result.output


def test_review_show_run_with_trace(tmp_path):
    """'review <id> --trace' shows trace details."""
    from peven.petri.types import GenerateOutput, RunResult, TransitionResult

    db = tmp_path / "test.db"
    results = [
        RunResult(
            status="completed",
            score=0.9,
            trace=[
                TransitionResult(
                    transition_id="gen", status="completed", output=GenerateOutput(text="hello")
                )
            ],
        )
    ]
    run_id = save(results, file="eval.py", db_path=db)

    with patch(
        "peven.cli.commands.review.get",
        wraps=lambda rid, **kw: __import__("peven.petri.store", fromlist=["get"]).get(
            rid, db_path=db
        ),
    ):
        result = runner.invoke(app, ["review", run_id, "--trace"])

    assert result.exit_code == 0
    assert "gen" in result.output
