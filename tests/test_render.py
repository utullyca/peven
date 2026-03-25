"""Test rich rendering: results table, traces, net topology."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from peven.petri.dsl import NetBuilder, agent, judge
from peven.petri.render import _status, render, render_net
from peven.petri.types import GenerateOutput, JudgeOutput, RunResult, TransitionResult


def _console() -> tuple[Console, StringIO]:
    buf = StringIO()
    con = Console(file=buf, force_terminal=True, width=120)
    return con, buf


# -- render() ------------------------------------------------------------------


def test_render_empty():
    con, buf = _console()
    render([], console=con)
    assert "No results" in buf.getvalue()


def test_render_single_run_shows_trace():
    """Single run (no run_id) renders trace directly, no table."""
    con, buf = _console()
    results = [
        RunResult(
            status="completed",
            score=0.85,
            trace=[
                TransitionResult(
                    transition_id="gen",
                    status="completed",
                    output=GenerateOutput(text="hello world"),
                ),
                TransitionResult(
                    transition_id="jdg",
                    status="completed",
                    output=JudgeOutput(score=0.85),
                ),
            ],
        )
    ]
    render(results, console=con)
    out = buf.getvalue()
    assert "gen" in out
    assert "jdg" in out
    assert "0.85" in out
    assert "run_id" not in out.lower().split("run ")[0]  # no table header


def test_render_batch_shows_table():
    """Multiple runs render a summary table."""
    con, buf = _console()
    results = [
        RunResult(
            run_id="r1",
            status="completed",
            score=0.9,
            trace=[TransitionResult(transition_id="t", status="completed")],
        ),
        RunResult(
            run_id="r2",
            status="failed",
            error="boom",
            trace=[TransitionResult(transition_id="t", status="failed", error="boom")],
        ),
        RunResult(
            run_id="r3",
            status="completed",
            score=0.8,
            trace=[TransitionResult(transition_id="t", status="completed")],
        ),
    ]
    render(results, console=con)
    out = buf.getvalue()
    assert "r1" in out
    assert "r2" in out
    assert "r3" in out
    assert "boom" in out
    # Rich markup splits tokens with ANSI codes — check parts separately
    assert "3" in out and "runs" in out
    assert "2" in out and "completed" in out
    assert "failed" in out
    assert "0.8500" in out  # mean of 0.9 and 0.8


def test_render_batch_with_trace():
    """trace=True shows per-run traces after the table."""
    con, buf = _console()
    results = [
        RunResult(
            run_id="r1",
            status="completed",
            score=0.7,
            trace=[
                TransitionResult(
                    transition_id="gen",
                    status="completed",
                    output=GenerateOutput(text="output"),
                    run_id="r1",
                ),
            ],
        ),
    ]
    render(results, trace=True, console=con)
    out = buf.getvalue()
    assert "r1" in out
    assert "gen" in out
    assert "output" in out


def test_render_failed_trace():
    """Failed transitions show error in trace."""
    con, buf = _console()
    results = [
        RunResult(
            status="failed",
            error="exploded",
            trace=[
                TransitionResult(
                    transition_id="t1",
                    status="completed",
                    output=GenerateOutput(text="ok"),
                ),
                TransitionResult(transition_id="t2", status="failed", error="exploded"),
            ],
        )
    ]
    render(results, console=con)
    out = buf.getvalue()
    assert "✗" in out
    assert "exploded" in out


def test_render_truncates_long_text():
    """Long output text is truncated."""
    con, buf = _console()
    long_text = "a" * 200
    results = [
        RunResult(
            status="completed",
            trace=[
                TransitionResult(
                    transition_id="gen",
                    status="completed",
                    output=GenerateOutput(text=long_text),
                ),
            ],
        )
    ]
    render(results, console=con)
    out = buf.getvalue()
    assert "…" in out


def test_render_judge_rubric_breakdown():
    """Trace view shows per-criterion rubric breakdown for judge outputs."""
    from rubric import CriterionReport

    con, buf = _console()
    results = [
        RunResult(
            status="completed",
            score=0.85,
            trace=[
                TransitionResult(
                    transition_id="jdg",
                    status="completed",
                    output=JudgeOutput(
                        score=0.85,
                        report=[
                            CriterionReport(
                                weight=1.0, requirement="clear writing", verdict="MET", reason="ok"
                            ),
                            CriterionReport(
                                weight=1.0,
                                requirement="factual accuracy",
                                verdict="UNMET",
                                reason="wrong date",
                            ),
                        ],
                    ),
                ),
            ],
        )
    ]
    render(results, console=con)
    out = buf.getvalue()
    assert "MET" in out, "should show MET verdict"
    assert "UNMET" in out, "should show UNMET verdict"
    assert "clear writing" in out, "should show requirement text"
    assert "wrong date" in out, "should show reason"


# -- render_net() --------------------------------------------------------------


def test_render_net_simple():
    """Simple chain topology renders correctly."""
    n = NetBuilder()
    p1 = n.place("start")
    t = n.transition("gen", agent(model="m", prompt="{text}"))
    p2 = n.place("end")
    p1 >> t >> p2

    con, buf = _console()
    render_net(n.build(), console=con)
    out = buf.getvalue()
    assert "start" in out
    assert "gen" in out
    assert "end" in out
    assert "2 places" in out
    assert "1 transition" in out


def test_render_net_fork_join():
    """Fork/join topology shows branching."""
    n = NetBuilder()
    start = n.place("start")
    left = n.place("left")
    right = n.place("right")
    end = n.place("end")
    go_l = n.transition("go_left", agent(model="m", prompt="{text}"))
    go_r = n.transition("go_right", agent(model="m", prompt="{text}"))
    join = n.transition("merge", agent(model="m", prompt="{text}"))

    start >> go_l >> left >> join
    start >> go_r >> right >> join >> end

    con, buf = _console()
    render_net(n.build(), console=con)
    out = buf.getvalue()
    assert "go_left" in out
    assert "go_right" in out
    assert "merge" in out


def test_render_net_cycle():
    """Cycle shows back-reference with ↩."""
    n = NetBuilder()
    prompt = n.place("prompt")
    scored = n.place("scored")
    final = n.place("final")
    gen = n.transition("gen", agent(model="m", prompt="{text}"))
    jdg = n.transition("judge", judge(model="test", rubric=[{"weight": 1.0, "requirement": "ok"}]))
    loop = n.transition("loop", agent(model="m", prompt="{text}"))
    done = n.transition("done", agent(model="m", prompt="{text}"))

    prompt >> gen >> scored >> jdg
    scored >> loop >> prompt  # cycle
    scored >> done >> final

    con, buf = _console()
    render_net(n.build(), console=con)
    out = buf.getvalue()
    assert "↩" in out  # back-reference to prompt
    assert "prompt" in out


def test_status_unknown():
    """Unknown status gets no styling."""
    text = _status("unknown")
    assert text.plain == "unknown"


def test_render_no_output_transition():
    """Transition with no output and no error shows dash."""
    con, buf = _console()
    results = [
        RunResult(
            status="completed",
            trace=[TransitionResult(transition_id="t", status="completed")],
        )
    ]
    render(results, console=con)
    assert "—" in buf.getvalue()


def test_render_net_with_guard():
    """Transitions with guards show 'when' in metadata."""
    n = NetBuilder()
    scored = n.place("scored")
    good = n.place("good")
    t = n.transition("accept", agent(model="m", prompt="{text}"))
    scored >> t.when(lambda tokens: tokens[0].score >= 0.7) >> good

    con, buf = _console()
    render_net(n.build(), console=con)
    out = buf.getvalue()
    assert "when" in out
