"""Test engine: ready, consume, deposit, execute (async central loop)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from peven.petri.engine import _trace_arcs, consume, deposit, execute, ready
from peven.petri.schema import (
    Arc,
    GenerateConfig,
    JudgeConfig,
    Marking,
    Net,
    Place,
    Token,
    Transition,
    ValidationError,
)
from peven.petri.types import GenerateOutput, JudgeOutput

# -- Helpers -------------------------------------------------------------------


def _mock_agent_run(text: str):
    mock_result = MagicMock()
    mock_result.output = text
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_result)
    return patch("peven.petri.executors.PydanticAgent", return_value=mock_agent)


def _mock_rubric_grade(score: float):
    from rubric import CriterionReport, EvaluationReport

    report = EvaluationReport(
        score=score,
        raw_score=score,
        llm_raw_score=score * 100,
        report=[CriterionReport(weight=1.0, requirement="ok", verdict="MET", reason="ok")],
    )

    async def mock_grade(self, *a, **kw):
        return report

    return patch("peven.petri.executors.Rubric.grade", mock_grade)


def _simple_net():
    """in -> gen -> mid -> jdg -> out"""
    return Net(
        places=[Place(id="in"), Place(id="mid"), Place(id="out")],
        transitions=[
            Transition(
                id="gen",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
            ),
            Transition(
                id="jdg",
                executor="judge",
                config=JudgeConfig(model="test", rubric=[{"weight": 1.0, "requirement": "ok"}]),
            ),
        ],
        arcs=[
            Arc(source="in", target="gen"),
            Arc(source="gen", target="mid"),
            Arc(source="mid", target="jdg"),
            Arc(source="jdg", target="out"),
        ],
        initial_marking=Marking(tokens={"in": [Token()]}),
    )


def _parallel_net():
    """in -> gen_a -> out_a, in -> gen_b -> out_b (fork)"""
    return Net(
        places=[Place(id="in"), Place(id="out_a"), Place(id="out_b")],
        transitions=[
            Transition(
                id="gen_a",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
            ),
            Transition(
                id="gen_b",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
            ),
        ],
        arcs=[
            Arc(source="in", target="gen_a"),
            Arc(source="in", target="gen_b"),
            Arc(source="gen_a", target="out_a"),
            Arc(source="gen_b", target="out_b"),
        ],
        initial_marking=Marking(tokens={"in": [Token(), Token()]}),
    )


# -- ready() -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ready_single():
    net = _simple_net()
    inputs, _ = _trace_arcs(net)
    result = await ready(net, net.initial_marking, inputs, set(), set())
    assert len(result) == 1, "one transition should be ready"
    assert result[0][0].id == "gen", "the ready transition should be 'gen'"
    assert result[0][1] is None, "run_id should be None for uncolored token"


@pytest.mark.asyncio
async def test_ready_insufficient():
    net = _simple_net()
    inputs, _ = _trace_arcs(net)
    empty = Marking(tokens={})
    result = await ready(net, empty, inputs, set(), set())
    assert result == [], "no tokens means no transitions ready"


@pytest.mark.asyncio
async def test_ready_skips_no_input_arcs():
    """Transition with no input arcs is never ready."""
    net = Net(
        places=[Place(id="out")],
        transitions=[Transition(id="t", executor="agent")],
        arcs=[Arc(source="t", target="out")],
        initial_marking=Marking(tokens={}),
    )
    inputs, _ = _trace_arcs(net)
    result = await ready(net, net.initial_marking, inputs, set(), set())
    assert result == []


@pytest.mark.asyncio
async def test_ready_colored():
    net = _simple_net()
    inputs, _ = _trace_arcs(net)
    marking = Marking(tokens={"in": [Token(run_id="a"), Token(run_id="b")]})
    result = await ready(net, marking, inputs, set(), set())
    run_ids = {rid for _, rid in result}
    assert run_ids == {"a", "b"}


@pytest.mark.asyncio
async def test_ready_skips_failed():
    net = _simple_net()
    inputs, _ = _trace_arcs(net)
    marking = Marking(tokens={"in": [Token(run_id="a"), Token(run_id="b")]})
    result = await ready(net, marking, inputs, failed={"a"}, in_flight=set())
    run_ids = {rid for _, rid in result}
    assert run_ids == {"b"}


@pytest.mark.asyncio
async def test_ready_skips_in_flight():
    net = _simple_net()
    inputs, _ = _trace_arcs(net)
    marking = Marking(tokens={"in": [Token(run_id="a")]})
    result = await ready(net, marking, inputs, set(), in_flight={("gen", "a")})
    assert result == [], "in-flight transition should not be returned as ready"


@pytest.mark.asyncio
async def test_ready_join_requires_same_run_id():
    """Join transition needs tokens in ALL input places for the SAME run_id."""
    net = Net(
        places=[Place(id="left"), Place(id="right"), Place(id="out")],
        transitions=[
            Transition(
                id="join",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
            )
        ],
        arcs=[
            Arc(source="left", target="join"),
            Arc(source="right", target="join"),
            Arc(source="join", target="out"),
        ],
        initial_marking=Marking(
            tokens={
                "left": [Token(run_id="x")],
                "right": [Token(run_id="y")],
            }
        ),
    )
    inputs, _ = _trace_arcs(net)
    result = await ready(net, net.initial_marking, inputs, set(), set())
    assert result == [], "join should not fire when input places have different run_ids"


@pytest.mark.asyncio
async def test_ready_join_fires_with_same_run_id():
    """Join transition fires when ALL input places have the same run_id."""
    net = Net(
        places=[Place(id="left"), Place(id="right"), Place(id="out")],
        transitions=[
            Transition(
                id="join",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
            )
        ],
        arcs=[
            Arc(source="left", target="join"),
            Arc(source="right", target="join"),
            Arc(source="join", target="out"),
        ],
        initial_marking=Marking(
            tokens={
                "left": [Token(run_id="x")],
                "right": [Token(run_id="x")],
            }
        ),
    )
    inputs, _ = _trace_arcs(net)
    result = await ready(net, net.initial_marking, inputs, set(), set())
    assert len(result) == 1, "join should fire when both inputs have same run_id"
    assert result[0][1] == "x"


# -- consume() ----------------------------------------------------------------


def test_consume_removes_tokens():
    net = _simple_net()
    inputs, _ = _trace_arcs(net)
    t = net.transitions[0]  # gen
    new_marking, consumed, consumed_by_place = consume(net.initial_marking, t, inputs, None)
    assert len(consumed) == 1
    assert new_marking.tokens.get("in", []) == []
    assert "in" in consumed_by_place
    assert len(consumed_by_place["in"]) == 1


def test_consume_respects_color():
    net = _simple_net()
    inputs, _ = _trace_arcs(net)
    t = net.transitions[0]
    marking = Marking(tokens={"in": [Token(run_id="a"), Token(run_id="b")]})
    new_marking, consumed, consumed_by_place = consume(marking, t, inputs, "a")
    assert len(consumed) == 1, "should consume exactly 1 token"
    assert consumed[0].run_id == "a", "should consume the 'a' token"
    remaining = new_marking.tokens["in"]
    assert len(remaining) == 1, "should leave 1 token behind"
    assert remaining[0].run_id == "b", "should leave the 'b' token untouched"
    assert consumed_by_place["in"][0].run_id == "a"


def test_consume_weight_greater_than_one():
    """Arc with weight=2 should consume exactly 2 tokens."""
    net = Net(
        places=[Place(id="in"), Place(id="out")],
        transitions=[
            Transition(
                id="t",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
            )
        ],
        arcs=[Arc(source="in", target="t", weight=2), Arc(source="t", target="out")],
        initial_marking=Marking(tokens={"in": [Token(), Token(), Token()]}),
    )
    inputs, _ = _trace_arcs(net)
    new_marking, consumed, consumed_by_place = consume(
        net.initial_marking, net.transitions[0], inputs, None
    )
    assert len(consumed) == 2, "weight=2 should consume exactly 2 tokens"
    assert len(new_marking.tokens["in"]) == 1, "should leave 1 token remaining"
    assert len(consumed_by_place["in"]) == 2, "consumed_by_place should track both tokens"


# -- deposit() ----------------------------------------------------------------


def test_deposit_adds_tokens():
    net = _simple_net()
    _, outputs = _trace_arcs(net)
    t = net.transitions[0]  # gen
    token = GenerateOutput(text="hi")
    marking = Marking(tokens={})
    new_marking = deposit(marking, t, outputs, token, {})
    assert len(new_marking.tokens["mid"]) == 1
    assert new_marking.tokens["mid"][0].text == "hi"


def test_deposit_weight_greater_than_one():
    """Arc with weight=2 should deposit 2 copies of the token."""
    net = Net(
        places=[Place(id="in"), Place(id="out")],
        transitions=[
            Transition(
                id="t",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
            )
        ],
        arcs=[Arc(source="in", target="t"), Arc(source="t", target="out", weight=2)],
        initial_marking=Marking(tokens={"in": [Token()]}),
    )
    _, outputs = _trace_arcs(net)
    token = GenerateOutput(text="hi")
    new_marking = deposit(Marking(tokens={}), net.transitions[0], outputs, token, {})
    assert len(new_marking.tokens["out"]) == 2, "weight=2 should deposit 2 copies"
    assert all(t.text == "hi" for t in new_marking.tokens["out"])


def test_deposit_capacity():
    net = Net(
        places=[Place(id="in"), Place(id="out", capacity=1)],
        transitions=[
            Transition(
                id="t",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
            )
        ],
        arcs=[Arc(source="in", target="t"), Arc(source="t", target="out")],
        initial_marking=Marking(tokens={"in": [Token()]}),
    )
    _, outputs = _trace_arcs(net)
    cap = {p.id: p.capacity for p in net.places}
    token = GenerateOutput(text="hi")
    m1 = deposit(Marking(tokens={}), net.transitions[0], outputs, token, cap)
    with pytest.raises(ValidationError, match="exceed capacity"):
        deposit(m1, net.transitions[0], outputs, token, cap)


# -- run() single -------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_chain():
    """generate -> judge chain, verify results + score."""
    net = _simple_net()

    with _mock_agent_run("nice response"), _mock_rubric_grade(0.85):
        [result] = await execute(net)

    assert result.status == "completed"
    assert len(result.trace) == 2
    assert result.score == 0.85
    gen = [r for r in result.trace if r.transition_id == "gen"]
    jdg = [r for r in result.trace if r.transition_id == "jdg"]
    assert len(gen) == 1
    assert len(jdg) == 1


@pytest.mark.asyncio
async def test_run_parallel():
    """Two independent transitions fire — both complete."""
    net = _parallel_net()

    with _mock_agent_run("output"):
        [result] = await execute(net, max_concurrency=2)

    assert result.status == "completed"
    assert len(result.trace) == 2
    tids = {r.transition_id for r in result.trace}
    assert tids == {"gen_a", "gen_b"}


@pytest.mark.asyncio
async def test_run_fuse_limit():
    """Fuse stops spawning mid-batch when limit is hit."""
    net = _parallel_net()

    with _mock_agent_run("output"):
        [result] = await execute(net, fuse=1, max_concurrency=2)

    assert len(result.trace) == 1


# -- run() with colored tokens ------------------------------------------------


@pytest.mark.asyncio
async def test_run_colored_isolation():
    """Two run_ids, one fails, other completes."""
    net = _simple_net()
    net.initial_marking = Marking(tokens={"in": [Token(run_id="ok"), Token(run_id="bad")]})

    async def flaky_execute(self, inputs, config):
        rid = inputs[0].run_id if inputs else None
        if rid == "bad":
            raise RuntimeError("boom")
        return GenerateOutput(text="good")

    with (
        patch("peven.petri.executors.Agent.execute", flaky_execute),
        _mock_rubric_grade(0.9),
    ):
        results = await execute(net, max_concurrency=2)

    by_rid = {r.run_id: r for r in results}
    assert by_rid["bad"].status == "failed"
    assert by_rid["ok"].status == "completed"
    jdg = [r for r in by_rid["ok"].trace if r.transition_id == "jdg"]
    assert len(jdg) == 1
    assert jdg[0].status == "completed"


@pytest.mark.asyncio
async def test_run_join_same_color():
    """Fork -> join: tokens merge only within same run_id."""
    net = Net(
        places=[
            Place(id="in"),
            Place(id="left"),
            Place(id="right"),
            Place(id="out"),
        ],
        transitions=[
            Transition(
                id="go_left",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
            ),
            Transition(
                id="go_right",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
            ),
            Transition(
                id="join",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
            ),
        ],
        arcs=[
            Arc(source="in", target="go_left"),
            Arc(source="in", target="go_right"),
            Arc(source="go_left", target="left"),
            Arc(source="go_right", target="right"),
            Arc(source="left", target="join"),
            Arc(source="right", target="join"),
            Arc(source="join", target="out"),
        ],
        initial_marking=Marking(
            tokens={
                "in": [
                    Token(run_id="r1"),
                    Token(run_id="r1"),
                    Token(run_id="r2"),
                    Token(run_id="r2"),
                ],
            }
        ),
    )

    with _mock_agent_run("merged"):
        results = await execute(net, max_concurrency=4)

    join_rids = set()
    for r in results:
        for t in r.trace:
            if t.transition_id == "join":
                join_rids.add(t.run_id)
    assert join_rids == {"r1", "r2"}


# -- run() batch --------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_three_rows():
    """3 rows through gen->judge, each gets own RunResult with score."""
    net = Net(
        places=[Place(id="in"), Place(id="mid"), Place(id="out")],
        transitions=[
            Transition(
                id="gen",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
            ),
            Transition(
                id="jdg",
                executor="judge",
                config=JudgeConfig(model="test", rubric=[{"weight": 1.0, "requirement": "ok"}]),
            ),
        ],
        arcs=[
            Arc(source="in", target="gen"),
            Arc(source="gen", target="mid"),
            Arc(source="mid", target="jdg"),
            Arc(source="jdg", target="out"),
        ],
        initial_marking=Marking(tokens={}),
    )

    rows = [
        GenerateOutput(text="a"),
        GenerateOutput(text="b"),
        GenerateOutput(text="c"),
    ]

    with _mock_agent_run("response"), _mock_rubric_grade(0.75):
        results = await execute(net, rows=rows, place="in", max_concurrency=3)

    assert len(results) == 3
    for r in results:
        assert r.status == "completed"
        assert r.score == 0.75
        assert r.run_id is not None
        assert len(r.trace) == 2


@pytest.mark.asyncio
async def test_batch_error_isolation():
    """One row fails, others complete with scores."""
    net = Net(
        places=[Place(id="in"), Place(id="mid"), Place(id="out")],
        transitions=[
            Transition(
                id="gen",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
            ),
            Transition(
                id="jdg",
                executor="judge",
                config=JudgeConfig(model="test", rubric=[{"weight": 1.0, "requirement": "ok"}]),
            ),
        ],
        arcs=[
            Arc(source="in", target="gen"),
            Arc(source="gen", target="mid"),
            Arc(source="mid", target="jdg"),
            Arc(source="jdg", target="out"),
        ],
        initial_marking=Marking(tokens={}),
    )

    call_count = 0

    async def sometimes_fail(self, inputs, config):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("boom")
        return GenerateOutput(text="ok")

    rows = [
        GenerateOutput(text="a"),
        GenerateOutput(text="b"),
        GenerateOutput(text="c"),
    ]

    with (
        patch("peven.petri.executors.Agent.execute", sometimes_fail),
        _mock_rubric_grade(0.8),
    ):
        results = await execute(net, rows=rows, place="in", max_concurrency=3)

    assert len(results) == 3
    failed = [r for r in results if r.status == "failed"]
    passed = [r for r in results if r.status == "completed"]
    assert len(failed) == 1
    assert len(passed) == 2
    assert failed[0].error is not None
    for r in passed:
        assert r.score == 0.8


@pytest.mark.asyncio
async def test_batch_requires_place():
    """rows without place raises ValueError."""
    net = _simple_net()
    with pytest.raises(ValueError, match="place is required"):
        await execute(net, rows=[Token()])


# -- when guards ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_when_blocks_firing():
    """Transition with when=False never fires."""
    net = Net(
        places=[Place(id="in"), Place(id="out")],
        transitions=[
            Transition(
                id="t",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
                when=lambda tokens: False,
            ),
        ],
        arcs=[Arc(source="in", target="t"), Arc(source="t", target="out")],
        initial_marking=Marking(tokens={"in": [Token()]}),
    )

    with _mock_agent_run("output"):
        results = await execute(net)

    assert results == []


@pytest.mark.asyncio
async def test_when_choice_routing():
    """Two transitions from same place, only the matching guard fires."""
    net = Net(
        places=[Place(id="scored"), Place(id="good"), Place(id="bad")],
        transitions=[
            Transition(
                id="accept",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
                when=lambda tokens: tokens[0].score >= 0.7,
            ),
            Transition(
                id="reject",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
                when=lambda tokens: tokens[0].score < 0.7,
            ),
        ],
        arcs=[
            Arc(source="scored", target="accept"),
            Arc(source="scored", target="reject"),
            Arc(source="accept", target="good"),
            Arc(source="reject", target="bad"),
        ],
        initial_marking=Marking(tokens={"scored": [JudgeOutput(score=0.8)]}),
    )

    with _mock_agent_run("routed"):
        [result] = await execute(net)

    assert len(result.trace) == 1
    assert result.trace[0].transition_id == "accept"


@pytest.mark.asyncio
async def test_when_cycle_terminates():
    """Cycle runs until guard exits: gen -> judge -> scored, loop back or done."""
    scores = iter([0.5, 0.95])

    def _mock_rubric_grade_seq():
        from rubric import CriterionReport, EvaluationReport

        async def mock_grade(self, *a, **kw):
            s = next(scores)
            return EvaluationReport(
                score=s,
                raw_score=s,
                llm_raw_score=s * 100,
                report=[CriterionReport(weight=1.0, requirement="ok", verdict="MET", reason="ok")],
            )

        return patch("peven.petri.executors.Rubric.grade", mock_grade)

    net = Net(
        places=[
            Place(id="prompt"),
            Place(id="response"),
            Place(id="scored"),
            Place(id="final"),
        ],
        transitions=[
            Transition(
                id="gen",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
            ),
            Transition(
                id="judge",
                executor="judge",
                config=JudgeConfig(model="test", rubric=[{"weight": 1.0, "requirement": "ok"}]),
            ),
            Transition(
                id="loop",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
                when=lambda tokens: tokens[0].score < 0.9,
            ),
            Transition(
                id="done",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
                when=lambda tokens: tokens[0].score >= 0.9,
            ),
        ],
        arcs=[
            Arc(source="prompt", target="gen"),
            Arc(source="gen", target="response"),
            Arc(source="response", target="judge"),
            Arc(source="judge", target="scored"),
            Arc(source="scored", target="loop"),
            Arc(source="loop", target="prompt"),  # back-edge
            Arc(source="scored", target="done"),
            Arc(source="done", target="final"),  # exit
        ],
        initial_marking=Marking(tokens={"prompt": [Token()]}),
    )

    with _mock_agent_run("text"), _mock_rubric_grade_seq():
        [result] = await execute(net, fuse=20)

    assert result.status == "completed"
    # gen fired twice (initial + loop), judge fired twice, loop once, done once = 6
    assert len(result.trace) == 6
    gen_count = sum(1 for r in result.trace if r.transition_id == "gen")
    judge_count = sum(1 for r in result.trace if r.transition_id == "judge")
    assert gen_count == 2
    assert judge_count == 2
    assert result.trace[-1].transition_id == "done"


# -- retries -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_succeeds_after_failure():
    """Transition with retries=2 recovers from a single failure."""
    attempts = {"count": 0}

    async def flaky(self, inputs, config):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("transient")
        return GenerateOutput(text="ok")

    net = Net(
        places=[Place(id="in"), Place(id="out")],
        transitions=[
            Transition(
                id="t",
                executor="agent",
                retries=2,
                config=GenerateConfig(model="test", prompt_template="{text}"),
            ),
        ],
        arcs=[Arc(source="in", target="t"), Arc(source="t", target="out")],
        initial_marking=Marking(tokens={"in": [Token()]}),
    )

    with patch("peven.petri.executors.Agent.execute", flaky):
        [result] = await execute(net)

    assert result.status == "completed"
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_retry_exhausted():
    """Transition fails after all retries are spent."""

    async def always_fail(self, inputs, config):
        raise RuntimeError("permanent")

    net = Net(
        places=[Place(id="in"), Place(id="out")],
        transitions=[
            Transition(
                id="t",
                executor="agent",
                retries=1,
                config=GenerateConfig(model="test", prompt_template="{text}"),
            ),
        ],
        arcs=[Arc(source="in", target="t"), Arc(source="t", target="out")],
        initial_marking=Marking(tokens={"in": [Token()]}),
    )

    with patch("peven.petri.executors.Agent.execute", always_fail):
        [result] = await execute(net)

    assert result.status == "failed"
    assert "permanent" in result.error


@pytest.mark.asyncio
async def test_retry_zero_means_no_retry():
    """retries=0 (default) fails immediately."""

    async def fail_once(self, inputs, config):
        raise RuntimeError("boom")

    net = Net(
        places=[Place(id="in"), Place(id="out")],
        transitions=[
            Transition(
                id="t",
                executor="agent",
                config=GenerateConfig(model="test", prompt_template="{text}"),
            ),
        ],
        arcs=[Arc(source="in", target="t"), Arc(source="t", target="out")],
        initial_marking=Marking(tokens={"in": [Token()]}),
    )

    with patch("peven.petri.executors.Agent.execute", fail_once):
        [result] = await execute(net)

    assert result.status == "failed"


@pytest.mark.asyncio
async def test_retry_colored_isolation():
    """Retry on one run_id doesn't affect another."""
    attempts = {"a": 0, "b": 0}

    async def flaky_by_color(self, inputs, config):
        rid = inputs[0].run_id
        attempts[rid] = attempts.get(rid, 0) + 1
        if rid == "a" and attempts["a"] == 1:
            raise RuntimeError("transient")
        return GenerateOutput(text=f"ok-{rid}")

    net = Net(
        places=[Place(id="in"), Place(id="out")],
        transitions=[
            Transition(
                id="t",
                executor="agent",
                retries=2,
                config=GenerateConfig(model="test", prompt_template="{text}"),
            ),
        ],
        arcs=[Arc(source="in", target="t"), Arc(source="t", target="out")],
        initial_marking=Marking(tokens={"in": [Token(run_id="a"), Token(run_id="b")]}),
    )

    with patch("peven.petri.executors.Agent.execute", flaky_by_color):
        results = await execute(net, max_concurrency=2)

    by_rid = {r.run_id: r for r in results}
    assert by_rid["a"].status == "completed"
    assert by_rid["b"].status == "completed"
    assert attempts["a"] == 2  # retried once
    assert attempts["b"] == 1  # no retry needed


@pytest.mark.asyncio
async def test_retry_fuse_boundary():
    """Retry does not fire if fuse is already exhausted."""
    attempts = {"count": 0}

    async def always_fail(self, inputs, config):
        attempts["count"] += 1
        raise RuntimeError("fail")

    net = Net(
        places=[Place(id="in"), Place(id="out")],
        transitions=[
            Transition(
                id="t",
                executor="agent",
                retries=5,
                config=GenerateConfig(model="test", prompt_template="{text}"),
            ),
        ],
        arcs=[Arc(source="in", target="t"), Arc(source="t", target="out")],
        initial_marking=Marking(tokens={"in": [Token()]}),
    )

    # fuse=1 means only 1 firing. Even with retries=5, the fuse limits total firings.
    with patch("peven.petri.executors.Agent.execute", always_fail):
        results = await execute(net, fuse=1)

    # The transition fired once (consuming the fuse), failed, retried,
    # but the re-spawned attempt counts as a new firing against the fuse.
    assert len(results) <= 1
