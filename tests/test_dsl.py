"""Test DSL: NetBuilder, proxies, config helpers, >> chaining."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from peven.petri.dsl import NetBuilder, agent, judge
from peven.petri.engine import execute
from peven.petri.schema import GenerateConfig, JudgeConfig, Token
from peven.petri.types import GenerateOutput

# -- Config helpers ------------------------------------------------------------


def test_agent_helper():
    name, config = agent(model="opus", prompt="say {text}", system="be nice")
    assert name == "agent"
    assert isinstance(config, GenerateConfig)
    assert config.model == "opus"
    assert config.prompt_template == "say {text}"
    assert config.system_prompt == "be nice"


def test_judge_helper():
    rubric = [{"weight": 1.0, "requirement": "clear"}]
    name, config = judge(model="test", rubric=rubric, threshold=0.8)
    assert name == "judge"
    assert isinstance(config, JudgeConfig)
    assert config.rubric == [{"weight": 1.0, "requirement": "clear"}]
    assert config.pass_threshold == 0.8


# -- Topology ------------------------------------------------------------------


def test_simple_chain():
    """p >> t >> p2 creates two arcs."""
    n = NetBuilder()
    p1 = n.place("p1")
    t = n.transition("t", agent(model="m", prompt="{text}"))
    p2 = n.place("p2")

    p1 >> t >> p2

    net = n.build()
    assert len(net.places) == 2
    assert len(net.transitions) == 1
    assert len(net.arcs) == 2
    assert net.arcs[0].source == "p1" and net.arcs[0].target == "t"
    assert net.arcs[1].source == "t" and net.arcs[1].target == "p2"


def test_fork():
    """One place >> two transitions."""
    n = NetBuilder()
    p = n.place("p")
    ta = n.transition("a", agent(model="m", prompt="{text}"))
    tb = n.transition("b", agent(model="m", prompt="{text}"))
    out_a = n.place("out_a")
    out_b = n.place("out_b")

    p >> ta >> out_a
    p >> tb >> out_b

    net = n.build()
    arcs_from_p = [a for a in net.arcs if a.source == "p"]
    assert len(arcs_from_p) == 2
    assert {a.target for a in arcs_from_p} == {"a", "b"}


def test_join():
    """Two places >> one transition."""
    n = NetBuilder()
    left = n.place("left")
    right = n.place("right")
    t = n.transition("join", agent(model="m", prompt="{text}"))
    out = n.place("out")

    left >> t
    right >> t >> out

    net = n.build()
    arcs_to_t = [a for a in net.arcs if a.target == "join"]
    assert len(arcs_to_t) == 2
    assert {a.source for a in arcs_to_t} == {"left", "right"}


def test_cycle_topology():
    """Back-edge arc creates a cycle in the topology."""
    n = NetBuilder()
    prompt = n.place("prompt")
    scored = n.place("scored")
    gen = n.transition("gen", agent(model="m", prompt="{text}"))
    loop = n.transition("loop", agent(model="m", prompt="{text}"))

    prompt >> gen >> scored
    scored >> loop >> prompt  # back-edge

    net = n.build()
    back_arc = [a for a in net.arcs if a.source == "loop" and a.target == "prompt"]
    assert len(back_arc) == 1


# -- Guards --------------------------------------------------------------------


def test_guard_on_transition():
    """.when() sets the guard on the transition."""
    n = NetBuilder()
    p = n.place("p")

    def guard_fn(tokens):
        return tokens[0].score >= 0.7

    t = n.transition("t", agent(model="m", prompt="{text}"))
    out = n.place("out")

    p >> t.when(guard_fn) >> out

    net = n.build()
    assert net.transitions[0].when is guard_fn


def test_guard_chaining():
    """Guard works in a chain: p >> t.when(...) >> p2."""
    n = NetBuilder()
    scored = n.place("scored")
    accept = n.transition("accept", agent(model="m", prompt="{text}"))
    reject = n.transition("reject", agent(model="m", prompt="{text}"))
    good = n.place("good")
    bad = n.place("bad")

    scored >> accept.when(lambda t: t[0].score >= 0.7) >> good
    scored >> reject.when(lambda t: t[0].score < 0.7) >> bad

    net = n.build()
    assert net.transitions[0].when is not None
    assert net.transitions[1].when is not None
    arcs_from_scored = [a for a in net.arcs if a.source == "scored"]
    assert len(arcs_from_scored) == 2


# -- Tokens and capacity ------------------------------------------------------


def test_initial_tokens():
    """.token() adds to initial marking."""
    n = NetBuilder()
    p = n.place("p")
    n.transition("t", agent(model="m", prompt="{text}"))

    p.token(Token())
    p.token(GenerateOutput(text="hello"))

    net = n.build()
    assert len(net.initial_marking.tokens["p"]) == 2


def test_capacity():
    """place with capacity is set on Place."""
    n = NetBuilder()
    n.place("limited", capacity=3)
    net = n.build()
    assert net.places[0].capacity == 3


# -- Retries -------------------------------------------------------------------


def test_dsl_retries_on_transition():
    """retries parameter flows through DSL to Transition."""
    n = NetBuilder()
    p = n.place("in")
    out = n.place("out")
    t = n.transition("t", agent(model="m", prompt="{text}"), retries=3)
    p >> t >> out

    net = n.build()
    assert net.transitions[0].retries == 3


@pytest.mark.asyncio
async def test_dsl_retries_end_to_end():
    """DSL-built net with retries recovers from transient failure."""
    attempts = {"count": 0}

    async def flaky(self, inputs, config):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("transient")
        return GenerateOutput(text="recovered")

    n = NetBuilder()
    p = n.place("in")
    out = n.place("out")
    t = n.transition("t", agent(model="test", prompt="{text}"), retries=2)
    p >> t >> out
    p.token(Token())

    from unittest.mock import patch

    with patch("peven.petri.executors.Agent.execute", flaky):
        [result] = await execute(n.build())

    assert result.status == "completed"
    assert attempts["count"] == 2


# -- Type enforcement ----------------------------------------------------------


def test_place_to_place_raises():
    n = NetBuilder()
    a = n.place("a")
    b = n.place("b")
    with pytest.raises(TypeError, match="Transition"):
        a >> b


def test_transition_to_transition_raises():
    n = NetBuilder()
    a = n.transition("a", agent(model="m", prompt="{text}"))
    b = n.transition("b", agent(model="m", prompt="{text}"))
    with pytest.raises(TypeError, match="Place"):
        a >> b


# -- End-to-end: build + run --------------------------------------------------


def _mock_agent():
    mock_result = MagicMock()
    mock_result.output = "response"
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_result)
    return patch("peven.petri.executors.PydanticAgent", return_value=mock_agent)


def _mock_rubric(score: float):
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


@pytest.mark.asyncio
async def test_build_and_run():
    """Build a gen→judge net via DSL, run it, verify RunResult."""
    n = NetBuilder()

    start = n.place("start")
    mid = n.place("mid")
    end = n.place("end")

    gen = n.transition("gen", agent(model="test", prompt="{text}"))
    jdg = n.transition("jdg", judge(model="test", rubric=[{"weight": 1.0, "requirement": "ok"}]))

    start >> gen >> mid >> jdg >> end
    start.token(Token())

    net = n.build()

    with _mock_agent(), _mock_rubric(0.92):
        [result] = await execute(net)

    assert result.status == "completed"
    assert result.score == 0.92
    assert len(result.trace) == 2
