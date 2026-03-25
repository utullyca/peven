"""Integration tests: concurrency, complex topologies, edge cases, failure modes.

Tests use custom executors with asyncio.sleep and execution logs to prove
the engine runs transitions in parallel, respects semaphore limits, handles
complex topologies, and fails gracefully.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from peven.petri.engine import execute
from peven.petri.executors import register
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

# -- Reusable test executors ---------------------------------------------------


class PassExecutor:
    """Returns GenerateOutput with configurable text."""

    def __init__(self, text: str = "ok"):
        self._text = text

    async def execute(self, inputs, config):
        return GenerateOutput(text=self._text)


class SlowExecutor:
    """Sleeps then returns. Records start/end for concurrency assertions."""

    def __init__(self, delay: float = 0.05):
        self.log: list[tuple[str, str, float]] = []
        self.delay = delay

    async def execute(self, inputs, config):
        tid = config.prompt_template if config else "?"
        self.log.append(("start", tid, time.monotonic()))
        await asyncio.sleep(self.delay)
        self.log.append(("end", tid, time.monotonic()))
        return GenerateOutput(text=tid)


class ScoreExecutor:
    """Returns JudgeOutput with a fixed score."""

    def __init__(self, score: float = 0.8):
        self._score = score

    async def execute(self, inputs, config):
        return JudgeOutput(score=self._score)


class SeqScoreExecutor:
    """Returns JudgeOutput with scores from a sequence."""

    def __init__(self, scores: list[float]):
        self._scores = iter(scores)

    async def execute(self, inputs, config):
        return JudgeOutput(score=next(self._scores))


# ==============================================================================
# CONCURRENCY PROOFS
# ==============================================================================


@pytest.mark.asyncio
async def test_parallel_actually_concurrent():
    """Two transitions sleep 0.1s each. If parallel, total < 0.18s."""
    logger = SlowExecutor(delay=0.1)
    register("slow", logger)

    net = Net(
        places=[Place(id="in"), Place(id="out_a"), Place(id="out_b")],
        transitions=[
            Transition(
                id="a",
                executor="slow",
                config=GenerateConfig(model="t", prompt_template="a"),
            ),
            Transition(
                id="b",
                executor="slow",
                config=GenerateConfig(model="t", prompt_template="b"),
            ),
        ],
        arcs=[
            Arc(source="in", target="a"),
            Arc(source="a", target="out_a"),
            Arc(source="in", target="b"),
            Arc(source="b", target="out_b"),
        ],
        initial_marking=Marking(tokens={"in": [Token(), Token()]}),
    )

    [result] = await execute(net, max_concurrency=2)

    assert result.status == "completed"
    assert len(result.trace) == 2
    # Behavioral check: second task started before first ended (true concurrency)
    starts = [e for e in logger.log if e[0] == "start"]
    ends = [e for e in logger.log if e[0] == "end"]
    assert starts[1][2] < ends[0][2], "b didn't start before a ended"


@pytest.mark.asyncio
async def test_semaphore_serializes():
    """max_concurrency=1 forces sequential. Two 0.1s tasks take >= 0.18s."""
    logger = SlowExecutor(delay=0.1)
    register("slow", logger)

    net = Net(
        places=[Place(id="in"), Place(id="out_a"), Place(id="out_b")],
        transitions=[
            Transition(
                id="a",
                executor="slow",
                config=GenerateConfig(model="t", prompt_template="a"),
            ),
            Transition(
                id="b",
                executor="slow",
                config=GenerateConfig(model="t", prompt_template="b"),
            ),
        ],
        arcs=[
            Arc(source="in", target="a"),
            Arc(source="a", target="out_a"),
            Arc(source="in", target="b"),
            Arc(source="b", target="out_b"),
        ],
        initial_marking=Marking(tokens={"in": [Token(), Token()]}),
    )

    [result] = await execute(net, max_concurrency=1)

    assert len(result.trace) == 2

    # Behavioral check: first task ended before second started (serialized)
    starts = sorted([e for e in logger.log if e[0] == "start"], key=lambda e: e[2])
    ends = sorted([e for e in logger.log if e[0] == "end"], key=lambda e: e[2])
    assert ends[0][2] <= starts[1][2], "Tasks overlapped despite max_concurrency=1"


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    """4 tasks, max_concurrency=2. Never more than 2 active at once."""
    active = []
    max_active = [0]

    class TrackingExecutor:
        async def execute(self, inputs, config):
            active.append(1)
            if len(active) > max_active[0]:
                max_active[0] = len(active)
            await asyncio.sleep(0.05)
            active.pop()
            return GenerateOutput(text="ok")

    register("tracking", TrackingExecutor())

    net = Net(
        places=[Place(id="in")] + [Place(id=f"out_{i}") for i in range(4)],
        transitions=[
            Transition(
                id=f"t{i}",
                executor="tracking",
                config=GenerateConfig(model="t", prompt_template=f"t{i}"),
            )
            for i in range(4)
        ],
        arcs=(
            [Arc(source="in", target=f"t{i}") for i in range(4)]
            + [Arc(source=f"t{i}", target=f"out_{i}") for i in range(4)]
        ),
        initial_marking=Marking(tokens={"in": [Token() for _ in range(4)]}),
    )

    [result] = await execute(net, max_concurrency=2)

    assert len(result.trace) == 4
    assert max_active[0] <= 2, f"Max concurrent was {max_active[0]}"


# ==============================================================================
# COMPLEX TOPOLOGIES
# ==============================================================================


@pytest.mark.asyncio
async def test_diamond_fork_join():
    """Diamond: start → left + right → join → end."""
    register("pass", PassExecutor())

    net = Net(
        places=[
            Place(id="start"),
            Place(id="left"),
            Place(id="right"),
            Place(id="end"),
        ],
        transitions=[
            Transition(
                id="go_left",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="l"),
            ),
            Transition(
                id="go_right",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="r"),
            ),
            Transition(
                id="join",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="j"),
            ),
        ],
        arcs=[
            Arc(source="start", target="go_left"),
            Arc(source="go_left", target="left"),
            Arc(source="start", target="go_right"),
            Arc(source="go_right", target="right"),
            Arc(source="left", target="join"),
            Arc(source="right", target="join"),
            Arc(source="join", target="end"),
        ],
        initial_marking=Marking(tokens={"start": [Token(), Token()]}),
    )

    [result] = await execute(net, max_concurrency=4)

    assert len(result.trace) == 3
    tids = [r.transition_id for r in result.trace]
    assert tids.index("join") == 2  # join fires last


@pytest.mark.asyncio
async def test_multi_stage_pipeline():
    """Four stages in sequence. Execution order must be preserved."""
    order = []

    class OrderExecutor:
        async def execute(self, inputs, config):
            order.append(config.prompt_template)
            return GenerateOutput(text=config.prompt_template)

    register("order", OrderExecutor())

    places = [Place(id=f"p{i}") for i in range(5)]
    transitions = [
        Transition(
            id=f"t{i}",
            executor="order",
            config=GenerateConfig(model="t", prompt_template=f"stage{i}"),
        )
        for i in range(4)
    ]
    arcs = []
    for i in range(4):
        arcs.append(Arc(source=f"p{i}", target=f"t{i}"))
        arcs.append(Arc(source=f"t{i}", target=f"p{i + 1}"))

    net = Net(
        places=places,
        transitions=transitions,
        arcs=arcs,
        initial_marking=Marking(tokens={"p0": [Token()]}),
    )

    [result] = await execute(net, max_concurrency=10)

    assert len(result.trace) == 4
    assert order == ["stage0", "stage1", "stage2", "stage3"]


@pytest.mark.asyncio
async def test_partial_join_waits():
    """Fork→join with colored tokens. Join waits until BOTH branches complete per run_id.

    Left branch is fast (0.01s), right branch is slow (0.1s).
    Join must not fire until both arrive for the same run_id.
    """
    order = []

    class TimedExecutor:
        async def execute(self, inputs, config):
            name = config.prompt_template
            delay = 0.01 if "left" in name else 0.1 if "right" in name else 0.0
            await asyncio.sleep(delay)
            order.append(name)
            return GenerateOutput(text=name)

    register("timed", TimedExecutor())

    net = Net(
        places=[
            Place(id="start"),
            Place(id="left"),
            Place(id="right"),
            Place(id="end"),
        ],
        transitions=[
            Transition(
                id="go_left",
                executor="timed",
                config=GenerateConfig(model="t", prompt_template="left"),
            ),
            Transition(
                id="go_right",
                executor="timed",
                config=GenerateConfig(model="t", prompt_template="right"),
            ),
            Transition(
                id="join",
                executor="timed",
                config=GenerateConfig(model="t", prompt_template="join"),
            ),
        ],
        arcs=[
            Arc(source="start", target="go_left"),
            Arc(source="go_left", target="left"),
            Arc(source="start", target="go_right"),
            Arc(source="go_right", target="right"),
            Arc(source="left", target="join"),
            Arc(source="right", target="join"),
            Arc(source="join", target="end"),
        ],
        initial_marking=Marking(tokens={"start": [Token(run_id="r1"), Token(run_id="r1")]}),
    )

    results = await execute(net, max_concurrency=4)
    by_rid = {r.run_id: r for r in results}

    assert by_rid["r1"].status == "completed"
    # Join must fire after both branches
    assert order.index("join") > order.index("left")
    assert order.index("join") > order.index("right")


@pytest.mark.asyncio
async def test_concurrent_deposits_to_same_place():
    """Two transitions both output to the same place. Both tokens land."""
    register("pass", PassExecutor("deposited"))

    net = Net(
        places=[Place(id="in"), Place(id="shared_out")],
        transitions=[
            Transition(
                id="a",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="a"),
            ),
            Transition(
                id="b",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="b"),
            ),
        ],
        arcs=[
            Arc(source="in", target="a"),
            Arc(source="a", target="shared_out"),
            Arc(source="in", target="b"),
            Arc(source="b", target="shared_out"),
        ],
        initial_marking=Marking(tokens={"in": [Token(), Token()]}),
    )

    [result] = await execute(net, max_concurrency=2)

    assert len(result.trace) == 2
    assert all(r.status == "completed" for r in result.trace)


@pytest.mark.asyncio
async def test_arc_weight_requires_multiple_tokens():
    """Transition with arc weight=2 only fires when 2 tokens are present."""
    register("pass", PassExecutor())

    # weight=2 means we need 2 tokens in "in" for the transition to fire
    net = Net(
        places=[Place(id="in"), Place(id="out")],
        transitions=[
            Transition(
                id="t",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="t"),
            ),
        ],
        arcs=[
            Arc(source="in", target="t", weight=2),
            Arc(source="t", target="out"),
        ],
        initial_marking=Marking(tokens={"in": [Token()]}),  # only 1 token — not enough
    )

    results = await execute(net)
    assert results == []  # never fires — not enough tokens

    # Now with 2 tokens — should fire
    net.initial_marking = Marking(tokens={"in": [Token(), Token()]})
    results = await execute(net)
    assert len(results) == 1
    assert len(results[0].trace) == 1


@pytest.mark.asyncio
async def test_fan_out_weight():
    """Output arc weight=2 deposits 2 copies of the token."""
    register("pass", PassExecutor("doubled"))

    net = Net(
        places=[Place(id="in"), Place(id="out"), Place(id="final")],
        transitions=[
            Transition(
                id="t",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="t"),
            ),
            # This transition needs 2 tokens (weight=2 input) to prove fan-out worked
            Transition(
                id="consume_both",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="c"),
            ),
        ],
        arcs=[
            Arc(source="in", target="t"),
            Arc(source="t", target="out", weight=2),  # deposits 2 tokens
            Arc(source="out", target="consume_both", weight=2),  # needs both
            Arc(source="consume_both", target="final"),
        ],
        initial_marking=Marking(tokens={"in": [Token()]}),
    )

    [result] = await execute(net)

    assert len(result.trace) == 2  # t fired, then consume_both fired
    assert result.trace[1].transition_id == "consume_both"


@pytest.mark.asyncio
async def test_token_data_preserved_through_pipeline():
    """Verify .text and .score values flow correctly through stages."""
    collected = {}

    class InspectAgent:
        async def execute(self, inputs, config):
            text = ""
            for tok in inputs:
                if hasattr(tok, "text"):
                    text = tok.text
            collected["agent_saw"] = text
            return GenerateOutput(text=f"processed:{text}")

    class InspectJudge:
        async def execute(self, inputs, config):
            text = ""
            for tok in inputs:
                if hasattr(tok, "text"):
                    text = tok.text
            collected["judge_saw"] = text
            return JudgeOutput(score=0.42)

    register("inspect_agent", InspectAgent())
    register("inspect_judge", InspectJudge())

    net = Net(
        places=[Place(id="in"), Place(id="mid"), Place(id="out")],
        transitions=[
            Transition(
                id="gen",
                executor="inspect_agent",
                config=GenerateConfig(model="t", prompt_template="g"),
            ),
            Transition(
                id="jdg",
                executor="inspect_judge",
                config=JudgeConfig(model="test", rubric=[{"weight": 1.0, "requirement": "ok"}]),
            ),
        ],
        arcs=[
            Arc(source="in", target="gen"),
            Arc(source="gen", target="mid"),
            Arc(source="mid", target="jdg"),
            Arc(source="jdg", target="out"),
        ],
        initial_marking=Marking(tokens={"in": [GenerateOutput(text="hello world")]}),
    )

    [result] = await execute(net)

    assert collected["agent_saw"] == "hello world"
    assert collected["judge_saw"] == "processed:hello world"
    assert result.score == 0.42


@pytest.mark.asyncio
async def test_multiple_judges_last_score_wins():
    """Two judges in sequence. Score is from the last one."""
    register("judge_high", ScoreExecutor(0.95))
    register("judge_low", ScoreExecutor(0.3))

    net = Net(
        places=[Place(id="p0"), Place(id="p1"), Place(id="p2"), Place(id="p3")],
        transitions=[
            Transition(
                id="j1",
                executor="judge_high",
                config=JudgeConfig(model="test", rubric=[{"weight": 1.0, "requirement": "ok"}]),
            ),
            Transition(
                id="j2",
                executor="judge_low",
                config=JudgeConfig(model="test", rubric=[{"weight": 1.0, "requirement": "ok"}]),
            ),
        ],
        arcs=[
            Arc(source="p0", target="j1"),
            Arc(source="j1", target="p1"),
            Arc(source="p1", target="j2"),
            Arc(source="j2", target="p2"),
        ],
        initial_marking=Marking(tokens={"p0": [GenerateOutput(text="candidate")]}),
    )

    [result] = await execute(net)

    assert result.score == 0.3  # last judge wins


@pytest.mark.asyncio
async def test_competing_transitions_no_guards():
    """Two transitions from same place, no guards, one token.

    Both get spawned from the initial ready() check (stale list),
    but only the first one gets the actual token. The second runs
    with empty inputs. Both fire — this is expected v0.1 behavior.
    """
    received_inputs = {}

    class TrackInputsExecutor:
        async def execute(self, inputs, config):
            received_inputs[config.prompt_template] = len(inputs)
            return GenerateOutput(text="ok")

    register("track", TrackInputsExecutor())

    net = Net(
        places=[Place(id="in"), Place(id="out_a"), Place(id="out_b")],
        transitions=[
            Transition(
                id="a",
                executor="track",
                config=GenerateConfig(model="t", prompt_template="a"),
            ),
            Transition(
                id="b",
                executor="track",
                config=GenerateConfig(model="t", prompt_template="b"),
            ),
        ],
        arcs=[
            Arc(source="in", target="a"),
            Arc(source="a", target="out_a"),
            Arc(source="in", target="b"),
            Arc(source="b", target="out_b"),
        ],
        initial_marking=Marking(tokens={"in": [Token()]}),  # only ONE token
    )

    [result] = await execute(net)

    # Both fire, but only one got the token
    assert len(result.trace) == 2
    input_counts = sorted(received_inputs.values())
    assert input_counts == [0, 1]  # one got the token, one got nothing


@pytest.mark.asyncio
async def test_same_transition_refires_in_cycle():
    """Cycle: same transition fires 3 times for same run_id before exiting."""
    register("pass", PassExecutor())
    register("seq_score", SeqScoreExecutor([0.3, 0.5, 0.95]))

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
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="g"),
            ),
            Transition(
                id="judge",
                executor="seq_score",
                config=JudgeConfig(model="test", rubric=[{"weight": 1.0, "requirement": "ok"}]),
            ),
            Transition(
                id="loop",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="l"),
                when=lambda tokens: tokens[0].score < 0.9,
            ),
            Transition(
                id="done",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="d"),
                when=lambda tokens: tokens[0].score >= 0.9,
            ),
        ],
        arcs=[
            Arc(source="prompt", target="gen"),
            Arc(source="gen", target="response"),
            Arc(source="response", target="judge"),
            Arc(source="judge", target="scored"),
            Arc(source="scored", target="loop"),
            Arc(source="loop", target="prompt"),
            Arc(source="scored", target="done"),
            Arc(source="done", target="final"),
        ],
        initial_marking=Marking(tokens={"prompt": [Token()]}),
    )

    [result] = await execute(net, fuse=50)

    gen_count = sum(1 for r in result.trace if r.transition_id == "gen")
    judge_count = sum(1 for r in result.trace if r.transition_id == "judge")
    loop_count = sum(1 for r in result.trace if r.transition_id == "loop")
    done_count = sum(1 for r in result.trace if r.transition_id == "done")

    assert gen_count == 3  # initial + 2 loops
    assert judge_count == 3
    assert loop_count == 2  # looped twice
    assert done_count == 1  # exited once


# ==============================================================================
# BATCH WITH FIXED RUN_IDS
# ==============================================================================


@pytest.mark.asyncio
async def test_batch_choice_routing_fixed_rids():
    """3 rows with fixed run_ids hit a choice. Each routes based on score.

    r1: score=0.9 → accept
    r2: score=0.4 → reject
    r3: score=0.8 → accept
    """
    scores = {"r1": 0.9, "r2": 0.4, "r3": 0.8}

    class RidJudge:
        async def execute(self, inputs, config):
            rid = inputs[0].run_id
            return JudgeOutput(score=scores.get(rid, 0.5))

    register("rid_judge", RidJudge())
    register("pass", PassExecutor())

    net = Net(
        places=[
            Place(id="in"),
            Place(id="scored"),
            Place(id="accepted"),
            Place(id="rejected"),
        ],
        transitions=[
            Transition(
                id="judge",
                executor="rid_judge",
                config=JudgeConfig(model="test", rubric=[{"weight": 1.0, "requirement": "ok"}]),
            ),
            Transition(
                id="accept",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="a"),
                when=lambda tokens: tokens[0].score >= 0.7,
            ),
            Transition(
                id="reject",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="r"),
                when=lambda tokens: tokens[0].score < 0.7,
            ),
        ],
        arcs=[
            Arc(source="in", target="judge"),
            Arc(source="judge", target="scored"),
            Arc(source="scored", target="accept"),
            Arc(source="accept", target="accepted"),
            Arc(source="scored", target="reject"),
            Arc(source="reject", target="rejected"),
        ],
        initial_marking=Marking(
            tokens={
                "in": [
                    GenerateOutput(text="case1", run_id="r1"),
                    GenerateOutput(text="case2", run_id="r2"),
                    GenerateOutput(text="case3", run_id="r3"),
                ]
            }
        ),
    )

    results = await execute(net, max_concurrency=3)
    by_rid = {r.run_id: r for r in results}

    # r1 and r3 accepted, r2 rejected
    r1_tids = {t.transition_id for t in by_rid["r1"].trace}
    r2_tids = {t.transition_id for t in by_rid["r2"].trace}
    r3_tids = {t.transition_id for t in by_rid["r3"].trace}

    assert "accept" in r1_tids and "reject" not in r1_tids
    assert "reject" in r2_tids and "accept" not in r2_tids
    assert "accept" in r3_tids and "reject" not in r3_tids


@pytest.mark.asyncio
async def test_batch_10_rows_stress():
    """10 rows through gen→judge, concurrency=3. Timing proves parallelism."""

    class FastAgent:
        async def execute(self, inputs, config):
            await asyncio.sleep(0.01)
            return GenerateOutput(text="fast")

    register("fast_agent", FastAgent())
    register("fast_judge", ScoreExecutor(0.77))

    net = Net(
        places=[Place(id="in"), Place(id="mid"), Place(id="out")],
        transitions=[
            Transition(
                id="gen",
                executor="fast_agent",
                config=GenerateConfig(model="t", prompt_template="{text}"),
            ),
            Transition(
                id="jdg",
                executor="fast_judge",
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

    rows = [GenerateOutput(text=f"row_{i}") for i in range(10)]
    results = await execute(net, rows=rows, place="in", max_concurrency=3, fuse=100)

    assert len(results) == 10
    for r in results:
        assert r.status == "completed"
        assert r.score == 0.77


@pytest.mark.asyncio
async def test_batch_with_cycles():
    """Batch of 3 rows cycling independently until judge score >= 0.9."""
    register("pass", PassExecutor())
    register("seq_score", SeqScoreExecutor([0.95, 0.5, 0.3, 0.92, 0.4, 0.91]))

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
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="g"),
            ),
            Transition(
                id="judge",
                executor="seq_score",
                config=JudgeConfig(model="test", rubric=[{"weight": 1.0, "requirement": "ok"}]),
            ),
            Transition(
                id="loop",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="l"),
                when=lambda tokens: tokens[0].score < 0.9,
            ),
            Transition(
                id="done",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="d"),
                when=lambda tokens: tokens[0].score >= 0.9,
            ),
        ],
        arcs=[
            Arc(source="prompt", target="gen"),
            Arc(source="gen", target="response"),
            Arc(source="response", target="judge"),
            Arc(source="judge", target="scored"),
            Arc(source="scored", target="loop"),
            Arc(source="loop", target="prompt"),
            Arc(source="scored", target="done"),
            Arc(source="done", target="final"),
        ],
        initial_marking=Marking(tokens={}),
    )

    rows = [GenerateOutput(text=f"row{i}") for i in range(3)]
    results = await execute(net, rows=rows, place="prompt", max_concurrency=3, fuse=50)

    assert len(results) == 3
    for r in results:
        assert r.status == "completed"
        assert any(t.transition_id == "done" for t in r.trace)


# ==============================================================================
# ERROR / FAILURE SCENARIOS
# ==============================================================================


@pytest.mark.asyncio
async def test_error_mid_pipeline():
    """Error in stage 2 of 3. Stage 1 in trace, stage 3 never runs."""

    class StageExecutor:
        async def execute(self, inputs, config):
            if config.prompt_template == "stage2":
                raise RuntimeError("stage2 exploded")
            return GenerateOutput(text=config.prompt_template)

    register("staged", StageExecutor())

    net = Net(
        places=[Place(id="p0"), Place(id="p1"), Place(id="p2"), Place(id="p3")],
        transitions=[
            Transition(
                id="t0",
                executor="staged",
                config=GenerateConfig(model="t", prompt_template="stage1"),
            ),
            Transition(
                id="t1",
                executor="staged",
                config=GenerateConfig(model="t", prompt_template="stage2"),
            ),
            Transition(
                id="t2",
                executor="staged",
                config=GenerateConfig(model="t", prompt_template="stage3"),
            ),
        ],
        arcs=[
            Arc(source="p0", target="t0"),
            Arc(source="t0", target="p1"),
            Arc(source="p1", target="t1"),
            Arc(source="t1", target="p2"),
            Arc(source="p2", target="t2"),
            Arc(source="t2", target="p3"),
        ],
        initial_marking=Marking(tokens={"p0": [Token()]}),
    )

    [result] = await execute(net)

    assert result.status == "failed"
    assert result.error == "stage2 exploded"
    assert len(result.trace) == 2
    assert result.trace[0].status == "completed"
    assert result.trace[1].status == "failed"


@pytest.mark.asyncio
async def test_cascading_failure_skips_all_downstream():
    """Failure in stage 1 prevents stages 2, 3, 4 from running."""

    class FailFirst:
        async def execute(self, inputs, config):
            if config.prompt_template == "s1":
                raise RuntimeError("s1 died")
            return GenerateOutput(text="ok")

    register("fail_first", FailFirst())

    places = [Place(id=f"p{i}") for i in range(5)]
    transitions = [
        Transition(
            id=f"t{i}",
            executor="fail_first",
            config=GenerateConfig(model="t", prompt_template=f"s{i}"),
        )
        for i in range(4)
    ]
    arcs = []
    for i in range(4):
        arcs.append(Arc(source=f"p{i}", target=f"t{i}"))
        arcs.append(Arc(source=f"t{i}", target=f"p{i + 1}"))

    net = Net(
        places=places,
        transitions=transitions,
        arcs=arcs,
        initial_marking=Marking(tokens={"p0": [Token(run_id="x")]}),
    )

    results = await execute(net)
    by_rid = {r.run_id: r for r in results}

    assert by_rid["x"].status == "failed"
    # Only t0 (s0) completed, t1 (s1) failed, t2 and t3 never ran
    assert len(by_rid["x"].trace) == 2


@pytest.mark.asyncio
async def test_parallel_error_isolation_fixed_rids():
    """3 fixed run_ids, r2 fails. r1 and r3 complete independently."""
    register("pass", PassExecutor())

    class FlakyByRid:
        async def execute(self, inputs, config):
            rid = inputs[0].run_id if inputs else None
            await asyncio.sleep(0.02)
            if rid == "r2":
                raise RuntimeError("r2 boom")
            return GenerateOutput(text="ok")

    register("flaky_rid", FlakyByRid())
    register("score", ScoreExecutor(0.88))

    net = Net(
        places=[Place(id="in"), Place(id="mid"), Place(id="out")],
        transitions=[
            Transition(
                id="gen",
                executor="flaky_rid",
                config=GenerateConfig(model="t", prompt_template="g"),
            ),
            Transition(
                id="jdg",
                executor="score",
                config=JudgeConfig(model="test", rubric=[{"weight": 1.0, "requirement": "ok"}]),
            ),
        ],
        arcs=[
            Arc(source="in", target="gen"),
            Arc(source="gen", target="mid"),
            Arc(source="mid", target="jdg"),
            Arc(source="jdg", target="out"),
        ],
        initial_marking=Marking(
            tokens={
                "in": [
                    Token(run_id="r1"),
                    Token(run_id="r2"),
                    Token(run_id="r3"),
                ]
            }
        ),
    )

    results = await execute(net, max_concurrency=3)
    by_rid = {r.run_id: r for r in results}

    assert by_rid["r1"].status == "completed"
    assert by_rid["r1"].score == 0.88
    assert by_rid["r2"].status == "failed"
    assert by_rid["r2"].error == "r2 boom"
    assert by_rid["r3"].status == "completed"
    assert by_rid["r3"].score == 0.88


@pytest.mark.asyncio
async def test_guard_throws_exception():
    """when() callable raises — transition is skipped, engine doesn't crash."""
    register("pass", PassExecutor())

    def bad_guard(tokens):
        raise ValueError("guard broke")

    net = Net(
        places=[Place(id="in"), Place(id="out_a"), Place(id="out_b")],
        transitions=[
            Transition(
                id="guarded",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="g"),
                when=bad_guard,
            ),
            Transition(
                id="unguarded",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="u"),
            ),
        ],
        arcs=[
            Arc(source="in", target="guarded"),
            Arc(source="guarded", target="out_a"),
            Arc(source="in", target="unguarded"),
            Arc(source="unguarded", target="out_b"),
        ],
        initial_marking=Marking(tokens={"in": [Token(), Token()]}),
    )

    # Guard exception should not crash the engine — guarded transition skipped
    [result] = await execute(net)

    # At minimum, the unguarded transition should fire
    assert any(r.transition_id == "unguarded" for r in result.trace)


@pytest.mark.asyncio
async def test_guard_truthy_nonbool():
    """when() returns truthy non-bool (int 1). Should still work."""
    register("pass", PassExecutor())

    net = Net(
        places=[Place(id="in"), Place(id="out")],
        transitions=[
            Transition(
                id="t",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="t"),
                when=lambda tokens: 1,
            ),  # truthy int, not bool
        ],
        arcs=[Arc(source="in", target="t"), Arc(source="t", target="out")],
        initial_marking=Marking(tokens={"in": [Token()]}),
    )

    [result] = await execute(net)
    assert len(result.trace) == 1


@pytest.mark.asyncio
async def test_executor_returns_wrong_type():
    """Executor returns a string instead of Token. Should fail gracefully."""

    class BadExecutor:
        async def execute(self, inputs, config):
            return "not a token"  # type: ignore

    register("bad", BadExecutor())

    net = Net(
        places=[Place(id="in"), Place(id="out")],
        transitions=[
            Transition(
                id="t",
                executor="bad",
                config=GenerateConfig(model="t", prompt_template="t"),
            ),
        ],
        arcs=[Arc(source="in", target="t"), Arc(source="t", target="out")],
        initial_marking=Marking(tokens={"in": [Token()]}),
    )

    [result] = await execute(net)
    assert result.status == "failed"
    assert result.error is not None


@pytest.mark.asyncio
async def test_capacity_exceeded_under_concurrency():
    """Place with capacity=1. Two concurrent transitions try to deposit. One fails."""
    register("pass", PassExecutor())

    net = Net(
        places=[Place(id="in"), Place(id="out", capacity=1)],
        transitions=[
            Transition(
                id="a",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="a"),
            ),
            Transition(
                id="b",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="b"),
            ),
        ],
        arcs=[
            Arc(source="in", target="a"),
            Arc(source="a", target="out"),
            Arc(source="in", target="b"),
            Arc(source="b", target="out"),
        ],
        initial_marking=Marking(tokens={"in": [Token(), Token()]}),
    )

    [result] = await execute(net, max_concurrency=2)

    # One succeeds, one fails due to capacity
    completed = [r for r in result.trace if r.status == "completed"]
    failed = [r for r in result.trace if r.status == "failed"]
    assert len(completed) == 1
    assert len(failed) == 1
    assert "capacity" in failed[0].error.lower()


@pytest.mark.asyncio
async def test_invalid_batch_place():
    """run() with rows + place that doesn't exist. Validation should catch it."""
    net = Net(
        places=[Place(id="in"), Place(id="out")],
        transitions=[
            Transition(
                id="t",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="t"),
            ),
        ],
        arcs=[Arc(source="in", target="t"), Arc(source="t", target="out")],
        initial_marking=Marking(tokens={}),
    )

    with pytest.raises(ValidationError):
        await execute(net, rows=[Token()], place="nonexistent")


@pytest.mark.asyncio
async def test_empty_net_no_transitions():
    """Net with no transitions returns immediately."""
    net = Net(
        places=[Place(id="p")],
        transitions=[],
        arcs=[],
        initial_marking=Marking(tokens={"p": [Token()]}),
    )

    results = await execute(net)
    assert results == []


@pytest.mark.asyncio
async def test_empty_batch():
    """run() with rows=[] returns []."""
    register("pass", PassExecutor())

    net = Net(
        places=[Place(id="in"), Place(id="out")],
        transitions=[
            Transition(
                id="t",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="t"),
            ),
        ],
        arcs=[Arc(source="in", target="t"), Arc(source="t", target="out")],
        initial_marking=Marking(tokens={}),
    )

    results = await execute(net, rows=[], place="in")
    assert results == []


@pytest.mark.asyncio
async def test_guard_deadlock_terminates():
    """Token in place but no guard passes. Engine terminates cleanly (no hang)."""
    register("pass", PassExecutor())

    net = Net(
        places=[Place(id="in"), Place(id="out_a"), Place(id="out_b")],
        transitions=[
            Transition(
                id="a",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="a"),
                when=lambda tokens: False,
            ),
            Transition(
                id="b",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="b"),
                when=lambda tokens: False,
            ),
        ],
        arcs=[
            Arc(source="in", target="a"),
            Arc(source="a", target="out_a"),
            Arc(source="in", target="b"),
            Arc(source="b", target="out_b"),
        ],
        initial_marking=Marking(tokens={"in": [Token()]}),
    )

    results = await execute(net)
    assert results == []  # nothing fires, clean exit


@pytest.mark.asyncio
async def test_fuse_stops_infinite_cycle():
    """Guard never passes — cycle would run forever. Fuse stops it."""
    register("pass", PassExecutor())
    register("always_low", ScoreExecutor(0.1))

    net = Net(
        places=[Place(id="prompt"), Place(id="response"), Place(id="scored")],
        transitions=[
            Transition(
                id="gen",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="g"),
            ),
            Transition(
                id="judge",
                executor="always_low",
                config=JudgeConfig(model="test", rubric=[{"weight": 1.0, "requirement": "ok"}]),
            ),
            # Loop always fires (score always < 0.9), no exit transition
            Transition(
                id="loop",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="l"),
                when=lambda tokens: tokens[0].score < 0.9,
            ),
        ],
        arcs=[
            Arc(source="prompt", target="gen"),
            Arc(source="gen", target="response"),
            Arc(source="response", target="judge"),
            Arc(source="judge", target="scored"),
            Arc(source="scored", target="loop"),
            Arc(source="loop", target="prompt"),
        ],
        initial_marking=Marking(tokens={"prompt": [Token()]}),
    )

    [result] = await execute(net, fuse=10)

    # Fuse=10 means at most 10 firings. Cycle is gen→judge→loop = 3 per iteration.
    # So ~3 iterations = 9 firings, then fuse stops at 10.
    assert len(result.trace) == 10
    assert result.status == "completed"  # fuse is not a failure, just a stop


@pytest.mark.asyncio
async def test_async_guard_gates_transition():
    """Async guard (e.g. LLM judge gate) controls whether transition fires."""
    register("pass", PassExecutor())

    async def async_judge_gate(tokens):
        """Simulate an async LLM call that decides if we should proceed."""
        await asyncio.sleep(0.01)
        # Check if input has sufficient quality
        for t in tokens:
            if hasattr(t, "text") and "good" in t.text:
                return True
        return False

    net = Net(
        places=[Place(id="in"), Place(id="out")],
        transitions=[
            Transition(
                id="gated",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="t"),
                when=async_judge_gate,
            ),
        ],
        arcs=[Arc(source="in", target="gated"), Arc(source="gated", target="out")],
        initial_marking=Marking(tokens={"in": [GenerateOutput(text="this is good stuff")]}),
    )

    [result] = await execute(net)
    assert len(result.trace) == 1  # guard passed, transition fired


@pytest.mark.asyncio
async def test_async_guard_blocks_transition():
    """Async guard returns False — transition doesn't fire."""
    register("pass", PassExecutor())

    async def async_reject(tokens):
        await asyncio.sleep(0.01)
        return False

    net = Net(
        places=[Place(id="in"), Place(id="out")],
        transitions=[
            Transition(
                id="gated",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="t"),
                when=async_reject,
            ),
        ],
        arcs=[Arc(source="in", target="gated"), Arc(source="gated", target="out")],
        initial_marking=Marking(tokens={"in": [Token()]}),
    )

    results = await execute(net)
    assert results == []  # guard blocked, nothing fired


@pytest.mark.asyncio
async def test_async_guard_exception_handled():
    """Async guard that raises — transition skipped, engine continues."""
    register("pass", PassExecutor())

    async def async_crash(tokens):
        raise RuntimeError("async guard exploded")

    net = Net(
        places=[Place(id="in"), Place(id="out_a"), Place(id="out_b")],
        transitions=[
            Transition(
                id="guarded",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="g"),
                when=async_crash,
            ),
            Transition(
                id="unguarded",
                executor="pass",
                config=GenerateConfig(model="t", prompt_template="u"),
            ),
        ],
        arcs=[
            Arc(source="in", target="guarded"),
            Arc(source="guarded", target="out_a"),
            Arc(source="in", target="unguarded"),
            Arc(source="unguarded", target="out_b"),
        ],
        initial_marking=Marking(tokens={"in": [Token(), Token()]}),
    )

    [result] = await execute(net)
    assert any(r.transition_id == "unguarded" for r in result.trace)
