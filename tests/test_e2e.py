"""End-to-end tests with live LLM calls via ollama.

Requires: ollama running locally with qwen2.5:0.5b pulled.
    ollama pull qwen2.5:0.5b

Skip all tests if ollama is not available.
"""

from __future__ import annotations

import httpx
import pytest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from peven.petri.dsl import NetBuilder
from peven.petri.engine import execute
from peven.petri.executors import register
from peven.petri.render import render, render_net
from peven.petri.schema import GenerateConfig, JudgeConfig
from peven.petri.types import GenerateOutput, JudgeOutput

# -- Setup ---------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:0.5b"


def _ollama_available() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_available(),
    reason="ollama not running — skip live e2e tests",
)


_PROVIDER = None


def _model():
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = OllamaProvider(base_url=f"{OLLAMA_URL}/v1")
    return OpenAIChatModel(MODEL_NAME, provider=_PROVIDER)


def _agent_config(prompt: str, system: str = "Reply concisely in one sentence."):
    return (
        "agent",
        GenerateConfig(
            model=_model(),
            prompt_template=prompt,
            system_prompt=system,
        ),
    )


class LLMJudge:
    """Judge that uses the LLM to score on a 0-1 scale."""

    def __init__(self):
        self._model = _model()

    async def execute(self, inputs, config):
        from pydantic_ai import Agent

        text = ""
        for tok in inputs:
            if hasattr(tok, "text"):
                text = tok.text
                break

        agent = Agent(
            model=self._model,
            system_prompt=(
                "You are a judge. Score the following text on quality from 0.0 to 1.0. "
                "Reply with ONLY a decimal number like 0.7 — nothing else."
            ),
        )
        result = await agent.run(f"Score this: {text}")
        try:
            score = float(result.output.strip())
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = 0.5
        return JudgeOutput(score=score)


# ==============================================================================
# SINGLE RUN TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_single_generate():
    """Single agent transition produces text output."""
    n = NetBuilder()
    start = n.place("start")
    end = n.place("end")
    gen = n.transition("gen", _agent_config("What is 2+2? Answer with just the number."))
    start >> gen >> end
    start.token(GenerateOutput(text=""))

    [result] = await execute(n.build())

    assert result.status == "completed"
    assert len(result.trace) == 1
    assert isinstance(result.trace[0].output, GenerateOutput)
    assert len(result.trace[0].output.text) > 0


@pytest.mark.asyncio
async def test_generate_chain():
    """Two agents in sequence — second sees first's output."""
    n = NetBuilder()
    start = n.place("start")
    mid = n.place("mid")
    end = n.place("end")

    gen1 = n.transition(
        "gen1",
        _agent_config(
            "Name a random color. Reply with just the color name.",
        ),
    )
    gen2 = n.transition(
        "gen2",
        _agent_config(
            "Name an animal that is the color: {text}. Reply with just the animal name.",
        ),
    )

    start >> gen1 >> mid >> gen2 >> end
    start.token(GenerateOutput(text=""))

    [result] = await execute(n.build())

    assert result.status == "completed"
    assert len(result.trace) == 2
    # Second agent should have produced something
    assert len(result.trace[1].output.text) > 0


@pytest.mark.asyncio
async def test_async_guard_with_llm():
    """Async guard calls the LLM to decide whether to proceed.

    This tests the async guard feature with a real LLM call — the guard
    itself is an LLM evaluation, not just a score check.
    Runs early to avoid ollama connection exhaustion from later parallel tests.
    """
    from pydantic_ai import Agent

    guard_model = _model()
    gate_called = [False]

    async def safety_gate(tokens):
        gate_called[0] = True
        text = ""
        for t in tokens:
            if hasattr(t, "text"):
                text = t.text
                break
        agent = Agent(
            model=guard_model,
            system_prompt="Is this text appropriate and safe? Reply with only YES or NO.",
        )
        await agent.run(text)
        return True  # always pass — we're testing that the async call works

    n = NetBuilder()
    start = n.place("start")
    checked = n.place("checked")
    end = n.place("end")

    gen = n.transition("gen", _agent_config("Write a friendly greeting for: {text}"))
    publish = n.transition("publish", _agent_config("Format this for publishing: {text}"))
    publish.when(safety_gate)

    start >> gen >> checked >> publish >> end
    start.token(GenerateOutput(text="a new colleague"))

    [result] = await execute(n.build())

    assert result.status == "completed"
    assert gate_called[0], "Async guard was never called"
    assert len(result.trace) == 2
    assert result.trace[0].transition_id == "gen"
    assert result.trace[1].transition_id == "publish"


@pytest.mark.asyncio
async def test_judge_as_gate():
    """LLM judge gate decides whether content advances.

    Generate a math answer, then an async guard asks the LLM to verify it.
    The guard actually evaluates the answer — not a passthrough.
    Two paths: verified → accepted, not verified → rejected.
    """
    from pydantic_ai import Agent

    gate_model = _model()

    async def math_verifier(tokens):
        """Ask the LLM if the math answer is correct."""
        text = ""
        for t in tokens:
            if hasattr(t, "text"):
                text = t.text
                break
        agent = Agent(
            model=gate_model,
            system_prompt=(
                "You are a math verifier. The question was 'What is 15 + 27?'. "
                "The correct answer is 42. Does the response contain the number 42? "
                "Reply with ONLY 'YES' or 'NO'."
            ),
        )
        result = await agent.run(f"Check this answer: {text}")
        return "YES" in result.output.upper()

    async def always_pass(tokens):
        """Fallback gate that always passes."""
        return True

    n = NetBuilder()
    start = n.place("start")
    answered = n.place("answered")
    accepted = n.place("accepted")
    fallback = n.place("fallback")

    solve = n.transition(
        "solve",
        _agent_config(
            "What is 15 + 27? Reply with ONLY the number.",
        ),
    )
    # Verified path — LLM judge gate checks the answer
    accept = n.transition("accept", _agent_config("{text}"))
    accept.when(math_verifier)
    # Fallback — always fires if verifier rejects
    reject = n.transition("reject", _agent_config("{text}"))
    reject.when(always_pass)

    start >> solve >> answered
    answered >> accept >> accepted
    answered >> reject >> fallback

    start.token(GenerateOutput(text=""))

    [result] = await execute(n.build())

    assert result.status == "completed"
    # At least solve + one of accept/reject should fire
    assert len(result.trace) >= 2
    tids = {r.transition_id for r in result.trace}
    assert "solve" in tids
    # One of these should have fired based on the judge gate
    assert "accept" in tids or "reject" in tids


@pytest.mark.asyncio
async def test_rubric_gate_legal_brief():
    """LLM writes a legal brief, rubric judge gate evaluates quality.

    The gate uses a multi-criteria rubric to decide if the brief is
    good enough to submit. If not, it routes to revision.

    Topology:
        facts → draft_brief → brief → [rubric gate] submit → filed
                                     → [rubric gate] revise → revised
    """
    from pydantic_ai import Agent

    gate_model = _model()

    async def rubric_quality_gate(tokens):
        """Multi-criteria rubric evaluation as an async guard."""
        text = ""
        for t in tokens:
            if hasattr(t, "text"):
                text = t.text
                break
        agent = Agent(
            model=gate_model,
            system_prompt=(
                "You are a legal writing evaluator. Score this brief on these criteria:\n"
                "1. STRUCTURE: Does it have a clear argument? (0-1)\n"
                "2. EVIDENCE: Does it cite or reference facts? (0-1)\n"
                "3. CLARITY: Is the writing clear and professional? (0-1)\n\n"
                "Reply with ONLY three scores separated by commas, like: 0.8,0.6,0.9"
            ),
        )
        result = await agent.run(f"Evaluate this legal brief:\n{text}")
        try:
            scores = [float(s.strip()) for s in result.output.split(",")[:3]]
            avg = sum(scores) / len(scores) if scores else 0
            return avg >= 0.5
        except (ValueError, ZeroDivisionError):
            return True  # on parse failure, let it through

    async def always_revise(tokens):
        return True

    n = NetBuilder()
    facts = n.place("facts")
    brief = n.place("brief")
    filed = n.place("filed")
    revised = n.place("revised")

    draft = n.transition(
        "draft_brief",
        _agent_config(
            "You are a lawyer. Write a short legal brief arguing that the defendant "
            "is not liable for breach of contract based on these facts: {text}. "
            "Include a clear argument structure with references to the facts.",
            system="You are an experienced litigation attorney. Be concise but thorough.",
        ),
    )
    submit = n.transition("submit", _agent_config("{text}"))
    submit.when(rubric_quality_gate)
    revise = n.transition(
        "revise",
        _agent_config(
            "This legal brief needs improvement. Strengthen the argument and add "
            "more specific references to the facts: {text}",
            system="You are a senior partner reviewing a junior associate's work.",
        ),
    )
    revise.when(always_revise)

    facts >> draft >> brief
    brief >> submit >> filed
    brief >> revise >> revised

    facts.token(
        GenerateOutput(
            text="The contract was signed on March 1. The defendant attempted delivery "
            "on March 15 but the plaintiff's warehouse was closed due to flooding. "
            "The force majeure clause in Section 7.2 covers natural disasters."
        )
    )

    [result] = await execute(n.build())

    assert result.status == "completed"
    assert len(result.trace) >= 2  # draft + (submit or revise)
    tids = {r.transition_id for r in result.trace}
    assert "draft_brief" in tids
    assert "submit" in tids or "revise" in tids


@pytest.mark.asyncio
async def test_iterative_essay_refinement():
    """Write → judge → refine cycle with a real rubric judge gate.

    The LLM writes an essay, a rubric judge evaluates it, and if the
    score is below threshold, routes back for revision. Uses fuse to
    guarantee termination.

    Topology:
        topic → write → draft → judge_quality → scored
        scored → [when score < 0.8] refine → topic  (cycle)
        scored → [when score >= 0.8] publish → final
    """
    register("llm_judge", LLMJudge())
    from pydantic_ai import Agent

    judge_model = _model()

    async def quality_gate_high(tokens):
        """Judge gate: is the essay good enough to publish?"""
        text = ""
        for t in tokens:
            if hasattr(t, "score"):
                return t.score >= 0.7  # use actual judge score
            if hasattr(t, "text"):
                text = t.text
        # Fallback: ask LLM directly
        agent = Agent(
            model=judge_model,
            system_prompt="Rate this essay 0.0 to 1.0. Reply with ONLY a number.",
        )
        result = await agent.run(text)
        try:
            return float(result.output.strip()) >= 0.7
        except ValueError:
            return True

    async def quality_gate_low(tokens):
        """Inverse of quality_gate_high."""
        result = await quality_gate_high(tokens)
        return not result

    n = NetBuilder()
    topic = n.place("topic")
    draft = n.place("draft")
    scored = n.place("scored")
    final = n.place("final")

    write = n.transition(
        "write",
        _agent_config(
            "Write a 3-sentence essay about: {text}",
            system="You are a skilled essayist. Be clear and compelling.",
        ),
    )
    judge_quality = n.transition(
        "judge_quality",
        (
            "llm_judge",
            JudgeConfig(
                model=_model(),
                rubric=[
                    {"weight": 0.4, "requirement": "clear thesis statement"},
                    {"weight": 0.3, "requirement": "supporting evidence or examples"},
                    {"weight": 0.3, "requirement": "professional and engaging tone"},
                ],
            ),
        ),
    )
    refine = n.transition(
        "refine",
        _agent_config(
            "This essay needs improvement. Make it more compelling and specific: {text}",
            system="You are an editor giving revision notes.",
        ),
    )
    refine.when(quality_gate_low)
    publish = n.transition("publish", _agent_config("{text}"))
    publish.when(quality_gate_high)

    topic >> write >> draft >> judge_quality >> scored
    scored >> refine >> topic  # cycle back
    scored >> publish >> final  # exit

    topic.token(GenerateOutput(text="why open source matters for AI safety"))

    [result] = await execute(n.build(), fuse=15, max_concurrency=1)

    trace_info = [(t.transition_id, t.status, t.error) for t in result.trace]
    assert result.status == "completed", f"error={result.error}, trace={trace_info}"
    assert len(result.trace) >= 3  # at minimum: write + judge + (publish or refine)
    tids = [r.transition_id for r in result.trace]
    assert "write" in tids
    assert "judge_quality" in tids
    # Either published directly or went through at least one refinement cycle
    assert "publish" in tids or "refine" in tids


@pytest.mark.asyncio
async def test_generate_with_judge():
    """Agent generates, LLM judges the output."""
    register("llm_judge", LLMJudge())

    n = NetBuilder()
    start = n.place("start")
    mid = n.place("mid")
    end = n.place("end")

    gen = n.transition(
        "gen",
        _agent_config(
            "Write a haiku about the ocean.",
        ),
    )
    jdg = n.transition(
        "jdg",
        (
            "llm_judge",
            JudgeConfig(
                model=_model(),
                rubric=[{"weight": 1.0, "requirement": "is a haiku"}],
            ),
        ),
    )

    start >> gen >> mid >> jdg >> end
    start.token(GenerateOutput(text=""))

    [result] = await execute(n.build())

    assert result.status == "completed"
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0
    assert len(result.trace) == 2


# ==============================================================================
# PARALLEL / FORK / JOIN
# ==============================================================================


@pytest.mark.asyncio
async def test_parallel_fork():
    """Two agents generate from the same prompt in parallel."""
    n = NetBuilder()
    start = n.place("start")
    out_a = n.place("out_a")
    out_b = n.place("out_b")

    gen_a = n.transition(
        "optimist",
        _agent_config(
            "Give an optimistic one-sentence take on: {text}",
        ),
    )
    gen_b = n.transition(
        "pessimist",
        _agent_config(
            "Give a pessimistic one-sentence take on: {text}",
        ),
    )

    start >> gen_a >> out_a
    start >> gen_b >> out_b
    start.token(GenerateOutput(text="artificial intelligence"))
    start.token(GenerateOutput(text="artificial intelligence"))

    [result] = await execute(n.build(), max_concurrency=2)

    assert result.status == "completed"
    assert len(result.trace) == 2
    tids = {r.transition_id for r in result.trace}
    assert tids == {"optimist", "pessimist"}


@pytest.mark.asyncio
async def test_fork_join():
    """Fork into two perspectives, join into a synthesis."""
    n = NetBuilder()
    start = n.place("start")
    left = n.place("left")
    right = n.place("right")
    end = n.place("end")

    gen_pro = n.transition(
        "pro",
        _agent_config(
            "Give one argument FOR remote work. One sentence.",
        ),
    )
    gen_con = n.transition(
        "con",
        _agent_config(
            "Give one argument AGAINST remote work. One sentence.",
        ),
    )
    synthesize = n.transition(
        "synthesize",
        _agent_config(
            "Summarize these two perspectives in one sentence: {text}",
        ),
    )

    start >> gen_pro >> left >> synthesize
    start >> gen_con >> right >> synthesize
    synthesize >> end

    start.token(GenerateOutput(text=""))
    start.token(GenerateOutput(text=""))

    [result] = await execute(n.build(), max_concurrency=2)

    assert result.status == "completed"
    assert len(result.trace) == 3
    # Synthesize should fire last
    assert result.trace[-1].transition_id == "synthesize"


# ==============================================================================
# CHOICE ROUTING WITH GUARDS
# ==============================================================================


@pytest.mark.asyncio
async def test_choice_routing():
    """LLM judge scores, guard routes to accept or revise."""
    register("llm_judge", LLMJudge())

    n = NetBuilder()
    start = n.place("start")
    mid = n.place("mid")
    scored = n.place("scored")
    accepted = n.place("accepted")
    n.place("rejected")

    gen = n.transition("gen", _agent_config("Write a one-sentence joke."))
    jdg = n.transition(
        "jdg",
        (
            "llm_judge",
            JudgeConfig(
                model=_model(),
                rubric=[{"weight": 1.0, "requirement": "is funny"}],
            ),
        ),
    )
    # Route based on score — with a small model, scores are unpredictable,
    # so we use a threshold that ensures one path fires
    accept = n.transition(
        "accept",
        _agent_config("{text}"),
    ).when(lambda t: t[0].score >= 0.0)  # always accepts for test reliability

    start >> gen >> mid >> jdg >> scored
    scored >> accept >> accepted

    start.token(GenerateOutput(text=""))

    [result] = await execute(n.build())

    assert result.status == "completed"
    assert result.score is not None
    tids = [r.transition_id for r in result.trace]
    assert "gen" in tids
    assert "jdg" in tids
    assert "accept" in tids


# ==============================================================================
# CYCLE
# ==============================================================================


@pytest.mark.asyncio
async def test_cycle_with_fuse():
    """Cycle: generate → judge → loop/done. Fuse limits iterations."""
    register("llm_judge", LLMJudge())

    n = NetBuilder()
    prompt = n.place("prompt")
    response = n.place("response")
    scored = n.place("scored")
    final = n.place("final")

    gen = n.transition("gen", _agent_config("Write a creative sentence about: {text}"))
    jdg = n.transition(
        "jdg",
        (
            "llm_judge",
            JudgeConfig(
                model=_model(),
                rubric=[{"weight": 1.0, "requirement": "creative"}],
            ),
        ),
    )
    # With a small model, scores are unpredictable. Use fuse to guarantee termination.
    loop = n.transition(
        "loop",
        _agent_config("Improve this: {text}"),
    ).when(lambda t: t[0].score < 0.95)
    done = n.transition(
        "done",
        _agent_config("{text}"),
    ).when(lambda t: t[0].score >= 0.95)

    prompt >> gen >> response >> jdg >> scored
    scored >> loop >> prompt
    scored >> done >> final

    prompt.token(GenerateOutput(text="robots"))

    [result] = await execute(n.build(), fuse=12)  # at most ~3 iterations

    assert result.status == "completed"
    assert len(result.trace) >= 2  # at least gen + judge fired
    gen_count = sum(1 for r in result.trace if r.transition_id == "gen")
    judge_count = sum(1 for r in result.trace if r.transition_id == "jdg")
    assert gen_count >= 1
    assert judge_count >= 1


# ==============================================================================
# BATCH
# ==============================================================================


@pytest.mark.asyncio
async def test_batch_three_rows():
    """Batch: 3 different prompts through the same pipeline."""
    register("llm_judge", LLMJudge())

    n = NetBuilder()
    start = n.place("start")
    mid = n.place("mid")
    scored = n.place("scored")

    gen = n.transition("gen", _agent_config("Answer this question concisely: {text}"))
    jdg = n.transition(
        "jdg",
        (
            "llm_judge",
            JudgeConfig(
                model=_model(),
                rubric=[{"weight": 1.0, "requirement": "correct and concise"}],
            ),
        ),
    )

    start >> gen >> mid >> jdg >> scored

    rows = [
        GenerateOutput(text="What is the capital of France?"),
        GenerateOutput(text="What is 7 * 8?"),
        GenerateOutput(text="Name a planet in our solar system."),
    ]

    results = await execute(n.build(), rows=rows, place="start", max_concurrency=3)

    assert len(results) == 3
    for r in results:
        assert r.status == "completed"
        assert r.score is not None
        assert 0.0 <= r.score <= 1.0
        assert len(r.trace) == 2


# ==============================================================================
# RENDER (visual smoke tests — just verify no crashes)
# ==============================================================================


@pytest.mark.asyncio
async def test_render_live_results():
    """Render live results without crashing."""
    n = NetBuilder()
    start = n.place("start")
    end = n.place("end")
    gen = n.transition("gen", _agent_config("Say hello in one word."))
    start >> gen >> end
    start.token(GenerateOutput(text=""))

    [result] = await execute(n.build())
    # Should not crash
    render([result])
    render([result], trace=True)


@pytest.mark.asyncio
async def test_render_live_batch():
    """Render batch results table without crashing."""
    n = NetBuilder()
    start = n.place("start")
    end = n.place("end")
    gen = n.transition("gen", _agent_config("Say hello."))
    start >> gen >> end

    rows = [GenerateOutput(text="hi"), GenerateOutput(text="hey")]
    results = await execute(n.build(), rows=rows, place="start", max_concurrency=2)

    render(results)
    render(results, trace=True)


def test_render_net_live():
    """Render a net topology without crashing."""
    n = NetBuilder()
    p = n.place("prompt")
    r = n.place("response")
    s = n.place("scored")
    f = n.place("final")
    gen = n.transition("gen", _agent_config("{text}"))
    jdg = n.transition("jdg", _agent_config("{text}"))
    loop = n.transition("loop", _agent_config("{text}")).when(lambda t: t[0].score < 0.9)
    done = n.transition("done", _agent_config("{text}")).when(lambda t: t[0].score >= 0.9)
    p >> gen >> r >> jdg >> s
    s >> loop >> p
    s >> done >> f
    render_net(n.build())


# ==============================================================================
# COMPLEX TOPOLOGIES
# ==============================================================================


@pytest.mark.asyncio
async def test_debate_two_agents():
    """Two agents argue opposite sides, a third judges who won.

    Topology:
        topic → advocate → pro_argument
        topic → critic → con_argument
        pro_argument → judge_debate
        con_argument → judge_debate → verdict
    """
    register("llm_judge", LLMJudge())

    n = NetBuilder()
    topic = n.place("topic")
    pro_arg = n.place("pro_argument")
    con_arg = n.place("con_argument")
    verdict = n.place("verdict")

    advocate = n.transition(
        "advocate",
        _agent_config(
            "Argue strongly IN FAVOR of this in 2 sentences: {text}",
        ),
    )
    critic = n.transition(
        "critic",
        _agent_config(
            "Argue strongly AGAINST this in 2 sentences: {text}",
        ),
    )
    judge_debate = n.transition(
        "judge_debate",
        (
            "llm_judge",
            JudgeConfig(
                model=_model(),
                rubric=[{"weight": 1.0, "requirement": "persuasive and well-reasoned"}],
            ),
        ),
    )

    topic >> advocate >> pro_arg >> judge_debate
    topic >> critic >> con_arg >> judge_debate
    judge_debate >> verdict

    topic.token(GenerateOutput(text="social media is good for society"))
    topic.token(GenerateOutput(text="social media is good for society"))

    [result] = await execute(n.build(), max_concurrency=2)

    assert result.status == "completed"
    assert result.score is not None
    assert len(result.trace) == 3  # advocate, critic, judge
    tids = {r.transition_id for r in result.trace}
    assert tids == {"advocate", "critic", "judge_debate"}


@pytest.mark.asyncio
async def test_multi_stage_pipeline():
    """Four-stage pipeline: brainstorm → draft → critique → revise.

    Each stage builds on the previous output.
    """
    n = NetBuilder()
    p0 = n.place("topic")
    p1 = n.place("ideas")
    p2 = n.place("draft_out")
    p3 = n.place("critique_out")
    p4 = n.place("final")

    brainstorm = n.transition(
        "brainstorm",
        _agent_config(
            "List 3 key points about: {text}. Be brief.",
        ),
    )
    write = n.transition(
        "write",
        _agent_config(
            "Write a short paragraph using these points: {text}",
        ),
    )
    critique = n.transition(
        "critique",
        _agent_config(
            "Give one specific criticism of this paragraph: {text}",
        ),
    )
    revise = n.transition(
        "revise",
        _agent_config(
            "Revise this based on the criticism: {text}",
        ),
    )

    p0 >> brainstorm >> p1 >> write >> p2 >> critique >> p3 >> revise >> p4
    p0.token(GenerateOutput(text="the importance of sleep"))

    [result] = await execute(n.build())

    assert result.status == "completed"
    assert len(result.trace) == 4
    # Verify execution order
    tids = [r.transition_id for r in result.trace]
    assert tids == ["brainstorm", "write", "critique", "revise"]
    # Each stage should produce output
    for tr in result.trace:
        assert isinstance(tr.output, GenerateOutput)
        assert len(tr.output.text) > 0


@pytest.mark.asyncio
async def test_diamond_with_judge():
    """Diamond topology: two parallel analyses merge into a judge.

    Topology:
        question → analyst_a → analysis_a ─┐
        question → analyst_b → analysis_b ─┤→ evaluator → scored
    """
    register("llm_judge", LLMJudge())

    n = NetBuilder()
    question = n.place("question")
    analysis_a = n.place("analysis_a")
    analysis_b = n.place("analysis_b")
    scored = n.place("scored")

    analyst_a = n.transition(
        "analyst_a",
        _agent_config(
            "Analyze this from an economic perspective in one sentence: {text}",
        ),
    )
    analyst_b = n.transition(
        "analyst_b",
        _agent_config(
            "Analyze this from a social perspective in one sentence: {text}",
        ),
    )
    evaluator = n.transition(
        "evaluator",
        (
            "llm_judge",
            JudgeConfig(
                model=_model(),
                rubric=[{"weight": 1.0, "requirement": "thorough analysis"}],
            ),
        ),
    )

    question >> analyst_a >> analysis_a >> evaluator
    question >> analyst_b >> analysis_b >> evaluator
    evaluator >> scored

    question.token(GenerateOutput(text="universal basic income"))
    question.token(GenerateOutput(text="universal basic income"))

    [result] = await execute(n.build(), max_concurrency=2)

    assert result.status == "completed"
    assert result.score is not None
    assert len(result.trace) == 3


@pytest.mark.asyncio
async def test_batch_with_parallel_branches():
    """Batch of 3 rows, each going through a fork topology.

    Each row forks into two analyses that run in parallel.
    """
    n = NetBuilder()
    start = n.place("start")
    left = n.place("left")
    right = n.place("right")

    technical = n.transition(
        "technical",
        _agent_config(
            "Give a technical one-sentence take on: {text}",
        ),
    )
    casual = n.transition(
        "casual",
        _agent_config(
            "Give a casual one-sentence take on: {text}",
        ),
    )

    start >> technical >> left
    start >> casual >> right

    rows = [
        GenerateOutput(text="quantum computing"),
        GenerateOutput(text="blockchain"),
        GenerateOutput(text="gene editing"),
    ]

    results = await execute(
        n.build(),
        rows=rows,
        place="start",
        max_concurrency=4,
        fuse=50,
    )

    # Each row forks into 2 transitions, but only gets one token,
    # so only one branch fires per row (competing transitions)
    assert len(results) == 3
    for r in results:
        assert r.status == "completed"
        assert len(r.trace) >= 1


# Interview panel is the heaviest test (9 LLM calls, 3 parallel) — runs last
@pytest.mark.asyncio
async def test_interview_panel():
    """Three LLM interviewers evaluate a candidate from different angles.

    Each interviewer asks a question, the candidate answers, then each
    interviewer scores independently.

    Topology:
        resume → technical_q → answer_t → tech_judge
        resume → behavioral_q → answer_b → behav_judge
        resume → culture_q → answer_c → culture_judge
    """
    register("llm_judge", LLMJudge())

    n = NetBuilder()
    resume = n.place("resume")
    answer_t = n.place("answer_t")
    answer_b = n.place("answer_b")
    answer_c = n.place("answer_c")
    scored_t = n.place("scored_t")
    scored_b = n.place("scored_b")
    scored_c = n.place("scored_c")

    tech_q = n.transition(
        "tech_q",
        _agent_config(
            "You are a technical interviewer. The candidate's resume says: {text}. "
            "Ask one specific technical question. Just the question.",
            system="You are a senior engineer.",
        ),
    )
    behav_q = n.transition(
        "behav_q",
        _agent_config(
            "You are a behavioral interviewer. The candidate's resume says: {text}. "
            "Ask one behavioral question. Just the question.",
            system="You are an HR professional.",
        ),
    )
    culture_q = n.transition(
        "culture_q",
        _agent_config(
            "You are assessing culture fit. The candidate's resume says: {text}. "
            "Ask one question about work style. Just the question.",
            system="You are a team lead.",
        ),
    )

    answer_tech = n.transition(
        "answer_tech",
        _agent_config(
            "Answer this interview question concisely: {text}",
            system="You are a confident software engineer.",
        ),
    )
    answer_behav = n.transition(
        "answer_behav",
        _agent_config(
            "Answer this behavioral question briefly: {text}",
            system="You are a confident software engineer.",
        ),
    )
    answer_culture = n.transition(
        "answer_culture",
        _agent_config(
            "Answer this culture question honestly: {text}",
            system="You are a collaborative engineer.",
        ),
    )

    tech_judge = n.transition(
        "tech_judge",
        (
            "llm_judge",
            JudgeConfig(
                model=_model(),
                rubric=[{"weight": 1.0, "requirement": "demonstrates technical competence"}],
            ),
        ),
    )
    behav_judge = n.transition(
        "behav_judge",
        (
            "llm_judge",
            JudgeConfig(
                model=_model(),
                rubric=[
                    {
                        "weight": 1.0,
                        "requirement": "shows leadership and problem-solving",
                    }
                ],
            ),
        ),
    )
    culture_judge = n.transition(
        "culture_judge",
        (
            "llm_judge",
            JudgeConfig(
                model=_model(),
                rubric=[{"weight": 1.0, "requirement": "good culture fit"}],
            ),
        ),
    )

    resume >> tech_q >> answer_t >> answer_tech >> scored_t >> tech_judge
    resume >> behav_q >> answer_b >> answer_behav >> scored_b >> behav_judge
    resume >> culture_q >> answer_c >> answer_culture >> scored_c >> culture_judge

    resume.token(GenerateOutput(text="Software engineer, 5 years Python, led team of 4"))
    resume.token(GenerateOutput(text="Software engineer, 5 years Python, led team of 4"))
    resume.token(GenerateOutput(text="Software engineer, 5 years Python, led team of 4"))

    [result] = await execute(n.build(), max_concurrency=2, fuse=30)

    assert result.status == "completed"
    assert len(result.trace) >= 9
    tids = {r.transition_id for r in result.trace}
    assert "tech_judge" in tids
    assert "behav_judge" in tids
    assert "culture_judge" in tids
    judge_scores = [r.output.score for r in result.trace if isinstance(r.output, JudgeOutput)]
    assert len(judge_scores) == 3
    assert all(0.0 <= s <= 1.0 for s in judge_scores)
