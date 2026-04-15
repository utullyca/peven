"""Test executor protocol, built-in executors, and registry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rubric import CriterionReport, EvaluationReport

from peven.petri.executors import Agent, RubricJudge, _extract_text, get, register
from peven.petri.schema import GenerateConfig, JudgeConfig, Token
from peven.petri.types import GenerateOutput, JudgeOutput


def _mock_agent_run(text: str):
    """Patch pydantic_ai.Agent to return static text via async run."""
    mock_result = MagicMock()
    mock_result.output = text
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_result)
    return patch("peven.petri.executors.PydanticAgent", return_value=mock_agent)


def _mock_rubric_grade(score: float):
    """Patch Rubric.grade to return a fixed score."""
    report = EvaluationReport(
        score=score,
        raw_score=score,
        llm_raw_score=score * 100,
        report=[
            CriterionReport(weight=1.0, requirement="be nice", verdict="MET", reason="was nice"),
        ],
    )

    async def mock_grade(self, *a, **kw):
        return report

    return patch("peven.petri.executors.Rubric.grade", mock_grade)


# -- _extract_text() -----------------------------------------------------------


def test_extract_text_from_generate_output():
    tokens = [GenerateOutput(text="hello")]
    assert _extract_text(tokens) == "hello"


def test_extract_text_empty_inputs():
    assert _extract_text([]) == "", "empty input list should return empty string"


def test_extract_text_no_generate_output():
    """Bare tokens have no text — should return empty string."""
    assert _extract_text([Token()]) == ""


def test_extract_text_skips_non_generate():
    """Should find GenerateOutput even if JudgeOutput comes first."""
    tokens = [JudgeOutput(score=0.5), GenerateOutput(text="found")]
    assert _extract_text(tokens) == "found"


def test_extract_text_joins_multiple_inputs():
    """Multiple textual inputs should be joined in order."""
    tokens = [GenerateOutput(text="first"), GenerateOutput(text="second")]
    assert _extract_text(tokens) == "first\n\nsecond"


# -- Agent -------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_returns_output():
    config = GenerateConfig(model="test:model", prompt_template="say {text}")
    inputs = [GenerateOutput(text="hello")]

    with _mock_agent_run("hello world"):
        result = await Agent().execute(inputs, config)

    assert isinstance(result, GenerateOutput)
    assert result.text == "hello world"


@pytest.mark.asyncio
async def test_agent_with_tools():
    def my_tool(query: str) -> str:
        return "tool result"

    config = GenerateConfig(model="test:model", prompt_template="say {text}", tools=[my_tool])
    inputs = [GenerateOutput(text="hello")]

    with _mock_agent_run("used tool"):
        result = await Agent().execute(inputs, config)

    assert isinstance(result, GenerateOutput)
    assert result.text == "used tool"


@pytest.mark.asyncio
async def test_agent_with_model_settings():
    config = GenerateConfig(
        model="test:model",
        prompt_template="say {text}",
        model_settings={"temperature": 0.2, "max_tokens": 100},
    )
    inputs = [GenerateOutput(text="hello")]

    with _mock_agent_run("configured"):
        result = await Agent().execute(inputs, config)

    assert result.text == "configured"


@pytest.mark.asyncio
async def test_agent_bad_config():
    config = JudgeConfig(model="test", rubric=[{"weight": 1.0, "requirement": "be nice"}])
    with pytest.raises(TypeError, match="requires GenerateConfig"):
        await Agent().execute([Token()], config)


# -- RubricJudge -------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_returns_score():
    config = JudgeConfig(model="test", rubric=[{"weight": 1.0, "requirement": "be nice"}])
    inputs = [GenerateOutput(text="I am nice")]

    with _mock_rubric_grade(0.9):
        result = await RubricJudge().execute(inputs, config)

    assert isinstance(result, JudgeOutput)
    assert result.score == 0.9, "score should match rubric grade"
    assert result.raw_score == 0.9, "raw_score should be populated from report"
    assert result.llm_raw_score == 90.0, "llm_raw_score should be populated from report"
    assert result.report is not None, "report should be populated"
    assert len(result.report) == 1


@pytest.mark.asyncio
async def test_judge_bad_config():
    config = GenerateConfig(model="test:model", prompt_template="{text}")
    with pytest.raises(TypeError, match="requires JudgeConfig"):
        await RubricJudge().execute([Token()], config)


# -- Registry ------------------------------------------------------------------


def test_registry_get():
    assert isinstance(get("agent"), Agent)
    assert isinstance(get("judge"), RubricJudge)


def test_registry_unknown():
    with pytest.raises(KeyError, match="Unknown executor"):
        get("nonexistent")


def test_registry_custom():
    class Stub:
        async def execute(self, inputs, config):
            return Token()

    register("stub", Stub())
    assert isinstance(get("stub"), Stub)
