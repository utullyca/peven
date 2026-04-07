"""Executor protocol, built-in executors, and registry."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic_ai import Agent as PydanticAgent

from peven.petri.schema import Token, TransitionConfig
from peven.petri.types import GenerateOutput, JudgeOutput
from rubric import Rubric


def _extract_text(tokens: list[Token]) -> str:
    """Pull text from the first GenerateOutput token."""
    for token in tokens:
        if isinstance(token, GenerateOutput):
            return token.text
    return ""


# -- Protocol ------------------------------------------------------------------


@runtime_checkable
class Executor(Protocol):
    async def execute(self, inputs: list[Token], config: TransitionConfig | None) -> Token: ...


# -- Built-in executors --------------------------------------------------------


class Agent:
    """LLM generation via pydantic_ai."""

    async def execute(self, inputs: list[Token], config: TransitionConfig | None) -> Token:
        from peven.petri.schema import GenerateConfig

        if not isinstance(config, GenerateConfig):
            raise TypeError(f"Agent executor requires GenerateConfig, got {type(config)}")

        text = _extract_text(inputs)
        prompt = config.prompt_template.format(text=text)
        kwargs = {"model": config.model, "system_prompt": config.system_prompt or ""}
        if config.tools:
            kwargs["tools"] = config.tools
        agent = PydanticAgent(**kwargs)
        run_kwargs = {}
        if config.model_settings:
            run_kwargs["model_settings"] = config.model_settings
        result = await agent.run(prompt, **run_kwargs)
        return GenerateOutput(text=result.output)


_GRADERS = {
    "per_criterion": ("PerCriterionGrader", "PerCriterionOutput"),
    "oneshot": ("PerCriterionOneShotGrader", "OneShotOutput"),
    "rubric_as_judge": ("RubricAsJudgeGrader", "RubricAsJudgeOutput"),
}


class RubricJudge:
    """Rubric-based scoring."""

    async def execute(self, inputs: list[Token], config: TransitionConfig | None) -> Token:
        import rubric.autograders as graders
        import rubric.autograders.schemas as schemas
        from peven.petri.schema import JudgeConfig

        if not isinstance(config, JudgeConfig):
            raise TypeError(f"RubricJudge executor requires JudgeConfig, got {type(config)}")

        grader_name, output_name = _GRADERS[config.strategy]
        grader_cls = getattr(graders, grader_name)
        output_cls = getattr(schemas, output_name)

        candidate = _extract_text(inputs)
        r = Rubric.from_dict(config.rubric)

        agent = PydanticAgent(model=config.model, output_type=output_cls)

        async def generate_fn(system_prompt: str, user_prompt: str, **kwargs):
            result = await agent.run(user_prompt, instructions=system_prompt)
            return result.output

        grader = grader_cls(generate_fn=generate_fn)
        report = await r.grade(candidate, autograder=grader)
        return JudgeOutput(
            score=report.score,
            raw_score=report.raw_score,
            llm_raw_score=report.llm_raw_score,
            report=report.report,
        )


# -- Registry ------------------------------------------------------------------

_REGISTRY: dict[str, Executor] = {
    "agent": Agent(),
    "judge": RubricJudge(),
}


def register(name: str, executor: Executor) -> None:
    """Register a custom executor."""
    _REGISTRY[name] = executor


def get(name: str) -> Executor:
    """Look up an executor by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown executor: {name!r}. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name]
