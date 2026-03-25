"""Petri net schema."""

from __future__ import annotations

from typing import Annotated, Any, Callable, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag
from pydantic_ai.models import Model


class ValidationError(ValueError):
    """Raised when the net violates structural rules."""


class Token(BaseModel):
    """Base token. A bare token is just a signal — subclass to carry data."""

    run_id: Optional[str] = Field(default=None)


# -- Transition configs --------------------------------------------------------


class GenerateConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["generate"] = "generate"
    model: Union[str, Model] = Field(exclude=True)
    prompt_template: str
    system_prompt: Optional[str] = Field(default=None)
    tools: Optional[list[Callable]] = Field(default=None, exclude=True)
    model_settings: Optional[dict[str, Any]] = Field(default=None)


class JudgeConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["judge"] = "judge"
    model: Union[str, Model] = Field(exclude=True)
    rubric: list[dict[str, Any]]
    strategy: Literal["per_criterion", "oneshot", "rubric_as_judge"] = Field(
        default="per_criterion"
    )
    pass_threshold: float = Field(default=0.5)


TransitionConfig = Annotated[
    Union[
        Annotated[GenerateConfig, Tag("generate")],
        Annotated[JudgeConfig, Tag("judge")],
    ],
    Discriminator("type"),
]


class Place(BaseModel):
    id: str
    capacity: Optional[int] = Field(default=None)


class Transition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    executor: str
    config: Optional[TransitionConfig] = Field(default=None)
    when: Optional[Callable] = Field(default=None, exclude=True)
    retries: int = Field(
        default=0,
        description="Number of retry attempts after failure. "
        "retries=0 means no retries (fail immediately). "
        "retries=2 means up to 2 retries (3 total attempts). "
        "On retry, consumed tokens are re-deposited to their input places.",
    )


class Arc(BaseModel):
    source: str
    target: str
    weight: int = Field(default=1)


class Marking(BaseModel):
    tokens: dict[str, list[Token]] = Field(default_factory=dict)


class Net(BaseModel):
    places: list[Place]
    transitions: list[Transition]
    arcs: list[Arc]
    initial_marking: Marking
