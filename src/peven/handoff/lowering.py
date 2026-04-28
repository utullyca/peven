"""Authoring-to-adapter packaging and compiled callback bindings."""

from __future__ import annotations

from dataclasses import dataclass

import msgspec

from ..authoring.executor import get_executor, validate_executor_signature
from ..authoring.ir import EnvSpec, ExecutorSpec
from ..shared.errors import PevenValidationError, ValidationIssue
from ..shared.token import Marking, Token, validate_structured_payload
from ._tokens import normalize_marking


__all__ = [
    "AUTHORED_ENV_SCHEMA_VERSION",
    "CompiledEnv",
    "EnvSpecMessage",
    "InputArcSpecMessage",
    "OutputArcSpecMessage",
    "PlaceSpecMessage",
    "TransitionBinding",
    "TransitionSpecMessage",
    "compile_env",
    "normalize_initial_marking",
    "package_env_spec",
]


AUTHORED_ENV_SCHEMA_VERSION = 1


class PlaceSpecMessage(msgspec.Struct, frozen=True):
    """Transport form of one authored place definition."""

    id: str
    capacity: int | None = None
    schema: object | None = None


class InputArcSpecMessage(msgspec.Struct, frozen=True):
    """Transport form of one authored input arc."""

    place: str
    weight: int = 1
    optional: bool = False


class OutputArcSpecMessage(msgspec.Struct, frozen=True):
    """Transport form of one authored output arc."""

    place: str


class TransitionSpecMessage(msgspec.Struct, frozen=True):
    """Transport form of one authored transition definition."""

    id: str
    executor: str
    inputs: list[InputArcSpecMessage]
    outputs: list[OutputArcSpecMessage]
    guard_spec: object | None = None
    retries: int = 0
    join_by_spec: object | None = None


class EnvSpecMessage(msgspec.Struct, frozen=True):
    """Top-level authored-IR payload sent to the Julia adapter."""

    schema_version: int
    env_name: str
    places: list[PlaceSpecMessage]
    transitions: list[TransitionSpecMessage]


@dataclass(frozen=True, slots=True)
class TransitionBinding:
    """Python-only callback dispatch metadata for one authored transition."""

    transition_id: str
    executor_name: str
    executor_spec: ExecutorSpec
    input_places: tuple[str, ...]
    input_weights: tuple[int, ...]
    input_optional: tuple[bool, ...]
    output_places: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompiledEnv:
    """Frozen authored env plus the Python callback bindings needed at runtime."""

    env_spec: EnvSpec
    authored_env: EnvSpecMessage
    transition_bindings: dict[str, TransitionBinding]


def package_env_spec(spec: EnvSpec) -> EnvSpecMessage:
    """Package one authored env spec into the adapter-facing authored-IR payload."""
    env = EnvSpecMessage(
        schema_version=AUTHORED_ENV_SCHEMA_VERSION,
        env_name=spec.env_name,
        places=[
            PlaceSpecMessage(
                id=place.id,
                capacity=place.capacity,
                schema=place.schema,
            )
            for place in spec.places
        ],
        transitions=[
            TransitionSpecMessage(
                id=transition.id,
                executor=transition.executor,
                inputs=[
                    InputArcSpecMessage(
                        place=arc.place,
                        weight=arc.weight,
                        optional=arc.optional,
                    )
                    for arc in transition.inputs
                ],
                outputs=[
                    OutputArcSpecMessage(place=arc.place)
                    for arc in transition.outputs
                ],
                guard_spec=transition.guard_spec,
                retries=transition.retries,
                join_by_spec=transition.join_by_spec,
            )
            for transition in spec.transitions
        ],
    )
    _validate_authored_env_message(env)
    return env


def compile_env(spec: EnvSpec) -> CompiledEnv:
    """Freeze one authored env into handoff-ready authored IR and callback bindings."""
    authored_env = package_env_spec(spec)
    executors: dict[str, ExecutorSpec] = {}
    transition_bindings: dict[str, TransitionBinding] = {}

    for transition in spec.transitions:
        executor_spec = executors.get(transition.executor)
        if executor_spec is None:
            executor_spec = get_executor(transition.executor)
            if executor_spec is None:
                raise PevenValidationError(
                    [
                        ValidationIssue(
                            "unknown_executor",
                            transition.id,
                            f"unknown executor {transition.executor}",
                        )
                    ]
                )
            executors[transition.executor] = executor_spec
        validate_executor_signature(
            executor_spec,
            input_count=len(transition.inputs),
            object_id=transition.id,
        )
        callback_inputs = transition.inputs
        if transition.join_by_spec is not None:
            callback_inputs = tuple(sorted(transition.inputs, key=lambda arc: arc.place))
        transition_bindings[transition.id] = TransitionBinding(
            transition_id=transition.id,
            executor_name=transition.executor,
            executor_spec=executor_spec,
            input_places=tuple(arc.place for arc in callback_inputs),
            input_weights=tuple(arc.weight for arc in callback_inputs),
            input_optional=tuple(arc.optional for arc in callback_inputs),
            output_places=tuple(arc.place for arc in transition.outputs),
        )

    return CompiledEnv(
        env_spec=spec,
        authored_env=authored_env,
        transition_bindings=transition_bindings,
    )


def normalize_initial_marking(marking: Marking) -> dict[str, list[Token]]:
    """Normalize one authored marking into the engine token-bucket shape."""
    return normalize_marking(marking)

def _validate_authored_env_message(env: EnvSpecMessage) -> None:
    """Enforce authored-IR transport shape without re-validating engine semantics."""
    if type(env.schema_version) is not int or env.schema_version <= 0:
        raise ValueError("authored env schema_version must be a positive integer")
    if type(env.env_name) is not str or not env.env_name:
        raise TypeError("authored env env_name must be a non-empty string")
    if type(env.places) is not list:
        raise TypeError("authored env places must be a list")
    if type(env.transitions) is not list:
        raise TypeError("authored env transitions must be a list")
    for place in env.places:
        _validate_place_spec_message(place)
    for transition in env.transitions:
        _validate_transition_spec_message(transition)


def _validate_place_spec_message(place: PlaceSpecMessage) -> None:
    """Validate one authored place payload as schema-safe transport data."""
    if type(place.id) is not str or not place.id:
        raise TypeError("authored place id must be a non-empty string")
    if place.capacity is not None and (
        type(place.capacity) is not int or place.capacity <= 0
    ):
        raise ValueError("authored place capacity must be a positive int or None")
    if place.schema is not None:
        validate_structured_payload(place.schema)


def _validate_transition_spec_message(transition: TransitionSpecMessage) -> None:
    """Validate one authored transition payload as schema-safe transport data."""
    if type(transition.id) is not str or not transition.id:
        raise TypeError("authored transition id must be a non-empty string")
    if type(transition.executor) is not str or not transition.executor:
        raise TypeError("authored transition executor must be a non-empty string")
    if type(transition.inputs) is not list:
        raise TypeError("authored transition inputs must be a list")
    if type(transition.outputs) is not list:
        raise TypeError("authored transition outputs must be a list")
    if type(transition.retries) is not int or transition.retries < 0:
        raise ValueError("authored transition retries must be a non-negative int")
    if transition.guard_spec is not None:
        validate_structured_payload(transition.guard_spec)
    if transition.join_by_spec is not None:
        validate_structured_payload(transition.join_by_spec)
    for arc in transition.inputs:
        if type(arc.place) is not str or not arc.place:
            raise TypeError("authored input arc place must be a non-empty string")
        if type(arc.weight) is not int or arc.weight <= 0:
            raise ValueError("authored input arc weight must be a positive int")
        if type(getattr(arc, "optional", False)) is not bool:
            raise TypeError("authored input arc optional must be a bool")
    for arc in transition.outputs:
        if type(arc.place) is not str or not arc.place:
            raise TypeError("authored output arc place must be a non-empty string")
