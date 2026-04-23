"""Callback argument adaptation and executor output normalization."""

from __future__ import annotations

from collections.abc import Sequence

from ..authoring.executor import ExecutorContext
from ..runtime.sinks import Sink
from ..shared.events import BundleRef
from ..shared.token import Token
from ._tokens import normalize_token_bucket, validate_token_list
from .lowering import CompiledEnv


__all__ = [
    "adapt_weighted_inputs",
    "invoke_transition",
    "normalize_transition_outputs",
]


def adapt_weighted_inputs(
    tokens: Sequence[Token],
    *,
    input_weights: tuple[int, ...],
) -> tuple[Token | list[Token], ...]:
    """Slice one reserved token vector into executor arguments by authored weights."""
    expected = sum(input_weights)
    if len(tokens) != expected:
        raise ValueError(
            f"reserved token count {len(tokens)} does not match authored input weight total {expected}"
        )
    cursor = 0
    args: list[Token | list[Token]] = []
    for weight in input_weights:
        if weight == 1:
            args.append(tokens[cursor])
        else:
            args.append(list(tokens[cursor : cursor + weight]))
        cursor += weight
    return tuple(args)


def normalize_transition_outputs(
    value: object,
    *,
    run_key: str,
    output_places: tuple[str, ...],
) -> dict[str, list[Token]]:
    """Normalize one executor return value into canonical outputs-by-place."""
    if not output_places:
        if value is None:
            return {}
        if type(value) is dict and not value:
            return {}
        raise TypeError("zero-output transitions must return None or an empty map")
    if len(output_places) == 1:
        place = output_places[0]
        if type(value) is dict:
            raise TypeError("single-output transitions must return one token or a list of tokens")
        if type(value) is list and not value:
            raise ValueError("single-output transitions must not return an empty list")
        return {place: _normalize_output_bucket(value, run_key=run_key)}
    if type(value) is not dict:
        raise TypeError("multi-output transitions must return a map keyed by output place")
    if set(value) != set(output_places):
        raise ValueError(
            "multi-output transition replies must include every declared output place exactly once"
        )
    return {
        place: _normalize_output_bucket(value[place], run_key=run_key)
        for place in output_places
    }


def _normalize_output_bucket(value: object, *, run_key: str) -> list[Token]:
    return normalize_token_bucket(
        value,
        message="transition outputs must be Token values",
        expected_run_key=run_key,
        run_key_container="transition outputs",
        run_key_error_message="emitted tokens must preserve the current firing run_key",
    )


async def invoke_transition(
    compiled_env: CompiledEnv,
    transition_id: str,
    env: object,
    bundle: BundleRef,
    tokens: Sequence[Token],
    *,
    attempt: int,
    inputs_by_place: dict[str, list[Token]] | None = None,
    sink: Sink | None = None,
) -> dict[str, list[Token]]:
    """Invoke one compiled transition callback from reserved engine tokens."""
    binding = compiled_env.transition_bindings.get(transition_id)
    if binding is None:
        raise ValueError(f"unknown compiled transition {transition_id}")
    if bundle.transition_id != transition_id:
        raise ValueError("bundle transition_id must match the invoked transition id")
    validate_token_list(
        list(tokens),
        message="reserved tokens must be a list of Token values",
        expected_run_key=bundle.run_key,
        run_key_container="reserved tokens",
    )
    ctx = ExecutorContext(
        env=env,
        bundle=bundle,
        executor_name=binding.executor_name,
        attempt=attempt,
        inputs_by_place=inputs_by_place,
        sink=sink,
    )
    args = adapt_weighted_inputs(tokens, input_weights=binding.input_weights)
    result = await binding.executor_spec.fn(ctx, *args)
    return normalize_transition_outputs(
        result,
        run_key=bundle.run_key,
        output_places=binding.output_places,
    )
