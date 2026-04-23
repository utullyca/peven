"""Adapter transport models plus protocol encode/decode helpers."""

from __future__ import annotations

from typing import Literal, TypeAlias, cast

import msgspec

from ..shared.events import (
    RUN_STATUS_VALUES,
    TERMINAL_REASON_VALUES,
    TRANSITION_STATUS_VALUES,
    BundleRef,
    GuardErrored,
    RunFinished,
    RunResult,
    RunStatus,
    RuntimeEvent,
    SelectionErrored,
    TerminalReason,
    TransitionCompleted,
    TransitionFailed,
    TransitionResult,
    TransitionStarted,
    TransitionStatus,
)
from ..shared.token import Token, validate_structured_payload
from ._tokens import normalize_token_buckets, validate_token_list
from .lowering import EnvSpecMessage


class LoadEnv(msgspec.Struct, frozen=True, tag="load_env", tag_field="kind"):
    """Authoring payload sent from Python to the Julia adapter."""

    req_id: int
    env: EnvSpecMessage


class LoadEnvOk(msgspec.Struct, frozen=True, tag="load_env_ok", tag_field="kind"):
    """Positive acknowledgement for one authored env load."""

    req_id: int


class LoadEnvError(msgspec.Struct, frozen=True, tag="load_env_error", tag_field="kind"):
    """Adapter-side rejection of one authored env load."""

    req_id: int
    error: str


class RunEnv(msgspec.Struct, frozen=True, tag="run_env", tag_field="kind"):
    """Run-start request carrying the normalized initial marking for one env run."""

    req_id: int
    env_run_id: int
    initial_marking: dict[str, list[Token]]
    fuse: int | None = None


class RunEnvOk(msgspec.Struct, frozen=True, tag="run_env_ok", tag_field="kind"):
    req_id: int
    env_run_id: int


class RunEnvError(msgspec.Struct, frozen=True, tag="run_env_error", tag_field="kind"):
    req_id: int
    env_run_id: int
    error: str


class CallbackBundle(msgspec.Struct, frozen=True):
    transition_id: str
    run_key: str
    ordinal: int
    selected_key: object | None = None


class CallbackRequest(
    msgspec.Struct, frozen=True, tag="callback_request", tag_field="kind"
):
    req_id: int
    env_run_id: int
    transition_id: str
    bundle: CallbackBundle
    tokens: list[Token]
    attempt: int
    inputs_by_place: dict[str, list[Token]] = msgspec.field(default_factory=dict)


class CallbackReply(msgspec.Struct, frozen=True, tag="callback_reply", tag_field="kind"):
    req_id: int
    env_run_id: int
    outputs: dict[str, list[Token]]


class CallbackError(msgspec.Struct, frozen=True, tag="callback_error", tag_field="kind"):
    req_id: int
    env_run_id: int
    error: str


class TransitionStartedMessage(
    msgspec.Struct, frozen=True, tag="transition_started", tag_field="kind"
):
    env_run_id: int
    bundle: CallbackBundle
    firing_id: int
    attempt: int
    inputs: list[Token]
    inputs_by_place: dict[str, list[Token]] = msgspec.field(default_factory=dict)


class TransitionCompletedMessage(
    msgspec.Struct, frozen=True, tag="transition_completed", tag_field="kind"
):
    env_run_id: int
    bundle: CallbackBundle
    firing_id: int
    attempt: int
    outputs: dict[str, list[Token]]


class TransitionFailedMessage(
    msgspec.Struct, frozen=True, tag="transition_failed", tag_field="kind"
):
    env_run_id: int
    bundle: CallbackBundle
    firing_id: int
    attempt: int
    error: str
    retrying: bool


class GuardErroredMessage(
    msgspec.Struct, frozen=True, tag="guard_errored", tag_field="kind"
):
    env_run_id: int
    bundle: CallbackBundle
    error: str


class SelectionErroredMessage(
    msgspec.Struct, frozen=True, tag="selection_errored", tag_field="kind"
):
    env_run_id: int
    transition_id: str
    run_key: str
    error: str


class TransitionResultMessage(msgspec.Struct, frozen=True):
    bundle: CallbackBundle
    firing_id: int
    status: str
    outputs: dict[str, list[Token]]
    error: str | None = None
    attempts: int = 1


class RunResultMessage(msgspec.Struct, frozen=True):
    run_key: str
    status: str
    error: str | None = None
    terminal_reason: str | None = None
    terminal_bundle: CallbackBundle | None = None
    terminal_transition: str | None = None
    trace: list[TransitionResultMessage] = msgspec.field(default_factory=list)
    final_marking: dict[str, list[Token]] = msgspec.field(default_factory=dict)


class RunFinishedMessage(
    msgspec.Struct, frozen=True, tag="run_finished", tag_field="kind"
):
    env_run_id: int
    result: RunResultMessage


LoadEnvReply: TypeAlias = LoadEnvOk | LoadEnvError
RunEnvReply: TypeAlias = RunEnvOk | RunEnvError
CallbackReplyOrError: TypeAlias = CallbackReply | CallbackError
RuntimeEventMessage: TypeAlias = (
    TransitionStartedMessage
    | TransitionCompletedMessage
    | TransitionFailedMessage
    | GuardErroredMessage
    | SelectionErroredMessage
    | RunFinishedMessage
)
AdapterMessage: TypeAlias = CallbackRequest | RuntimeEventMessage


_load_env_reply_decoder = msgspec.msgpack.Decoder(LoadEnvReply)
_run_env_reply_decoder = msgspec.msgpack.Decoder(RunEnvReply)
_callback_request_decoder = msgspec.msgpack.Decoder(CallbackRequest)
_callback_reply_decoder = msgspec.msgpack.Decoder(CallbackReplyOrError)
_runtime_event_decoder = msgspec.msgpack.Decoder(RuntimeEventMessage)
_adapter_message_decoder = msgspec.msgpack.Decoder(AdapterMessage)


__all__ = [
    "CallbackBundle",
    "CallbackError",
    "CallbackReply",
    "CallbackRequest",
    "GuardErroredMessage",
    "LoadEnv",
    "LoadEnvError",
    "LoadEnvOk",
    "RunEnv",
    "RunEnvError",
    "RunEnvOk",
    "RunFinishedMessage",
    "RunResultMessage",
    "SelectionErroredMessage",
    "TransitionCompletedMessage",
    "TransitionFailedMessage",
    "TransitionResultMessage",
    "TransitionStartedMessage",
    "bundle_ref_from_callback_bundle",
    "decode_adapter_message",
    "decode_callback_reply",
    "decode_callback_request",
    "decode_load_env_reply",
    "decode_run_env_reply",
    "decode_runtime_event",
    "make_callback_error",
    "make_callback_reply",
    "make_load_env",
    "make_run_env",
    "normalize_runtime_event",
    "validate_callback_request",
    "validate_structured_payload",
]


def make_load_env(*, req_id: int, env: EnvSpecMessage) -> LoadEnv:
    """Build one authored env load request after request-id validation."""
    _validate_req_id(req_id)
    return LoadEnv(req_id=req_id, env=env)


def make_run_env(
    *,
    req_id: int,
    env_run_id: int,
    initial_marking: dict[str, list[Token]],
    fuse: int | None = None,
) -> RunEnv:
    _validate_req_id(req_id)
    _validate_env_run_id(env_run_id)
    _validate_fuse(fuse)
    normalized = normalize_token_buckets(
        initial_marking,
        container_name="run initial_marking",
    )
    return RunEnv(
        req_id=req_id,
        env_run_id=env_run_id,
        initial_marking=normalized,
        fuse=fuse,
    )


def make_callback_reply(
    *,
    req_id: int,
    env_run_id: int,
    outputs: dict[str, list[Token]],
) -> CallbackReply:
    _validate_adapter_req_id(req_id)
    _validate_env_run_id(env_run_id)
    normalized = normalize_token_buckets(
        outputs,
        container_name="callback outputs",
    )
    return CallbackReply(req_id=req_id, env_run_id=env_run_id, outputs=normalized)


def make_callback_error(*, req_id: int, env_run_id: int, error: str) -> CallbackError:
    _validate_adapter_req_id(req_id)
    _validate_env_run_id(env_run_id)
    if not error:
        raise TypeError("callback error must be a non-empty string")
    return CallbackError(req_id=req_id, env_run_id=env_run_id, error=error)


def decode_load_env_reply(payload: bytes) -> LoadEnvOk | LoadEnvError:
    """Decode one adapter reply for an authored env load request."""
    reply = _decode_union(_load_env_reply_decoder, payload, descriptor="load reply")
    _validate_req_id(reply.req_id)
    if isinstance(reply, LoadEnvError) and not reply.error:
        raise TypeError("load reply error must be a non-empty string")
    return reply


def decode_run_env_reply(payload: bytes) -> RunEnvOk | RunEnvError:
    reply = _decode_union(_run_env_reply_decoder, payload, descriptor="run reply")
    _validate_req_id(reply.req_id)
    _validate_env_run_id(reply.env_run_id)
    if isinstance(reply, RunEnvError) and not reply.error:
        raise TypeError("run reply error must be a non-empty string")
    return reply


def decode_callback_request(payload: bytes) -> CallbackRequest:
    request = _decode_union(
        _callback_request_decoder, payload, descriptor="callback request"
    )
    _validate_callback_request(request)
    return request


def decode_callback_reply(payload: bytes) -> CallbackReply | CallbackError:
    reply = _decode_union(_callback_reply_decoder, payload, descriptor="callback reply")
    _validate_adapter_req_id(reply.req_id)
    _validate_env_run_id(reply.env_run_id)
    if isinstance(reply, CallbackError):
        if not reply.error:
            raise TypeError("callback error must be a non-empty string")
        return reply
    normalize_token_buckets(reply.outputs, container_name="callback outputs")
    return reply


def decode_runtime_event(payload: bytes) -> tuple[int, RuntimeEvent]:
    event = _decode_union(_runtime_event_decoder, payload, descriptor="runtime event")
    return _normalize_runtime_event(event)


def decode_adapter_message(payload: bytes) -> AdapterMessage:
    return _decode_union(_adapter_message_decoder, payload, descriptor="adapter message")


def validate_callback_request(request: CallbackRequest) -> CallbackRequest:
    """Apply post-decode validation to one callback request and return it."""
    _validate_callback_request(request)
    return request


def normalize_runtime_event(event: RuntimeEventMessage) -> tuple[int, RuntimeEvent]:
    """Normalize one decoded runtime-event message into its public event form."""
    return _normalize_runtime_event(event)


_UnionDescriptor: TypeAlias = Literal[
    "load reply",
    "run reply",
    "callback request",
    "callback reply",
    "runtime event",
    "adapter message",
]


def _decode_union(
    decoder: msgspec.msgpack.Decoder, payload: bytes, *, descriptor: _UnionDescriptor
):
    try:
        return decoder.decode(payload)
    except (msgspec.DecodeError, msgspec.ValidationError) as exc:
        raise ValueError(f"malformed {descriptor} payload") from exc


def _validate_callback_request(request: CallbackRequest) -> None:
    _validate_adapter_req_id(request.req_id)
    _validate_env_run_id(request.env_run_id)
    if not request.transition_id:
        raise TypeError("callback request transition_id must be a non-empty string")
    if request.transition_id != request.bundle.transition_id:
        raise ValueError("callback request transition_id must match bundle.transition_id")
    if request.attempt <= 0:
        raise ValueError("callback request attempt must be a positive integer")
    _validate_callback_bundle(request.bundle)
    validate_token_list(
        request.tokens,
        message="callback request tokens must be a list of Token values",
    )
    normalize_token_buckets(
        request.inputs_by_place,
        container_name="callback request inputs_by_place",
        expected_run_key=request.bundle.run_key,
    )


def _normalize_runtime_event(event: RuntimeEventMessage) -> tuple[int, RuntimeEvent]:
    _validate_env_run_id(event.env_run_id)
    if isinstance(event, TransitionStartedMessage):
        bundle = bundle_ref_from_callback_bundle(event.bundle)
        _validate_firing_id_and_attempt(event.firing_id, event.attempt, event_name="transition_started")
        inputs = validate_token_list(
            event.inputs,
            message="transition_started inputs must be a list of Token values",
            expected_run_key=bundle.run_key,
            run_key_container="transition_started inputs",
        )
        inputs_by_place = normalize_token_buckets(
            event.inputs_by_place,
            container_name="transition_started inputs_by_place",
            expected_run_key=bundle.run_key,
        )
        return event.env_run_id, TransitionStarted(
            bundle=bundle,
            firing_id=event.firing_id,
            attempt=event.attempt,
            inputs=inputs,
            inputs_by_place=inputs_by_place,
        )
    if isinstance(event, TransitionCompletedMessage):
        bundle = bundle_ref_from_callback_bundle(event.bundle)
        _validate_firing_id_and_attempt(event.firing_id, event.attempt, event_name="transition_completed")
        outputs = normalize_token_buckets(
            event.outputs,
            container_name="transition_completed outputs",
            expected_run_key=bundle.run_key,
        )
        return event.env_run_id, TransitionCompleted(
            bundle=bundle,
            firing_id=event.firing_id,
            attempt=event.attempt,
            outputs=outputs,
        )
    if isinstance(event, TransitionFailedMessage):
        bundle = bundle_ref_from_callback_bundle(event.bundle)
        _validate_firing_id_and_attempt(event.firing_id, event.attempt, event_name="transition_failed")
        if not event.error:
            raise TypeError("transition_failed error must be a non-empty string")
        return event.env_run_id, TransitionFailed(
            bundle=bundle,
            firing_id=event.firing_id,
            attempt=event.attempt,
            error=event.error,
            retrying=event.retrying,
        )
    if isinstance(event, GuardErroredMessage):
        bundle = bundle_ref_from_callback_bundle(event.bundle)
        if not event.error:
            raise TypeError("guard_errored error must be a non-empty string")
        return event.env_run_id, GuardErrored(bundle=bundle, error=event.error)
    if isinstance(event, SelectionErroredMessage):
        if not event.transition_id:
            raise TypeError("selection_errored transition_id must be a non-empty string")
        if not event.run_key:
            raise TypeError("selection_errored run_key must be a non-empty string")
        if not event.error:
            raise TypeError("selection_errored error must be a non-empty string")
        return event.env_run_id, SelectionErrored(
            transition_id=event.transition_id,
            run_key=event.run_key,
            error=event.error,
        )
    return event.env_run_id, RunFinished(result=_decode_run_result(event.result))


def _validate_firing_id_and_attempt(firing_id: int, attempt: int, *, event_name: str) -> None:
    if firing_id <= 0:
        raise ValueError(f"{event_name} firing_id must be positive")
    if attempt <= 0:
        raise ValueError(f"{event_name} attempt must be positive")


def _validate_req_id(req_id: int) -> None:
    if type(req_id) is not int or req_id <= 0 or req_id % 2 == 0:
        raise ValueError("protocol req_id must be an odd positive integer")


def _validate_env_run_id(env_run_id: int) -> None:
    if type(env_run_id) is not int or env_run_id <= 0:
        raise ValueError("protocol env_run_id must be a positive integer")


def _validate_fuse(fuse: int | None) -> None:
    if fuse is None:
        return
    if type(fuse) is not int or fuse < 0:
        raise ValueError("fuse must be a non-negative integer when present")


def _validate_adapter_req_id(req_id: int) -> None:
    if type(req_id) is not int or req_id <= 0 or req_id % 2 != 0:
        raise ValueError("adapter req_id must be an even positive integer")


def _validate_callback_bundle(bundle: CallbackBundle) -> None:
    if not bundle.transition_id:
        raise TypeError("callback bundle transition_id must be a non-empty string")
    if not bundle.run_key:
        raise TypeError("callback bundle run_key must be a non-empty string")
    if bundle.ordinal <= 0:
        raise ValueError("callback bundle ordinal must be a positive integer")
    validate_structured_payload(bundle.selected_key)


def bundle_ref_from_callback_bundle(bundle: CallbackBundle) -> BundleRef:
    _validate_callback_bundle(bundle)
    return BundleRef(
        transition_id=bundle.transition_id,
        run_key=bundle.run_key,
        ordinal=bundle.ordinal,
        selected_key=bundle.selected_key,
    )


def _decode_transition_result(
    result: TransitionResultMessage,
    *,
    run_key: str,
) -> TransitionResult:
    bundle = bundle_ref_from_callback_bundle(result.bundle)
    if bundle.run_key != run_key:
        raise ValueError("run_finished trace bundles must preserve the run_result run_key")
    if result.firing_id <= 0:
        raise ValueError("run_finished trace firing_id must be positive")
    if result.status not in TRANSITION_STATUS_VALUES:
        raise ValueError(f"unexpected transition result status: {result.status!r}")
    outputs = normalize_token_buckets(
        result.outputs,
        container_name="run_finished trace outputs",
        expected_run_key=run_key,
    )
    if result.error is not None and not result.error:
        raise TypeError("run_finished trace error must be a non-empty string or None")
    if result.attempts <= 0:
        raise ValueError("run_finished trace attempts must be positive")
    return TransitionResult(
        bundle=bundle,
        firing_id=result.firing_id,
        status=cast("TransitionStatus", result.status),
        outputs=outputs,
        error=result.error,
        attempts=result.attempts,
    )


def _decode_run_result(result: RunResultMessage) -> RunResult:
    if not result.run_key:
        raise TypeError("run_finished result run_key must be a non-empty string")
    if result.status not in RUN_STATUS_VALUES:
        raise ValueError(f"unexpected run result status: {result.status!r}")
    if result.error is not None and not result.error:
        raise TypeError("run_finished result error must be a non-empty string or None")
    if result.terminal_reason not in TERMINAL_REASON_VALUES:
        raise ValueError(f"unexpected terminal reason: {result.terminal_reason!r}")
    terminal_bundle = None
    if result.terminal_bundle is not None:
        terminal_bundle = bundle_ref_from_callback_bundle(result.terminal_bundle)
        if terminal_bundle.run_key != result.run_key:
            raise ValueError(
                "run_finished terminal_bundle must preserve the run_result run_key"
            )
    if result.terminal_transition is not None and not result.terminal_transition:
        raise TypeError("run_finished terminal_transition must be a non-empty string or None")
    trace = [
        _decode_transition_result(item, run_key=result.run_key) for item in result.trace
    ]
    final_marking = normalize_token_buckets(
        result.final_marking,
        container_name="run_finished final_marking",
        expected_run_key=result.run_key,
    )
    return RunResult(
        run_key=result.run_key,
        status=cast("RunStatus", result.status),
        error=result.error,
        terminal_reason=cast("TerminalReason", result.terminal_reason),
        terminal_bundle=terminal_bundle,
        terminal_transition=result.terminal_transition,
        trace=trace,
        final_marking=final_marking,
    )
