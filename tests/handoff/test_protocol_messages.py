from __future__ import annotations

import msgspec
import pytest

import peven
import peven.handoff.messages as messages_module
from peven.handoff.callbacks import adapt_weighted_inputs, normalize_transition_outputs
from peven.handoff.lowering import normalize_initial_marking, package_env_spec
from peven.handoff.messages import (
    CallbackBundle,
    CallbackError,
    CallbackReply,
    CallbackRequest,
    GuardErroredMessage,
    LoadEnv,
    LoadEnvError,
    LoadEnvOk,
    RunEnv,
    RunEnvError,
    RunEnvOk,
    RunFinishedMessage,
    RunResultMessage,
    SelectionErroredMessage,
    TransitionCompletedMessage,
    TransitionFailedMessage,
    TransitionResultMessage,
    TransitionStartedMessage,
    bundle_ref_from_callback_bundle,
    decode_callback_reply,
    decode_callback_request,
    decode_load_env_reply,
    decode_run_env_reply,
    decode_runtime_event,
    make_callback_error,
    make_callback_reply,
    make_load_env,
    make_run_env,
)
from peven.shared.token import Token


def test_protocol_messages_do_not_export_dead_structured_payload_helpers() -> None:
    assert not hasattr(messages_module, "encode_structured_payload")
    assert not hasattr(messages_module, "decode_structured_payload")


def test_normalize_initial_marking_requires_concrete_run_keys() -> None:
    token = peven.token({"value": 1}, run_key="rk-1")
    marking = peven.Marking({"ready": [token]})

    assert normalize_initial_marking(marking) == {"ready": [token]}


def test_adapt_weighted_inputs_matches_authored_input_weights() -> None:
    left = peven.token({"side": "left"}, run_key="rk")
    right_1 = peven.token({"side": "right-1"}, run_key="rk")
    right_2 = peven.token({"side": "right-2"}, run_key="rk")

    adapted = adapt_weighted_inputs(
        [left, right_1, right_2],
        input_weights=(1, 2),
    )

    assert adapted == (left, [right_1, right_2])

    with pytest.raises(ValueError, match="does not match authored input weight total"):
        adapt_weighted_inputs([left], input_weights=(1, 2))


def test_normalize_transition_outputs_single_output_requires_concrete_tokens() -> None:
    emission = normalize_transition_outputs(
        [
            peven.token({"value": 1}, run_key="rk-1"),
            peven.token({"value": 2}, run_key="rk-1", color="scored"),
        ],
        run_key="rk-1",
        output_places=("done",),
    )

    assert emission == {
        "done": [
            Token(run_key="rk-1", color="default", payload={"value": 1}),
            Token(run_key="rk-1", color="scored", payload={"value": 2}),
        ],
    }

    single = normalize_transition_outputs(
        peven.token({"value": 3}, run_key="rk-1"),
        run_key="rk-1",
        output_places=("done",),
    )

    assert single == {
        "done": [Token(run_key="rk-1", color="default", payload={"value": 3})],
    }

    with pytest.raises(ValueError, match="must not return an empty list"):
        normalize_transition_outputs(
            [],
            run_key="rk-1",
            output_places=("done",),
        )


def test_normalize_transition_outputs_supports_multi_output_engine_contract() -> None:
    emission = normalize_transition_outputs(
        {
            "accepted": peven.token({"value": "ok"}, run_key="rk-2"),
            "rejected": [peven.token({"value": "no"}, run_key="rk-2")],
        },
        run_key="rk-2",
        output_places=("accepted", "rejected"),
    )

    assert emission == {
        "accepted": [Token(run_key="rk-2", color="default", payload={"value": "ok"})],
        "rejected": [Token(run_key="rk-2", color="default", payload={"value": "no"})],
    }

    with pytest.raises(ValueError, match="every declared output place exactly once"):
        normalize_transition_outputs(
            {"accepted": peven.token({"value": "ok"}, run_key="rk-2")},
            run_key="rk-2",
            output_places=("accepted", "rejected"),
        )

    with pytest.raises(TypeError, match="must return a map keyed by output place"):
        normalize_transition_outputs(
            peven.token({"value": "bad"}, run_key="rk-2"),
            run_key="rk-2",
            output_places=("accepted", "rejected"),
        )

    with pytest.raises(TypeError, match="must return one token or a list of tokens"):
        normalize_transition_outputs(
            {"done": peven.token({"value": "bad"}, run_key="rk-2")},
            run_key="rk-2",
            output_places=("done",),
        )

    with pytest.raises(ValueError, match="preserve the current firing run_key"):
        normalize_transition_outputs(
            peven.token({"value": "bad"}, run_key="other"),
            run_key="rk-2",
            output_places=("done",),
        )

    with pytest.raises(TypeError, match="Token values"):
        normalize_transition_outputs(
            {"accepted": {"value": "ok"}, "rejected": []},
            run_key="rk-2",
            output_places=("accepted", "rejected"),
        )


def test_normalize_transition_outputs_allows_empty_output_buckets() -> None:
    emission = normalize_transition_outputs(
        {"accepted": [], "rejected": [peven.token({"value": "no"}, run_key="rk-2")]},
        run_key="rk-2",
        output_places=("accepted", "rejected"),
    )

    assert emission == {
        "accepted": [],
        "rejected": [Token(run_key="rk-2", color="default", payload={"value": "no"})],
    }


def test_normalize_transition_outputs_supports_zero_output_transitions() -> None:
    assert normalize_transition_outputs(
        None,
        run_key="rk-2",
        output_places=(),
    ) == {}
    assert normalize_transition_outputs(
        {},
        run_key="rk-2",
        output_places=(),
    ) == {}

    with pytest.raises(TypeError, match="zero-output transitions must return None or an empty map"):
        normalize_transition_outputs(
            peven.token({"value": "bad"}, run_key="rk-2"),
            run_key="rk-2",
            output_places=(),
        )


def test_load_and_run_messages_are_transportable_and_validate_ids() -> None:
    namespace = {
        "__name__": "tests.handoff.generated_protocol_message_executors",
        "peven": peven,
    }
    exec(
        """
@peven.executor("default_protocol_message")
async def default_protocol_message(ctx, ready):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
""",
        namespace,
    )

    @peven.env("protocol_message_env")
    class ProtocolMessageEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="default_protocol_message",
        )

    authored_env = package_env_spec(ProtocolMessageEnv.spec())
    marking = normalize_initial_marking(
        peven.Marking({"ready": [peven.token({"seed": 1}, run_key="rk-1")]})
    )

    load_message = make_load_env(req_id=1, env=authored_env)
    run_message = make_run_env(req_id=3, env_run_id=1, initial_marking=marking, fuse=7)

    assert isinstance(load_message, LoadEnv)
    assert isinstance(run_message, RunEnv)
    encoded_load = msgspec.msgpack.decode(
        msgspec.msgpack.encode(load_message),
        type=dict[str, object],
    )
    encoded_run = msgspec.msgpack.decode(
        msgspec.msgpack.encode(run_message),
        type=dict[str, object],
    )
    assert encoded_load["kind"] == "load_env"
    assert isinstance(encoded_load["env"], dict)
    assert encoded_run["kind"] == "run_env"
    assert encoded_run["fuse"] == 7

    with pytest.raises(ValueError, match="odd positive integer"):
        make_load_env(req_id=2, env=authored_env)

    with pytest.raises(ValueError, match="positive integer"):
        make_run_env(req_id=3, env_run_id=0, initial_marking=marking)

    with pytest.raises(ValueError, match="fuse must be a non-negative integer when present"):
        make_run_env(req_id=3, env_run_id=1, initial_marking=marking, fuse=-1)


def test_decode_load_and_run_replies_return_typed_structs() -> None:
    assert decode_load_env_reply(
        msgspec.msgpack.encode(LoadEnvOk(req_id=1))
    ) == LoadEnvOk(req_id=1)
    assert decode_load_env_reply(
        msgspec.msgpack.encode(LoadEnvError(req_id=1, error="bad env"))
    ) == LoadEnvError(req_id=1, error="bad env")

    assert decode_run_env_reply(
        msgspec.msgpack.encode(RunEnvOk(req_id=3, env_run_id=1))
    ) == RunEnvOk(req_id=3, env_run_id=1)
    assert decode_run_env_reply(
        msgspec.msgpack.encode(RunEnvError(req_id=3, env_run_id=1, error="bad run"))
    ) == RunEnvError(req_id=3, env_run_id=1, error="bad run")

    with pytest.raises(ValueError, match="malformed load reply payload"):
        decode_load_env_reply(
            msgspec.msgpack.encode({"kind": "nope", "req_id": 1})
        )

    with pytest.raises(ValueError, match="malformed run reply payload"):
        decode_run_env_reply(
            msgspec.msgpack.encode({"kind": "nope", "req_id": 3, "env_run_id": 1})
        )


def test_decode_callback_request_and_replies_return_typed_structs() -> None:
    request = CallbackRequest(
        req_id=2,
        env_run_id=1,
        transition_id="finish",
        bundle=CallbackBundle(transition_id="finish", run_key="rk-1", ordinal=1),
        tokens=[peven.token({"seed": 1}, run_key="rk-1")],
        attempt=1,
        inputs_by_place={"ready": [peven.token({"seed": 1}, run_key="rk-1")]},
    )
    reply = make_callback_reply(
        req_id=2,
        env_run_id=1,
        outputs={"done": [peven.token({"ok": True}, run_key="rk-1")]},
    )
    error = make_callback_error(req_id=2, env_run_id=1, error="boom")

    assert decode_callback_request(msgspec.msgpack.encode(request)) == request
    assert decode_callback_reply(msgspec.msgpack.encode(reply)) == reply
    assert decode_callback_reply(msgspec.msgpack.encode(error)) == error
    assert bundle_ref_from_callback_bundle(request.bundle) == peven.BundleRef(
        transition_id="finish",
        run_key="rk-1",
        ordinal=1,
    )

    with pytest.raises(ValueError, match="even positive integer"):
        decode_callback_request(
            msgspec.msgpack.encode(
                CallbackRequest(
                    req_id=1,
                    env_run_id=1,
                    transition_id="finish",
                    bundle=CallbackBundle(
                        transition_id="finish",
                        run_key="rk-1",
                        ordinal=1,
                    ),
                    tokens=[peven.token({"seed": 1}, run_key="rk-1")],
                    attempt=1,
                    inputs_by_place={"ready": [peven.token({"seed": 1}, run_key="rk-1")]},
                )
            )
        )


def test_callback_message_validation_rejects_invalid_shapes() -> None:
    with pytest.raises(TypeError, match="callback error must be a non-empty string"):
        make_callback_error(req_id=2, env_run_id=1, error="")

    with pytest.raises(ValueError, match="malformed callback request payload"):
        decode_callback_request(
            msgspec.msgpack.encode(
                {
                    "kind": "nope",
                    "req_id": 2,
                    "env_run_id": 1,
                    "transition_id": "finish",
                    "bundle": {
                        "transition_id": "finish",
                        "run_key": "rk-1",
                        "ordinal": 1,
                    },
                    "tokens": [],
                    "attempt": 1,
                }
            )
        )

    with pytest.raises(TypeError, match="transition_id must be a non-empty string"):
        decode_callback_request(
            msgspec.msgpack.encode(
                CallbackRequest(
                    req_id=2,
                    env_run_id=1,
                    transition_id="",
                    bundle=CallbackBundle(
                        transition_id="",
                        run_key="rk-1",
                        ordinal=1,
                    ),
                    tokens=[],
                    attempt=1,
                )
            )
        )

    with pytest.raises(ValueError, match="attempt must be a positive integer"):
        decode_callback_request(
            msgspec.msgpack.encode(
                CallbackRequest(
                    req_id=2,
                    env_run_id=1,
                    transition_id="finish",
                    bundle=CallbackBundle(
                        transition_id="finish",
                        run_key="rk-1",
                        ordinal=1,
                    ),
                    tokens=[],
                    attempt=0,
                )
            )
        )

    with pytest.raises(TypeError, match="callback error must be a non-empty string"):
        decode_callback_reply(
            msgspec.msgpack.encode(
                CallbackError(req_id=2, env_run_id=1, error="")
            )
        )

    with pytest.raises(ValueError, match="malformed callback reply payload"):
        decode_callback_reply(
            msgspec.msgpack.encode({"kind": "nope", "req_id": 2, "env_run_id": 1})
        )

    with pytest.raises(TypeError, match="callback outputs place ids must be non-empty strings"):
        decode_callback_reply(
            msgspec.msgpack.encode(
                CallbackReply(req_id=2, env_run_id=1, outputs={"": []})
            )
        )

    with pytest.raises(TypeError, match="callback outputs buckets must be lists of Token values"):
        make_callback_reply(
            req_id=2,
            env_run_id=1,
            outputs={"done": peven.token({"ok": True}, run_key="rk-1")},  # type: ignore[dict-item]
        )

    with pytest.raises(TypeError, match="callback outputs buckets must be lists of Token values"):
        make_callback_reply(
            req_id=2,
            env_run_id=1,
            outputs={"done": ["bad"]},  # type: ignore[list-item]
        )


def test_decode_runtime_event_rejects_token_buckets_with_the_wrong_run_key() -> None:
    with pytest.raises(ValueError, match="preserve the bundle run_key"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                TransitionCompletedMessage(
                    env_run_id=7,
                    bundle=CallbackBundle(
                        transition_id="finish",
                        run_key="rk-1",
                        ordinal=1,
                    ),
                    firing_id=3,
                    attempt=1,
                    outputs={"done": [peven.token({"ok": True}, run_key="other")]},
                )
            )
        )

    with pytest.raises(ValueError, match="preserve the bundle run_key"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                TransitionStartedMessage(
                    env_run_id=7,
                    bundle=CallbackBundle(
                        transition_id="finish",
                        run_key="rk-1",
                        ordinal=1,
                    ),
                    firing_id=3,
                    attempt=1,
                    inputs=[peven.token({"ok": True}, run_key="rk-1")],
                    inputs_by_place={"ready": [peven.token({"ok": True}, run_key="other")]},
                )
            )
        )


def test_decode_runtime_event_preserves_transition_started_input_places() -> None:
    env_run_id, event = decode_runtime_event(
        msgspec.msgpack.encode(
            TransitionStartedMessage(
                env_run_id=7,
                bundle=CallbackBundle(
                    transition_id="finish",
                    run_key="rk-1",
                    ordinal=1,
                    selected_key="case-17",
                ),
                firing_id=3,
                attempt=2,
                inputs=[
                    peven.token({"side": "left"}, run_key="rk-1"),
                    peven.token({"side": "right"}, run_key="rk-1"),
                ],
                inputs_by_place={
                    "left": [peven.token({"side": "left"}, run_key="rk-1")],
                    "right": [peven.token({"side": "right"}, run_key="rk-1")],
                },
            )
        )
    )

    assert env_run_id == 7
    assert isinstance(event, peven.TransitionStarted)
    assert event.bundle == peven.BundleRef(
        transition_id="finish",
        run_key="rk-1",
        selected_key="case-17",
        ordinal=1,
    )
    assert event.inputs_by_place == {
        "left": [Token(run_key="rk-1", color="default", payload={"side": "left"})],
        "right": [Token(run_key="rk-1", color="default", payload={"side": "right"})],
    }


def test_decode_runtime_event_accepts_run_finished_payloads() -> None:
    env_run_id, event = decode_runtime_event(
        msgspec.msgpack.encode(
            RunFinishedMessage(
                env_run_id=11,
                result=RunResultMessage(
                    run_key="rk-1",
                    status="completed",
                    trace=[
                        TransitionResultMessage(
                            bundle=CallbackBundle(
                                transition_id="finish",
                                run_key="rk-1",
                                ordinal=1,
                            ),
                            firing_id=5,
                            status="completed",
                            outputs={
                                "done": [peven.token({"ok": True}, run_key="rk-1")]
                            },
                        )
                    ],
                    final_marking={
                        "done": [peven.token({"ok": True}, run_key="rk-1")]
                    },
                ),
            )
        )
    )

    assert env_run_id == 11
    assert isinstance(event, peven.RunFinished)
    assert event.result.run_key == "rk-1"
    assert event.result.final_marking == {
        "done": [Token(run_key="rk-1", color="default", payload={"ok": True})]
    }


def test_decode_runtime_event_rejects_invalid_failure_and_error_payloads() -> None:
    with pytest.raises(TypeError, match="transition_failed error must be a non-empty string"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                TransitionFailedMessage(
                    env_run_id=7,
                    bundle=CallbackBundle(
                        transition_id="finish",
                        run_key="rk-1",
                        ordinal=1,
                    ),
                    firing_id=3,
                    attempt=1,
                    error="",
                    retrying=False,
                )
            )
        )

    with pytest.raises(TypeError, match="guard_errored error must be a non-empty string"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                GuardErroredMessage(
                    env_run_id=7,
                    bundle=CallbackBundle(
                        transition_id="finish",
                        run_key="rk-1",
                        ordinal=1,
                    ),
                    error="",
                )
            )
        )

    with pytest.raises(TypeError, match="selection_errored transition_id must be a non-empty string"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                SelectionErroredMessage(
                    env_run_id=7,
                    transition_id="",
                    run_key="rk-1",
                    error="boom",
                )
            )
        )

    with pytest.raises(TypeError, match="selection_errored run_key must be a non-empty string"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                SelectionErroredMessage(
                    env_run_id=7,
                    transition_id="finish",
                    run_key="",
                    error="boom",
                )
            )
        )

    with pytest.raises(TypeError, match="selection_errored error must be a non-empty string"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                SelectionErroredMessage(
                    env_run_id=7,
                    transition_id="finish",
                    run_key="rk-1",
                    error="",
                )
            )
        )

    with pytest.raises(ValueError, match="malformed runtime event payload"):
        decode_runtime_event(msgspec.msgpack.encode({"kind": "nope", "env_run_id": 7}))


def test_decode_runtime_event_rejects_invalid_run_finished_shapes() -> None:
    with pytest.raises(TypeError, match="run_key must be a non-empty string"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                RunFinishedMessage(
                    env_run_id=11,
                    result=RunResultMessage(
                        run_key="",
                        status="completed",
                    ),
                )
            )
        )

    with pytest.raises(ValueError, match="unexpected run result status"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                RunFinishedMessage(
                    env_run_id=11,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="unknown",
                    ),
                )
            )
        )

    with pytest.raises(TypeError, match="result error must be a non-empty string or None"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                RunFinishedMessage(
                    env_run_id=11,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="failed",
                        error="",
                    ),
                )
            )
        )

    with pytest.raises(ValueError, match="unexpected terminal reason"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                RunFinishedMessage(
                    env_run_id=11,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="failed",
                        terminal_reason="unknown",
                    ),
                )
            )
        )

    with pytest.raises(ValueError, match="terminal_bundle must preserve the run_result run_key"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                RunFinishedMessage(
                    env_run_id=11,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="failed",
                        terminal_bundle=CallbackBundle(
                            transition_id="finish",
                            run_key="other",
                            ordinal=1,
                        ),
                    ),
                )
            )
        )

    with pytest.raises(ValueError, match="trace bundles must preserve the run_result run_key"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                RunFinishedMessage(
                    env_run_id=11,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="completed",
                        trace=[
                            TransitionResultMessage(
                                bundle=CallbackBundle(
                                    transition_id="finish",
                                    run_key="other",
                                    ordinal=1,
                                ),
                                firing_id=1,
                                status="completed",
                                outputs={},
                            )
                        ],
                    ),
                )
            )
        )

    with pytest.raises(ValueError, match="unexpected transition result status"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                RunFinishedMessage(
                    env_run_id=11,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="completed",
                        trace=[
                            TransitionResultMessage(
                                bundle=CallbackBundle(
                                    transition_id="finish",
                                    run_key="rk-1",
                                    ordinal=1,
                                ),
                                firing_id=1,
                                status="unknown",
                                outputs={},
                            )
                        ],
                    ),
                )
            )
        )

    with pytest.raises(TypeError, match="trace error must be a non-empty string or None"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                RunFinishedMessage(
                    env_run_id=11,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="completed",
                        trace=[
                            TransitionResultMessage(
                                bundle=CallbackBundle(
                                    transition_id="finish",
                                    run_key="rk-1",
                                    ordinal=1,
                                ),
                                firing_id=1,
                                status="failed",
                                outputs={},
                                error="",
                            )
                        ],
                    ),
                )
            )
        )

    with pytest.raises(ValueError, match="final_marking must preserve the bundle run_key"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                RunFinishedMessage(
                    env_run_id=11,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="completed",
                        final_marking={
                            "done": [peven.token({"ok": True}, run_key="other")]
                        },
                    ),
                )
            )
        )

    with pytest.raises(TypeError, match="terminal_transition must be a non-empty string or None"):
        decode_runtime_event(
            msgspec.msgpack.encode(
                RunFinishedMessage(
                    env_run_id=11,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="failed",
                        terminal_transition="",
                    ),
                )
            )
        )


def test_protocol_decode_helpers_reject_non_map_and_bad_scalar_shapes() -> None:
    with pytest.raises(ValueError, match="malformed load reply payload"):
        decode_load_env_reply(b"\xc1")

    with pytest.raises(ValueError, match="malformed load reply payload"):
        decode_load_env_reply(msgspec.msgpack.encode([]))

    with pytest.raises(ValueError, match="malformed load reply payload"):
        decode_load_env_reply(msgspec.msgpack.encode({1: "bad"}))

    with pytest.raises(ValueError, match="malformed load reply payload"):
        decode_load_env_reply(
            msgspec.msgpack.encode({"kind": "load_env_ok", "req_id": "1"})
        )

    with pytest.raises(ValueError, match="env_run_id must be a positive integer"):
        decode_run_env_reply(
            msgspec.msgpack.encode(
                {"kind": "run_env_ok", "req_id": 3, "env_run_id": 0}
            )
        )

    with pytest.raises(TypeError, match="load reply error must be a non-empty string"):
        decode_load_env_reply(
            msgspec.msgpack.encode(
                {"kind": "load_env_error", "req_id": 1, "error": ""}
            )
        )

    with pytest.raises(TypeError, match="transition_id must be a non-empty string"):
        bundle_ref_from_callback_bundle(
            CallbackBundle(transition_id="", run_key="rk-1", ordinal=1)
        )

    with pytest.raises(TypeError, match="run_key must be a non-empty string"):
        bundle_ref_from_callback_bundle(
            CallbackBundle(transition_id="finish", run_key="", ordinal=1)
        )

    with pytest.raises(ValueError, match="ordinal must be a positive integer"):
        bundle_ref_from_callback_bundle(
            CallbackBundle(transition_id="finish", run_key="rk-1", ordinal=0)
        )


def test_decode_callback_request_rejects_non_structured_selected_keys() -> None:
    with pytest.raises(TypeError, match="StructuredPayload unsupported type: bytes"):
        decode_callback_request(
            msgspec.msgpack.encode(
                CallbackRequest(
                    req_id=2,
                    env_run_id=1,
                    transition_id="finish",
                    bundle=CallbackBundle(
                        transition_id="finish",
                        run_key="rk-1",
                        ordinal=1,
                        selected_key=b"bad",
                    ),
                    tokens=[peven.token({"seed": 1}, run_key="rk-1")],
                    attempt=1,
                )
            )
        )

    with pytest.raises(ValueError, match=r"must match bundle\.transition_id"):
        decode_callback_request(
            msgspec.msgpack.encode(
                CallbackRequest(
                    req_id=2,
                    env_run_id=1,
                    transition_id="other",
                    bundle=CallbackBundle(
                        transition_id="finish",
                        run_key="rk-1",
                        ordinal=1,
                    ),
                    tokens=[peven.token({"seed": 1}, run_key="rk-1")],
                    attempt=1,
                )
            )
        )
