from __future__ import annotations

import pytest

import peven
import peven.runtime.bridge as bridge_module
from peven.handoff.lowering import compile_env
from peven.handoff.messages import (
    CallbackBundle,
    CallbackError,
    CallbackReply,
    CallbackRequest,
    decode_callback_reply,
)
from peven.runtime.store import activate_store, clear_store, open_store, reset_store

from .conftest import make_transition_callback


def _register_executor(source: str) -> dict[str, object]:
    namespace = {
        "__name__": "tests.runtime.generated_bridge_callback_executors",
        "peven": peven,
    }
    exec(source, namespace)
    return namespace


@pytest.mark.asyncio
async def test_reply_to_callback_request_returns_a_canonical_callback_reply() -> None:
    _register_executor(
        """
@peven.executor("bridge_callback_weighted")
async def bridge_callback_weighted(ctx, left, rights):
    return peven.token(
        {
            "run_key": ctx.bundle.run_key,
            "left": left.payload,
            "right_count": len(rights),
            "right_payloads": [token.payload for token in rights],
            "input_places": sorted(ctx.inputs_by_place),
        },
        run_key=ctx.bundle.run_key,
    )
"""
    )

    @peven.env("bridge_callback_reply_env")
    class BridgeCallbackReplyEnv(peven.Env):
        left = peven.place()
        right = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["left", peven.input("right", weight=2)],
            outputs=["done"],
            executor="bridge_callback_weighted",
        )

    env = BridgeCallbackReplyEnv()
    compiled = compile_env(BridgeCallbackReplyEnv.spec())
    callback = make_transition_callback(compiled, env)
    request = CallbackRequest(
        req_id=2,
        env_run_id=7,
        transition_id="finish",
        bundle=CallbackBundle(transition_id="finish", run_key="rk-1", ordinal=1),
        tokens=[
            peven.token({"side": "left"}, run_key="rk-1"),
            peven.token({"side": "right-1"}, run_key="rk-1"),
            peven.token({"side": "right-2"}, run_key="rk-1"),
        ],
        attempt=1,
        inputs_by_place={
            "left": [peven.token({"side": "left"}, run_key="rk-1")],
            "right": [
                peven.token({"side": "right-1"}, run_key="rk-1"),
                peven.token({"side": "right-2"}, run_key="rk-1"),
            ],
        },
    )

    payload, reply = await bridge_module._reply_to_callback_request(
        callback=callback,
        active_env_run_id=7,
        request=request,
    )
    assert decode_callback_reply(payload) == reply

    assert isinstance(reply, CallbackReply)
    assert reply.req_id == 2
    assert reply.env_run_id == 7
    assert reply.outputs == {
        "done": [
            peven.Token(
                run_key="rk-1",
                color="default",
                payload={
                    "run_key": "rk-1",
                    "left": {"side": "left"},
                    "right_count": 2,
                    "right_payloads": [{"side": "right-1"}, {"side": "right-2"}],
                    "input_places": ["left", "right"],
                },
            )
        ]
    }


@pytest.mark.asyncio
async def test_reply_to_callback_request_returns_callback_error_on_callback_failure() -> None:
    _register_executor(
        """
@peven.executor("bridge_callback_failure")
async def bridge_callback_failure(ctx):
    raise RuntimeError("callback boom")
"""
    )

    @peven.env("bridge_callback_error_env")
    class BridgeCallbackErrorEnv(peven.Env):
        done = peven.place()

        finish = peven.transition(
            inputs=[],
            outputs=["done"],
            executor="bridge_callback_failure",
        )

    env = BridgeCallbackErrorEnv()
    compiled = compile_env(BridgeCallbackErrorEnv.spec())
    callback = make_transition_callback(compiled, env)
    request = CallbackRequest(
        req_id=2,
        env_run_id=3,
        transition_id="finish",
        bundle=CallbackBundle(transition_id="finish", run_key="rk-1", ordinal=1),
        tokens=[],
        attempt=1,
    )

    payload, reply = await bridge_module._reply_to_callback_request(
        callback=callback,
        active_env_run_id=3,
        request=request,
    )
    assert decode_callback_reply(payload) == reply

    assert reply == CallbackError(req_id=2, env_run_id=3, error="callback boom")


@pytest.mark.asyncio
async def test_reply_to_callback_request_returns_callback_error_for_unknown_transition() -> None:
    @peven.env("bridge_callback_unknown_transition_env")
    class BridgeCallbackUnknownTransitionEnv(peven.Env):
        done = peven.place()

    env = BridgeCallbackUnknownTransitionEnv()
    compiled = compile_env(BridgeCallbackUnknownTransitionEnv.spec())
    callback = make_transition_callback(compiled, env)
    request = CallbackRequest(
        req_id=2,
        env_run_id=13,
        transition_id="missing",
        bundle=CallbackBundle(transition_id="missing", run_key="rk-1", ordinal=1),
        tokens=[],
        attempt=1,
    )

    _, reply = await bridge_module._reply_to_callback_request(
        callback=callback,
        active_env_run_id=13,
        request=request,
    )

    assert reply == CallbackError(
        req_id=2,
        env_run_id=13,
        error="unknown compiled transition missing",
    )


@pytest.mark.asyncio
async def test_reply_to_callback_request_rejects_the_wrong_env_run() -> None:
    _register_executor(
        """
@peven.executor("bridge_callback_env_run")
async def bridge_callback_env_run(ctx):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("bridge_callback_env_run_env")
    class BridgeCallbackEnvRunEnv(peven.Env):
        done = peven.place()

        finish = peven.transition(
            inputs=[],
            outputs=["done"],
            executor="bridge_callback_env_run",
        )

    env = BridgeCallbackEnvRunEnv()
    compiled = compile_env(BridgeCallbackEnvRunEnv.spec())
    callback = make_transition_callback(compiled, env)
    request = CallbackRequest(
        req_id=2,
        env_run_id=4,
        transition_id="finish",
        bundle=CallbackBundle(transition_id="finish", run_key="rk-1", ordinal=1),
        tokens=[],
        attempt=1,
    )

    with pytest.raises(
        bridge_module.AdapterProtocolError,
        match="env_run_id did not match the active run",
    ):
        await bridge_module._reply_to_callback_request(
            callback=callback,
            active_env_run_id=5,
            request=request,
        )


@pytest.mark.asyncio
async def test_reply_to_callback_request_uses_the_existing_run_store_across_multiple_requests() -> None:
    _register_executor(
        """
@peven.executor("bridge_callback_store_writer")
async def bridge_callback_store_writer(ctx, ready):
    ref = peven.store.put({"seed": ready.payload["seed"]})
    return peven.token({"ref": ref}, run_key=ctx.bundle.run_key)

@peven.executor("bridge_callback_store_reader")
async def bridge_callback_store_reader(ctx, stored):
    value = peven.store.get(stored.payload["ref"])
    return peven.token({"payload": value}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("bridge_callback_store_env")
    class BridgeCallbackStoreEnv(peven.Env):
        ready = peven.place()
        stored = peven.place()
        done = peven.place()

        write = peven.transition(
            inputs=["ready"],
            outputs=["stored"],
            executor="bridge_callback_store_writer",
        )
        read = peven.transition(
            inputs=["stored"],
            outputs=["done"],
            executor="bridge_callback_store_reader",
        )

    env = BridgeCallbackStoreEnv()
    compiled = compile_env(BridgeCallbackStoreEnv.spec())
    callback = make_transition_callback(compiled, env)
    store = open_store(9)
    store_token = activate_store(store)

    try:
        stored_payload, stored_reply = await bridge_module._reply_to_callback_request(
            callback=callback,
            active_env_run_id=9,
            request=CallbackRequest(
                req_id=2,
                env_run_id=9,
                transition_id="write",
                bundle=CallbackBundle(transition_id="write", run_key="rk-1", ordinal=1),
                tokens=[peven.token({"seed": 9}, run_key="rk-1")],
                attempt=1,
            ),
        )
        assert decode_callback_reply(stored_payload) == stored_reply
        assert isinstance(stored_reply, CallbackReply)

        done_payload, done_reply = await bridge_module._reply_to_callback_request(
            callback=callback,
            active_env_run_id=9,
            request=CallbackRequest(
                req_id=4,
                env_run_id=9,
                transition_id="read",
                bundle=CallbackBundle(transition_id="read", run_key="rk-1", ordinal=2),
                tokens=stored_reply.outputs["stored"],
                attempt=1,
            ),
        )
        assert decode_callback_reply(done_payload) == done_reply
        assert isinstance(done_reply, CallbackReply)
    finally:
        clear_store(store)
        reset_store(store_token)

    assert done_reply.outputs == {
        "done": [
            peven.Token(
                run_key="rk-1",
                color="default",
                payload={"payload": {"seed": 9}},
            )
        ]
    }
