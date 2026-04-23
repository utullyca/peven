from __future__ import annotations

import math

import msgspec
import pytest

import peven
from peven.handoff import messages as handoff_messages
from peven.handoff.lowering import package_env_spec
from peven.shared.token import validate_structured_payload


def test_structured_payload_roundtrip_and_error_paths() -> None:
    value = {"ok": [1, True, None, {"score": 1.5}]}

    validate_structured_payload(value)
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(value)) == value

    with pytest.raises(OverflowError, match="64-bit"):
        validate_structured_payload(2**80)

    with pytest.raises(ValueError, match="finite"):
        validate_structured_payload(math.inf)

    with pytest.raises(TypeError, match="string keys"):
        validate_structured_payload({1: "bad"})

    with pytest.raises(TypeError, match="unsupported type"):
        validate_structured_payload(object())


def test_messages_module_no_longer_exports_structured_payload_helpers() -> None:
    assert not hasattr(handoff_messages, "encode_structured_payload")
    assert not hasattr(handoff_messages, "decode_structured_payload")


def test_package_env_spec_keeps_authoring_payload_transportable() -> None:
    namespace = {
        "__name__": "tests.handoff.generated_payload_handoff_executors",
        "peven": peven,
    }
    exec(
        """
@peven.executor("payload_writer")
async def payload_writer(ctx, prompt):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
""",
        namespace,
    )

    @peven.env("payload_handoff")
    class PayloadHandoff(peven.Env):
        prompt = peven.place(capacity=1, schema={"kind": "prompt"})
        done = peven.place()

        finish = peven.transition(
            inputs=["prompt"],
            outputs=["done"],
            executor="payload_writer",
            guard=(peven.f.score > 0) & ~peven.isempty(peven.f.items),
        )

    packaged = package_env_spec(PayloadHandoff.spec())
    encoded = msgspec.msgpack.encode(packaged)
    decoded = msgspec.msgpack.decode(encoded, type=dict[str, object])

    assert isinstance(packaged.schema_version, int)
    assert decoded["env_name"] == "payload_handoff"
    assert isinstance(decoded["places"], list)
    assert isinstance(decoded["transitions"], list)


def test_package_env_spec_rejects_non_structured_schema_payloads() -> None:
    def define_bad_schema_payload() -> None:
        @peven.env("bad_schema_payload")
        class BadSchemaPayload(peven.Env):
            prompt = peven.place(schema=object())

        package_env_spec(BadSchemaPayload.spec())

    with pytest.raises(TypeError, match="unsupported type"):
        define_bad_schema_payload()
