from __future__ import annotations

from types import SimpleNamespace

import msgspec
import pytest

import peven
import peven.handoff.lowering as lowering_module
from peven.handoff.lowering import (
    AUTHORED_ENV_SCHEMA_VERSION,
    EnvSpecMessage,
    InputArcSpecMessage,
    OutputArcSpecMessage,
    PlaceSpecMessage,
    TransitionSpecMessage,
    package_env_spec,
)


namespace = {
    "__name__": "tests.handoff.generated_executors",
    "peven": peven,
}
exec(
    """
@peven.executor("handoff_writer")
async def handoff_writer(ctx, prompt):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)

@peven.executor("handoff_joiner")
async def handoff_joiner(ctx, left, right):
    return {"done": peven.token({"ok": True}, run_key=ctx.bundle.run_key)}

@peven.executor("handoff_optional")
async def handoff_optional(ctx, ready, plan=None):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
""",
    namespace,
)


@peven.env("handoff_demo")
class HandoffDemo(peven.Env):
    prompt = peven.place(capacity=2, schema={"kind": "prompt"})
    ready = peven.place()
    done = peven.place(terminal=True)

    write = peven.transition(
        inputs=[peven.input("prompt", weight=2)],
        outputs=[peven.output("ready")],
        executor="handoff_writer",
        retries=2,
    )
    join = peven.transition(
        inputs=["ready", "prompt"],
        outputs=["done"],
        executor="handoff_joiner",
        join_by=peven.join_key(peven.place_id, peven.payload.case_id),
    )


@peven.env("guarded_handoff")
class GuardedHandoff(peven.Env):
    ready = peven.place()
    done = peven.place()

    finish = peven.transition(
        inputs=["ready"],
        outputs=["done"],
        executor="handoff_writer",
        guard=(peven.f.score > 0) & ~peven.isempty(peven.f.items),
    )


@peven.env("optional_handoff")
class OptionalHandoff(peven.Env):
    ready = peven.place()
    plan = peven.place()
    done = peven.place()

    finish = peven.transition(
        inputs=["ready", peven.input("plan", optional=True)],
        outputs=["done"],
        executor="handoff_optional",
    )


def test_package_env_spec_produces_authored_ir_payload() -> None:
    lowered = package_env_spec(HandoffDemo.spec())

    assert HandoffDemo.spec().places[2].terminal is True
    assert lowered == EnvSpecMessage(
        schema_version=AUTHORED_ENV_SCHEMA_VERSION,
        env_name="handoff_demo",
        places=[
            PlaceSpecMessage(id="prompt", capacity=2, schema={"kind": "prompt"}),
            PlaceSpecMessage(id="ready", capacity=None, schema=None),
            PlaceSpecMessage(id="done", capacity=None, schema=None),
        ],
        transitions=[
            TransitionSpecMessage(
                id="write",
                executor="handoff_writer",
                inputs=[InputArcSpecMessage(place="prompt", weight=2, optional=False)],
                outputs=[OutputArcSpecMessage(place="ready")],
                guard_spec=None,
                retries=2,
                join_by_spec=None,
            ),
            TransitionSpecMessage(
                id="join",
                executor="handoff_joiner",
                inputs=[
                    InputArcSpecMessage(place="ready", weight=1),
                    InputArcSpecMessage(place="prompt", weight=1),
                ],
                outputs=[OutputArcSpecMessage(place="done")],
                guard_spec=None,
                retries=0,
                join_by_spec={
                    "kind": "tuple",
                    "items": [
                        {"kind": "place_id"},
                        {"kind": "payload_ref", "path": ["case_id"]},
                    ],
                },
            ),
        ],
    )


def test_package_env_spec_preserves_optional_input_arcs() -> None:
    lowered = package_env_spec(OptionalHandoff.spec())

    assert lowered.transitions[0].inputs == [
        InputArcSpecMessage(place="ready", weight=1, optional=False),
        InputArcSpecMessage(place="plan", weight=1, optional=True),
    ]

    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(lowered),
        type=EnvSpecMessage,
    )
    assert decoded.transitions[0].inputs[1].optional is True


def test_package_env_spec_passes_guard_specs_through_unchanged() -> None:
    lowered = package_env_spec(GuardedHandoff.spec())

    assert lowered.transitions[0].guard_spec == {
        "kind": "and",
        "children": [
            {
                "kind": "cmp",
                "op": ">",
                "left": {"kind": "field_ref", "path": ["score"]},
                "right": {"kind": "literal", "value": 0},
            },
            {
                "kind": "not",
                "child": {
                    "kind": "call",
                    "name": "isempty",
                    "args": [{"kind": "field_ref", "path": ["items"]}],
                },
            },
        ],
    }


def test_packaged_authored_env_is_immediately_transportable() -> None:
    lowered = package_env_spec(HandoffDemo.spec())

    payload = msgspec.msgpack.encode(lowered)

    assert isinstance(payload, bytes)
    assert payload


def test_authored_ir_shape_validators_reject_bad_transport_shapes() -> None:
    with pytest.raises(ValueError, match="schema_version must be a positive integer"):
        lowering_module._validate_authored_env_message(
            SimpleNamespace(
                schema_version=0,
                env_name="env",
                places=[],
                transitions=[],
            )
        )

    with pytest.raises(TypeError, match="env_name must be a non-empty string"):
        lowering_module._validate_authored_env_message(
            SimpleNamespace(
                schema_version=1,
                env_name="",
                places=[],
                transitions=[],
            )
        )

    with pytest.raises(TypeError, match="places must be a list"):
        lowering_module._validate_authored_env_message(
            SimpleNamespace(
                schema_version=1,
                env_name="env",
                places=(),
                transitions=[],
            )
        )

    with pytest.raises(TypeError, match="transitions must be a list"):
        lowering_module._validate_authored_env_message(
            SimpleNamespace(
                schema_version=1,
                env_name="env",
                places=[],
                transitions=(),
            )
        )

    with pytest.raises(TypeError, match="place id must be a non-empty string"):
        lowering_module._validate_place_spec_message(
            SimpleNamespace(id="", capacity=None, schema=None)
        )

    with pytest.raises(ValueError, match="place capacity must be a positive int or None"):
        lowering_module._validate_place_spec_message(
            SimpleNamespace(id="ready", capacity=0, schema=None)
        )

    with pytest.raises(TypeError, match="transition id must be a non-empty string"):
        lowering_module._validate_transition_spec_message(
            SimpleNamespace(
                id="",
                executor="writer",
                inputs=[],
                outputs=[],
                retries=0,
                guard_spec=None,
                join_by_spec=None,
            )
        )

    with pytest.raises(TypeError, match="transition executor must be a non-empty string"):
        lowering_module._validate_transition_spec_message(
            SimpleNamespace(
                id="finish",
                executor="",
                inputs=[],
                outputs=[],
                retries=0,
                guard_spec=None,
                join_by_spec=None,
            )
        )

    with pytest.raises(TypeError, match="transition inputs must be a list"):
        lowering_module._validate_transition_spec_message(
            SimpleNamespace(
                id="finish",
                executor="writer",
                inputs=(),
                outputs=[],
                retries=0,
                guard_spec=None,
                join_by_spec=None,
            )
        )

    with pytest.raises(TypeError, match="transition outputs must be a list"):
        lowering_module._validate_transition_spec_message(
            SimpleNamespace(
                id="finish",
                executor="writer",
                inputs=[],
                outputs=(),
                retries=0,
                guard_spec=None,
                join_by_spec=None,
            )
        )

    with pytest.raises(ValueError, match="transition retries must be a non-negative int"):
        lowering_module._validate_transition_spec_message(
            SimpleNamespace(
                id="finish",
                executor="writer",
                inputs=[],
                outputs=[],
                retries=-1,
                guard_spec=None,
                join_by_spec=None,
            )
        )

    with pytest.raises(TypeError, match="input arc place must be a non-empty string"):
        lowering_module._validate_transition_spec_message(
            SimpleNamespace(
                id="finish",
                executor="writer",
                inputs=[SimpleNamespace(place="", weight=1)],
                outputs=[],
                retries=0,
                guard_spec=None,
                join_by_spec=None,
            )
        )

    with pytest.raises(ValueError, match="input arc weight must be a positive int"):
        lowering_module._validate_transition_spec_message(
            SimpleNamespace(
                id="finish",
                executor="writer",
                inputs=[SimpleNamespace(place="ready", weight=0)],
                outputs=[],
                retries=0,
                guard_spec=None,
                join_by_spec=None,
            )
        )

    with pytest.raises(TypeError, match="input arc optional must be a bool"):
        lowering_module._validate_transition_spec_message(
            SimpleNamespace(
                id="finish",
                executor="writer",
                inputs=[SimpleNamespace(place="ready", weight=1, optional="yes")],
                outputs=[],
                retries=0,
                guard_spec=None,
                join_by_spec=None,
            )
        )

    with pytest.raises(TypeError, match="output arc place must be a non-empty string"):
        lowering_module._validate_transition_spec_message(
            SimpleNamespace(
                id="finish",
                executor="writer",
                inputs=[],
                outputs=[SimpleNamespace(place="")],
                retries=0,
                guard_spec=None,
                join_by_spec=None,
            )
        )
