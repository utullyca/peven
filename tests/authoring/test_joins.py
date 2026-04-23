from __future__ import annotations

import pytest

import peven
from peven.authoring.join import (
    JoinLiteral,
    JoinNode,
    JoinTuple,
    PayloadRef,
    join_literal,
    validate_join_tree,
)
from peven.shared.errors import PevenValidationError


def test_join_dsl_serializes_composite_specs_exactly() -> None:
    selector = peven.join_key(peven.place_id, peven.payload.case_id, "judge")

    assert selector.to_spec() == {
        "kind": "tuple",
        "items": [
            {"kind": "place_id"},
            {"kind": "payload_ref", "path": ["case_id"]},
            {"kind": "literal", "value": "judge"},
        ],
    }


def test_transition_rejects_non_join_node_join_by() -> None:
    namespace = {
        "__name__": "tests.authoring.generated_join_executors",
        "peven": peven,
    }
    exec(
        """
@peven.executor("join_writer")
async def join_writer(ctx, left, right):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
""",
        namespace,
    )

    with pytest.raises(PevenValidationError, match=r"join_by must be a peven\.join selector"):
        @peven.env("bad_join")
        class BadJoinEnv(peven.Env):
            left = peven.place()
            right = peven.place()
            done = peven.place()

            finish = peven.transition(
                inputs=["left", "right"],
                outputs=["done"],
                executor="join_writer",
                join_by=object(),
            )


def test_join_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="requires at least one selector part"):
        peven.join_key()

    with pytest.raises(TypeError, match="scalar structured values"):
        join_literal(object())

    with pytest.raises(TypeError, match="do not support indexing"):
        peven.payload["bad"]  # type: ignore[index]

    with pytest.raises(ValueError, match="identifiers"):
        peven.payload.__getattr__("bad-name")


def test_validate_join_tree_rejects_bad_internal_shapes() -> None:
    with pytest.raises(TypeError, match="scalar structured values"):
        join_literal(JoinTuple((peven.place_id,)))

    with pytest.raises(ValueError, match="require at least one item"):
        validate_join_tree(JoinTuple(()))

    class CustomJoin(JoinNode):
        def to_spec(self) -> dict[str, object]:
            return {"kind": "custom"}

    with pytest.raises(TypeError, match="unsupported join_by node type"):
        validate_join_tree(CustomJoin())

    with pytest.raises(ValueError, match="identifier"):
        validate_join_tree(PayloadRef(("",)))


def test_join_helpers_cover_single_part_and_payloadref_methods() -> None:
    nested = peven.payload.case_id.__getattr__("value")
    assert nested.to_spec() == {"kind": "payload_ref", "path": ["case_id", "value"]}
    assert peven.join_key(peven.payload.case_id).to_spec() == {
        "kind": "payload_ref",
        "path": ["case_id"],
    }
    literal_node = JoinLiteral("x")
    assert join_literal(literal_node) is literal_node
    assert peven.place_id.to_spec() == {"kind": "place_id"}

    with pytest.raises(TypeError, match="do not support indexing"):
        peven.payload.case_id["bad"]  # type: ignore[index]

    base = JoinNode()
    with pytest.raises(NotImplementedError):
        base.to_spec()
