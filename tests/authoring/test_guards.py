from __future__ import annotations

import pytest

import peven
from peven.authoring.guard import (
    And,
    Call,
    FieldRef,
    GuardNode,
    In,
    Literal,
    Not,
    Or,
    coerce_guard_node,
    literal,
    validate_guard_tree,
)
from peven.shared.errors import PevenValidationError


def test_guard_dsl_serializes_composite_specs_exactly() -> None:
    guard = (~(peven.f.x == 1)) & (peven.f.y > 0)

    assert guard.to_spec() == {
        "kind": "and",
        "children": [
            {
                "kind": "not",
                "child": {
                    "kind": "cmp",
                    "op": "==",
                    "left": {"kind": "field_ref", "path": ["x"]},
                    "right": {"kind": "literal", "value": 1},
                },
            },
            {
                "kind": "cmp",
                "op": ">",
                "left": {"kind": "field_ref", "path": ["y"]},
                "right": {"kind": "literal", "value": 0},
            },
        ],
    }


def test_transition_rejects_non_boolean_guard_root() -> None:
    namespace = {
        "__name__": "tests.authoring.generated_guard_executors",
        "peven": peven,
    }
    exec(
        """
@peven.executor("guard_writer")
async def guard_writer(ctx, ready):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
""",
        namespace,
    )

    with pytest.raises(PevenValidationError, match="guard root must produce a boolean"):
        @peven.env("bad_guard")
        class BadGuardEnv(peven.Env):
            ready = peven.place()
            done = peven.place()

            finish = peven.transition(
                inputs=["ready"],
                outputs=["done"],
                executor="guard_writer",
                guard=peven.f.value,
            )


def test_transition_allows_guard_on_multi_input_or_weighted_transition_to_reach_adapter_boundary() -> None:
    namespace = {
        "__name__": "tests.authoring.generated_weighted_guard_executors",
        "peven": peven,
    }
    exec(
        """
@peven.executor("weighted_guard_writer")
async def weighted_guard_writer(ctx, ready):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)

@peven.executor("multi_guard_writer")
async def multi_guard_writer(ctx, left, right):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
""",
        namespace,
    )

    @peven.env("weighted_guard")
    class WeightedGuardEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=[peven.input("ready", weight=2)],
            outputs=["done"],
            executor="weighted_guard_writer",
            guard=~peven.isempty(peven.f.items),
        )

    assert WeightedGuardEnv.spec().transitions[0].guard_spec == {
        "kind": "not",
        "child": {
            "kind": "call",
            "name": "isempty",
            "args": [{"kind": "field_ref", "path": ["items"]}],
        },
    }


def test_validate_guard_tree_rejects_boolean_producers_in_scalar_positions() -> None:
    with pytest.raises(ValueError, match="guard cmp children must be scalar-valued"):
        validate_guard_tree(peven.guard.Cmp("==", peven.isempty(peven.f.items), literal(True)))

    with pytest.raises(ValueError, match="guard in ref must be scalar-valued"):
        validate_guard_tree(peven.in_(peven.isempty(peven.f.items), [True]))


def test_guard_helpers_reject_invalid_inputs_and_boolean_misuse() -> None:
    with pytest.raises(TypeError, match="do not define truthiness"):
        bool(peven.f.value)

    with pytest.raises(TypeError, match="scalar structured values"):
        literal(object())

    with pytest.raises(TypeError, match=r"must be built from peven\.guard nodes"):
        coerce_guard_node(object())

    with pytest.raises(TypeError, match="flat list of scalar literals"):
        peven.in_(peven.f.value, "not-a-list")

    with pytest.raises(TypeError, match="do not support indexing"):
        peven.f["bad"]  # type: ignore[index]

    with pytest.raises(ValueError, match="identifiers"):
        peven.f.__getattr__("bad-name")


def test_validate_guard_tree_rejects_bad_internal_shapes() -> None:
    with pytest.raises(ValueError, match="unknown guard comparison op"):
        validate_guard_tree(peven.guard.Cmp("~=", peven.f.value, literal(1)))

    with pytest.raises(ValueError, match="expects exactly one argument"):
        validate_guard_tree(Call("isempty", (peven.f.value, peven.f.other)))

    with pytest.raises(TypeError, match="flat literal list"):
        validate_guard_tree(In(peven.f.value, (Literal(1), FieldRef(("x",)))))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="boolean combinators require at least one child"):
        validate_guard_tree(And(()))

    with pytest.raises(ValueError, match="boolean-producing children"):
        validate_guard_tree(Not(peven.f.value))

    class CustomGuard(GuardNode):
        @property
        def produces_bool(self) -> bool:
            return True

        def to_spec(self) -> dict[str, object]:
            return {"kind": "custom"}

    with pytest.raises(TypeError, match="unsupported guard node type"):
        validate_guard_tree(CustomGuard())


def test_guard_operator_helpers_and_fieldref_methods_are_explicit() -> None:
    or_guard = (peven.f.left != 1) | (peven.f.right <= 3)
    ge_guard = peven.f.score >= 2
    lt_guard = peven.f.rank < 5
    nested = peven.f.payload.__getattr__("case_id")

    assert isinstance(or_guard, Or)
    assert ge_guard.to_spec()["op"] == ">="
    assert lt_guard.to_spec()["op"] == "<"
    assert nested.to_spec() == {"kind": "field_ref", "path": ["payload", "case_id"]}
    assert peven.isnothing(peven.f.value).to_spec()["name"] == "isnothing"
    assert peven.length(peven.f.items).to_spec()["name"] == "length"
    assert peven.in_(peven.f.kind, ["a"]).to_spec()["kind"] == "in"
    assert literal(Literal(1)).value == 1

    with pytest.raises(TypeError, match="do not support indexing"):
        peven.f.value["bad"]  # type: ignore[index]

    base = GuardNode()
    with pytest.raises(NotImplementedError):
        _ = base.produces_bool
    with pytest.raises(NotImplementedError):
        base.to_spec()

    @peven.env("multi_guard")
    class MultiGuardEnv(peven.Env):
        left = peven.place()
        right = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["left", "right"],
            outputs=["done"],
            executor="multi_guard_writer",
            guard=~peven.isempty(peven.f.items),
        )

    assert MultiGuardEnv.spec().transitions[0].guard_spec == {
        "kind": "not",
        "child": {
            "kind": "call",
            "name": "isempty",
            "args": [{"kind": "field_ref", "path": ["items"]}],
        },
    }
