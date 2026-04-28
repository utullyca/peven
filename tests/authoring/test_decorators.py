from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import peven
from peven.shared.errors import PevenValidationError


def _register_executor(source: str) -> dict[str, object]:
    namespace = {
        "__name__": "tests.authoring.generated_executors",
        "peven": peven,
    }
    exec(source, namespace)
    return namespace


def test_env_decorator_finalizes_one_explicit_authoring_ir() -> None:
    _register_executor(
        """
@peven.executor("writer_finalize")
async def writer_finalize(ctx, ready):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key, color="scored")
"""
    )

    @peven.env("judge_net")
    class JudgeEnv(peven.Env):
        ready = peven.place(capacity=2, schema={"kind": "input"})
        scored = peven.place(terminal=True)

        judge = peven.transition(
            inputs=["ready"],
            outputs=[peven.output("scored")],
            executor="writer_finalize",
            guard=~peven.isempty(peven.f.items),
            retries=2,
        )

    spec = JudgeEnv.__peven_env_spec__

    assert spec.env_name == "judge_net"
    assert [place.id for place in spec.places] == ["ready", "scored"]
    assert spec.places[0].capacity == 2
    assert spec.places[0].schema == {"kind": "input"}
    assert spec.places[0].terminal is False
    assert spec.places[1].terminal is True
    assert spec.transitions[0].id == "judge"
    assert spec.transitions[0].executor == "writer_finalize"
    assert spec.transitions[0].inputs[0].place == "ready"
    assert spec.transitions[0].inputs[0].weight == 1
    assert spec.transitions[0].inputs[0].optional is False
    assert spec.transitions[0].outputs[0].place == "scored"
    assert spec.transitions[0].guard_spec == {
        "kind": "not",
        "child": {
            "kind": "call",
            "name": "isempty",
            "args": [{"kind": "field_ref", "path": ["items"]}],
        },
    }
    assert spec.transitions[0].retries == 2


def test_transition_declaration_order_is_stable() -> None:
    _register_executor(
        """
@peven.executor("ordered_writer_first")
async def ordered_writer_first(ctx, queued):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)

@peven.executor("ordered_writer_second")
async def ordered_writer_second(ctx, ready):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("ordered_net")
    class OrderedEnv(peven.Env):
        queued = peven.place()
        ready = peven.place()
        done = peven.place()

        first = peven.transition(
            inputs=["queued"],
            outputs=["ready"],
            executor="ordered_writer_first",
        )
        second = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="ordered_writer_second",
        )

    spec = OrderedEnv.__peven_env_spec__

    assert [place.id for place in spec.places] == ["queued", "ready", "done"]
    assert [transition.id for transition in spec.transitions] == ["first", "second"]


def test_executor_reuse_allows_different_input_place_order() -> None:
    _register_executor(
        """
@peven.executor("ordered_inputs_writer")
async def ordered_inputs_writer(ctx, left, right):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("input_order_contract_env")
    class InputOrderContractEnv(peven.Env):
        left = peven.place()
        right = peven.place()
        done = peven.place()

        first = peven.transition(
            inputs=["left", "right"],
            outputs=["done"],
            executor="ordered_inputs_writer",
        )
        second = peven.transition(
            inputs=["right", "left"],
            outputs=["done"],
            executor="ordered_inputs_writer",
        )

    spec = InputOrderContractEnv.spec()
    assert [transition.id for transition in spec.transitions] == ["first", "second"]


def test_executor_reuse_allows_multi_output_place_reordering() -> None:
    _register_executor(
        """
@peven.executor("unordered_outputs_writer")
async def unordered_outputs_writer(ctx, ready):
    return {
        "accepted": [peven.token({"ok": True}, run_key=ctx.bundle.run_key)],
        "rejected": [],
    }
"""
    )

    @peven.env("output_order_contract_env")
    class OutputOrderContractEnv(peven.Env):
        ready = peven.place()
        accepted = peven.place()
        rejected = peven.place()

        first = peven.transition(
            inputs=["ready"],
            outputs=["accepted", "rejected"],
            executor="unordered_outputs_writer",
        )
        second = peven.transition(
            inputs=["ready"],
            outputs=["rejected", "accepted"],
            executor="unordered_outputs_writer",
        )

    spec = OutputOrderContractEnv.spec()
    assert [transition.id for transition in spec.transitions] == ["first", "second"]


def test_executor_reuse_allows_different_single_output_places() -> None:
    _register_executor(
        """
@peven.executor("single_output_writer")
async def single_output_writer(ctx, ready):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("single_output_reuse_env")
    class SingleOutputReuseEnv(peven.Env):
        ready = peven.place()
        accepted = peven.place()
        rejected = peven.place()

        accept = peven.transition(
            inputs=["ready"],
            outputs=["accepted"],
            executor="single_output_writer",
        )
        reject = peven.transition(
            inputs=["ready"],
            outputs=["rejected"],
            executor="single_output_writer",
        )

    spec = SingleOutputReuseEnv.spec()
    assert [transition.id for transition in spec.transitions] == ["accept", "reject"]


def test_transition_requires_a_known_executor() -> None:
    with pytest.raises(PevenValidationError, match="unknown executor"):
        @peven.env("missing_executor")
        class MissingExecutorEnv(peven.Env):
            ready = peven.place()
            done = peven.place()

            finish = peven.transition(
                inputs=["ready"],
                outputs=["done"],
                executor="missing_executor",
            )


def test_transition_authoring_allows_place_references_to_reach_the_adapter_boundary() -> None:
    _register_executor(
        """
@peven.executor("known_place_writer")
async def known_place_writer(ctx, ready):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("missing_input_place")
    class MissingInputPlaceEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["missing"],
            outputs=["done"],
            executor="known_place_writer",
        )

    @peven.env("missing_output_place")
    class MissingOutputPlaceEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=["missing"],
            executor="known_place_writer",
        )

    assert MissingInputPlaceEnv.spec().transitions[0].inputs[0].place == "missing"
    assert MissingOutputPlaceEnv.spec().transitions[0].outputs[0].place == "missing"


def test_zero_input_transitions_are_supported_when_executor_is_ctx_only() -> None:
    _register_executor(
        """
@peven.executor("ctx_only_writer")
async def ctx_only_writer(ctx):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("zero_input")
    class ZeroInputEnv(peven.Env):
        done = peven.place()

        finish = peven.transition(
            inputs=[],
            outputs=["done"],
            executor="ctx_only_writer",
        )

    assert ZeroInputEnv.spec().transitions[0].inputs == ()


def test_zero_output_transitions_are_supported_during_authoring() -> None:
    _register_executor(
        """
@peven.executor("sink_writer")
async def sink_writer(ctx, ready):
    return None
"""
    )

    @peven.env("sink_env")
    class SinkEnv(peven.Env):
        ready = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=[],
            executor="sink_writer",
        )

    assert SinkEnv.spec().transitions[0].outputs == ()


def test_duplicate_authored_declarations_are_rejected() -> None:
    with pytest.raises(PevenValidationError, match="duplicate declaration ready"):
        @peven.env("duplicate_place")
        class DuplicatePlaceEnv(peven.Env):
            ready = peven.place()
            ready = peven.place()  # noqa: PIE794

    _register_executor(
        """
@peven.executor("duplicate_stub_writer")
async def duplicate_stub_writer(ctx, ready):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    with pytest.raises(PevenValidationError, match="duplicate declaration finish"):
        @peven.env("duplicate_transition")
        class DuplicateTransitionEnv(peven.Env):
            ready = peven.place()
            done = peven.place()

            finish = peven.transition(
                inputs=["ready"],
                outputs=["done"],
                executor="duplicate_stub_writer",
            )

            finish = peven.transition(  # noqa: PIE794
                inputs=["ready"],
                outputs=["done"],
                executor="duplicate_stub_writer",
            )


def test_authored_declarations_cannot_be_overwritten_by_plain_values() -> None:
    with pytest.raises(PevenValidationError, match="authored declaration ready was overwritten"):
        @peven.env("overwritten_place")
        class OverwrittenPlaceEnv(peven.Env):
            ready = peven.place()
            ready = 1  # noqa: PIE794

    _register_executor(
        """
@peven.executor("overwrite_writer")
async def overwrite_writer(ctx, ready):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    with pytest.raises(PevenValidationError, match="authored declaration finish was overwritten"):
        @peven.env("overwritten_transition")
        class OverwrittenTransitionEnv(peven.Env):
            ready = peven.place()
            done = peven.place()

            finish = peven.transition(
                inputs=["ready"],
                outputs=["done"],
                executor="overwrite_writer",
            )

            def finish():  # noqa: F811
                ...


def test_executor_signature_matches_normalized_input_arcs() -> None:
    _register_executor(
        """
@peven.executor("merge_writer")
async def merge_writer(ctx, left, right_pair):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("merge_net")
    class MergeEnv(peven.Env):
        left = peven.place()
        right = peven.place()
        done = peven.place()

        merge = peven.transition(
            inputs=[peven.input("left"), peven.input("right", weight=2)],
            outputs=["done"],
            executor="merge_writer",
        )

    spec = MergeEnv.__peven_env_spec__

    assert tuple(arc.weight for arc in spec.transitions[0].inputs) == (1, 2)
    assert tuple(arc.optional for arc in spec.transitions[0].inputs) == (False, False)

    _register_executor(
        """
@peven.executor("bad_merge_writer")
async def bad_merge_writer(ctx, left):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    with pytest.raises(PevenValidationError, match="executor signature does not match"):
        @peven.env("bad_merge_net")
        class BadMergeEnv(peven.Env):
            left = peven.place()
            right = peven.place()
            done = peven.place()

            merge = peven.transition(
                inputs=[peven.input("left"), peven.input("right", weight=2)],
                outputs=["done"],
                executor="bad_merge_writer",
            )


def test_env_ir_preserves_optional_input_arcs() -> None:
    _register_executor(
        """
@peven.executor("optional_plan_writer")
async def optional_plan_writer(ctx, ready, plan=None):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("optional_plan_net")
    class OptionalPlanEnv(peven.Env):
        ready = peven.place()
        plan = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["ready", peven.input("plan", optional=True)],
            outputs=["done"],
            executor="optional_plan_writer",
        )

    assert [(arc.place, arc.weight, arc.optional) for arc in OptionalPlanEnv.spec().transitions[0].inputs] == [
        ("ready", 1, False),
        ("plan", 1, True),
    ]


def test_executor_registration_is_top_level_and_unique() -> None:
    async def local_executor(ctx, ready):
        return peven.token({"ok": True}, run_key=ctx.bundle.run_key)

    with pytest.raises(PevenValidationError, match="top-level async functions"):
        peven.executor("local_only")(local_executor)

    _register_executor(
        """
@peven.executor("duplicate_executor")
async def duplicate_executor(ctx, ready):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    with pytest.raises(PevenValidationError, match="duplicate executor"):
        _register_executor(
            """
@peven.executor("duplicate_executor")
async def duplicate_executor_again(ctx, ready):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
        )


def test_env_inheritance_is_not_supported_in_v1() -> None:
    @peven.env("base_env")
    class BaseEnv(peven.Env):
        ready = peven.place()

    with pytest.raises(PevenValidationError, match="env inheritance is not supported"):
        @peven.env("child_env")
        class ChildEnv(BaseEnv):
            done = peven.place()


def test_output_weights_and_emit_many_are_not_public_authoring_surface() -> None:
    assert not hasattr(peven, "emit_many")

    with pytest.raises(TypeError):
        peven.output("done", weight=2)


def test_public_package_surface_exposes_the_implemented_subset() -> None:
    exports = set(peven.__all__)

    assert {
        "BundleRef",
        "Env",
        "Marking",
        "Token",
        "TransitionResult",
        "RunResult",
        "completed_firings",
        "env",
        "executor",
        "input",
        "install_runtime",
        "output",
        "place",
        "run_keys",
        "run_marking",
        "unregister_executor",
    }.issubset(exports)
    assert {"emit_many", "bootstrap", "_runtime"}.isdisjoint(exports)


def test_obsolete_private_authoring_modules_are_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("peven._authoring")


def test_package_import_origin_is_clean_src_tree() -> None:
    package_path = Path(peven.__file__).resolve()

    assert package_path.parts[-3:] == ("src", "peven", "__init__.py")
