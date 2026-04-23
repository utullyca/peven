from __future__ import annotations

from types import SimpleNamespace

import pytest

import peven
from peven.authoring.executor import (
    ExecutorContext,
    get_executor,
    unregister_executor,
    validate_executor_signature,
)
from peven.authoring.ir import ExecutorSpec
from peven.shared.errors import PevenValidationError


def test_executor_decorator_rejects_invalid_name() -> None:
    with pytest.raises(ValueError, match="executor name must be a non-empty string"):
        peven.executor("")


def test_executor_signature_requires_ctx_and_positional_parameters() -> None:
    async def wrong_ctx(not_ctx):  # type: ignore[no-untyped-def]
        return None

    async def keyword_only(ctx, *, ready):  # type: ignore[no-untyped-def]
        return None

    with pytest.raises(PevenValidationError, match="start with a positional `ctx`"):
        validate_executor_signature(
            ExecutorSpec(name="wrong_ctx", fn=wrong_ctx),
            input_count=0,
            object_id="wrong_ctx",
        )

    with pytest.raises(PevenValidationError, match="must be positional"):
        validate_executor_signature(
            ExecutorSpec(name="keyword_only", fn=keyword_only),
            input_count=1,
            object_id="keyword_only",
        )


def test_executor_signature_accepts_exact_ctx_only_shape() -> None:
    async def ctx_only(ctx):  # type: ignore[no-untyped-def]
        return None

    validate_executor_signature(
        ExecutorSpec(name="ctx_only", fn=ctx_only),
        input_count=0,
        object_id="ctx_only",
    )


def test_executor_context_is_small_and_explicit() -> None:
    ctx = ExecutorContext(
        env=SimpleNamespace(name="env"),
        bundle=SimpleNamespace(run_key="rk"),
        executor_name="judge",
        attempt=2,
    )

    assert ctx.executor_name == "judge"
    assert ctx.attempt == 2


def test_executor_context_trace_writes_to_sink_when_present() -> None:
    events: list[object] = []
    sink = SimpleNamespace(write=events.append)
    ctx = ExecutorContext(
        env=SimpleNamespace(name="env"),
        bundle=SimpleNamespace(run_key="rk"),
        executor_name="judge",
        attempt=2,
        sink=sink,
    )

    ctx.trace({"kind": "agent_trace", "step": "tool"})

    assert events == [{"kind": "agent_trace", "step": "tool"}]


def test_unregister_executor_removes_registered_executor() -> None:
    namespace = {
        "__name__": "tests.authoring.generated_removable_executor",
        "peven": peven,
    }
    exec(
        """
@peven.executor("removable_executor")
async def removable_executor(ctx):
    return None
""",
        namespace,
    )

    assert get_executor("removable_executor") is not None
    unregister_executor("removable_executor")
    assert get_executor("removable_executor") is None


def test_executor_decorator_allows_reimport_of_the_same_top_level_executor() -> None:
    first_namespace = {
        "__name__": "tests.authoring.generated_reloadable_executor",
        "peven": peven,
    }
    second_namespace = dict(first_namespace)
    source = """
@peven.executor("reloadable_executor")
async def reloadable_executor(ctx):
    return None
"""

    exec(source, first_namespace)
    first_registered = get_executor("reloadable_executor")

    exec(source, second_namespace)
    second_registered = get_executor("reloadable_executor")

    assert first_registered is not None
    assert second_registered is not None
    assert second_registered.fn is second_namespace["reloadable_executor"]
    unregister_executor("reloadable_executor")
