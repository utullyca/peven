from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import peven.runtime as runtime_pkg
import peven.runtime.bridge as bridge_module


EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
INTEGRATIONS_DIR = Path(__file__).resolve().parents[2] / "src" / "peven" / "integrations"


def test_examples_directory_is_a_namespace_package() -> None:
    sys.modules.pop("examples", None)
    sys.modules.pop("examples.guarded_batch", None)
    sys.modules.pop("examples.keyed_join", None)
    sys.modules.pop("examples.trace", None)

    module = importlib.import_module("examples")

    assert not (EXAMPLES_DIR / "__init__.py").exists()
    assert "examples.guarded_batch" not in sys.modules
    assert "examples.keyed_join" not in sys.modules
    assert "examples.trace" not in sys.modules
    assert getattr(module, "__all__", []) == []


def test_examples_directory_contains_only_the_curated_examples() -> None:
    assert {path.name for path in EXAMPLES_DIR.glob("*.py")} == {
        "guarded_batch.py",
        "keyed_join.py",
        "trace.py",
    }


def test_trace_example_can_be_reimported_after_module_eviction() -> None:
    first = importlib.import_module("examples.trace")
    sys.modules.pop("examples.trace", None)
    second = importlib.import_module("examples.trace")

    assert first is not second
    assert second.run_trace is not None


def test_examples_do_not_import_private_runtime_bridge_runner() -> None:
    for path in EXAMPLES_DIR.glob("*.py"):
        if path.name in {"__init__.py", "_runtime.py"}:
            continue
        source = path.read_text(encoding="utf-8")
        assert "_run_until_terminal_result" not in source
        assert "import peven.runtime.bridge as bridge_module" not in source


def test_small_examples_expose_runnable_module_entrypoints() -> None:
    for module_name in ("examples.guarded_batch", "examples.keyed_join"):
        module = importlib.import_module(module_name)

        assert callable(module.main)


def test_runtime_bridge_private_runner_alias_is_removed() -> None:
    assert not hasattr(bridge_module, "_run_until_terminal_result")
    assert not hasattr(bridge_module, "dispatch_transition_callback")


def test_runtime_package_no_longer_re_exports_bridge_runners() -> None:
    assert not hasattr(runtime_pkg, "run_env")
    assert not hasattr(runtime_pkg, "run_until_terminal_result")
    assert hasattr(runtime_pkg, "store")


def test_guarded_batch_example_uses_a_supported_guard_shape() -> None:
    from examples.guarded_batch import GuardedBatchEnv

    batch = next(
        transition
        for transition in GuardedBatchEnv.spec().transitions
        if transition.id == "batch"
    )

    if batch.guard_spec is not None:
        assert len(batch.inputs) == 1
        assert batch.inputs[0].weight == 1


def test_rich_reasoning_wrapper_example_is_removed() -> None:
    assert not (EXAMPLES_DIR / "rich_reasoning.py").exists()


def test_smoke_example_is_moved_into_tests() -> None:
    assert not (EXAMPLES_DIR / "smoke.py").exists()


def test_trace_examples_are_merged_into_one_fuse_aware_example() -> None:
    assert not (EXAMPLES_DIR / "mock_tool_trace.py").exists()

    module = importlib.import_module("examples.trace")
    signature = inspect.signature(module.run_trace)

    assert "fuse" in signature.parameters


def test_integrations_package_is_a_namespace_without_a_reexport_shim() -> None:
    module = importlib.import_module("peven.integrations")

    assert not (INTEGRATIONS_DIR / "__init__.py").exists()
    assert getattr(module, "__all__", []) == []


def test_pydantic_ai_integration_module_stays_directly_importable() -> None:
    module = importlib.import_module("peven.integrations.pydantic_ai")

    assert module.event_stream_handler is not None
