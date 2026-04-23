from __future__ import annotations

import io
import json

import pytest
from pydantic_ai.messages import FinalResultEvent
from rich.console import Console

import peven
import peven.authoring as authoring
from peven.runtime import store as runtime_store


def test_store_api_requires_an_active_run_scope() -> None:
    with pytest.raises(RuntimeError, match=r"active Env\.run"):
        peven.store.put(object())


def test_authoring_namespace_exports_the_public_sink_helpers() -> None:
    exports = set(authoring.__all__)

    assert {"CompositeSink", "JSONLSink", "RichSink"}.issubset(exports)
    assert authoring.CompositeSink is peven.CompositeSink
    assert authoring.JSONLSink is peven.JSONLSink
    assert authoring.RichSink is peven.RichSink


def test_jsonl_sink_writes_one_event_per_line(tmp_path) -> None:
    sink = peven.JSONLSink(tmp_path / "events.jsonl")
    bundle = peven.events.BundleRef(transition_id="judge", run_key="rk-1", ordinal=1)
    event = peven.events.TransitionStarted(
        bundle=bundle,
        firing_id=11,
        attempt=1,
        inputs=[peven.token({"kind": "prompt"}, run_key="rk-1", color="prompt")],
    )

    sink.write(event)

    payload = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())

    assert payload["kind"] == "transition_started"
    assert payload["bundle"]["transition_id"] == "judge"
    assert payload["inputs"][0]["payload"] == {"kind": "prompt"}


def test_jsonl_sink_accepts_generic_trace_records(tmp_path) -> None:
    sink = peven.JSONLSink(tmp_path / "events.jsonl")

    sink.write(
        {
            "kind": "agent_trace",
            "transition_id": "answer",
            "attempt": 1,
            "event": {"phase": "tool_call", "tool": "search"},
        }
    )

    payload = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())

    assert payload == {
        "attempt": 1,
        "event": {"phase": "tool_call", "tool": "search"},
        "kind": "agent_trace",
        "transition_id": "answer",
    }


def test_jsonl_sink_serializes_pydantic_ai_stream_events(tmp_path) -> None:
    sink = peven.JSONLSink(tmp_path / "events.jsonl")

    sink.write(
        {
            "kind": "agent_trace",
            "event": FinalResultEvent(tool_name=None, tool_call_id=None),
        }
    )

    payload = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())

    assert payload == {
        "event": {
            "event_kind": "final_result",
            "tool_call_id": None,
            "tool_name": None,
        },
        "kind": "agent_trace",
    }


def test_composite_sink_fans_out_writes_and_close() -> None:
    left: list[object] = []
    right: list[object] = []
    closed: list[tuple[str, BaseException | None]] = []

    class _Sink:
        def __init__(self, name: str, bucket: list[object]) -> None:
            self.name = name
            self.bucket = bucket

        def write(self, event: object) -> None:
            self.bucket.append(event)

        def close(self, exc: BaseException | None) -> None:
            closed.append((self.name, exc))

    sink = peven.CompositeSink(_Sink("left", left), _Sink("right", right))

    sink.write({"kind": "agent_trace", "step": "tool"})
    sink.close(None)

    assert left == [{"kind": "agent_trace", "step": "tool"}]
    assert right == [{"kind": "agent_trace", "step": "tool"}]
    assert closed == [("left", None), ("right", None)]


def test_rich_sink_renders_bundle_and_hides_completed_terminal_reason() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=120)
    time_values = iter([0.0, 0.02, 0.60, 1.84])
    sink = peven.RichSink(console=console, time_fn=lambda: next(time_values))
    bundle = peven.events.BundleRef(
        transition_id="judge",
        run_key="rk-1",
        selected_key=("case-17", "rubric"),
        ordinal=2,
    )

    sink.write(
        peven.events.TransitionStarted(
            bundle=bundle,
            firing_id=11,
            attempt=1,
            inputs=[peven.token({"kind": "prompt"}, run_key="rk-1")],
            inputs_by_place={"prompt": [peven.token({"kind": "prompt"}, run_key="rk-1")]},
        )
    )
    sink.write(
        peven.events.TransitionCompleted(
            bundle=bundle,
            firing_id=11,
            attempt=1,
            outputs={"report": [peven.token({"ok": True}, run_key="rk-1")]},
        )
    )
    sink.write(
        peven.events.RunFinished(
            peven.RunResult(
                run_key="rk-1",
                status="completed",
                terminal_reason="no_enabled_transition",
                final_marking={"report": [peven.token({"ok": True}, run_key="rk-1")]},
            )
        )
    )

    text = buffer.getvalue()

    assert "run" in text
    assert "rk-1" in text
    assert "start" in text
    assert "prompt -> [judge]" in text
    assert "key=('case-17', 'rubric')#2" in text
    assert "bundle[" not in text
    assert "status=running" not in text
    assert "ok" in text
    assert "[judge] -> report" in text
    assert "0.58s" in text
    assert "dt=" not in text
    assert "completed" in text
    assert "1.84s" in text
    assert "wall=" not in text
    assert "terminal_reason=no_enabled_transition" not in text
    assert "marking" in text
    assert "report:1" in text


def test_rich_sink_shows_failure_reason_and_filters_stream_parts_by_default() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=120)
    sink = peven.RichSink(console=console)

    sink.write(
        {
            "kind": "agent_trace",
            "transition_id": "answer",
            "run_key": "rk-2",
            "event_kind": "part_start",
            "event": FinalResultEvent(tool_name=None, tool_call_id=None),
        }
    )
    sink.write(
        {
            "kind": "agent_trace",
            "transition_id": "judge",
            "run_key": "rk-2",
            "event_kind": "final_result",
            "event": FinalResultEvent(tool_name=None, tool_call_id=None),
        }
    )
    sink.write(
        peven.events.RunFinished(
            peven.RunResult(
                run_key="rk-2",
                status="failed",
                terminal_reason="executor_failed",
            )
        )
    )

    text = buffer.getvalue()

    assert "part_start" not in text
    assert "agent" in text
    assert "final_result" in text
    assert "model=" not in text
    assert "rk-2" in text
    assert "status=running" not in text
    assert "failed" in text
    assert "terminal_reason=executor_failed" in text


def test_rich_sink_shows_payload_previews_and_debug_metadata() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=160)
    time_values = iter([0.0, 0.05, 0.85, 1.25])
    sink = peven.RichSink(
        console=console,
        show_payloads=True,
        payload_preview_chars=400,
        debug=True,
        time_fn=lambda: next(time_values),
    )
    bundle = peven.events.BundleRef(transition_id="judge", run_key="rk-debug", ordinal=1)
    draft = peven.token(
        {"draft_answer": "mars", "verdict": "correct"},
        run_key="rk-debug",
        color="draft",
    )

    sink.write(
        peven.events.TransitionStarted(
            bundle=bundle,
            firing_id=17,
            attempt=1,
            inputs=[draft],
            inputs_by_place={"draft": [draft]},
        )
    )
    sink.write(
        peven.events.TransitionCompleted(
            bundle=bundle,
            firing_id=17,
            attempt=1,
            outputs={"done": [draft]},
        )
    )
    sink.write(
        peven.events.RunFinished(
            peven.RunResult(
                run_key="rk-debug",
                status="completed",
                final_marking={"done": [draft]},
            )
        )
    )

    text = buffer.getvalue()

    assert "draft -> [judge]" in text
    assert "outputs" in text
    assert "attempt=1" in text
    assert "firing=17" in text
    assert "inputs=1" not in text
    assert '"draft_answer": "mars"' in text
    assert '"verdict": "correct"' in text
    assert "result" in text
    assert "completed" in text
    assert "rk-debug" in text
    assert "1.25s" in text
    assert "wall=" not in text


def test_rich_sink_clears_run_timing_state_after_finish_and_close() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=120)
    time_values = iter([0.0, 1.5])
    sink = peven.RichSink(console=console, time_fn=lambda: next(time_values))

    sink.write(
        peven.events.RunFinished(
            peven.RunResult(
                run_key="rk-3",
                status="completed",
                final_marking={},
            )
        )
    )

    assert sink._run_started_at == {}

    sink._run_started_at["stale"] = 99.0
    sink.close(None)

    assert sink._run_started_at == {}
    assert sink._firing_started_at == {}


def test_rich_sink_evicts_run_timing_state_on_finish_and_close() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=120)
    time_values = iter([0.0, 1.25, 2.50, 2.75])
    sink = peven.RichSink(console=console, time_fn=lambda: next(time_values))

    sink.write(
        peven.events.RunFinished(
            peven.RunResult(
                run_key="rk-3",
                status="completed",
                final_marking={},
            )
        )
    )

    assert sink._run_started_at == {}

    sink.write(
        peven.events.TransitionStarted(
            bundle=peven.events.BundleRef(
                transition_id="judge",
                run_key="rk-4",
                ordinal=1,
            ),
            firing_id=99,
            attempt=1,
            inputs=[],
        )
    )
    assert sink._run_started_at == {"rk-4": 2.5}

    sink.close(None)

    assert sink._run_started_at == {}
    assert sink._firing_started_at == {}


def test_store_lifecycle_helpers_round_trip_values() -> None:
    store = runtime_store.open_store(11)
    token = runtime_store.activate_store(store)
    try:
        ref = peven.store.put({"ok": True})
        assert peven.store.get(ref) == {"ok": True}
        peven.store.release(ref)
        ref2 = peven.store.put({"other": True})
        runtime_store.clear_store(store)
        assert store.refs == {}
        peven.store.release(ref2)
    finally:
        runtime_store.reset_store(token)
