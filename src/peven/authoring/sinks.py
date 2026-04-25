"""Authoring-side sink helpers."""

from __future__ import annotations

import datetime as _datetime
import json
import time
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TextIO

import msgspec

from ..shared.events import (
    BundleRef,
    GuardErrored,
    RunFinished,
    SelectionErrored,
    TransitionCompleted,
    TransitionFailed,
    TransitionStarted,
)
from ..shared.token import Token


try:
    from rich.table import Table as _RichTable
    from rich.text import Text as _RichText
except ImportError:  # pragma: no cover - rich is optional
    _RichTable = _RichText = None  # type: ignore[assignment]


__all__ = ["CompositeSink", "JSONLSink", "RichSink"]


_AGENT_PART_KINDS = frozenset({"part_start", "part_delta", "part_end"})
_TOOL_CALL_KINDS = frozenset({"function_tool_call", "builtin_tool_call"})
_TOOL_RESULT_KINDS = frozenset({"function_tool_result", "builtin_tool_result"})


class JSONLSink:
    """Append each received event as one JSON line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO | None = None

    def write(self, event: object) -> None:
        if self._handle is None:
            self._handle = self.path.open("a", encoding="utf-8")
        self._handle.write(json.dumps(_event_to_json_ready(event), sort_keys=True))
        self._handle.write("\n")
        self._handle.flush()

    def close(self, exc: BaseException | None) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class CompositeSink:
    """Fan out one sink lifecycle to multiple child sinks."""

    def __init__(self, *children: object) -> None:
        if not children:
            raise ValueError("CompositeSink requires at least one child sink")
        self.children = tuple(children)

    def write(self, record: object) -> None:
        for child in self.children:
            child.write(record)

    def close(self, exc: BaseException | None) -> None:
        first_error: BaseException | None = None
        for child in self.children:
            try:
                child.close(exc)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


class RichSink:
    """Render one live run trace to the terminal via Rich."""

    def __init__(
        self,
        *,
        console: object | None = None,
        show_agent_traces: bool = True,
        show_payloads: bool = False,
        payload_preview_chars: int = 120,
        show_final_marking: bool = True,
        debug: bool = False,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        if console is None:
            try:
                from rich.console import Console
            except ImportError as exc:  # pragma: no cover - depends on optional install path
                raise RuntimeError(
                    "RichSink requires `rich`; install `peven[rich]` or `pip install rich`"
                ) from exc
            console = Console()
        self._console = console
        self.show_agent_traces = show_agent_traces
        self.show_payloads = show_payloads
        self.payload_preview_chars = payload_preview_chars
        self.show_final_marking = show_final_marking
        self.debug = debug
        self._time_fn = time.perf_counter if time_fn is None else time_fn
        self._run_started_at: dict[str, float] = {}
        self._firing_started_at: dict[int, float] = {}

    def write(self, record: object) -> None:
        run_key = _record_run_key(record)
        if run_key is not None and run_key not in self._run_started_at:
            self._run_started_at[run_key] = self._time_fn()
            self._row("▸", "run", run_key, style="cyan")

        if isinstance(record, TransitionStarted):
            self._on_started(record)
        elif isinstance(record, TransitionCompleted):
            self._on_completed(record)
        elif isinstance(record, TransitionFailed):
            self._on_failed(record)
        elif isinstance(record, GuardErrored):
            self._row(
                "?",
                "guard",
                record.bundle.transition_id,
                meta=_format_bundle(record.bundle),
                style="yellow",
            )
            self._row("·", "error", record.error, style="yellow")
        elif isinstance(record, SelectionErrored):
            self._row("?", "select", record.transition_id, style="yellow")
            self._row("·", "error", record.error, style="yellow")
        elif isinstance(record, RunFinished):
            self._on_run_finished(record)
        elif isinstance(record, dict) and record.get("kind") == "agent_trace":
            self._on_agent_trace(record)

    def close(self, exc: BaseException | None) -> None:
        self._run_started_at.clear()
        self._firing_started_at.clear()
        if exc is not None:
            self._row("·", "error", f"sink closed with error: {exc}", style="red")

    def _on_started(self, record: TransitionStarted) -> None:
        self._firing_started_at[record.firing_id] = self._time_fn()
        flow = _format_flow_places(record.inputs_by_place)
        arrow = f"{flow} -> " if flow else "-> "
        meta = _join_meta(
            _format_bundle(record.bundle),
            f"attempt={record.attempt}" if self.debug or record.attempt > 1 else None,
            f"firing={record.firing_id}" if self.debug else None,
        )
        self._row("▸", "start", f"{arrow}[{record.bundle.transition_id}]", meta=meta)

    def _on_completed(self, record: TransitionCompleted) -> None:
        flow = _format_flow_places(record.outputs) or ""
        meta = _join_meta(
            self._pop_elapsed(record.firing_id),
            f"firing={record.firing_id}" if self.debug else None,
        )
        message = f"[{record.bundle.transition_id}] -> {flow}"
        self._row("✓", "ok", message, meta=meta, style="green")
        if self.show_payloads:
            self._emit_buckets("outputs", record.outputs)

    def _on_failed(self, record: TransitionFailed) -> None:
        meta = _join_meta(
            self._pop_elapsed(record.firing_id),
            f"firing={record.firing_id}" if self.debug else None,
            "retrying=yes" if record.retrying else None,
        )
        self._row("✗", "fail", record.bundle.transition_id, meta=meta, style="red")
        self._row("·", "error", record.error, style="red")

    def _on_run_finished(self, record: RunFinished) -> None:
        result = record.result
        now = self._time_fn()
        started = self._run_started_at.pop(result.run_key, now)
        style = "green" if result.status == "completed" else "red"
        glyph = "✓" if result.status == "completed" else "✗"
        meta = _join_meta(
            f"{now - started:.2f}s",
            f"terminal_reason={result.terminal_reason}"
            if result.status != "completed" and result.terminal_reason is not None
            else None,
        )
        self._row(glyph, "run", f"{result.run_key}  {result.status}", meta=meta, style=style)
        if self.show_final_marking and result.final_marking:
            marking = " ".join(
                f"{place}:{len(bucket)}" for place, bucket in result.final_marking.items()
            )
            self._row("·", "marking", marking, style="dim")
            if self.show_payloads:
                self._emit_buckets("result", result.final_marking)

    def _on_agent_trace(self, record: dict[str, object]) -> None:
        if not self.show_agent_traces:
            return
        event_kind = _agent_event_kind(record)
        if event_kind in _AGENT_PART_KINDS:
            return
        self._row(
            "▸",
            "agent",
            _format_agent_trace(record, event_kind, self.payload_preview_chars),
            style="magenta",
        )

    def _pop_elapsed(self, firing_id: int) -> str | None:
        started = self._firing_started_at.pop(firing_id, None)
        if started is None:
            return None
        return f"{self._time_fn() - started:.2f}s"

    def _emit_buckets(self, label: str, buckets: dict[str, list[object]]) -> None:
        for place, bucket in buckets.items():
            preview = _preview_bucket(bucket, self.payload_preview_chars)
            self._row("·", label, f"{place}: {preview}", style="dim")

    def _row(
        self,
        glyph: str,
        label: str,
        message: str,
        *,
        meta: str | None = None,
        style: str = "",
    ) -> None:
        table = _RichTable.grid(expand=False, padding=(0, 1))
        table.add_column(width=1, no_wrap=True)
        table.add_column(width=7, no_wrap=True)
        table.add_column(ratio=1)
        table.add_column()
        table.add_row(
            _RichText(glyph, style=style or "bold dim"),
            _RichText(label, style="bold dim"),
            _RichText(message, style=style),
            _RichText(meta or "", style="dim"),
        )
        self._console.print(table)


def _event_to_json_ready(value: object) -> object:
    if isinstance(value, (_datetime.datetime, _datetime.date, _datetime.time)):
        return value.isoformat()
    if is_dataclass(value):
        return {key: _event_to_json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, msgspec.Struct):
        return _event_to_json_ready(msgspec.to_builtins(value))
    if isinstance(value, tuple):
        return [_event_to_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _event_to_json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_event_to_json_ready(item) for item in value]
    return value


def _record_run_key(record: object) -> str | None:
    if isinstance(
        record, (TransitionStarted, TransitionCompleted, TransitionFailed, GuardErrored)
    ):
        return record.bundle.run_key
    if isinstance(record, SelectionErrored):
        return record.run_key
    if isinstance(record, RunFinished):
        return record.result.run_key
    if isinstance(record, dict):
        run_key = record.get("run_key")
        if isinstance(run_key, str) and run_key:
            return run_key
    return None


def _format_bundle(bundle: BundleRef) -> str | None:
    if bundle.selected_key is None and bundle.ordinal == 1:
        return None
    if bundle.selected_key is None:
        return f"#{bundle.ordinal}"
    selected = (
        bundle.selected_key if isinstance(bundle.selected_key, str) else repr(bundle.selected_key)
    )
    if bundle.ordinal == 1:
        return f"key={selected}"
    return f"key={selected}#{bundle.ordinal}"


def _format_flow_places(buckets: dict[str, list[object]]) -> str | None:
    if not buckets:
        return None
    return " + ".join(
        place if len(bucket) == 1 else f"{place}:{len(bucket)}"
        for place, bucket in buckets.items()
    )


def _truncate(rendered: str, max_chars: int) -> str:
    if len(rendered) <= max_chars:
        return rendered
    if max_chars <= 1:
        return rendered[:max_chars]
    return rendered[: max_chars - 1] + "…"


def _compact_dump(value: object, max_chars: int) -> str:
    ready = _event_to_json_ready(value)
    return _truncate(json.dumps(ready, sort_keys=True, separators=(", ", ": ")), max_chars)


def _payload_for_preview(value: object) -> object:
    if isinstance(value, Token):
        value = value.payload
    if isinstance(value, dict):
        return {
            key: item
            for key, item in value.items()
            if not (isinstance(key, str) and key.endswith("_latency_s"))
        }
    return value


def _preview_bucket(bucket: list[object], max_chars: int) -> str:
    if not bucket:
        return "empty"
    first = _compact_dump(_payload_for_preview(bucket[0]), max_chars)
    if len(bucket) == 1:
        return first
    return f"{first} (+{len(bucket) - 1} more)"


def _format_agent_trace(
    record: dict[str, object], event_kind: str | None, max_chars: int
) -> str:
    if not event_kind:
        return "agent_trace"
    is_call = event_kind in _TOOL_CALL_KINDS
    is_result = event_kind in _TOOL_RESULT_KINDS
    if is_call or is_result:
        event = record.get("event")
        parts = ["tool_call" if is_call else "tool_result"]
        tool_name = _tool_name_from_event(event)
        if tool_name:
            parts.append(f"name={tool_name}")
        if is_result:
            content = getattr(getattr(event, "result", None), "content", None)
            if content not in (None, ""):
                parts.append(f"result={_compact_dump(content, max_chars)}")
        return "  ".join(parts)
    model = record.get("model")
    if isinstance(model, str) and model:
        return f"{event_kind}  model={model}"
    return event_kind


def _tool_name_from_event(event: object) -> str | None:
    for attr in ("part", "result"):
        name = getattr(getattr(event, attr, None), "tool_name", None)
        if isinstance(name, str) and name:
            return name
    return None


def _agent_event_kind(record: dict[str, object]) -> str | None:
    kind = record.get("event_kind")
    if isinstance(kind, str) and kind:
        return kind
    kind = getattr(record.get("event"), "event_kind", None)
    return kind if isinstance(kind, str) and kind else None


def _join_meta(*parts: str | None) -> str | None:
    return "  ".join(part for part in parts if part) or None
