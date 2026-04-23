from __future__ import annotations

import io
import os

import pytest
from examples.trace import run_trace
from rich.console import Console

import peven

from .conftest import require_external_pevenpy_adapter_command


@pytest.mark.integration
@pytest.mark.slow
def test_trace_rich_sink_renders_a_full_e2e_transcript() -> None:
    if os.environ.get("PEVEN_RUN_OLLAMA_SMOKE") != "1":
        pytest.skip("set PEVEN_RUN_OLLAMA_SMOKE=1 to run the local Ollama smoke test")

    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=200)
    sink = peven.RichSink(
        console=console,
        show_payloads=True,
        payload_preview_chars=600,
        debug=True,
    )

    result = run_trace(
        command=require_external_pevenpy_adapter_command(),
        sink=sink,
    )

    text = buffer.getvalue()

    assert result.status == "completed"
    assert result.terminal_reason is None
    assert "run" in text
    assert "prompt -> [answer]" in text
    assert "[answer] -> draft" in text
    assert "draft -> [judge]" in text
    assert "[judge] -> done" in text
    assert "agent" in text
    assert "agent stream" not in text
    assert "part_delta" not in text
    assert any(kind in text for kind in ("tool_call", "tool_result", "final_result"))
    assert "model=qwen3.5:9b" in text
    assert "model=deepseek-r1:7b" in text
    assert "outputs" in text
    assert "result" in text
    assert "marking" in text
    assert "done:1" in text
    assert "completed" in text
