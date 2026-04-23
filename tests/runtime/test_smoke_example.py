from __future__ import annotations

import os
import time

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.ollama import OllamaProvider

import peven
from peven.integrations.pydantic_ai import event_stream_handler

from .conftest import require_external_pevenpy_adapter_command


TASKS = (
    {"branch": "left", "question": "What planet is known as the red planet?", "expected": "mars"},
    {
        "branch": "right",
        "question": "What gas do humans need to breathe to survive?",
        "expected": "oxygen",
    },
)

_AGENT = Agent(
    OpenAIChatModel(
        "qwen3.5:0.8b",
        provider=OllamaProvider(base_url="http://127.0.0.1:11434/v1"),
    ),
    output_type=NativeOutput(str),
    model_settings={"temperature": 0.6, "max_tokens": 256, "seed": 1},
)


@peven.executor("test_smoke_answer")
async def smoke_answer_executor(ctx, prompt):
    task = prompt.payload
    started = time.perf_counter()
    result = await _AGENT.run(
        f"Answer with one short lowercase noun.\nQuestion: {task['question']}",
        event_stream_handler=event_stream_handler(ctx, model="qwen3.5:0.8b"),
    )
    text = result.output.strip().lower()
    return ctx.token(
        {
            **task,
            "answer": text,
            "exact_match": text == task["expected"],
            "latency_s": time.perf_counter() - started,
        }
    )


@peven.executor("test_smoke_collect")
async def smoke_collect_executor(ctx, done):
    return ctx.token({token.payload["branch"]: token.payload for token in done})


@peven.env("test_smoke_ollama_env")
class TestSmokeOllamaEnv(peven.Env):
    prompt = peven.place(schema={"kind": "question"})
    done = peven.place()
    report = peven.place()

    def initial_marking(self) -> peven.Marking:
        return peven.marking(prompt=TASKS)

    answer = peven.transition(
        inputs=["prompt"],
        outputs=[peven.output("done")],
        executor="test_smoke_answer",
    )
    collect = peven.transition(
        inputs=[peven.input("done", weight=len(TASKS))],
        outputs=[peven.output("report")],
        executor="test_smoke_collect",
    )


def _run_smoke_ollama_e2e(*, command: tuple[str, ...]) -> dict[str, dict]:
    return TestSmokeOllamaEnv().run(command=command).final_marking["report"][0].payload


@pytest.mark.integration
@pytest.mark.slow
def test_smoke_example_runs_end_to_end_with_local_ollama() -> None:
    if os.environ.get("PEVEN_RUN_OLLAMA_SMOKE") != "1":
        pytest.skip("set PEVEN_RUN_OLLAMA_SMOKE=1 to run the local Ollama smoke test")
    report = _run_smoke_ollama_e2e(command=require_external_pevenpy_adapter_command())

    assert set(report) == {"left", "right"}
    assert report["left"]["answer"] == "mars"
    assert report["right"]["answer"] == "oxygen"
    assert report["left"]["exact_match"] is True
    assert report["right"]["exact_match"] is True
    assert report["left"]["latency_s"] > 0
    assert report["right"]["latency_s"] > 0
