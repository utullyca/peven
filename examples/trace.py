"""Tool and reasoning traces with an optional fuse budget.

Run with ``fuse=1`` to stop after the first traced transition and inspect
``RunResult.status`` / ``RunResult.terminal_reason``.
"""

import time

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.ollama import OllamaProvider

import peven
from peven.integrations.pydantic_ai import event_stream_handler


TASKS = (
    {
        "question": "What planet is known as the red planet?",
    },
)

TOOL_MODEL = "qwen3.5:9b"
REASON_MODEL = "deepseek-r1:7b"
_PROVIDER = OllamaProvider(base_url="http://127.0.0.1:11434/v1")

tool_agent = Agent(
    OpenAIChatModel(
        TOOL_MODEL,
        provider=_PROVIDER,
    ),
    output_type=NativeOutput(str),
    instructions=(
        "Always call the `lookup_fact` tool exactly once before answering. "
        "Return one short lowercase noun."
    ),
    model_settings={"temperature": 0.1, "max_tokens": 128, "seed": 1},
)
reason_agent = Agent(
    OpenAIChatModel(
        REASON_MODEL,
        provider=_PROVIDER,
    ),
    output_type=NativeOutput(str),
    model_settings={"temperature": 0.2, "max_tokens": 256, "seed": 1},
)


@tool_agent.tool_plain
def lookup_fact(question: str) -> str:
    question = question.strip().lower()
    if "red planet" in question:
        return "mars"
    if "breathe" in question or "survive" in question:
        return "oxygen"
    return "unknown"


@peven.executor("tool_answer")
async def tool_answer(ctx, prompt):
    task = prompt.payload
    started = time.perf_counter()
    result = await tool_agent.run(
        task["question"],
        event_stream_handler=event_stream_handler(ctx, model=TOOL_MODEL),
    )
    return ctx.token(
        {
            "question": task["question"],
            "draft_answer": result.output.strip().lower().rstrip("."),
            "tool_model": TOOL_MODEL,
            "tool_latency_s": time.perf_counter() - started,
        }
    )


@peven.executor("judge_answer")
async def judge_answer(ctx, draft):
    payload = draft.payload
    started = time.perf_counter()
    result = await reason_agent.run(
        "Think carefully, then answer with exactly `correct` or `incorrect`.\n"
        f"Question: {payload['question']}\n"
        f"Draft answer: {payload['draft_answer']}",
        event_stream_handler=event_stream_handler(ctx, model=REASON_MODEL),
    )
    return ctx.token(
        {
            **payload,
            "reason_model": REASON_MODEL,
            "verdict": result.output.strip().lower(),
            "reason_latency_s": time.perf_counter() - started,
        }
    )


@peven.env("trace")
class TraceEnv(peven.Env):
    prompt = peven.place(schema={"kind": "trace_prompt"})
    draft = peven.place()
    done = peven.place()

    def initial_marking(self) -> peven.Marking:
        return peven.marking(prompt=TASKS)

    answer = peven.transition(
        inputs=["prompt"],
        outputs=[peven.output("draft")],
        executor="tool_answer",
    )
    judge = peven.transition(
        inputs=["draft"],
        outputs=[peven.output("done")],
        executor="judge_answer",
    )


def run_trace(
    *,
    command: tuple[str, ...],
    sink: object | None = None,
    fuse: int | None = None,
) -> peven.RunResult:
    return TraceEnv().run(command=command, sink=sink, fuse=fuse)
