"""Branch each task to two models, then keyed-join their answers by case_id.

Run with:

    uv run python -m examples.keyed_join

The output is one JSON object keyed by ``case_id``. Each value contains the
same question answered by the fast and full branches.
"""

import json
import time

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.ollama import OllamaProvider

import peven


TASKS = (
    {
        "case_id": "planet",
        "question": "What planet is known as the red planet?",
    },
    {
        "case_id": "gas",
        "question": "What gas do humans need to breathe to survive?",
    },
)

FAST_MODEL = "qwen3.5:0.8b"
FULL_MODEL = "qwen3.5:9b"
_PROVIDER = OllamaProvider(base_url="http://127.0.0.1:11434/v1")
fast_agent = Agent(
    OpenAIChatModel(FAST_MODEL, provider=_PROVIDER),
    output_type=NativeOutput(str),
    model_settings={"temperature": 0.2, "max_tokens": 128, "seed": 1},
)
full_agent = Agent(
    OpenAIChatModel(FULL_MODEL, provider=_PROVIDER),
    output_type=NativeOutput(str),
    model_settings={"temperature": 0.4, "max_tokens": 128, "seed": 1},
)


@peven.executor("tee_prompt")
async def tee_prompt(ctx, prompt):
    payload = prompt.payload
    return {
        "fast_prompt": ctx.token(payload),
        "full_prompt": ctx.token(payload),
    }


@peven.executor("answer_fast")
async def answer_fast(ctx, fast_prompt):
    task = fast_prompt.payload
    started = time.perf_counter()
    result = await fast_agent.run(
        f"Answer with one short lowercase noun.\nQuestion: {task['question']}"
    )
    return ctx.token(
        {
            "case_id": task["case_id"],
            "question": task["question"],
            "answer": result.output.strip().lower(),
            "model": FAST_MODEL,
            "latency_s": time.perf_counter() - started,
        }
    )


@peven.executor("answer_full")
async def answer_full(ctx, full_prompt):
    task = full_prompt.payload
    started = time.perf_counter()
    result = await full_agent.run(
        f"Answer in one short lowercase sentence.\nQuestion: {task['question']}"
    )
    return ctx.token(
        {
            "case_id": task["case_id"],
            "question": task["question"],
            "answer": result.output.strip().lower().rstrip("."),
            "model": FULL_MODEL,
            "latency_s": time.perf_counter() - started,
        }
    )


@peven.executor("merge_answers")
async def merge_answers(ctx, fast_answer, full_answer):
    fast = fast_answer.payload
    full = full_answer.payload
    return ctx.token(
        {
            "case_id": fast["case_id"],
            "question": fast["question"],
            "fast": {"model": fast["model"], "answer": fast["answer"]},
            "full": {"model": full["model"], "answer": full["answer"]},
        }
    )


@peven.env("keyed_join")
class KeyedJoinEnv(peven.Env):
    prompt = peven.place(schema={"kind": "question"})
    fast_prompt = peven.place()
    full_prompt = peven.place()
    fast_answer = peven.place()
    full_answer = peven.place()
    merged = peven.place(terminal=True)

    def initial_marking(self, seed: int | None = None) -> peven.Marking:
        del seed
        return peven.marking(prompt=TASKS)

    tee = peven.transition(
        inputs=["prompt"],
        outputs=[peven.output("fast_prompt"), peven.output("full_prompt")],
        executor="tee_prompt",
    )
    answer_fast = peven.transition(
        inputs=["fast_prompt"],
        outputs=[peven.output("fast_answer")],
        executor="answer_fast",
    )
    answer_full = peven.transition(
        inputs=["full_prompt"],
        outputs=[peven.output("full_answer")],
        executor="answer_full",
    )
    merge = peven.transition(
        inputs=["fast_answer", "full_answer"],
        outputs=[peven.output("merged")],
        executor="merge_answers",
        join_by=peven.join_key(peven.payload.case_id),
    )


def run_keyed_join(*, command: tuple[str, ...] | None = None) -> dict[str, dict]:
    result = KeyedJoinEnv().run(command=command)
    return {token.payload["case_id"]: token.payload for token in result.final_marking["merged"]}


def main() -> None:
    report = run_keyed_join()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
