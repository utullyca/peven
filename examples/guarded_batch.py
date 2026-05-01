"""Guard and retry one batched model call.

Run with:

    uv run python -m examples.guarded_batch

This example keeps the topology intentionally small: a non-empty batch guard
decides whether the transition can fire, and ``retries=1`` retries the first
executor failure.
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
        "slot": "planet",
        "question": "What planet is known as the red planet?",
    },
    {
        "slot": "gas",
        "question": "What gas do humans need to breathe to survive?",
    },
)
TASK_BATCH = {"items": list(TASKS)}

agent = Agent(
    OpenAIChatModel(
        "qwen3.5:9b",
        provider=OllamaProvider(base_url="http://127.0.0.1:11434/v1"),
    ),
    output_type=NativeOutput(str),
    model_settings={"temperature": 0.3, "max_tokens": 128, "seed": 1},
)


@peven.executor("batch_answer")
async def batch_answer(ctx, prompt):
    tasks = prompt.payload["items"]
    if ctx.attempt == 1:
        ctx.trace(
            {
                "kind": "retry_demo",
                "transition_id": ctx.bundle.transition_id,
                "run_key": ctx.bundle.run_key,
                "attempt": ctx.attempt,
            }
        )
        raise RuntimeError("retry once to demonstrate transition.retries")

    started = time.perf_counter()
    result = await agent.run(
        "Answer each question with one short lowercase noun separated by `;` in input order.\n"
        + "\n".join(
            f"{index}. {task['question']}" for index, task in enumerate(tasks, start=1)
        )
    )
    pieces = [
        piece.strip(" .").lower()
        for piece in result.output.replace("\n", ";").split(";")
        if piece.strip()
    ]
    answers = {
        task["slot"]: (pieces[index] if index < len(pieces) else "")
        for index, task in enumerate(tasks)
    }
    return ctx.token(
        {
            "model": "qwen3.5:9b",
            "attempt": ctx.attempt,
            "answers": answers,
            "latency_s": time.perf_counter() - started,
        }
    )


@peven.env("guarded_batch")
class GuardedBatchEnv(peven.Env):
    prompt = peven.place(
        capacity=1,
        schema={"kind": "question_batch", "size": len(TASKS)},
    )
    report = peven.place(terminal=True)

    def initial_marking(self, seed: int | None = None) -> peven.Marking:
        del seed
        return peven.marking(prompt=[TASK_BATCH])

    batch = peven.transition(
        inputs=["prompt"],
        outputs=[peven.output("report")],
        executor="batch_answer",
        guard=~peven.isempty(peven.f.items),
        retries=1,
    )


def run_guarded_batch(*, command: tuple[str, ...] | None = None) -> dict[str, object]:
    return GuardedBatchEnv().run(command=command).final_marking["report"][0].payload


def main() -> None:
    report = run_guarded_batch()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
