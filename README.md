# Peven

Peven is Python authoring for structured LLM environments backed by a Julia Petri-net runtime.

If PydanticAI makes it easy to build agents, Peven makes it easy to build the environment around them: places, transitions, joins, guards, retries, and the topology you want to evaluate.

## Why use it

- Author environments in Python, next to the agents and tools you already write.
- Make topology explicit instead of hiding it inside one giant agent loop.
- Run the hard state-machine part on a Julia engine built for Petri nets and concurrent firing.
- Compare workflows: single-shot, judge loops, keyed joins, guarded retries, branch-and-merge topologies.

## Install

First install the Python package:

```bash
uv add peven
```

or

```bash
pip install peven
```

Peven also needs a Julia runtime. `peven.install_runtime()` provisions that layer through `juliapkg`, including:

- Julia itself if it is not already available
- [`PevenPy.jl`](https://github.com/jammy-eggs/PevenPy.jl)
- [`Peven.jl`](https://github.com/jammy-eggs/Peven.jl)

Recommended: do that immediately after install so the one-time Julia download, package resolution, and precompile work does not happen on your first real run.

```bash
uv run peven-install
```

or

```bash
uv run peven install-runtime
```

If you skip that step, the first `Env.run()` will do the same provisioning automatically.


## Quickstart

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.ollama import OllamaProvider

import peven


agent = Agent(
    OpenAIChatModel(
        "qwen3.5:9b",
        provider=OllamaProvider(base_url="http://127.0.0.1:11434/v1"),
    ),
    output_type=NativeOutput(str),
)


@peven.executor("answer")
async def answer(ctx, prompt):
    question = prompt.payload["question"]
    result = await agent.run(question)
    return ctx.token(
        {
            "question": question,
            "answer": result.output.strip().lower(),
        }
    )


@peven.env("single_question")
class SingleQuestionEnv(peven.Env):
    prompt = peven.place(schema={"kind": "question"})
    report = peven.place(terminal=True)

    def initial_marking(self) -> peven.Marking:
        return peven.marking(
            prompt=[{"question": "What planet is known as the red planet?"}]
        )

    solve = peven.transition(
        inputs=["prompt"],
        outputs=["report"],
        executor="answer",
    )


result = SingleQuestionEnv().run()
print(result.status)
print(result.final_marking["report"][0].payload)
```

That same pattern scales to richer topologies:

- tee one prompt into multiple branches
- join outputs back together by key
- gate transitions with guards
- retry transitions without rewriting control flow
- cap the run with `fuse`

Mark a final place with `terminal=True` when a token there means the run is
complete. The Julia engine still reports the real Petri-net condition; Python
normalizes `no_enabled_transition` into a completed `RunResult` only when a
terminal place contains a token.

Use `peven.input("place", optional=True)` when a transition should fire from
its required inputs and receive `None` when that optional place is absent. This
maps directly to `ArcFrom(...; optional=true)` in `Peven.jl`. Optional arcs must
not be used on keyed joins, and a transition still needs at least one required
input.

## Why Julia

The Julia side is not there for novelty. It keeps the engine closer to the real Petri-net model.

Python is a great place to author agents and executors, but it pushes engine code toward shims, wrappers, and dynamic glue. Julia is a better fit for the symbolic runtime: markings, firing rules, joins, guards, retries, and termination stay explicit instead of dissolving into spaghetti soup.

## Architecture

Peven has three layers:

- `peven` — Python authoring, executors, integrations, sinks, and runtime ownership.
- [`PevenPy.jl`](https://github.com/jammy-eggs/PevenPy.jl) — the narrow Julia adapter boundary.
- [`Peven.jl`](https://github.com/jammy-eggs/Peven.jl) — the execution engine.

Technically, Python authors the env and owns transition callbacks. `PevenPy.jl` lowers the authored env into Julia runtime structures, runs the net, and streams runtime events back. `Peven.jl` owns the actual Petri-net execution semantics.

## Examples

The repo examples are intentionally small but representative:

- `examples/trace.py` — PydanticAI trace integration, `fuse`, and rich run output
- `examples/guarded_batch.py` — guarded retries around a batch step
- `examples/keyed_join.py` — branch, answer in parallel, and keyed-join the results
- `examples/minigrid/` — MiniGrid DoorKey with a mover, planner, fog memory, and terminal scoring

## Release notes

### 0.2.2

- Added optional input arcs via `peven.input(..., optional=True)`.
- Updated the MiniGrid DoorKey example so planner advice is an optional token,
  not a sentinel `{"advice": "none"}` token.
- Updated the packaged Julia runtime pins for optional-arc support.
- Added adapter parity coverage for optional inputs, optional-only rejection,
  and optional keyed-join rejection.

### 0.2.1

- Added `peven.place(terminal=True)` for Python-side completion normalization.
- Updated Rich output to hide `no_enabled_transition` for completed terminal-place runs.
- Added the MiniGrid DoorKey example under the `examples` dependency group.
- Added `gymnasium` and `minigrid` to the optional examples dependencies.

## Inspiration

Peven is inspired by a couple different things. For starters the name is taken from Patricia A. McKillip's Riddle-Master trilogy. Peven of Aum is a king, a ghost, and a master riddler who has only ever lost once. In the Riddle-Master trilogy, riddles are made up of three parts: questions, answers, and strictures. My hope for Peven is that it can help you explore evaluations by providing a runtime where you can ask a question, iterate based on the stricture, and, eventually, get to an answer. "Beware the unanswered Riddle."

My second point of inspiration comes from my time working at The LLM Data Company, where I had the chance to learn and experiment to my heart's content. A lot of my work centered around environments and benchmarks. I often wished I had a reusable framework or package to support my work here, something like a pydantic (which I love) but for evaluations.

Most of the architectural decisions I made regarding the engine are because I thought the math was cool. Peven should give you a pretty clear sense of (1) how I think about evaluations and (2) what types of evaluations I'm interested in.
