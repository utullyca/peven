# Peven

## Elevator Pitch

Peven is a rich topological engine for multi-agent evaluations.

## Inspiration

Peven is inspired by a couple different things. For starters the name is taken from Patricia A. McKillip's Riddle-Master trilogy. Peven of Aum is a king, a ghost, and a master riddler who has only ever lost once. In the Riddle-Master trilogy, riddles are made up of three parts: questions, answers, and strictures. My hope for Peven is that it can help you explore evaluations by providing a runtime where you can ask a question, iterate based on the stricture, and, eventually, get to an answer. "Beware the unanswered Riddle."

My second point of inspiration comes from my time working at The LLM Data Company, where I had the chance to learn and experiment to my heart's content. A lot of my work centered around environments and benchmarks. I often wished I had a reusable framework or package to support my work here, something like a pydantic (which I love) but for evaluations.

Most of the architectural decisions I made regarding the engine are because I thought the math was cool. Peven should give you a pretty clear sense of (1) how I think about evaluations and (2) what types of evaluations I'm interested in.

This is my first contribution to the open source ecosystem (not sure I can claim that before I even have 1 star so please star!) but selfishly, I do hope it evolves into something more meaningful. If this package is useful, or better yet interesting, to one person then I'll be happy.

## Bets

1. Petri nets are the best way to express multi-agent evaluations. Every agent loop, adversarial interaction, and multi-turn evaluation is a concurrent stateful system with shared resources. That's what Petri nets were invented to model.

2. Structure where you want it, chaos where you need it. Acyclic nets give you deterministic parallel experiments. Cyclic nets give you dynamic agent loops. One engine, one formalism.

## Heritage

An older version of this package was tested extensively with frontier OpenAI and Anthropic models and used complex DAGs instead of Petri nets. Some examples of the tests include: Supreme Court simulations where multiple justices drafted memos and voted, code repair leagues where solutions were executed and critiqued in loops, red-team tournaments with adversarial pairwise comparison, diplomacy crisis negotiations between multi-agent panels, and incident response exercises with staged evidence packets. Those experiments shaped key design decisions in this engine: why colored tokens for batch isolation, why async guards for judge gates, why the consume-eagerly pattern avoids locks, and why the harness itself is the experiment.

What matters in serious evaluations, in my opinion, is the shape of interaction: who sees what, in what order, what actions they can take, and how we judge those actions across individual or shared states. Peven favors radical explicitness: evaluation work should never hide inside implicit assumptions.

## Petri Nets

A Petri net is a mathematical model for concurrent systems. The key concepts:

- **Places** — containers that hold tokens. Think of them as states or buffers.
- **Transitions** — actions that consume tokens from input places and produce tokens in output places. In peven, transitions are LLM calls (agents, judges).
- **Arcs** — directed edges connecting places to transitions (inputs) and transitions to places (outputs). A net is always bipartite: places only connect to transitions, never directly to each other. Each arc has a **weight** (default 1): an input arc with weight $w$ means the transition needs $w$ tokens from that place to fire; an output arc with weight $w$ deposits $w$ copies of the result.
- **Tokens** — data flowing through the net. In peven, tokens carry text (GenerateOutput) or scores (JudgeOutput).
- **Firing rule** — a transition $t$ is enabled when every input place $p$ has at least $w(p, t)$ tokens. Firing consumes those tokens and deposits outputs. Multiple transitions can fire concurrently if they have independent inputs.
- **Colored tokens** — tokens tagged with a `run_id` for batch isolation. Multiple independent evaluations run through the same net simultaneously without interfering.

A marking $M$ maps each place to its token count. Transition $t$ is enabled when every input place has enough:

$$M(p) \geq w(p, t) \quad \forall \, p \in \bullet t$$

$\bullet t$ is the set of input places. $w(p, t)$ is the arc weight from place $p$ to transition $t$ (0 if no arc). When $t$ fires, every place updates:

$$M'(p) = M(p) - w(p, t) + w(t, p)$$

What was there, minus what the transition consumed, plus what it produced. $w(t, p)$ is the arc weight from $t$ back to $p$ (0 if no arc).

**Example:** A simple generate net with two places and one transition:

```python
prompt = n.place("prompt")      # holds the input token
response = n.place("response")  # receives the agent's output
generate = n.transition("generate", agent(model="openai:gpt-4o", prompt="{text}"))
prompt >> generate >> response   # arc weight = 1 (default)
prompt.token(GenerateOutput(text="hello"))  # initial marking: 1 token in prompt
```

```
Before:  prompt=1  response=0
         generate is enabled: prompt has 1 token >= arc weight 1 ✓

generate fires (calls gpt-4o, gets a response):
         prompt   = 1 - 1 + 0 = 0  (consumed 1 token, generate doesn't output here)
         response = 0 - 0 + 1 = 1  (generate deposits its output here)

After:   prompt=0  response=1
```

Most arcs in peven have weight 1, so it's just "take one, put one." Weights > 1 are for when a transition needs multiple tokens to fire, like a join that requires evidence from two places.

The engine uses an event-driven loop: transitions fire as soon as they're enabled (`asyncio.wait(FIRST_COMPLETED)`), tokens are consumed eagerly at spawn time and deposited on completion, and all marking mutations happen in a single-threaded central loop without locks.

Tokens are the unit of inter-node communication, not agent state. An agent can run multi-turn conversations, use tools, and accumulate context internally. The net only sees the final result as a token passed to the next transition. This is sub-optimal for any actor that needs to perceive internal agent state: the experiment designer debugging a run, a monitor agent screening intermediate reasoning, or a judge that needs to evaluate process rather than output. This is something I will be thinking about improving immediately.

## Install

```bash
uv add peven
```

## Quickstart

```python
from peven import NetBuilder, agent, judge, execute, GenerateOutput

n = NetBuilder()
prompt = n.place("prompt")
response = n.place("response")
scored = n.place("scored")

gen = n.transition("gen", agent(model="openai:gpt-4o", prompt="Write about {text}"))
jdg = n.transition("jdg", judge(model="openai:gpt-4o", rubric=[{"weight": 1.0, "requirement": "clear and engaging"}]))

prompt >> gen >> response >> jdg >> scored
prompt.token(GenerateOutput(text="the importance of testing"))

net = n.build()
results = await execute(net)  # in an async function

# Or from a script:
# import asyncio
# results = asyncio.run(execute(net))
```

## Nodes

Every transition in a net is either an **agent** (generates text) or a **judge** (scores text). Each node gets its own model and configuration.

### agent

```python
agent(model, prompt, system=None, tools=None, model_settings=None)
```

```python
gen = n.transition("gen", agent(
    model="openai:gpt-4o",
    prompt="Write a poem about {text}",
    system="You are a poet.",
))

revise = n.transition("revise", agent(
    model="anthropic:claude-sonnet-4-20250514",
    prompt="Revise this: {text}",
))
```

Models use pydantic-ai routing: `"openai:gpt-4o"`, `"anthropic:claude-sonnet-4-20250514"`, `"ollama:qwen2.5:0.5b"`, etc.

### judge

```python
judge(model, rubric, strategy="per_criterion", threshold=0.5)
```

```python
jdg = n.transition("jdg", judge(
    model="openai:gpt-4o",
    rubric=[
        {"weight": 1.0, "requirement": "clear and well-structured"},
        {"weight": 0.5, "requirement": "uses specific examples"},
    ],
))
```

Judges use the [rubric](https://pypi.org/project/rubric/) package, built at [The LLM Data Company](https://www.llmdata.com/). Three grading strategies:

| Strategy          | LLM calls                   | Output                                         |
| ----------------- | --------------------------- | ---------------------------------------------- |
| `per_criterion`   | One per criterion (default) | Per-criterion MET/UNMET with reasons           |
| `oneshot`         | One for all criteria        | Per-criterion MET/UNMET in a single call       |
| `rubric_as_judge` | One holistic call           | Single score 0-100, no per-criterion breakdown |

Different judges in the same net can use different models and strategies:

```python
jdg_deep = n.transition("deep", judge(model="openai:gpt-4o", rubric=rubric))
jdg_fast = n.transition("fast", judge(model="openai:gpt-4o-mini", rubric=rubric, strategy="oneshot"))
```

## CLI

```bash
# Run an eval file
peven run eval.py
peven run eval.py --concurrency 5 --fuse 500

# Show per-transition execution trace (which transitions fired, outputs, scores)
peven run eval.py --trace

# Validate and inspect topology
peven validate eval.py

# Review stored runs
peven review all              # every run
peven review last 10          # last N runs
peven review <run_id>
peven review <run_id> --trace
```

Every `peven run` automatically persists results to `~/.peven/runs.db` (created on first run, no setup needed) so you can review them later with `peven review`.

## Examples

The `examples/` folder has toy examples to get started:

- **`simple.py`** — Single generate. Minimal net.
- **`refine.py`** — Generate-judge-revise loop. Cycles back until score passes threshold.
- **`debate.py`** — Two agents argue in parallel, judge scores the result. Fork-join topology.

```bash
peven run examples/refine.py --trace
peven validate examples/debate.py
```

## Releases

This is v0.1. See [ROADMAP.md](./ROADMAP.md) for what's next.

## Tests

```bash
# Unit + integration
uv run pytest tests/ --ignore=tests/test_e2e.py -v

# E2E with live LLM (requires ollama)
uv run pytest tests/test_e2e.py -v
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).
