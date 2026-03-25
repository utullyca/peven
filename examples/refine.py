"""Generate-judge-revise loop.

Writes an essay, judges it against a rubric, and revises until the
score passes threshold or the fuse runs out.

    peven run examples/refine.py --trace
    peven validate examples/refine.py

Requires ollama with qwen2.5:0.5b:
    ollama pull qwen2.5:0.5b
"""

from peven.petri.dsl import NetBuilder, agent, judge
from peven.petri.types import GenerateOutput

# -- Model ---------------------------------------------------------------------
# Using ollama locally. Swap for "openai:gpt-4o" or "anthropic:claude-sonnet-4-20250514" etc.

MODEL = "ollama:qwen2.5:0.5b"

# -- Net -----------------------------------------------------------------------

n = NetBuilder()

# Places
topic = n.place("topic")
draft = n.place("draft")
scored = n.place("scored")
final = n.place("final")

# Transitions
write = n.transition(
    "write",
    agent(
        model=MODEL,
        prompt="Write a 3-sentence essay about: {text}",
        system="You are a clear, concise writer.",
    ),
)

# Toy rubric for demonstration — real rubrics should be atomic and verifiable.
# Using rubric_as_judge strategy since small models struggle with per-criterion structured output.
evaluate = n.transition(
    "evaluate",
    judge(
        model=MODEL,
        rubric=[
            {"weight": 0.4, "requirement": "has a clear thesis statement"},
            {"weight": 0.3, "requirement": "includes a specific example or evidence"},
            {"weight": 0.3, "requirement": "writing is professional and engaging"},
        ],
        strategy="rubric_as_judge",
    ),
)

revise = n.transition(
    "revise",
    agent(
        model=MODEL,
        prompt="This essay scored below threshold. Improve it — make the thesis clearer, "
        "add a specific example, and tighten the prose: {text}",
        system="You are a strict editor.",
    ),
)
revise.when(lambda tokens: tokens[0].score < 0.7)

accept = n.transition(
    "accept",
    agent(
        model=MODEL,
        prompt="{text}",
    ),
)
accept.when(lambda tokens: tokens[0].score >= 0.7)

# Topology
topic >> write >> draft >> evaluate >> scored
scored >> revise >> topic  # cycle back for revision
scored >> accept >> final  # exit when good enough

# Seed
topic.token(GenerateOutput(text="why open source matters for AI safety"))

net = n.build()
