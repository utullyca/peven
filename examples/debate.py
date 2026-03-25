"""Parallel debate with judge.

Two agents argue opposite sides of a topic in parallel, then a judge
scores the combined arguments. Shows fork-join topology.

    peven run examples/debate.py --trace
    peven validate examples/debate.py

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
pro_argument = n.place("pro_argument")
con_argument = n.place("con_argument")
verdict = n.place("verdict")

# Transitions — two agents argue in parallel
advocate = n.transition(
    "advocate",
    agent(
        model=MODEL,
        prompt="Argue strongly IN FAVOR of this in 2-3 sentences: {text}",
        system="You are a persuasive debater. Be specific and compelling.",
    ),
)

critic = n.transition(
    "critic",
    agent(
        model=MODEL,
        prompt="Argue strongly AGAINST this in 2-3 sentences: {text}",
        system="You are a sharp critic. Be specific and compelling.",
    ),
)

# Toy rubric for demonstration — real rubrics should be atomic and verifiable.
# Using rubric_as_judge strategy since small models struggle with per-criterion structured output.
score = n.transition(
    "score",
    judge(
        model=MODEL,
        rubric=[
            {"weight": 0.5, "requirement": "arguments are specific, not vague generalities"},
            {"weight": 0.5, "requirement": "arguments directly address the topic"},
        ],
        strategy="rubric_as_judge",
    ),
)

# Topology — fork from topic, join at judge
topic >> advocate >> pro_argument >> score
topic >> critic >> con_argument >> score
score >> verdict

# Seed — two copies so both branches can fire
topic.token(GenerateOutput(text="remote work is better than office work"))
topic.token(GenerateOutput(text="remote work is better than office work"))

net = n.build()
