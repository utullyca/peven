"""Simple generate pipeline.

Minimal net: one agent, one prompt, one response.

    peven run examples/simple.py
    peven validate examples/simple.py

Requires ollama with qwen2.5:0.5b:
    ollama pull qwen2.5:0.5b
"""

from peven.petri.dsl import NetBuilder, agent
from peven.petri.types import GenerateOutput

MODEL = "ollama:qwen2.5:0.5b"

n = NetBuilder()
prompt = n.place("prompt")
response = n.place("response")
gen = n.transition("gen", agent(model=MODEL, prompt="Write a short poem about {text}"))
prompt >> gen >> response
prompt.token(GenerateOutput(text="the color of rain"))

net = n.build()
