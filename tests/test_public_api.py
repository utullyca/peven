"""Root package exports should stay intentional."""

from peven import (
    GenerateOutput,
    NetBuilder,
    __version__,
    agent,
    judge,
    score_at_least,
)


def test_public_api_exports():
    assert __version__ == "0.1.1"
    assert callable(agent)
    assert callable(judge)
    assert callable(score_at_least)

    n = NetBuilder()
    start = n.place("start")
    out = n.place("out")
    gen = n.transition("gen", agent(model="test", prompt="{text}"))
    start >> gen >> out
    start.token(GenerateOutput(text="hi"))

    net = n.build()
    assert net.places[0].id == "start"
