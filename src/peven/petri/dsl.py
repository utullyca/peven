"""DSL for authoring Petri nets.

Builder API with >> chaining that compiles to the Pydantic IR (Net).

    from peven.petri.dsl import NetBuilder, agent, judge

    n = NetBuilder()
    prompt = n.place("prompt")
    response = n.place("response")
    gen = n.transition("gen", agent(model="opus", prompt="{text}"))
    prompt >> gen >> response
    net = n.build()
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from peven.petri.schema import (
    Arc,
    GenerateConfig,
    JudgeConfig,
    Marking,
    Net,
    Place,
    Token,
    Transition,
    TransitionConfig,
)


# -- Config helpers ------------------------------------------------------------


def agent(
    model: str,
    prompt: str,
    system: str | None = None,
    tools: list[Callable] | None = None,
    model_settings: dict[str, Any] | None = None,
) -> tuple[str, GenerateConfig]:
    """Create an agent executor config."""
    return (
        "agent",
        GenerateConfig(
            model=model,
            prompt_template=prompt,
            system_prompt=system,
            tools=tools,
            model_settings=model_settings,
        ),
    )


def judge(
    model: str,
    rubric: list[dict[str, Any]],
    strategy: str = "per_criterion",
    threshold: float = 0.5,
) -> tuple[str, JudgeConfig]:
    """Create a rubric judge executor config."""
    return (
        "judge",
        JudgeConfig(model=model, rubric=rubric, strategy=strategy, pass_threshold=threshold),
    )


# -- Proxy objects -------------------------------------------------------------


class PlaceProxy:
    """Proxy for a place. Supports >> to create arcs."""

    def __init__(self, builder: NetBuilder, id: str, capacity: int | None = None):
        self._builder = builder
        self.id = id
        self.capacity = capacity

    def __rshift__(self, other: TransitionProxy) -> TransitionProxy:
        if not isinstance(other, TransitionProxy):
            raise TypeError(f"Place can only connect to a Transition, got {type(other).__name__}")
        self._builder._arc(self.id, other.id)
        return other

    def token(self, tok: Token) -> PlaceProxy:
        """Add an initial token to this place."""
        self._builder._token(self.id, tok)
        return self


class TransitionProxy:
    """Proxy for a transition. Supports >> and .when() for guards."""

    def __init__(
        self,
        builder: NetBuilder,
        id: str,
        executor: str,
        config: TransitionConfig | None,
        retries: int = 0,
    ):
        self._builder = builder
        self.id = id
        self.executor = executor
        self.config = config
        self._when = None
        self._retries = retries

    def __rshift__(self, other: PlaceProxy) -> PlaceProxy:
        if not isinstance(other, PlaceProxy):
            raise TypeError(f"Transition can only connect to a Place, got {type(other).__name__}")
        self._builder._arc(self.id, other.id)
        return other

    def when(self, predicate: Callable[[list[Token]], Any]) -> TransitionProxy:
        """Attach a guard. Predicate receives list[Token], returns bool. Can be async."""
        self._when = predicate
        return self


# -- Builder -------------------------------------------------------------------


class NetBuilder:
    """Build a Petri net with >> chaining, then compile to Net IR."""

    def __init__(self):
        self._places: dict[str, PlaceProxy] = {}
        self._transitions: dict[str, TransitionProxy] = {}
        self._arcs: list[tuple[str, str, int]] = []
        self._tokens: dict[str, list[Token]] = {}

    def place(self, id: str, capacity: int | None = None) -> PlaceProxy:
        """Create a place."""
        p = PlaceProxy(self, id, capacity)
        self._places[id] = p
        return p

    def transition(self, id: str, executor_config: tuple, retries: int = 0) -> TransitionProxy:
        """Create a transition. executor_config from agent() or judge()."""
        name, config = executor_config
        t = TransitionProxy(self, id, name, config, retries=retries)
        self._transitions[id] = t
        return t

    def _arc(self, source: str, target: str, weight: int = 1):
        self._arcs.append((source, target, weight))

    def _token(self, place_id: str, tok: Token):
        self._tokens.setdefault(place_id, []).append(tok)

    def build(self) -> Net:
        """Compile to Pydantic Net IR."""
        return Net(
            places=[Place(id=p.id, capacity=p.capacity) for p in self._places.values()],
            transitions=[
                Transition(
                    id=t.id,
                    executor=t.executor,
                    config=t.config,
                    when=t._when,
                    retries=t._retries,
                )
                for t in self._transitions.values()
            ],
            arcs=[Arc(source=s, target=t, weight=w) for s, t, w in self._arcs],
            initial_marking=Marking(tokens=self._tokens),
        )
