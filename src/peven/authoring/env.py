"""Env base class and class-finalization for Python authoring."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final, cast

from ..handoff.lowering import compile_env
from ..shared.errors import PevenValidationError, ValidationIssue
from .executor import get_executor_registry_version
from .ir import EnvSpec, InputArcSpec, OutputArcSpec, PlaceSpec, TransitionSpec
from .topology import PlaceDecl, TransitionDecl


__all__ = ["Env", "env"]

_PEVEN_ENV_SPEC_ATTR: Final[str] = "__peven_env_spec__"
_PEVEN_COMPILED_ENV_ATTR: Final[str] = "__peven_compiled_env__"
_PEVEN_COMPILED_ENV_VERSION_ATTR: Final[str] = "__peven_compiled_env_version__"


def _is_authored_decl(value: object) -> bool:
    return isinstance(value, (PlaceDecl, TransitionDecl))


class _AuthoringNamespace(dict[str, object]):
    """Class namespace that records overwritten authoring declarations.

    Python class bodies overwrite earlier names before decorators ever see the
    finished class object. We keep a tiny custom namespace only so `@env(...)`
    can still reject duplicate places/transitions without scanning source text.
    """

    def __init__(self) -> None:
        super().__init__()
        self.duplicate_authored_names: set[str] = set()
        self.overwritten_authored_names: set[str] = set()

    def __setitem__(self, key: str, value: object) -> None:
        previous = self.get(key)
        if _is_authored_decl(previous):
            if _is_authored_decl(value):
                self.duplicate_authored_names.add(key)
            else:
                self.overwritten_authored_names.add(key)
        super().__setitem__(key, value)


class _AuthoringEnvMeta(type):
    """Metaclass that carries duplicate authoring names onto the final class."""

    @classmethod
    def __prepare__(
        mcls, name: str, bases: tuple[type[object], ...], **kwargs: object
    ) -> _AuthoringNamespace:
        return _AuthoringNamespace()

    def __new__(
        mcls,
        name: str,
        bases: tuple[type[object], ...],
        namespace: dict[str, object],
        **kwargs: object,
    ) -> type[object]:
        cls = super().__new__(mcls, name, bases, dict(namespace), **kwargs)
        duplicates = ()
        overwritten = ()
        if isinstance(namespace, _AuthoringNamespace):
            duplicates = tuple(sorted(namespace.duplicate_authored_names))
            overwritten = tuple(sorted(namespace.overwritten_authored_names))
        typed_cls = cast(Any, cls)
        typed_cls.__peven_duplicate_authored_names__ = duplicates
        typed_cls.__peven_overwritten_authored_names__ = overwritten
        return cls


class Env(metaclass=_AuthoringEnvMeta):
    """Base class for Python-authored peven topology."""

    @classmethod
    def spec(cls) -> EnvSpec:
        """Return the authored IR cached on the decorated env class."""
        spec = cls.__dict__.get(_PEVEN_ENV_SPEC_ATTR)
        if not isinstance(spec, EnvSpec):
            raise TypeError("env class has not been decorated with @peven.env(...)")
        return spec

    @classmethod
    def compiled(cls) -> object:
        """Return the compiled handoff artifact cached on the decorated env class."""
        registry_version = get_executor_registry_version()
        compiled = cls.__dict__.get(_PEVEN_COMPILED_ENV_ATTR)
        compiled_version = cls.__dict__.get(_PEVEN_COMPILED_ENV_VERSION_ATTR)
        if compiled is None or compiled_version != registry_version:
            compiled = compile_env(cls.spec())
            typed_cls = cast(Any, cls)
            typed_cls.__peven_compiled_env__ = compiled
            typed_cls.__peven_compiled_env_version__ = registry_version
        return compiled

    def initial_marking(
        self, seed: int | None = None
    ):  # pragma: no cover - later layer
        raise NotImplementedError(
            "initial_marking belongs to the handoff/runtime layers"
        )

    def run(self, **kwargs: object):
        """Run one authored env through the implemented bridge on the sync runtime loop."""
        from ..runtime.bridge import run_env
        from ..runtime.state import run_sync

        return run_sync(run_env(self, **kwargs))


def env(name: str) -> Callable[[type[Env]], type[Env]]:
    """Finalize one Env subclass into immutable authoring IR."""
    if type(name) is not str or not name:
        raise ValueError("env name must be a non-empty string")

    def decorator(cls: type[Env]) -> type[Env]:
        if not issubclass(cls, Env):
            raise TypeError("@peven.env(...) requires an Env subclass")
        if cls.__bases__ != (Env,):
            raise PevenValidationError(
                [
                    ValidationIssue(
                        "invalid_env",
                        cls.__name__,
                        "env inheritance is not supported",
                    )
                ]
            )
        duplicates = getattr(cls, "__peven_duplicate_authored_names__", ())
        overwritten = getattr(cls, "__peven_overwritten_authored_names__", ())
        if duplicates:
            issues = [
                ValidationIssue(
                    "duplicate_declaration", name, f"duplicate declaration {name}"
                )
                for name in duplicates
            ]
            raise PevenValidationError(issues)
        if overwritten:
            issues = [
                ValidationIssue(
                    "overwritten_declaration",
                    name,
                    f"authored declaration {name} was overwritten before class finalization",
                )
                for name in overwritten
            ]
            raise PevenValidationError(issues)
        spec = _collect_env_spec(cls, env_name=name)
        typed_cls = cast(Any, cls)
        typed_cls.__peven_env_spec__ = spec
        typed_cls.__peven_compiled_env__ = compile_env(spec)
        typed_cls.__peven_compiled_env_version__ = get_executor_registry_version()
        return cls

    return decorator


def _collect_env_spec(cls: type[Env], *, env_name: str) -> EnvSpec:
    places: list[PlaceSpec] = []
    transitions: list[TransitionSpec] = []

    for attribute_name, value in cls.__dict__.items():
        if isinstance(value, PlaceDecl):
            places.append(
                PlaceSpec(
                    id=attribute_name,
                    capacity=value.capacity,
                    schema=value.schema,
                    terminal=value.terminal,
                )
            )
            continue
        if not isinstance(value, TransitionDecl):
            continue
        declaration = value
        transitions.append(
            TransitionSpec(
                id=attribute_name,
                executor=declaration.executor,
                inputs=tuple(
                    InputArcSpec(
                        place=arc.place,
                        weight=arc.weight,
                        optional=arc.optional,
                    )
                    for arc in declaration.inputs
                ),
                outputs=tuple(
                    OutputArcSpec(place=arc.place) for arc in declaration.outputs
                ),
                guard_spec=None
                if declaration.guard is None
                else declaration.guard.to_spec(),
                retries=declaration.retries,
                join_by_spec=(
                    None
                    if declaration.join_by is None
                    else declaration.join_by.to_spec()
                ),
            )
        )

    spec = EnvSpec(
        env_name=env_name, places=tuple(places), transitions=tuple(transitions)
    )
    return spec
