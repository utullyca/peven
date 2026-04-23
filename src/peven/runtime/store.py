"""Run-scoped opaque-ref store used by authored env callbacks."""

from __future__ import annotations

from contextvars import ContextVar, Token as ContextToken
from dataclasses import dataclass, field


__all__ = [
    "activate_store",
    "clear_store",
    "get",
    "open_store",
    "put",
    "release",
    "reset_store",
]


@dataclass(slots=True)
class _RunStore:
    """Mutable ref store for one active env run."""

    env_run_id: int
    refs: dict[str, object] = field(default_factory=dict)
    next_ref_id: int = 1

    def put(self, value: object) -> str:
        ref = f"peven-ref:{self.env_run_id}:{self.next_ref_id}"
        self.next_ref_id += 1
        self.refs[ref] = value
        return ref

    def get(self, ref: str) -> object:
        return self.refs[ref]

    def release(self, ref: str) -> None:
        self.refs.pop(ref, None)

    def clear(self) -> None:
        self.refs.clear()


_ACTIVE_STORE: ContextVar[_RunStore | None] = ContextVar("peven_active_store", default=None)


def open_store(env_run_id: int) -> _RunStore:
    """Construct one new empty run-scoped store."""
    return _RunStore(env_run_id=env_run_id)


def activate_store(store: _RunStore) -> ContextToken[_RunStore | None]:
    """Install one run store into the ambient context."""
    return _ACTIVE_STORE.set(store)


def reset_store(token: ContextToken[_RunStore | None]) -> None:
    """Restore the prior ambient store binding."""
    _ACTIVE_STORE.reset(token)


def clear_store(store: _RunStore) -> None:
    """Release every remaining ref in one run store."""
    store.clear()


def put(value: object) -> str:
    """Store one opaque Python object in the active run store."""
    return _require_active_store().put(value)


def get(ref: str) -> object:
    """Resolve one ref string inside the active run store."""
    return _require_active_store().get(ref)


def release(ref: str) -> None:
    """Release one ref string from the active run store."""
    _require_active_store().release(ref)


def _require_active_store() -> _RunStore:
    store = _ACTIVE_STORE.get()
    if store is None:
        raise RuntimeError("peven.store may only be used during an active Env.run()")
    return store
