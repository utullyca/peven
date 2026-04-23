"""Declarative join-selector DSL for keyed joins in Python-authored peven nets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ._expr import (
    coerce_scalar_literal,
    extend_identifier_path,
    reject_indexing,
    require_identifier_path,
    root_identifier_path,
)


__all__ = ["join_key", "payload", "place_id"]


class JoinNode:
    """Base class for immutable join-selector expression nodes."""

    __slots__ = ()

    def to_spec(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PayloadRef(JoinNode):
    """Path reference rooted at one input token payload."""

    path: tuple[str, ...]

    def __getattr__(self, name: str) -> PayloadRef:
        return PayloadRef(
            extend_identifier_path(
                self.path,
                name,
                error_message="join_by payload path segments must be identifiers",
            )
        )

    def __getitem__(self, key: object) -> PayloadRef:
        del key
        reject_indexing(error_message="join_by payload refs do not support indexing in v0.4")

    def to_spec(self) -> dict[str, object]:
        return {"kind": "payload_ref", "path": list(self.path)}


@dataclass(frozen=True, slots=True)
class PlaceIdRef(JoinNode):
    """Reference to the current input place id inside a join selector."""

    def to_spec(self) -> dict[str, object]:
        return {"kind": "place_id"}


@dataclass(frozen=True, slots=True)
class JoinLiteral(JoinNode):
    """Scalar literal embedded into a join selector."""

    value: object

    def to_spec(self) -> dict[str, object]:
        return {"kind": "literal", "value": self.value}


@dataclass(frozen=True, slots=True)
class JoinTuple(JoinNode):
    """Composite join key built from multiple selector parts."""

    items: tuple[JoinNode, ...]

    def to_spec(self) -> dict[str, object]:
        return {"kind": "tuple", "items": [item.to_spec() for item in self.items]}


class _PayloadRoot:
    """Attribute-only root for the public `payload` join selector proxy."""

    __slots__ = ()

    def __getattr__(self, name: str) -> PayloadRef:
        return PayloadRef(
            root_identifier_path(
                name,
                error_message="join_by payload path segments must be identifiers",
            )
        )

    def __getitem__(self, key: object) -> PayloadRef:
        del key
        reject_indexing(error_message="join_by payload refs do not support indexing in v0.4")


payload: Final[_PayloadRoot] = _PayloadRoot()
place_id: Final[PlaceIdRef] = PlaceIdRef()


def join_key(*parts: object) -> JoinNode:
    """Build a declarative join selector from one or more selector parts."""
    if not parts:
        raise ValueError("join_key() requires at least one selector part")
    if len(parts) == 1:
        return coerce_join_node(parts[0])
    return JoinTuple(tuple(coerce_join_node(part) for part in parts))


def join_literal(value: object) -> JoinLiteral:
    """Normalize one scalar literal captured into a join selector."""
    return coerce_scalar_literal(
        value,
        base_type=JoinNode,
        literal_type=JoinLiteral,
        error_message="join_by literals must be scalar structured values",
    )


def coerce_join_node(value: object) -> JoinNode:
    """Normalize one public join-selector operand into a concrete node."""
    if isinstance(value, JoinNode):
        return value
    return join_literal(value)


def validate_join_tree(node: JoinNode) -> None:
    """Validate one join-selector tree against the v0.4 keyed-join contract."""
    _validate_join_node(node)


def _validate_join_node(node: JoinNode) -> None:
    if isinstance(node, PayloadRef):
        require_identifier_path(
            node.path,
            error_message="join_by payload path segments must be identifiers",
        )
        return
    if isinstance(node, PlaceIdRef):
        return
    if isinstance(node, JoinLiteral):
        join_literal(node)
        return
    if isinstance(node, JoinTuple):
        if not node.items:
            raise ValueError("join_by tuple selectors require at least one item")
        for item in node.items:
            _validate_join_node(item)
        return
    raise TypeError(f"unsupported join_by node type: {type(node).__name__}")
