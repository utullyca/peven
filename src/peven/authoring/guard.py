"""Declarative guard DSL for Python-authored peven transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

from ._expr import (
    coerce_scalar_literal,
    extend_identifier_path,
    reject_indexing,
    require_identifier_path,
    root_identifier_path,
)


_COMPARISON_OPERATORS: Final[frozenset[str]] = frozenset({"==", "!=", "<", "<=", ">", ">="})
_CALL_ARITY: Final[dict[str, int]] = {
    "isnothing": 1,
    "isempty": 1,
    "length": 1,
}
__all__ = ["f", "in_", "isempty", "isnothing", "length"]


class GuardNode:
    """Base class for immutable guard expression nodes."""

    __slots__ = ()

    @property
    def produces_bool(self) -> bool:
        raise NotImplementedError

    def to_spec(self) -> dict[str, object]:
        raise NotImplementedError

    def __bool__(self) -> bool:
        raise TypeError("guard expressions do not define truthiness; use &, |, and ~")

    def __and__(self, other: object) -> And:
        return And((coerce_guard_node(self), coerce_guard_node(other)))

    def __or__(self, other: object) -> Or:
        return Or((coerce_guard_node(self), coerce_guard_node(other)))

    def __invert__(self) -> Not:
        return Not(coerce_guard_node(self))

    def __eq__(self, other: object) -> bool:
        # The DSL returns a comparison node at runtime, but we keep the
        # signature object-compatible so static type checkers accept it.
        return cast(bool, Cmp("==", coerce_guard_node(self), literal(other)))

    def __ne__(self, other: object) -> bool:
        return cast(bool, Cmp("!=", coerce_guard_node(self), literal(other)))

    def __lt__(self, other: object) -> Cmp:
        return Cmp("<", coerce_guard_node(self), literal(other))

    def __le__(self, other: object) -> Cmp:
        return Cmp("<=", coerce_guard_node(self), literal(other))

    def __gt__(self, other: object) -> Cmp:
        return Cmp(">", coerce_guard_node(self), literal(other))

    def __ge__(self, other: object) -> Cmp:
        return Cmp(">=", coerce_guard_node(self), literal(other))


@dataclass(frozen=True, slots=True, eq=False)
class FieldRef(GuardNode):
    """Path reference rooted at the selected input token payload."""

    path: tuple[str, ...]

    @property
    def produces_bool(self) -> bool:
        return False

    def __getattr__(self, name: str) -> FieldRef:
        return FieldRef(
            extend_identifier_path(
                self.path,
                name,
                error_message="guard field path segments must be identifiers",
            )
        )

    def __getitem__(self, key: object) -> FieldRef:
        del key
        reject_indexing(error_message="guard field refs do not support indexing in v0.2a")

    def to_spec(self) -> dict[str, object]:
        return {"kind": "field_ref", "path": list(self.path)}


@dataclass(frozen=True, slots=True, eq=False)
class Literal(GuardNode):
    """Structured scalar literal embedded into a guard expression."""

    value: object

    @property
    def produces_bool(self) -> bool:
        return False

    def to_spec(self) -> dict[str, object]:
        return {"kind": "literal", "value": self.value}


@dataclass(frozen=True, slots=True, eq=False)
class Cmp(GuardNode):
    """Binary comparison over guard operands."""

    op: str
    left: GuardNode
    right: GuardNode

    @property
    def produces_bool(self) -> bool:
        return True

    def to_spec(self) -> dict[str, object]:
        return {
            "kind": "cmp",
            "op": self.op,
            "left": self.left.to_spec(),
            "right": self.right.to_spec(),
        }


@dataclass(frozen=True, slots=True, eq=False)
class Call(GuardNode):
    """Whitelisted guard callable."""

    name: str
    args: tuple[GuardNode, ...]

    @property
    def produces_bool(self) -> bool:
        return self.name != "length"

    def to_spec(self) -> dict[str, object]:
        return {
            "kind": "call",
            "name": self.name,
            "args": [arg.to_spec() for arg in self.args],
        }


@dataclass(frozen=True, slots=True, eq=False)
class In(GuardNode):
    """Membership predicate over a field ref and flat literal list."""

    ref: GuardNode
    values: tuple[Literal, ...]

    @property
    def produces_bool(self) -> bool:
        return True

    def to_spec(self) -> dict[str, object]:
        return {
            "kind": "in",
            "ref": self.ref.to_spec(),
            "values": [value.to_spec() for value in self.values],
        }


@dataclass(frozen=True, slots=True, eq=False)
class And(GuardNode):
    """Boolean conjunction."""

    children: tuple[GuardNode, ...]

    @property
    def produces_bool(self) -> bool:
        return True

    def to_spec(self) -> dict[str, object]:
        return {"kind": "and", "children": [child.to_spec() for child in self.children]}


@dataclass(frozen=True, slots=True, eq=False)
class Or(GuardNode):
    """Boolean disjunction."""

    children: tuple[GuardNode, ...]

    @property
    def produces_bool(self) -> bool:
        return True

    def to_spec(self) -> dict[str, object]:
        return {"kind": "or", "children": [child.to_spec() for child in self.children]}


@dataclass(frozen=True, slots=True, eq=False)
class Not(GuardNode):
    """Boolean negation."""

    child: GuardNode

    @property
    def produces_bool(self) -> bool:
        return True

    def to_spec(self) -> dict[str, object]:
        return {"kind": "not", "child": self.child.to_spec()}


class _FieldRoot:
    """Attribute-only root for the public `f` field proxy."""

    __slots__ = ()

    def __getattr__(self, name: str) -> FieldRef:
        return FieldRef(
            root_identifier_path(
                name,
                error_message="guard field path segments must be identifiers",
            )
        )

    def __getitem__(self, key: object) -> FieldRef:
        del key
        reject_indexing(error_message="guard field refs do not support indexing in v0.2a")


f: Final[_FieldRoot] = _FieldRoot()


def literal(value: object) -> Literal:
    """Construct one scalar literal node after validation."""
    return coerce_scalar_literal(
        value,
        base_type=GuardNode,
        literal_type=Literal,
        error_message="guard literals must be scalar structured values",
    )


def isnothing(ref: object) -> Call:
    return Call("isnothing", (coerce_guard_node(ref),))


def isempty(ref: object) -> Call:
    return Call("isempty", (coerce_guard_node(ref),))


def length(ref: object) -> Call:
    return Call("length", (coerce_guard_node(ref),))


def in_(ref: object, values: object) -> In:
    node = coerce_guard_node(ref)
    if type(values) is not list:
        raise TypeError("guard in_() values must be a flat list of scalar literals")
    return In(node, tuple(literal(value) for value in values))


def coerce_guard_node(value: object) -> GuardNode:
    """Normalize one public guard operand into a concrete node."""
    if isinstance(value, GuardNode):
        return value
    raise TypeError("guard operands must be built from peven.guard nodes")


def validate_guard_tree(node: GuardNode) -> None:
    """Validate one guard tree against the v0.2a public DSL contract."""
    _validate_guard_node(node)


def _validate_guard_node(node: GuardNode) -> None:
    if isinstance(node, FieldRef):
        require_identifier_path(
            node.path,
            error_message="guard field path segments must be identifiers",
        )
        return
    if isinstance(node, Literal):
        literal(node)
        return
    if isinstance(node, Cmp):
        if node.op not in _COMPARISON_OPERATORS:
            raise ValueError(f"unknown guard comparison op {node.op!r}")
        _validate_guard_node(node.left)
        _validate_guard_node(node.right)
        if node.left.produces_bool or node.right.produces_bool:
            raise ValueError("guard cmp children must be scalar-valued")
        return
    if isinstance(node, Call):
        arity = _CALL_ARITY.get(node.name)
        if arity is None:
            raise ValueError(f"unknown guard callable {node.name!r}")
        if len(node.args) != arity:
            raise ValueError(f"{node.name}() expects exactly one argument")
        for argument in node.args:
            _validate_guard_node(argument)
        return
    if isinstance(node, In):
        _validate_guard_node(node.ref)
        if node.ref.produces_bool:
            raise ValueError("guard in ref must be scalar-valued")
        if not node.values:
            raise ValueError(
                "guard in_() values must be a flat literal list of scalar structured values"
            )
        for value in node.values:
            if not isinstance(value, Literal):
                raise TypeError(
                    "guard in_() values must be a flat literal list of scalar structured values"
                )
            try:
                _validate_guard_node(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "guard in_() values must be a flat literal list of scalar structured values"
                ) from exc
        return
    if isinstance(node, And | Or):
        if not node.children:
            raise ValueError("guard boolean combinators require at least one child")
        for child in node.children:
            _validate_guard_node(child)
            if not child.produces_bool:
                raise ValueError("guard boolean combinators require boolean-producing children")
        return
    if isinstance(node, Not):
        _validate_guard_node(node.child)
        if not node.child.produces_bool:
            raise ValueError("guard boolean combinators require boolean-producing children")
        return
    raise TypeError(
        f"unsupported guard node type {type(node).__name__}; use the built-in peven.guard DSL only"
    )
