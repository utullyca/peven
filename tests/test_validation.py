"""Test validation: every error branch."""

from __future__ import annotations

import pytest

from peven.petri.schema import Arc, Marking, Net, Place, Token, Transition, ValidationError
from peven.petri.validation import (
    arc_directionality,
    arc_integrity,
    id_uniqueness,
    marking_validity,
    reachability,
    validate,
)

# -- validate() orchestration --------------------------------------------------


def test_validate_calls_all_checks():
    """validate() catches errors from each check, not just the first."""
    # Net with duplicate places (caught by id_uniqueness, not the others)
    net = Net(
        places=[Place(id="a"), Place(id="a")],
        transitions=[Transition(id="t", executor="agent")],
        arcs=[Arc(source="a", target="t")],
        initial_marking=Marking(tokens={}),
    )
    with pytest.raises(ValidationError, match="Duplicate place ID"):
        validate(net)


def test_validate_catches_reachability():
    """validate() runs reachability check (the last check in the chain)."""
    net = Net(
        places=[Place(id="p"), Place(id="q")],
        transitions=[Transition(id="t1", executor="agent"), Transition(id="t2", executor="agent")],
        arcs=[Arc(source="p", target="t1"), Arc(source="t1", target="q")],
        initial_marking=Marking(tokens={"p": [Token()]}),
    )
    with pytest.raises(ValidationError, match="Unreachable"):
        validate(net)


# -- id_uniqueness ----------------------------------------------------------------


def test_duplicate_place():
    with pytest.raises(ValidationError, match="Duplicate place ID"):
        id_uniqueness([Place(id="a"), Place(id="a")], [])


def test_duplicate_transition():
    with pytest.raises(ValidationError, match="Duplicate ID"):
        id_uniqueness([Place(id="a")], [Transition(id="a", executor="agent")])


# -- arc_integrity -------------------------------------------------------------


def test_arc_weight_invalid():
    arcs = [Arc(source="p", target="t", weight=0)]
    with pytest.raises(ValidationError, match="invalid weight"):
        arc_integrity(arcs, {"p"}, {"t"})


def test_unknown_arc_source():
    arcs = [Arc(source="ghost", target="t")]
    with pytest.raises(ValidationError, match="unknown source"):
        arc_integrity(arcs, {"p"}, {"t"})


def test_unknown_arc_target():
    arcs = [Arc(source="p", target="ghost")]
    with pytest.raises(ValidationError, match="unknown target"):
        arc_integrity(arcs, {"p"}, {"t"})


# -- arc_directionality --------------------------------------------------------


def test_place_to_place():
    arcs = [Arc(source="p1", target="p2")]
    with pytest.raises(ValidationError, match="place to place"):
        arc_directionality(arcs, {"p1", "p2"}, set())


def test_transition_to_transition():
    arcs = [Arc(source="t1", target="t2")]
    with pytest.raises(ValidationError, match="transition to transition"):
        arc_directionality(arcs, set(), {"t1", "t2"})


# -- marking_validity ----------------------------------------------------------


def test_marking_unknown_place():
    marking = Marking(tokens={"ghost": [Token()]})
    with pytest.raises(ValidationError, match="unknown place"):
        marking_validity(marking, [Place(id="real")])


def test_marking_exceeds_capacity():
    marking = Marking(tokens={"p": [Token(), Token()]})
    with pytest.raises(ValidationError, match="exceeds capacity"):
        marking_validity(marking, [Place(id="p", capacity=1)])


# -- reachability --------------------------------------------------------------


def test_no_transitions():
    # Should return without error
    reachability([], Marking(tokens={}), {"p"}, set())


def test_unreachable_transition():
    arcs = [Arc(source="p", target="t1")]
    marking = Marking(tokens={"p": [Token()]})
    with pytest.raises(ValidationError, match="Unreachable"):
        reachability(arcs, marking, {"p"}, {"t1", "t2"})
