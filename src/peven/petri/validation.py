"""Petri net validation."""

from __future__ import annotations

from peven.petri.schema import Arc, Marking, Net, Place, Transition, ValidationError


def validate(net: Net) -> None:
    """Run all structural checks. Raises ValidationError on failure."""
    place_ids = {p.id for p in net.places}
    transition_ids = {t.id for t in net.transitions}

    id_uniqueness(net.places, net.transitions)
    score_transition_validity(net.score_transition_id, transition_ids)
    arc_integrity(net.arcs, place_ids, transition_ids)
    arc_directionality(net.arcs, place_ids, transition_ids)
    marking_validity(net.initial_marking, net.places)
    reachability(net.arcs, net.initial_marking, place_ids, transition_ids)


def id_uniqueness(places: list[Place], transitions: list[Transition]) -> None:
    """Every place and transition ID must be unique across the net."""
    seen: set[str] = set()
    for place in places:
        if place.id in seen:
            raise ValidationError(f"Duplicate place ID: {place.id!r}")
        seen.add(place.id)
    for transition in transitions:
        if transition.id in seen:
            raise ValidationError(f"Duplicate ID: {transition.id!r}")
        seen.add(transition.id)


def score_transition_validity(score_transition_id: str | None, transition_ids: set[str]) -> None:
    """score_transition_id must point at an existing transition when set."""
    if score_transition_id is None:
        return
    if score_transition_id not in transition_ids:
        raise ValidationError(f"Unknown score transition: {score_transition_id!r}")


def arc_integrity(arcs: list[Arc], place_ids: set[str], transition_ids: set[str]) -> None:
    """Every arc must reference existing IDs and have positive weight."""
    all_ids = place_ids | transition_ids
    for arc in arcs:
        if arc.weight < 1:
            raise ValidationError(
                f"Arc {arc.source!r} -> {arc.target!r} has invalid weight: {arc.weight}"
            )
        if arc.source not in all_ids:
            raise ValidationError(f"Arc references unknown source: {arc.source!r}")
        if arc.target not in all_ids:
            raise ValidationError(f"Arc references unknown target: {arc.target!r}")


def arc_directionality(arcs: list[Arc], place_ids: set[str], transition_ids: set[str]) -> None:
    """Arcs must go place -> transition or transition -> place."""
    for arc in arcs:
        src_is_place = arc.source in place_ids
        tgt_is_place = arc.target in place_ids
        if src_is_place and tgt_is_place:
            raise ValidationError(f"Arc from place to place: {arc.source!r} -> {arc.target!r}")
        if not src_is_place and not tgt_is_place:
            raise ValidationError(
                f"Arc from transition to transition: {arc.source!r} -> {arc.target!r}"
            )


def marking_validity(marking: Marking, places: list[Place]) -> None:
    """Initial marking must reference existing places and respect capacity."""
    place_map = {p.id: p for p in places}
    for place_id, tokens in marking.tokens.items():
        if place_id not in place_map:
            raise ValidationError(f"Marking references unknown place: {place_id!r}")
        capacity = place_map[place_id].capacity
        if capacity is not None and len(tokens) > capacity:
            raise ValidationError(
                f"Marking exceeds capacity for place {place_id!r}: "
                f"{len(tokens)} tokens, capacity {capacity}"
            )


def reachability(
    arcs: list[Arc],
    marking: Marking,
    place_ids: set[str],
    transition_ids: set[str],
) -> None:
    """Every transition must be reachable from some initially-marked place."""
    if not transition_ids:
        return

    children: dict[str, list[str]] = {id: [] for id in place_ids | transition_ids}
    for arc in arcs:
        children[arc.source].append(arc.target)

    visited: set[str] = set()
    stack = [pid for pid in marking.tokens if pid in place_ids]
    while stack:
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)
        stack.extend(children[nid])

    unreachable = transition_ids - visited
    if unreachable:
        raise ValidationError(f"Unreachable transitions: {sorted(unreachable)}")
