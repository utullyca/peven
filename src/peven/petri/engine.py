"""Petri net engine — async, event-driven, color-aware."""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from peven.petri.executors import get
from peven.petri.schema import JudgeConfig, Marking, Net, Token, Transition, ValidationError
from peven.petri.types import JudgeOutput, RunResult, TransitionResult
from peven.petri.validation import validate


logger = logging.getLogger(__name__)

EventHandler = Callable[[dict[str, Any]], Awaitable[None] | None]


async def _emit(event_handler: EventHandler | None, event: dict[str, Any]) -> None:
    """Dispatch a runtime event to an optional handler."""
    if event_handler is None:
        return
    result = event_handler(event)
    if inspect.isawaitable(result):
        await result


def _copy_marking(marking: Marking) -> Marking:
    """Deep copy a marking for external observers."""
    return marking.model_copy(deep=True)


def _sorted_in_flight(in_flight: set[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
    """Stable ordering for in-flight transition identifiers."""
    return sorted(in_flight, key=lambda item: (item[0], item[1] or ""))


def _trace_arcs(
    net: Net,
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, list[tuple[str, int]]]]:
    """Precompute input and output arcs per transition."""
    place_ids = {p.id for p in net.places}
    inputs: dict[str, list[tuple[str, int]]] = {t.id: [] for t in net.transitions}
    outputs: dict[str, list[tuple[str, int]]] = {t.id: [] for t in net.transitions}

    for arc in net.arcs:
        if arc.source in place_ids:
            inputs[arc.target].append((arc.source, arc.weight))
        else:
            outputs[arc.source].append((arc.target, arc.weight))

    return inputs, outputs


async def ready(
    net: Net,
    marking: Marking,
    inputs: dict[str, list[tuple[str, int]]],
    failed: set[str | None],
    in_flight: set[tuple[str, str | None]],
    guard_errors: dict[str | None, list[str]] | None = None,
    guard_failed: set[tuple[str, str | None]] | None = None,
    guard_failures: list[TransitionResult] | None = None,
) -> list[tuple[Transition, str | None]]:
    """Return (transition, run_id) pairs that can fire.

    Each transition is checked independently — parallel paths that never
    join are never gated on each other. Only at join points (a transition
    with multiple input places) does the run_id intersection narrow which
    runs can fire, ensuring both inputs come from the same evaluation.

    Guards can be sync or async callables (e.g. LLM judge gates).
    """
    if guard_errors is None:
        guard_errors = {}
    if guard_failed is None:
        guard_failed = set()
    if guard_failures is None:
        guard_failures = []
    ready: list[tuple[Transition, str | None]] = []

    for t in net.transitions:
        arcs = inputs[t.id]
        if not arcs:
            continue

        # Which run_ids can fire this transition?
        # A join (e.g. score with inputs pro_argument + con_argument) should
        # only fire when the SAME run_id has enough tokens in EVERY input place.
        # So: for each input place, find which run_ids have >= weight tokens,
        # then intersect across all input places. Only the survivors can fire.
        #
        # Example: score needs tokens from pro_argument AND con_argument.
        #   pro_argument has: [Token(run_id="abc"), Token(run_id="xyz")]
        #   con_argument has: [Token(run_id="abc")]
        #   → pro_argument sufficient: {"abc", "xyz"}
        #   → con_argument sufficient: {"abc"}
        #   → intersection: {"abc"} — only "abc" can fire, "xyz" is still waiting
        candidates: set[str | None] | None = None
        for place_id, weight in arcs:
            toks = marking.tokens.get(place_id, [])
            # Count tokens per run_id in this place
            counts: dict[str | None, int] = {}
            for tok in toks:
                counts[tok.run_id] = counts.get(tok.run_id, 0) + 1
            # Keep only run_ids with enough tokens to satisfy this arc
            sufficient = {rid for rid, cnt in counts.items() if cnt >= weight}
            # Intersect: first place seeds the set, subsequent places narrow it
            if candidates is None:
                candidates = sufficient
            else:
                candidates &= sufficient

        if not candidates:
            continue

        for rid in sorted(candidates, key=lambda value: value or ""):
            if rid in failed:
                continue
            if (t.id, rid) in in_flight:
                continue
            if (t.id, rid) in guard_failed:
                continue
            if t.when is not None:
                peek = []
                for place_id, weight in arcs:
                    count = 0
                    for tok in marking.tokens.get(place_id, []):
                        if count < weight and tok.run_id == rid:
                            peek.append(tok)
                            count += 1
                try:
                    result = t.when(peek)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if not result:
                        continue
                except Exception as e:
                    logger.warning(
                        "Guard on transition %r (run_id=%s) raised an exception",
                        t.id,
                        rid,
                        exc_info=True,
                    )
                    error = f"Guard on transition {t.id!r} raised {type(e).__name__}: {e}"
                    guard_failed.add((t.id, rid))
                    guard_errors.setdefault(rid, []).append(error)
                    guard_failures.append(
                        TransitionResult(
                            transition_id=t.id,
                            status="failed",
                            error=error,
                            run_id=rid,
                        )
                    )
                    continue
            ready.append((t, rid))

    return ready


def try_consume(
    marking: Marking,
    transition: Transition,
    inputs: dict[str, list[tuple[str, int]]],
    run_id: str | None,
) -> tuple[Marking, list[Token], dict[str, list[Token]]] | None:
    """Atomically remove input tokens for (transition, run_id) from marking.

    Returns (new_marking, consumed_flat, consumed_by_place).
    consumed_flat is for passing to executors.
    consumed_by_place maps place_id -> tokens, used for retry re-deposit.

    Returns None if the current marking no longer has enough tokens to satisfy
    every required input arc for this run_id.
    """
    # Copy marking — don't mutate the original
    new_tokens = {pid: list(toks) for pid, toks in marking.tokens.items()}
    consumed: list[Token] = []  # flat list passed to the executor as input
    consumed_by_place: dict[str, list[Token]] = {}  # for retry re-deposit

    for place_id, weight in inputs[transition.id]:
        available = sum(1 for tok in new_tokens.get(place_id, []) if tok.run_id == run_id)
        if available < weight:
            return None

        taken = 0
        remaining = []
        place_consumed: list[Token] = []
        for tok in new_tokens.get(place_id, []):
            # Take tokens matching this run_id until we satisfy the arc weight
            if taken < weight and tok.run_id == run_id:
                consumed.append(tok)
                place_consumed.append(tok)
                taken += 1
            else:
                remaining.append(tok)
        new_tokens[place_id] = remaining
        if place_consumed:
            consumed_by_place[place_id] = place_consumed

    return Marking(tokens=new_tokens), consumed, consumed_by_place


def consume(
    marking: Marking,
    transition: Transition,
    inputs: dict[str, list[tuple[str, int]]],
    run_id: str | None,
) -> tuple[Marking, list[Token], dict[str, list[Token]]]:
    """Remove input tokens for (transition, run_id) from marking.

    Raises ValidationError if the marking cannot satisfy every input arc.
    """
    consumed = try_consume(marking, transition, inputs, run_id)
    if consumed is None:
        raise ValidationError(
            f"Transition {transition.id!r} is no longer enabled for run_id={run_id!r}"
        )
    return consumed


def deposit(
    marking: Marking,
    transition: Transition,
    outputs: dict[str, list[tuple[str, int]]],
    token: Token,
    capacity: dict[str, int | None],
) -> Marking:
    """Place output token into all output places. Returns new marking."""
    # Copy marking — mirror of consume
    new_tokens = {pid: list(toks) for pid, toks in marking.tokens.items()}

    # Place the output token into each output place, weight times (usually 1)
    for place_id, weight in outputs[transition.id]:
        if place_id not in new_tokens:
            new_tokens[place_id] = []
        for _ in range(weight):
            # Enforce place capacity if set (e.g. bounded buffers)
            cap = capacity.get(place_id)
            if cap is not None and len(new_tokens[place_id]) >= cap:
                raise ValidationError(
                    f"Firing {transition.id!r} would exceed capacity "
                    f"for place {place_id!r}: capacity {cap}"
                )
            new_tokens[place_id].append(token)

    return Marking(tokens=new_tokens)


def _seed_run_ids(marking: Marking) -> list[str | None]:
    """Return distinct run_ids in first-seen order."""
    seen: set[str | None] = set()
    ordered: list[str | None] = []
    for tokens in marking.tokens.values():
        for token in tokens:
            if token.run_id in seen:
                continue
            seen.add(token.run_id)
            ordered.append(token.run_id)
    return ordered


def _resolve_score_transition(net: Net) -> tuple[str | None, bool]:
    """Resolve which transition supplies the scalar run score and whether it was inferred."""
    if net.score_transition_id is not None:
        return net.score_transition_id, False

    judge_ids = [t.id for t in net.transitions if isinstance(t.config, JudgeConfig)]
    if len(judge_ids) == 1:
        return judge_ids[0], True
    return None, False


def _score_values(trace: list[TransitionResult], transition_id: str | None) -> list[float]:
    """Collect scores emitted by the designated judge transition."""
    if transition_id is None:
        return []

    values: list[float] = []
    for result in trace:
        if result.transition_id == transition_id and isinstance(result.output, JudgeOutput):
            values.append(result.output.score)
    return values


def _score(trace: list[TransitionResult], transition_id: str | None) -> float | None:
    """Extract the mean score from the designated judge transition."""
    values = _score_values(trace, transition_id)
    if not values:
        return None
    return sum(values) / len(values)


def _has_score(trace: list[TransitionResult], transition_id: str | None) -> bool:
    """Return True if the designated score transition emitted a JudgeOutput."""
    return bool(_score_values(trace, transition_id))


def _run_tokens(marking: Marking, run_id: str | None) -> list[tuple[str, Token]]:
    """Return every remaining token for a run_id with its place."""
    tokens: list[tuple[str, Token]] = []
    for place_id, place_tokens in marking.tokens.items():
        for token in place_tokens:
            if token.run_id == run_id:
                tokens.append((place_id, token))
    return tokens


def _run_completed(
    marking: Marking,
    run_id: str | None,
    source_places: set[str],
) -> bool:
    """A run is complete when no active tokens remain outside sink places."""
    remaining = _run_tokens(marking, run_id)
    if not remaining:
        return True
    return all(place_id not in source_places for place_id, _ in remaining)


async def _execute(
    net: Net,
    fuse: int,
    max_concurrency: int,
    score_transition_id: str | None,
    event_handler: EventHandler | None = None,
) -> tuple[
    list[TransitionResult],
    set[str | None],
    Marking,
    dict[str | None, list[str]],
    dict[str | None, str],
    set[str | None],
]:
    """Run the engine loop.

    The core loop is five steps:

        1. ready()   — which transitions can fire right now?
        2. consume() — remove input tokens so nothing else grabs them
        3. execute   — call the LLM (agent or judge)
        4. deposit() — place the output token into output places
        5. goto 1    — new tokens may have enabled new transitions

    Steps 1-4 are interleaved, not batched — multiple transitions run
    concurrently and we react as soon as any one finishes (FIRST_COMPLETED)
    rather than waiting for all of them. This single-threaded central loop
    owns all marking mutations; executors are pure functions with no shared
    state, so no locks are needed.
    """
    # -- Setup: validate, build lookup tables, initialize state ---------------
    validate(net)
    inputs_idx, outputs_idx = _trace_arcs(net)  # transition → [(place_id, weight)]
    cap = {p.id: p.capacity for p in net.places}

    marking = net.initial_marking
    trace: list[TransitionResult] = []
    firings = 0
    failed_run_ids: set[str | None] = set()
    guard_errors: dict[str | None, list[str]] = {}
    guard_failed: set[tuple[str, str | None]] = set()
    guard_failures: list[TransitionResult] = []
    executor_errors: dict[str | None, str] = {}
    fuse_exhausted_run_ids: set[str | None] = set()

    sem = asyncio.Semaphore(max_concurrency)
    pending: dict[asyncio.Task, tuple[Transition, str | None, list[Token]]] = {}
    in_flight: set[tuple[str, str | None]] = set()  # so ready() skips these
    retry_counts: dict[tuple[str, str | None], int] = {}

    # -- Spawn: check what's ready, consume tokens, kick off tasks ----------
    async def _spawn():
        nonlocal marking, firings

        guard_failure_count = len(guard_failures)
        primed = await ready(
            net,
            marking,
            inputs_idx,
            failed_run_ids,
            in_flight,
            guard_errors,
            guard_failed,
            guard_failures,
        )
        for failure in guard_failures[guard_failure_count:]:
            trace.append(failure)
            await _emit(
                event_handler,
                {
                    "type": "transition_failed",
                    "transition_id": failure.transition_id,
                    "run_id": failure.run_id,
                    "error": failure.error,
                    "retrying": False,
                    "marking": _copy_marking(marking),
                    "in_flight": _sorted_in_flight(in_flight),
                },
            )
        if not primed:
            return

        available_slots = max_concurrency - len(pending)
        if available_slots <= 0:
            return

        if firings >= fuse:
            fuse_exhausted_run_ids.update(rid for _, rid in primed)
            return

        for idx, (t, rid) in enumerate(primed):
            if available_slots <= 0:
                return
            if firings >= fuse:
                fuse_exhausted_run_ids.update(pending_rid for _, pending_rid in primed[idx:])
                return

            # Consume eagerly — remove tokens NOW so no other transition grabs them
            # while the LLM call is in flight
            consumed = try_consume(marking, t, inputs_idx, rid)
            if consumed is None:
                continue
            marking, consumed_tokens, consumed_by_place = consumed
            firings += 1
            in_flight.add((t.id, rid))
            await _emit(
                event_handler,
                {
                    "type": "transition_started",
                    "transition_id": t.id,
                    "run_id": rid,
                    "inputs": [token.model_copy(deep=True) for token in consumed_tokens],
                    "marking": _copy_marking(marking),
                    "in_flight": _sorted_in_flight(in_flight),
                },
            )

            # Default args (t=t, etc.) capture the current loop values
            async def _exec(t=t, consumed_tokens=consumed_tokens, rid=rid):
                async with sem:
                    executor = get(t.executor)
                    output = await executor.execute(consumed_tokens, t.config)
                    if not isinstance(output, Token):
                        raise TypeError(
                            f"Executor {t.executor!r} returned {type(output).__name__}, expected Token"
                        )
                    if score_transition_id == t.id and not isinstance(output, JudgeOutput):
                        raise TypeError(
                            f"Score transition {t.id!r} returned {type(output).__name__}, "
                            "expected JudgeOutput"
                        )
                    output.run_id = rid
                    return output

            task = asyncio.create_task(_exec())
            pending[task] = (t, rid, consumed_by_place)
            available_slots -= 1

    # -- Initial spawn ------------------------------------------------------
    await _spawn()

    # -- Main loop: react to completions, deposit results, spawn new --------
    while pending:
        # Wait for ANY task to finish — event-driven, not batch
        done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            t, rid, consumed_by_place = pending.pop(task)
            in_flight.discard((t.id, rid))

            try:
                # Success — deposit output token into output places
                output_token = task.result()
                marking = deposit(marking, t, outputs_idx, output_token, cap)

                result = TransitionResult(
                    transition_id=t.id,
                    status="completed",
                    output=output_token,
                    run_id=rid,
                )
                await _emit(
                    event_handler,
                    {
                        "type": "transition_completed",
                        "transition_id": t.id,
                        "run_id": rid,
                        "output": output_token.model_copy(deep=True),
                        "marking": _copy_marking(marking),
                        "in_flight": _sorted_in_flight(in_flight),
                    },
                )
            except Exception as e:
                key = (t.id, rid)
                retry_counts[key] = retry_counts.get(key, 0) + 1

                if retry_counts[key] <= t.retries:
                    # Put consumed tokens back where they came from and try again
                    new_tokens = {pid: list(ts) for pid, ts in marking.tokens.items()}
                    for place_id, toks in consumed_by_place.items():
                        new_tokens.setdefault(place_id, []).extend(toks)
                    marking = Marking(tokens=new_tokens)
                    await _emit(
                        event_handler,
                        {
                            "type": "transition_failed",
                            "transition_id": t.id,
                            "run_id": rid,
                            "error": str(e),
                            "retrying": True,
                            "marking": _copy_marking(marking),
                            "in_flight": _sorted_in_flight(in_flight),
                        },
                    )
                    continue

                # Retries exhausted — record failure
                result = TransitionResult(
                    transition_id=t.id,
                    status="failed",
                    error=str(e),
                    run_id=rid,
                )
                # Poison the entire run_id so no more transitions fire for it
                failed_run_ids.add(rid)
                executor_errors.setdefault(rid, str(e))
                await _emit(
                    event_handler,
                    {
                        "type": "transition_failed",
                        "transition_id": t.id,
                        "run_id": rid,
                        "error": str(e),
                        "retrying": False,
                        "marking": _copy_marking(marking),
                        "in_flight": _sorted_in_flight(in_flight),
                    },
                )

            trace.append(result)

        # New tokens may have been deposited — check for newly enabled transitions
        await _spawn()

    # Loop exits when nothing is pending and nothing new can spawn (fixed point)
    return trace, failed_run_ids, marking, guard_errors, executor_errors, fuse_exhausted_run_ids


async def execute(
    net: Net,
    rows: list[Token] | None = None,
    place: str | None = None,
    fuse: int = 1000,
    max_concurrency: int = 10,
    event_handler: EventHandler | None = None,
) -> list[RunResult]:
    """Run the net. Returns one RunResult per run.

    Single: execute(net) -> [RunResult(run_id=None)]
    Batch:  execute(net, rows=[...], place="in") -> [RunResult per row]
    """
    if rows is not None:
        if place is None:
            raise ValueError("place is required when rows are provided")

        # Stamp each row with a unique run_id, merge into marking
        tokens = {pid: list(toks) for pid, toks in net.initial_marking.tokens.items()}
        if place not in tokens:
            tokens[place] = []
        for row in rows:
            rid = uuid.uuid4().hex[:12]
            tokens[place].append(row.model_copy(update={"run_id": rid}))

        net = net.model_copy(update={"initial_marking": Marking(tokens=tokens)})

    seeded_run_ids = _seed_run_ids(net.initial_marking)
    if not seeded_run_ids:
        return []

    score_transition_id, inferred_score_transition = _resolve_score_transition(net)
    await _emit(
        event_handler,
        {
            "type": "run_started",
            "run_ids": list(seeded_run_ids),
            "score_transition_id": score_transition_id,
            "score_transition_inferred": inferred_score_transition,
            "marking": _copy_marking(net.initial_marking),
        },
    )
    trace, failed, marking, guard_errors, executor_errors, fuse_exhausted = await _execute(
        net,
        fuse,
        max_concurrency,
        score_transition_id,
        event_handler=event_handler,
    )
    place_ids = {p.id for p in net.places}
    source_places = {arc.source for arc in net.arcs if arc.source in place_ids}

    # Partition trace by run_id — works for single (run_id=None) and batch
    by_rid: dict[str | None, list[TransitionResult]] = {}
    for r in trace:
        by_rid.setdefault(r.run_id, []).append(r)

    results: list[RunResult] = []
    for rid in seeded_run_ids:
        row_trace = by_rid.get(rid, [])
        score = _score(row_trace, score_transition_id)
        if rid in failed:
            result = RunResult(
                run_id=rid,
                status="failed",
                terminal_reason="executor_failed",
                score=score,
                error=executor_errors.get(rid),
                trace=row_trace,
            )
            results.append(result)
            await _emit(
                event_handler,
                {
                    "type": "run_finished",
                    "run_id": rid,
                    "result": result.model_copy(deep=True),
                },
            )
            continue

        if _run_completed(marking, rid, source_places):
            if (
                score_transition_id is not None
                and not inferred_score_transition
                and not _has_score(row_trace, score_transition_id)
            ):
                result = RunResult(
                    run_id=rid,
                    status="incomplete",
                    terminal_reason="missing_score",
                    score=None,
                    error=(
                        f"Run reached completion without emitting a score from "
                        f"transition {score_transition_id!r}"
                    ),
                    trace=row_trace,
                )
            else:
                result = RunResult(
                    run_id=rid,
                    status="completed",
                    terminal_reason="completed",
                    score=score,
                    trace=row_trace,
                )
            results.append(result)
            await _emit(
                event_handler,
                {
                    "type": "run_finished",
                    "run_id": rid,
                    "result": result.model_copy(deep=True),
                },
            )
            continue

        if rid in fuse_exhausted:
            terminal_reason = "fuse_exhausted"
            error = f"Execution stopped after reaching fuse={fuse}"
        elif rid in guard_errors:
            terminal_reason = "guard_error"
            error = "; ".join(guard_errors[rid])
        else:
            terminal_reason = "no_enabled_transition"
            error = "No enabled transitions for remaining tokens"

        result = RunResult(
            run_id=rid,
            status="incomplete",
            terminal_reason=terminal_reason,
            score=score,
            error=error,
            trace=row_trace,
        )
        results.append(result)
        await _emit(
            event_handler,
            {
                "type": "run_finished",
                "run_id": rid,
                "result": result.model_copy(deep=True),
            },
        )

    return results
