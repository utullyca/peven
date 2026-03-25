"""Petri net engine — async, event-driven, color-aware."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

from peven.petri.executors import get
from peven.petri.schema import Marking, Net, Token, Transition, ValidationError
from peven.petri.types import GenerateOutput, JudgeOutput, RunResult, TransitionResult
from peven.petri.validation import validate

logger = logging.getLogger(__name__)


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
    failed: set[str],
    in_flight: set[tuple[str, Optional[str]]],
) -> list[tuple[Transition, Optional[str]]]:
    """Return (transition, run_id) pairs that can fire.

    Each transition is checked independently — parallel paths that never
    join are never gated on each other. Only at join points (a transition
    with multiple input places) does the run_id intersection narrow which
    runs can fire, ensuring both inputs come from the same evaluation.

    Guards can be sync or async callables (e.g. LLM judge gates).
    """
    ready: list[tuple[Transition, Optional[str]]] = []

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
        candidates: Optional[set[Optional[str]]] = None
        for place_id, weight in arcs:
            toks = marking.tokens.get(place_id, [])
            # Count tokens per run_id in this place
            counts: dict[Optional[str], int] = {}
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

        for rid in candidates:
            if rid is not None and rid in failed:
                continue
            if (t.id, rid) in in_flight:
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
                except Exception:
                    logger.warning(
                        "Guard on transition %r (run_id=%s) raised an exception",
                        t.id,
                        rid,
                        exc_info=True,
                    )
                    continue
            ready.append((t, rid))

    return ready


def consume(
    marking: Marking,
    transition: Transition,
    inputs: dict[str, list[tuple[str, int]]],
    run_id: Optional[str],
) -> tuple[Marking, list[Token], dict[str, list[Token]]]:
    """Remove input tokens for (transition, run_id) from marking.

    Returns (new_marking, consumed_flat, consumed_by_place).
    consumed_flat is for passing to executors.
    consumed_by_place maps place_id -> tokens, used for retry re-deposit.
    """
    # Copy marking — don't mutate the original
    new_tokens = {pid: list(toks) for pid, toks in marking.tokens.items()}
    consumed: list[Token] = []  # flat list passed to the executor as input
    consumed_by_place: dict[str, list[Token]] = {}  # for retry re-deposit

    for place_id, weight in inputs[transition.id]:
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


def deposit(
    marking: Marking,
    transition: Transition,
    outputs: dict[str, list[tuple[str, int]]],
    token: Token,
    capacity: dict[str, Optional[int]],
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


def _score(trace: list[TransitionResult]) -> Optional[float]:
    """Extract score from the last judge result in a trace."""
    last = None
    for r in trace:
        if isinstance(r.output, JudgeOutput):
            last = r.output.score
    return last


async def _execute(
    net: Net,
    fuse: int,
    max_concurrency: int,
) -> tuple[list[TransitionResult], set[str]]:
    """Run the engine loop. Returns (trace, failed_run_ids).

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
    failed_run_ids: set[str] = set()

    sem = asyncio.Semaphore(max_concurrency)  # limit concurrent LLM calls
    pending: dict[asyncio.Task, tuple[Transition, Optional[str], list[Token]]] = {}
    in_flight: set[tuple[str, Optional[str]]] = set()  # so ready() skips these
    retry_counts: dict[tuple[str, Optional[str]], int] = {}

    # -- Spawn: check what's ready, consume tokens, kick off tasks ----------
    async def _spawn():
        nonlocal marking, firings

        primed = await ready(net, marking, inputs_idx, failed_run_ids, in_flight)
        for t, rid in primed:
            if firings >= fuse:
                break

            # Consume eagerly — remove tokens NOW so no other transition grabs them
            # while the LLM call is in flight
            marking, consumed, consumed_by_place = consume(marking, t, inputs_idx, rid)
            firings += 1

            # Default args (t=t, etc.) capture the current loop values
            async def _exec(t=t, consumed=consumed, rid=rid):
                async with sem:
                    executor = get(t.executor)
                    output = await executor.execute(consumed, t.config)
                    output.run_id = rid
                    return output

            task = asyncio.create_task(_exec())
            pending[task] = (t, rid, consumed_by_place)
            in_flight.add((t.id, rid))

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

                output = (
                    output_token
                    if isinstance(output_token, (GenerateOutput, JudgeOutput))
                    else None
                )
                result = TransitionResult(
                    transition_id=t.id,
                    status="completed",
                    output=output,
                    run_id=rid,
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
                    continue

                # Retries exhausted — record failure
                result = TransitionResult(
                    transition_id=t.id,
                    status="failed",
                    error=str(e),
                    run_id=rid,
                )
                # Poison the entire run_id so no more transitions fire for it
                if rid is not None:
                    failed_run_ids.add(rid)

            trace.append(result)

        # New tokens may have been deposited — check for newly enabled transitions
        await _spawn()

    # Loop exits when nothing is pending and nothing new can spawn (fixed point)
    return trace, failed_run_ids


async def execute(
    net: Net,
    rows: Optional[list[Token]] = None,
    place: Optional[str] = None,
    fuse: int = 1000,
    max_concurrency: int = 10,
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

    trace, failed = await _execute(net, fuse, max_concurrency)

    # Partition trace by run_id — works for single (run_id=None) and batch
    by_rid: dict[Optional[str], list[TransitionResult]] = {}
    for r in trace:
        by_rid.setdefault(r.run_id, []).append(r)

    return [
        RunResult(
            run_id=rid,
            status="failed"
            if rid is not None and rid in failed
            else ("failed" if any(r.status == "failed" for r in row_trace) else "completed"),
            score=_score(row_trace),
            error=next((r.error for r in row_trace if r.error), None),
            trace=row_trace,
        )
        for rid, row_trace in by_rid.items()
    ]
