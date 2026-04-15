"""Rich rendering for Petri net results and topology."""

from __future__ import annotations

import json
import re

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from peven.petri.schema import Net
from peven.petri.types import (
    GenerateOutput,
    JudgeOutput,
    RunResult,
    TokenSnapshot,
    TransitionResult,
)

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _status(value: str) -> Text:
    if value == "completed":
        return Text(value, style="green")
    if value == "failed":
        return Text(value, style="red")
    if value == "incomplete":
        return Text(value, style="yellow")
    return Text(value)


def _score_text(score: float | None) -> str:
    if score is None:
        return "—"
    return f"{score:.4f}"


def _sanitize_text(text: str) -> str:
    text = _ANSI_ESCAPE_RE.sub("", text)
    return "".join(ch if ch.isprintable() or ch.isspace() else " " for ch in text)


def _truncate(text: str, length: int = 60) -> str:
    text = _sanitize_text(text)
    text = " ".join(text.split())
    if len(text) > length:
        return text[: length - 1] + "…"
    return text


def _generic_output_summary(token) -> str:
    """Compact summary for non-built-in token outputs."""
    if isinstance(token, TokenSnapshot):
        payload = {k: v for k, v in token.payload.items() if k != "run_id"}
        text = payload.get("text")
        if isinstance(text, str):
            return _truncate(text)
        body = json.dumps(payload, sort_keys=True, default=str) if payload else ""
        return _truncate(f"{token.type_name} {body}".strip())

    text = getattr(token, "text", None)
    if isinstance(text, str):
        return _truncate(text)

    payload = token.model_dump(mode="json")
    payload.pop("run_id", None)
    body = json.dumps(payload, sort_keys=True, default=str) if payload else ""
    return _truncate(f"{type(token).__name__} {body}".strip())


def _output_summary(tr: TransitionResult) -> str:
    """One-line summary of a transition result's output."""
    if tr.error:
        return _truncate(tr.error)
    if isinstance(tr.output, JudgeOutput):
        return f"score={_score_text(tr.output.score)}"
    if isinstance(tr.output, GenerateOutput):
        return _truncate(tr.output.text)
    if tr.output is not None:
        return _generic_output_summary(tr.output)
    return "—"


def _trace_tree(result: RunResult) -> Tree:
    """Build a Rich tree for a single run's trace."""
    rid = _sanitize_text(result.run_id or "single")
    score_part = f" ({_score_text(result.score)})" if result.score is not None else ""
    reason_part = (
        f" [{result.terminal_reason}]"
        if result.terminal_reason and result.terminal_reason != "completed"
        else ""
    )
    label = Text.assemble(
        "Run ",
        Text(rid, style="bold"),
        " — ",
        _status(result.status),
        score_part,
        Text(reason_part, style="dim"),
    )
    tree = Tree(label)
    for tr in result.trace:
        icon = "✓" if tr.status == "completed" else "✗"
        style = "green" if tr.status == "completed" else "red"
        node = Text.assemble(
            Text(f"{icon} ", style=style),
            Text(tr.transition_id, style="bold"),
            "  ",
            _output_summary(tr),
        )
        branch = tree.add(node)
        # Expand rubric breakdown for judge outputs
        if isinstance(tr.output, JudgeOutput) and tr.output.report:
            for cr in tr.output.report:
                v_style = "green" if cr.verdict == "MET" else "red"
                cr_label = Text.assemble(
                    Text(cr.verdict, style=v_style),
                    f"  {_sanitize_text(cr.requirement)}",
                    Text(f"  ({_sanitize_text(cr.reason)})", style="dim"),
                )
                branch.add(cr_label)
    return tree


def render(
    results: list[RunResult],
    trace: bool = False,
    console: Console | None = None,
) -> None:
    """Render run results to the terminal.

    Single run: shows trace directly.
    Multiple runs: summary table + optional per-run traces.
    """
    con = console or Console()

    if not results:
        con.print("[dim]No results.[/dim]")
        return

    # Single run — just show trace
    if len(results) == 1 and not results[0].run_id:
        con.print(_trace_tree(results[0]))
        return

    # Summary table
    table = Table(show_header=True, header_style="bold")
    table.add_column("run_id")
    table.add_column("status")
    table.add_column("score", justify="right")
    table.add_column("error")

    for r in results:
        table.add_row(
            r.run_id or "—",
            _status(r.status),
            _score_text(r.score),
            _truncate(r.error, 40) if r.error else "",
        )
    con.print(table)

    # Stats line
    total = len(results)
    completed = sum(1 for r in results if r.status == "completed")
    failed = sum(1 for r in results if r.status == "failed")
    incomplete = total - completed - failed
    scores = [r.score for r in results if r.status == "completed" and r.score is not None]
    mean = sum(scores) / len(scores) if scores else None
    parts = [f"{total} runs", f"{completed} completed"]
    if failed:
        parts.append(f"{failed} failed")
    if incomplete:
        parts.append(f"{incomplete} incomplete")
    if mean is not None:
        label = "mean" if completed == total else "completed mean"
        parts.append(f"{label}: {mean:.4f}")
    con.print(f"[dim]{' · '.join(parts)}[/dim]")

    # Per-run traces
    if trace:
        con.print()
        for r in results:
            con.print(_trace_tree(r))
            con.print()


def render_net(net: Net, console: Console | None = None) -> None:
    """Render the net topology as a tree."""
    con = console or Console()

    place_ids = {p.id for p in net.places}

    # Build adjacency: place -> transitions, transition -> places
    p_to_t: dict[str, list[str]] = {p.id: [] for p in net.places}
    t_to_p: dict[str, list[str]] = {t.id: [] for t in net.transitions}

    for arc in net.arcs:
        if arc.source in place_ids:
            p_to_t[arc.source].append(arc.target)
        else:
            t_to_p[arc.source].append(arc.target)

    # Transition metadata
    t_meta = {}
    for t in net.transitions:
        parts = [t.executor]
        if t.when:
            parts.append("when")
        t_meta[t.id] = " ".join(parts)

    # Find root places (in initial marking or no incoming arcs)
    has_incoming = set()
    for arc in net.arcs:
        if arc.target in place_ids:
            has_incoming.add(arc.target)
    roots = [p.id for p in net.places if p.id not in has_incoming]
    if not roots:
        roots = list(net.initial_marking.tokens.keys())
    if not roots and net.places:
        roots = [net.places[0].id]

    n_places = len(net.places)
    n_transitions = len(net.transitions)
    tree = Tree(
        Text.assemble(
            "Net",
            Text(f" ({n_places} places, {n_transitions} transitions)", style="dim"),
        )
    )

    visited: set[str] = set()

    def _add_place(parent: Tree, pid: str) -> None:
        if pid in visited:
            parent.add(Text.assemble("[", Text(pid, style="cyan"), "] ↩"))
            return
        visited.add(pid)
        for tid in p_to_t.get(pid, []):
            _add_transition(parent, pid, tid)
        if not p_to_t.get(pid):
            parent.add(Text.assemble("[", Text(pid, style="cyan"), "]"))

    def _add_transition(parent: Tree, from_place: str, tid: str) -> None:
        meta = t_meta.get(tid, "")
        label = Text.assemble(
            "[",
            Text(from_place, style="cyan"),
            "] → ",
            Text(tid, style="yellow bold"),
            Text(f" ({meta})", style="dim") if meta else "",
        )
        targets = t_to_p.get(tid, [])
        if not targets:
            parent.add(label)
            return
        branch = parent.add(label)
        for target_pid in targets:
            _add_place(branch, target_pid)

    for root in roots:
        if root not in visited:
            _add_place(tree, root)

    con.print(tree)
