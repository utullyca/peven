"""peven review — review stored run results."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from peven.petri.render import render
from peven.petri.store import get, list_runs


console = Console()


def review(
    run_id: Annotated[str, typer.Argument(help="Run ID, 'all', or 'last N'")],
    n: Annotated[int, typer.Argument(help="Number of recent runs (with 'last')")] = 0,
    trace: Annotated[bool, typer.Option(help="Show execution trace")] = False,
):
    """Review stored runs.

    peven review all              Show every run
    peven review last 10          Show last N runs (default 10)
    peven review <run_id>         Show a specific run
    peven review <run_id> --trace Show a specific run with execution trace
    """
    if run_id == "all":
        _list_all(limit=None)
    elif run_id == "last":
        _list_all(limit=n or 10)
    else:
        _show_run(run_id, trace)


def _list_all(limit: int | None):
    runs = list_runs(limit=limit)
    if not runs:
        typer.echo("No runs found.")
        return

    table = Table(title="Recent Runs")
    table.add_column("ID", style="cyan")
    table.add_column("Time")
    table.add_column("File")
    table.add_column("Status")
    table.add_column("Score", justify="right")
    table.add_column("Results", justify="right")

    for r in runs:
        status_style = "green" if r.status == "completed" else "red"
        score = f"{r.score:.2f}" if r.score is not None else "—"
        table.add_row(
            r.id,
            r.timestamp.replace("T", " ").split(".")[0],
            r.file or "—",
            f"[{status_style}]{r.status}[/{status_style}]",
            score,
            str(r.result_count),
        )

    console.print(table)


def _show_run(run_id: str, trace: bool):
    data = get(run_id)
    if data is None:
        typer.echo(f"Run {run_id} not found.", err=True)
        raise typer.Exit(1)

    ts = data.timestamp.replace("T", " ").split(".")[0]
    typer.echo(f"Run {data.id}  |  {ts}  |  {data.file or '—'}\n")
    render(data.results, trace=trace)
