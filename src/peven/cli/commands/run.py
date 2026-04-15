"""peven run — execute an eval file."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer

from peven.cli.loader import LoadError, load
from peven.petri.engine import execute as engine_run
from peven.petri.render import render
from peven.petri.store import save


def run(
    file: Annotated[
        str,
        typer.Argument(help="Path to a trusted eval .py file (executed as Python)"),
    ],
    concurrency: Annotated[
        int,
        typer.Option(help="Max concurrent transition executions", min=1),
    ] = 10,
    fuse: Annotated[int, typer.Option(help="Max total firings before stopping", min=1)] = 1000,
    trace: Annotated[bool, typer.Option(help="Show execution trace")] = False,
    save_run: Annotated[
        bool,
        typer.Option("--save/--no-save", help="Persist results to ~/.peven/runs.db"),
    ] = True,
):
    """Run a trusted Petri net eval file."""
    try:
        net, rows, place = load(file)
    except LoadError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None

    try:
        results = asyncio.run(
            engine_run(net, rows=rows, place=place, fuse=fuse, max_concurrency=concurrency)
        )
    except Exception as e:
        typer.echo(f"Execution failed: {e}", err=True)
        raise typer.Exit(1) from None

    if save_run:
        run_id = save(results, file=file)
        typer.echo(f"Run {run_id}\n")
    else:
        typer.echo("Run not saved (--no-save)\n")
    render(results, trace=trace)
