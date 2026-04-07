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
    file: Annotated[str, typer.Argument(help="Path to eval .py file")],
    concurrency: Annotated[int, typer.Option(help="Max concurrent transitions", min=1)] = 10,
    fuse: Annotated[int, typer.Option(help="Max total firings before stopping", min=1)] = 1000,
    trace: Annotated[bool, typer.Option(help="Show execution trace")] = False,
):
    """Run a Petri net evaluation."""
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

    run_id = save(results, file=file)
    typer.echo(f"Run {run_id}\n")
    render(results, trace=trace)
