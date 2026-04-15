"""peven validate — check an eval file and show its structure."""

from __future__ import annotations

from typing import Annotated

import typer

from peven.cli.loader import LoadError, load
from peven.petri.render import render_net
from peven.petri.schema import ValidationError
from peven.petri.validation import validate as validate_net


def validate(
    file: Annotated[
        str,
        typer.Argument(help="Path to a trusted eval .py file (executed as Python)"),
    ],
):
    """Validate a trusted eval file and show its topology."""
    try:
        net, _, _ = load(file)
    except LoadError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None

    try:
        validate_net(net)
    except ValidationError as e:
        typer.echo(f"Validation failed: {e}", err=True)
        raise typer.Exit(1) from None

    typer.echo("Valid.\n")
    render_net(net)
