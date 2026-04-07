"""CLI entry point."""

import typer

from peven.cli.commands.review import review
from peven.cli.commands.run import run
from peven.cli.commands.validate import validate


app = typer.Typer(
    name="peven",
    no_args_is_help=True,
    help="Petri net engine for multi-agent evaluations.",
)
app.command()(run)
app.command()(validate)
app.command()(review)


if __name__ == "__main__":
    app()
