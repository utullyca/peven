"""Small command-line surface for runtime provisioning."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .runtime.bootstrap import InstalledRuntime, ensure_runtime_installed


def main(argv: Sequence[str] | None = None) -> int:
    """Run the peven CLI."""
    parser = argparse.ArgumentParser(prog="peven")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "install-runtime",
        help="install and precompile the Julia runtime dependencies",
    )

    args = parser.parse_args(argv)
    if args.command == "install-runtime":
        _print_runtime_ready(ensure_runtime_installed())
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


def install_runtime_main() -> int:
    """Run the direct install-runtime alias."""
    return main(["install-runtime"])


def _print_runtime_ready(installed: InstalledRuntime) -> None:
    print("runtime_ready")
    print(f"julia_executable={installed.julia_executable}")
    print(f"julia_project={installed.julia_project}")
