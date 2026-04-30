"""Small command-line surface for runtime provisioning."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence

from .runtime.bootstrap import InstalledRuntime, ensure_runtime_installed


def main(argv: Sequence[str] | None = None) -> int:
    """Run the peven CLI."""
    parser = argparse.ArgumentParser(prog="peven")
    subparsers = parser.add_subparsers(dest="command", required=True)

    install_parser = subparsers.add_parser(
        "install-runtime",
        help="install and precompile the Julia runtime dependencies",
    )
    install_parser.add_argument(
        "--verbose",
        action="store_true",
        help="print runtime resolution progress",
    )
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="check Python and Julia runtime wiring",
    )
    doctor_parser.add_argument(
        "--verbose",
        action="store_true",
        help="print runtime resolution progress",
    )

    args = parser.parse_args(argv)
    if args.command == "install-runtime":
        return _install_runtime(verbose=args.verbose)
    if args.command == "doctor":
        return _doctor(verbose=args.verbose)
    parser.error(f"unknown command: {args.command}")
    return 2


def install_runtime_main() -> int:
    """Run the direct install-runtime alias."""
    return main(["install-runtime"])


def _install_runtime(*, verbose: bool = False) -> int:
    if verbose:
        print("resolving_runtime")
    _print_runtime_ready(ensure_runtime_installed())
    return 0


def _doctor(*, verbose: bool = False) -> int:
    print("python_package=ok")
    if verbose:
        print("resolving_runtime")
    try:
        installed = ensure_runtime_installed()
    except Exception as exc:
        print(f"juliapkg=error {exc}")
        return 1
    print("juliapkg=ok")
    print(f"julia_executable={installed.julia_executable}")
    print(f"julia_project={installed.julia_project}")
    try:
        print(f"julia_version={_read_julia_version(installed.julia_executable)}")
    except Exception as exc:
        print(f"julia_version=error {exc}")
        return 1
    print("doctor_ok")
    return 0


def _read_julia_version(julia_executable: str) -> str:
    result = subprocess.run(
        [julia_executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _print_runtime_ready(installed: InstalledRuntime) -> None:
    print("runtime_ready")
    print(f"julia_executable={installed.julia_executable}")
    print(f"julia_project={installed.julia_project}")
