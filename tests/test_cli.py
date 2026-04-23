from __future__ import annotations

from pathlib import Path

import peven.cli as cli_module
from peven.runtime.bootstrap import InstalledRuntime


def test_install_runtime_subcommand_resolves_the_runtime_and_reports_paths(
    monkeypatch, capsys
) -> None:
    installed = InstalledRuntime(
        julia_executable="/tmp/julia",
        julia_project=Path("/tmp/julia-env"),
    )
    monkeypatch.setattr(cli_module, "ensure_runtime_installed", lambda: installed)

    exit_code = cli_module.main(["install-runtime"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "julia_executable=/tmp/julia" in captured.out
    assert "julia_project=/tmp/julia-env" in captured.out


def test_peven_install_alias_reuses_the_install_runtime_subcommand(
    monkeypatch, capsys
) -> None:
    installed = InstalledRuntime(
        julia_executable="/tmp/julia",
        julia_project=Path("/tmp/julia-env"),
    )
    monkeypatch.setattr(cli_module, "ensure_runtime_installed", lambda: installed)

    exit_code = cli_module.install_runtime_main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "runtime_ready" in captured.out
