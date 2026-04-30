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


def test_install_runtime_verbose_reports_resolution_step(monkeypatch, capsys) -> None:
    installed = InstalledRuntime(
        julia_executable="/tmp/julia",
        julia_project=Path("/tmp/julia-env"),
    )
    monkeypatch.setattr(cli_module, "ensure_runtime_installed", lambda: installed)

    exit_code = cli_module.main(["install-runtime", "--verbose"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "resolving_runtime" in captured.out
    assert "runtime_ready" in captured.out


def test_doctor_reports_runtime_paths_and_julia_version(monkeypatch, capsys) -> None:
    installed = InstalledRuntime(
        julia_executable="/tmp/julia",
        julia_project=Path("/tmp/julia-env"),
    )
    monkeypatch.setattr(cli_module, "ensure_runtime_installed", lambda: installed)
    monkeypatch.setattr(cli_module, "_read_julia_version", lambda _: "julia version 1.12.6")

    exit_code = cli_module.main(["doctor"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "python_package=ok" in captured.out
    assert "juliapkg=ok" in captured.out
    assert "julia_executable=/tmp/julia" in captured.out
    assert "julia_project=/tmp/julia-env" in captured.out
    assert "julia_version=julia version 1.12.6" in captured.out
    assert "doctor_ok" in captured.out


def test_doctor_reports_runtime_resolution_failure(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_module,
        "ensure_runtime_installed",
        lambda: (_ for _ in ()).throw(RuntimeError("missing Julia")),
    )

    exit_code = cli_module.main(["doctor"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "python_package=ok" in captured.out
    assert "juliapkg=error" in captured.out
    assert "missing Julia" in captured.out
    assert "doctor_ok" not in captured.out


def test_doctor_reports_julia_version_failure(monkeypatch, capsys) -> None:
    installed = InstalledRuntime(
        julia_executable="/tmp/julia",
        julia_project=Path("/tmp/julia-env"),
    )
    monkeypatch.setattr(cli_module, "ensure_runtime_installed", lambda: installed)
    monkeypatch.setattr(
        cli_module,
        "_read_julia_version",
        lambda _: (_ for _ in ()).throw(RuntimeError("bad julia")),
    )

    exit_code = cli_module.main(["doctor"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "juliapkg=ok" in captured.out
    assert "julia_version=error" in captured.out
    assert "bad julia" in captured.out
    assert "doctor_ok" not in captured.out
