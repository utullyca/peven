"""Load a Net from a trusted Python eval file by convention.

Discovers:
    net   — required, a Net instance
    rows  — optional, list[Token] for batch
    place — optional, str (required if rows is set)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from peven.petri.schema import Net, Token


class LoadError(Exception):
    """Raised when the eval file can't be loaded or is missing required objects."""


def load(path: str) -> tuple[Net, list[Token] | None, str | None]:
    """Import and execute a trusted eval file, then extract net, rows, place."""
    p = Path(path).resolve()
    if not p.exists():
        raise LoadError(f"File not found: {path}")
    if not p.suffix == ".py":
        raise LoadError(f"Expected a .py file, got: {p.suffix}")

    module_name = f"_peven_eval_{p.stem}"
    spec = importlib.util.spec_from_file_location(module_name, p)
    if spec is None or spec.loader is None:
        raise LoadError(f"Cannot load module from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise LoadError(f"Error executing {path}: {e}") from e
    finally:
        sys.modules.pop(module_name, None)

    net = getattr(module, "net", None)
    if net is None:
        raise LoadError(f"No 'net' variable found in {path}")
    if not isinstance(net, Net):
        raise LoadError(f"'net' must be a Net instance, got {type(net).__name__}")

    rows = getattr(module, "rows", None)
    place = getattr(module, "place", None)

    if rows is not None and place is None:
        raise LoadError("'rows' requires a 'place' variable")

    return net, rows, place
