from __future__ import annotations

import math
import re

from peven.handoff.framing import DEFAULT_MAX_FRAME_BYTES
from peven.runtime.bootstrap import HANDSHAKE_TAG, PEVEN_VERSION, PROTOCOL_VERSION

from .conftest import require_pevenpy_project


def _julia_protocol_constants() -> dict[str, str | int]:
    source = (require_pevenpy_project() / "src" / "protocol.jl").read_text(encoding="utf-8")

    def read_string(name: str) -> str:
        match = re.search(rf'^const {name} = "([^"]+)"$', source, flags=re.MULTILINE)
        assert match is not None, f"missing Julia protocol constant {name}"
        return match.group(1)

    max_frame_match = re.search(r"^const MAX_FRAME_BYTES = ([^\n]+)$", source, flags=re.MULTILINE)
    assert max_frame_match is not None, "missing Julia protocol constant MAX_FRAME_BYTES"
    max_frame_expr = max_frame_match.group(1).strip()
    max_frame_parts = [part.strip() for part in max_frame_expr.split("*")]
    assert max_frame_parts
    assert all(part.isdigit() for part in max_frame_parts)

    return {
        "HANDSHAKE_TAG": read_string("HANDSHAKE_TAG"),
        "PROTOCOL_VERSION": read_string("PROTOCOL_VERSION"),
        "PEVEN_VERSION": read_string("PEVEN_VERSION"),
        "MAX_FRAME_BYTES": math.prod(int(part) for part in max_frame_parts),
    }


def test_python_and_julia_protocol_constants_stay_in_sync() -> None:
    julia = _julia_protocol_constants()

    assert julia == {
        "HANDSHAKE_TAG": HANDSHAKE_TAG,
        "PROTOCOL_VERSION": PROTOCOL_VERSION,
        "PEVEN_VERSION": PEVEN_VERSION,
        "MAX_FRAME_BYTES": DEFAULT_MAX_FRAME_BYTES,
    }
