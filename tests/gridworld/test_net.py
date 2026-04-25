from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pytest


pytest.importorskip("minigrid")

from examples.minigrid.net import DoorKeyEnv, run_minigrid
from examples.minigrid.render import render_episode

import peven

from ..runtime.conftest import require_external_pevenpy_adapter_command


def test_spec_exposes_mover_planner_and_environment_step() -> None:
    spec = DoorKeyEnv.spec()
    transition_ids = {transition.id for transition in spec.transitions}
    assert transition_ids == {"mover", "planner", "env_step"}


def test_spec_places_are_core_environment_state() -> None:
    spec = DoorKeyEnv.spec()
    place_ids = {place.id for place in spec.places}
    assert place_ids == {
        "obs",
        "memory",
        "plan",
        "move_request",
        "action",
        "plan_request",
        "done",
    }


@pytest.mark.integration
@pytest.mark.ollama
@pytest.mark.slow
def test_doorkey_runs_through_peven_net_with_llm_tools() -> None:
    if os.environ.get("PEVEN_RUN_MINIGRID_OLLAMA") != "1":
        pytest.skip("set PEVEN_RUN_MINIGRID_OLLAMA=1 to run the Minigrid LLM test")

    command = require_external_pevenpy_adapter_command()
    log_dir = Path("logs") / "doorkey_8x8" / "net_seed0_test"
    log_dir.mkdir(parents=True, exist_ok=True)
    episode_log = log_dir / "episode.jsonl"
    episode_log.unlink(missing_ok=True)
    sink = peven.JSONLSink(episode_log)

    result = run_minigrid(command=command, seed=0, fuse=160, sink=sink)

    assert result.status == "completed", result
    assert result.terminal_reason is None
    actions: list[str] = []
    cells: set[tuple[int, int]] = set()
    for index, entry in enumerate(result.trace[:-1]):
        if entry.bundle.transition_id != "mover":
            continue
        if entry.outputs["plan_request"]:
            assert entry.outputs["action"] == []
            assert result.trace[index + 1].bundle.transition_id == "planner"
        else:
            assert entry.outputs["action"]
            action_payload = cast(dict[str, Any], entry.outputs["action"][0].payload)
            actions.append(str(action_payload["kind"]))
            assert result.trace[index + 1].bundle.transition_id == "env_step"
    for entry in result.trace:
        for token in entry.outputs.get("obs", []):
            payload = cast(dict[str, Any], token.payload)
            pos = cast(list[int], payload["agent_pos"])
            cells.add((int(pos[0]), int(pos[1])))

    memory_tokens = result.final_marking.get("memory", [])
    if memory_tokens:
        memory_payload = cast(dict[str, Any], memory_tokens[0].payload)
        assert memory_payload["planner_calls"] <= memory_payload["planner_limit"]

    done_tokens = result.final_marking.get("done", [])
    assert done_tokens
    done_payload = cast(dict[str, Any], done_tokens[0].payload)
    assert done_payload["terminated"] is True
    assert done_payload["reward"] > 0.0
    assert done_payload["score"] <= done_payload["reward"]
    assert done_payload["planner_calls"] <= done_payload["planner_limit"]
    assert {"forward", "pickup", "toggle"}.issubset(actions)
    assert len(cells) > 3

    rendered = render_episode(episode_log)
    assert "^" in rendered or ">" in rendered or "v" in rendered or "<" in rendered
    assert "done" in rendered
    assert "run completed" in rendered
