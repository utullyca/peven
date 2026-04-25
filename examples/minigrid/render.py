"""CLI helper for rendering Minigrid JSONL sink output as an episode trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def render_episode(path: str | Path) -> str:
    env_actions: dict[int, dict[str, Any]] = {}
    planner_requests: dict[int, dict[str, Any]] = {}
    lines: list[str] = []

    for record in _records(path):
        transition_id = record.get("bundle", {}).get("transition_id")
        if record.get("kind") == "transition_started" and transition_id == "env_step":
            action = _payload(record.get("inputs_by_place", {}), "action")
            if action is not None:
                env_actions[int(record["firing_id"])] = action
            continue
        if record.get("kind") == "transition_started" and transition_id == "planner":
            request = _payload(record.get("inputs_by_place", {}), "plan_request")
            if request is not None:
                planner_requests[int(record["firing_id"])] = request
            continue

        if record.get("kind") != "transition_completed":
            if record.get("kind") == "run_finished":
                result = record.get("result", {})
                status = result.get("status")
                reason = result.get("terminal_reason")
                if status:
                    suffix = f" terminal_reason={reason}" if reason else ""
                    lines.append(f"run {status}{suffix}")
            continue

        outputs = record.get("outputs", {})
        if transition_id == "planner":
            plan = _payload(outputs, "plan")
            request = planner_requests.get(int(record["firing_id"]))
            if plan is not None and request is not None:
                lines.append(
                    f"turn {int(request['turn']):02d} planner -> {_format_plan(plan)}"
                )
        elif transition_id == "env_step":
            action = env_actions.get(int(record["firing_id"]), {})
            snapshot = _payload(outputs, "obs")
            done = _payload(outputs, "done")
            memory = _payload(outputs, "memory")
            if snapshot is not None:
                lines.append(_format_step(action, snapshot, memory, done))

    return "\n".join(lines)


def _records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _payload(bucketed: Any, place: str) -> dict[str, Any] | None:
    if not isinstance(bucketed, dict):
        return None
    bucket = bucketed.get(place) or []
    if not bucket:
        return None
    payload = bucket[0].get("payload")
    return payload if isinstance(payload, dict) else None


def _format_plan(plan: dict[str, Any]) -> str:
    if plan.get("instruction"):
        return str(plan["instruction"])
    if plan.get("advice"):
        return str(plan["advice"])
    subgoal = plan.get("subgoal", "none")
    complete_when = plan.get("complete_when", "none")
    avoid = plan.get("avoid", "none")
    return f"subgoal={subgoal}; complete_when={complete_when}; avoid={avoid}"


def _format_step(
    action: dict[str, Any],
    snapshot: dict[str, Any],
    memory: dict[str, Any] | None,
    done: dict[str, Any] | None,
) -> str:
    pos = snapshot["agent_pos"]
    reward = float(snapshot["reward"])
    line = (
        f"turn {int(action.get('turn', 0)):02d} {action.get('kind', '?'):<7} "
        f"step={int(snapshot['step_count']):02d} "
        f"pos=({int(pos[0])},{int(pos[1])}) "
        f"dir={int(snapshot['agent_dir'])} "
        f"key={'yes' if snapshot['has_key'] else 'no'} "
        f"reward={reward:.8g}"
    )
    if done is not None:
        line += f" score={float(done['score']):.8g} done"

    fog = memory.get("fog") if memory is not None else None
    memory_grid = fog if isinstance(fog, list) else snapshot["view"]
    agent_dir = int(snapshot["agent_dir"])
    agent_pos = (int(pos[0]), int(pos[1]))
    view_pos = _view_agent_pos(snapshot["view"])
    return (
        f"{line}\n"
        f"view:\n{_render_grid(snapshot['view'], agent_pos=view_pos, agent_dir=agent_dir)}\n"
        f"memory:\n{_render_grid(memory_grid, agent_pos=agent_pos, agent_dir=agent_dir)}"
    )


def _view_agent_pos(view: list[list[str]]) -> tuple[int, int]:
    return (len(view[0]) // 2, len(view) - 1)


def _render_grid(
    grid: list[list[str]],
    *,
    agent_pos: tuple[int, int] | None = None,
    agent_dir: int | None = None,
) -> str:
    rows = [list(row) for row in grid]
    if agent_pos is not None and agent_dir is not None:
        x, y = agent_pos
        if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
            rows[y][x] = ">v<^"[agent_dir % 4]
    return "\n".join("".join(row) for row in rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    print(render_episode(args.path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
