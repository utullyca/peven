"""Pydantic AI planner and mover runners for the DoorKey net."""

from __future__ import annotations

from typing import Any

from examples.minigrid.tools import ACTION_OUTPUTS, action_kind
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from peven.integrations.pydantic_ai import event_stream_handler


mover = Agent(
    OpenAIChatModel(
        "gemma4:e4b",
        provider=OllamaProvider(base_url="http://127.0.0.1:11434/v1"),
    ),
    output_type=ACTION_OUTPUTS,
    output_retries=2,
    instructions=(
        "You are the MiniGrid DoorKey mover. Choose exactly one action tool. "
        "The current inventory_key, tile_ahead, planner advice, and planner_calls_remaining "
        "in the prompt are authoritative. Never call ask_planner when planner_calls_remaining "
        "is 0. Only call pick_up_key when tile_ahead is key. Only call open_door "
        "when tile_ahead is closed_door; move_forward through open_door. "
        "Never move_forward into wall, unseen, key, or closed_door. "
        "If the current objective is tile_left or tile_right, turn toward it first."
    ),
    model_settings={
        "temperature": 0.2,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "max_tokens": 64,
        "seed": 1,
        "extra_body": {"reasoning_effort": "none"},
    },
)

planner = Agent(
    OpenAIChatModel(
        "deepseek-r1:7b",
        provider=OllamaProvider(base_url="http://127.0.0.1:11434/v1"),
    ),
    output_type=str,
    model_settings={
        "temperature": 0.1,
        "max_tokens": 256,
        "seed": 1,
        "extra_body": {"reasoning_effort": "low"},
    },
)


async def choose_move(
    ctx: Any,
    *,
    obs: dict[str, Any],
    memory: dict[str, Any],
    plan: dict[str, Any] | None,
) -> dict[str, str]:
    remaining = int(memory["planner_limit"]) - int(memory["planner_calls"])
    result = await mover.run(
        _mover_prompt(obs=obs, plan=plan, remaining=remaining),
        output_type=ACTION_OUTPUTS,
        event_stream_handler=event_stream_handler(ctx, model="gemma4:e4b"),
    )
    return {"kind": action_kind(result.output)}


async def make_plan(
    ctx: Any,
    *,
    obs: dict[str, Any],
    memory: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, str]:
    result = await planner.run(
        _planner_prompt(obs=obs, memory=memory),
        event_stream_handler=event_stream_handler(ctx, model="deepseek-r1:7b"),
    )
    return {
        "advice": _plan_advice(str(result.output)),
        "model": "deepseek-r1:7b",
    }


def _mover_prompt(
    *,
    obs: dict[str, Any],
    plan: dict[str, Any] | None,
    remaining: int,
) -> str:
    ahead = _tile_name(_tile_ahead(obs))
    left = _tile_name(_tile_left(obs))
    right = _tile_name(_tile_right(obs))
    advice = None if plan is None else str(plan.get("advice") or "")
    planner_advice = "" if advice is None else _planner_advice(advice)
    action_basis = (
        "live state.\n"
        if advice is None
        else "live state or planner advice.\n"
    )
    return (
        "Choose one MiniGrid action.\n"
        f"Mission: {obs['mission']}\n"
        f"inventory_key={'yes' if obs['has_key'] else 'no'}\n"
        f"tile_ahead={ahead}\n"
        f"tile_left={left}\n"
        f"tile_right={right}\n"
        f"local_objective={_local_objective(obs)}\n"
        f"planner_calls_remaining={remaining}\n"
        f"{planner_advice}"
        "DoorKey objective order: get the key, open the closed door, then reach the goal.\n"
        "Available tools: move_forward, turn_left, turn_right, pick_up_key, open_door, ask_planner.\n"
        "Current inventory_key and tile_ahead are authoritative.\n"
        f"{_planning_rule(advice=advice, remaining=remaining)}"
        "Tool preconditions:\n"
        "- pick_up_key works only with inventory_key=no and tile_ahead=key.\n"
        "- open_door works only with inventory_key=yes and tile_ahead=closed_door.\n"
        "- move_forward works only with tile_ahead=empty, tile_ahead=open_door, or tile_ahead=goal.\n"
        "- if local_objective says tile_left, use turn_left.\n"
        "- if local_objective says tile_right, use turn_right.\n"
        "- interact only after the target becomes tile_ahead.\n"
        "- if tile_ahead=wall or tile_ahead=unseen, turn instead of move_forward.\n"
        "- turn_left and turn_right always rotate in place.\n"
        "- ask_planner spends one planner call and does not move.\n"
        f"Choose an action whose preconditions match the {action_basis}"
        f"{_view_notes(obs)}"
        "Current agent view:\n"
        f"{_agent_view_grid(obs['view'])}\n"
    )


def _planner_prompt(
    *,
    obs: dict[str, Any],
    memory: dict[str, Any],
) -> str:
    return (
        "You advise a smaller mover model in MiniGrid DoorKey. "
        "The mover chooses the actual action; you provide objective guidance only.\n"
        "Use only the current view, inventory, and observed memory below.\n"
        "Return exactly three lines: subgoal, tool, reason. No extra text.\n"
        f"Mission: {obs['mission']}\n"
        f"Inventory key: {'yes' if obs['has_key'] else 'no'}\n"
        f"Tile directly ahead: {_tile_label(_tile_ahead(obs))}\n"
        f"{_view_notes(obs)}"
        "Memory map: only previously observed tiles; ? is unknown, not empty.\n"
        "DoorKey target rule:\n"
        "- if inventory has no key and k appears in either grid, subgoal is get_key.\n"
        "- if inventory has key and D appears in either grid, subgoal is open_door.\n"
        "- if inventory has key and G appears in either grid, subgoal is reach_goal.\n"
        "- otherwise, subgoal is explore.\n"
        "A plan may name one immediate tool, but not a route or hidden-map guess.\n"
        "Use egocentric words like ahead, left, right, and front-left; do not use compass directions.\n"
        "Explore is a subgoal, never a tool.\n"
        "Primitive action rule:\n"
        "- tool pick_up_key only if the tile directly ahead is key and inventory has no key.\n"
        "- tool open_door only if the tile directly ahead is a closed door and inventory has key.\n"
        "- tool move_forward only if the tile directly ahead is empty, open door, or goal.\n"
        "- if the tile directly ahead is wall or unseen, tool is turn_left or turn_right.\n"
        "Use only the grids below.\n"
        "subgoal: one of get_key, open_door, reach_goal, explore\n"
        "tool: one of pick_up_key, open_door, turn_left, turn_right, move_forward, ask_planner\n"
        "reason: one short sentence grounded in visible or remembered tiles\n"
        "Current agent view:\n"
        f"{_agent_view_grid(obs['view'])}\n"
        "Memory map:\n"
        f"{_grid(memory['fog'])}\n"
    )


def _view_notes(obs: dict[str, Any]) -> str:
    view = obs.get("view")
    size = "odd-sized"
    if isinstance(view, list) and view and isinstance(view[0], list):
        size = f"{len(view[0])}x{len(view)}"
    return (
        "Legend: .=empty, #=wall, k=key, D=closed door, o=open door, G=goal, ?=unseen.\n"
        f"Agent view: egocentric {size}; agent is bottom center, facing the top row.\n"
        "Rows above ^ are in front of the agent.\n"
        "The tile directly ahead is the cell immediately above ^.\n"
        "Only same-row neighbors beside ^ are immediately left or right.\n"
    )


def _planner_advice(advice: str | dict[str, Any]) -> str:
    if isinstance(advice, dict):
        advice = str(advice.get("advice") or "none")
    return f"Planner advice:\n{advice}\n"


def _planning_rule(*, advice: str | None, remaining: int) -> str:
    if advice is not None and advice.strip():
        return "Use planner advice only when it matches the live preconditions.\n"
    if remaining > 0:
        return (
            "Planner calls are scarce and optional; use ask_planner when "
            "the next DoorKey object is not visible or the current primitive action is unclear.\n"
        )
    return (
        "No planner calls remain; ask_planner is a lost turn. "
        "Choose a primitive action from the current view.\n"
    )


def _plan_advice(output: str) -> str:
    prefixes = ("subgoal:", "tool:", "reason:")
    seen: set[str] = set()
    lines = []
    for line in output.splitlines():
        normalized = " ".join(line.split())
        lowered = normalized.lower()
        key = next((prefix for prefix in prefixes if lowered.startswith(prefix)), None)
        if key is not None and key not in seen:
            seen.add(key)
            lines.append(normalized)
    return "\n".join(lines)


def _tile_ahead(obs: dict[str, Any]) -> str:
    view = obs.get("view")
    if not isinstance(view, list) or len(view) < 2:
        return "?"
    row = view[-2]
    if not isinstance(row, list) or not row:
        return "?"
    return str(row[len(row) // 2])


def _tile_left(obs: dict[str, Any]) -> str:
    view = obs.get("view")
    if not isinstance(view, list) or not view:
        return "?"
    row = view[-1]
    if not isinstance(row, list) or len(row) < 2:
        return "?"
    return str(row[max(0, len(row) // 2 - 1)])


def _tile_right(obs: dict[str, Any]) -> str:
    view = obs.get("view")
    if not isinstance(view, list) or not view:
        return "?"
    row = view[-1]
    if not isinstance(row, list) or len(row) < 2:
        return "?"
    return str(row[min(len(row) - 1, len(row) // 2 + 1)])


def _local_objective(obs: dict[str, Any]) -> str:
    has_key = bool(obs["has_key"])
    target = "key" if not has_key else "closed_door"
    tiles = {
        "tile_ahead": _tile_name(_tile_ahead(obs)),
        "tile_left": _tile_name(_tile_left(obs)),
        "tile_right": _tile_name(_tile_right(obs)),
    }
    if target not in tiles.values() and has_key:
        target = "goal"
    for location, tile in tiles.items():
        if tile == target:
            return f"{target} at {location}"
    return "not in local tiles"


def _tile_label(tile: str) -> str:
    labels = {
        ".": "empty",
        "#": "wall",
        "k": "key",
        "D": "closed door",
        "o": "open door",
        "G": "goal",
        "?": "unseen",
    }
    return f"{labels.get(tile, 'unknown')} ({tile})"


def _tile_name(tile: str) -> str:
    names = {
        ".": "empty",
        "#": "wall",
        "k": "key",
        "D": "closed_door",
        "o": "open_door",
        "G": "goal",
        "?": "unseen",
    }
    return names.get(tile, "unknown")


def _grid(rows: object) -> str:
    if not isinstance(rows, list):
        return ""
    return "\n".join(
        "".join(str(cell) for cell in row) for row in rows if isinstance(row, list)
    )


def _agent_view_grid(rows: object) -> str:
    if not isinstance(rows, list) or not rows or not isinstance(rows[-1], list):
        return _grid(rows)
    marked = [list(row) for row in rows if isinstance(row, list)]
    marked[-1][len(marked[-1]) // 2] = "^"
    return _grid(marked)
