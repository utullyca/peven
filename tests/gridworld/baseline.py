from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from examples.minigrid.gridworld import DIR_X, DIR_Y, DoorKeyWorld


State = tuple[int, int, int]


def scripted_action(world: DoorKeyWorld) -> str:
    base: Any = world.env.unwrapped
    pos = (int(base.agent_pos[0]), int(base.agent_pos[1]))
    direction = int(base.agent_dir)
    key_pos = _find_one(world, "key")
    door_pos = _find_one(world, "door")
    door_cell = base.grid.get(*door_pos) if door_pos is not None else None
    goal_pos = _find_one(world, "goal")

    if not world.has_key() and key_pos is not None:
        return _approach(world, pos, direction, key_pos, interact="pickup")
    if door_cell is not None and not door_cell.is_open and door_pos is not None:
        return _approach(world, pos, direction, door_pos, interact="toggle")
    if goal_pos is not None:
        return _walk_to(world, pos, direction, goal_pos)
    return "forward"


def _find_one(world: DoorKeyWorld, obj_type: str) -> tuple[int, int] | None:
    base: Any = world.env.unwrapped
    grid = base.grid
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == obj_type:
                return (x, y)
    return None


def _approach(
    world: DoorKeyWorld,
    pos: tuple[int, int],
    direction: int,
    target: tuple[int, int],
    *,
    interact: str,
) -> str:
    front = (pos[0] + DIR_X[direction], pos[1] + DIR_Y[direction])
    if front == target:
        return interact
    return _bfs_first(
        world,
        pos,
        direction,
        goal=lambda s: (s[0] + DIR_X[s[2]], s[1] + DIR_Y[s[2]]) == target,
        allow_stepping_on=target,
        block_target=True,
    )


def _walk_to(
    world: DoorKeyWorld,
    pos: tuple[int, int],
    direction: int,
    target: tuple[int, int],
) -> str:
    return _bfs_first(
        world,
        pos,
        direction,
        goal=lambda s: (s[0], s[1]) == target,
        allow_stepping_on=target,
        block_target=False,
    )


def _bfs_first(
    world: DoorKeyWorld,
    pos: tuple[int, int],
    direction: int,
    *,
    goal: Callable[[State], bool],
    allow_stepping_on: tuple[int, int],
    block_target: bool,
) -> str:
    base: Any = world.env.unwrapped
    grid = base.grid
    start = (pos[0], pos[1], direction)
    if goal(start):
        return "forward"

    frontier: deque[tuple[State, str | None]] = deque([(start, None)])
    seen = {start}
    while frontier:
        state, first = frontier.popleft()
        x, y, d = state
        for action in ("left", "right", "forward"):
            if action == "forward":
                nx, ny = x + DIR_X[d], y + DIR_Y[d]
                if block_target and (nx, ny) == allow_stepping_on:
                    continue
                if not _is_walkable(grid.get(nx, ny)):
                    continue
                next_state = (nx, ny, d)
            elif action == "left":
                next_state = (x, y, (d - 1) % 4)
            else:
                next_state = (x, y, (d + 1) % 4)

            if next_state in seen:
                continue
            seen.add(next_state)
            first_action = first if first is not None else action
            if goal(next_state):
                return first_action
            frontier.append((next_state, first_action))
    return "forward"


def _is_walkable(cell: Any) -> bool:
    return (
        cell is None
        or cell.type == "goal"
        or bool(cell.type == "door" and cell.is_open)
    )
