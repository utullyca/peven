from __future__ import annotations

import pytest


pytest.importorskip("minigrid")

from examples.minigrid.gridworld import merge_fog


def _blank_fog(size: int = 8) -> list[list[str]]:
    return [["?"] * size for _ in range(size)]


def _ego_with_one(glyph: str, *, row: int, col: int, size: int = 5) -> list[list[str]]:
    view = [["?"] * size for _ in range(size)]
    view[row][col] = glyph
    return view


def test_merge_fog_direction_east_places_front_cell_east_of_agent() -> None:
    fog = _blank_fog()
    ego = _ego_with_one("K", row=3, col=2)
    merged = merge_fog(fog, ego, agent_pos=(3, 5), agent_dir=0)
    assert merged[5][4] == "K"
    assert fog[5][4] == "?"


def test_merge_fog_direction_south_places_front_cell_south_of_agent() -> None:
    fog = _blank_fog()
    ego = _ego_with_one("K", row=3, col=2)
    merged = merge_fog(fog, ego, agent_pos=(3, 5), agent_dir=1)
    assert merged[6][3] == "K"


def test_merge_fog_direction_west_places_front_cell_west_of_agent() -> None:
    fog = _blank_fog()
    ego = _ego_with_one("K", row=3, col=2)
    merged = merge_fog(fog, ego, agent_pos=(3, 5), agent_dir=2)
    assert merged[5][2] == "K"


def test_merge_fog_direction_north_places_front_cell_north_of_agent() -> None:
    fog = _blank_fog()
    ego = _ego_with_one("K", row=3, col=2)
    merged = merge_fog(fog, ego, agent_pos=(3, 5), agent_dir=3)
    assert merged[4][3] == "K"


def test_merge_fog_right_of_agent_east_facing_goes_south_in_world() -> None:
    fog = _blank_fog()
    ego = _ego_with_one("K", row=4, col=3)
    merged = merge_fog(fog, ego, agent_pos=(3, 5), agent_dir=0)
    assert merged[6][3] == "K"


def test_merge_fog_does_not_overwrite_known_cells_with_question_marks() -> None:
    fog = _blank_fog()
    fog[5][4] = "K"
    ego = [["?"] * 5 for _ in range(5)]
    merged = merge_fog(fog, ego, agent_pos=(3, 5), agent_dir=0)
    assert merged[5][4] == "K"


def test_merge_fog_overwrites_stale_cells_with_visible_tiles() -> None:
    fog = _blank_fog()
    fog[5][4] = "k"
    ego = _ego_with_one(".", row=3, col=2)
    merged = merge_fog(fog, ego, agent_pos=(3, 5), agent_dir=0)
    assert merged[5][4] == "."


def test_merge_fog_clips_at_world_bounds() -> None:
    fog = _blank_fog(size=6)
    ego = _ego_with_one("W", row=0, col=2)
    merged = merge_fog(fog, ego, agent_pos=(3, 1), agent_dir=3)
    for row in merged:
        for glyph in row:
            assert glyph == "?"
