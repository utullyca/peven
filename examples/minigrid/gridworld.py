"""MiniGrid DoorKey world plus observed-world memory.

MiniGrid observations are egocentric: an odd-sized image with the agent at
bottom center, facing toward the top row. The mover sees that view as-is.
``merge_fog`` only projects those observed tiles into a fixed world map for
planner memory and trace rendering.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401  (registers MiniGrid-* envs with gym)
from minigrid.core.actions import Actions
from minigrid.core.constants import DIR_TO_VEC, IDX_TO_OBJECT, STATE_TO_IDX
from minigrid.wrappers import ReseedWrapper, ViewSizeWrapper


ACTION_KINDS: tuple[str, ...] = tuple(action.name for action in Actions)
_ACTION_INDEX: dict[str, int] = {action.name: int(action) for action in Actions}
DIR_X: tuple[int, ...] = tuple(int(vec[0]) for vec in DIR_TO_VEC)
DIR_Y: tuple[int, ...] = tuple(int(vec[1]) for vec in DIR_TO_VEC)
_STATE_BY_INDEX = {index: state for state, index in STATE_TO_IDX.items()}


class DoorKeyWorld:
    """Thin owner for a standard MiniGrid DoorKey environment.

    The wrapper keeps the Gym/MiniGrid lifecycle in one place and exposes only
    JSON-friendly snapshots for Peven tokens.
    """

    env: Any

    def __init__(
        self,
        *,
        env_id: str = "MiniGrid-DoorKey-8x8-v0",
        seed: int = 0,
        view_size: int = 5,
    ) -> None:
        """Create the MiniGrid environment with deterministic reseeding."""
        self.env_id = env_id
        self.view_size = view_size
        self.env = self._make_env(seed)
        self.seed = seed
        self._obs: dict[str, Any] | None = None
        self._reward: float = 0.0
        self._terminated: bool = False
        self._truncated: bool = False

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset deterministically; ``None`` reuses the current seed."""
        if seed is not None and seed != self.seed:
            self.env = self._make_env(seed)
            self.seed = seed
        self._obs, _ = self.env.reset()
        self._reward = 0.0
        self._terminated = False
        self._truncated = False
        return self.snapshot()

    def step(self, action_kind: str) -> dict[str, Any]:
        """Apply one MiniGrid action by name and return the next snapshot."""
        if action_kind not in _ACTION_INDEX:
            raise ValueError(f"unknown MiniGrid action: {action_kind!r}")
        self._obs, reward, terminated, truncated, _ = self.env.step(
            _ACTION_INDEX[action_kind]
        )
        self._reward = float(reward)
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Return the model/log-facing state as plain Python data."""
        obs = self._current_obs()
        base: Any = self.env.unwrapped
        return {
            "agent_pos": [int(base.agent_pos[0]), int(base.agent_pos[1])],
            "agent_dir": int(base.agent_dir),
            "has_key": self.has_key(),
            "mission": str(obs["mission"]),
            "view": self.ego_view(),
            "step_count": int(base.step_count),
            "max_steps": int(base.max_steps),
            "reward": self._reward,
            "terminated": self._terminated,
            "truncated": self._truncated,
        }

    def ego_view(self) -> list[list[str]]:
        """Decode MiniGrid's egocentric image into compact tile glyphs."""
        image = self._current_obs()["image"]
        width, height = int(image.shape[0]), int(image.shape[1])
        rows = [
            [_encoded_tile_glyph(image[x, y]) for x in range(width)]
            for y in range(height)
        ]
        if self.has_key() and rows[-1][width // 2] == "k":
            rows[-1][width // 2] = "."
        return rows

    def has_key(self) -> bool:
        """Return whether the agent is currently carrying the DoorKey key."""
        base: Any = self.env.unwrapped
        carrying = base.carrying
        return getattr(carrying, "type", None) == "key"

    @property
    def grid_size(self) -> tuple[int, int]:
        """Return the underlying MiniGrid world dimensions."""
        base: Any = self.env.unwrapped
        return int(base.width), int(base.height)

    def _current_obs(self) -> dict[str, Any]:
        """Return the latest MiniGrid observation, failing before reset."""
        if self._obs is None:
            raise RuntimeError("DoorKeyWorld has not been reset")
        return self._obs

    def _make_env(self, seed: int) -> Any:
        """Build the standard DoorKey env with deterministic and view wrappers."""
        base = gym.make(self.env_id)
        base = ReseedWrapper(base, seeds=[seed])
        return ViewSizeWrapper(base, agent_view_size=self.view_size)


def blank_fog(width: int, height: int) -> list[list[str]]:
    """Create an unknown-tile memory map matching the world size."""
    return [["?"] * width for _ in range(height)]


def _encoded_tile_glyph(tile: Any) -> str:
    """Map MiniGrid's compact tile encoding to the example's glyphs."""
    obj = IDX_TO_OBJECT[int(tile[0])]
    if obj == "unseen":
        return "?"
    if obj == "empty":
        return "."
    if obj == "door":
        return "o" if _STATE_BY_INDEX[int(tile[2])] == "open" else "D"
    if obj == "key":
        return "k"
    if obj == "goal":
        return "G"
    if obj == "wall":
        return "#"
    return "?"


def _rotate_ego(forward: int, right: int, agent_dir: int) -> tuple[int, int]:
    """Convert agent-relative offsets to MiniGrid world-space offsets."""
    if agent_dir == 0:
        return forward, right
    if agent_dir == 1:
        return -right, forward
    if agent_dir == 2:
        return -forward, -right
    return right, -forward


def merge_fog(
    fog: list[list[str]],
    ego_view: list[list[str]],
    agent_pos: tuple[int, int],
    agent_dir: int,
) -> list[list[str]]:
    """Project one egocentric observation into persistent world memory.

    MiniGrid already gives the mover the egocentric view. This helper is only
    for memory: it takes each visible tile's ``forward/right`` offset from the
    agent and writes it into the fixed top-down fog map.
    """
    merged = [row[:] for row in fog]
    size = len(ego_view)
    cx = size // 2
    cy = size - 1
    height = len(merged)
    width = len(merged[0]) if height else 0
    ax, ay = agent_pos
    for row in range(size):
        for col in range(size):
            glyph = ego_view[row][col]
            if glyph == "?":
                continue
            forward = cy - row
            right = col - cx
            dx, dy = _rotate_ego(forward, right, agent_dir)
            wx, wy = ax + dx, ay + dy
            if 0 <= wx < width and 0 <= wy < height:
                merged[wy][wx] = glyph
    return merged
