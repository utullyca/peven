"""Peven net for DoorKey: mover chooses, planner advises, env_step mutates MiniGrid.

``memory`` is a place because it is Peven-visible state: the observed fog map,
planner budget, and last action are consumed and re-emitted by the transitions
that must preserve or update them. Keeping it in the marking makes the run
traceable and prevents hidden Python state from steering the net.
"""

from __future__ import annotations

from examples.minigrid.agents import choose_move, make_plan
from examples.minigrid.gridworld import (
    DoorKeyWorld,
    blank_fog,
    merge_fog,
)

import peven


@peven.executor("minigrid_mover")
async def mover_executor(ctx, move_request, obs, memory, plan):
    """Ask the small mover model for one action or a planner call."""
    memory_payload = memory.payload
    plan_payload = plan.payload
    turn = move_request.payload["turn"]
    decision = await choose_move(
        ctx,
        obs=obs.payload,
        memory=memory_payload,
        plan=plan_payload,
    )

    if decision["kind"] == "ask_planner":
        if memory_payload["planner_calls"] >= memory_payload["planner_limit"]:
            return {
                "action": ctx.token(
                    {
                        "kind": "ask_planner",
                        "turn": turn,
                        "invalid": "planner_exhausted",
                        "env_action": "done",
                    }
                ),
                "plan_request": [],
                "obs": ctx.token(obs.payload),
                "memory": ctx.token(memory_payload),
                "plan": ctx.token(plan_payload),
            }
        # Asking the planner is exclusive with moving: no action token is emitted.
        return {
            "action": [],
            "plan_request": ctx.token(
                {
                    "turn": turn,
                    "version": plan_payload["version"] + 1,
                    "reason": "ask_planner",
                }
            ),
            "obs": ctx.token(obs.payload),
            "memory": ctx.token(memory_payload),
            "plan": [],
        }

    return {
        "action": ctx.token({**decision, "turn": turn}),
        "plan_request": [],
        "obs": ctx.token(obs.payload),
        "memory": ctx.token(memory_payload),
        "plan": ctx.token({"advice": "none", "version": plan_payload["version"]}),
    }


@peven.executor("minigrid_planner")
async def planner_executor(ctx, plan_request, obs, memory):
    """Ask the planner model for the next high-level directive."""
    request = plan_request.payload
    memory_payload = memory.payload
    turn = request["turn"] + 1
    plan = await make_plan(ctx, obs=obs.payload, memory=memory_payload, request=request)
    return {
        "plan": ctx.token({**plan, "version": request["version"]}),
        "obs": ctx.token(obs.payload),
        "memory": ctx.token(
            {
                **memory_payload,
                "planner_calls": memory_payload["planner_calls"] + 1,
                "steps_since_planner": 0,
            }
        ),
        "move_request": ctx.token({"turn": turn}),
    }


@peven.executor("minigrid_env_step")
async def env_step_executor(ctx, action, obs, memory):
    """Apply one action to MiniGrid and update Peven-visible memory."""
    del obs
    action_kind = action.payload["kind"]
    env_action = action.payload.get("env_action", action_kind)
    memory_payload = memory.payload
    next_turn = action.payload["turn"] + 1

    world: DoorKeyWorld = ctx.env.world
    snapshot = world.step(env_action)
    fog = merge_fog(
        memory_payload["fog"],
        world.ego_view(),
        (int(snapshot["agent_pos"][0]), int(snapshot["agent_pos"][1])),
        snapshot["agent_dir"],
    )
    new_memory = {
        **memory_payload,
        "fog": fog,
        "steps_since_planner": memory_payload["steps_since_planner"] + 1,
        "last_action": action_kind,
    }
    terminal = snapshot["terminated"] or snapshot["truncated"]

    return {
        "obs": ctx.token(snapshot),
        "memory": ctx.token(new_memory),
        "move_request": [] if terminal else ctx.token({"turn": next_turn}),
        "done": (
            ctx.token(
                {
                    "turns": next_turn,
                    "terminated": snapshot["terminated"],
                    "truncated": snapshot["truncated"],
                    "reward": snapshot["reward"],
                    "score": snapshot["reward"],
                    "planner_calls": new_memory["planner_calls"],
                    "planner_limit": new_memory["planner_limit"],
                }
            )
            if terminal
            else []
        ),
    }


@peven.env("minigrid_doorkey")
class DoorKeyEnv(peven.Env):
    """DoorKey net with one live token per state place."""

    obs = peven.place(capacity=1)
    memory = peven.place(capacity=1)  # observed fog, planner budget, last action
    plan = peven.place(capacity=1)
    move_request = peven.place(capacity=1)
    action = peven.place(capacity=1)
    plan_request = peven.place(capacity=1)
    done = peven.place(capacity=1, terminal=True)

    def __init__(self) -> None:
        self.world = DoorKeyWorld()

    def initial_marking(self, seed: int | None = None) -> peven.Marking:
        snapshot = self.world.reset(seed=seed)
        width, height = self.world.grid_size
        fog = merge_fog(
            blank_fog(width=width, height=height),
            self.world.ego_view(),
            (int(snapshot["agent_pos"][0]), int(snapshot["agent_pos"][1])),
            int(snapshot["agent_dir"]),
        )
        return peven.marking(
            obs=[snapshot],
            memory=[
                {
                    "fog": fog,
                    "steps_since_planner": 0,
                    "last_action": "reset",
                    "planner_calls": 0,
                    "planner_limit": 2,
                }
            ],
            plan=[
                {
                    "advice": "none",
                    "version": 0,
                }
            ],
            move_request=[{"turn": 0}],
        )

    mover = peven.transition(
        inputs=["move_request", "obs", "memory", "plan"],
        outputs=["action", "plan_request", "obs", "memory", "plan"],
        executor="minigrid_mover",
    )
    planner = peven.transition(
        inputs=["plan_request", "obs", "memory"],
        outputs=["plan", "obs", "memory", "move_request"],
        executor="minigrid_planner",
    )
    env_step = peven.transition(
        inputs=["action", "obs", "memory"],
        outputs=["obs", "memory", "move_request", "done"],
        executor="minigrid_env_step",
    )


def run_minigrid(
    *,
    command: tuple[str, ...],
    seed: int = 0,
    sink: object | None = None,
    fuse: int | None = 1_500,
) -> peven.RunResult:
    return DoorKeyEnv().run(command=command, sink=sink, fuse=fuse, seed=seed)
