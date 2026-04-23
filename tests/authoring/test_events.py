from __future__ import annotations

import peven


def test_public_event_and_result_shapes_follow_engine_contract() -> None:
    token = peven.token({"ok": True}, run_key="rk")
    bundle = peven.events.BundleRef(transition_id="judge", run_key="rk", ordinal=1)

    started = peven.events.TransitionStarted(bundle=bundle, firing_id=1, attempt=1, inputs=[token])
    completed = peven.events.TransitionCompleted(
        bundle=bundle,
        firing_id=1,
        attempt=1,
        outputs={"done": [token]},
    )
    failed = peven.events.TransitionFailed(
        bundle=bundle,
        firing_id=1,
        attempt=2,
        error="boom",
        retrying=True,
    )
    guard_errored = peven.events.GuardErrored(bundle=bundle, error="guard")
    selection_errored = peven.events.SelectionErrored(
        transition_id="judge",
        run_key="rk",
        error="select",
    )
    completed_result = peven.events.TransitionResult(
        bundle=bundle,
        firing_id=1,
        status="completed",
        outputs={"done": [token]},
        attempts=1,
    )
    failed_result = peven.events.TransitionResult(
        bundle=bundle,
        firing_id=2,
        status="failed",
        outputs={},
        error="boom",
        attempts=2,
    )
    fuse_blocked_result = peven.events.TransitionResult(
        bundle=bundle,
        firing_id=3,
        status="fuse_blocked",
        outputs={},
        attempts=1,
    )
    run_result = peven.events.RunResult(
        run_key="rk",
        status="failed",
        error="boom",
        terminal_reason="executor_failed",
        terminal_bundle=bundle,
        terminal_transition="judge",
        trace=[completed_result, failed_result, fuse_blocked_result],
        final_marking={"done": [token]},
    )
    finished = peven.events.RunFinished(result=run_result)

    assert started.kind == "transition_started"
    assert completed.outputs == {"done": [token]}
    assert failed.error == "boom"
    assert guard_errored.bundle is bundle
    assert selection_errored.transition_id == "judge"
    assert finished.result is run_result


def test_bundle_ref_preserves_the_documented_positional_order() -> None:
    bundle = peven.events.BundleRef("judge", "rk", None, 7)

    assert bundle.transition_id == "judge"
    assert bundle.run_key == "rk"
    assert bundle.selected_key is None
    assert bundle.ordinal == 7
    assert bundle.key is None
    assert bundle.idx == 7

    keyword_bundle = peven.events.BundleRef(
        transition_id="judge",
        run_key="rk",
        selected_key="case-17",
        ordinal=3,
    )
    assert keyword_bundle.selected_key == "case-17"
    assert keyword_bundle.ordinal == 3
    assert keyword_bundle.key == "case-17"
    assert keyword_bundle.idx == 3


def test_run_result_trace_helpers_follow_engine_result_surface() -> None:
    token = peven.token({"ok": True}, run_key="rk")
    bundle = peven.events.BundleRef(transition_id="judge", run_key="rk", ordinal=1)
    completed = peven.events.TransitionResult(
        bundle=bundle,
        firing_id=1,
        status="completed",
        outputs={"done": [token]},
    )
    failed = peven.events.TransitionResult(
        bundle=bundle,
        firing_id=2,
        status="failed",
        outputs={},
        error="boom",
    )
    fuse_blocked = peven.events.TransitionResult(
        bundle=bundle,
        firing_id=3,
        status="fuse_blocked",
        outputs={},
    )
    result = peven.events.RunResult(
        run_key="rk",
        status="incomplete",
        trace=[completed, failed, fuse_blocked],
        final_marking={"done": [token]},
    )

    assert peven.events.completed_firings(result) == [completed]
    assert peven.events.failed_firings(result) == [failed]
    assert peven.events.fuse_blocked_firings(result) == [fuse_blocked]
    assert peven.events.firing_result(result, 2) == failed
    assert peven.events.firing_result(result, 99) is None
    assert peven.events.firing_status(result, 1) == "completed"
    assert peven.events.firing_status(result, 99) is None
