from __future__ import annotations

import pytest

import peven
from peven.authoring.executor import ExecutorContext


def test_marking_defaults_to_empty_and_returns_copied_dict() -> None:
    marking = peven.Marking()

    assert marking.tokens_by_place == {}
    assert marking.to_dict() == {}


def test_marking_validates_places_buckets_and_tokens() -> None:
    with pytest.raises(ValueError, match="place ids must be non-empty strings"):
        peven.Marking({"": []})

    with pytest.raises(TypeError, match="buckets must be sequences of Token values"):
        peven.Marking({"ready": 1})  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must contain Token values"):
        peven.Marking({"ready": [object()]})


def test_marking_to_dict_returns_mutable_copy() -> None:
    token = peven.token({"ok": True}, run_key="rk")
    marking = peven.Marking({"ready": [token]})

    first = marking.to_dict()
    second = marking.to_dict()
    first["ready"].append(token)

    assert len(second["ready"]) == 1
    assert len(marking.tokens_by_place["ready"]) == 1


def test_run_keys_and_run_marking_follow_engine_marking_helpers() -> None:
    first = peven.token({"value": 1}, run_key="rk-1")
    second = peven.token({"value": 2}, run_key="rk-2")
    third = peven.token({"value": 3}, run_key="rk-1")
    marking = peven.Marking({"ready": [first, second], "done": [third]})

    assert peven.run_keys(marking) == ["rk-1", "rk-2"]
    assert peven.run_marking(marking, "rk-1") == peven.Marking({"ready": [first], "done": [third]})

    with pytest.raises(ValueError, match="run_key must be a non-empty string"):
        peven.run_marking(marking, "")


def test_token_validates_structured_payloads() -> None:
    with pytest.raises(TypeError):
        peven.token({"bad": object()})

    with pytest.raises(TypeError, match="run_key"):
        peven.token({"value": 1})  # type: ignore[call-arg]


def test_public_token_constructor_validates_fields_and_payload() -> None:
    with pytest.raises(TypeError, match="run_key"):
        peven.Token(run_key="")

    with pytest.raises(TypeError, match="color"):
        peven.Token(run_key="rk", color="")

    with pytest.raises(OverflowError, match="signed 64-bit"):
        peven.Token(run_key="rk", payload=2**100)


def test_marking_wraps_payloads_under_one_generated_run_key() -> None:
    marking = peven.marking(prompt=[{"question": "left"}, {"question": "right"}])

    prompt_tokens = marking.tokens_by_place["prompt"]

    assert len(prompt_tokens) == 2
    assert prompt_tokens[0].payload == {"question": "left"}
    assert prompt_tokens[1].payload == {"question": "right"}
    assert prompt_tokens[0].run_key == prompt_tokens[1].run_key


def test_marking_uses_the_explicit_run_key_for_all_places() -> None:
    marking = peven.marking(run_key="rk-explicit", prompt=[1], done=[2])

    assert marking.tokens_by_place["prompt"][0].run_key == "rk-explicit"
    assert marking.tokens_by_place["done"][0].run_key == "rk-explicit"


@pytest.mark.parametrize("bucket", ["hi", {"question": "hi"}])
def test_marking_rejects_non_bucket_single_payload_values(bucket: object) -> None:
    with pytest.raises(TypeError, match="wrap single payloads in a list"):
        peven.marking(prompt=bucket)  # type: ignore[arg-type]


def test_executor_context_token_uses_the_active_bundle_run_key() -> None:
    ctx = ExecutorContext(
        env=object(),
        bundle=peven.BundleRef(transition_id="finish", run_key="rk-bundle", ordinal=1),
        executor_name="finish",
        attempt=2,
    )

    default_token = ctx.token({"ok": True})
    custom_token = ctx.token({"ok": "blue"}, color="blue")

    assert default_token.run_key == "rk-bundle"
    assert default_token.color == "default"
    assert custom_token.run_key == "rk-bundle"
    assert custom_token.color == "blue"
