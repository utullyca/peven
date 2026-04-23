from __future__ import annotations

import pytest

import peven
from peven.authoring.join import JoinTuple
from peven.shared.errors import PevenValidationError


def test_topology_constructors_validate_basic_arguments() -> None:
    with pytest.raises(ValueError, match="positive int or None"):
        peven.place(capacity=0)

    with pytest.raises(ValueError, match="place ids must be non-empty strings"):
        peven.input("")

    with pytest.raises(ValueError, match="input weight must be a positive int"):
        peven.input("ready", weight=0)

    with pytest.raises(ValueError, match="place ids must be non-empty strings"):
        peven.output("")


def test_transition_decorator_validates_executor_retries_and_dsl_types() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        peven.transition(inputs=["ready"], outputs=["done"], executor="")

    with pytest.raises(ValueError, match="non-negative int"):
        peven.transition(inputs=["ready"], outputs=["done"], executor="judge", retries=-1)

    with pytest.raises(PevenValidationError, match=r"guard must be a peven\.guard expression"):
        peven.transition(inputs=["ready"], outputs=["done"], executor="judge", guard=object())

    with pytest.raises(PevenValidationError, match=r"join_by must be a peven\.join selector"):
        peven.transition(inputs=["ready"], outputs=["done"], executor="judge", join_by=object())

    with pytest.raises(PevenValidationError, match=r"inputs must be str or peven\.input"):
        peven.transition(inputs=[object()], outputs=["done"], executor="judge")

    with pytest.raises(PevenValidationError, match=r"outputs must be str or peven\.output"):
        peven.transition(inputs=["ready"], outputs=[object()], executor="judge")

    with pytest.raises(PevenValidationError, match="tuple selectors require at least one item"):
        peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="judge",
            join_by=JoinTuple(()),
        )


def test_transition_decorator_treats_bare_string_inputs_outputs_as_single_declarations() -> None:
    declaration = peven.transition(inputs="ready", outputs="done", executor="judge")

    assert [input_decl.place for input_decl in declaration.inputs] == ["ready"]
    assert [output_decl.place for output_decl in declaration.outputs] == ["done"]


def test_transition_decorator_allows_duplicate_output_places_to_reach_adapter_boundary() -> None:
    declaration = peven.transition(
        inputs=["ready"],
        outputs=["done", "done"],
        executor="judge",
    )

    assert [output_decl.place for output_decl in declaration.outputs] == ["done", "done"]
