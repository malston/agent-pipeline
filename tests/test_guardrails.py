"""Guardrails are the Harness gates every agent output must clear.

Plan guardrail: the plan the Model emits may only use granted tools, must fit
the step budget, and must end by emitting the contract.
Output guardrail: A1 may not emit a bundle citing a source it did not retrieve
(no fabricated citations).
"""
import pytest

from agent_pipeline.agents.plan import Plan, PlanStep
from agent_pipeline.agents.guardrails import (
    GuardrailViolation,
    validate_plan,
    validate_retrieval_output,
)
from agent_pipeline.contracts.retrieval import Passage, RetrievalBundle

ALLOWED = {"search_knowledge", "get_source", "save_scratch", "load_scratch", "emit_contract"}


def _plan(*tools):
    return Plan(steps=[PlanStep(step_id=i, intent="do", tool=t) for i, t in enumerate(tools)])


def test_plan_rejects_tool_outside_grant():
    with pytest.raises(GuardrailViolation):
        validate_plan(_plan("search_knowledge", "check_claim", "emit_contract"),
                      allowed_tools=ALLOWED, max_steps=5)


def test_plan_rejects_exceeding_step_budget():
    with pytest.raises(GuardrailViolation):
        validate_plan(_plan("search_knowledge", "search_knowledge", "emit_contract"),
                      allowed_tools=ALLOWED, max_steps=2)


def test_plan_rejects_missing_terminal_emit():
    with pytest.raises(GuardrailViolation):
        validate_plan(_plan("search_knowledge", "save_scratch"),
                      allowed_tools=ALLOWED, max_steps=5)


def test_plan_rejects_nonterminal_emit():
    # An emit before other steps short-circuits the executor (it returns on the
    # first emit_contract), silently skipping retrieval. Reject it at the plan gate.
    with pytest.raises(GuardrailViolation) as exc:
        validate_plan(_plan("emit_contract", "search_knowledge", "emit_contract"),
                      allowed_tools=ALLOWED, max_steps=5)
    assert exc.value.code == "PREMATURE_EMIT"


def test_plan_accepts_single_terminal_emit():
    # A2/A3 emit exactly [emit_contract]; steps[:-1] is empty here, so the
    # premature-emit gate must not reject a legitimate single-step plan.
    validate_plan(_plan("emit_contract"), allowed_tools=ALLOWED, max_steps=5)  # no raise


def test_plan_accepts_valid_plan():
    validate_plan(_plan("search_knowledge", "save_scratch", "emit_contract"),
                  allowed_tools=ALLOWED, max_steps=5)  # no raise


def test_output_rejects_ungrounded_citation():
    bundle = RetrievalBundle(
        request_id="r1", normalized_query="q",
        passages=[Passage(source_id="ghost", text="made up", score=0.9)],
        coverage=0.9,
    )
    with pytest.raises(GuardrailViolation):
        validate_retrieval_output(bundle, known_source_ids={"mito", "econ"})


def test_output_accepts_grounded_bundle():
    bundle = RetrievalBundle(
        request_id="r1", normalized_query="q",
        passages=[Passage(source_id="mito", text="real", score=0.9)],
        coverage=0.9,
    )
    validate_retrieval_output(bundle, known_source_ids={"mito", "econ"})  # no raise
