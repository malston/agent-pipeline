"""A3 Composer: Model (composer) + Harness (Plan-Execute loop, guardrails).

The keyless RuleBasedComposer lets the agent run offline with real guardrails.
The LLM-backed composer is exercised in test_a3_e2e.py, gated on a provider key.
"""
import pytest

from agent_pipeline.agents.composer import A3Composer, RuleBasedComposer, CompositionPlan
from agent_pipeline.agents.plan import PlanStep
from agent_pipeline.agents.guardrails import GuardrailViolation
from agent_pipeline.contracts.composition import (
    ComposerInput,
    Point,
    Section,
    Draft,
)


def _input():
    return ComposerInput(
        request_id="r1",
        points=[
            Point(statement="Cells produce ATP in mitochondria", sources=["mito"], confidence=0.9),
            Point(statement="Plants also use photosynthesis", sources=["photo"], confidence=0.6),
        ],
        gaps=["nothing on bacteria"],
    )


def test_rule_based_composer_emits_plan_ending_in_emit():
    plan = RuleBasedComposer().compose(_input())
    assert plan.steps[-1].tool == "emit_contract"


def test_a3_produces_grounded_draft_end_to_end():
    draft = A3Composer(RuleBasedComposer()).run(_input())

    assert isinstance(draft, Draft)
    assert draft.request_id == "r1"
    assert draft.sections, "expected at least one section"
    assert draft.style_profile
    available = {"mito", "photo"}
    for section in draft.sections:
        assert set(section.cited_sources) <= available  # every citation is grounded


def test_a3_rejects_plan_using_ungranted_tool():
    class _RetrievingComposer:
        def compose(self, composer_input: ComposerInput) -> CompositionPlan:
            return CompositionPlan(
                steps=[
                    PlanStep(step_id=0, intent="retrieve", tool="search_knowledge"),
                    PlanStep(step_id=1, intent="emit", tool="emit_contract"),
                ],
                sections=[Section(heading="H", body="B", cited_sources=["mito"])],
                style_profile="x",
            )

    with pytest.raises(GuardrailViolation) as exc:
        A3Composer(_RetrievingComposer()).run(_input())
    assert exc.value.code == "TOOL_NOT_GRANTED"


def test_a3_rejects_fabricated_citation():
    class _FabricatingComposer:
        def compose(self, composer_input: ComposerInput) -> CompositionPlan:
            return CompositionPlan(
                steps=[PlanStep(step_id=0, intent="emit", tool="emit_contract")],
                sections=[Section(heading="H", body="B", cited_sources=["ghost"])],
                style_profile="x",
            )

    with pytest.raises(GuardrailViolation) as exc:
        A3Composer(_FabricatingComposer()).run(_input())
    assert exc.value.code == "UNGROUNDED_SECTION"
