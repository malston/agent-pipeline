"""A2 Analyst: Model (analyst) + Harness (Plan-Execute loop, guardrails).

The keyless RuleBasedAnalyst lets the agent run end-to-end offline with real
guardrails and no LLM. The LLM-backed analyst (real Model) is exercised in
test_a2_e2e.py, gated on a provider key.
"""
import pytest

from agent_pipeline.agents.analyst import A2Analyst, RuleBasedAnalyst, AnalysisPlan
from agent_pipeline.agents.plan import PlanStep
from agent_pipeline.agents.guardrails import GuardrailViolation
from agent_pipeline.contracts.analysis import (
    AnalystInput,
    Evidence,
    Finding,
    AnalysisReport,
)


def _input():
    return AnalystInput(
        request_id="r1",
        question="how do cells produce energy?",
        evidence_pool=[
            Evidence(id="mito", text="mitochondria produce ATP"),
            Evidence(id="photo", text="plants convert sunlight to sugars"),
        ],
        retrieval_confidence=0.8,
    )


def test_rule_based_analyst_emits_plan_ending_in_emit(_input=_input):
    plan = RuleBasedAnalyst().analyze(_input())
    assert plan.steps[-1].tool == "emit_contract"


def test_a2_produces_grounded_report_end_to_end():
    report = A2Analyst(RuleBasedAnalyst()).run(_input())

    assert isinstance(report, AnalysisReport)
    assert report.request_id == "r1"
    assert report.findings, "expected at least one finding"
    pool_ids = {"mito", "photo"}
    for finding in report.findings:
        assert set(finding.evidence) <= pool_ids  # every finding grounded in the pool


def test_a2_rejects_plan_that_uses_ungranted_tool():
    class _UngrantedToolAnalyst:
        def analyze(self, analyst_input: AnalystInput) -> AnalysisPlan:
            return AnalysisPlan(
                steps=[
                    PlanStep(step_id=0, intent="judge", tool="check_claim"),
                    PlanStep(step_id=1, intent="emit", tool="emit_contract"),
                ],
                findings=[Finding(claim="c", evidence=["mito"], confidence=0.5)],
                gaps=[],
            )

    with pytest.raises(GuardrailViolation) as exc:
        A2Analyst(_UngrantedToolAnalyst()).run(_input())
    assert exc.value.code == "TOOL_NOT_GRANTED"


def test_a2_rejects_ungrounded_finding():
    class _FabricatingAnalyst:
        def analyze(self, analyst_input: AnalystInput) -> AnalysisPlan:
            return AnalysisPlan(
                steps=[PlanStep(step_id=0, intent="emit", tool="emit_contract")],
                findings=[Finding(claim="made up", evidence=["ghost"], confidence=0.9)],
                gaps=[],
            )

    with pytest.raises(GuardrailViolation) as exc:
        A2Analyst(_FabricatingAnalyst()).run(_input())
    assert exc.value.code == "UNGROUNDED_FINDING"
