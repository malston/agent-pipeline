"""A2 output guardrail: a finding may only cite evidence from the input pool.

This is the analysis-stage analogue of A1's ungrounded-citation guard -- it stops
the analyst fabricating evidence ids the retriever never surfaced.
"""
import pytest

from agent_pipeline.agents.guardrails import GuardrailViolation, validate_analysis_output
from agent_pipeline.contracts.analysis import Finding, AnalysisReport


def test_analysis_rejects_finding_citing_evidence_outside_pool():
    report = AnalysisReport(
        request_id="r1",
        findings=[Finding(claim="made up", evidence=["ghost"], confidence=0.9)],
    )
    with pytest.raises(GuardrailViolation) as exc:
        validate_analysis_output(report, evidence_ids={"mito", "econ"})
    assert exc.value.code == "UNGROUNDED_FINDING"


def test_analysis_accepts_findings_grounded_in_pool():
    report = AnalysisReport(
        request_id="r1",
        findings=[
            Finding(claim="cells make ATP", evidence=["mito"], confidence=0.8),
            Finding(claim="plants use light", evidence=["mito", "photo"], confidence=0.6),
        ],
    )
    validate_analysis_output(report, evidence_ids={"mito", "photo"})  # no raise
