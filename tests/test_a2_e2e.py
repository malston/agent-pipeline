"""End-to-end A2 with the REAL provider-agnostic Model (LLMAnalyst).

Real LLM, real guardrails. Skips (never mocks) without a provider key.
"""
import os

import pytest

from agent_pipeline.agents.analyst import A2Analyst, LLMAnalyst
from agent_pipeline.contracts.analysis import AnalystInput, Evidence, AnalysisReport


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="no ANTHROPIC_API_KEY; real-LLM e2e skipped (not mocked)",
)
def test_a2_end_to_end_with_real_llm():
    analyst_input = AnalystInput(
        request_id="r1",
        question="how do cells produce energy?",
        evidence_pool=[
            Evidence(id="mito", text="Mitochondria produce ATP, the cell's energy currency."),
            Evidence(id="econ", text="Central banks adjust interest rates to steer inflation."),
        ],
        retrieval_confidence=0.8,
    )
    report = A2Analyst(LLMAnalyst()).run(analyst_input)

    assert isinstance(report, AnalysisReport)
    assert report.findings
    # the output guardrail already blocks ungrounded citations; assert it explicitly
    for finding in report.findings:
        assert set(finding.evidence) <= {"mito", "econ"}
