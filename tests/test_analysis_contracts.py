"""A2's output contracts: Finding and AnalysisReport.

A finding must cite at least one piece of evidence -- an uncited claim is
ungrounded by construction.
"""
import pytest
from pydantic import ValidationError

from agent_pipeline.contracts.analysis import Finding, AnalysisReport


def test_finding_requires_at_least_one_evidence_id():
    with pytest.raises(ValidationError):
        Finding(claim="the sky is blue", evidence=[], confidence=0.9)


def test_finding_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        Finding(claim="c", evidence=["s1"], confidence=1.5)


def test_finding_rejects_confidence_below_zero():
    with pytest.raises(ValidationError):
        Finding(claim="c", evidence=["s1"], confidence=-0.1)


def test_finding_rejects_empty_claim():
    with pytest.raises(ValidationError):
        Finding(claim="", evidence=["s1"], confidence=0.9)


def test_analysis_report_allows_no_findings_with_gaps():
    report = AnalysisReport(request_id="r1", findings=[], gaps=["no evidence on X"])
    assert report.findings == []
    assert report.gaps == ["no evidence on X"]


def test_analysis_report_json_round_trip():
    report = AnalysisReport(
        request_id="r1",
        findings=[Finding(claim="cells make ATP", evidence=["mito"], confidence=0.8)],
        gaps=[],
    )
    restored = AnalysisReport.model_validate_json(report.model_dump_json())
    assert restored == report
    assert restored.findings[0].evidence == ["mito"]
