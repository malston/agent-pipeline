"""Context Translation for the A2 -> A3 boundary.

Findings (analysis vocabulary) become points (composition vocabulary); gaps carry
through. Deterministic and Model-free.
"""
from agent_pipeline.contracts.analysis import AnalysisReport, Finding
from agent_pipeline.contracts.composition import ComposerInput
from agent_pipeline.translators.analysis_to_composition import (
    translate_analysis_to_composition,
)


def _report():
    return AnalysisReport(
        request_id="r1",
        findings=[
            Finding(claim="cells make ATP", evidence=["mito"], confidence=0.9),
            Finding(claim="plants use light", evidence=["photo", "mito"], confidence=0.6),
        ],
        gaps=["nothing on bacteria"],
    )


def test_translation_produces_valid_composer_input():
    out = translate_analysis_to_composition(_report())
    assert isinstance(out, ComposerInput)
    assert out.request_id == "r1"


def test_findings_map_to_points_and_gaps_carry_through():
    out = translate_analysis_to_composition(_report())
    assert [(p.statement, p.sources, p.confidence) for p in out.points] == [
        ("cells make ATP", ["mito"], 0.9),
        ("plants use light", ["photo", "mito"], 0.6),
    ]
    assert out.gaps == ["nothing on bacteria"]
