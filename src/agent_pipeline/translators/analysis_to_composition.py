"""Context Translation for the A2 -> A3 boundary.

Maps the analysis vocabulary (findings: claim / evidence) into the composition
vocabulary (points: statement / sources). Gaps carry through unchanged. This is
the only place that vocabulary change happens, and it is Model-free.
"""
from agent_pipeline.contracts.analysis import AnalysisReport
from agent_pipeline.contracts.composition import ComposerInput, Point


def translate_analysis_to_composition(report: AnalysisReport) -> ComposerInput:
    return ComposerInput(
        request_id=report.request_id,
        points=[
            Point(statement=f.claim, sources=f.evidence, confidence=f.confidence)
            for f in report.findings
        ],
        gaps=report.gaps,
    )
