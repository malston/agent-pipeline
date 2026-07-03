"""Context Translation for the A1 -> A2 boundary.

Maps the retrieval vocabulary (normalized_query / passages / coverage) into the
analysis vocabulary (question / evidence_pool / retrieval_confidence). This is
the only place that vocabulary change happens, and it is Model-free.
"""
from agent_pipeline.contracts.retrieval import RetrievalBundle
from agent_pipeline.contracts.analysis import AnalystInput, Evidence


def translate_retrieval_to_analysis(bundle: RetrievalBundle) -> AnalystInput:
    return AnalystInput(
        request_id=bundle.request_id,
        question=bundle.normalized_query,
        evidence_pool=[
            Evidence(id=p.source_id, text=p.text) for p in bundle.passages
        ],
        retrieval_confidence=bundle.coverage,
    )
