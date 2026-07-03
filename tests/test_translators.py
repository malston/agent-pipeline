"""Context Translation: A1's output vocabulary -> A2's input vocabulary.

Deterministic, Model-free, and unit-testable. A producer-side rename breaks
these tests, not the downstream Model silently.
"""
from agent_pipeline.contracts.retrieval import Passage, RetrievalBundle
from agent_pipeline.contracts.analysis import AnalystInput
from agent_pipeline.translators.retrieval_to_analysis import (
    translate_retrieval_to_analysis,
)


def _bundle():
    return RetrievalBundle(
        request_id="r1",
        normalized_query="how do cells produce energy",
        passages=[
            Passage(source_id="mito", text="ATP is made in mitochondria", score=0.9),
            Passage(source_id="photo", text="plants use sunlight", score=0.6),
        ],
        coverage=0.75,
    )


def test_translation_produces_valid_analyst_input():
    out = translate_retrieval_to_analysis(_bundle())
    assert isinstance(out, AnalystInput)


def test_translation_maps_vocabulary_fields():
    out = translate_retrieval_to_analysis(_bundle())
    assert out.request_id == "r1"
    assert out.question == "how do cells produce energy"        # normalized_query -> question
    assert out.retrieval_confidence == 0.75                     # coverage -> retrieval_confidence


def test_translation_maps_passages_to_evidence():
    out = translate_retrieval_to_analysis(_bundle())
    assert [(e.id, e.text) for e in out.evidence_pool] == [
        ("mito", "ATP is made in mitochondria"),
        ("photo", "plants use sunlight"),
    ]
