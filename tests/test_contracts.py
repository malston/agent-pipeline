"""Contracts are the typed boundaries between pipeline agents.

These tests pin the invariants each contract must enforce so a producing agent
cannot emit a structurally-invalid artifact to the next stage.
"""
import pytest
from pydantic import ValidationError

from agent_pipeline.contracts.retrieval import (
    Passage,
    RetrievalRequest,
    RetrievalBundle,
)
from agent_pipeline.contracts.analysis import AnalystInput, Evidence


def test_passage_rejects_score_above_one():
    with pytest.raises(ValidationError):
        Passage(source_id="s1", text="hello", score=1.5)


def test_passage_rejects_score_below_zero():
    with pytest.raises(ValidationError):
        Passage(source_id="s1", text="hello", score=-0.1)


def test_passage_accepts_score_in_unit_interval():
    p = Passage(source_id="s1", text="hello", score=0.42)
    assert p.score == 0.42


def test_retrieval_bundle_rejects_coverage_outside_unit_interval():
    with pytest.raises(ValidationError):
        RetrievalBundle(
            request_id="r1",
            normalized_query="q",
            passages=[Passage(source_id="s1", text="t", score=0.9)],
            coverage=1.2,
        )


def test_retrieval_bundle_json_round_trip_preserves_passages():
    bundle = RetrievalBundle(
        request_id="r1",
        normalized_query="what is add",
        passages=[
            Passage(source_id="s1", text="alpha", score=0.9),
            Passage(source_id="s2", text="beta", score=0.7),
        ],
        coverage=0.8,
    )
    restored = RetrievalBundle.model_validate_json(bundle.model_dump_json())
    assert restored == bundle
    assert [p.source_id for p in restored.passages] == ["s1", "s2"]


def test_retrieval_request_requires_raw_query():
    with pytest.raises(ValidationError):
        RetrievalRequest(request_id="r1")  # missing raw_query


def test_analyst_input_carries_retrieval_confidence():
    ai = AnalystInput(
        request_id="r1",
        question="what is add",
        evidence_pool=[Evidence(id="s1", text="alpha")],
        retrieval_confidence=0.8,
    )
    assert ai.retrieval_confidence == 0.8
    assert ai.evidence_pool[0].id == "s1"
