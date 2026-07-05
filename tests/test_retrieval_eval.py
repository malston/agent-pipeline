"""Agent-scope eval for A1 retrieval, over real embeddings.

Scores the RAG gate (recall@k, MRR) on a golden dataset and binds each score to the
trace of its run. Real fastembed embeddings, no mocks, no keys.
"""
import pytest
from pydantic import ValidationError

from agent_pipeline.evals.retrieval import RetrievalExample, evaluate_retrieval


def _golden():
    return [
        RetrievalExample(
            id="bio", query="how do cells produce energy", relevant_sources=["mito", "photo"]
        ),
        RetrievalExample(
            id="econ", query="how do central banks affect inflation", relevant_sources=["econ"]
        ),
    ]


def test_retrieval_eval_binds_scores_to_traces(knowledge):
    report = evaluate_retrieval(knowledge, _golden(), k=2)
    # 2 examples x 2 metrics
    assert len(report.scores) == 4
    trace_ids = {t.trace_id for t in report.traces}
    assert trace_ids and all(s.trace_id in trace_ids for s in report.scores)


def test_retrieval_eval_measures_quality_on_the_corpus(knowledge):
    report = evaluate_retrieval(knowledge, _golden(), k=2)
    # the topically-distinct corpus retrieves the relevant sources at k=2
    assert report.aggregate("recall_at_k") == 1.0
    assert report.aggregate("reciprocal_rank") == 1.0


def test_retrieval_eval_k1_is_a_drift_resistant_sentinel(knowledge):
    # k=1: 'bio' has two relevant sources but one slot -> recall 0.5; 'econ' -> 1.0.
    # mean 0.75 only moves if the ranking genuinely changes (not ceiling-saturated).
    report = evaluate_retrieval(knowledge, _golden(), k=1)
    assert report.aggregate("recall_at_k") == 0.75
    assert report.aggregate("reciprocal_rank") == 1.0  # top-1 is relevant for both


def test_retrieval_example_requires_relevant_sources():
    with pytest.raises(ValidationError):
        RetrievalExample(id="x", query="q", relevant_sources=[])
