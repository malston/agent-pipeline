"""Retrieval-quality metrics for evaluating A1 (the RAG gate)."""
import pytest

from agent_pipeline.evals.metrics import (
    recall_at_k,
    reciprocal_rank,
    set_recall,
    set_precision,
)


def test_recall_at_k_counts_relevant_within_top_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=2) == 0.5  # only 'a' in top-2
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 1.0  # both in top-3


def test_recall_at_k_handles_fewer_results_than_k():
    assert recall_at_k(["a"], {"a", "b"}, k=5) == 0.5


def test_recall_at_k_requires_relevant_set():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=3)


def test_recall_at_k_rejects_non_positive_k():
    # k<=0 is a misconfiguration, not a measurement (negative k silently truncates)
    with pytest.raises(ValueError):
        recall_at_k(["a", "b"], {"a"}, k=0)
    with pytest.raises(ValueError):
        recall_at_k(["a", "b"], {"a"}, k=-1)


def test_reciprocal_rank_of_first_relevant():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5  # rank 2
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0  # rank 1


def test_reciprocal_rank_zero_when_none_relevant():
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_set_recall_and_precision():
    # predicted {a,b,x} vs relevant {a,b,c}: recall 2/3, precision 2/3
    assert set_recall(["a", "b", "x"], {"a", "b", "c"}) == 2 / 3
    assert set_precision(["a", "b", "x"], {"a", "b", "c"}) == 2 / 3


def test_set_precision_of_empty_prediction_is_one():
    # predicting nothing makes no false claims
    assert set_precision([], {"a"}) == 1.0


def test_set_recall_requires_relevant_set():
    with pytest.raises(ValueError):
        set_recall(["a"], set())


def test_set_metrics_are_deduped_but_reciprocal_rank_is_positional():
    # set metrics treat inputs as sets: duplicates don't change the denominator
    assert set_precision(["a", "a"], {"a"}) == 1.0
    # reciprocal_rank scans positionally and does NOT dedup: 'b' sits at rank 3
    assert reciprocal_rank(["a", "a", "b"], {"b"}) == 1 / 3
