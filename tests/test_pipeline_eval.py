"""System-scope eval: the whole keyless pipeline, scored on citation quality.

Runs A1->A2->A3->A4 (keyless) on a golden request and scores whether the final
brief cites the relevant sources (recall) without over-citing irrelevant ones
(precision). Deterministic, real, keyless -- and it quantifies the pipeline's
tendency to carry irrelevant retrieved evidence through to the brief.
"""
import pytest
from pydantic import ValidationError

from agent_pipeline.evals.pipeline import SystemExample, evaluate_pipeline


def _golden():
    return [
        SystemExample(
            id="energy",
            request="how do cells make energy?",
            relevant_sources=["mito", "photo"],
        )
    ]


def test_pipeline_eval_scores_bind_to_traces(knowledge):
    report = evaluate_pipeline(knowledge, _golden())
    trace_ids = {t.trace_id for t in report.traces}
    assert trace_ids and all(s.trace_id in trace_ids for s in report.scores)
    assert len(report.scores) == 2  # 1 example x 2 metrics; catches a dropped metric


def test_pipeline_eval_measures_citation_recall_and_precision(knowledge):
    report = evaluate_pipeline(knowledge, _golden())
    # the brief cites both relevant sources (mito, photo)...
    assert report.aggregate("citation_recall") == 1.0
    # ...but the keyless pipeline carries the irrelevant 'econ' source through: 2 of 3
    # cited are relevant. Pinning the exact value catches both over- and under-citation.
    assert report.aggregate("citation_precision") == 2 / 3


def test_system_example_requires_relevant_sources():
    with pytest.raises(ValidationError):
        SystemExample(id="x", request="q", relevant_sources=[])
