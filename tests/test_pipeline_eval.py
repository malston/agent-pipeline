"""System-scope eval: the whole keyless pipeline, scored on citation quality.

Runs A1->A2->A3->A4 (keyless) on a golden request and scores whether the final
brief cites the relevant sources (recall) without over-citing irrelevant ones
(precision). Deterministic, real, keyless -- and it quantifies the pipeline's
tendency to carry irrelevant retrieved evidence through to the brief.
"""
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


def test_pipeline_eval_measures_citation_recall_and_precision(knowledge):
    report = evaluate_pipeline(knowledge, _golden())
    # the brief cites both relevant sources...
    assert report.aggregate("citation_recall") == 1.0
    # ...but the keyless pipeline also carries the irrelevant 'econ' source through,
    # so precision is below 1.0 -- the eval makes that visible
    assert report.aggregate("citation_precision") < 1.0
