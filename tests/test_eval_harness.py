"""The eval harness: run examples, score outputs, bind every score to a trace id.

Trace-id binding is the design's requirement (evals attach to the trace of the run
that produced them); the LangSmith adapter later pushes these as feedback-on-run.
"""
from pydantic import BaseModel

from agent_pipeline.evals.harness import evaluate


class _Example(BaseModel):
    id: str
    text: str


def _dataset():
    return [_Example(id="e1", text="ab"), _Example(id="e2", text="abcd")]


def test_evaluate_binds_each_score_to_its_run_trace_id():
    trace_ids = iter(["t1", "t2"])
    report = evaluate(
        _dataset(),
        run_fn=lambda ex: ex.text,
        metrics={"length": lambda out, ex: float(len(out))},
        trace_id_factory=lambda: next(trace_ids),
    )
    # each score carries the trace id of the run that produced it
    assert {(s.trace_id, s.metric, s.value) for s in report.scores} == {
        ("t1", "length", 2.0),
        ("t2", "length", 4.0),
    }
    # traces link trace id -> example
    assert {t.trace_id: t.example_id for t in report.traces} == {"t1": "e1", "t2": "e2"}


def test_report_aggregates_mean_of_a_metric():
    report = evaluate(
        _dataset(),
        run_fn=lambda ex: ex.text,
        metrics={"length": lambda out, ex: float(len(out))},
    )
    assert report.aggregate("length") == 3.0  # mean(2, 4)


def test_trace_ids_are_unique_per_run_by_default():
    report = evaluate(_dataset(), run_fn=lambda ex: ex.text, metrics={})
    assert len({t.trace_id for t in report.traces}) == 2
