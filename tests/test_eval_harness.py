"""The eval harness: run examples, score outputs, bind every score to a trace id.

Trace-id binding is the design's requirement (evals attach to the trace of the run
that produced them). It mirrors LangSmith's feedback-on-run shape so tracing can be
layered on later without reshaping scores.
"""
import pytest
from pydantic import BaseModel, ValidationError

from agent_pipeline.evals.harness import evaluate, EvalReport, Trace, Score


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


def test_aggregate_raises_on_a_metric_with_no_scores():
    report = evaluate(_dataset(), run_fn=lambda ex: ex.text, metrics={})
    with pytest.raises(ValueError):
        report.aggregate("length")


def test_evaluate_isolates_per_example_failures():
    # one example's run raises; the batch continues and records the failure
    def run(ex):
        if ex.id == "e1":
            raise RuntimeError("boom")
        return ex.text

    report = evaluate(
        _dataset(), run_fn=run, metrics={"length": lambda out, ex: float(len(out))}
    )
    errored = [t for t in report.traces if t.error is not None]
    assert [t.example_id for t in errored] == ["e1"]
    assert "boom" in errored[0].error
    # the healthy example was still scored
    assert [s.value for s in report.scores] == [4.0]  # only e2 ("abcd")


def test_report_rejects_scores_referencing_unknown_traces():
    with pytest.raises(ValidationError):
        EvalReport(
            traces=[Trace(trace_id="t1", example_id="e1", output=None)],
            scores=[Score(trace_id="ghost", metric="m", value=1.0)],
        )
