"""The eval harness: run examples through a function, score the outputs, and bind
every score to the trace id of the run that produced it.

Trace-id binding (score.trace_id <-> the run it scored) is chosen to match
LangSmith's runs + feedback-on-run shape, so a tracing adapter can be layered on
later without reshaping scores. That adapter is not part of this module.

A run that raises is isolated: its failure is recorded on the trace and the batch
continues, so one bad example does not suppress the rest (for a system eval, a
rejected brief is a result to count, not a reason to abort).
"""
import uuid
from collections.abc import Callable, Iterable, Mapping
from statistics import mean
from typing import Any

from pydantic import BaseModel, model_validator


class Trace(BaseModel):
    trace_id: str
    example_id: str
    output: Any = None
    error: str | None = None  # set instead of output when the run raised


class Score(BaseModel):
    trace_id: str
    metric: str
    value: float


class EvalReport(BaseModel):
    traces: list[Trace]
    scores: list[Score]

    @model_validator(mode="after")
    def _scores_reference_known_traces(self) -> "EvalReport":
        known = {t.trace_id for t in self.traces}
        dangling = {s.trace_id for s in self.scores} - known
        if dangling:
            raise ValueError(f"scores reference unknown trace ids: {sorted(dangling)}")
        return self

    def aggregate(self, metric: str) -> float:
        """Mean of a metric over the runs that recorded it."""
        values = [s.value for s in self.scores if s.metric == metric]
        if not values:
            raise ValueError(f"no scores recorded for metric '{metric}'")
        return mean(values)

    @property
    def errors(self) -> list[Trace]:
        """Traces whose run raised."""
        return [t for t in self.traces if t.error is not None]


def _new_trace_id() -> str:
    return str(uuid.uuid4())


def evaluate(
    dataset: Iterable[Any],
    run_fn: Callable[[Any], Any],
    metrics: Mapping[str, Callable[[Any, Any], float]],
    trace_id_factory: Callable[[], str] = _new_trace_id,
) -> EvalReport:
    traces: list[Trace] = []
    scores: list[Score] = []
    for example in dataset:
        trace_id = trace_id_factory()
        try:
            output = run_fn(example)
        except Exception as exc:
            # Record the failure and keep going -- surfaced in report.errors, not swallowed.
            traces.append(
                Trace(
                    trace_id=trace_id,
                    example_id=example.id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        traces.append(Trace(trace_id=trace_id, example_id=example.id, output=output))
        for name, metric_fn in metrics.items():
            scores.append(
                Score(trace_id=trace_id, metric=name, value=metric_fn(output, example))
            )
    return EvalReport(traces=traces, scores=scores)
