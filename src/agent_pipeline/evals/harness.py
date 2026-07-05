"""The eval harness: run examples through a function, score the outputs, and bind
every score to the trace id of the run that produced it.

Local and keyless. A LangSmith adapter attaches these traces + scores as runs and
feedback-on-run when LANGSMITH_API_KEY is configured; the binding shape here mirrors
that model (score.trace_id <-> the run it scored).
"""
import uuid
from collections.abc import Callable, Iterable, Mapping
from statistics import mean
from typing import Any

from pydantic import BaseModel


class Trace(BaseModel):
    trace_id: str
    example_id: str
    output: Any


class Score(BaseModel):
    trace_id: str
    metric: str
    value: float


class EvalReport(BaseModel):
    traces: list[Trace]
    scores: list[Score]

    def aggregate(self, metric: str) -> float:
        """Mean of a metric across all runs."""
        values = [s.value for s in self.scores if s.metric == metric]
        if not values:
            raise ValueError(f"no scores recorded for metric '{metric}'")
        return mean(values)


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
        output = run_fn(example)
        traces.append(Trace(trace_id=trace_id, example_id=example.id, output=output))
        for name, metric_fn in metrics.items():
            scores.append(
                Score(trace_id=trace_id, metric=name, value=metric_fn(output, example))
            )
    return EvalReport(traces=traces, scores=scores)
