"""A2 Analyst = Model (an analyst) + Harness (Plan-Execute loop + guardrails).

The analyst reasons over the evidence pool A1 handed it and emits evidence-bound
findings. Two analysts share one seam:

* ``RuleBasedAnalyst`` -- keyless, deterministic stand-in; treats each evidence
  item as a finding citing itself. Lets the agent run with no provider.
* ``LLMAnalyst`` -- provider-agnostic (build_model seam); the real Model. Needs a
  provider key, so it is exercised only by the gated end-to-end test.
"""
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from agent_pipeline.agents.plan import Plan, PlanStep
from agent_pipeline.agents.guardrails import (
    GuardrailViolation,
    validate_plan,
    validate_analysis_output,
)
from agent_pipeline.config import A2_TOOL_GRANT, A2_MAX_PLAN_STEPS
from agent_pipeline.model import build_model
from agent_pipeline.contracts.analysis import (
    AnalystInput,
    Finding,
    AnalysisReport,
)


class AnalysisPlan(BaseModel):
    """The A2 Model's output: the auditable step list plus the analysis payload."""

    steps: list[PlanStep]
    findings: list[Finding]
    gaps: list[str] = []


class Analyst(Protocol):
    def analyze(self, analyst_input: AnalystInput) -> AnalysisPlan: ...


class RuleBasedAnalyst:
    """Keyless stand-in: each evidence item becomes a finding citing itself."""

    def analyze(self, analyst_input: AnalystInput) -> AnalysisPlan:
        # The claim is the evidence verbatim, so it is trivially fully supported:
        # confidence 1.0. This stand-in does no real scoring.
        findings = [
            Finding(claim=item.text, evidence=[item.id], confidence=1.0)
            for item in analyst_input.evidence_pool
        ]
        gaps = [] if findings else ["no evidence retrieved for the question"]
        return AnalysisPlan(
            steps=[PlanStep(step_id=0, intent="emit the report", tool="emit_contract")],
            findings=findings,
            gaps=gaps,
        )


class LLMAnalyst:
    """Provider-agnostic Model analyst. Receives the model by injection."""

    _SYSTEM = (
        "You are the analyst in a RAG pipeline. Given a question and a pool of "
        "evidence (each with an id), extract findings: each finding is a claim with "
        "the evidence ids that support it (cite ONLY ids present in the pool) and a "
        "confidence in [0,1]. List gaps the evidence cannot answer. You analyze the "
        "given evidence only -- you have no retrieval tools. Produce a plan whose "
        "single step uses the tool emit_contract."
    )

    def __init__(self, model: BaseChatModel | None = None) -> None:
        base = model if model is not None else build_model()
        self._model = base.with_structured_output(AnalysisPlan)

    def analyze(self, analyst_input: AnalystInput) -> AnalysisPlan:
        return self._model.invoke(
            [("system", self._SYSTEM), ("human", analyst_input.model_dump_json())]
        )


class A2Analyst:
    def __init__(
        self,
        analyst: Analyst,
        max_plan_steps: int = A2_MAX_PLAN_STEPS,
    ) -> None:
        self._analyst = analyst
        self._max_steps = max_plan_steps

    def run(self, analyst_input: AnalystInput) -> AnalysisReport:
        plan = self._analyst.analyze(analyst_input)  # PLAN (Model)
        validate_plan(Plan(steps=plan.steps), A2_TOOL_GRANT, self._max_steps)  # guardrail

        evidence_ids = {item.id for item in analyst_input.evidence_pool}
        for step in plan.steps:  # EXECUTE
            if step.tool == "emit_contract":
                report = AnalysisReport(
                    request_id=analyst_input.request_id,
                    findings=plan.findings,
                    gaps=plan.gaps,
                )
                validate_analysis_output(report, evidence_ids)  # guardrail
                return report
        # Unreachable: validate_plan guarantees a terminal emit_contract. Defensive
        # backstop so a future weakening of that guard fails loudly, not silently.
        raise GuardrailViolation("NO_EMIT", "plan executed without emitting a contract")
