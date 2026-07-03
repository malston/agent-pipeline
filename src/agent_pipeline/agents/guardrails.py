"""Harness guardrails: gates an agent's plan and output must clear.

A ``GuardrailViolation`` is a typed, traceable failure -- never a silent pass.
"""
from agent_pipeline.agents.plan import Plan
from agent_pipeline.contracts.retrieval import RetrievalBundle
from agent_pipeline.contracts.analysis import AnalysisReport

TERMINAL_TOOL = "emit_contract"


class GuardrailViolation(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def validate_plan(plan: Plan, allowed_tools: set[str], max_steps: int) -> None:
    """Reject plans that use ungranted tools, exceed the step budget, or fail to
    terminate by emitting the contract."""
    if len(plan.steps) > max_steps:
        raise GuardrailViolation(
            "PLAN_TOO_LONG",
            f"plan has {len(plan.steps)} steps, budget is {max_steps}",
        )
    for step in plan.steps:
        if step.tool not in allowed_tools:
            raise GuardrailViolation(
                "TOOL_NOT_GRANTED",
                f"step {step.step_id} calls '{step.tool}', not in agent grant",
            )
    if not plan.steps or plan.steps[-1].tool != TERMINAL_TOOL:
        raise GuardrailViolation(
            "MISSING_TERMINAL_EMIT",
            f"plan must end with '{TERMINAL_TOOL}'",
        )


def validate_retrieval_output(
    bundle: RetrievalBundle, known_source_ids: set[str]
) -> None:
    """Reject a bundle that cites a source A1 did not actually retrieve."""
    for passage in bundle.passages:
        if passage.source_id not in known_source_ids:
            raise GuardrailViolation(
                "UNGROUNDED_CITATION",
                f"passage cites unknown source '{passage.source_id}'",
            )


def validate_analysis_output(
    report: AnalysisReport, evidence_ids: set[str]
) -> None:
    """Reject a report whose findings cite evidence outside the input pool."""
    for finding in report.findings:
        for source_id in finding.evidence:
            if source_id not in evidence_ids:
                raise GuardrailViolation(
                    "UNGROUNDED_FINDING",
                    f"finding cites evidence '{source_id}' not in the input pool",
                )
