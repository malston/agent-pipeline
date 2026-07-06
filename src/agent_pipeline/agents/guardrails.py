"""Harness guardrails: gates an agent's plan and output must clear.

A ``GuardrailViolation`` is a typed, traceable failure -- never a silent pass.
"""
from agent_pipeline.agents.plan import Plan
from agent_pipeline.contracts.retrieval import RetrievalBundle
from agent_pipeline.contracts.analysis import AnalysisReport
from agent_pipeline.contracts.composition import Draft
from agent_pipeline.contracts.validation import ValidatedBrief

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
    for step in plan.steps[:-1]:
        if step.tool == TERMINAL_TOOL:
            raise GuardrailViolation(
                "PREMATURE_EMIT",
                f"step {step.step_id} emits before the plan terminates; the "
                f"executor returns on the first '{TERMINAL_TOOL}', skipping later steps",
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


def validate_composition_output(
    draft: Draft, available_sources: set[str]
) -> None:
    """Reject a draft whose sections cite sources not available to the composer, or
    that composed nothing from a non-empty point set (an empty input legitimately
    yields an empty draft, so the check is conditional on there being sources)."""
    if available_sources and not draft.sections:
        raise GuardrailViolation(
            "EMPTY_DRAFT", "no sections composed from a non-empty point set"
        )
    for section in draft.sections:
        for source_id in section.cited_sources:
            if source_id not in available_sources:
                raise GuardrailViolation(
                    "UNGROUNDED_SECTION",
                    f"section cites source '{source_id}' not available to compose from",
                )


def validate_brief_output(brief: ValidatedBrief) -> None:
    """The terminal gate: no brief leaves with a failed check."""
    checks = brief.checks
    if not checks.grounding_ok:
        raise GuardrailViolation("GROUNDING_FAILED", "a claim is not supported by its sources")
    if not checks.policy_ok:
        raise GuardrailViolation("POLICY_FAILED", "the brief violates a policy rule")
    if not checks.format_ok:
        raise GuardrailViolation("FORMAT_FAILED", "the brief is not well-formed")
