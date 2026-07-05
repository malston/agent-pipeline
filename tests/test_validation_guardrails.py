"""A4 output guardrail: the hard gate. No brief leaves with a failed check.

This is the terminal grounding/policy/format gate -- the reason the pipeline can
call its output "validated".
"""
import pytest

from agent_pipeline.agents.guardrails import GuardrailViolation, validate_brief_output
from agent_pipeline.contracts.validation import ValidationChecks, ValidatedBrief


def _brief(grounding=True, policy=True, fmt=True):
    return ValidatedBrief(
        request_id="r1",
        body="Cells make ATP.",
        citations=["mito"],
        checks=ValidationChecks(grounding_ok=grounding, policy_ok=policy, format_ok=fmt),
    )


def test_gate_rejects_failed_grounding():
    with pytest.raises(GuardrailViolation) as exc:
        validate_brief_output(_brief(grounding=False))
    assert exc.value.code == "GROUNDING_FAILED"


def test_gate_rejects_failed_policy():
    with pytest.raises(GuardrailViolation) as exc:
        validate_brief_output(_brief(policy=False))
    assert exc.value.code == "POLICY_FAILED"


def test_gate_rejects_failed_format():
    with pytest.raises(GuardrailViolation) as exc:
        validate_brief_output(_brief(fmt=False))
    assert exc.value.code == "FORMAT_FAILED"


def test_gate_accepts_all_checks_passing():
    validate_brief_output(_brief())  # no raise
