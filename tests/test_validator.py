"""A4 Validator: Model-backed check_claim + Harness gate.

The keyless StructuralClaimVerifier checks that each claim's sources are among the
available sources (defense-in-depth over the upstream grounding). Policy and format
are deterministic. The gate rejects any brief with a failed check. The semantic
LLM verifier is exercised in test_a4_e2e.py, gated on a provider key.
"""
import pytest

from agent_pipeline.agents.validator import A4Validator, StructuralClaimVerifier
from agent_pipeline.agents.guardrails import GuardrailViolation
from agent_pipeline.contracts.validation import Claim, BriefInput, ValidatedBrief


def _input(claim_sources=("mito",), available=("mito", "photo"), body="Cells make ATP."):
    return BriefInput(
        request_id="r1",
        claims=[Claim(text="Cells make ATP.", sources=list(claim_sources))],
        body=body,
        available_sources=list(available),
    )


def test_a4_validates_grounded_brief():
    brief = A4Validator(StructuralClaimVerifier()).run(_input())
    assert isinstance(brief, ValidatedBrief)
    assert brief.checks.grounding_ok and brief.checks.policy_ok and brief.checks.format_ok
    assert brief.citations == ["mito", "photo"]


def test_a4_gates_ungrounded_claim():
    # claim cites a source not in the available set
    with pytest.raises(GuardrailViolation) as exc:
        A4Validator(StructuralClaimVerifier()).run(
            _input(claim_sources=("ghost",), available=("mito",))
        )
    assert exc.value.code == "GROUNDING_FAILED"


def test_a4_gates_policy_violation():
    with pytest.raises(GuardrailViolation) as exc:
        A4Validator(StructuralClaimVerifier(), banned_phrases=frozenset({"forbidden"})).run(
            _input(body="This body says forbidden things.")
        )
    assert exc.value.code == "POLICY_FAILED"


def test_a4_gates_malformed_body():
    with pytest.raises(GuardrailViolation) as exc:
        A4Validator(StructuralClaimVerifier()).run(_input(body="   "))
    assert exc.value.code == "FORMAT_FAILED"
