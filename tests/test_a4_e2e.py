"""End-to-end A4 with the REAL semantic verifier (LLMClaimVerifier).

Real LLM judges whether the cited source text supports the claim. Real knowledge
store, real gate. Skips (never mocks) without a provider key.
"""
import os

import pytest

from agent_pipeline.agents.validator import A4Validator, LLMClaimVerifier
from agent_pipeline.contracts.validation import Claim, BriefInput


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="no ANTHROPIC_API_KEY; real-LLM e2e skipped (not mocked)",
)
def test_a4_semantically_verifies_a_supported_claim(knowledge):
    # 'mito' in the shared corpus is about the mitochondrion producing ATP, so the
    # Model should judge this claim supported.
    brief_input = BriefInput(
        request_id="r1",
        claims=[Claim(text="The mitochondrion produces ATP.", sources=["mito"])],
        body="The mitochondrion produces ATP, the cell's energy currency.",
        available_sources=["mito"],
    )
    brief = A4Validator(LLMClaimVerifier(knowledge)).run(brief_input)
    assert brief.checks.grounding_ok


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="no ANTHROPIC_API_KEY; real-LLM e2e skipped (not mocked)",
)
def test_a4_gates_a_claim_the_sources_do_not_support(knowledge):
    # 'econ' is about central banks and interest rates -- it does not support a
    # claim about cellular energy, so the Model should judge it unsupported and the
    # gate should reject the brief.
    from agent_pipeline.agents.guardrails import GuardrailViolation

    brief_input = BriefInput(
        request_id="r1",
        claims=[Claim(text="Mitochondria generate ATP for the cell.", sources=["econ"])],
        body="Mitochondria generate ATP for the cell.",
        available_sources=["econ"],
    )
    with pytest.raises(GuardrailViolation) as exc:
        A4Validator(LLMClaimVerifier(knowledge)).run(brief_input)
    assert exc.value.code == "GROUNDING_FAILED"
