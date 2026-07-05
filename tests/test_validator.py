"""A4 Validator: Model-backed check_claim + Harness gate.

The keyless StructuralClaimVerifier checks that each claim's sources are among the
available sources (defense-in-depth over the upstream grounding). Policy and format
are deterministic. The gate rejects any brief with a failed check. The semantic
LLM verifier is exercised in test_a4_e2e.py, gated on a provider key.
"""
import pytest
from pydantic import ValidationError

from agent_pipeline.agents.validator import (
    A4Validator,
    StructuralClaimVerifier,
    LLMClaimVerifier,
    ValidationOutcome,
)
from agent_pipeline.agents.guardrails import GuardrailViolation
from agent_pipeline.contracts.validation import (
    Claim,
    BriefInput,
    ValidatedBrief,
    ValidationChecks,
)


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
    # citations are the sources the claims actually cite, not the whole pool
    assert brief.citations == ["mito"]


def test_a4_gates_when_one_of_several_claims_is_ungrounded():
    bi = BriefInput(
        request_id="r1",
        claims=[
            Claim(text="grounded", sources=["mito"]),
            Claim(text="not grounded", sources=["ghost"]),
        ],
        body="Some body.",
        available_sources=["mito", "photo"],
    )
    with pytest.raises(GuardrailViolation) as exc:
        A4Validator(StructuralClaimVerifier()).run(bi)
    assert exc.value.code == "GROUNDING_FAILED"


def test_a4_treats_a_claimless_brief_as_grounded():
    # A gaps-only brief has nothing to verify; grounding is vacuously satisfied.
    bi = BriefInput(request_id="r1", claims=[], body="Only open questions.", available_sources=[])
    brief = A4Validator(StructuralClaimVerifier()).run(bi)
    assert brief.checks.grounding_ok is True
    assert brief.citations == []


def test_a4_policy_check_is_case_insensitive():
    with pytest.raises(GuardrailViolation) as exc:
        A4Validator(StructuralClaimVerifier(), banned_phrases=frozenset({"forbidden"})).run(
            _input(body="This body shouts FORBIDDEN loudly.")
        )
    assert exc.value.code == "POLICY_FAILED"


def test_a4_rejects_empty_banned_phrase_at_construction():
    with pytest.raises(ValueError):
        A4Validator(StructuralClaimVerifier(), banned_phrases=frozenset({""}))


def test_llm_verifier_rejects_claim_citing_outside_pool_without_calling_model(knowledge):
    # Offline: the subset short-circuit returns False before any Model call.
    verifier = LLMClaimVerifier(knowledge)
    assert verifier.verify(Claim(text="x", sources=["ghost"]), {"mito"}) is False


def test_llm_verifier_raises_on_unresolvable_source(knowledge):
    # Offline: source passes the subset check but is not in the store -> infra fault,
    # a distinct SOURCE_UNRESOLVED, not a content GROUNDING_FAILED.
    verifier = LLMClaimVerifier(knowledge)
    with pytest.raises(GuardrailViolation) as exc:
        verifier.verify(Claim(text="x", sources=["phantom"]), {"phantom"})
    assert exc.value.code == "SOURCE_UNRESOLVED"


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


def test_a4_check_reports_unsupported_without_raising():
    # a claim citing a source outside the pool is unsupported (structural verifier)
    bi = BriefInput(
        request_id="r1",
        claims=[
            Claim(text="grounded claim", sources=["mito"]),
            Claim(text="ungrounded claim", sources=["ghost"]),
        ],
        body="Some body.",
        available_sources=["mito", "photo"],
    )
    outcome = A4Validator(StructuralClaimVerifier()).check(bi)
    assert outcome.brief.checks.grounding_ok is False
    assert outcome.unsupported == ["ungrounded claim"]  # only the failing claim's text


def test_a4_check_reports_grounded_for_a_good_brief():
    outcome = A4Validator(StructuralClaimVerifier()).check(_input())
    assert outcome.brief.checks.grounding_ok is True
    assert outcome.unsupported == []


def test_a4_check_propagates_source_unresolved():
    class _UnresolvedVerifier:
        def verify(self, claim, available_sources):
            raise GuardrailViolation("SOURCE_UNRESOLVED", "missing doc")

    with pytest.raises(GuardrailViolation) as exc:
        A4Validator(_UnresolvedVerifier()).check(_input())
    assert exc.value.code == "SOURCE_UNRESOLVED"


def _brief(grounding_ok: bool) -> ValidatedBrief:
    return ValidatedBrief(
        request_id="r1",
        body="Cells make ATP.",
        citations=["mito"],
        checks=ValidationChecks(grounding_ok=grounding_ok, policy_ok=True, format_ok=True),
    )


def test_validation_outcome_rejects_grounded_brief_with_unsupported_claims():
    # grounding_ok True but a non-empty witness set is a self-contradiction the loop
    # would misread (route to the gate while still feeding claims back to A3).
    with pytest.raises(ValidationError):
        ValidationOutcome(brief=_brief(grounding_ok=True), unsupported=["x"])


def test_validation_outcome_rejects_ungrounded_brief_with_no_unsupported_claims():
    with pytest.raises(ValidationError):
        ValidationOutcome(brief=_brief(grounding_ok=False), unsupported=[])


def test_validation_outcome_accepts_the_two_consistent_states():
    ValidationOutcome(brief=_brief(grounding_ok=True), unsupported=[])
    ValidationOutcome(brief=_brief(grounding_ok=False), unsupported=["x"])
