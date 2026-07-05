"""A4's contracts: Claim/BriefInput (input) and ValidationChecks/ValidatedBrief (output).

A claim must state something and cite a source. ValidatedBrief is the pipeline's
final deliverable, carrying the checks A4 ran.
"""
import pytest
from pydantic import ValidationError

from agent_pipeline.contracts.validation import (
    Claim,
    ValidationChecks,
    BriefInput,
    ValidatedBrief,
)


def test_claim_requires_nonempty_text():
    with pytest.raises(ValidationError):
        Claim(text="", sources=["mito"])


def test_claim_requires_at_least_one_source():
    with pytest.raises(ValidationError):
        Claim(text="cells make ATP", sources=[])


def test_brief_input_carries_claims_body_and_available_sources():
    bi = BriefInput(
        request_id="r1",
        claims=[Claim(text="cells make ATP", sources=["mito"])],
        body="Cells make ATP in mitochondria.",
        available_sources=["mito", "photo"],
    )
    assert bi.claims[0].sources == ["mito"]
    assert bi.available_sources == ["mito", "photo"]


def test_brief_input_rejects_empty_uncited_assertion():
    # an empty-string "assertion" would put "" into unsupported and fail grounding on
    # nothing; uncited assertions are section bodies, which are always non-empty
    with pytest.raises(ValidationError):
        BriefInput(
            request_id="r1",
            claims=[Claim(text="cells make ATP", sources=["mito"])],
            body="Cells make ATP.",
            available_sources=["mito"],
            uncited_assertions=[""],
        )


def test_brief_input_carries_uncited_assertions():
    bi = BriefInput(
        request_id="r1",
        claims=[Claim(text="cells make ATP", sources=["mito"])],
        body="Cells make ATP.",
        available_sources=["mito"],
        uncited_assertions=["a section that cites nothing"],
    )
    assert bi.uncited_assertions == ["a section that cites nothing"]


def test_validated_brief_rejects_empty_body():
    with pytest.raises(ValidationError):
        ValidatedBrief(
            request_id="r1",
            body="",
            citations=["mito"],
            checks=ValidationChecks(grounding_ok=True, policy_ok=True, format_ok=True),
        )


def test_validated_brief_round_trip():
    brief = ValidatedBrief(
        request_id="r1",
        body="Cells make ATP.",
        citations=["mito"],
        checks=ValidationChecks(grounding_ok=True, policy_ok=True, format_ok=True),
    )
    restored = ValidatedBrief.model_validate_json(brief.model_dump_json())
    assert restored == brief
    assert restored.checks.grounding_ok is True
