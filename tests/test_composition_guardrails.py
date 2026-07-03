"""A3 output guardrail: a section may only cite sources available to the composer.

The composition-stage analogue of A1/A2's grounding guards -- it stops the
composer inventing citations the analysis never supplied.
"""
import pytest

from agent_pipeline.agents.guardrails import (
    GuardrailViolation,
    validate_composition_output,
)
from agent_pipeline.contracts.composition import Section, Draft


def _draft(cited):
    return Draft(
        request_id="r1",
        sections=[Section(heading="Energy", body="Cells make ATP.", cited_sources=cited)],
        style_profile="concise",
    )


def test_composition_rejects_section_citing_unavailable_source():
    with pytest.raises(GuardrailViolation) as exc:
        validate_composition_output(_draft(["ghost"]), available_sources={"mito", "photo"})
    assert exc.value.code == "UNGROUNDED_SECTION"


def test_composition_accepts_sections_citing_available_sources():
    validate_composition_output(_draft(["mito"]), available_sources={"mito", "photo"})


def test_composition_accepts_section_that_cites_nothing():
    validate_composition_output(_draft([]), available_sources={"mito"})  # no raise
