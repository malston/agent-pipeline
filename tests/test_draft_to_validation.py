"""Context Translation for the A3 -> A4 boundary.

Cited sections become claims to verify; sections that cite nothing -- and gaps A2
never reported -- become uncited assertions (ungrounded, no grounding attempt); the
section text plus the acknowledged gaps become the brief body; cited source ids become
the available-sources set. Deterministic, Model-free.
"""
import pytest

from agent_pipeline.contracts.composition import Draft, Section
from agent_pipeline.contracts.validation import BriefInput
from agent_pipeline.agents.guardrails import GuardrailViolation
from agent_pipeline.translators.draft_to_validation import translate_draft_to_validation

_LEGIT_GAPS = ["No bacteria data."]

def test_empty_everything_draft_raises_empty_brief():
    # no sections and no gaps -> there is nothing to validate; fail with a domain-typed
    # guardrail rather than a generic pydantic error on the empty body.
    draft = Draft(request_id="r1", sections=[], gaps=[], style_profile="concise")
    with pytest.raises(GuardrailViolation) as exc:
        translate_draft_to_validation(draft, [])
    assert exc.value.code == "EMPTY_BRIEF"


def _draft():
    return Draft(
        request_id="r1",
        sections=[
            Section(heading="Energy", body="Cells make ATP.", cited_sources=["mito"]),
            Section(heading="Plants", body="Plants photosynthesize.", cited_sources=["photo", "mito"]),
        ],
        gaps=_LEGIT_GAPS,
        style_profile="concise",
    )


def test_translation_produces_valid_brief_input():
    out = translate_draft_to_validation(_draft(), _LEGIT_GAPS)
    assert isinstance(out, BriefInput)
    assert out.request_id == "r1"


def test_cited_sections_become_claims():
    out = translate_draft_to_validation(_draft(), _LEGIT_GAPS)
    assert [(c.text, c.sources) for c in out.claims] == [
        ("Cells make ATP.", ["mito"]),
        ("Plants photosynthesize.", ["photo", "mito"]),
    ]


def test_available_sources_are_deduped_in_first_seen_order():
    out = translate_draft_to_validation(_draft(), _LEGIT_GAPS)
    assert out.available_sources == ["mito", "photo"]


def test_body_includes_every_section_and_the_gaps():
    out = translate_draft_to_validation(_draft(), _LEGIT_GAPS)
    assert "Cells make ATP." in out.body
    assert "Plants photosynthesize." in out.body
    assert "No bacteria data." in out.body  # gaps render into the body


def test_gaps_are_neither_claims_nor_uncited_assertions():
    out = translate_draft_to_validation(_draft(), _LEGIT_GAPS)
    assert len(out.claims) == 2
    assert out.uncited_assertions == []


def test_uncited_content_section_becomes_an_uncited_assertion():
    draft = Draft(
        request_id="r1",
        sections=[
            Section(heading="Grounded", body="Cells make ATP.", cited_sources=["mito"]),
            Section(heading="Floating", body="Cells also feel joy.", cited_sources=[]),
        ],
        style_profile="concise",
    )
    out = translate_draft_to_validation(draft, [])
    assert [(c.text, c.sources) for c in out.claims] == [("Cells make ATP.", ["mito"])]
    assert out.uncited_assertions == ["Cells also feel joy."]
    assert "Cells also feel joy." in out.body  # still ships in the body


def test_gaps_only_draft_yields_no_claims_or_assertions_but_a_body():
    draft = Draft(request_id="r1", sections=[], gaps=["No data at all."], style_profile="concise")
    out = translate_draft_to_validation(draft, ["No data at all."])
    assert out.claims == []
    assert out.uncited_assertions == []
    assert out.available_sources == []
    assert "No data at all." in out.body


def test_fabricated_gap_becomes_an_uncited_assertion():
    # a gap A2 never reported is unbacked body text -> an uncited assertion; a legitimate
    # gap (in legitimate_gaps) is not.
    draft = Draft(
        request_id="r1",
        sections=[Section(heading="Energy", body="Cells make ATP.", cited_sources=["mito"])],
        gaps=["No bacteria data.", "Mitochondria secretly plot."],
        style_profile="concise",
    )
    out = translate_draft_to_validation(draft, legitimate_gaps=["No bacteria data."])
    assert out.uncited_assertions == ["Mitochondria secretly plot."]  # only the fabricated one
    assert [(c.text, c.sources) for c in out.claims] == [("Cells make ATP.", ["mito"])]
    assert "Mitochondria secretly plot." in out.body  # still ships in the body
