"""Context Translation for the A3 -> A4 boundary.

Cited sections become claims to verify; the assembled section text becomes the
brief body; cited source ids become the available-sources set. Uncited sections
(intro/gaps) contribute to the body but are not claims. Deterministic, Model-free.
"""
from agent_pipeline.contracts.composition import Draft, Section
from agent_pipeline.contracts.validation import BriefInput
from agent_pipeline.translators.draft_to_validation import translate_draft_to_validation


def _draft():
    return Draft(
        request_id="r1",
        sections=[
            Section(heading="Energy", body="Cells make ATP.", cited_sources=["mito"]),
            Section(heading="Plants", body="Plants photosynthesize.", cited_sources=["photo", "mito"]),
            Section(heading="Open questions", body="No bacteria data.", cited_sources=[]),
        ],
        style_profile="concise",
    )


def test_translation_produces_valid_brief_input():
    out = translate_draft_to_validation(_draft())
    assert isinstance(out, BriefInput)
    assert out.request_id == "r1"


def test_cited_sections_become_claims_uncited_do_not():
    out = translate_draft_to_validation(_draft())
    assert [(c.text, c.sources) for c in out.claims] == [
        ("Cells make ATP.", ["mito"]),
        ("Plants photosynthesize.", ["photo", "mito"]),
    ]


def test_available_sources_are_deduped_in_first_seen_order():
    out = translate_draft_to_validation(_draft())
    assert out.available_sources == ["mito", "photo"]


def test_body_includes_every_section():
    out = translate_draft_to_validation(_draft())
    assert "Cells make ATP." in out.body
    assert "Plants photosynthesize." in out.body
    assert "No bacteria data." in out.body  # uncited section still in the body
