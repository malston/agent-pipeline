"""Context Translation for the A3 -> A4 boundary.

Cited sections become claims to verify; sections that cite nothing become uncited
assertions (ungrounded, with no grounding attempt); the assembled section text plus
the acknowledged gaps become the brief body; cited source ids become the
available-sources set. Deterministic, Model-free.
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
        ],
        gaps=["No bacteria data."],
        style_profile="concise",
    )


def test_translation_produces_valid_brief_input():
    out = translate_draft_to_validation(_draft())
    assert isinstance(out, BriefInput)
    assert out.request_id == "r1"


def test_cited_sections_become_claims():
    out = translate_draft_to_validation(_draft())
    assert [(c.text, c.sources) for c in out.claims] == [
        ("Cells make ATP.", ["mito"]),
        ("Plants photosynthesize.", ["photo", "mito"]),
    ]


def test_available_sources_are_deduped_in_first_seen_order():
    out = translate_draft_to_validation(_draft())
    assert out.available_sources == ["mito", "photo"]


def test_body_includes_every_section_and_the_gaps():
    out = translate_draft_to_validation(_draft())
    assert "Cells make ATP." in out.body
    assert "Plants photosynthesize." in out.body
    assert "No bacteria data." in out.body  # gaps render into the body


def test_gaps_are_neither_claims_nor_uncited_assertions():
    out = translate_draft_to_validation(_draft())
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
    out = translate_draft_to_validation(draft)
    assert [(c.text, c.sources) for c in out.claims] == [("Cells make ATP.", ["mito"])]
    assert out.uncited_assertions == ["Cells also feel joy."]
    assert "Cells also feel joy." in out.body  # still ships in the body


def test_gaps_only_draft_yields_no_claims_or_assertions_but_a_body():
    draft = Draft(request_id="r1", sections=[], gaps=["No data at all."], style_profile="concise")
    out = translate_draft_to_validation(draft)
    assert out.claims == []
    assert out.uncited_assertions == []
    assert out.available_sources == []
    assert "No data at all." in out.body
