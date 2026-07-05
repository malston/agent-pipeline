"""A3's contracts: ComposerInput/Point (input vocabulary) and Section/Draft (output).

A point must state something and cite a source; a section must have a heading and
body. Citations are validated against the available sources by a guardrail, not the
type (the type cannot see the pool).
"""
import pytest
from pydantic import ValidationError

from agent_pipeline.contracts.composition import (
    Point,
    ComposerInput,
    Section,
    Draft,
)


def test_point_requires_nonempty_statement():
    with pytest.raises(ValidationError):
        Point(statement="", sources=["mito"], confidence=0.9)


def test_point_requires_at_least_one_source():
    with pytest.raises(ValidationError):
        Point(statement="cells make ATP", sources=[], confidence=0.9)


def test_section_requires_heading_and_body():
    with pytest.raises(ValidationError):
        Section(heading="", body="text", cited_sources=["mito"])
    with pytest.raises(ValidationError):
        Section(heading="Energy", body="", cited_sources=["mito"])


def test_section_may_cite_nothing():
    section = Section(heading="Overview", body="A short intro.")
    assert section.cited_sources == []


def test_draft_rejects_empty_style_profile():
    with pytest.raises(ValidationError):
        Draft(request_id="r1", sections=[], style_profile="")


def test_draft_rejects_empty_gap():
    # a gap is a non-empty acknowledgment; an empty one would ship as unbacked whitespace
    with pytest.raises(ValidationError):
        Draft(request_id="r1", sections=[], gaps=[""], style_profile="x")


def test_draft_json_round_trip():
    draft = Draft(
        request_id="r1",
        sections=[Section(heading="Energy", body="Cells make ATP.", cited_sources=["mito"])],
        style_profile="concise-technical",
    )
    restored = Draft.model_validate_json(draft.model_dump_json())
    assert restored == draft


def test_draft_carries_gaps():
    draft = Draft(request_id="r1", sections=[], gaps=["no data on X"], style_profile="x")
    assert draft.gaps == ["no data on X"]


def test_composer_input_carries_points_and_gaps():
    ci = ComposerInput(
        request_id="r1",
        points=[Point(statement="cells make ATP", sources=["mito"], confidence=0.9)],
        gaps=["nothing on bacteria"],
    )
    assert ci.points[0].sources == ["mito"]
    assert ci.gaps == ["nothing on bacteria"]
