"""Contracts owned by the A3 Composer boundary.

``ComposerInput`` is A3's input contract -- the target vocabulary the A2->A3
translator maps ``AnalysisReport`` into. ``Draft`` is A3's output.
"""
from pydantic import BaseModel, Field


class Point(BaseModel):
    """A point to make in the draft, bound to its supporting sources."""

    statement: str = Field(min_length=1)
    sources: list[str] = Field(min_length=1)  # source ids
    confidence: float = Field(ge=0.0, le=1.0)


class ComposerInput(BaseModel):
    """A3's input: the points to compose from and the gaps to acknowledge."""

    request_id: str
    points: list[Point]
    gaps: list[str] = []


class Section(BaseModel):
    """A section of the draft: an assertion, with the source ids that support it."""

    heading: str = Field(min_length=1)
    body: str = Field(min_length=1)
    # source ids; empty marks an assertion with no grounding attempt, which A4 rejects
    # (fed back to A3 to recompose). Gaps that assert nothing live in Draft.gaps, not here.
    cited_sources: list[str] = []


class Draft(BaseModel):
    """A3's output: the composed sections, the acknowledged gaps, and the style."""

    request_id: str
    sections: list[Section]
    gaps: list[str] = []
    style_profile: str = Field(min_length=1)
