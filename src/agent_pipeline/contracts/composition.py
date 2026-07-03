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
    """A section of the draft; may cite the sources it draws on."""

    heading: str = Field(min_length=1)
    body: str = Field(min_length=1)
    cited_sources: list[str] = []  # source ids


class Draft(BaseModel):
    """A3's output: the composed sections plus the style they follow."""

    request_id: str
    sections: list[Section]
    style_profile: str
