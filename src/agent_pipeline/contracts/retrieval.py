"""Contracts owned by the A1 Retriever boundary.

``RetrievalRequest`` is the pipeline entry point; ``RetrievalBundle`` is A1's
output contract handed (via a translator) to A2.
"""
from pydantic import BaseModel, Field


class Passage(BaseModel):
    """A retrieved span of a source document, with a relevance score."""

    source_id: str
    text: str
    score: float = Field(ge=0.0, le=1.0)
    span: str | None = None


class RetrievalRequest(BaseModel):
    """The raw request entering the pipeline."""

    request_id: str
    raw_query: str


class RetrievalBundle(BaseModel):
    """A1's output: the normalized query plus the evidence it retrieved."""

    request_id: str
    normalized_query: str
    passages: list[Passage]
    coverage: float = Field(ge=0.0, le=1.0)
