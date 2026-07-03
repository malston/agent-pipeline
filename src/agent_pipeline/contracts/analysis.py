"""Contracts owned by the A2 Analyst boundary.

``AnalystInput`` is A2's input contract -- the target vocabulary that the
A1->A2 translator maps ``RetrievalBundle`` into.
"""
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A unit of evidence in the analyst's vocabulary."""

    id: str
    text: str


class AnalystInput(BaseModel):
    """A2's input: the question plus a pool of evidence to reason over."""

    request_id: str
    question: str
    evidence_pool: list[Evidence]
    retrieval_confidence: float = Field(ge=0.0, le=1.0)


class Finding(BaseModel):
    """A claim the analyst extracted, bound to the evidence that supports it."""

    claim: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)  # source ids; a claim must cite one
    confidence: float = Field(ge=0.0, le=1.0)


class AnalysisReport(BaseModel):
    """A2's output: evidence-bound findings plus the gaps it could not fill."""

    request_id: str
    findings: list[Finding]
    gaps: list[str] = []
