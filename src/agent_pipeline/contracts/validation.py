"""Contracts owned by the A4 Validator boundary.

``BriefInput`` is A4's input contract -- the target vocabulary the A3->A4
translator maps ``Draft`` into. ``ValidatedBrief`` is the pipeline's final output.
"""
from pydantic import BaseModel, Field


class Claim(BaseModel):
    """A statement A4 must verify, bound to the sources meant to support it."""

    text: str = Field(min_length=1)
    sources: list[str] = Field(min_length=1)  # source ids


class ValidationChecks(BaseModel):
    """The gate results A4 records on the brief."""

    grounding_ok: bool
    policy_ok: bool
    format_ok: bool


class BriefInput(BaseModel):
    """A4's input: the claims to verify, the assembled body, and the source ids
    legitimately available to cite."""

    request_id: str
    claims: list[Claim]
    body: str = Field(min_length=1)
    available_sources: list[str]


class ValidatedBrief(BaseModel):
    """The pipeline's final output: the brief plus the checks that cleared it."""

    request_id: str
    body: str = Field(min_length=1)
    citations: list[str]  # source ids the claims actually cite
    checks: ValidationChecks
