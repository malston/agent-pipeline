"""Context Translation for the A3 -> A4 boundary.

Maps the composition vocabulary (draft of sections) into the validation vocabulary
(claims to verify + an assembled body + the available source ids). A cited section
becomes a claim; an uncited section (intro/gaps) still contributes to the body but
is nothing to verify. Deterministic and Model-free.
"""
from agent_pipeline.contracts.composition import Draft
from agent_pipeline.contracts.validation import BriefInput, Claim


def translate_draft_to_validation(draft: Draft) -> BriefInput:
    claims = [
        Claim(text=section.body, sources=section.cited_sources)
        for section in draft.sections
        if section.cited_sources
    ]

    available: list[str] = []
    for section in draft.sections:
        for source_id in section.cited_sources:
            if source_id not in available:
                available.append(source_id)

    body = "\n\n".join(
        f"## {section.heading}\n{section.body}" for section in draft.sections
    )

    return BriefInput(
        request_id=draft.request_id,
        claims=claims,
        body=body,
        available_sources=available,
    )
