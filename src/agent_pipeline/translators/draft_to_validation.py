"""Context Translation for the A3 -> A4 boundary.

Maps the composition vocabulary (draft of sections + gaps) into the validation
vocabulary. A cited section becomes a claim; a section that cites nothing becomes an
uncited assertion (ungrounded -- no grounding attempt). The section text and the gaps
assemble into the body; cited source ids become the available-sources set.
Deterministic and Model-free.
"""
from agent_pipeline.contracts.composition import Draft
from agent_pipeline.contracts.validation import BriefInput, Claim
from agent_pipeline.agents.guardrails import GuardrailViolation


def translate_draft_to_validation(draft: Draft) -> BriefInput:
    if not draft.sections and not draft.gaps:
        # nothing to ground and nothing to acknowledge -- the body would be empty; fail
        # with a domain-typed guardrail, not a generic pydantic error on BriefInput.body.
        raise GuardrailViolation(
            "EMPTY_BRIEF", "draft has no sections and no gaps -- nothing to validate"
        )

    claims = [
        Claim(text=section.body, sources=section.cited_sources)
        for section in draft.sections
        if section.cited_sources
    ]
    uncited_assertions = [
        section.body for section in draft.sections if not section.cited_sources
    ]

    available: list[str] = []
    seen: set[str] = set()
    for section in draft.sections:
        for source_id in section.cited_sources:
            if source_id not in seen:
                seen.add(source_id)
                available.append(source_id)

    parts = [f"## {section.heading}\n{section.body}" for section in draft.sections]
    if draft.gaps:
        parts.append("## Open questions\n" + "\n".join(f"- {gap}" for gap in draft.gaps))
    body = "\n\n".join(parts)

    return BriefInput(
        request_id=draft.request_id,
        claims=claims,
        body=body,
        available_sources=available,
        uncited_assertions=uncited_assertions,
    )
