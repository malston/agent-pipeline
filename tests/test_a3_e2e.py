"""End-to-end A3 with the REAL provider-agnostic Model (LLMComposer).

Real LLM, real guardrails. Skips (never mocks) without a provider key.
"""
import os

import pytest

from agent_pipeline.agents.composer import A3Composer, LLMComposer
from agent_pipeline.contracts.composition import ComposerInput, Point, Draft


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="no ANTHROPIC_API_KEY; real-LLM e2e skipped (not mocked)",
)
def test_a3_end_to_end_with_real_llm():
    composer_input = ComposerInput(
        request_id="r1",
        points=[
            Point(statement="Mitochondria produce ATP, the cell's energy currency.", sources=["mito"], confidence=0.9),
            Point(statement="Plants also capture energy via photosynthesis.", sources=["photo"], confidence=0.7),
        ],
        gaps=["No detail on bacterial energy metabolism."],
    )
    draft = A3Composer(LLMComposer()).run(composer_input)

    assert isinstance(draft, Draft)
    assert draft.sections
    # the output guardrail already blocks fabricated citations; assert it explicitly
    for section in draft.sections:
        assert set(section.cited_sources) <= {"mito", "photo"}
