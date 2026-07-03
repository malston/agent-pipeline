"""End-to-end A1 with the REAL provider-agnostic Model (LLMPlanner).

Per the no-mocks-in-e2e rule this uses a real LLM and real retrieval. With no
provider key configured it SKIPS -- it is never faked. Set ANTHROPIC_API_KEY
(and optionally MODEL_ID) to run it.
"""
import os

import pytest

from agent_pipeline.agents.retriever import A1Retriever, LLMPlanner
from agent_pipeline.contracts.retrieval import RetrievalBundle


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="no ANTHROPIC_API_KEY; real-LLM e2e skipped (not mocked)",
)
def test_a1_end_to_end_with_real_llm(knowledge, sample_request):
    agent = A1Retriever(knowledge, LLMPlanner())
    bundle = agent.run(sample_request)
    assert isinstance(bundle, RetrievalBundle)
    assert bundle.passages
    assert all(p.source_id in knowledge.known_ids() for p in bundle.passages)
    assert bundle.passages[0].source_id in {"mito", "photo"}
