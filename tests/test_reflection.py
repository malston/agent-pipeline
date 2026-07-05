"""The A3 <-> A4 reflection loop: recompose on grounding failure, gate at the end.

Deterministic and keyless -- an injected ClaimVerifier drives the critic so the loop
mechanism is tested without an LLM.
"""
import pytest
from langchain_core.documents import Document

from agent_pipeline.tools.embeddings import LocalEmbeddings
from agent_pipeline.tools.knowledge import KnowledgeStore
from agent_pipeline.graph.pipeline import build_graph
from agent_pipeline.agents.retriever import A1Retriever, RuleBasedPlanner
from agent_pipeline.agents.analyst import A2Analyst, RuleBasedAnalyst
from agent_pipeline.agents.composer import A3Composer, RuleBasedComposer
from agent_pipeline.agents.validator import A4Validator
from agent_pipeline.agents.guardrails import GuardrailViolation
from agent_pipeline.contracts.retrieval import RetrievalRequest

def _one_doc_store():
    store = KnowledgeStore(LocalEmbeddings())
    store.index([Document(id="mito", page_content="Mitochondria produce ATP.", metadata={})])
    return store

def _initial(request):
    return {
        "request": request,
        "retrieval": None,
        "analysis": None,
        "draft": None,
        "brief": None,
        "feedback": None,
        "attempt": 0,
    }

class _CapturingComposer:
    """Wraps RuleBasedComposer and records the feedback passed on each compose."""

    def __init__(self):
        self._inner = RuleBasedComposer()
        self.feedbacks = []

    def compose(self, composer_input, feedback=None):
        self.feedbacks.append(feedback)
        return self._inner.compose(composer_input, feedback)

class _FailOnceVerifier:
    """Fails the claim on the first check round, passes afterward."""

    def __init__(self):
        self.calls = 0

    def verify(self, claim, available_sources):
        self.calls += 1
        return self.calls > 1  # False on the 1st call, True after

class _AlwaysFailVerifier:
    def verify(self, claim, available_sources):
        return False

def _app(store, composer, verifier):
    return build_graph(
        A1Retriever(store, RuleBasedPlanner()),
        A2Analyst(RuleBasedAnalyst()),
        composer,
        A4Validator(verifier),
    )

def test_loop_recomposes_with_feedback_then_grounds():
    composer = _CapturingComposer()
    app = _app(_one_doc_store(), A3Composer(composer), _FailOnceVerifier())
    request = RetrievalRequest(request_id="r1", raw_query="how do cells make energy?")
    result = app.invoke(_initial(request), {"configurable": {"thread_id": "r1"}})

    brief = result["brief"]
    assert brief is not None and brief.checks.grounding_ok is True
    # composed twice: first with no feedback, then with the unsupported claim
    assert len(composer.feedbacks) == 2
    assert composer.feedbacks[0] is None
    assert composer.feedbacks[1] == ["Mitochondria produce ATP."]

def test_loop_raises_after_max_attempts():
    app = _app(_one_doc_store(), A3Composer(RuleBasedComposer()), _AlwaysFailVerifier())
    request = RetrievalRequest(request_id="r1", raw_query="how do cells make energy?")
    with pytest.raises(GuardrailViolation) as exc:
        app.invoke(_initial(request), {"configurable": {"thread_id": "r1"}})
    assert exc.value.code == "GROUNDING_FAILED"
