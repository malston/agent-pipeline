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
from agent_pipeline.agents.composer import A3Composer, RuleBasedComposer, CompositionPlan
from agent_pipeline.agents.plan import PlanStep
from agent_pipeline.agents.validator import A4Validator, StructuralClaimVerifier
from agent_pipeline.agents.guardrails import GuardrailViolation
from agent_pipeline.config import MAX_COMPOSE_ATTEMPTS
from agent_pipeline.contracts.retrieval import RetrievalRequest
from agent_pipeline.contracts.composition import Section

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


class _SourceUnresolvedVerifier:
    """Raises the infra fault a missing-document verifier would raise."""

    def verify(self, claim, available_sources):
        raise GuardrailViolation("SOURCE_UNRESOLVED", "cited source missing from store")


def test_source_unresolved_aborts_the_run_without_looping():
    # SOURCE_UNRESOLVED is an infra fault, not a grounding failure: recomposing cannot
    # add a missing document, so check() must let it propagate out of the graph rather
    # than feed it back as a retry. Proven here through the graph, not just check().
    composer = _CapturingComposer()
    app = _app(_one_doc_store(), A3Composer(composer), _SourceUnresolvedVerifier())
    request = RetrievalRequest(request_id="r1", raw_query="how do cells make energy?")
    with pytest.raises(GuardrailViolation) as exc:
        app.invoke(_initial(request), {"configurable": {"thread_id": "r1"}})
    assert exc.value.code == "SOURCE_UNRESOLVED"
    assert len(composer.feedbacks) == 1  # composed once; the fault did not drive a retry


class _PassVerifier:
    def verify(self, claim, available_sources):
        return True


class _BannedPhraseComposer:
    """Emits one grounded section whose body carries a banned phrase."""

    def __init__(self):
        self.calls = 0

    def compose(self, composer_input, feedback=None):
        self.calls += 1
        return CompositionPlan(
            steps=[PlanStep(step_id=0, intent="emit", tool="emit_contract")],
            sections=[Section(heading="H", body="Cells make ATP; forbidden.", cited_sources=["mito"])],
            style_profile="plain",
        )


def test_policy_failure_routes_to_the_gate_without_looping():
    # route() keys only on grounding; a grounded-but-policy-failing brief must fall
    # straight through to the gate and raise, never spin the recompose loop.
    composer = _BannedPhraseComposer()
    app = build_graph(
        A1Retriever(_one_doc_store(), RuleBasedPlanner()),
        A2Analyst(RuleBasedAnalyst()),
        A3Composer(composer),
        A4Validator(_PassVerifier(), banned_phrases=frozenset({"forbidden"})),
    )
    request = RetrievalRequest(request_id="r1", raw_query="how do cells make energy?")
    with pytest.raises(GuardrailViolation) as exc:
        app.invoke(_initial(request), {"configurable": {"thread_id": "r1"}})
    assert exc.value.code == "POLICY_FAILED"
    assert composer.calls == 1  # policy failure did not trigger the recompose loop


class _UncitedSectionComposer:
    """Emits one content section that cites nothing, every time -- it never grounds."""

    def __init__(self):
        self.calls = 0

    def compose(self, composer_input, feedback=None):
        self.calls += 1
        return CompositionPlan(
            steps=[PlanStep(step_id=0, intent="emit", tool="emit_contract")],
            sections=[Section(heading="Floating", body="Unbacked assertion.", cited_sources=[])],
            style_profile="plain",
        )


def test_uncited_section_drives_the_loop_to_exhaustion():
    composer = _UncitedSectionComposer()
    app = _app(_one_doc_store(), A3Composer(composer), StructuralClaimVerifier())
    request = RetrievalRequest(request_id="r1", raw_query="how do cells make energy?")
    with pytest.raises(GuardrailViolation) as exc:
        app.invoke(_initial(request), {"configurable": {"thread_id": "r1"}})
    assert exc.value.code == "GROUNDING_FAILED"
    assert composer.calls == MAX_COMPOSE_ATTEMPTS  # an uncited section can never ground


def test_gaps_only_draft_ships_grounded_without_looping():
    # the no-evidence path: an empty store -> analyst finds nothing -> a gaps-only draft
    # (no sections, no claims, no uncited assertions) must ship grounded on the first pass,
    # the symmetric complement of the uncited-section-loops-to-exhaustion case above.
    composer = _CapturingComposer()
    empty_store = KnowledgeStore(LocalEmbeddings())
    app = _app(empty_store, A3Composer(composer), StructuralClaimVerifier())
    request = RetrievalRequest(request_id="r1", raw_query="anything about bacteria?")
    result = app.invoke(_initial(request), {"configurable": {"thread_id": "r1"}})

    brief = result["brief"]
    assert brief is not None and brief.checks.grounding_ok is True
    assert len(composer.feedbacks) == 1  # composed once; gaps do not drive a recompose


class _FabricatedGapComposer:
    """Emits a grounded section but also invents a gap A2 never reported, every time."""

    def __init__(self):
        self.calls = 0

    def compose(self, composer_input, feedback=None):
        self.calls += 1
        sections = [
            Section(heading="Point 1", body=p.statement, cited_sources=p.sources)
            for p in composer_input.points
        ]
        return CompositionPlan(
            steps=[PlanStep(step_id=0, intent="emit", tool="emit_contract")],
            sections=sections,
            gaps=[*composer_input.gaps, "Cells secretly feel joy."],
            style_profile="plain",
        )

def test_fabricated_gap_drives_the_loop_to_exhaustion():
    # the section is grounded, but the invented gap is unbacked body text -> it fails
    # grounding and drives the loop to exhaustion, then the gate raises.
    composer = _FabricatedGapComposer()
    app = _app(_one_doc_store(), A3Composer(composer), StructuralClaimVerifier())
    request = RetrievalRequest(request_id="r1", raw_query="how do cells make energy?")
    with pytest.raises(GuardrailViolation) as exc:
        app.invoke(_initial(request), {"configurable": {"thread_id": "r1"}})
    assert exc.value.code == "GROUNDING_FAILED"
    assert composer.calls == MAX_COMPOSE_ATTEMPTS  # the invented gap never grounds
