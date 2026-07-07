"""A1 Retriever: Model (planner) + Harness (Plan-Execute loop, guardrails).

The keyless RuleBasedPlanner lets the whole agent run end-to-end with real RAG
and real guardrails -- no LLM key, no mocks. The LLM-backed planner (real Model)
is exercised in test_a1_e2e.py, gated on a provider key.
"""
import pytest

from agent_pipeline.agents.retriever import (
    A1Retriever,
    RuleBasedPlanner,
    RetrievalPlan,
)
from agent_pipeline.agents.plan import PlanStep
from agent_pipeline.agents.guardrails import GuardrailViolation
from agent_pipeline.contracts.retrieval import RetrievalRequest, RetrievalBundle
from agent_pipeline.tools.memory import WorkingMemory


def test_rule_based_planner_emits_plan_ending_in_emit(sample_request):
    plan = RuleBasedPlanner().plan(sample_request)
    assert plan.normalized_query
    assert plan.search_queries
    assert plan.steps[-1].tool == "emit_contract"


def test_a1_produces_grounded_bundle_end_to_end(knowledge, sample_request):
    agent = A1Retriever(knowledge, RuleBasedPlanner())
    bundle = agent.run(sample_request)

    assert isinstance(bundle, RetrievalBundle)
    assert bundle.request_id == sample_request.request_id
    assert bundle.passages, "expected non-empty evidence"
    # every citation is grounded in a really-retrieved source
    assert all(p.source_id in knowledge.known_ids() for p in bundle.passages)
    assert 0.0 <= bundle.coverage <= 1.0
    # real semantic retrieval put a biology doc on top for an energy question
    assert bundle.passages[0].source_id in {"mito", "photo"}


def test_a1_writes_candidates_to_scoped_working_memory(knowledge, sample_request):
    mem = WorkingMemory()
    A1Retriever(knowledge, RuleBasedPlanner(), memory=mem).run(sample_request)
    assert mem.load(sample_request.request_id, "candidates")


def test_a1_emits_empty_bundle_for_no_hit_retrieval(local_embeddings, sample_request):
    # An empty corpus yields a zero-hit retrieval. A1 emits a valid, empty bundle
    # (coverage 0.0) that clears the grounding gate inside run() -- a legitimate
    # "no evidence" result, not a failure. A2 later surfaces it as a "no evidence" gap.
    from agent_pipeline.tools.knowledge import KnowledgeStore

    empty_store = KnowledgeStore(local_embeddings)
    bundle = A1Retriever(empty_store, RuleBasedPlanner()).run(sample_request)

    assert isinstance(bundle, RetrievalBundle)
    assert bundle.passages == []
    assert bundle.coverage == 0.0


def test_a1_rejects_plan_that_uses_ungranted_tool(knowledge, sample_request):
    class _UngrantedToolPlanner:
        def plan(self, request: RetrievalRequest) -> RetrievalPlan:
            return RetrievalPlan(
                normalized_query=request.raw_query,
                search_queries=[request.raw_query],
                k=4,
                steps=[
                    PlanStep(step_id=0, intent="judge", tool="check_claim"),
                    PlanStep(step_id=1, intent="emit", tool="emit_contract"),
                ],
            )

    agent = A1Retriever(knowledge, _UngrantedToolPlanner())
    with pytest.raises(GuardrailViolation) as exc:
        agent.run(sample_request)
    assert exc.value.code == "TOOL_NOT_GRANTED"


@pytest.mark.parametrize("unhandled_tool", ["get_source", "load_scratch"])
def test_a1_rejects_plan_using_tool_the_executor_does_not_honor(
    knowledge, sample_request, unhandled_tool
):
    """A tool the executor cannot run must be rejected loudly by the plan
    guardrail, never granted and then silently skipped."""

    class _UnhandledToolPlanner:
        def plan(self, request: RetrievalRequest) -> RetrievalPlan:
            return RetrievalPlan(
                normalized_query=request.raw_query,
                search_queries=[request.raw_query],
                k=4,
                steps=[
                    PlanStep(step_id=0, intent="unhandled", tool=unhandled_tool),
                    PlanStep(step_id=1, intent="emit", tool="emit_contract"),
                ],
            )

    agent = A1Retriever(knowledge, _UnhandledToolPlanner())
    with pytest.raises(GuardrailViolation) as exc:
        agent.run(sample_request)
    assert exc.value.code == "TOOL_NOT_GRANTED"


def test_a1_raises_on_granted_but_unhandled_step(knowledge, sample_request, monkeypatch):
    """Defense against grant/executor drift: a tool that is granted but has no
    executor branch must fail loudly (UNHANDLED_STEP), never be silently skipped."""
    monkeypatch.setattr(
        "agent_pipeline.agents.retriever.A1_TOOL_GRANT",
        {"search_knowledge", "save_scratch", "emit_contract", "mystery_tool"},
    )

    class _UnhandledStepPlanner:
        def plan(self, request: RetrievalRequest) -> RetrievalPlan:
            return RetrievalPlan(
                normalized_query=request.raw_query,
                search_queries=[request.raw_query],
                k=4,
                steps=[
                    PlanStep(step_id=0, intent="mystery", tool="mystery_tool"),
                    PlanStep(step_id=1, intent="emit", tool="emit_contract"),
                ],
            )

    agent = A1Retriever(knowledge, _UnhandledStepPlanner())
    with pytest.raises(GuardrailViolation) as exc:
        agent.run(sample_request)
    assert exc.value.code == "UNHANDLED_STEP"
