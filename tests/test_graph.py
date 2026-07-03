"""The full A1 -> A2 -> A3 -> A4 pipeline wired as LangGraph nodes over shared state.

Real langgraph, real checkpointer. Proves the four nodes run, the translator edges
feed each stage, the terminal gate produces a ValidatedBrief with passing checks,
and the per-stage checkpoint that stops cascades. Translator field-mapping
correctness is covered in the translator tests.
"""
from langgraph.checkpoint.memory import InMemorySaver

from agent_pipeline.graph.pipeline import build_graph
from agent_pipeline.agents.retriever import A1Retriever, RuleBasedPlanner
from agent_pipeline.agents.analyst import A2Analyst, RuleBasedAnalyst
from agent_pipeline.agents.composer import A3Composer, RuleBasedComposer
from agent_pipeline.agents.validator import A4Validator, StructuralClaimVerifier


def _app(knowledge, checkpointer=None):
    return build_graph(
        A1Retriever(knowledge, RuleBasedPlanner()),
        A2Analyst(RuleBasedAnalyst()),
        A3Composer(RuleBasedComposer()),
        A4Validator(StructuralClaimVerifier()),
        checkpointer=checkpointer,
    )


def _initial(request):
    return {
        "request": request,
        "retrieval": None,
        "analysis": None,
        "draft": None,
        "brief": None,
    }


def test_graph_runs_all_stages_and_produces_a_validated_brief(knowledge, sample_request):
    app = _app(knowledge)
    config = {"configurable": {"thread_id": sample_request.request_id}}
    result = app.invoke(_initial(sample_request), config)

    for stage in ("retrieval", "analysis", "draft", "brief"):
        assert result[stage] is not None and result[stage].request_id == sample_request.request_id

    brief = result["brief"]
    assert brief.checks.grounding_ok and brief.checks.policy_ok and brief.checks.format_ok
    # the brief's citations trace back to what A1 actually retrieved
    retrieved_ids = {p.source_id for p in result["retrieval"].passages}
    assert retrieved_ids and set(brief.citations) <= retrieved_ids


def test_graph_checkpoints_stage_output(knowledge, sample_request):
    saver = InMemorySaver()
    app = _app(knowledge, checkpointer=saver)
    config = {"configurable": {"thread_id": "t-123"}}
    app.invoke(_initial(sample_request), config)
    snapshot = app.get_state(config)
    for stage in ("retrieval", "analysis", "draft", "brief"):
        assert snapshot.values[stage].request_id == sample_request.request_id
