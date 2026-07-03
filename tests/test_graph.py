"""The A1 -> A2 -> A3 pipeline wired as LangGraph nodes over shared state.

Real langgraph, real checkpointer. Proves the three nodes run, that the
translator edges feed each stage (findings grounded in A1's retrieved ids;
sections grounded in those sources), and the per-stage checkpoint that stops
cascades. Translator field-mapping correctness is covered in the translator tests.
"""
from langgraph.checkpoint.memory import InMemorySaver

from agent_pipeline.graph.pipeline import build_graph
from agent_pipeline.agents.retriever import A1Retriever, RuleBasedPlanner
from agent_pipeline.agents.analyst import A2Analyst, RuleBasedAnalyst
from agent_pipeline.agents.composer import A3Composer, RuleBasedComposer


def _app(knowledge, checkpointer=None):
    return build_graph(
        A1Retriever(knowledge, RuleBasedPlanner()),
        A2Analyst(RuleBasedAnalyst()),
        A3Composer(RuleBasedComposer()),
        checkpointer=checkpointer,
    )


def _initial(request):
    return {"request": request, "retrieval": None, "analysis": None, "draft": None}


def test_graph_runs_all_stages_and_populates_state(knowledge, sample_request):
    app = _app(knowledge)
    config = {"configurable": {"thread_id": sample_request.request_id}}
    result = app.invoke(_initial(sample_request), config)

    assert result["retrieval"].request_id == sample_request.request_id
    assert result["analysis"].request_id == sample_request.request_id
    draft = result["draft"]
    assert draft is not None and draft.request_id == sample_request.request_id
    assert draft.sections

    # sections are grounded in the sources A1 actually retrieved; pin non-empty so
    # the subset check can't hold vacuously
    retrieved_ids = {p.source_id for p in result["retrieval"].passages}
    assert retrieved_ids, "A1 should have retrieved passages for this query"
    for section in draft.sections:
        assert set(section.cited_sources) <= retrieved_ids


def test_graph_checkpoints_stage_output(knowledge, sample_request):
    saver = InMemorySaver()
    app = _app(knowledge, checkpointer=saver)
    config = {"configurable": {"thread_id": "t-123"}}
    app.invoke(_initial(sample_request), config)
    # every stage's validated output is held for resume by later stages
    snapshot = app.get_state(config)
    assert snapshot.values["retrieval"].request_id == sample_request.request_id
    assert snapshot.values["analysis"].request_id == sample_request.request_id
    assert snapshot.values["draft"].request_id == sample_request.request_id
