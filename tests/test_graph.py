"""A1 wired as a LangGraph node over shared pipeline state, with checkpointing.

Real langgraph, real checkpointer -- proves the topology seam and the per-stage
checkpoint that stops cascade failures.
"""
from langgraph.checkpoint.memory import InMemorySaver

from agent_pipeline.graph.pipeline import build_graph
from agent_pipeline.agents.retriever import A1Retriever, RuleBasedPlanner


def _app(knowledge, checkpointer=None):
    return build_graph(A1Retriever(knowledge, RuleBasedPlanner()), checkpointer=checkpointer)


def test_graph_runs_a1_and_populates_state(knowledge, sample_request):
    app = _app(knowledge)
    config = {"configurable": {"thread_id": sample_request.request_id}}
    result = app.invoke({"request": sample_request, "retrieval": None}, config)
    bundle = result["retrieval"]
    assert bundle is not None
    assert bundle.request_id == sample_request.request_id
    assert bundle.passages


def test_graph_checkpoints_stage_output(knowledge, sample_request):
    saver = InMemorySaver()
    app = _app(knowledge, checkpointer=saver)
    config = {"configurable": {"thread_id": "t-123"}}
    app.invoke({"request": sample_request, "retrieval": None}, config)
    # the checkpointer holds A1's validated output for resume by later stages
    snapshot = app.get_state(config)
    assert snapshot.values["retrieval"].request_id == sample_request.request_id
