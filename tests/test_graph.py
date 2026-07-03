"""The A1 -> A2 pipeline wired as LangGraph nodes over shared state, checkpointed.

Real langgraph, real checkpointer -- proves the topology seam, the Context
Translation on the A1->A2 edge, and the per-stage checkpoint that stops cascades.
"""
from langgraph.checkpoint.memory import InMemorySaver

from agent_pipeline.graph.pipeline import build_graph
from agent_pipeline.agents.retriever import A1Retriever, RuleBasedPlanner
from agent_pipeline.agents.analyst import A2Analyst, RuleBasedAnalyst


def _app(knowledge, checkpointer=None):
    return build_graph(
        A1Retriever(knowledge, RuleBasedPlanner()),
        A2Analyst(RuleBasedAnalyst()),
        checkpointer=checkpointer,
    )


def _initial(request):
    return {"request": request, "retrieval": None, "analysis": None}


def test_graph_runs_both_stages_and_populates_state(knowledge, sample_request):
    app = _app(knowledge)
    config = {"configurable": {"thread_id": sample_request.request_id}}
    result = app.invoke(_initial(sample_request), config)

    assert result["retrieval"].request_id == sample_request.request_id
    report = result["analysis"]
    assert report is not None and report.request_id == sample_request.request_id
    assert report.findings
    # findings are grounded in the passages A1 actually retrieved
    retrieved_ids = {p.source_id for p in result["retrieval"].passages}
    for finding in report.findings:
        assert set(finding.evidence) <= retrieved_ids


def test_graph_checkpoints_stage_output(knowledge, sample_request):
    saver = InMemorySaver()
    app = _app(knowledge, checkpointer=saver)
    config = {"configurable": {"thread_id": "t-123"}}
    app.invoke(_initial(sample_request), config)
    # both stages' validated outputs are held for resume by later stages
    snapshot = app.get_state(config)
    assert snapshot.values["retrieval"].request_id == sample_request.request_id
    assert snapshot.values["analysis"].request_id == sample_request.request_id
