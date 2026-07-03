"""The Pipeline topology as a LangGraph StateGraph.

Currently A1 -> A2; A3/A4 attach as further nodes with translator edges. The
checkpointer persists each stage's validated output so a downstream failure
resumes from the last good contract instead of re-running the pipeline.
"""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from agent_pipeline.agents.retriever import A1Retriever
from agent_pipeline.agents.analyst import A2Analyst
from agent_pipeline.translators.retrieval_to_analysis import (
    translate_retrieval_to_analysis,
)
from agent_pipeline.contracts.retrieval import RetrievalRequest, RetrievalBundle
from agent_pipeline.contracts.analysis import AnalysisReport


class PipelineState(TypedDict):
    request: RetrievalRequest
    retrieval: RetrievalBundle | None
    analysis: AnalysisReport | None


def build_graph(retriever: A1Retriever, analyst: A2Analyst, checkpointer=None):
    def retriever_node(state: PipelineState) -> dict:
        return {"retrieval": retriever.run(state["request"])}

    def analyst_node(state: PipelineState) -> dict:
        # Context Translation on the A1 -> A2 boundary.
        analyst_input = translate_retrieval_to_analysis(state["retrieval"])
        return {"analysis": analyst.run(analyst_input)}

    graph = StateGraph(PipelineState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
