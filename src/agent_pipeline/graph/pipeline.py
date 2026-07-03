"""The Pipeline topology as a LangGraph StateGraph.

Currently one node (A1); A2/A3/A4 attach as further nodes with translator edges.
The checkpointer persists each stage's validated output so a downstream failure
resumes from the last good contract instead of re-running the pipeline.
"""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from agent_pipeline.agents.retriever import A1Retriever
from agent_pipeline.contracts.retrieval import RetrievalRequest, RetrievalBundle


class PipelineState(TypedDict):
    request: RetrievalRequest
    retrieval: RetrievalBundle | None


def build_graph(retriever: A1Retriever, checkpointer=None):
    def retriever_node(state: PipelineState) -> dict:
        return {"retrieval": retriever.run(state["request"])}

    graph = StateGraph(PipelineState)
    graph.add_node("retriever", retriever_node)
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
