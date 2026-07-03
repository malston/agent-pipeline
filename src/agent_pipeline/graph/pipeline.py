"""The complete Pipeline topology as a LangGraph StateGraph.

The graph runs A1 -> A2 -> A3 -> A4, joined by translator edges, and produces a
ValidatedBrief. The checkpointer persists each stage's validated output so a
downstream failure resumes from the last good contract instead of re-running the
pipeline.
"""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from agent_pipeline.agents.retriever import A1Retriever
from agent_pipeline.agents.analyst import A2Analyst
from agent_pipeline.agents.composer import A3Composer
from agent_pipeline.agents.validator import A4Validator
from agent_pipeline.translators.retrieval_to_analysis import (
    translate_retrieval_to_analysis,
)
from agent_pipeline.translators.analysis_to_composition import (
    translate_analysis_to_composition,
)
from agent_pipeline.translators.draft_to_validation import translate_draft_to_validation
from agent_pipeline.contracts.retrieval import RetrievalRequest, RetrievalBundle
from agent_pipeline.contracts.analysis import AnalysisReport
from agent_pipeline.contracts.composition import Draft
from agent_pipeline.contracts.validation import ValidatedBrief


class PipelineState(TypedDict):
    request: RetrievalRequest
    retrieval: RetrievalBundle | None
    analysis: AnalysisReport | None
    draft: Draft | None
    brief: ValidatedBrief | None


def build_graph(
    retriever: A1Retriever,
    analyst: A2Analyst,
    composer: A3Composer,
    validator: A4Validator,
    checkpointer=None,
):
    def retriever_node(state: PipelineState) -> dict:
        return {"retrieval": retriever.run(state["request"])}

    def analyst_node(state: PipelineState) -> dict:
        bundle = state["retrieval"]
        if bundle is None:
            raise ValueError(
                "analyst_node reached with no retrieval bundle; "
                "A1 did not populate state['retrieval']"
            )
        return {"analysis": analyst.run(translate_retrieval_to_analysis(bundle))}

    def composer_node(state: PipelineState) -> dict:
        report = state["analysis"]
        if report is None:
            raise ValueError(
                "composer_node reached with no analysis report; "
                "A2 did not populate state['analysis']"
            )
        return {"draft": composer.run(translate_analysis_to_composition(report))}

    def validator_node(state: PipelineState) -> dict:
        draft = state["draft"]
        if draft is None:
            raise ValueError(
                "validator_node reached with no draft; "
                "A3 did not populate state['draft']"
            )
        return {"brief": validator.run(translate_draft_to_validation(draft))}

    graph = StateGraph(PipelineState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("composer", composer_node)
    graph.add_node("validator", validator_node)
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "composer")
    graph.add_edge("composer", "validator")
    graph.add_edge("validator", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
