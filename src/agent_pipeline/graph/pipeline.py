"""The Pipeline topology as a LangGraph StateGraph, with an A3 <-> A4 reflection loop.

The graph runs A1 -> A2 -> A3 -> A4. On a grounding failure with attempts remaining,
A4's report loops back to A3 to recompose with per-claim feedback; otherwise a terminal
gate validates and raises if any check still fails. A1 and A2 run once. The checkpointer
persists each stage's output. See docs/architecture/pipeline-graph.md.
"""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from agent_pipeline.agents.retriever import A1Retriever
from agent_pipeline.agents.analyst import A2Analyst
from agent_pipeline.agents.composer import A3Composer
from agent_pipeline.agents.validator import A4Validator
from agent_pipeline.agents.guardrails import validate_brief_output
from agent_pipeline.config import MAX_COMPOSE_ATTEMPTS
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
    feedback: list[str] | None  # unsupported claim texts fed back to A3 on retry
    attempt: int  # A3 compose count

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
        draft = composer.run(
            translate_analysis_to_composition(report), feedback=state.get("feedback")
        )
        return {"draft": draft, "attempt": state.get("attempt", 0) + 1}

    def validator_node(state: PipelineState) -> dict:
        draft = state["draft"]
        if draft is None:
            raise ValueError(
                "validator_node reached with no draft; "
                "A3 did not populate state['draft']"
            )
        outcome = validator.check(translate_draft_to_validation(draft))
        return {"brief": outcome.brief, "feedback": outcome.unsupported}

    def gate_node(state: PipelineState) -> dict:
        brief = state["brief"]
        if brief is None:
            raise ValueError(
                "gate_node reached with no brief; "
                "A4 did not populate state['brief']"
            )
        validate_brief_output(brief)  # raises if any check still fails
        return {}

    def route(state: PipelineState) -> str:
        brief = state["brief"]
        if brief is None:
            raise ValueError(
                "route reached with no brief; "
                "A4 did not populate state['brief']"
            )
        if brief.checks.grounding_ok or state["attempt"] >= MAX_COMPOSE_ATTEMPTS:
            return "gate"
        return "composer"

    graph = StateGraph(PipelineState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("composer", composer_node)
    graph.add_node("validator", validator_node)
    graph.add_node("gate", gate_node)
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "composer")
    graph.add_edge("composer", "validator")
    graph.add_conditional_edges("validator", route, {"gate": "gate", "composer": "composer"})
    graph.add_edge("gate", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
