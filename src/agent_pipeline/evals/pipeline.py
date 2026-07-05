"""System-scope eval: the full pipeline scored end to end.

Runs the keyless A1->A2->A3->A4 graph on a golden request and scores the final
brief's citations against the sources that should back the answer. Bound to traces
via the harness, like the agent-scope evals.
"""
from pydantic import BaseModel

from agent_pipeline.agents.retriever import A1Retriever, RuleBasedPlanner
from agent_pipeline.agents.analyst import A2Analyst, RuleBasedAnalyst
from agent_pipeline.agents.composer import A3Composer, RuleBasedComposer
from agent_pipeline.agents.validator import A4Validator, StructuralClaimVerifier
from agent_pipeline.graph.pipeline import build_graph
from agent_pipeline.contracts.retrieval import RetrievalRequest
from agent_pipeline.evals.harness import evaluate, EvalReport
from agent_pipeline.evals.metrics import set_recall, set_precision
from agent_pipeline.tools.knowledge import KnowledgeStore


class SystemExample(BaseModel):
    id: str
    request: str
    relevant_sources: list[str]


def evaluate_pipeline(
    knowledge: KnowledgeStore, dataset: list[SystemExample]
) -> EvalReport:
    app = build_graph(
        A1Retriever(knowledge, RuleBasedPlanner()),
        A2Analyst(RuleBasedAnalyst()),
        A3Composer(RuleBasedComposer()),
        A4Validator(StructuralClaimVerifier()),
    )

    def run(example: SystemExample) -> list[str]:
        result = app.invoke(
            {
                "request": RetrievalRequest(request_id=example.id, raw_query=example.request),
                "retrieval": None,
                "analysis": None,
                "draft": None,
                "brief": None,
            },
            {"configurable": {"thread_id": example.id}},
        )
        return result["brief"].citations

    return evaluate(
        dataset,
        run_fn=run,
        metrics={
            "citation_recall": lambda out, ex: set_recall(out, set(ex.relevant_sources)),
            "citation_precision": lambda out, ex: set_precision(out, set(ex.relevant_sources)),
        },
    )
