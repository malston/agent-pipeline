"""Agent-scope eval for A1 retrieval (the RAG gate).

Scores a golden dataset (query -> the source ids that should be retrieved) with
recall@k and MRR, binding each score to its run's trace via the harness. This is the
evidence base for HDD tuning of retrieval (starting with k).
"""
from pydantic import BaseModel, Field

from agent_pipeline.evals.harness import evaluate, EvalReport
from agent_pipeline.evals.metrics import recall_at_k, reciprocal_rank
from agent_pipeline.tools.knowledge import KnowledgeStore


class RetrievalExample(BaseModel):
    id: str
    query: str
    relevant_sources: list[str] = Field(min_length=1)  # a golden example must label >=1


def evaluate_retrieval(
    knowledge: KnowledgeStore,
    dataset: list[RetrievalExample],
    k: int,
) -> EvalReport:
    return evaluate(
        dataset,
        run_fn=lambda ex: [p.source_id for p in knowledge.search(ex.query, k=k)],
        metrics={
            "recall_at_k": lambda out, ex: recall_at_k(out, set(ex.relevant_sources), k),
            "reciprocal_rank": lambda out, ex: reciprocal_rank(out, set(ex.relevant_sources)),
        },
    )
