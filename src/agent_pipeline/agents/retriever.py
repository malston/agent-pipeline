"""A1 Retriever = Model (a planner) + Harness (Plan-Execute loop + guardrails).

Plan-Execute: the planner (Model) emits an explicit, auditable plan up front; the
Harness validates it (plan guardrail), executes the granted tools it names, then
guardrails the output before handing off. Two planners share one seam:

* ``RuleBasedPlanner`` -- keyless, deterministic; lets the agent run with no
  provider. The HDD-friendly baseline.
* ``LLMPlanner`` -- provider-agnostic (init_chat_model); the real Model. Needs a
  provider key, so it is exercised only by the gated end-to-end test.
"""
from typing import Protocol

from pydantic import BaseModel, Field

from agent_pipeline.agents.plan import Plan, PlanStep
from agent_pipeline.agents.guardrails import (
    GuardrailViolation,
    validate_plan,
    validate_retrieval_output,
)
from agent_pipeline.config import A1_TOOL_GRANT, A1_MAX_PLAN_STEPS, DEFAULT_MODEL_ID
from agent_pipeline.contracts.retrieval import (
    Passage,
    RetrievalRequest,
    RetrievalBundle,
)
from agent_pipeline.tools.knowledge import KnowledgeStore
from agent_pipeline.tools.memory import WorkingMemory


class RetrievalPlan(BaseModel):
    """The A1 Model's output: how to retrieve, plus the auditable step list."""

    normalized_query: str
    search_queries: list[str]
    k: int = Field(default=4, ge=1, le=20)
    steps: list[PlanStep]


class Planner(Protocol):
    def plan(self, request: RetrievalRequest) -> RetrievalPlan: ...


class RuleBasedPlanner:
    """Keyless baseline planner. Real, deterministic, no LLM."""

    def plan(self, request: RetrievalRequest) -> RetrievalPlan:
        query = " ".join(request.raw_query.split())
        return RetrievalPlan(
            normalized_query=query,
            search_queries=[query],
            k=4,
            steps=[
                PlanStep(step_id=0, intent="retrieve evidence", tool="search_knowledge"),
                PlanStep(step_id=1, intent="cache candidate ids", tool="save_scratch"),
                PlanStep(step_id=2, intent="emit the bundle", tool="emit_contract"),
            ],
        )


class LLMPlanner:
    """Provider-agnostic Model planner. Swappable by MODEL_ID; needs a key."""

    _SYSTEM = (
        "You are the retrieval planner for a RAG pipeline. Given a raw request, "
        "produce a normalized query, one or more semantic search queries, an "
        "integer k, and a plan. Plan steps may use ONLY these tools: "
        "search_knowledge, get_source, save_scratch, load_scratch, emit_contract. "
        "The final step MUST be emit_contract."
    )

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        from langchain.chat_models import init_chat_model

        self._model = init_chat_model(model_id).with_structured_output(RetrievalPlan)

    def plan(self, request: RetrievalRequest) -> RetrievalPlan:
        return self._model.invoke(
            [("system", self._SYSTEM), ("human", request.raw_query)]
        )


class A1Retriever:
    def __init__(
        self,
        knowledge: KnowledgeStore,
        planner: Planner,
        memory: WorkingMemory | None = None,
        max_plan_steps: int = A1_MAX_PLAN_STEPS,
    ) -> None:
        self._knowledge = knowledge
        self._planner = planner
        self._memory = memory or WorkingMemory()
        self._max_steps = max_plan_steps

    def run(self, request: RetrievalRequest) -> RetrievalBundle:
        plan = self._planner.plan(request)  # PLAN (Model)
        validate_plan(Plan(steps=plan.steps), A1_TOOL_GRANT, self._max_steps)  # guardrail

        scope = request.request_id
        passages: list[Passage] = []
        for step in plan.steps:  # EXECUTE (Harness runs the tools the Model planned)
            if step.tool == "search_knowledge":
                passages = self._retrieve(plan)
            elif step.tool == "save_scratch":
                self._memory.save(scope, "candidates", [p.source_id for p in passages])
            elif step.tool == "emit_contract":
                bundle = self._emit(request, plan, passages)
                validate_retrieval_output(bundle, self._knowledge.known_ids())  # guardrail
                return bundle
            # get_source / load_scratch are granted but unused by this executor
        raise GuardrailViolation("NO_EMIT", "plan executed without emitting a contract")

    def _retrieve(self, plan: RetrievalPlan) -> list[Passage]:
        seen: set[str] = set()
        passages: list[Passage] = []
        for query in plan.search_queries:
            for passage in self._knowledge.search(query, k=plan.k):
                if passage.source_id not in seen:
                    seen.add(passage.source_id)
                    passages.append(passage)
        passages.sort(key=lambda p: p.score, reverse=True)
        return passages

    def _emit(
        self, request: RetrievalRequest, plan: RetrievalPlan, passages: list[Passage]
    ) -> RetrievalBundle:
        coverage = passages[0].score if passages else 0.0
        return RetrievalBundle(
            request_id=request.request_id,
            normalized_query=plan.normalized_query,
            passages=passages,
            coverage=coverage,
        )
