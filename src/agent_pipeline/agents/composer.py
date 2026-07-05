"""A3 Composer = Model (a composer) + Harness (Plan-Execute loop + guardrails).

The composer turns the points A2 supplied into a drafted deliverable. Two
composers share one seam:

* ``RuleBasedComposer`` -- keyless, deterministic stand-in; one section per point
  citing that point's sources, plus an "Open questions" section when gaps exist.
  Lets the agent run with no provider.
* ``LLMComposer`` -- provider-agnostic (build_model seam); the real Model. Needs a
  provider key, so it is exercised only by the gated end-to-end test.
"""
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from agent_pipeline.agents.plan import Plan, PlanStep
from agent_pipeline.agents.guardrails import (
    GuardrailViolation,
    validate_plan,
    validate_composition_output,
)
from agent_pipeline.config import A3_TOOL_GRANT, A3_MAX_PLAN_STEPS
from agent_pipeline.model import build_model
from agent_pipeline.contracts.composition import (
    ComposerInput,
    Section,
    Draft,
)


class CompositionPlan(BaseModel):
    """The A3 Model's output: the auditable step list plus the draft payload."""

    steps: list[PlanStep]
    sections: list[Section]
    style_profile: str


class Composer(Protocol):
    def compose(self, composer_input: ComposerInput) -> CompositionPlan: ...


class RuleBasedComposer:
    """Keyless stand-in: one section per point citing that point's sources, plus an
    "Open questions" section when the input has gaps."""

    def compose(self, composer_input: ComposerInput) -> CompositionPlan:
        sections = [
            Section(
                heading=f"Point {i + 1}",
                body=point.statement,
                cited_sources=point.sources,
            )
            for i, point in enumerate(composer_input.points)
        ]
        if composer_input.gaps:
            sections.append(
                Section(heading="Open questions", body="\n".join(composer_input.gaps))
            )
        return CompositionPlan(
            steps=[PlanStep(step_id=0, intent="emit the draft", tool="emit_contract")],
            sections=sections,
            style_profile="outline",
        )


class LLMComposer:
    """Provider-agnostic Model composer. Receives the model by injection."""

    _SYSTEM = (
        "You are the composer in a RAG pipeline. Given a set of points (each a "
        "statement with the source ids that support it) and known gaps, write a "
        "draft as titled sections. Compose STRICTLY from the given points: restate "
        "and organize only what the points assert. Do NOT add facts, mechanisms, "
        "inferences, or links between points that are not explicitly stated in them, "
        "even if you believe them true -- any added detail will be rejected by the "
        "downstream grounding check. Each section cites ONLY the source ids of the "
        "points it draws on, and every sentence must be supported by those cited "
        "points. Put unanswered gaps in their own section. Pick a concise "
        "style_profile. You have no retrieval tools: produce a plan whose single "
        "step uses the tool emit_contract."
    )

    def __init__(self, model: BaseChatModel | None = None) -> None:
        base = model if model is not None else build_model()
        self._model = base.with_structured_output(CompositionPlan)

    def compose(self, composer_input: ComposerInput) -> CompositionPlan:
        return self._model.invoke(
            [("system", self._SYSTEM), ("human", composer_input.model_dump_json())]
        )


class A3Composer:
    def __init__(
        self,
        composer: Composer,
        max_plan_steps: int = A3_MAX_PLAN_STEPS,
    ) -> None:
        self._composer = composer
        self._max_steps = max_plan_steps

    def run(self, composer_input: ComposerInput) -> Draft:
        plan = self._composer.compose(composer_input)  # PLAN (Model)
        validate_plan(Plan(steps=plan.steps), A3_TOOL_GRANT, self._max_steps)  # guardrail

        available = {
            source for point in composer_input.points for source in point.sources
        }
        for step in plan.steps:  # EXECUTE
            if step.tool == "emit_contract":
                draft = Draft(
                    request_id=composer_input.request_id,
                    sections=plan.sections,
                    style_profile=plan.style_profile,
                )
                validate_composition_output(draft, available)  # guardrail
                return draft
        # Unreachable: validate_plan guarantees a terminal emit_contract. Defensive
        # backstop so a future weakening of that guard fails loudly, not silently.
        raise GuardrailViolation("NO_EMIT", "plan executed without emitting a contract")
