"""A4 Validator = Model-backed check_claim + Harness gate.

A4 is the terminal gate: it verifies each claim against its cited sources, applies
policy and format checks, and refuses to emit a brief that fails any of them.

Unlike A1-A3, A4's plan is Harness-built ([check_claim, emit_contract]) rather than
Model-generated: A4's Model contribution is the per-claim *verification*, not
planning. Two verifiers share one seam:

* ``StructuralClaimVerifier`` -- keyless; a claim is grounded iff its cited sources
  are all among the available sources (defense-in-depth over upstream grounding).
* ``LLMClaimVerifier`` -- provider-agnostic; fetches each source's text and has the
  Model judge whether it actually supports the claim. Needs a key, so it is
  exercised only by the gated end-to-end test.
"""
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from agent_pipeline.agents.plan import Plan, PlanStep
from agent_pipeline.agents.guardrails import (
    GuardrailViolation,
    validate_plan,
    validate_brief_output,
)
from agent_pipeline.config import A4_TOOL_GRANT, A4_MAX_PLAN_STEPS, A4_BANNED_PHRASES
from agent_pipeline.model import build_model
from agent_pipeline.tools.knowledge import KnowledgeStore
from agent_pipeline.contracts.validation import (
    Claim,
    BriefInput,
    ValidationChecks,
    ValidatedBrief,
)


class ClaimVerifier(Protocol):
    def verify(self, claim: Claim, available_sources: set[str]) -> bool: ...


class StructuralClaimVerifier:
    """Keyless: a claim is grounded iff every cited source is available."""

    def verify(self, claim: Claim, available_sources: set[str]) -> bool:
        return set(claim.sources) <= available_sources


class _ClaimVerdict(BaseModel):
    supported: bool


class LLMClaimVerifier:
    """Provider-agnostic: the Model judges whether the cited source text supports
    the claim. Fetches source text via the knowledge store (check_claim tool)."""

    _SYSTEM = (
        "You verify grounding. Given a claim and the text of its cited sources, "
        "answer supported=true only if the sources substantiate the claim; "
        "otherwise supported=false."
    )

    def __init__(self, knowledge: KnowledgeStore, model: BaseChatModel | None = None) -> None:
        self._knowledge = knowledge
        base = model if model is not None else build_model()
        self._model = base.with_structured_output(_ClaimVerdict)

    def verify(self, claim: Claim, available_sources: set[str]) -> bool:
        if not set(claim.sources) <= available_sources:
            return False  # cites something outside the pool -- not grounded
        texts = []
        for source_id in claim.sources:
            document = self._knowledge.get(source_id)
            if document is None:
                return False  # cited source does not resolve
            texts.append(document.page_content)
        prompt = f"Claim: {claim.text}\n\nSources:\n" + "\n".join(texts)
        verdict = self._model.invoke([("system", self._SYSTEM), ("human", prompt)])
        return verdict.supported


class A4Validator:
    def __init__(
        self,
        verifier: ClaimVerifier,
        banned_phrases: frozenset[str] = A4_BANNED_PHRASES,
        max_plan_steps: int = A4_MAX_PLAN_STEPS,
    ) -> None:
        self._verifier = verifier
        self._banned = banned_phrases
        self._max_steps = max_plan_steps

    def run(self, brief_input: BriefInput) -> ValidatedBrief:
        # A4's plan is Harness-built: verify every claim, then emit.
        plan = Plan(
            steps=[
                PlanStep(step_id=0, intent="verify claims", tool="check_claim"),
                PlanStep(step_id=1, intent="emit the brief", tool="emit_contract"),
            ]
        )
        validate_plan(plan, A4_TOOL_GRANT, self._max_steps)  # guardrail

        available = set(brief_input.available_sources)
        grounding_ok = True
        for step in plan.steps:  # EXECUTE
            if step.tool == "check_claim":
                grounding_ok = all(
                    self._verifier.verify(claim, available)
                    for claim in brief_input.claims
                )
            elif step.tool == "emit_contract":
                brief = ValidatedBrief(
                    request_id=brief_input.request_id,
                    body=brief_input.body,
                    citations=brief_input.available_sources,
                    checks=ValidationChecks(
                        grounding_ok=grounding_ok,
                        policy_ok=not any(p in brief_input.body for p in self._banned),
                        format_ok=bool(brief_input.body.strip()),
                    ),
                )
                validate_brief_output(brief)  # the gate
                return brief
        # Unreachable: validate_plan guarantees a terminal emit_contract.
        raise GuardrailViolation("NO_EMIT", "plan executed without emitting a contract")
