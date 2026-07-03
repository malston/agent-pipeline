"""The Plan is the Harness-internal contract a Plan-Execute agent emits up front.

It is not a cross-agent contract -- it is the auditable list of steps the
Harness validates (guardrail) before executing any of them.
"""
from pydantic import BaseModel


class PlanStep(BaseModel):
    step_id: int
    intent: str
    tool: str


class Plan(BaseModel):
    steps: list[PlanStep]
