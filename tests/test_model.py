"""Provider-agnostic Model layer (DESIGN §9): the single build_model() factory.

Config tests run offline -- construction needs no key on the installed
langchain-anthropic (only .invoke() does). The live smoke test is gated on a real
key and never mocked.
"""
import os

import pytest
from langchain_core.language_models import BaseChatModel


def test_build_model_uses_config_default(monkeypatch):
    monkeypatch.delenv("MODEL_ID", raising=False)
    monkeypatch.delenv("MODEL_TEMPERATURE", raising=False)
    from agent_pipeline.model import build_model
    from agent_pipeline.config import DEFAULT_MODEL_ID

    model = build_model()
    assert isinstance(model, BaseChatModel)
    # DEFAULT_MODEL_ID carries the provider prefix; the client stores the bare id
    assert DEFAULT_MODEL_ID.endswith(model.model)
    assert model.temperature is None  # nothing sent -> no 400 on current models


def test_build_model_honors_model_id_override(monkeypatch):
    monkeypatch.setenv("MODEL_ID", "anthropic:claude-opus-4-8")
    monkeypatch.delenv("MODEL_TEMPERATURE", raising=False)
    from agent_pipeline.model import build_model

    assert build_model().model == "claude-opus-4-8"


def test_build_model_forwards_temperature_only_when_set(monkeypatch):
    monkeypatch.delenv("MODEL_ID", raising=False)
    monkeypatch.setenv("MODEL_TEMPERATURE", "0")
    from agent_pipeline.model import build_model

    assert build_model().temperature == 0.0


def test_llm_planner_accepts_injected_model():
    # Consolidation: LLMPlanner receives the model via the seam, it does not
    # build its own. Constructs offline; wrapping is a pure bind, no network.
    from agent_pipeline.model import build_model
    from agent_pipeline.agents.retriever import LLMPlanner

    planner = LLMPlanner(model=build_model())
    assert planner is not None


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="live Anthropic smoke; set ANTHROPIC_API_KEY to run (never mocked)",
)
def test_build_model_invokes_live():
    from langchain_core.messages import AIMessage
    from agent_pipeline.model import build_model

    reply = build_model().invoke("Reply with the single word: ready")
    assert isinstance(reply, AIMessage)
    assert reply.content
