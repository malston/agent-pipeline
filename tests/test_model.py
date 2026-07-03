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
    assert model.temperature is None  # omitted -> provider default applies


def test_build_model_honors_model_id_override(monkeypatch):
    monkeypatch.setenv("MODEL_ID", "anthropic:claude-opus-4-8")
    monkeypatch.delenv("MODEL_TEMPERATURE", raising=False)
    from agent_pipeline.model import build_model

    assert build_model().model == "claude-opus-4-8"


def test_build_model_treats_blank_model_id_as_default(monkeypatch):
    # A blank MODEL_ID (e.g. `MODEL_ID=` in a .env) means "unset", not an empty id
    # that init_chat_model would choke on.
    monkeypatch.delenv("MODEL_TEMPERATURE", raising=False)
    from agent_pipeline.model import build_model
    from agent_pipeline.config import DEFAULT_MODEL_ID

    for blank in ("", "   "):
        monkeypatch.setenv("MODEL_ID", blank)
        assert DEFAULT_MODEL_ID.endswith(build_model().model)


def test_build_model_forwards_temperature_when_set(monkeypatch):
    # A non-zero value proves the env string is parsed and forwarded, not defaulted.
    monkeypatch.delenv("MODEL_ID", raising=False)
    monkeypatch.setenv("MODEL_TEMPERATURE", "0.7")
    from agent_pipeline.model import build_model

    assert build_model().temperature == 0.7


def test_build_model_treats_blank_temperature_as_unset(monkeypatch):
    # A blank MODEL_TEMPERATURE (e.g. `MODEL_TEMPERATURE=` in a .env) means "unset",
    # not "temperature = error".
    monkeypatch.delenv("MODEL_ID", raising=False)
    from agent_pipeline.model import build_model

    for blank in ("", "   "):
        monkeypatch.setenv("MODEL_TEMPERATURE", blank)
        assert build_model().temperature is None


def test_build_model_rejects_malformed_temperature_descriptively(monkeypatch):
    monkeypatch.delenv("MODEL_ID", raising=False)
    monkeypatch.setenv("MODEL_TEMPERATURE", "high")
    from agent_pipeline.model import build_model

    with pytest.raises(ValueError, match="MODEL_TEMPERATURE"):
        build_model()


def test_llm_planner_uses_injected_model_without_calling_factory(monkeypatch):
    # Prove the injected model is actually used: a tripwire makes the factory
    # fallback fail loudly, so this passes only if LLMPlanner honors the argument
    # instead of quietly building its own.
    from agent_pipeline.model import build_model
    from agent_pipeline.agents import retriever

    injected = build_model()

    def _factory_must_not_run():
        raise AssertionError("build_model() must not run when a model is injected")

    monkeypatch.setattr(retriever, "build_model", _factory_must_not_run)
    planner = retriever.LLMPlanner(model=injected)
    assert planner._model is not None


def test_llm_planner_defaults_to_build_model():
    # The model=None path builds from the factory; keyless, so it constructs offline.
    from agent_pipeline.agents.retriever import LLMPlanner

    assert LLMPlanner()._model is not None


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
