"""Provider-agnostic Model layer (DESIGN §9).

One place builds the chat model; every agent receives it by injection, so swapping
providers is a config change (MODEL_ID), not a code change. init_chat_model infers
the provider from the id (an ``anthropic:claude-*`` id resolves via
langchain-anthropic). The client reads ANTHROPIC_API_KEY from the environment --
the key is never referenced here.
"""
import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from agent_pipeline.config import DEFAULT_MODEL_ID


def build_model() -> BaseChatModel:
    model_id = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
    kwargs: dict[str, object] = {}
    # Current Claude models reject a non-default temperature; forward it only when
    # explicitly configured (e.g. pinning an older model that still accepts it).
    temperature = os.getenv("MODEL_TEMPERATURE")
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    return init_chat_model(model_id, **kwargs)
