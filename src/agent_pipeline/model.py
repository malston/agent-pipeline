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
    # A blank/whitespace MODEL_ID (e.g. `MODEL_ID=` in a .env) counts as unset.
    model_id = (os.getenv("MODEL_ID") or "").strip() or DEFAULT_MODEL_ID
    kwargs: dict[str, float] = {}
    # claude-sonnet-5 returns HTTP 400 ("`temperature` is deprecated for this model")
    # for a non-default temperature, so forward it only when MODEL_TEMPERATURE is
    # explicitly set (for pinning an older model that still accepts it); otherwise
    # omit it and let the provider default stand. A blank value counts as unset.
    raw_temperature = os.getenv("MODEL_TEMPERATURE")
    if raw_temperature is not None and raw_temperature.strip():
        try:
            kwargs["temperature"] = float(raw_temperature)
        except ValueError as exc:
            raise ValueError(
                f"MODEL_TEMPERATURE must be a number (e.g. '0' or '0.7'); "
                f"got {raw_temperature!r}"
            ) from exc
    return init_chat_model(model_id, **kwargs)
