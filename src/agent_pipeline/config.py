"""Harness configuration -- the provider-agnostic seams live here.

Nothing in this module contains agent reasoning; it only names the knobs the
Harness turns (model id, tool grants, step budgets).
"""

# The single tool catalog the Harness exposes to the Model (ADD: ~6 tools).
# Each agent's Harness grants only the subset inside its Context Boundary.
TOOL_CATALOG = {
    "search_knowledge",  # semantic retrieval (RAG + semantic memory)
    "get_source",        # fetch a full source document by id
    "save_scratch",      # write task-scoped working memory
    "load_scratch",      # read task-scoped working memory
    "check_claim",       # verify a claim is supported by cited sources (A4)
    "emit_contract",     # validate + hand off this agent's typed output
}

# A1 Retriever's Context Boundary: it retrieves and hands off; it does not judge
# claims (that is A4's grant).
A1_TOOL_GRANT = {
    "search_knowledge",
    "get_source",
    "save_scratch",
    "load_scratch",
    "emit_contract",
}

A1_MAX_PLAN_STEPS = 6

# Default provider-agnostic Model id ("provider:model" for init_chat_model).
# The single source of truth for the default; build_model() applies the MODEL_ID
# env override at call time. Shared across agents; reasoning-capable by default.
DEFAULT_MODEL_ID = "anthropic:claude-sonnet-5"
