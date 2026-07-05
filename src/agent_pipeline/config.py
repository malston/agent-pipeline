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
# claims (that is A4's grant). The grant lists only tools the executor actually
# honors, so a planned tool it cannot run is rejected loudly (TOOL_NOT_GRANTED)
# rather than silently skipped.
A1_TOOL_GRANT = {
    "search_knowledge",
    "save_scratch",
    "emit_contract",
}

A1_MAX_PLAN_STEPS = 6

# A2 Analyst's Context Boundary: it reasons over the evidence pool it is given and
# emits. It does not retrieve more evidence (a future enhancement would add
# search_knowledge/get_source) nor judge claim support (that is A4's `check_claim`).
# The grant lists only tools the executor honors, so a planned retrieval step is
# rejected loudly rather than silently ignored.
A2_TOOL_GRANT = {"emit_contract"}

A2_MAX_PLAN_STEPS = 6

# A3 Composer's Context Boundary: it composes a draft from the points it is given
# and emits. It does not retrieve or judge; its only tool is the hand-off. (Style
# exemplars via retrieval would be a future enhancement adding search_knowledge.)
A3_TOOL_GRANT = {"emit_contract"}

A3_MAX_PLAN_STEPS = 6

# A4 Validator's Context Boundary: it verifies each claim against its cited sources
# (check_claim) and gates the brief; it is the only agent granted check_claim.
A4_TOOL_GRANT = {"check_claim", "emit_contract"}

A4_MAX_PLAN_STEPS = 6

# Phrases the validator's policy check rejects in the brief body. Empty by default
# (permissive); set to enforce a content policy.
A4_BANNED_PHRASES: frozenset[str] = frozenset()

# Reflection loop: max A3 compose attempts before A4's gate raises (1 initial + retries).
MAX_COMPOSE_ATTEMPTS = 3

# Default provider-agnostic Model id ("provider:model" for init_chat_model).
# The single source of truth for the default; build_model() applies the MODEL_ID
# env override at call time. Shared across agents; reasoning-capable by default.
DEFAULT_MODEL_ID = "anthropic:claude-sonnet-5"
