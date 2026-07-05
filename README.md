# agent-pipeline

A four-agent **Pipeline** built with Agent-Driven Design: every agent is a
**Model + Harness** pair, runs a **Plan-Execute** loop, and hands a **typed
contract** to the next agent through explicit **Context Translation**. The Model
is provider-agnostic (LangChain / LangGraph); improvement follows Harness-Driven
Design. Full design in [DESIGN.md](DESIGN.md).

```text
request -> A1 Retriever -> A2 Analyst -> A3 Composer -> A4 Validator -> ValidatedBrief
```

See the **[pipeline graph](docs/architecture/pipeline-graph.md)** for the full LangGraph
topology, including the A3 &#8646; A4 reflection loop.

## Status

- All four agents built end-to-end (TDD): A1 Retriever, A2 Analyst, A3 Composer, and
  A4 Validator, wired `A1 -> A2 -> A3 -> A4` with a Context Translator on every edge and
  guardrails on every output.
- An eval harness scores retrieval and citation quality, with results bound to trace ids.
- In progress: the A3 &#8646; A4 reflection loop (see the graph above).

## Layout

```text
src/agent_pipeline/
  contracts/     typed boundaries between agents (Pydantic)
  tools/         knowledge store (RAG), embeddings, working memory
  agents/        agent harness: planners, Plan-Execute loop, guardrails
  translators/   Context Translation between agent vocabularies
  graph/         LangGraph StateGraph wiring + checkpointing
  evals/         agent/system evals (metrics + harness) bound to trace ids
```

## Develop

Requires [uv](https://docs.astral.sh/uv/). Python 3.12+ is fetched automatically.

```bash
uv sync                 # install (add --extra anthropic for the LLM path)
uv run pytest -q        # run the suite (first run downloads a small embedding model)
```

## Model provider

The retriever runs today with `RuleBasedPlanner` (keyless, deterministic). The
real Model path uses `LLMPlanner`, swappable by `MODEL_ID` (a `provider:model`
string for `init_chat_model`) and gated on a provider key:

```bash
export ANTHROPIC_API_KEY=...        # then the real-LLM end-to-end test runs
export MODEL_ID=anthropic:claude-haiku-4-5-20251001   # optional override
```

Without a key, the real-LLM end-to-end test skips (it is never mocked).
