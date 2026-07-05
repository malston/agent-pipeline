# Agent Pipeline — Design (Agent-Driven Design)

A four-agent **Pipeline** where each agent is a **Model + Harness** pair, runs a
**Plan-Execute** loop, and hands a **typed contract** to the next agent through an
explicit **Context Translation** step. The Model is provider-agnostic; the Harness is
built on LangChain / LangGraph. Improvement follows **Harness-Driven Design (HDD)**:
the Model is fixed, and we treat failures as Harness failures until proven otherwise.

> **Illustrative domain.** To keep the design concrete, the pipeline turns a raw
> request into a validated, sourced brief: **Retrieve → Analyze → Compose → Validate.**
> The architecture is domain-agnostic; swap the stage semantics and the contracts and
> the same skeleton holds.

> **Implementation note (honesty).** Exact LangChain / LangGraph / LangSmith API
> signatures below are illustrative of the intended shape. Pin them against current docs
> at build time (`init_chat_model`, `StateGraph`, checkpointer classes, `evaluate`,
> feedback-on-run APIs all exist but their exact parameters move between versions).

---

## 1. ADD layer map for this system

```text
  USER GOAL: "raw request → validated, sourced brief"
        │
  ┌─────▼──────────────────────────────────────────────────────┐
  │ TOPOLOGY LAYER — Pipeline of 4 (sequential, contract handoff)│
  └─────┬──────────────────────────────────────────────────────┘
  ┌─────▼──────────────────────────────────────────────────────┐
  │ AGENT LAYER — each agent = Harness + Model                   │
  │   Harness: prompts, 6-tool catalog (subset per agent),      │
  │            Plan-Execute loop, memory, guardrails, translators│
  │   Model:   reasoning / generation / judgment (swappable)     │
  └─────┬──────────────────────────────────────────────────────┘
  ┌─────▼──────────────────────────────────────────────────────┐
  │ INTEGRATION LAYER — tools (MCP/direct), typed contracts (A2A)│
  └─────┬──────────────────────────────────────────────────────┘
  ┌─────▼──────────────────────────────────────────────────────┐
  │ INFRA LAYER — vector DB, state/memory store, LangSmith        │
  └─────────────────────────────────────────────────────────────┘
```

Every requirement maps to exactly one layer. Reasoning is the Model's; everything
structural (flow, tools, memory, validation, translation) is the Harness's; storage and
tracing are Infra that _serve_ the Harness and hold no agent logic.

---

## 2. Topology — Pipeline of 4

```text
  request
    │
    ▼                       ▼ = Context Translation boundary (typed contract, never NL)
  ┌───────────────┐   RetrievalBundle   ┌───────────────┐
  │ A1 Retriever  │───────▼─────────────│ A2 Analyst    │
  │ (RAG intake)  │                     │ (findings)    │
  └───────────────┘                     └──────┬────────┘
                                     AnalysisReport
                                                │
  ┌───────────────┐    Draft            ┌───────▼───────┐
  │ A4 Validator  │◄──────▼─────────────│ A3 Composer   │
  │ (grounding /  │                     │ (deliverable) │
  │  policy gate) │                     └───────────────┘
  └──────┬────────┘
         ▼
   ValidatedBrief  (final output)
```

| Agent            | Role          | Transformation                     | Owns                                             |
| ---------------- | ------------- | ---------------------------------- | ------------------------------------------------ |
| **A1 Retriever** | RAG intake    | `request → RetrievalBundle`        | query normalization, retrieval, source selection |
| **A2 Analyst**   | reasoning     | `RetrievalBundle → AnalysisReport` | claim extraction, evidence binding               |
| **A3 Composer**  | generation    | `AnalysisReport → Draft`           | drafting the deliverable to spec                 |
| **A4 Validator** | judgment/gate | `Draft → ValidatedBrief`           | grounding check, policy/format gate              |

**Why Pipeline (not Orchestrator or Fan-out):** the steps are inherently sequential
(each depends on the prior), the input/output contracts are clean, and debuggability is
first-class — you can inspect the typed artifact between every stage.

**Cascade caution.** Pipeline failures cascade: A2 cannot run on a bad `RetrievalBundle`.
Mitigations (all Harness):

- **Output guardrail per agent** (§6) — a stage cannot emit an invalid contract.
- **Checkpoint after every stage** via the LangGraph checkpointer, so a failed A3 resumes
  from A2's validated output instead of re-running the whole pipeline.
- **Bounded re-prompt** on guardrail failure, then hard-fail the stage with a typed error
  captured in the trace.

**Reflection loop.** A3 ⇄ A4 form an ADD Reflection loop: when A4's grounding check
rejects a section, the graph recomposes A3 with the unsupported claims as feedback, up
to `MAX_COMPOSE_ATTEMPTS`, then raises. See
[docs/architecture/pipeline-graph.md](docs/architecture/pipeline-graph.md).

---

## 3. Anatomy of one agent (Model + Harness)

Each of the four agents is the same skeleton with a different prompt, tool subset, and
contract pair. The Harness is deterministic and testable; the Model does only reasoning,
generation, and judgment.

```bash
  incoming contract
        │
  ┌─────▼─────────────────────────────────────────────────────────┐
  │ HARNESS                                                        │
  │                                                                │
  │  (a) INPUT GUARDRAIL   validate incoming contract vs schema    │
  │  (b) CONTEXT ASSEMBLY  system prompt + RAG retrieval +         │
  │                        working-memory scratch + tool defs      │
  │        │                                                       │
  │        ▼   PLAN                                                 │
  │  ┌───────────────┐  Model emits explicit plan (list of steps)  │
  │  │    MODEL      │  Harness validates plan (allowed tools,     │
  │  │  (plan step)  │  step budget) BEFORE any execution          │
  │  └──────┬────────┘                                             │
  │        │   EXECUTE (Harness iterates plan steps in order)      │
  │        ▼                                                       │
  │  for each step:  Model chooses tool → Harness executes tool →  │
  │                  result appended to working memory → next step │
  │        │   final step MUST call emit_contract                  │
  │        ▼                                                       │
  │  (c) OUTPUT GUARDRAIL  schema + policy + grounding checks;     │
  │                        re-prompt ≤N, else typed stage failure  │
  │        │                                                       │
  │  (d) CONTEXT TRANSLATION  map this agent's vocabulary →        │
  │                           next agent's input contract          │
  └────────┼───────────────────────────────────────────────────────┘
           ▼
      outgoing contract (checkpointed)
```

### Plan-Execute loop (the required loop pattern)

Chosen over ReAct because the plan is **visible and auditable before execution** — the
Harness can validate it, bound it, or (optionally) gate it for human approval.

1. **Plan.** One Model call produces an explicit, ordered plan: a list of
   `{step_id, intent, tool, why}`. No side effects yet.
2. **Validate plan (Harness guardrail).** Reject plans that reference tools outside this
   agent's grant, exceed the step budget, or skip the mandatory terminal
   `emit_contract` step. Re-prompt on rejection.
3. **Execute.** The Harness iterates the plan. Each step is a Model call scoped to that
   step; the Model requests a tool call, the Harness executes it, and the result is
   written to **working memory** (not dumped back into the context window wholesale — the
   Harness injects only what the next step needs, to avoid context bloat).
4. **Emit.** The terminal step calls `emit_contract`, which runs the output guardrail.

---

## 4. Contracts & Context Translation

**Rule: agents never exchange natural language.** Every boundary is a typed contract
(Pydantic), and every boundary has an explicit **Translator** — deterministic Harness
code that maps the producer's vocabulary into the consumer's. The Translator is where
field projection, renaming, and enrichment happen; it is unit-testable and independent of
any Model.

```python
# Contracts — one per pipeline edge (Pydantic v2, illustrative)
class RetrievalBundle(BaseModel):
    request_id: str
    normalized_query: str
    passages: list[Passage]         # {source_id, span, text, score}
    coverage: float                 # retrieval confidence signal

class AnalysisReport(BaseModel):
    request_id: str
    findings: list[Finding]         # {claim, evidence: list[source_id], confidence}
    gaps: list[str]                 # what evidence is missing

class Draft(BaseModel):
    request_id: str
    sections: list[Section]         # {heading, body, cited_sources: list[str]}
    style_profile: str

class ValidatedBrief(BaseModel):
    request_id: str
    body: str
    citations: list[Citation]
    checks: ValidationChecks        # grounding_ok, policy_ok, format_ok
```

```python
# Translator — the Context Translation step. Vocabulary changes here, in code.
def translate_retrieval_to_analysis(b: RetrievalBundle) -> AnalystInput:
    # "passages / coverage" (retrieval vocab)  →  "evidence pool" (analysis vocab)
    return AnalystInput(
        request_id=b.request_id,
        question=b.normalized_query,
        evidence_pool=[Evidence(id=p.source_id, text=p.text) for p in b.passages],
        retrieval_confidence=b.coverage,
    )
```

Each edge (A1→A2, A2→A3, A3→A4) has its own translator. Because contracts are typed and
translators are explicit, a producer-side rename never silently breaks the consumer — it
breaks the translator's tests instead.

### Integration boundary: why in-process, not A2A

The agents talk to each other **in-process**: they are nodes of one LangGraph `StateGraph`,
and a boundary is a function call passing a validated Pydantic contract through a translator.
We deliberately do **not** use the A2A wire protocol (the ADD integration layer's horizontal
option) for these edges.

A2A earns its cost when agents are _distributed or independently owned_ — separate
deployments, cross-framework/polyglot agents, dynamic capability discovery via Agent Cards,
or trust boundaries between parties. It brings HTTP transport, a task lifecycle, artifacts,
and per-hop auth to solve those problems. This pipeline has none of them: four agents, one
process, one framework, one team, a static sequence. A2A here would wrap the same typed
contracts in a heavier envelope and duplicate orchestration LangGraph already provides — for
zero functional gain.

Crucially, we already keep A2A's one load-bearing idea for free: **agents are opaque to each
other**, exchanging only typed contracts and never reaching into another's memory, tools, or
prompt. That is the property that matters, realized at function-call cost.

**When to introduce A2A** — it is a _per-edge_ decision, not all-or-nothing. Adopt it on the
specific edge that first crosses a process, org, or vendor line (e.g. a stage becomes an
independently-scaled service, or A4 delegates to a third-party validator). Because the
boundaries are already clean, the migration is mechanical: each contract → an A2A
artifact/`DataPart` schema, each translator → the boundary adapter, each agent's `run()` → a
`/tasks/send` handler. Deferring A2A costs nothing today and forecloses nothing later.

---

## 5. The six tools

A single tool catalog is exposed to the Model layer; each agent's Harness **grants only
the subset inside its Context Boundary** (ADD: small, relevant tool set per agent). Tools
are named for what they do in the domain, return typed output, and fail loudly.

| #   | Tool                                  | Purpose                                                                | Backed by               |
| --- | ------------------------------------- | ---------------------------------------------------------------------- | ----------------------- |
| 1   | `search_knowledge(query, k, filters)` | Semantic retrieval — RAG + semantic memory                             | Vector DB               |
| 2   | `get_source(source_id)`               | Fetch full source doc by id                                            | Doc store               |
| 3   | `save_scratch(key, value)`            | Write task-scoped working memory                                       | State store             |
| 4   | `load_scratch(key)`                   | Read task-scoped working memory                                        | State store             |
| 5   | `check_claim(claim, source_ids)`      | Verify a claim is supported by cited sources                           | Vector DB + Model judge |
| 6   | `emit_contract(payload)`              | Validate payload vs downstream schema and set it as the agent's output | Harness                 |

**Per-agent grants (Context Boundary enforcement):**

| Agent        | 1 search | 2 get_source | 3 save | 4 load | 5 check_claim | 6 emit |
| ------------ | :------: | :----------: | :----: | :----: | :-----------: | :----: |
| A1 Retriever |    ✅    |      ✅      |   ✅   |   ✅   |               |   ✅   |
| A2 Analyst   |    ✅    |      ✅      |   ✅   |   ✅   |               |   ✅   |
| A3 Composer  |    ✅    |              |   ✅   |   ✅   |               |   ✅   |
| A4 Validator |    ✅    |      ✅      |        |   ✅   |      ✅       |   ✅   |

`emit_contract` is universal — it is how the Model signals "done, here is my typed
output," and it is the choke point where the output guardrail runs.

---

## 6. Validation & guardrails (every output, before it leaves)

Three guardrail gates per agent, all Harness:

1. **Input guardrail** — incoming contract validated against its schema before the agent
   runs. Cascade stops here if the upstream produced junk.
2. **Plan guardrail** — plan validated for allowed tools + step budget + mandatory
   terminal emit (§3).
3. **Output guardrail** — before the contract leaves the agent:
   - **Schema**: `emit_contract` payload must parse into the outgoing Pydantic model.
   - **Policy**: PII / safety / banned-content checks (deterministic rules; escalate to a
     Model judge only where rules can't decide).
   - **Grounding** (A4 especially): every citation must resolve to a real `source_id` and
     `check_claim` must pass — no unsupported claims leave the pipeline.
   - On failure: **re-prompt ≤ N times** with the validator's structured error injected,
     then emit a typed stage failure that is checkpointed and traced (never a silent pass).

---

## 7. Memory

| Memory                    | Scope                        | Where                                      | Accessed via                                                   |
| ------------------------- | ---------------------------- | ------------------------------------------ | -------------------------------------------------------------- |
| **Working** (short-term)  | one task / one pipeline run  | State store (LangGraph checkpointer state) | `save_scratch` / `load_scratch`, plus Harness context assembly |
| **Semantic** (embeddings) | global, shared across agents | Vector DB                                  | `search_knowledge`, RAG pipeline                               |

Both are **Harness-owned** and read/written on the Model's behalf. Working memory holds
intermediate step results so the Plan-Execute loop can pass data between steps _without_
letting the context window grow unbounded (anti-pattern: context bloat). Semantic memory
is the retrieval corpus that RAG draws on.

---

## 8. RAG pipeline (Harness pattern, injected before the Model reasons)

```text
  agent query ─► embed ─► vector search (k) ─► re-rank ─► top-k by relevance+diversity
             ─► assemble into context window (most-relevant last) ─► Model reasons
```

Retrieval happens in the Harness _before_ the Model call; the Model only reasons over what
was retrieved. All RAG knobs — embedding model, chunk size/overlap, `k`, re-rank
strategy, hybrid (semantic + keyword), assembly order — are Harness config and therefore
HDD levers. A1 leans on RAG hardest; A2/A3/A4 use it for targeted lookups
(evidence, style exemplars, policy text).

---

## 9. Provider-agnostic Model layer

The Model sits behind LangChain's chat-model interface so it can be swapped by config
without touching Harness code:

```python
# One place defines the Model (model.py: build_model()); every agent receives it by
# injection. build_model() reads MODEL_ID (default DEFAULT_MODEL_ID) at call time.
model = init_chat_model(settings.MODEL_ID)   # e.g. anthropic:claude-sonnet-5 / gpt / gemini
# NOTE: no default temperature. Current Claude models (e.g. claude-sonnet-5) reject a
# non-default temperature with HTTP 400 ("`temperature` is deprecated for this model"),
# so build_model() forwards temperature only when MODEL_TEMPERATURE is explicitly set.
# Structured output binds a contract to the Model's response:
planner = model.with_structured_output(RetrievalPlan)
emitter = model.with_structured_output(AnalysisReport)   # per-agent outgoing contract
```

Swapping providers = changing `MODEL_ID`. Prompts, tools, contracts, translators, memory,
and graph wiring are untouched. This is what makes HDD possible: the Model is a fixed,
replaceable engine.

---

## 10. LangGraph implementation shape

The pipeline is a `StateGraph` with one node per agent and translator edges over a shared
typed state. Each agent node is itself a compiled Plan-Execute runnable.

```python
class PipelineState(TypedDict):
    request_id: str
    retrieval: RetrievalBundle | None
    analysis: AnalysisReport | None
    draft: Draft | None
    result: ValidatedBrief | None
    errors: list[StageError]

g = StateGraph(PipelineState)
g.add_node("retriever", run_retriever)     # each wraps: guardrail→plan→execute→guardrail
g.add_node("analyst",   run_analyst)
g.add_node("composer",  run_composer)
g.add_node("validator", run_validator)

g.add_edge(START, "retriever")
g.add_edge("retriever", "analyst")         # translator invoked inside the node boundary
g.add_edge("analyst",   "composer")
g.add_edge("composer",  "validator")
g.add_edge("validator", END)

app = g.compile(checkpointer=checkpointer)  # checkpoint after every stage
```

- **Checkpointer** = the state/memory store (e.g., a Postgres-backed saver). Gives
  per-stage resume and holds working memory.
- **HITL (optional)** — LangGraph `interrupt` after the plan guardrail lets a human
  approve a plan before execution, exactly where Plan-Execute makes it cheap.

---

## 11. Infrastructure (serves the Harness, holds no agent logic)

| Infra                    | Responsibility                                             | Provider-agnostic seam                                           |
| ------------------------ | ---------------------------------------------------------- | ---------------------------------------------------------------- |
| **Vector DB**            | semantic memory + RAG corpus + `check_claim` support       | LangChain `VectorStore` interface (pgvector / Qdrant / Pinecone) |
| **State / memory store** | working memory, pipeline state, per-stage checkpoints      | LangGraph checkpointer (Postgres/SQLite saver)                   |
| **LangSmith**            | one full trace per run; eval scores bound to trace/run IDs | `LANGSMITH_TRACING=true` + tracing callbacks                     |

**Tracing + eval binding (required):**

- Every pipeline run emits **one trace** spanning all four agents, every Model call, and
  every tool call — correlated by `request_id` / `session_id`.
- Evals run via LangSmith `evaluate()` over a dataset; **every eval score is attached to
  the run/trace ID** it scored (LangSmith feedback-on-run), so a failing score links
  straight to the trace that produced it. This is the evidence base for HDD decisions.

Per-agent signals to emit (Observability): `agent_id`, `model_id`, in/out tokens,
latency, `tool_calls[]`, `success`/`error_type`, and per stage the incoming/outgoing
contract ids.

---

## 12. Improvement strategy — Harness-Driven Design (HDD)

**The Model is fixed.** When a stage underperforms, diagnose the Harness first and exhaust
HDD before considering any Model change. Most failures here are Harness failures — a bad
retrieval, a vague tool description, a leaky contract, a weak prompt.

| Symptom (seen in a trace)          | Most-likely Harness cause    | HDD intervention (cheapest first)                             |
| ---------------------------------- | ---------------------------- | ------------------------------------------------------------- |
| A1 misses relevant sources         | retrieval config             | tune `k` / chunking / hybrid / re-rank in RAG                 |
| A2 cites wrong evidence            | ambiguous contract or prompt | tighten `Finding` schema; sharpen system prompt; add few-shot |
| Model calls the wrong tool         | tool naming/description      | rename tool; add examples to tool description                 |
| A3 output drifts off-spec          | context assembly             | inject style exemplars via RAG; add output few-shot           |
| Consumer breaks on producer change | informal handoff             | fix the **translator**; add contract tests                    |
| Quality decays over long steps     | context bloat                | prune working memory; inject only step-relevant context       |
| Stage fails silently               | missing guardrail            | add/strengthen output guardrail + re-prompt                   |
| Wrong task reaches a stage         | routing/edges                | fix graph wiring / plan guardrail                             |

**Escalate to LLM-Driven Design only when** evals (attached to trace IDs) show the failure
persists across HDD variants with correct, complete context and well-described tools —
i.e., the Model reasons wrong despite a clean Harness. Then, in cost order: model-tier
upgrade → chain-of-thought → (last resort) fine-tuning. **Never fine-tune to inject
changing knowledge (use RAG), to fix a prompt, or to paper over a Harness bug.**

The loop, concretely: reproduce the failure → open its trace → form a Harness hypothesis →
change one Harness lever → re-run the eval → compare scores on the same dataset → keep or
revert. The trace is the unit of debugging; the eval score bound to it is the unit of
proof.

---

## 13. Status / open items

**Shipped (on `main`, TDD throughout):**

- The full `A1 → A2 → A3 → A4` pipeline producing a `ValidatedBrief`, with typed contracts
  and Context Translators on every edge and guardrails on every output.
- Provider-agnostic Model layer (`model.build_model()`, default `anthropic:claude-sonnet-5`),
  swappable by `MODEL_ID`. Dev defaults: `InMemoryVectorStore` (semantic memory) and the
  in-memory LangGraph checkpointer.
- Eval harness (#5): retrieval recall@k / MRR and system-scope citation recall/precision,
  every score bound to a per-run trace id; golden datasets live in `tests/`.

**In progress:**

- A3 ⇄ A4 reflection loop — recompose on grounding failure with per-claim feedback, up to
  `MAX_COMPOSE_ATTEMPTS`, then raise (this branch; see
  [docs/architecture/pipeline-graph.md](docs/architecture/pipeline-graph.md)).

**Open (key/service-gated, tracked in #6):**

- Production infra: pgvector (or Qdrant) behind the `VectorStore` seam; a Postgres-backed
  checkpointer; LangSmith tracing (one trace per run) plus the eval LangSmith adapter
  (push each trace as a run and each score as feedback-on-run).
- Richer eval datasets and metrics as a real corpus replaces the toy fixtures.
