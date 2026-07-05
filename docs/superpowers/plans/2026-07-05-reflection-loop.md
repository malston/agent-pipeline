# Reflection loop (A3 ⇄ A4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the all-LLM pipeline reliably clear A4's grounding gate by letting A3 recompose with per-claim feedback when a section is unsupported, up to a bounded number of attempts, then raising if still failing.

**Architecture:** A4 gains a non-raising `check()` report mode returning the brief plus the unsupported claim texts; `run()` stays the standalone hard gate. The LangGraph pipeline adds a `composer ⇄ validator` cycle: on grounding failure with attempts remaining, loop back to A3 with feedback; otherwise route to a terminal `gate` node that raises if any check still fails. A1 and A2 run once.

**Tech Stack:** Python 3.12+, uv, pytest, Pydantic v2, LangChain v1, LangGraph 1.2.x, fastembed (keyless embeddings).

## Global Constraints

- TDD: write the failing test first, watch it fail, implement minimal, watch it pass, commit. One behavior per test.
- No mocks in tests. Use real code and injected test doubles (in-test `ClaimVerifier`/`Composer` implementing the real Protocol), never `unittest.mock`.
- Keyless offline is the default; any real-LLM path is gated on `ANTHROPIC_API_KEY` and skips (never mocks) without it.
- Run the full suite with `uv run python -m pytest -q`; output must be pristine (no warnings/errors).
- Commit on branch `feat/reflection-loop`. Push over HTTPS (`git -c credential.helper='!gh auth git-credential' push https://github.com/malston/agent-pipeline.git feat/reflection-loop:feat/reflection-loop`) — the SSH agent is currently failing.
- `MAX_COMPOSE_ATTEMPTS = 3` (1 initial compose + 2 retries).

## File structure

- `src/agent_pipeline/agents/validator.py` — add `ValidationOutcome`, add `A4Validator.check()`, refactor `A4Validator.run()` to use it.
- `src/agent_pipeline/config.py` — add `MAX_COMPOSE_ATTEMPTS`.
- `src/agent_pipeline/agents/composer.py` — thread optional `feedback` through `Composer.compose`, `RuleBasedComposer`, `LLMComposer`, `A3Composer.run`.
- `src/agent_pipeline/graph/pipeline.py` — add `feedback`/`attempt` to state, a `gate` node, and the conditional loop edge.
- `tests/test_validator.py` — `check()` behavior.
- `tests/test_composer.py` — feedback threading.
- `tests/test_reflection.py` (new) — deterministic loop mechanism.
- `tests/test_graph.py` — update `_initial()` for the new state fields.

---

### Task 1: A4 report mode — `check()` + `ValidationOutcome`

**Files:**

- Modify: `src/agent_pipeline/agents/validator.py`
- Test: `tests/test_validator.py`

**Interfaces:**

- Consumes: existing `A4Validator.__init__(verifier, banned_phrases, max_plan_steps)`, `ClaimVerifier.verify`, `validate_brief_output`, `ValidatedBrief`, `ValidationChecks`, `_cited_sources`.
- Produces: `ValidationOutcome(brief: ValidatedBrief, unsupported: list[str])`; `A4Validator.check(brief_input: BriefInput) -> ValidationOutcome` (no raise on failed check; still raises `SOURCE_UNRESOLVED` from the verifier); `A4Validator.run(brief_input) -> ValidatedBrief` (unchanged external behavior — gate raises).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_validator.py`:

```python
def test_a4_check_reports_unsupported_without_raising():
    # a claim citing a source outside the pool is unsupported (structural verifier)
    bi = BriefInput(
        request_id="r1",
        claims=[
            Claim(text="grounded claim", sources=["mito"]),
            Claim(text="ungrounded claim", sources=["ghost"]),
        ],
        body="Some body.",
        available_sources=["mito", "photo"],
    )
    outcome = A4Validator(StructuralClaimVerifier()).check(bi)
    assert outcome.brief.checks.grounding_ok is False
    assert outcome.unsupported == ["ungrounded claim"]  # only the failing claim's text

def test_a4_check_reports_grounded_for_a_good_brief():
    outcome = A4Validator(StructuralClaimVerifier()).check(_input())
    assert outcome.brief.checks.grounding_ok is True
    assert outcome.unsupported == []

def test_a4_check_propagates_source_unresolved():
    class _UnresolvedVerifier:
        def verify(self, claim, available_sources):
            raise GuardrailViolation("SOURCE_UNRESOLVED", "missing doc")

    with pytest.raises(GuardrailViolation) as exc:
        A4Validator(_UnresolvedVerifier()).check(_input())
    assert exc.value.code == "SOURCE_UNRESOLVED"
```

(`_input()` and the imports `A4Validator, StructuralClaimVerifier, GuardrailViolation, Claim, BriefInput` already exist in this file.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m pytest tests/test_validator.py -k "check" -q`
Expected: FAIL — `AttributeError: 'A4Validator' object has no attribute 'check'`.

- [ ] **Step 3: Implement `ValidationOutcome` and `check()`, refactor `run()`**

In `src/agent_pipeline/agents/validator.py`, add the model just above `class A4Validator:`:

```python
class ValidationOutcome(BaseModel):
    """A4's report-mode result: the brief plus the claim texts judged unsupported."""

    brief: ValidatedBrief
    unsupported: list[str]
```

Replace the body of `A4Validator.run` (the whole method, lines building the plan through the final `raise`) with a `check()` method plus a thin `run()`:

```python
    def check(self, brief_input: BriefInput) -> ValidationOutcome:
        # A4's plan is Harness-built: verify every claim, then emit. This is the report
        # mode -- it records failed checks but does not raise (the loop needs to retry).
        plan = Plan(
            steps=[
                PlanStep(step_id=0, intent="verify claims", tool="check_claim"),
                PlanStep(step_id=1, intent="emit the brief", tool="emit_contract"),
            ]
        )
        validate_plan(plan, A4_TOOL_GRANT, self._max_steps)  # guardrail

        available = set(brief_input.available_sources)
        unsupported: list[str] = []
        for step in plan.steps:  # EXECUTE
            if step.tool == "check_claim":
                # A verifier may raise SOURCE_UNRESOLVED (infra fault) -- let it propagate.
                unsupported = [
                    claim.text
                    for claim in brief_input.claims
                    if not self._verifier.verify(claim, available)
                ]
            elif step.tool == "emit_contract":
                body_folded = brief_input.body.casefold()
                brief = ValidatedBrief(
                    request_id=brief_input.request_id,
                    body=brief_input.body,
                    citations=_cited_sources(brief_input),
                    checks=ValidationChecks(
                        grounding_ok=not unsupported,
                        policy_ok=not any(p in body_folded for p in self._banned),
                        format_ok=bool(brief_input.body.strip()),
                    ),
                )
                return ValidationOutcome(brief=brief, unsupported=unsupported)
        # Unreachable: validate_plan guarantees a terminal emit_contract.
        raise GuardrailViolation("NO_EMIT", "plan executed without emitting a contract")

    def run(self, brief_input: BriefInput) -> ValidatedBrief:
        outcome = self.check(brief_input)
        validate_brief_output(outcome.brief)  # the standalone hard gate
        return outcome.brief
```

- [ ] **Step 4: Run the tests to verify they pass (and nothing regressed)**

Run: `uv run python -m pytest tests/test_validator.py -q`
Expected: PASS (the new `check` tests plus all existing `run`/gate tests).

- [ ] **Step 5: Commit**

```bash
git add src/agent_pipeline/agents/validator.py tests/test_validator.py
git commit -m "feat: A4 report mode -- check() returns brief + unsupported, no raise"
```

---

### Task 2: A3 accepts per-claim feedback + `MAX_COMPOSE_ATTEMPTS`

**Files:**

- Modify: `src/agent_pipeline/config.py`
- Modify: `src/agent_pipeline/agents/composer.py`
- Test: `tests/test_composer.py`

**Interfaces:**

- Consumes: existing `A3Composer`, `RuleBasedComposer`, `LLMComposer`, `Composer` protocol, `ComposerInput`, `CompositionPlan`.
- Produces: `Composer.compose(composer_input, feedback: list[str] | None = None)`; `A3Composer.run(composer_input, feedback: list[str] | None = None) -> Draft`; `config.MAX_COMPOSE_ATTEMPTS: int`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_composer.py`:

```python
def test_rule_based_composer_ignores_feedback():
    # the keyless composer is already faithful; feedback must not change its output
    ci = _input()
    assert RuleBasedComposer().compose(ci, feedback=["anything"]) == RuleBasedComposer().compose(ci)

def test_a3_run_threads_feedback_to_the_composer():
    class _CapturingComposer:
        def __init__(self):
            self.received = "unset"

        def compose(self, composer_input, feedback=None):
            self.received = feedback
            return CompositionPlan(
                steps=[PlanStep(step_id=0, intent="emit", tool="emit_contract")],
                sections=[Section(heading="H", body="B", cited_sources=["mito"])],
                style_profile="x",
            )

    capturing = _CapturingComposer()
    A3Composer(capturing).run(_input(), feedback=["unsupported claim"])
    assert capturing.received == ["unsupported claim"]
```

(`_input`, `RuleBasedComposer`, `A3Composer`, `CompositionPlan`, `PlanStep`, `Section` are already imported in this file.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m pytest tests/test_composer.py -k "feedback" -q`
Expected: FAIL — `TypeError: compose() got an unexpected keyword argument 'feedback'`.

- [ ] **Step 3: Add the config constant**

In `src/agent_pipeline/config.py`, after the `A4_BANNED_PHRASES` line, add:

```python
# Reflection loop: max A3 compose attempts before A4's gate raises (1 initial + retries).
MAX_COMPOSE_ATTEMPTS = 3
```

- [ ] **Step 4: Thread `feedback` through the composer**

In `src/agent_pipeline/agents/composer.py`:

Update the `Composer` protocol:

```python
class Composer(Protocol):
    def compose(
        self, composer_input: ComposerInput, feedback: list[str] | None = None
    ) -> CompositionPlan: ...
```

Update `RuleBasedComposer.compose` signature (body unchanged — it ignores feedback):

```python
    def compose(
        self, composer_input: ComposerInput, feedback: list[str] | None = None
    ) -> CompositionPlan:
```

Replace `LLMComposer.compose` with:

```python
    def compose(
        self, composer_input: ComposerInput, feedback: list[str] | None = None
    ) -> CompositionPlan:
        human = composer_input.model_dump_json()
        if feedback:
            human += (
                "\n\nYour previous draft made these statements that were NOT supported "
                "by their cited sources. Recompose stating only what the points assert; "
                "drop or rephrase each of these:\n"
                + "\n".join(f"- {statement}" for statement in feedback)
            )
        return self._model.invoke([("system", self._SYSTEM), ("human", human)])
```

Update `A3Composer.run` to accept and pass feedback:

```python
    def run(
        self, composer_input: ComposerInput, feedback: list[str] | None = None
    ) -> Draft:
        plan = self._composer.compose(composer_input, feedback)  # PLAN (Model)
```

(the rest of `run` is unchanged.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run python -m pytest tests/test_composer.py -q`
Expected: PASS (the two new tests plus all existing composer tests).

- [ ] **Step 6: Commit**

```bash
git add src/agent_pipeline/config.py src/agent_pipeline/agents/composer.py tests/test_composer.py
git commit -m "feat: A3 accepts per-claim feedback; add MAX_COMPOSE_ATTEMPTS"
```

---

### Task 3: Reflection graph — `composer ⇄ validator` cycle + terminal gate

**Files:**

- Modify: `src/agent_pipeline/graph/pipeline.py`
- Modify: `tests/test_graph.py` (update `_initial()`)
- Test: `tests/test_reflection.py` (new)

**Interfaces:**

- Consumes: `A4Validator.check` (Task 1), `A3Composer.run(..., feedback=...)` (Task 2), `config.MAX_COMPOSE_ATTEMPTS`, `validate_brief_output`, the three translators.
- Produces: `build_graph(retriever, analyst, composer, validator, checkpointer=None)` (same signature) compiling a graph with a `composer ⇄ validator` loop and a `gate` node; `PipelineState` with added `feedback: list[str] | None` and `attempt: int`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reflection.py`:

```python
"""The A3 <-> A4 reflection loop: recompose on grounding failure, gate at the end.

Deterministic and keyless -- an injected ClaimVerifier drives the critic so the loop
mechanism is tested without an LLM.
"""
import pytest
from langchain_core.documents import Document

from agent_pipeline.tools.embeddings import LocalEmbeddings
from agent_pipeline.tools.knowledge import KnowledgeStore
from agent_pipeline.graph.pipeline import build_graph
from agent_pipeline.agents.retriever import A1Retriever, RuleBasedPlanner
from agent_pipeline.agents.analyst import A2Analyst, RuleBasedAnalyst
from agent_pipeline.agents.composer import A3Composer, RuleBasedComposer
from agent_pipeline.agents.validator import A4Validator
from agent_pipeline.agents.guardrails import GuardrailViolation
from agent_pipeline.contracts.retrieval import RetrievalRequest

def _one_doc_store():
    store = KnowledgeStore(LocalEmbeddings())
    store.index([Document(id="mito", page_content="Mitochondria produce ATP.", metadata={})])
    return store

def _initial(request):
    return {
        "request": request,
        "retrieval": None,
        "analysis": None,
        "draft": None,
        "brief": None,
        "feedback": None,
        "attempt": 0,
    }

class _CapturingComposer:
    """Wraps RuleBasedComposer and records the feedback passed on each compose."""

    def __init__(self):
        self._inner = RuleBasedComposer()
        self.feedbacks = []

    def compose(self, composer_input, feedback=None):
        self.feedbacks.append(feedback)
        return self._inner.compose(composer_input, feedback)

class _FailOnceVerifier:
    """Fails the claim on the first check round, passes afterward."""

    def __init__(self):
        self.calls = 0

    def verify(self, claim, available_sources):
        self.calls += 1
        return self.calls > 1  # False on the 1st call, True after

class _AlwaysFailVerifier:
    def verify(self, claim, available_sources):
        return False

def _app(store, composer, verifier):
    return build_graph(
        A1Retriever(store, RuleBasedPlanner()),
        A2Analyst(RuleBasedAnalyst()),
        composer,
        A4Validator(verifier),
    )

def test_loop_recomposes_with_feedback_then_grounds():
    composer = _CapturingComposer()
    app = _app(_one_doc_store(), A3Composer(composer), _FailOnceVerifier())
    request = RetrievalRequest(request_id="r1", raw_query="how do cells make energy?")
    result = app.invoke(_initial(request), {"configurable": {"thread_id": "r1"}})

    brief = result["brief"]
    assert brief is not None and brief.checks.grounding_ok is True
    # composed twice: first with no feedback, then with the unsupported claim
    assert len(composer.feedbacks) == 2
    assert composer.feedbacks[0] is None
    assert composer.feedbacks[1] == ["Mitochondria produce ATP."]

def test_loop_raises_after_max_attempts():
    app = _app(_one_doc_store(), A3Composer(RuleBasedComposer()), _AlwaysFailVerifier())
    request = RetrievalRequest(request_id="r1", raw_query="how do cells make energy?")
    with pytest.raises(GuardrailViolation) as exc:
        app.invoke(_initial(request), {"configurable": {"thread_id": "r1"}})
    assert exc.value.code == "GROUNDING_FAILED"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m pytest tests/test_reflection.py -q`
Expected: FAIL — the graph still routes `validator -> END` with no loop, so `test_loop_recomposes_with_feedback_then_grounds` fails on `len(composer.feedbacks) == 2` (it will be 1), and the always-fail test does not raise (A4Validator here has no gate on the linear graph). Confirm the failures are about loop behavior, not import errors.

- [ ] **Step 3: Rewrite `graph/pipeline.py` with the reflection loop**

Replace the entire contents of `src/agent_pipeline/graph/pipeline.py` with:

```python
"""The Pipeline topology as a LangGraph StateGraph, with an A3 <-> A4 reflection loop.

The graph runs A1 -> A2 -> A3 -> A4. On a grounding failure with attempts remaining,
A4's report loops back to A3 to recompose with per-claim feedback; otherwise a terminal
gate validates and raises if any check still fails. A1 and A2 run once. The checkpointer
persists each stage's output. See docs/architecture/pipeline-graph.md.
"""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from agent_pipeline.agents.retriever import A1Retriever
from agent_pipeline.agents.analyst import A2Analyst
from agent_pipeline.agents.composer import A3Composer
from agent_pipeline.agents.validator import A4Validator
from agent_pipeline.agents.guardrails import validate_brief_output
from agent_pipeline.config import MAX_COMPOSE_ATTEMPTS
from agent_pipeline.translators.retrieval_to_analysis import (
    translate_retrieval_to_analysis,
)
from agent_pipeline.translators.analysis_to_composition import (
    translate_analysis_to_composition,
)
from agent_pipeline.translators.draft_to_validation import translate_draft_to_validation
from agent_pipeline.contracts.retrieval import RetrievalRequest, RetrievalBundle
from agent_pipeline.contracts.analysis import AnalysisReport
from agent_pipeline.contracts.composition import Draft
from agent_pipeline.contracts.validation import ValidatedBrief

class PipelineState(TypedDict):
    request: RetrievalRequest
    retrieval: RetrievalBundle | None
    analysis: AnalysisReport | None
    draft: Draft | None
    brief: ValidatedBrief | None
    feedback: list[str] | None  # unsupported claim texts fed back to A3 on retry
    attempt: int  # A3 compose count

def build_graph(
    retriever: A1Retriever,
    analyst: A2Analyst,
    composer: A3Composer,
    validator: A4Validator,
    checkpointer=None,
):
    def retriever_node(state: PipelineState) -> dict:
        return {"retrieval": retriever.run(state["request"])}

    def analyst_node(state: PipelineState) -> dict:
        bundle = state["retrieval"]
        if bundle is None:
            raise ValueError(
                "analyst_node reached with no retrieval bundle; "
                "A1 did not populate state['retrieval']"
            )
        return {"analysis": analyst.run(translate_retrieval_to_analysis(bundle))}

    def composer_node(state: PipelineState) -> dict:
        report = state["analysis"]
        if report is None:
            raise ValueError(
                "composer_node reached with no analysis report; "
                "A2 did not populate state['analysis']"
            )
        draft = composer.run(
            translate_analysis_to_composition(report), feedback=state.get("feedback")
        )
        return {"draft": draft, "attempt": state.get("attempt", 0) + 1}

    def validator_node(state: PipelineState) -> dict:
        draft = state["draft"]
        if draft is None:
            raise ValueError(
                "validator_node reached with no draft; "
                "A3 did not populate state['draft']"
            )
        outcome = validator.check(translate_draft_to_validation(draft))
        return {"brief": outcome.brief, "feedback": outcome.unsupported}

    def gate_node(state: PipelineState) -> dict:
        validate_brief_output(state["brief"])  # raises if any check still fails
        return {}

    def route(state: PipelineState) -> str:
        if state["brief"].checks.grounding_ok or state["attempt"] >= MAX_COMPOSE_ATTEMPTS:
            return "gate"
        return "composer"

    graph = StateGraph(PipelineState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("composer", composer_node)
    graph.add_node("validator", validator_node)
    graph.add_node("gate", gate_node)
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "composer")
    graph.add_edge("composer", "validator")
    graph.add_conditional_edges("validator", route, {"gate": "gate", "composer": "composer"})
    graph.add_edge("gate", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
```

- [ ] **Step 4: Update `tests/test_graph.py` `_initial()` for the new state fields**

In `tests/test_graph.py`, replace the `_initial` helper with:

```python
def _initial(request):
    return {
        "request": request,
        "retrieval": None,
        "analysis": None,
        "draft": None,
        "brief": None,
        "feedback": None,
        "attempt": 0,
    }
```

- [ ] **Step 5: Run the reflection + graph tests to verify they pass**

Run: `uv run python -m pytest tests/test_reflection.py tests/test_graph.py -q`
Expected: PASS — the loop retries once then grounds; the always-fail case raises `GROUNDING_FAILED`; the existing 4-stage graph tests still produce a grounded brief on attempt 1.

- [ ] **Step 6: Run the full suite**

Run: `uv run python -m pytest -q`
Expected: PASS, pristine output (live e2e tests skip without a key).

- [ ] **Step 7: Commit**

```bash
git add src/agent_pipeline/graph/pipeline.py tests/test_reflection.py tests/test_graph.py
git commit -m "feat: A3 <-> A4 reflection loop in the pipeline graph"
```

---

### Task 4: Live verification + DESIGN.md note

**Files:**

- Modify: `DESIGN.md` (topology section)

**Interfaces:**

- Consumes: the merged reflection loop; a real `ANTHROPIC_API_KEY` (from `.env`) for the manual live check.

- [ ] **Step 1: Verify live that the loop improves the all-LLM pipeline**

Run (loads the key from `.env`, does not commit anything):

```bash
set -a; . ./.env; set +a
uv run python - <<'PY'
from langchain_core.documents import Document
from agent_pipeline.tools.embeddings import LocalEmbeddings
from agent_pipeline.tools.knowledge import KnowledgeStore
from agent_pipeline.agents.retriever import A1Retriever, RuleBasedPlanner
from agent_pipeline.agents.analyst import A2Analyst, LLMAnalyst
from agent_pipeline.agents.composer import A3Composer, LLMComposer
from agent_pipeline.agents.validator import A4Validator, LLMClaimVerifier
from agent_pipeline.graph.pipeline import build_graph
from agent_pipeline.contracts.retrieval import RetrievalRequest
ks = KnowledgeStore(LocalEmbeddings())
ks.index([
  Document(id="mito", page_content="The mitochondrion is the powerhouse of the cell, producing ATP.", metadata={}),
  Document(id="photo", page_content="Photosynthesis converts sunlight into chemical energy in plants.", metadata={}),
  Document(id="econ", page_content="Central banks adjust interest rates to influence inflation.", metadata={}),
])
app = build_graph(A1Retriever(ks, RuleBasedPlanner()), A2Analyst(LLMAnalyst()), A3Composer(LLMComposer()), A4Validator(LLMClaimVerifier(ks)))
for i in range(3):
    try:
        out = app.invoke({"request": RetrievalRequest(request_id=f"r{i}", raw_query="how do cells make energy?"),
                          "retrieval": None, "analysis": None, "draft": None, "brief": None, "feedback": None, "attempt": 0},
                         {"configurable": {"thread_id": f"r{i}"}})
        print(f"run {i}: grounded, attempts spent, citations={out['brief'].citations}")
    except Exception as e:
        print(f"run {i}: {type(e).__name__}: {str(e)[:80]}")
PY
```

Expected: runs now pass grounding markedly more often than before the loop (with the toy one-sentence corpus some runs may still exhaust — the loop raises reliability, it does not guarantee it on a thin corpus). Record the observed pass rate in the PR description. Do NOT commit a hard live test (it is stochastic; faithfulness rate belongs in the eval harness).

- [ ] **Step 2: Add a topology note to `DESIGN.md`**

In `DESIGN.md`, at the end of section `## 2. Topology — Pipeline of 4` (just before the `---` that precedes section 3), add:

```markdown
**Reflection loop.** A3 ⇄ A4 form an ADD Reflection loop: when A4's grounding check
rejects a section, the graph recomposes A3 with the unsupported claims as feedback, up
to `MAX_COMPOSE_ATTEMPTS`, then raises. See
[docs/architecture/pipeline-graph.md](docs/architecture/pipeline-graph.md).
```

- [ ] **Step 3: Commit**

```bash
git add DESIGN.md
git commit -m "docs: note the A3 <-> A4 reflection loop in DESIGN topology"
```

---

## Self-Review

**Spec coverage:** A4 report mode + `SOURCE_UNRESOLVED` propagation (Task 1); per-claim feedback + `MAX_COMPOSE_ATTEMPTS` (Task 2); reflection graph with conditional edge, gate, and raise-on-exhaustion (Task 3); deterministic loop tests + live smoke verification + DESIGN note (Tasks 3-4). Policy/format falling through to the gate is covered by `gate_node` calling the unchanged `validate_brief_output`. All spec sections map to a task.

**Placeholder scan:** none — every code and test block is complete and concrete.

**Type consistency:** `check() -> ValidationOutcome` (Task 1) is consumed by `validator_node` as `outcome.brief` / `outcome.unsupported` (Task 3); `compose(composer_input, feedback=None)` (Task 2) is called by `composer_node` via `A3Composer.run(..., feedback=...)` (Task 3); `MAX_COMPOSE_ATTEMPTS` defined in Task 2 and read by `route` in Task 3. `_FailOnceVerifier`/`_CapturingComposer` implement the real `ClaimVerifier`/`Composer` Protocols. Consistent.
