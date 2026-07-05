# Ground the Emitted Body (#19) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A4 grounds the prose that actually ships — a content section that cites nothing becomes an ungrounded assertion the reflection loop must fix, instead of escaping verification.

**Architecture:** Gaps move from an uncited `Section` to a first-class `Draft.gaps: list[str]`, so every `Section` is a content assertion. The A3→A4 translator routes cited sections to `claims` (as today) and uncited sections' text to a new `BriefInput.uncited_assertions`; `A4Validator.check()` folds those into `unsupported`, so they set `grounding_ok=False` and feed the reflection loop. `Claim.sources` keeps its `min_length=1` invariant — uncited prose is modeled as its own thing, not a degenerate claim.

**Tech Stack:** Python 3.12+, Pydantic v2, LangGraph, pytest, `uv`.

## Global Constraints

- TDD: failing test first, watch it fail, minimal code, watch it pass, commit. Never delete/weaken a test to pass.
- No mocks; keyless deterministic tests use real `fastembed` + Protocol test doubles. The LLM path is exercised only by key-gated e2e tests (skipped without `ANTHROPIC_API_KEY`), never mocked.
- Smallest reasonable change; no backward-compat shim for the old gaps-as-section shape.
- `Claim.sources` stays `min_length=1`; `test_claim_requires_at_least_one_source` is unchanged.
- Names/comments describe domain behavior, no temporal/historical narration ("new", "old", "was").
- Run `uv run python -m pytest -q` and keep output pristine (137 passed / 6 skipped is the current baseline).
- Detect-and-loop, not hard-reject: an uncited assertion drives `grounding_ok=False`, never a construction-time or immediate `GuardrailViolation`.

---

## File Structure

- `src/agent_pipeline/contracts/composition.py` — `Draft` gains `gaps`; `Section` comment updated. (Task 1)
- `src/agent_pipeline/agents/composer.py` — `CompositionPlan` gains `gaps`; both composers + `A3Composer.run` route gaps to `Draft.gaps`. (Task 1)
- `src/agent_pipeline/contracts/validation.py` — `BriefInput` gains `uncited_assertions`. (Task 2)
- `src/agent_pipeline/translators/draft_to_validation.py` — uncited sections → `uncited_assertions`; gaps rendered into `body`. (Task 2)
- `src/agent_pipeline/agents/validator.py` — `check()` folds `uncited_assertions` into `unsupported`. (Task 2)
- `DESIGN.md` — grounding-model note. (Task 2)
- Tests: `test_composition_contracts.py`, `test_composer.py` (Task 1); `test_validation_contracts.py`, `test_draft_to_validation.py`, `test_validator.py`, `test_reflection.py` (Task 2).

---

### Task 1: Gaps become a first-class `Draft.gaps` field

Gaps stop being an "Open questions" `Section` and become `Draft.gaps: list[str]`, carried through `CompositionPlan.gaps`. After this task, `Draft.gaps` is populated but not yet rendered into the shipped body (Task 2 does that) — a momentary gap the continuous execution closes immediately.

**Files:**

- Modify: `src/agent_pipeline/contracts/composition.py` (Draft ~line 35-40, Section ~line 25-32)
- Modify: `src/agent_pipeline/agents/composer.py` (CompositionPlan ~line 32-37, RuleBasedComposer ~line 46-69, LLMComposer `_SYSTEM` ~line 74-88, A3Composer.run ~line 138-150)
- Test: `tests/test_composition_contracts.py`, `tests/test_composer.py`

**Interfaces:**

- Produces: `Draft(request_id: str, sections: list[Section], gaps: list[str] = [], style_profile: str)`; `CompositionPlan(steps, sections, gaps: list[str] = [], style_profile)`. `RuleBasedComposer.compose` emits one section per point and sets `gaps = composer_input.gaps` (no "Open questions" section). `A3Composer.run` assembles `Draft(..., gaps=plan.gaps, ...)`.

- [ ] **Step 1: Update the gaps-only composer test to the new shape (failing test)**

Replace `test_a3_composes_gaps_only_input` in `tests/test_composer.py`:

```python
def test_a3_composes_gaps_only_input():
    gaps_only = ComposerInput(request_id="r1", points=[], gaps=["no data on bacteria"])
    draft = A3Composer(RuleBasedComposer()).run(gaps_only)
    assert draft.sections == []
    assert draft.gaps == ["no data on bacteria"]
```

Add, below it, a test that gaps ride the draft when points exist:

```python
def test_a3_carries_gaps_onto_the_draft():
    ci = ComposerInput(
        request_id="r1",
        points=[Point(statement="Cells make ATP", sources=["mito"], confidence=0.9)],
        gaps=["nothing on bacteria"],
    )
    draft = A3Composer(RuleBasedComposer()).run(ci)
    assert draft.gaps == ["nothing on bacteria"]
    assert [s.heading for s in draft.sections] == ["Point 1"]  # no "Open questions" section
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m pytest tests/test_composer.py::test_a3_composes_gaps_only_input tests/test_composer.py::test_a3_carries_gaps_onto_the_draft -v`
Expected: FAIL — `Draft` has no `gaps` field / `RuleBasedComposer` still emits an "Open questions" section (`AttributeError` or assertion mismatch).

- [ ] **Step 3: Add `gaps` to `Draft` and update the `Section` comment**

In `src/agent_pipeline/contracts/composition.py`, replace the `Section` and `Draft` classes:

```python
class Section(BaseModel):
    """A section of the draft: an assertion, with the source ids that support it."""

    heading: str = Field(min_length=1)
    body: str = Field(min_length=1)
    # source ids; empty marks an assertion with no grounding attempt, which A4 rejects
    # (fed back to A3 to recompose). Gaps that assert nothing live in Draft.gaps, not here.
    cited_sources: list[str] = []

class Draft(BaseModel):
    """A3's output: the composed sections, the acknowledged gaps, and the style."""

    request_id: str
    sections: list[Section]
    gaps: list[str] = []
    style_profile: str = Field(min_length=1)
```

- [ ] **Step 4: Add `gaps` to `CompositionPlan`, route gaps in both composers and `A3Composer.run`**

In `src/agent_pipeline/agents/composer.py`:

Add `gaps` to `CompositionPlan`:

```python
class CompositionPlan(BaseModel):
    """The A3 Model's output: the auditable step list plus the draft payload."""

    steps: list[PlanStep]
    sections: list[Section]
    gaps: list[str] = []
    style_profile: str
```

Replace `RuleBasedComposer`:

```python
class RuleBasedComposer:
    """Keyless stand-in: one section per point citing that point's sources; the input
    gaps carry through to the draft's gaps."""

    def compose(
        self, composer_input: ComposerInput, feedback: list[str] | None = None
    ) -> CompositionPlan:
        sections = [
            Section(
                heading=f"Point {i + 1}",
                body=point.statement,
                cited_sources=point.sources,
            )
            for i, point in enumerate(composer_input.points)
        ]
        return CompositionPlan(
            steps=[PlanStep(step_id=0, intent="emit the draft", tool="emit_contract")],
            sections=sections,
            gaps=composer_input.gaps,
            style_profile="outline",
        )
```

In `LLMComposer._SYSTEM`, replace the sentence

```bash
"cited points. Put any unanswered gaps in a final section that cites NO "
"sources (empty cited_sources) and lists the given gaps without adding "
"explanation. Pick a concise style_profile. You have no retrieval tools: "
```

with

```bash
"cited points. List any unanswered gaps in the gaps field as plain strings "
"(do not invent gaps); do not put them in a section. Every section must cite "
"the source ids it draws on. Pick a concise style_profile. You have no "
"retrieval tools: "
```

In `A3Composer.run`, pass `gaps` when assembling the draft:

```python
            if step.tool == "emit_contract":
                draft = Draft(
                    request_id=composer_input.request_id,
                    sections=plan.sections,
                    gaps=plan.gaps,
                    style_profile=plan.style_profile,
                )
                validate_composition_output(draft, available)  # guardrail
                return draft
```

- [ ] **Step 5: Run the two tests to verify they pass**

Run: `uv run python -m pytest tests/test_composer.py::test_a3_composes_gaps_only_input tests/test_composer.py::test_a3_carries_gaps_onto_the_draft -v`
Expected: PASS.

- [ ] **Step 6: Add a `Draft.gaps` contract test**

In `tests/test_composition_contracts.py`, add (after `test_draft_json_round_trip`):

```python
def test_draft_carries_gaps():
    draft = Draft(request_id="r1", sections=[], gaps=["no data on X"], style_profile="x")
    assert draft.gaps == ["no data on X"]
```

Run: `uv run python -m pytest tests/test_composition_contracts.py::test_draft_carries_gaps -v`
Expected: PASS.

- [ ] **Step 7: Run the full suite**

Run: `uv run python -m pytest -q`
Expected: PASS (137 passed / 6 skipped, or +1 for the added contract test). `test_draft_to_validation.py` still passes — it hand-builds drafts and is untouched by the composer change; its uncited "Open questions" section is still treated as body-not-claim by the unchanged translator.

- [ ] **Step 8: Commit**

```bash
git add src/agent_pipeline/contracts/composition.py src/agent_pipeline/agents/composer.py tests/test_composer.py tests/test_composition_contracts.py
git commit -m "feat: gaps become a first-class Draft.gaps field (#19)

Sections are now content assertions only; the composer routes gaps to
Draft.gaps (via CompositionPlan.gaps) instead of an uncited Open questions
section. Translator rendering of gaps into the body follows in the next step."
```

---

### Task 2: Ground the emitted body — uncited sections become `uncited_assertions`

The translator routes uncited content sections to a new `BriefInput.uncited_assertions`; `check()` folds them into `unsupported` so they fail grounding and feed the loop. Gaps render into the shipped body.

**Files:**

- Modify: `src/agent_pipeline/contracts/validation.py` (BriefInput ~line 20-26)
- Modify: `src/agent_pipeline/translators/draft_to_validation.py` (whole function)
- Modify: `src/agent_pipeline/agents/validator.py` (`check()` `check_claim` branch ~line 145-152)
- Modify: `DESIGN.md` (grounding note)
- Test: `tests/test_validation_contracts.py`, `tests/test_draft_to_validation.py`, `tests/test_validator.py`, `tests/test_reflection.py`

**Interfaces:**

- Consumes: `Draft.gaps` and the "sections are content assertions" model from Task 1.
- Produces: `BriefInput(request_id, claims, body, available_sources, uncited_assertions: list[str] = [])`. `translate_draft_to_validation` sets `uncited_assertions = [s.body for s in draft.sections if not s.cited_sources]` and renders `draft.gaps` into `body`. `A4Validator.check` computes `unsupported = [failing claim texts] + brief_input.uncited_assertions`.

- [ ] **Step 1: Headline failing test — an uncited assertion fails grounding**

In `tests/test_validator.py`, add:

```python
def test_a4_check_reports_uncited_assertion_as_unsupported():
    bi = BriefInput(
        request_id="r1",
        claims=[Claim(text="grounded claim", sources=["mito"])],
        body="Some body.",
        available_sources=["mito"],
        uncited_assertions=["a floating claim that cites nothing"],
    )
    outcome = A4Validator(StructuralClaimVerifier()).check(bi)
    assert outcome.brief.checks.grounding_ok is False
    assert "a floating claim that cites nothing" in outcome.unsupported

def test_a4_run_gates_an_uncited_assertion():
    bi = BriefInput(
        request_id="r1",
        claims=[Claim(text="grounded claim", sources=["mito"])],
        body="Some body.",
        available_sources=["mito"],
        uncited_assertions=["ungrounded floating claim"],
    )
    with pytest.raises(GuardrailViolation) as exc:
        A4Validator(StructuralClaimVerifier()).run(bi)
    assert exc.value.code == "GROUNDING_FAILED"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m pytest tests/test_validator.py::test_a4_check_reports_uncited_assertion_as_unsupported tests/test_validator.py::test_a4_run_gates_an_uncited_assertion -v`
Expected: FAIL — `BriefInput` has no `uncited_assertions` field (`ValidationError`/`TypeError`).

- [ ] **Step 3: Add `uncited_assertions` to `BriefInput`**

In `src/agent_pipeline/contracts/validation.py`, replace `BriefInput`:

```python
class BriefInput(BaseModel):
    """A4's input: the claims to verify, the assembled body, the source ids
    legitimately available to cite, and the texts of sections that cite nothing
    (assertions with no grounding attempt)."""

    request_id: str
    claims: list[Claim]
    body: str = Field(min_length=1)
    available_sources: list[str]
    uncited_assertions: list[str] = []
```

- [ ] **Step 4: Fold `uncited_assertions` into `unsupported` in `check()`**

In `src/agent_pipeline/agents/validator.py`, in the `check_claim` branch of `check()`, append the uncited assertions:

```python
            if step.tool == "check_claim":
                # A verifier may raise SOURCE_UNRESOLVED (infra fault) -- let it propagate.
                unsupported = [
                    claim.text
                    for claim in brief_input.claims
                    if not self._verifier.verify(claim, available)
                ]
                # Sections that cite nothing made no grounding attempt -- trivially unsupported.
                unsupported += brief_input.uncited_assertions
```

- [ ] **Step 5: Run the headline tests to verify they pass**

Run: `uv run python -m pytest tests/test_validator.py::test_a4_check_reports_uncited_assertion_as_unsupported tests/test_validator.py::test_a4_run_gates_an_uncited_assertion -v`
Expected: PASS.

- [ ] **Step 6: Translator failing tests — route uncited sections and render gaps**

Rewrite `tests/test_draft_to_validation.py` to the new model (gaps live in `Draft.gaps`; an uncited section is an ungrounded assertion). Replace the module docstring's second paragraph and the `_draft()` helper and the four behavior tests:

```python
"""Context Translation for the A3 -> A4 boundary.

Cited sections become claims to verify; sections that cite nothing become uncited
assertions (ungrounded, with no grounding attempt); the assembled section text plus
the acknowledged gaps become the brief body; cited source ids become the
available-sources set. Deterministic, Model-free.
"""
from agent_pipeline.contracts.composition import Draft, Section
from agent_pipeline.contracts.validation import BriefInput
from agent_pipeline.translators.draft_to_validation import translate_draft_to_validation

def _draft():
    return Draft(
        request_id="r1",
        sections=[
            Section(heading="Energy", body="Cells make ATP.", cited_sources=["mito"]),
            Section(heading="Plants", body="Plants photosynthesize.", cited_sources=["photo", "mito"]),
        ],
        gaps=["No bacteria data."],
        style_profile="concise",
    )

def test_translation_produces_valid_brief_input():
    out = translate_draft_to_validation(_draft())
    assert isinstance(out, BriefInput)
    assert out.request_id == "r1"

def test_cited_sections_become_claims():
    out = translate_draft_to_validation(_draft())
    assert [(c.text, c.sources) for c in out.claims] == [
        ("Cells make ATP.", ["mito"]),
        ("Plants photosynthesize.", ["photo", "mito"]),
    ]

def test_available_sources_are_deduped_in_first_seen_order():
    out = translate_draft_to_validation(_draft())
    assert out.available_sources == ["mito", "photo"]

def test_body_includes_every_section_and_the_gaps():
    out = translate_draft_to_validation(_draft())
    assert "Cells make ATP." in out.body
    assert "Plants photosynthesize." in out.body
    assert "No bacteria data." in out.body  # gaps render into the body

def test_gaps_are_neither_claims_nor_uncited_assertions():
    out = translate_draft_to_validation(_draft())
    assert len(out.claims) == 2
    assert out.uncited_assertions == []

def test_uncited_content_section_becomes_an_uncited_assertion():
    draft = Draft(
        request_id="r1",
        sections=[
            Section(heading="Grounded", body="Cells make ATP.", cited_sources=["mito"]),
            Section(heading="Floating", body="Cells also feel joy.", cited_sources=[]),
        ],
        style_profile="concise",
    )
    out = translate_draft_to_validation(draft)
    assert [(c.text, c.sources) for c in out.claims] == [("Cells make ATP.", ["mito"])]
    assert out.uncited_assertions == ["Cells also feel joy."]
    assert "Cells also feel joy." in out.body  # still ships in the body

def test_gaps_only_draft_yields_no_claims_or_assertions_but_a_body():
    draft = Draft(request_id="r1", sections=[], gaps=["No data at all."], style_profile="concise")
    out = translate_draft_to_validation(draft)
    assert out.claims == []
    assert out.uncited_assertions == []
    assert out.available_sources == []
    assert "No data at all." in out.body
```

- [ ] **Step 7: Run to verify failure**

Run: `uv run python -m pytest tests/test_draft_to_validation.py -v`
Expected: FAIL — translator does not populate `uncited_assertions` and does not render `draft.gaps` into the body.

- [ ] **Step 8: Implement the translator change**

Replace the body of `translate_draft_to_validation` in `src/agent_pipeline/translators/draft_to_validation.py`:

```python
def translate_draft_to_validation(draft: Draft) -> BriefInput:
    claims = [
        Claim(text=section.body, sources=section.cited_sources)
        for section in draft.sections
        if section.cited_sources
    ]
    uncited_assertions = [
        section.body for section in draft.sections if not section.cited_sources
    ]

    available: list[str] = []
    seen: set[str] = set()
    for section in draft.sections:
        for source_id in section.cited_sources:
            if source_id not in seen:
                seen.add(source_id)
                available.append(source_id)

    parts = [f"## {section.heading}\n{section.body}" for section in draft.sections]
    if draft.gaps:
        parts.append("## Open questions\n" + "\n".join(f"- {gap}" for gap in draft.gaps))
    body = "\n\n".join(parts)

    return BriefInput(
        request_id=draft.request_id,
        claims=claims,
        body=body,
        available_sources=available,
        uncited_assertions=uncited_assertions,
    )
```

Also update the module docstring at the top of `draft_to_validation.py` (which still says uncited sections are "intro/gaps") to match:

```python
"""Context Translation for the A3 -> A4 boundary.

Maps the composition vocabulary (draft of sections + gaps) into the validation
vocabulary. A cited section becomes a claim; a section that cites nothing becomes an
uncited assertion (ungrounded -- no grounding attempt). The section text and the gaps
assemble into the body; cited source ids become the available-sources set.
Deterministic and Model-free.
"""
```

- [ ] **Step 9: Run the translator tests**

Run: `uv run python -m pytest tests/test_draft_to_validation.py -v`
Expected: PASS (7 tests).

- [ ] **Step 10: Graph test — an uncited section drives the loop to exhaustion**

In `tests/test_reflection.py`, add imports at the top (alongside the existing composer/config imports):

```python
from agent_pipeline.agents.validator import A4Validator, StructuralClaimVerifier
from agent_pipeline.config import MAX_COMPOSE_ATTEMPTS
```

(`StructuralClaimVerifier` and `MAX_COMPOSE_ATTEMPTS` are the additions; `A4Validator` is already imported — merge, don't duplicate. `CompositionPlan`, `PlanStep`, `Section` are already imported.)

Add the test:

```python
class _UncitedSectionComposer:
    """Emits one content section that cites nothing, every time -- it never grounds."""

    def __init__(self):
        self.calls = 0

    def compose(self, composer_input, feedback=None):
        self.calls += 1
        return CompositionPlan(
            steps=[PlanStep(step_id=0, intent="emit", tool="emit_contract")],
            sections=[Section(heading="Floating", body="Unbacked assertion.", cited_sources=[])],
            style_profile="plain",
        )

def test_uncited_section_drives_the_loop_to_exhaustion():
    composer = _UncitedSectionComposer()
    app = _app(_one_doc_store(), A3Composer(composer), StructuralClaimVerifier())
    request = RetrievalRequest(request_id="r1", raw_query="how do cells make energy?")
    with pytest.raises(GuardrailViolation) as exc:
        app.invoke(_initial(request), {"configurable": {"thread_id": "r1"}})
    assert exc.value.code == "GROUNDING_FAILED"
    assert composer.calls == MAX_COMPOSE_ATTEMPTS  # an uncited section can never ground
```

- [ ] **Step 11: Run to verify it passes**

Run: `uv run python -m pytest tests/test_reflection.py::test_uncited_section_drives_the_loop_to_exhaustion -v`
Expected: PASS. (The composer emits a single uncited section; `validate_composition_output` allows it since it cites nothing unavailable and the draft is non-empty; the translator makes it an uncited assertion; `check()` reports `grounding_ok=False`; the loop recomposes to `MAX_COMPOSE_ATTEMPTS` then the gate raises `GROUNDING_FAILED`.)

- [ ] **Step 12: Add the `BriefInput.uncited_assertions` contract test**

In `tests/test_validation_contracts.py`, add (after `test_brief_input_carries_claims_body_and_available_sources`):

```python
def test_brief_input_carries_uncited_assertions():
    bi = BriefInput(
        request_id="r1",
        claims=[Claim(text="cells make ATP", sources=["mito"])],
        body="Cells make ATP.",
        available_sources=["mito"],
        uncited_assertions=["a section that cites nothing"],
    )
    assert bi.uncited_assertions == ["a section that cites nothing"]
```

Run: `uv run python -m pytest tests/test_validation_contracts.py -v`
Expected: PASS (existing `test_claim_requires_at_least_one_source` still passes — `Claim` is unchanged).

- [ ] **Step 13: Update DESIGN.md with the grounding-model note**

In `DESIGN.md`, find the A4 Validator description and add this note where it explains grounding (keep the surrounding prose; add these two sentences):

```markdown
Grounding covers the emitted body, not just the cited claims: a content section that
cites nothing becomes an _uncited assertion_ (`BriefInput.uncited_assertions`) that
counts as unsupported, so it fails grounding and feeds the A3⇄A4 loop. Acknowledged
gaps are not assertions -- they live in `Draft.gaps`, render into the body, and never
block grounding (#19).
```

- [ ] **Step 14: Run the full suite**

Run: `uv run python -m pytest -q`
Expected: PASS, pristine (baseline 137 + the tests added across both tasks; 6 skipped). If any pre-existing test fails, fix the root cause — do not weaken it.

- [ ] **Step 15: Commit**

```bash
git add src/agent_pipeline/contracts/validation.py src/agent_pipeline/translators/draft_to_validation.py src/agent_pipeline/agents/validator.py DESIGN.md tests/test_validation_contracts.py tests/test_draft_to_validation.py tests/test_validator.py tests/test_reflection.py
git commit -m "feat: A4 grounds the emitted body via uncited_assertions (#19)

Uncited content sections travel to BriefInput.uncited_assertions and fold into
the unsupported set, so ungrounded prose fails grounding and feeds the A3<->A4
loop instead of shipping unverified. Gaps render into the body from Draft.gaps.
Claim.sources keeps its min_length=1 invariant."
```

---

## Self-Review

**Spec coverage:** Draft.gaps + CompositionPlan.gaps + composer routing (Task 1); BriefInput.uncited_assertions + translator + check() fold + body rendering + DESIGN note (Task 2); headline, translator, check, composer, and graph/reflection tests all present; non-goal (gap-stuffing) left untouched as intended. Covered.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every test step shows the assertion; exact commands with expected results. Clean.

**Type consistency:** `Draft.gaps`/`CompositionPlan.gaps`/`BriefInput.uncited_assertions` are `list[str]` with `= []` defaults everywhere; `translate_draft_to_validation` returns `BriefInput` with all five fields; `check()` reads `brief_input.uncited_assertions`; `A3Composer.run` passes `gaps=plan.gaps`. Names match across tasks.
