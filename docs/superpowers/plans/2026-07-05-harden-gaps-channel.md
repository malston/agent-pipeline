# Harden the Gaps Channel (#22) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A gap A3 emits that A2 never reported (a fabricated gap) is treated as unbacked body text — folded into `uncited_assertions` so it fails grounding and feeds the A3⇄A4 loop, closing gap-stuffing.

**Architecture:** Reuse the #19 mechanism. `translate_draft_to_validation` gains a required `legitimate_gaps` parameter (the reference is `AnalysisReport.gaps`, already in `PipelineState["analysis"]`); any `draft.gaps` entry not in that set joins `uncited_assertions`. The `LLMComposer` prompt is tightened to echo gaps verbatim so the subset check stays exact and Model-free. No new A4 field, error code, or loop path.

**Tech Stack:** Python 3.12+, Pydantic v2, LangGraph, pytest, `uv`.

## Global Constraints

- TDD: failing test first, watch it fail, minimal code, watch it pass, commit. Never delete/weaken a test to pass.
- No mocks; keyless deterministic tests use real `fastembed` + Protocol test doubles. The LLM path is exercised only by key-gated e2e tests, never mocked.
- Reuse `uncited_assertions` — no new A4 contract field, no new error code, no change to loop routing or `MAX_COMPOSE_ATTEMPTS`.
- `legitimate_gaps` is a **required** parameter (no default) — a defaulted/None value would silently skip the gap check, which the project's fail-loud rule forbids.
- The reference set is `AnalysisReport.gaps`.
- `Claim.sources` stays `min_length=1`; `uncited_assertions` elements stay non-empty (PR #20) — the translator folds only non-empty fabricated gaps (gaps are non-empty strings anyway).
- Smallest reasonable change; no backward-compat shim. Names/comments describe domain behavior, no temporal/historical narration.
- Run `uv run python -m pytest -q` and keep output pristine (baseline on this branch: 148 passed / 6 skipped).

---

## File Structure

- `src/agent_pipeline/translators/draft_to_validation.py` — `legitimate_gaps` param; fabricated gaps fold into `uncited_assertions`. (Task 1)
- `src/agent_pipeline/graph/pipeline.py` — `validator_node` passes `state["analysis"].gaps`. (Task 1)
- `src/agent_pipeline/agents/composer.py` — `LLMComposer._SYSTEM` verbatim-echo tightening. (Task 2)
- `DESIGN.md` — grounding-model note extended for the gaps channel. (Task 2)
- Tests: `tests/test_draft_to_validation.py` (Task 1); `tests/test_reflection.py` (Task 2).

Direct callers of `translate_draft_to_validation` (must be updated for the new required param): `validator_node` (graph/pipeline.py) and `tests/test_draft_to_validation.py`. Confirm with `grep -rn "translate_draft_to_validation" src tests` before finishing Task 1 — there should be no others.

---

### Task 1: Translator folds fabricated gaps into `uncited_assertions`

`translate_draft_to_validation` gains a required `legitimate_gaps` parameter; a `draft.gaps` entry absent from it becomes an uncited assertion. `validator_node` passes `state["analysis"].gaps`. Both the signature change and the caller update land together so the suite stays green.

**Files:**

- Modify: `src/agent_pipeline/translators/draft_to_validation.py`
- Modify: `src/agent_pipeline/graph/pipeline.py` (`validator_node`, ~lines 71-79)
- Test: `tests/test_draft_to_validation.py`

**Interfaces:**

- Produces: `translate_draft_to_validation(draft: Draft, legitimate_gaps: list[str]) -> BriefInput`. `uncited_assertions` = (uncited section bodies) + (`draft.gaps` entries not in `legitimate_gaps`). `validator_node` calls it with `state["analysis"].gaps`.

- [ ] **Step 1: Update existing translator tests for the new required param (still-legit behavior)**

In `tests/test_draft_to_validation.py`, add a shared constant and thread it through every existing call so the drafts' gaps count as legitimate (behavior unchanged). Replace the top of the file (imports through `_draft`) and each existing call:

```python
"""Context Translation for the A3 -> A4 boundary.

Cited sections become claims to verify; sections that cite nothing -- and gaps A2
never reported -- become uncited assertions (ungrounded, no grounding attempt); the
section text plus the acknowledged gaps become the brief body; cited source ids become
the available-sources set. Deterministic, Model-free.
"""
from agent_pipeline.contracts.composition import Draft, Section
from agent_pipeline.contracts.validation import BriefInput
from agent_pipeline.translators.draft_to_validation import translate_draft_to_validation

_LEGIT_GAPS = ["No bacteria data."]

def _draft():
    return Draft(
        request_id="r1",
        sections=[
            Section(heading="Energy", body="Cells make ATP.", cited_sources=["mito"]),
            Section(heading="Plants", body="Plants photosynthesize.", cited_sources=["photo", "mito"]),
        ],
        gaps=_LEGIT_GAPS,
        style_profile="concise",
    )
```

Then update each existing call to pass the matching legitimate set:

- `test_translation_produces_valid_brief_input`, `test_cited_sections_become_claims`, `test_available_sources_are_deduped_in_first_seen_order`, `test_body_includes_every_section_and_the_gaps`, `test_gaps_are_neither_claims_nor_uncited_assertions` — change `translate_draft_to_validation(_draft())` to `translate_draft_to_validation(_draft(), _LEGIT_GAPS)`.
- `test_uncited_content_section_becomes_an_uncited_assertion` (its draft has no gaps) — change `translate_draft_to_validation(draft)` to `translate_draft_to_validation(draft, [])`.
- `test_gaps_only_draft_yields_no_claims_or_assertions_but_a_body` (gaps `["No data at all."]`) — change `translate_draft_to_validation(draft)` to `translate_draft_to_validation(draft, ["No data at all."])`.

- [ ] **Step 2: Add the failing fabricated-gap test**

Append to `tests/test_draft_to_validation.py`:

```python
def test_fabricated_gap_becomes_an_uncited_assertion():
    # a gap A2 never reported is unbacked body text -> an uncited assertion; a legitimate
    # gap (in legitimate_gaps) is not.
    draft = Draft(
        request_id="r1",
        sections=[Section(heading="Energy", body="Cells make ATP.", cited_sources=["mito"])],
        gaps=["No bacteria data.", "Mitochondria secretly plot."],
        style_profile="concise",
    )
    out = translate_draft_to_validation(draft, legitimate_gaps=["No bacteria data."])
    assert out.uncited_assertions == ["Mitochondria secretly plot."]  # only the fabricated one
    assert [(c.text, c.sources) for c in out.claims] == [("Cells make ATP.", ["mito"])]
    assert "Mitochondria secretly plot." in out.body  # still ships in the body
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run python -m pytest tests/test_draft_to_validation.py -v`
Expected: the existing calls FAIL with `TypeError` (missing required positional arg `legitimate_gaps`) and/or the new test fails — the function does not yet accept `legitimate_gaps`.

- [ ] **Step 4: Implement the translator change**

In `src/agent_pipeline/translators/draft_to_validation.py`, update the module docstring's second paragraph and the function:

```python
"""Context Translation for the A3 -> A4 boundary.

Maps the composition vocabulary (draft of sections + gaps) into the validation
vocabulary. A cited section becomes a claim; a section that cites nothing -- and a gap
A2 never reported -- becomes an uncited assertion (ungrounded, no grounding attempt).
The section text and the gaps assemble into the body; cited source ids become the
available-sources set. Deterministic and Model-free.
"""
from agent_pipeline.contracts.composition import Draft
from agent_pipeline.contracts.validation import BriefInput, Claim

def translate_draft_to_validation(draft: Draft, legitimate_gaps: list[str]) -> BriefInput:
    claims = [
        Claim(text=section.body, sources=section.cited_sources)
        for section in draft.sections
        if section.cited_sources
    ]
    known_gaps = set(legitimate_gaps)
    uncited_assertions = [
        section.body for section in draft.sections if not section.cited_sources
    ] + [gap for gap in draft.gaps if gap not in known_gaps]

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

- [ ] **Step 5: Update `validator_node` to pass the reference**

In `src/agent_pipeline/graph/pipeline.py`, replace `validator_node`:

```python
    def validator_node(state: PipelineState) -> dict:
        draft = state["draft"]
        if draft is None:
            raise ValueError(
                "validator_node reached with no draft; "
                "A3 did not populate state['draft']"
            )
        analysis = state["analysis"]
        if analysis is None:
            raise ValueError(
                "validator_node reached with no analysis; "
                "A2 did not populate state['analysis']"
            )
        outcome = validator.check(translate_draft_to_validation(draft, analysis.gaps))
        return {"brief": outcome.brief, "feedback": outcome.unsupported}
```

- [ ] **Step 6: Run the translator tests, then the full suite**

Run: `uv run python -m pytest tests/test_draft_to_validation.py -v`
Expected: PASS (all updated calls + the new fabricated-gap test).

Run: `uv run python -m pytest -q`
Expected: PASS, pristine. The pre-existing graph tests still pass — `RuleBasedComposer` echoes `analysis.gaps` verbatim, so `draft.gaps == analysis.gaps ⊆ legitimate_gaps` and no gap is fabricated (in particular `test_gaps_only_draft_ships_grounded_without_looping` still ships grounded, now also exercising the legit-echo path through the subset check).

- [ ] **Step 7: Confirm no other callers, then commit**

Run: `grep -rn "translate_draft_to_validation" src tests`
Expected: only `translators/draft_to_validation.py` (definition), `graph/pipeline.py` (updated), and `tests/test_draft_to_validation.py` (updated). If any other caller exists, update it to pass its legitimate gaps and re-run the suite.

```bash
git add src/agent_pipeline/translators/draft_to_validation.py src/agent_pipeline/graph/pipeline.py tests/test_draft_to_validation.py
git commit -m "feat: fold fabricated gaps into uncited_assertions (#22)

translate_draft_to_validation gains a required legitimate_gaps parameter; a
draft.gaps entry A2 never reported becomes an uncited assertion and fails
grounding via the existing #19 mechanism. validator_node passes the reference
from state['analysis'].gaps."
```

---

### Task 2: Verbatim-echo prompt + graph loop test + DESIGN note

Tighten the LLM composer to copy gaps verbatim (so the subset check is exact), prove a fabricated gap drives the loop end-to-end, and document the hardening.

**Files:**

- Modify: `src/agent_pipeline/agents/composer.py` (`LLMComposer._SYSTEM`)
- Modify: `DESIGN.md`
- Test: `tests/test_reflection.py`

**Interfaces:**

- Consumes: the Task 1 translator/`validator_node` wiring (fabricated gaps → `uncited_assertions` → loop).

- [ ] **Step 1: Add the failing graph test**

In `tests/test_reflection.py`, append (the imports `CompositionPlan`, `PlanStep`, `Section`, `StructuralClaimVerifier`, `MAX_COMPOSE_ATTEMPTS`, and the `_one_doc_store`/`_app`/`_initial` helpers already exist in this file):

```python
class _FabricatedGapComposer:
    """Emits a grounded section but also invents a gap A2 never reported, every time."""

    def __init__(self):
        self.calls = 0

    def compose(self, composer_input, feedback=None):
        self.calls += 1
        sections = [
            Section(heading="Point 1", body=p.statement, cited_sources=p.sources)
            for p in composer_input.points
        ]
        return CompositionPlan(
            steps=[PlanStep(step_id=0, intent="emit", tool="emit_contract")],
            sections=sections,
            gaps=[*composer_input.gaps, "Cells secretly feel joy."],
            style_profile="plain",
        )

def test_fabricated_gap_drives_the_loop_to_exhaustion():
    # the section is grounded, but the invented gap is unbacked body text -> it fails
    # grounding and drives the loop to exhaustion, then the gate raises.
    composer = _FabricatedGapComposer()
    app = _app(_one_doc_store(), A3Composer(composer), StructuralClaimVerifier())
    request = RetrievalRequest(request_id="r1", raw_query="how do cells make energy?")
    with pytest.raises(GuardrailViolation) as exc:
        app.invoke(_initial(request), {"configurable": {"thread_id": "r1"}})
    assert exc.value.code == "GROUNDING_FAILED"
    assert composer.calls == MAX_COMPOSE_ATTEMPTS  # the invented gap never grounds
```

- [ ] **Step 2: Run to verify it passes**

Run: `uv run python -m pytest tests/test_reflection.py::test_fabricated_gap_drives_the_loop_to_exhaustion -v`
Expected: PASS. (`_one_doc_store` + the query yields one point, so the composer emits a grounded `mito` section — the only grounding failure is the invented gap, which `validator_node` now detects against `analysis.gaps` and folds into `uncited_assertions`; the loop recomposes to `MAX_COMPOSE_ATTEMPTS`, then `gate_node` raises `GROUNDING_FAILED`.)

This test relies on Task 1's wiring; if it does not pass, Task 1 is incomplete — do not weaken the test.

- [ ] **Step 3: Tighten the `LLMComposer` prompt to echo gaps verbatim**

In `src/agent_pipeline/agents/composer.py`, in `LLMComposer._SYSTEM`, replace the gaps sentence. Change:

```python
        "cited points. List any unanswered gaps in the gaps field as plain strings "
        "(do not invent gaps); do not put them in a section. Every section must cite "
        "the source ids it draws on. Pick a concise style_profile. You have no "
```

to:

```python
        "cited points. Copy each given gap verbatim into the gaps field -- do not "
        "rephrase, merge, or invent gaps, and do not put them in a section. Every "
        "section must cite the source ids it draws on. Pick a concise style_profile. "
        "You have no "
```

- [ ] **Step 4: Run the composer tests**

Run: `uv run python -m pytest tests/test_composer.py -q`
Expected: PASS. (The prompt is a system-message string; the keyless `RuleBasedComposer` tests are unaffected, and the `_CapturingModel` feedback tests assert the human message, not the system prompt.)

- [ ] **Step 5: Extend the DESIGN.md grounding note**

In `DESIGN.md`, in the reflection-loop paragraph (section 2, the note ending "...do not block grounding (#19)."), append one sentence:

```markdown
A gap A3 emits that A2 never reported is likewise treated as an uncited assertion and
fails grounding, so the composer cannot smuggle unverified prose through the gaps
channel (#22).
```

- [ ] **Step 6: Run the full suite**

Run: `uv run python -m pytest -q`
Expected: PASS, pristine (baseline 148 + the tests added across both tasks; 6 skipped).

- [ ] **Step 7: Commit**

```bash
git add src/agent_pipeline/agents/composer.py DESIGN.md tests/test_reflection.py
git commit -m "feat: LLMComposer echoes gaps verbatim; loop-closes gap-stuffing (#22)

The LLM composer must copy each given gap verbatim so the fabricated-gap subset
check stays exact and Model-free. A graph test proves an invented gap drives the
A3<->A4 loop to GROUNDING_FAILED at exhaustion. DESIGN note extended."
```

---

## Self-Review

**Spec coverage:** `legitimate_gaps` param + fabricated-gap fold (Task 1 Step 4); `validator_node` passes `analysis.gaps` (Task 1 Step 5); verbatim-echo prompt (Task 2 Step 3); translator unit test + graph loop test (Task 1 Step 2, Task 2 Step 1); DESIGN note (Task 2 Step 5); non-goal (semantic tolerance) left untouched. Covered.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every test step shows assertions and expected results; the caller-audit grep is explicit. Clean.

**Type consistency:** `translate_draft_to_validation(draft, legitimate_gaps)` signature matches every updated call (validator_node passes `analysis.gaps: list[str]`; tests pass `list[str]`); `uncited_assertions` stays `list[str]`; `AnalysisReport.gaps` is `list[str]`. `_FabricatedGapComposer.compose` returns a `CompositionPlan` with the fields Task 1 (#19) established (`steps`, `sections`, `gaps`, `style_profile`). Consistent.
