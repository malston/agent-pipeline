# Ground the emitted body (fix #19) — Design

**Date:** 2026-07-05
**Status:** Approved (design)
**Issue:** [#19](https://github.com/malston/agent-pipeline/issues/19) — A4 grounds extracted claims, not the emitted body

## Problem

A4's grounding verdict is computed over the **claims extracted from cited sections**,
not over the **body that actually ships**. `translate_draft_to_validation` builds a
claim only from a section that cites sources (`if section.cited_sources`), but the
brief `body` concatenates _every_ section. So a section with body text and empty
`cited_sources` ships in the brief yet produces no claim, is never verified, and
cannot appear in `unsupported` — so `grounding_ok` reports `True` for prose that was
never grounding-checked.

`RuleBasedComposer` is safe (its only uncited section is the gap block, which asserts
nothing). The gap is on the `LLMComposer` path: an unfaithful model can place a
fabricated assertion in an uncited section, invisible to the critic and therefore to
the reflection loop that exists to catch exactly that faithlessness.

Pre-existing (single-shot A4 had the same gap); surfaced by the PR #17 whole-branch
review and deferred to this design because closing it changes what "grounded" means.

## Decisions

Two forks, both settled during brainstorming:

1. **Detect → loop, not hard-reject.** When A4 finds an uncited content section, its
   text becomes an unsupported claim so the reflection loop recomposes (cite it or
   drop it), consistent with #17 — grounding failures are recoverable. Not an
   immediate `GuardrailViolation`.
2. **Model gaps as a first-class `Draft.gaps: list[str]`, not a section.** Every
   `Section` becomes a grounded assertion; gaps are strings rendered into the body but
   never claimed. This gives the clean invariant "every Section is a grounded
   assertion" and mirrors `ComposerInput`, which already separates `points`/`gaps`.

## The change

### Contracts (`contracts/composition.py`)

- `Draft` gains `gaps: list[str] = []`. `sections` are content assertions only.
- `Section.cited_sources` stays optional. An empty one is **permitted by the type but
  judged ungrounded by A4** — making it `min_length=1` would hard-fail construction
  (e.g. reject the LLM's structured output outright) instead of feeding back to the
  loop, contradicting decision 1. Update its comment: empty `cited_sources` marks an
  ungrounded assertion the A4 gate rejects (fed back to A3 to recompose); gaps that
  assert nothing live in `Draft.gaps`, not sections.

### Composer output (`contracts` + `agents/composer.py`)

- `CompositionPlan` (A3's Model output) gains `gaps: list[str] = []`.
- `RuleBasedComposer`: drop the `"Open questions"` section; set
  `gaps = composer_input.gaps`. One section per point, unchanged.
- `LLMComposer` system prompt: replace "put unanswered gaps in a final section that
  cites NO sources" with "list unanswered gaps in the `gaps` field as plain strings
  (do not invent gaps); every section must cite the source ids it draws on."
- `A3Composer.run`: assemble `Draft(sections=plan.sections, gaps=plan.gaps,
style_profile=plan.style_profile)`.

### The fix (`translators/draft_to_validation.py` + `agents/validator.py`)

- `translate_draft_to_validation`: extract a claim from **every** section (remove the
  `if section.cited_sources` filter). Render `draft.gaps` into the shipped `body`
  under an "Open questions" block when non-empty, so the brief still surfaces gaps
  (presentation only). `available_sources` (union of cited sources) unchanged.
- Both verifiers: **a claim that cites no sources is ungrounded.**
  - `StructuralClaimVerifier.verify`: `return bool(claim.sources) and set(claim.sources) <= available_sources`.
  - `LLMClaimVerifier.verify`: `if not claim.sources: return False` before the subset
    check (an empty set is a subset of anything, so today it passes).

### Data flow after the change

An uncited content section → `Claim(text=body, sources=[])` → `verify` returns False
→ text lands in `unsupported` → `grounding_ok=False` → the loop feeds the text back to
A3 → recompose (cite or drop) → terminal gate. Gaps are never claims, so they ship in
the body without blocking grounding.

## Invariants preserved

- `test_a4_treats_a_claimless_brief_as_grounded`: a brief with no claims stays
  vacuously grounded (a gaps-only draft yields `sections==[]` → `claims==[]`).
- Analyst-found-nothing path (`findings==[]`, `gaps==[...]`): produces a gaps-only
  draft that ships grounded, as today.
- `EMPTY_DRAFT` (`validate_composition_output`): points present ⇒ sections must exist;
  unchanged (gaps do not satisfy the "composed something" obligation).
- Terminal gate and the `ValidationOutcome` `unsupported ⇔ grounding_ok` invariant:
  unaffected.

## Testing (TDD)

Failing test first, then minimal code, per task:

1. **Headline:** a `Draft` with one grounded cited section plus one uncited content
   section carrying a fabricated claim must **not** pass A4 — `grounding_ok=False` and
   the uncited section's text in `unsupported`.
2. **Verifiers:** `StructuralClaimVerifier` and `LLMClaimVerifier` each report a
   claim with empty `sources` as ungrounded (the LLM one without calling the model).
3. **Translator:** every section becomes a claim (cited or not); `gaps` render into
   `body` but produce no claim.
4. **Composer:** `RuleBasedComposer` puts gaps in `Draft.gaps` (updated gaps-only test:
   `sections==[]`, `gaps==["..."]`); empty input still yields an empty draft.
5. **Graph/reflection:** an uncited content section drives the loop and raises
   `GROUNDING_FAILED` at exhaustion (composer runs `MAX_COMPOSE_ATTEMPTS` times).

Test output stays pristine. The gated LLM e2e (`test_a3_e2e`, key-guarded) exercises
the new prompt/`gaps` field only with a provider key; never mocked.

## Non-goals

- **Gap-stuffing.** An LLM smuggling a fabricated assertion into `Draft.gaps` (which
  ships in body unverified) stays out of scope — a narrower threat than the
  content-elaboration case this closes. Hardening would verify `draft.gaps` against
  the known input gaps (they originate upstream in `AnalysisReport`), a separate
  follow-up.
- No change to policy/format checks, the loop's routing, or `MAX_COMPOSE_ATTEMPTS`.
- No backward-compatibility shim for the old gaps-as-section shape.

## Files touched

- `src/agent_pipeline/contracts/composition.py` — `Draft.gaps`, `Section` comment
- `src/agent_pipeline/agents/composer.py` — `CompositionPlan.gaps`, both composers,
  `A3Composer.run`
- `src/agent_pipeline/translators/draft_to_validation.py` — claims from every section,
  gaps into body
- `src/agent_pipeline/agents/validator.py` — empty-sources rule in both verifiers
- Tests: `test_validator.py`, `test_composer.py`, `test_reflection.py`, and a
  translator test
- Docs: `DESIGN.md` grounding note; close `#19`
