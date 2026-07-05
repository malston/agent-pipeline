# Harden the gaps channel (fix #22) — Design

**Date:** 2026-07-05
**Status:** Approved (design)
**Issue:** [#22](https://github.com/malston/agent-pipeline/issues/22) — verify `Draft.gaps` against upstream gaps to close gap-stuffing

## Problem

After #19, A4 grounds every content section, but `Draft.gaps` render into the shipped
body and are never grounding-checked (gaps are acknowledgments, not assertions). An
unfaithful `LLMComposer` could smuggle a fabricated factual assertion in as a "gap" and
it would ship unverified — the same shape of silent escape #19 closed, narrowed to the
gaps channel. Flagged as an explicit non-goal in the #19 spec and re-raised by two PR
#20 reviewers.

The legitimate gaps originate at `AnalysisReport.gaps` (A2's evidence assessment) and
reach A3 as `ComposerInput.gaps` unchanged (`translate_analysis_to_composition` copies
them). A3 should only _echo_ those, never invent new ones.

## Decision

A gap A3 emits that A2 never reported is a **fabricated gap**, and a fabricated gap is
just unbacked body text — the exact thing #19's `BriefInput.uncited_assertions` already
models. So a fabricated gap is folded into `uncited_assertions` and flows through the
existing #19 mechanism: `unsupported` → `grounding_ok=False` → the A3⇄A4 reflection loop
recomposes (the composer drops the invented gap) → terminal gate. Full reuse: no new A4
field, no new loop path, no new error code, detect-and-loop (not hard-reject).

The reference set is `AnalysisReport.gaps`, already present in `PipelineState["analysis"]`.

## The change

### Deterministic, Model-free check

`fabricated = [g for g in draft.gaps if g not in set(legitimate_gaps)]`. For this exact
subset check to be Model-free (no semantic-similarity call), the LLM composer must echo
gaps verbatim.

- `LLMComposer._SYSTEM` (`agents/composer.py`): tighten the gaps instruction from "do
  not invent gaps" to "copy each given gap verbatim into the gaps field — do not
  rephrase, merge, or add." A rephrasing composer gets looped like any faithlessness.
- `RuleBasedComposer` already sets `gaps = composer_input.gaps` verbatim, so its subset
  check always passes; the keyless path is unaffected.

### Translator (`translators/draft_to_validation.py`)

`translate_draft_to_validation(draft, legitimate_gaps)` gains a required `legitimate_gaps`
parameter. Fabricated gaps join `uncited_assertions`:

```python
uncited_assertions = [
    section.body for section in draft.sections if not section.cited_sources
] + [gap for gap in draft.gaps if gap not in set(legitimate_gaps)]
```

The body still renders all `draft.gaps` (a fabricated one appears in the body _and_ in
`uncited_assertions`, exactly like an uncited content section). `available_sources` and
`claims` are unchanged. Stays deterministic and Model-free.

### Graph (`graph/pipeline.py`)

`validator_node` passes the reference:

```python
analysis = state["analysis"]
if analysis is None:
    raise ValueError("validator_node reached with no analysis; A2 did not populate state['analysis']")
outcome = validator.check(translate_draft_to_validation(draft, analysis.gaps))
```

## Error handling

Fabricated gaps drive the existing loop: recompose, then ground. If a composer never
stops fabricating, the gate raises `GROUNDING_FAILED` at `MAX_COMPOSE_ATTEMPTS` — no new
code. The existing feedback message ("statements NOT supported by their cited sources;
drop or rephrase") reads slightly generically for a gap but produces the right action
(the composer drops the invented gap); refining it is out of scope.

## Invariants preserved

- The legitimate no-evidence path still ships grounded: A2 emits a real gap, A3 echoes
  it verbatim, `draft.gaps ⊆ legitimate_gaps`, nothing is fabricated, `uncited_assertions`
  gains nothing from the gaps.
- `Claim.sources` `min_length=1`, the terminal gate, and the `ValidationOutcome`
  `unsupported ⇔ grounding_ok` invariant are all unaffected (this only adds entries to
  `uncited_assertions`, which already fold into `unsupported`).
- The PR #20 non-empty constraint on `uncited_assertions` elements holds — gaps are
  non-empty strings; the translator folds only non-empty fabricated gaps.

## Testing (TDD)

1. **Translator:** a `draft.gaps` entry absent from `legitimate_gaps` lands in
   `uncited_assertions`; a legitimate one does not; `claims`/body unchanged.
2. **Graph:** a composer emitting a fabricated gap drives the loop to `GROUNDING_FAILED`
   at `MAX_COMPOSE_ATTEMPTS`; a composer echoing only legitimate gaps ships grounded with
   `composer.calls == 1`.
3. Existing translator callers (`validator_node`, direct unit tests) updated for the new
   required parameter.

Test output stays pristine. The gated LLM e2e exercises the verbatim-echo prompt only
with a provider key; never mocked.

## Non-goals

- **Semantic-tolerance matching** (accepting a rephrased gap) — out of scope; the
  verbatim-echo prompt makes exact subset match the contract. A rephrasing LLM is looped
  like any other faithfulness miss.
- No new error code, no A4-side contract field, no change to the loop routing or
  `MAX_COMPOSE_ATTEMPTS`.
- Refining the loop feedback wording for gaps specifically.

## Files touched

- `src/agent_pipeline/agents/composer.py` — `LLMComposer._SYSTEM` verbatim-echo tightening
- `src/agent_pipeline/translators/draft_to_validation.py` — `legitimate_gaps` parameter;
  fabricated gaps fold into `uncited_assertions`
- `src/agent_pipeline/graph/pipeline.py` — `validator_node` passes `state["analysis"].gaps`
- Tests: `test_draft_to_validation.py`, `test_reflection.py`
- Docs: `DESIGN.md` note; close `#22`
