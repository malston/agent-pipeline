# Reflection loop: A3 Composer ⇄ A4 Validator — Design

**Date:** 2026-07-05
**Status:** Implemented (PR #17); plan at `docs/superpowers/plans/2026-07-05-reflection-loop.md`
**Issue:** epic #7 follow-up (composer faithfulness → reliably-green all-LLM pipeline)

## Problem

The all-LLM pipeline (`LLMAnalyst` + `LLMComposer` + `LLMClaimVerifier`) intermittently
trips A4's grounding gate: the LLM composer elaborates beyond the points it is given
(adds facts/inferences absent from the cited sources), and A4's semantic verifier
correctly rejects the unfaithful sections. Tightening A3's prompt (PR #15) reduced but
did not eliminate this — a prompt cannot guarantee faithfulness. The reflection loop is
the Harness-level fix: let A3 recompose with feedback until the brief is grounded, or
fail loudly after a bounded number of attempts.

This is the ADD **Reflection Loop** topology: generator (A3) + critic (A4), iterate
until convergence or max rounds.

## Design decisions (approved)

1. **A4 gains a report mode.** A4 currently raises on a failed gate. For the loop it
   must be able to _report_ failure without aborting. A new `check()` returns the brief
   with its checks (no raise); `run()` stays as `check()` + raise (the standalone hard
   gate, unchanged).
2. **Per-claim feedback.** The loop feeds A3 the specific claim texts the verifier
   judged unsupported, so recomposition is targeted. No extra LLM call — reuses the
   per-claim verdicts A4 already computes.
3. **Raise on exhaustion.** After max attempts with grounding still failing, the gate
   raises `GuardrailViolation` — the invariant "no unsupported claim leaves the pipeline"
   is preserved. The loop raises reliability, not softens the gate.

## Components

### A4 report mode — `agents/validator.py`

- `ValidationOutcome` (new, internal model): `brief: ValidatedBrief`, `unsupported: list[str]`
  (the claim texts the verifier returned `False` for).
- `A4Validator.check(brief_input) -> ValidationOutcome`: computes grounding (per-claim
  `verify`), policy, and format checks; builds the `ValidatedBrief`; collects the
  unsupported claim texts. **Does not raise** on a failed check.
- `A4Validator.run(brief_input) -> ValidatedBrief`: `check()` then
  `validate_brief_output(outcome.brief)`. External behavior unchanged — still the
  standalone hard gate.
- `SOURCE_UNRESOLVED` raised by `LLMClaimVerifier.verify` still propagates immediately
  from `check()` — an infra/indexing fault, not a content-grounding failure the loop can
  fix by recomposing.

### A3 feedback — `agents/composer.py`

- `Composer` protocol: `compose(composer_input, feedback: list[str] | None = None)`.
- `A3Composer.run(composer_input, feedback=None)`: threads feedback to the composer.
- `LLMComposer.compose`: when `feedback` is present, appends to the human message:
  "Your previous draft made these statements that were NOT supported by their cited
  sources: [...]. Recompose stating only what the points assert; drop or rephrase the
  unsupported content."
- `RuleBasedComposer.compose`: accepts and ignores `feedback` (already faithful).

### Reflection graph — `graph/pipeline.py`

- `PipelineState` gains: `feedback: list[str] | None`, `attempt: int`.
- `composer_node`: runs `A3.run(translate_analysis_to_composition(analysis), feedback=state.get("feedback"))`;
  returns `{"draft": ..., "attempt": state.get("attempt", 0) + 1}`.
- `validator_node`: runs `A4.check(translate_draft_to_validation(draft))`;
  returns `{"brief": outcome.brief, "feedback": outcome.unsupported}`.
- Conditional edge after `validator`:
  `brief.checks.grounding_ok OR attempt >= MAX_COMPOSE_ATTEMPTS → "gate"`, else → `"composer"`.
- `gate_node`: `validate_brief_output(state["brief"])` (raises on any failed check,
  including grounding after exhaustion) → `END`.
- A1 → A2 run once; only A3 ⇄ A4 loops.

### Config — `config.py`

- `MAX_COMPOSE_ATTEMPTS = 3` (1 initial + 2 retries).

## Data flow

```text
request -> A1 -> A2 -> [composer -> validator]* -> gate -> ValidatedBrief
                          ^__________________|
                          recompose with per-claim feedback
                          while not grounded and attempt < MAX
```

## Error handling

- **Grounding failure (content):** loop recomposes with feedback, up to `MAX`.
- **`SOURCE_UNRESOLVED` (infra):** raises immediately from `check()` — recomposition can't
  fix a missing KB document.
- **Policy / format failure:** the loop is grounding-driven; these fall through to the
  gate node, which raises (recomposition would not fix a banned phrase or a malformed
  body).
- **Grounding still failing at `attempt >= MAX`:** routed to gate, which raises
  `GROUNDING_FAILED`.

## Testing

- **Deterministic loop mechanism (keyless, the key regression protection):**
  - Inject a stateful test `ClaimVerifier` that fails attempt 1 then passes: assert the
    loop retried, passed the unsupported feedback to a capturing composer, and produced
    a grounded brief with `attempt == 2`.
  - An always-fail verifier: assert the gate raises `GuardrailViolation("GROUNDING_FAILED")`
    after `MAX_COMPOSE_ATTEMPTS`.
- **RuleBased 4-stage:** still produces a grounded brief on attempt 1 (no retry); existing
  graph tests updated for the new state fields / signature.
- **A4 unit:** `check()` returns unsupported claims without raising; `run()` still raises;
  `check()` still propagates `SOURCE_UNRESOLVED`.
- **A3 unit:** `LLMComposer` receives feedback; `RuleBasedComposer` ignores it.
- **Live gated e2e:** the all-LLM pipeline with reflection produces a grounded brief.
  LLM-stochastic, so a smoke test (verified live during implementation), not a hard CI
  gate. Faithfulness _rate_ remains an eval-harness concern.

## Out of scope

- A separate critic LLM call (per-claim verdicts are reused instead).
- Reflecting on policy/format failures.
- Making the live full-pipeline e2e a hard CI test (it stays gated + smoke-only).
