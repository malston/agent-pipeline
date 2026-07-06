# Grounding-gate pilot — Design

**Date:** 2026-07-06
**Status:** Proposed (kill-criteria experiment; no build committed)
**Context:** The `claude-kb` corpus is populated by an external daily source-watcher that
distills canonical Anthropic/AWS sources into posts via an LLM, wires them into the
navigation layer, runs `check-kb.py` (structural integrity only), and opens a PR. Nothing
verifies that a claim in a generated post is actually supported by its cited source. This
pilot measures whether the `agent-pipeline` A4 grounding verifier (`LLMClaimVerifier`) would
plug that gap cleanly — before any gate is built into the watcher.

## Problem

The watcher's distiller is an LLM, so it carries the "elaborates beyond its source" failure
mode the pipeline's A4 gate was built to catch. Today the only automated check before a post
merges is `check-kb.py` (11 checks: links, tags, header fields, counts — all structural, no
claim verification), plus a human eyeballing the PR. There is no signal on whether the
corpus actually contains fabrication, and no evidence that an automated grounding gate would
catch it at a tolerable false-positive rate.

## The decision this informs

The pilot exists to produce a **build / don't-build / fix-verifier-first** decision, on two
numbers:

1. **Fabrication rate** — of the claims in recently-generated posts, how many are not
   supported by their cited source? (Is there a problem to solve?)
2. **Verifier precision** — of the claims `LLMClaimVerifier` flags, how many are real
   problems vs. the verifier crying wolf? (Would the gate be usable?)

**Decision rule (set before running):**

- Fabrication ≈ 0 → the corpus is already clean. **Do not build the gate.** (Best outcome —
  learned for a few dollars.)
- Fabrication meaningfully > 0 **and** precision ≥ ~75% at a tolerable flag rate → **build
  it**, wire into the watcher as the pre-PR gate.
- Fabrication > 0 **but** precision low (over-flags) → the verifier needs work (stricter
  prompt, better claim extraction) before it is gate-worthy. **Iterate the pilot; do not wire
  in yet.**

## Goal / Non-goals

**Goal:** a one-page report with the four metrics below and a one-line verdict against the
decision rule.

**Non-goals:**

- Not building or modifying the watcher.
- Not building the reflection-loop / redraft behavior — that is the gate's design, out of
  scope for a measurement pilot.
- Not measuring recall (requires planted fabrications; optional phase 2).
- Not characterizing the whole 268-post corpus — the sample targets the current watcher.

## Method

A standalone, offline script that reads the `claude-kb` repo files directly and reuses
`agent-pipeline`'s `LLMClaimVerifier` as a library. No watcher changes.

1. **Sample.** The ~50 most recent date-prefixed **Posts** in `claude/posts/` — the
   watcher-generated class. Skip `contributed/` (human-authored); the pilot tests the watcher.

2. **Acquire sources.** For each post, fetch the URLs in its `**Sources:**` block → text. A
   URL that will not fetch (moved / paywalled / JS-rendered) → its claims are
   `SOURCE_UNRESOLVED`, excluded from the fabrication measure and counted separately as
   **coverage**.

3. **Extract claims.** Per post, extract atomic factual claims from the **Summary + body
   sections** via a fixed LLM extraction prompt, excluding boilerplate (`Sources`, `Docs`,
   `Changelog`, `Related Posts`, headings). Each claim's `sources` = the post's Sources URL
   set. The extraction prompt is a controlled variable — held fixed across all 50 and
   documented, because granularity moves every downstream number.

4. **Verify.** Run `LLMClaimVerifier` per claim against the post's fetched sources →
   supported / unsupported. Configure it **strict** (default to _unsupported_ when uncertain
   — the refute-by-default posture). `grounded = every claim supported`.

5. **Adjudicate (ground truth).** A human labels every **flagged** (unsupported) claim into:
   - **fabrication** — the claim is not in the source (the target)
   - **drift** — it was faithful when written; the source has since changed (also useful)
   - **false positive** — the claim is supported; the verifier was wrong

   Only flags are adjudicated, so the cost scales with the flag rate.

## Metrics

- **Flag rate** = unsupported / total claims
- **Fabrication rate** = fabrications / total claims — _is there a problem?_
- **Verifier precision** = (fabrication + drift) / flagged — _is the gate usable?_ (the
  load-bearing number)
- **Coverage** = claims with a fetchable source / total claims

Recall is not measured. Optional phase 2 for recall: inject 5–10 synthetic fabrications into
copies of a few posts and measure catch rate.

## Effort & cost

~50 posts × (1 extraction call + K verify calls per post, K = claims/post). Bounded — a few
dollars of API, ~an hour of compute. Requires `ANTHROPIC_API_KEY` (the verifier is the gated
LLM path). The real cost is human adjudication of flags — small if the flag rate is low, and
a high flag rate is itself a finding.

## Threats to validity (report these; do not bury them)

- **Drift ≠ fabrication.** Verifying a committed post against its _current_ source conflates
  "the watcher fabricated" with "the source changed since." Step 5 splits them; report them
  separately or fabrication is overstated. (Both are legitimate catches for a real gate.)
- **LLM verifying an LLM.** The verifier has its own error rate — that is exactly what
  precision measures, which is why a flag is not treated as ground truth (step 5 exists).
- **Extraction granularity dominates.** Too coarse misses smuggled claims (understates
  fabrication); too fine false-positives on hedging prose (understates precision). One fixed
  prompt; the biggest knob.
- **Selection.** 50 recent posts test today's watcher (intended). Do not generalize the
  fabrication rate to the whole corpus without a stratified sample.

## Where it runs / dependencies

- Standalone script (`agent-pipeline` tooling), reads `claude/posts/` from a local `claude-kb`
  checkout, reuses `agent_pipeline.agents.validator.LLMClaimVerifier` backed by an in-memory
  `{url: fetched_text}` store rather than the pipeline's vector store.
- Assumes every sampled post carries a `**Sources:**` block of canonical URLs (the
  AUTHORING.md format).
- Output: a report file (the four metrics + the flagged-claim table with adjudication labels
  - the verdict).

## Open decisions

1. **Sample** — most-recent-50 (tests today's watcher; the default, right for a build/no-build
   decision on the gate) vs. a stratified sample across time/topics (characterizes the whole
   corpus).
2. **Drift confound** — treat "drifted since written" as a _catch_ (a real gate would flag
   stale posts too — arguably good) vs. isolate pure fabrication only (harder; needs source
   snapshots from post-authoring time).
