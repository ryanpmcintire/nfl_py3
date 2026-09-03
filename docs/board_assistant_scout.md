# Board assistant ("chatbot thing") — feasibility scout

**Status:** scouted 2026-09-03 on owner question; no build yet. The constraint
is fixed: the dashboard is a static GitHub Pages site (plus an optional
read-only unprivileged container — `docs/container_deployment.md`), so
anything needing secrets, egress billing, or a stateful server breaks the
deployment contract.

## Options, honestly graded

1. **In-browser LLM (WebLLM-style).** Technically hostable statically (model
   weights from a CDN, inference in WebGPU, no data leaves the device).
   Rejected for this board: multi-GB first download, mobile-hostile,
   uncontrolled answers that can contradict the research (a chatbot
   inventing cover probabilities would be worse than none), and no
   provenance story for anything it says.
2. **Hosted LLM backend (serverless + API key).** Rejected: needs secrets,
   egress, billing, and moderation — a second service with its own ops and
   threat model for a research dashboard. Revisit only if the board ever
   grows a real backend for another reason.
3. **Static guided assistant (recommended).** A chat panel over a
   publish-time-generated `assistant_knowledge.json`: every answer is
   retrieved, never generated. It can truthfully answer: why this pick
   (per-game model state, line, probability, policy flips), what Best Pick
   means, the model's record and intervals, glossary terms, findings
   summaries, and "what changed since Tuesday". It cannot hallucinate
   beyond its corpus because there is no generative step — unknown queries
   get an explicit "not in the published card" fallback with links.

## Recommended work packages (only option 3 is scoped)

- **A1 knowledge export:** `publish-board` writes `assistant_knowledge.json`
  (games, picks, probabilities, policy/flip reasons, best-pick nomination
  text, record chips, findings headlines, glossary) with the same provenance
  block as every other artifact. Read-only, deterministic, tested golden.
- **A2 matcher:** client-side intent ranking (keyword + synonym map, no
  network, no weights download). Returns ranked answers with citations to
  page anchors. Tested against a frozen Q/A fixture, including the
  must-deflect set (wagering advice, future games, non-public data).
- **A3 chat UI:** progressive-enhancement panel (works with JS disabled by
  degrading to the existing anchor-linked sections), keyboard accessible,
  `prefers-reduced-motion` honored, same terminal chrome. Mobile-collapsed
  by default.
- **A4 guardrails (release-blocking):** the assistant may never emit a
  probability, pick, or record not present in the knowledge file (tested by
  construction: renderer escapes + allowlist of answer templates); every
  answer carries its source anchor; the deflect set is pinned by tests.

Effort: A1–A4 is a medium dashboard build (new, no prerequisites beyond the
existing board pipeline). An LLM swap for A2 later would be a new decision
with its own threat model, not an upgrade.

## Decision needed: none yet

This stays a scout until the owner picks option 3 (or kills the idea).
Nothing was built, and no dashboard contract changed.
