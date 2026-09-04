# Board assistant ("chatbot thing") — feasibility scout

**Status:** BUILT 2026-09-04 (UI-16 ✅). `src/nfl_ats/board_assistant.py` +
panel wired into all four Terminal pages via `board_terminal.py`; corpus
built from the pages' own content dataclasses at render time; 33 tests in
`tests/test_board_assistant.py`; Python/JS parity verified by executing the
shipped script in Node over 18 frozen questions (harness kept outside the
repo -- DOM stub, not a repo fixture). The constraint is fixed: the
dashboard is a static GitHub Pages site (plus an optional read-only
unprivileged container — `docs/container_deployment.md`), so anything
needing secrets, egress billing, or a stateful server breaks the
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

## Build notes (2026-09-04, supersedes the A1-A4 plan below where noted)

Option 3 shipped, with one deliberate deviation from work package A1:
no separate `assistant_knowledge.json` file is written. The hosted
dashboard's nginx allowlist serves only the four HTML pages (anything
else 404s) and its CSP sets `connect-src 'none'`, so a sidecar file
would be unreachable exactly where it matters. Instead each page embeds
its corpus inline as `<script type="application/json">` (angle brackets
unicode-escaped; plain `json.loads` decodes it back). Same provenance
block, same determinism, same golden tests as the file plan -- only the
transport changed, and the no-fetch constraint the scout already
required is what forced it.

The A2 matcher is implemented twice from one source: the Python
reference (`board_assistant.answer`, frozen Q/A fixture) and a thin
inline-JS port that ranks the same embedded entries with the same
embedded synonym table and returns the winning entry's own body. The
STOP table and deflect rules are GENERATED from the Python constants at
render time, so the port cannot drift. Three real routing bugs were
caught by the fixture during the build (stopword noise, synonym
expansion leaking into reverse-substring matches, fragment-vs-exact
ties) and are pinned by regression tests.

Guardrails hold by construction (JS templates interpolate corpus
values only) and are pinned by a numeric-guard test: every number in
every fixture answer also occurs in the corpus dump. The
must-deflect set (wagering advice, future weeks, pick popularity,
exact scores) is pinned per rule.

## Decision needed: none

Shipped as scoped. Follow-ups are UI-17/18/19 in ROADMAP.md.
