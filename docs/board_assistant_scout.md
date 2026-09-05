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

## v2: compositional engine (2026-09-04, same constraints)

The v1 keyword-to-canned-row matcher was correctly judged a FAQ with
extra steps: it could not compose ("which confident Sunday dogs?"),
and real phrasings ("lock of the week", "most confident", "Sunday
night game") routinely misrouted. Rebuilt as intent parse (teams via
alias word-sets plus uppercase-code scan, days, glossary terms, counts)
with composed answers computed from structured corpus data: per-team
confidence with computed rank, top-N/bottom-N rankings, dog/favorite
lists, day schedules, same-game detection with cross-game comparison,
plus honest scope entries for what the board does not publish
(winners, injuries, weather). New coverage: lock slang, like/love
phrasing, teaser/buy-points/over-under/fade-the-public deflects,
movement sentences in clock answers. Spec: 120-question anticipated
battery (`tests/test_assistant_battery.py`); Python/JS parity verified
by executing the shipped script over the whole battery (120/120, one
documented intentional divergence: empty submits are silent in JS).
Still not a conversation: no follow-ups, no memory, no generation.

## Decision needed: none

Shipped as scoped. Follow-ups are UI-17/18/19 in ROADMAP.md.

## UI-18 / ENG-04: lineup answers (2026-09-04)

Four new intents read the published `lineups.json` artifact (see
`docs/projected_lineups.md` for the artifact itself) via a new module,
`src/nfl_ats/board_assistant_lineups.py`, kept separate from
`board_assistant.py` to limit merge-conflict surface on that large,
concurrently-edited file. `board_assistant.build_knowledge_for_board`
merges a precomputed `"lineups"` block into the corpus from the SAME
per-game `TeamLineup` objects `board_content.py` already attaches to each
game's dive (`home_lineup`/`away_lineup`) -- this code path never opens an
artifact itself.

**Intents:**

- *"Who is starting at QB for `<team>`?"* -- states the current
  depth-chart QB1, OR, when the fail-closed forecast/lineup consistency
  rule fires (`TeamLineup.note` is set -- the same signal
  `scripts/build_week_lineups.py` already stamps when the depth chart's
  QB1 disagrees with the forecast's assumed QB), names BOTH the
  forecast's assumed QB and the depth chart's current QB1 and explicitly
  refuses to state a single starter.
- *"Is `<player>` playing / available?"* -- play probability, injury
  status, and whether the player is the model's scored QB
  (`model_role == "base_model"`) or context-only, for any player
  resolved by name (exact full-name match, falling back to a unique last
  name) from the published snapshot. Requires an availability/status cue
  word alongside the resolved name so a shared surname token can never
  hijack an unrelated question.
- *"Any injuries for `<team>`?"* -- per-player injury designations when
  the artifact carries any, else the team-level injury-feed status
  string (today always "unavailable" -- no current injury source is
  attached; see `docs/projected_lineups.md`).
- *"Which games have a backup QB?"* -- lists every team whose current
  depth-chart QB1 disagrees with the forecast's assumed QB (the same
  fail-closed signal above, read across all published games rather than
  one team).

**Source-time anchors.** Every SUPPORTED answer above quotes its source
inline, `"as of <time> from <source>"`, taken verbatim from
`TeamLineup.as_of`/`.source`.

**Two documented fallbacks, never a guess:**

- *Absent* -- no lineup entry exists for the requested team/game (the
  artifact was never published this week, or this game isn't in it).
- *Stale* -- a lineup entry exists but its own `as_of` is older than a
  48-hour freshness budget (`LINEUP_STALE_BUDGET_HOURS`) relative to the
  page's own build time. That budget is new with this work (no prior
  document defined one): the scheduled refresh runs once a day
  (`docs/projected_lineups.md`), so 48h is one missed day of slack before
  the assistant treats the snapshot as too old to answer from. Staleness
  is computed once, at corpus-build time, and baked into the corpus as a
  plain boolean -- never re-derived at query time.

**Tests:** `tests/test_board_assistant_lineups.py` -- a synthetic
`lineups.json` written to `tmp_path` and loaded through the real
`nfl_ats.lineup_view` parser, then routed through `board_assistant.answer`
exactly like the site build does. Covers each intent, the consistency-rule
refusal, the stale and absent fallbacks, and that every supported answer
carries the source-time anchor. Verified against the real local
`artifacts/lineups/current/lineups.json` too, by rebuilding the full site
into a scratch directory (`scripts/build_full_site.py --out-dir`) and
querying the embedded corpus directly.

**JS port status:** closed by ENG-25 (below) -- the inline-JS port now
carries all four intents and is checked against the Python reference in
CI. See that section for the harness and how it's verified.

## Golden evaluation (ENG-05, 2026-09-04)

A broader, categorized fixture/evaluation suite, separate from the
anticipated-question battery above (`tests/test_assistant_battery.py`) and
the per-feature tests it sits beside: `tests/fixtures/assistant_golden/
questions.json` is a 103-row corpus (97 plus the six multi-word-glossary
routing rows ENG-36 added), each row a `question` / `expected_intent` /
`must_contain` / `must_not_contain` / `category` tuple, `category` one of
`routing`, `unsupported_fallback`, `numeric_provenance`, `stale_data`,
`lineup`, `accessibility_text`. Every intent `answer()` can return is
exercised -- deflects, scope answers, the four ENG-04 lineup intents, and
every glossary term, single- or multi-word (see ENG-36 below).

`src/nfl_ats/assistant_eval.py` grades the corpus
(`evaluate_golden(knowledge, questions, *, stale_knowledge=None) ->
EvalReport`, `render_report(report) -> str`): per-category pass counts,
every failing question with its actual answer, and an automatic check
that every `numeric_provenance` answer carries a recognized provenance
marker (`cover probability`, `as of ...`, `season-blocked`,
`most confident`, etc. -- see `PROVENANCE_MARKERS`) alongside its number,
never a bare digit. `make_stale_lineup_knowledge(knowledge)` returns a
deep copy with every published lineup entry forced past the 48h
staleness budget (`LINEUP_STALE_BUDGET_HOURS`) -- the "second knowledge
object" the `stale_data` category grades against, proven in
`tests/test_assistant_golden.py` to never name a starter and to always
carry the stale-fallback text.

`tests/test_assistant_golden.py` runs the full corpus against the same
`build_fixture_content()` fixture the other assistant tests share, plus a
small synthetic `lineups.json` (MIA/LV mismatch, NE/SEA clean, DEN/KC and
the rest deliberately unpublished) loaded through the real
`nfl_ats.lineup_view` parser -- 100% pass required overall and per
category, plus a parametrized test per question for isolated failures.
It also asserts the rendered chat panel's accessibility contract directly
on `board_terminal.render()` output: a labelled input
(`<label for="assistant-q">` / `<input id="assistant-q">`), a submit
button reachable by keyboard (`<form>` + `type="submit"`, no
`tabindex="-1"`, no click-only handler), an `aria-live="polite"` region
for answers, and a `<noscript>` fallback that both explains the assistant
itself needs JavaScript and leaves the page's own picks table rendering
unconditionally (proven by stripping every `<noscript>` block and
confirming the table markup survives). That last assertion FAILED against
the shipped markup -- the `<noscript>` block listed topic links but never
said the assistant needed JavaScript -- and was fixed with a one-paragraph
additive edit to `board_assistant.assistant_section` (a `<p>` before the
existing links; no CSS class added, so `board_terminal.TERMINAL_STYLE_CSS`
needed no change).

`scripts/assistant_eval.py [--json]` runs the SAME corpus against the
real, currently published corpus, built from local artifacts into memory
only via `board_site_content.load_site_content` (never writes a site or
knowledge file) -- exits non-zero on any failure. Run against this
session's real Week 1 artifacts it measured 68/97 (categories pinned to
week-invariant text -- `stale_data` 8/8, `unsupported_fallback` 17/17 --
passed clean; `routing`/`lineup`/`numeric_provenance`/
`accessibility_text` mostly failed on exact percentages, ranks, and
player names the fixture hard-codes from the synthetic card, not on
intent routing: every failure's actual topic matched its expected topic
except one, `refresh` vs `timing` for "what changed since Tuesday", which
is the CORRECT answer given no late-week refresh has run yet this week).
This is expected, not a defect: the corpus's substring checks are pinned
to `build_fixture_content()`'s fixed numbers on purpose, so CI grades it
against that fixture (`tests/test_assistant_golden.py`, 100% required);
the live script is a point-in-time diagnostic that a human runs to
sanity-check the router against real data, not an automated release gate
expected to pass every week.

**Multi-word glossary routing gap -- RESOLVED (ENG-36, 2026-09-04):** the
gap noted here at ENG-05 time -- `board_assistant._parse` tested a
glossary term's name for membership in a set of single query WORDS, so a
multi-word term (`cover probability`, `closing line`, `Best Pick`) could
never match -- is fixed. `_parse` (and the mirrored inline-JS `parse()`)
now do longest-match-first phrase matching over normalised n-grams: every
term name/alias is re-tokenised with the same `_tokens`/`tokens()`
normalisation as the query, and the longest n-gram of query tokens equal
to a candidate's token tuple wins (`_glossary_term_match` in Python,
`matchGlossaryTerm` in the JS port). Single-word terms match exactly as
before (a 1-token n-gram is just membership), so no existing routing
changed; this also incidentally makes the `ATS` alias "against the
spread" reachable, which the same bug had silently broken.
`test_golden_fixture_covers_every_router_intent` now asserts every
`GLOSSARY` entry is reachable (no more `" " not in item.term` exclusion),
and the golden corpus gained six rows -- "What is cover probability?" /
"What does cover probability mean?", "What is a closing line?" / "what
does closing line mean", "What is a Best Pick?" / "What does Best Pick
mean?" -- all passing in both the fixture-graded suite and the live
`scripts/assistant_eval.py` run against real Week 1 artifacts (measured
2026-09-04: 74/103 overall, `routing` 40/51, no new intent mismatches --
the same pre-existing `refresh` vs `timing` case is the only topic
mismatch anywhere in the report; every other routing/lineup/
numeric_provenance/accessibility_text failure is a fixture-pinned number
or player name, matching the ENG-05-era baseline's failure shape above).

## JS port + in-repo parity harness (ENG-25, 2026-09-04)

The four ENG-04 lineup intents are now ported into the inline-JS engine
(`board_assistant.assistant_script`), function-for-function against
`nfl_ats.board_assistant_lineups`: `lineupQbStarterAnswer`,
`lineupTeamInjuriesAnswer`, `lineupPlayerAvailabilityAnswer`,
`lineupBackupQbGamesAnswer`, plus the shared `lineupAnchorText` /
`lineupStaleText` / `lineupUnpublishedText` / `lineupTeamLookup` helpers.
Word-set gating (`QB_WORDS`, `BACKUP_WORDS`, `AVAILABILITY_WORDS`) is
GENERATED into the page from the same Python constants via `_INTENT_WORDS`
(`lineup_qb` / `lineup_backup` / `lineup_availability` keys) -- the existing
mechanism the rest of the router already used, so this port can't drift the
way the old placeholder gap could. The 48h staleness budget is never a
second hardcoded constant: both engines read it from the corpus itself
(`lineups.stale_budget_hours`, written once by
`board_assistant_lineups.build_lineup_knowledge`). Intent precedence,
consistency-rule refusal text, and the stale/absent fallback text are
copied verbatim from the Python reference.

**Pure engine, exposed for Node.** The submit-handler logic (deflect rules,
then `parse`/`answerParsed`) was hoisted out of the DOM click handler into
a standalone `answerQuestion(question, corpus)`, and the whole
`document.querySelectorAll(...)` DOM-wiring block is now guarded by
`typeof document !== 'undefined'`. At the end of the IIFE, a guarded
`if (typeof module !== 'undefined' && module.exports) { module.exports =
{ answer: answerQuestion }; }` exports that pure function -- a no-op in the
browser (`module` is undefined there), but it lets a Node process `require()`
the exact script every page ships and call the engine directly, no DOM stub
required.

**In-repo Node harness.** `tests/parity/assistant_parity.mjs` takes three
argv paths -- an extracted engine script (`.cjs`, so Node treats it as
CommonJS regardless of any ambient `package.json` `"type"`), a knowledge
JSON blob, and a JSON array of question strings -- calls `engine.answer`
on each, and prints a JSON array of `{question, topic, text, anchors}` to
stdout. It does no comparison itself.
`tests/test_assistant_js_parity.py` does the comparing: it builds ONE
synthetic knowledge corpus from the same fixtures ENG-05 uses
(`_board_content_fixtures.build_fixture_content` plus the mixed
MIA/LV-mismatch, NE/SEA-clean lineups artifact `test_assistant_golden.py`
already writes), extracts the live script straight out of
`assistant_script()`'s own return value (never a hand-copied duplicate),
runs every golden-corpus question (`tests/fixtures/assistant_golden/
questions.json`, 98 unique strings) plus 14 lineup-phrasing regression
questions harvested from `test_board_assistant_lineups.py`'s own Python
unit tests through both engines, and asserts the `{topic, text, anchors}`
triple matches exactly. This replaces the harness the "Build notes"
section above described as "kept outside the repo" -- it is now
version-controlled and runs in CI.

**Skip rule.** Node is a dev-time, not a runtime, dependency of this
project. `tests/test_assistant_js_parity.py` calls `shutil.which("node")`
and `pytest.skip`s with an explicit reason when it's absent -- a missing
local Node install is never treated as a test failure. When Node **is**
present (true in this session: `node --version` reported v22.19.0), the
check runs for real and every question must match or the test fails with
the full list of mismatches (question, Python answer, JS answer) inline in
the pytest output, not just a pass/fail bit.

One real bug was caught building this harness, in the harness itself, not
in either engine: comparing the two answers via `subprocess.run(...,
text=True)` without an explicit `encoding="utf-8"` mis-decoded a non-ASCII
character (an em dash in a fixture's injury-status string) through the
Windows console's default codepage, producing two false mismatches on an
otherwise-clean run. Fixed by passing `encoding="utf-8"` explicitly --
noted here because it is exactly the kind of divergence this harness exists
to catch, except this one lived in the test's own plumbing, not in either
assistant engine.

With the JS port and its own in-repo, CI-checked parity harness both done,
ENG-04 (UI-18: lineup answers) and ENG-05 (golden evaluation corpus) are
both fully shipped -- see ROADMAP.md Phase 13 for the status line.
