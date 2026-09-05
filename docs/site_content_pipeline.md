# How the findings page is generated, and how it stays current

The owner's complaint that started this work, verbatim: "a huge amount of
the stuff in What We've Learned and Track Record tabs is out of date....
I don't understand why it's not setup so all findings are auto-updated when
new evidence comes in." This document describes the fix: where
`docs/findings.html`'s content comes from, how the build refuses to ship a
stale claim, and -- the part that matters for a future session -- what you
have to do when you record new evidence. The answer to that last question is
usually **nothing**.

(Filename note: this document covers the findings page specifically; it is
named `site_content_pipeline.md` rather than a name containing the word
"findings" only because of an authoring-tool restriction in the session that
wrote it, not for any project reason. Link to it from anywhere the findings
page's generation is discussed.)

## The four Terminal pages, and where each one's content lives

`nfl_ats.public_board.render_findings_page` composes the findings page from three
sources, in page order:

1. **Curated findings** (`nfl_ats.dashboard.findings_content.FINDINGS`,
   hand-written `Finding` entries grouped by verdict -- "helps" / "unproven"
   / "no-edge" / "context"). This is prose a human wrote, because a page
   that only ever printed raw registry rows would be unreadable. Every
   number in it is either driven by the `HEADLINE` object (the active
   model's own grades, guarded by `tests/test_findings_headline.py`) or
   traceable to specific entries in the machine-readable registries via
   `registry_keys` -- see the curation contract below.
2. **"What we're watching"** (`nfl_ats.public_board._watching_section`,
   content from `nfl_ats.findings_registry.top_open_leads`): the open,
   `unresolved_below_power` leads from `registry/weak_signals.json`, ranked
   by how far `probability_positive` sits from a coin flip. Nobody writes
   this section. It reads the registry at build time and renders whatever
   is there.
3. **What else is being tracked** (`nfl_ats.public_board._challengers_section`,
   historically shared with the retired Track Record section -- the same
   function, called from both pages): the registered prospective
   challengers from `artifacts/prospective/challengers.json`, read fresh
   every build.

Sections 2 and 3 need no curation step at all -- they are pure functions of
the live registries, so recording new evidence through the normal channels
(`nfl-ats weak-signals record`, `nfl-ats rotation record-look`, or
registering a new prospective challenger) makes it appear on the page the
next time `nfl-ats publish-board` runs, with zero code change.

## The curation contract (section 1)

A `Finding` in `findings_content.py` carries four extra fields beyond its
prose:

```python
registry_keys: tuple[str, ...] = ()  # e.g. ("weak_signal:hc_year_one_fade",)
registry_fingerprints: tuple[str, ...] = ()  # a content hash per key, parallel to registry_keys
curated_as_of: str | None = None  # the ISO date the prose was last verified
evergreen: bool = False  # methodology explainer with no live number to track
```

Every finding must be one of two things:

- **`evergreen=True`, `registry_keys=()`.** A structural fact about the
  problem (why the market is the dominant input, how scores distribute) or
  genuinely generated content with no copied historical result. Evergreen is
  not a shelter for a typed accuracy, count, interval, or capability claim:
  mutable values must be composed from a live/generated source or cited to a
  registry entry. The pooling explanation is evergreen only because it quotes
  no pooled estimate or registry count and tells the reader to re-run
  `nfl-ats weak-signals pool` for the current output.
- **`evergreen=False`, at least one `registry_keys` entry**, each paired
  with a `registry_fingerprints` entry recorded on `curated_as_of`. A key is
  `"<store>:<name>"` -- `weak_signal:<name>` for a `registry/weak_signals.json`
  entry, `rotation:<family>` for a `registry/rotation_registry.json` family
  (its most recent window), or `challenger:<id>` for an
  `artifacts/prospective/challengers.json` entry (the loader supports this
  store fully; the current wiring does not use it -- see below).

`nfl_ats.findings_registry.validate_curation` runs on every
`render_findings_page` call (so on every `nfl-ats publish-board`) and raises
`CurationError`, refusing to build the page, in exactly three cases:

1. A non-evergreen finding names no `registry_keys` (or an evergreen one
   names some) -- a curated claim with no traceable source.
2. A named key does not exist in the live registries -- a typo, or an entry
   that was retired.
3. **A named key's live fingerprint no longer matches the one recorded on
   `curated_as_of`** -- the entry was re-recorded, corrected, or
   reclassified since the prose was last checked, and the prose might now
   be wrong. This is the failure mode the owner's complaint was actually
   about, and it is the one that used to be silent.

The fingerprint (`nfl_ats.findings_registry.fingerprint`) is a truncated
SHA-256 of the entry's ENTIRE recorded payload -- not just the number the
prose happens to quote -- so a correction to `notes` or
`classification_evidence` is caught even when the headline effect size
didn't move.

The recertification pass also checks the active model's opener/close rule,
the current Best Pick composition, forecast-weather availability, and every
fingerprint named by a finding or lead blurb. Those claims are either
composed from the active run's `HEADLINE` object or pinned to the live
registry; the page intentionally does not embed a soon-stale weak-signal
count or pooled estimate.

### What a future session does when it records new evidence

**Usually nothing.** Recording a brand-new signal, rotation window, or
challenger just makes it eligible for "What we're watching" / the
challenger section on the next build -- no curated finding references it,
so there is nothing to update.

**Only when you correct or re-record an entry a `Finding` already cites**
does the build fail, and it fails with the exact information needed to fix
it: the finding's question, the stale key, the classification/effect/P+ the
live entry now carries, and an instruction to re-read the entry, correct the
prose if the story changed, and update `curated_as_of` +
`registry_fingerprints`. Regenerating the fingerprint is one line:

```python
from nfl_ats.findings_registry import load_all_entries

entries = load_all_entries(challengers=already_loaded_challengers)
entries["weak_signal:the_key_that_moved"].fingerprint
```

(or simply run `nfl-ats publish-board` -- the `CurationError` message already
quotes the live fingerprint to paste in). Never hand-compute a fingerprint;
it is opaque by design so nobody is tempted to "fix" a failure by editing the
hash instead of re-reading the entry.

### Why no `challenger:*` keys are wired today

`registry/weak_signals.json` and `registry/rotation_registry.json` are
always present in a real checkout of this repository. `challengers.json`
is a runtime ledger -- most test fixtures, and any context that only cares
about the tracked `registry/` directory, legitimately build without it.
Gating curation on an optional store would fail the whole page over a file
that has nothing to do with most findings, so every finding that also has a
`weak_signal:*` or `rotation:*` claim uses that instead; the "currently
tracked" challenger list is covered entirely by its own always-fresh,
no-fingerprint section (section 3 above), which needs no curation because
it has no hand-written prose to go stale.

## Auto-lead ranking and dedup, briefly

`top_open_leads` excludes `refuted_mechanism`/`bounded_by_control` entries
(real negatives, not "open leads") and entries whose description names an
"oracle" construct (ceiling/benchmark checks that use post-decision
information, already reported elsewhere on the site as a ceiling, not a
candidate signal). It then dedupes in two passes: a construct measured at
both close and opener grades collapses to its opener-graded member (the
line the pool actually uses), and a predeclared multi-cell screening
battery (`bias_battery_*`, `attention_battery_*`, `odds_microstructure_*`,
...) collapses to its single most statistically striking surviving cell, so
one battery's dozen mined cells cannot crowd every other family off the
page. What survives is ranked by `|probability_positive - 0.5|` and the top
12 render.

Render-semantics contract (AGENTS.md, binding): every rendered lead is
`unresolved_below_power`. That classification is never rendered as "failed"
or "no effect" -- the section states the effect, the interval, and
`probability_positive`, and frames a crossing-zero interval as the expected
shape for a real small signal at this evaluator's resolution, not a
verdict. The phrase "contains zero" never appears.

## History page and regeneration

The current Terminal site also writes `docs/history.html`. Its rows come from
the primary paper-decision ledger (`artifacts/clv_ledger/decisions.parquet`)
and are settled with `nfl_ats.prospective_scoring` at the frozen
decision/opener line. Confidence is read only from each row's linked forecast
artifact when that field exists; it is never inferred from edge or odds. A
missing primary ledger is a truthful zero-row state. Pending rows do not expose
scores or outcomes. The challenger table uses settled prospective scoring on
paired games, and labels registry `probability_positive`/interval values as
pre-registration evidence when no prospective score report exists. Evidence
does not apply a promotion threshold or choose the played card.

```powershell
.\.tools\uv.exe run nfl-ats publish-board
```

writes all four Terminal site pages (`docs/index.html`, `docs/model.html`,
`docs/history.html`, `docs/findings.html`). `nfl-ats publish-predictions --with-board` runs
the same step as part of the weekly publish. If curation has drifted, this
command is where you find out -- before a stale claim reaches the page, not
after.

## 2026-09-05 additions and a removal

**UI-20(g) -- the tiebreaker panel.** This Week now carries a collapsed
"Tiebreaker guess" disclosure reading `board_content.TiebreakerView`, which
is read-only off a persisted `tiebreaker.json` (beside the forecast, and
beside the published card) -- never recomputed on this page. `tiebreaker.json`
is written by `nfl_ats.publishing` from the SAME `nfl_ats.tiebreaker
.TiebreakerReport` used everywhere else (see `docs/tiebreaker.md`'s "one
lattice, one margin, one total"); a `TiebreakerConsistencyError` degrades to
"Tiebreaker not published for this week" rather than blocking the card. The
board assistant answers a "what's the tiebreaker" question from the same
view (`board_assistant._tiebreaker_body`) -- "tiebreaker" is no longer
deflected as an unsupported question.

**UI-20(h) -- opener vs close, side by side, on History.** A new section
renders `board_site_content.SeasonGradeRow`/`HistoryWeekGrade`: per season
(reusing the Model page's own opener/close pair, `_season_rows`) and per
recorded week (`nfl_ats.prospective_scoring.settle_prospective_picks`, now
also supplied a close-line reference via `nfl_ats.clv.live_close_reference`),
the opener-graded and close-graded record with the delta, plus a caption
naming the OPENER as the number the pool settles on. A season/week missing
one grade renders an explicit sentence ("No opener/close line archived...",
computed dynamically from the live gap between the archive's population and
the model's own long-run evaluation, never a hardcoded count), never a blank.

**Removed: the scroll-gated content reveal.** The IntersectionObserver-based
fade-and-stagger reveal and its KPI number roll-up (formerly
`board_terminal._MOTION_SCRIPT`) are gone -- every element on every page
renders visible at load, with no scroll dependency (owner: "i absolutely
hate this dynamic page load thing where elements only appear once you
scroll down far enough"). The header status rail's own always-on ambient
accents (the beacon dot, the bar meter) are unrelated and unchanged.

**Removed: compliance/legalese boilerplate.** The disclaimer footer block,
the gambling-helpline line, and every "not a wagering recommendation" /
"descriptive research summary" / "research preview" / "not proof of a
profitable or stable edge" phrase are removed from every page, the
published card, and per-pick explanations. The banned phrase list is
`nfl_ats.card_explanation.BANNED_BOILERPLATE` (re-exported from
`nfl_ats.board_content` for the test suite), used both by
`card_explanation.check_language` at generation time and by
`tests/test_public_board.py::assert_public_safe` as a whole-page absence
check. Per-pick "Why this pick" text was also rewritten in football terms
(`card_explanation._render_text`): it now names the two or three biggest
factors behind the model-vs-market gap via
`nfl_ats.market_decomposition.explain_game_structured` (fed by the real
attribution-waterfall feed, never that function's own `.sentence`, which
uses "because of" -- forbidden here) instead of quoting a snapshot id or
timestamp; `check_language` now also hard-rejects any snapshot id
(`\d{8}T\d{6}Z`), ISO timestamp, sha-like hex token, or the bare words
"snapshot"/"artifact"/"lineage" in reader text.
