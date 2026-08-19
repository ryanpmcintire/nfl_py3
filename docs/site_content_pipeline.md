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

## The three sections, and where each one's content lives

`nfl_ats.public_board.render_findings_page` composes the page from three
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
   shared verbatim with `track_record.html`'s own D3(a) section -- the same
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
  problem (what the ceiling is, why the market is the dominant input, how
  scores distribute) or a live command's own output (the pooling section
  explicitly tells the reader to re-run `nfl-ats weak-signals pool` rather
  than trusting a quoted number). Nothing here is a single registry entry to
  fingerprint.
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

## Regenerating the page

```powershell
.\.tools\uv.exe run nfl-ats publish-board
```

writes all three site pages (`docs/index.html`, `docs/findings.html`,
`docs/track_record.html`). `nfl-ats publish-predictions --with-board` runs
the same step as part of the weekly publish. If curation has drifted, this
command is where you find out -- before a stale claim reaches the page, not
after.
