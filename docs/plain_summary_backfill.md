# Template-driven `plain_summary` backfill (MOD-18 cell batteries)

Lane Q, 2026-09-08. Extends `scripts/backfill_plain_summaries.py` (previously
a hand-written `PLAIN_SUMMARIES` mapping only, lane AQ 2026-09-05) with a
second, TEMPLATE-driven mode for large batteries of near-identical cells
that a name/field grammar can describe mechanically, one hand-written
sentence per row does not scale to.

## Why this exists

A research lane recorded roughly 1,800 (and still growing at the time of
writing -- the lane was actively running throughout this session)
`mod18_home_side_location_cfb_v1_*` rows -- a college-football replication
of the MOD-18 home-side push (`docs/cfb_home_side_replication.md`) -- with
no `--plain-summary`. Being `recorded_at` today, every one of them falls
inside `nfl_ats.findings_registry.recent_registry_activity`'s 7-day window,
which is exactly what
`tests/test_board_humanised.py::test_recent_activity_weak_signal_entries_have_no_plain_summary_backlog`
checks: the board refuses to show "Research this week" with a weak-signal
row that has no genuine plain-English summary.

## What was built

Two new template sets, selected with `--template-set`:

- **`cfb_home_side_location_v1`** -- for `mod18_home_side_location_cfb_v1_*`.
  Parses the row's own `description` field (`"CFB {arm} {era} {spread}
  {home}"`, e.g. `"CFB diagnosis era_2006_2011 7.5-10 home_underdog"`),
  which already tokenises cleanly on spaces -- the description, not the
  name, because the name packs the same pieces into one underscore run that
  is genuinely ambiguous to split back apart (`era_2006_2011` vs `7_5_10`).
  The year range always comes from the row's own `seasons` field, never the
  era token.
  - `diagnosis` arm (`ats_points`): "College football check of the
    home-team push: on 10.5+ point spreads with a home underdog, 2006-2011,
    the home team beat the model's number by about 3 points." (positive =
    home beat the model's number; negative = missed it -- the diagnosis
    arm's own sign convention, `docs/home_side_offset_promotion.md`'s
    "the 10.5+ home point error").
  - `s2`/`s3` arms (`accuracy_points`/`brier_improvement`/
    `log_loss_improvement`): "College football check of the home-team push:
    applying it only on spreads of seven points or more, 2021-2025, reads
    about 0.4 percentage points better than not applying it." (s3 names its
    own serving rule -- big spreads only, per `docs/home_side_offset_promotion.md`'s
    "S3 played" section; s2 serves every spread.)

- **`nfl_home_side_cell_v1`** -- for the cell grammar shared by
  `mod18_home_side_location_v1_{s3,s4,s5}_*` (`docs/home_side_offset_promotion.md`,
  `docs/home_side_side_aware.md`, `docs/home_side_prior.md`) and
  `mod18_conditional_margin_v1_{m1,mp1}_*` (`docs/big_spread_lattice.md`).
  These rows' own `description` is boilerplate ("Home-side location arm
  {name-suffix}, positive favours the candidate"), so this template parses
  the NAME directly against the shared shapes: `{group}_{variant}_cell_bucket_
  {bucket}[_{home}]_{metric}_{start}_{end}`, `..._overall_{metric}_{start}_{end}`,
  `..._season_{year}_{metric}_{start}_{end}`, `..._push_at_3_{metric}_{start}_{end}`,
  plus S3's own `s3_{year}_vs_s2` comparison shape. All 414 currently-recorded
  rows in these five families already carry a `plain_summary` (measured
  2026-09-08: `nfl-ats weak-signals status` / direct registry scan, zero
  missing across `s3`/`s4`/`s5`/`m1`/`mp1`); this template set is a
  forward-looking safety net for the next lane in this family, not an
  active backlog.

Both templates:

- Read every effect from the row's own `effect`/`effect_units` field, never
  re-derive it, and always follow the module-wide "positive favours the
  candidate" convention (`nfl_ats.weak_signals`'s own documented sign rule)
  -- so a negative `accuracy_points`/`brier_improvement`/`log_loss_improvement`
  effect always reads as worse than the comparison, never silently flipped.
- Raise `ValueError` (never guess) for any row whose description/name does
  not match the grammar the function knows -- an unrecognised arm, an
  unrecognised NFL sub-arm group, or a cell shape neither regex matches.
  The caller (`run_template_backfill`) catches this and lists the row under
  `skipped`, with the reason, rather than writing a fabricated sentence.
- Are checked, EVERY generated sentence, against `banned_tokens_in` --
  a standalone reimplementation of `tests/test_board_humanised.py`'s render
  contract scan (hex-looking ids, `..._v1`-style slugs, raw artifact/ISO
  timestamps, the literal "P+"/"week-blocked", bare snake_case identifiers,
  `nfl_ats.card_explanation.BANNED_BOILERPLATE`) -- before it is ever
  considered a candidate to write. A sentence that fails this check is
  treated exactly like an unparseable row: skipped, never written.

## Commands

Preview (default; writes nothing):

```powershell
.\.tools\uv.exe run --no-sync python scripts\backfill_plain_summaries.py `
    --prefix mod18_home_side_location_cfb_v1_ --template-set cfb_home_side_location_v1
```

Apply (writes through the registry's own `weak-signals record --replace`
path -- see "How it writes" below):

```powershell
.\.tools\uv.exe run --no-sync python scripts\backfill_plain_summaries.py `
    --prefix mod18_home_side_location_cfb_v1_ --template-set cfb_home_side_location_v1 --apply
```

The NFL cell grammar, once any of its five families has a real backlog
(none do as of this writing):

```powershell
.\.tools\uv.exe run --no-sync python scripts\backfill_plain_summaries.py `
    --family mod18_home_side_location_v1 --template-set nfl_home_side_cell_v1 --apply
.\.tools\uv.exe run --no-sync python scripts\backfill_plain_summaries.py `
    --family mod18_conditional_margin_v1 --template-set nfl_home_side_cell_v1 --apply
```

`--prefix` and `--family` may be combined (both filters must match); at
least one is required whenever `--template-set` is given. The legacy
`PLAIN_SUMMARIES` hand-written-mapping mode (lane AQ) is unaffected and
keeps its own, OPPOSITE default: it writes unless `--dry-run` is passed.
The template mode is deliberately the reverse -- preview unless `--apply`
is passed -- because it can touch orders of magnitude more rows per
invocation than a dozen hand-picked names ever would.

## How it writes

Exactly like the legacy mode: through `_record_args` + `nfl_ats_cli.main`,
i.e. the real `weak-signals record --replace` CLI path, so
`record_signal`'s `validate_closure`/`validate_coherence` run on every
write and every field except `plain_summary` is read back off the LIVE
record immediately before re-recording it -- nothing is hand-transcribed.
A before/after diff of the whole registry file asserts the only JSON keys
that changed anywhere are `plain_summary` leaves; `run_template_backfill`
raises if that assertion fails.

### `_forced_registry_dir` (why it exists)

`nfl_ats_cli.main(argv)` has no parameter for "which registry file" -- it
always resolves its own path via `nfl_ats.weak_signals.default_registry_path()`,
which reads `NFL_ATS_REGISTRY_DIR` from the process environment.
`run_template_backfill(registry_path, ...)` takes an explicit
`registry_path` for its OWN reads, and until this fix that path and the
CLI's env-driven path had no way to be forced to agree.

**Incident, 2026-09-08 (this lane, caught before commit):** this test
suite's first draft called `run_template_backfill(apply=True)` against a
synthetic tmp-path fixture without setting `NFL_ATS_REGISTRY_DIR`. The
function's own reads correctly targeted the fixture; the `nfl_ats_cli.main`
write inside it did not -- it silently resolved to the ambient default,
`registry/weak_signals.json`, and used `--replace` to overwrite three rows
that happened to share names with the fixture (`mod18_home_side_location_cfb_v1_diagnosis_all_all_all_ats_points`,
`mod18_home_side_location_cfb_v1_s2_all_all_all_accuracy_points`,
`mod18_conditional_margin_v1_m1_m1_overall_standalone_2020_2025`),
fabricating their `effect`/`interval`/`sample_games`/`notes`/etc. fields
with the test's synthetic values. Caught by the auto-mode classifier
refusing the write-permission this session was never granted for
`registry/weak_signals.json`, before the SECOND (repair) write could
happen either -- so the live file was left corrupted on exactly those
three rows rather than doubly so. The fix is `_forced_registry_dir`: every
apply-mode write in `run_template_backfill` now runs with
`NFL_ATS_REGISTRY_DIR` forced to `registry_path.parent`, restored
afterward, plus a hard `default_registry_path() == registry_path`
assertion immediately inside that block that refuses to proceed at all if
the two still disagree. `tests/test_backfill_plain_summaries.py` also
carries a belt-and-suspenders `autouse` fixture that points every test at
an isolated `NFL_ATS_REGISTRY_DIR` regardless. **The three corrupted rows
in the live registry were NOT repaired by this lane** (no registry-write
permission) -- see the session report for the exact recovered values (one
fully recoverable from git `HEAD`, two only to the rounded `effect` this
lane's own earlier preview run had already printed) and the coordinator
follow-up this needs.

## Known limitations (disclosed, not hidden)

- `_NFL_VARIANT_NOTE` names only the specific lettered-variant distinctions
  this lane could verify by reading `docs/home_side_side_aware.md` (S4b:
  50-game prior sensitivity) and `docs/home_side_prior.md` (S5a/b/c: 50/25/200-game
  priors) and `docs/big_spread_lattice.md` (M1b: also checks which side of
  the nearest key number the line sits on). A future lettered variant with
  no entry there still gets a correct base-arm sentence, just without its
  own distinguishing clause -- a degradation in detail, not a correctness
  bug, and never a source of banned-token leakage.
- Both `brier_improvement`/`log_loss_improvement` phrasing is deliberately
  generic ("the model's confidence numbers land about X points closer to /
  further from what actually happened") rather than naming "Brier score" or
  "log loss" -- board is for humans (AGENTS.md); a pool player does not
  need the metric's name, only its direction and size.
- `--missing-plain-summary` (lane AQ, unchanged by this lane) still scopes
  to what the live findings page actually renders (What we're watching /
  Research this week / Signal registry notable rows), not a full registry
  scan; it is the right tool to confirm a specific backlog is now empty
  after a `--template-set --apply` run.
