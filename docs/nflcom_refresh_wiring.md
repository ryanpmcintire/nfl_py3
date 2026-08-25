# NFL.com Friday out>=2 fade: refresh-path wiring

Status: **wired 2026-08-24, challenger-tracked only.** This document records
what was wired into the production refresh path, what deliberately did NOT
change, how 2026 prospective scoring consumes it, and the one predeclared
confirmation look this wiring enables. It complements
[`docs/nflcom_friday_refresh.md`](nflcom_friday_refresh.md) (the frozen rule
and composition study) and [`docs/late_week_refresh.md`](late_week_refresh.md)
(POL-11 refresh flow); it does not modify either.

## What was wired

The NFL.com Friday out>=2-starters fade
(`nflcom_friday_refresh_out2_starters_v1`) can now be **scored prospectively
by the refresh machinery itself**, alongside the market-follow rule:

- New module `src/nfl_ats/nflcom_refresh_overlay.py`, called from
  `refresh-picks` (`src/nfl_ats/cli.py::_cmd_refresh_picks`) AFTER
  `plan_refresh` has produced each game's post-market-follow played side.
  For every still-eligible game it computes the WOULD-BE pick under the
  frozen rule and appends it -- with the played side beside it -- to a
  SEPARATE append-only ledger,
  `artifacts/prospective/nflcom_friday_refresh_decisions.parquet`
  (`NFLCOM_REFRESH_OVERLAY_COLUMNS`). Recording is opt-in
  (`--record-decisions`) and guarded by the same
  `refuse_if_outside_recording_lock_window` rehearsal guard every other
  ledger uses.
- Signal inputs are REUSED VERBATIM, not reimplemented:
  `nfl_ats.prospective.nflcom_team_starter_out_counts` and
  `nfl_ats.prospective.nflcom_out2_starters_flip`. Those are the same
  normalization / starter-proxy / tie-rule machinery
  `scripts/nflcom_friday_designation_screen.py` built and
  `scripts/nflcom_friday_refresh_feature.py` reused
  (`build_out_counts`/`attach_counts`/`apply_overlay` are its archive-side
  wrappers around the identical logic) -- the exact machinery
  `registry/weak_signals.json:nflcom_refresh_out2_starters_on_chain` was
  measured with. There is no second or third copy anywhere; the counting
  function was extracted from the publish-time recorder's body verbatim so
  both recorders share one implementation.
- Base pick for the would-be arm: the PLAYED side after the observed-movement
  policy (`RefreshedGame.new_pick_side`), mirroring how the composition study
  applied the overlay on top of whatever the chain backs at refresh time.

**Bug fixed while wiring (measured this session):** extracting the counting
block exposed a latent crash in the publish-time recorder's inline version --
with `as_index=False`, `groupby(...)["col"].sum()` returns a DataFrame, so the
old `.rename("starter_out")` raised before any row could be written; the
publish-path try/except would have swallowed it as a silent skip. The
extracted `nflcom_team_starter_out_counts` uses the screen script's own
`agg(starter_out=("is_starter_caliber", "sum"))` idiom and is pinned by
`test_nflcom_team_starter_out_counts_aggregates_per_canonical_team_week`.

## What deliberately did NOT change

- **The played card is untouched. The overlay cannot alter any pick.**
  `plan_refresh`'s output is consumed read-only; nothing is written to
  `pick_revisions.parquet`, the published card, or the four-overlay Tuesday
  policy identity. Pinned by
  `tests/test_nflcom_refresh_overlay.py::test_overlay_can_never_alter_the_played_pick_or_the_revision_ledger`.
- **The market-follow rule stays the only decision rule at refresh time.**
  The reason this signal must stay CHALLENGER-TRACKED rather than auto-played
  is measured, not guessed: the max-EV composition study
  (`scripts/nflcom_friday_refresh_feature.py`,
  `docs/nflcom_friday_refresh.md`) showed that ADDING the overlay to the
  played chain LOWERS composed accuracy (in-stack marginal +0.20), even
  though the standalone composition (+2.1795 pts, P+ 0.9954 week-blocked,
  three seasons) looks strong. Wiring for SCORING is not wiring for PLAYING;
  promotion remains an owner decision under the forced-pick EV frame
  (AGENTS.md: a promotion bar governs claims, never which card is played).
- Publish-time tracking (`record_nflcom_refresh_out2_starters_challenger_
  decisions` inside `publish-predictions --record-decisions`) continues
  unchanged; the refresh-time ledger is an additional view of the SAME
  challenger, not a new one.

## 2026-08-25: the gate this wiring depended on could never open

Both recorders described here were, as wired, incapable of writing a single
row. The freshness gate required the page to post-date Friday 16:00 ET *and*
pre-date the week's earliest kickoff, which is a Thursday night in every week
but one — an empty window, measured unsatisfiable on 7 of 7 real weeks. The
"fail-open" contract below held perfectly and that was the problem: it failed
open into silence, every week, with no error to notice.

Corrected in both copies to a per-game boundary (`pick_refresh.pick_deadline`
= min(own kickoff, the week-wide Sunday 16:00 ET lock)), so a Friday page
scores the Sunday/Monday slate and drops only the Wed/Thu games it genuinely
post-dates. Full arithmetic, the re-scored effect, and the correction to the
headline number are in `docs/nflcom_friday_refresh.md`
("2026-08-25 correction"). Short version for anyone quoting this arm: the
production-reachable estimate is **+1.95 accuracy points (P+ 0.983
week-blocked, n=719)**, not the +2.18 measured on a population that included
games the corrected gate excludes.

Also required for any of this to produce evidence, and previously missing:
nothing captured a live NFL.com page at all. The only local snapshot covered
2022-2024. `scripts/ingest_nflcom_injuries.py --current` captures the live
week, and `scripts/capture_scheduler.py` decides when that is due (see
`docs/capture_scheduling.md`). The first live 2026 capture was taken
2026-08-25.

## Fail-open contract (pinned)

Every absent-input path is a documented NO-OP -- never an error, never a
flip: no `data/raw/nflcom_injuries/*/manifest.json` snapshot; no
`data/players/raw/*/snap_counts.parquet`; the week's page absent from the
manifest; or the page failing the freshness gate. An unexpected recorder
failure is caught in `_cmd_refresh_picks` so it cannot break the production
refresh pass either. Week 1 has no prior-week snaps, so the starter proxy is
unavailable there by construction -> counts 0 -> keep (the frozen rule).

## Leakage discipline

The flag may consume ONLY information available before kickoff:

- the week's FINAL NFL.com league injury page, manifest-gated to
  `fetched >= Friday 16:00 ET of the game week` (week-wide: this is what makes
  it the FINAL report) `AND fetched < each GAME's own pick deadline`
  (per-game: `pick_refresh.pick_deadline` = min(own kickoff, the week's Sunday
  16:00 ET lock)) -- identical gate to the publish-time recorder. Corrected
  2026-08-25; the superseded week-wide "< the week's earliest kickoff" form is
  described above;
- prior-week snap shares (<= week-1 REG games) for the starter proxy.

Regression pins in `tests/test_nflcom_refresh_overlay.py`:
`...page_fetched_at_or_after_kickoff_is_a_documented_noop` (a page post-dating
EVERY eligible game's deadline produces a skip with ZERO ledger writes --
post-deadline information is never consumed, never silently used),
`test_a_thursday_game_no_longer_silences_the_whole_week` (one Wed/Thu game
drops out alone instead of voiding the week), plus the pre-Friday-page pin.
`tests/test_prospective.py` carries the matching pair for the publish-time
recorder.

## How 2026 prospective scoring will consume it

Each row carries everything scoring needs at the frozen Tuesday grade:
`game_id`, `decision_home_spread` (the grading line),
`played_pick_side` vs `nflcom_would_be_pick_side`, both teams'
starter-out counts and flags, `injury_page_snapshot` +
`injury_page_fetched_at_utc` provenance, and `refresh_run_id` /
`revision_recorded_at_utc`. Passes append multiple rows per game across the
week's Thursday/Saturday/Sunday passes (deliberately not deduped);
settlement should read the LATEST pre-kickoff row per game (mirroring
`pick_refresh.final_pick_per_game`'s latest-revision rule), then score the
would-be side against outcomes at `decision_home_spread`, paired against the
played side, pushes excluded identically. The existing publish-time
challenger-ledger rows remain the Tuesday-grade arm of the same comparison;
the refresh ledger adds "what the refresh machinery would have done."

## The ONE predeclared confirmation look this wiring enables

Stated here BEFORE any 2026 data exists; do not run until the season
completes:

> After the 2026 regular season is fully settled (all Week 18 games graded),
> score the refresh-time would-be arm against the played arm ONCE: latest
> pre-kickoff overlay row per game, graded at the frozen Tuesday
> `decision_home_spread`, paired accuracy deltas in accuracy points, full
> slate, pushes excluded identically, week-blocked bootstrap primary /
> season-blocked secondary (same machinery and disclosure rules as
> `docs/nflcom_friday_refresh.md`). Classification follows the binding
> taxonomy: `probability_positive` is the reported quantity; an interval
> crossing zero NEVER grounds rejection; only a resolved wrong sign, zero
> split-half reliability, or a positive-control bound can close the line.
> This single look is the family's 2026 prospective confirmation; no
> intermediate mid-season peek is admitted, and no threshold re-tuning is
> admitted afterwards.
