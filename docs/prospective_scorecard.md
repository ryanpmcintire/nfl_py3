# Prospective evidence scorecards (ENG-06)

`nfl_ats/prospective_scorecard.py` + `scripts/prospective_scorecard.py`: one
settled-week REPORT per season (optionally capped at a week), covering the
active model and every challenger declared in
`artifacts/prospective/challengers.json`. It is read-only end to end -- it
never writes to `artifacts/prospective/` or `registry/`, and it never calls
`nfl-ats weak-signals record` / `nfl-ats rotation record-look`. Those two
commands are the only path to a verdict; this tool only reads the ledgers
those commands (and the weekly recorders that feed them) already wrote and
summarizes what has settled so far.

## Why it had to exist

`nfl-ats prospective-score` (`docs/prospective_evidence.md`) already settles
every recorded pick at both grades and reports accuracy with a week-blocked
interval. What it does not do in one place: coverage against the week's full
card, calibration, the marginal effect of a challenger's own disagreeing
picks, or the Tuesday-vs-refresh flip rate. Assembling those by hand from the
four separate ledgers (`artifacts/clv_ledger/decisions.parquet`,
`artifacts/prospective/challenger_decisions.parquet`,
`artifacts/prospective/pick_revisions.parquet`, and each week's
`recommendations.csv` card) is exactly the kind of task a session redoes
slightly differently every time. This tool is the one place that does it the
same way every time.

## Command

```powershell
.\.tools\uv.exe run --no-sync python scripts\prospective_scorecard.py --season 2025
.\.tools\uv.exe run --no-sync python scripts\prospective_scorecard.py --season 2026 --through-week 3 --json
```

Flags: `--season` (required), `--through-week` (caps the scope, inclusive;
default is the whole season), `--features` (default
`data/processed/game_features.parquet`), `--artifacts-root` (default
`artifacts/`), `--data-root` (default `data/`), `--out` (default
`artifacts/prospective_scorecards/<season>_<stamp>/`, an untracked/gitignored
path -- `artifacts/**` is gitignored except the explicitly un-ignored
`artifacts/prospective/challengers.json`), `--bootstrap-samples` /
`--bootstrap-seed` (defaults `2000` / `20260904`), `--registry-root` (ENG-33;
default is the tracked `registry/`, honouring `NFL_ATS_REGISTRY_DIR`;
read-only, see below), `--json` (also prints the full per-entrant JSON
payload to stdout).

Every run prints the Markdown table and writes `scorecard.md`,
`scorecard.json`, and `scorecard.csv` to the output directory. It never
writes under `artifacts/prospective/` itself (the two ledgers it reads live
there) and never touches `registry/`.

## What each row covers

One row per entrant (the active model, then every entry in
`challengers.json`, regardless of current status):

- **Coverage.** `games_on_card` (distinct games in the active model's own
  ledger for the scope), `games_recorded` (this entrant's own ledger),
  `coverage_ratio`. `None`, not a divide-by-zero, when nothing has been
  recorded yet.
- **Settled accuracy.** `accuracy_decision_line` (PRIMARY -- the line the
  pick was actually made at, per `docs/prospective_evidence.md`; this
  report treats it as the opener-equivalent grade) and
  `accuracy_close_line` (secondary), `settled_games`, `pushes`, `pending`.
  Reuses `nfl_ats.prospective_scoring.settle_prospective_picks` and
  `prospective_accuracy` -- the exact functions `nfl-ats prospective-score`
  uses -- rather than re-deriving push/pending handling.
- **The entrant's own accuracy interval** (`row["interval"]`): a
  week-blocked bootstrap over `prospective_accuracy_metrics`, the same
  metric/interval pairing `nfl-ats prospective-score` reports per entrant.
- **Paired delta vs. the active model** (`paired_vs_active`, challengers
  only): on games BOTH entrants recorded and settled, the accuracy-point
  delta, its week-blocked interval, and `probability_positive`.
- **Overlay marginal effect** (`overlay_marginal`, challengers only): the
  same paired-delta machinery restricted to the games where this entrant's
  pick actually differs from the active model's chain pick -- "the games it
  fired on". `docs/prospective_evidence.md` already frames challenger
  evidence this way ("the two arms disagree on 3 of 16 games ... which is
  where all of the paired evidence will come from"); this is that framing,
  computed automatically every week instead of read off a table by hand.
- **Refresh effect** (`refresh_effect`, active model row only): reads
  `artifacts/prospective/pick_revisions.parquet`
  (`nfl_ats.pick_refresh.load_pick_revisions`) for the Tuesday-vs-final-refresh
  flip count and the paired accuracy delta between the two, both sides graded
  against the same frozen `decision_home_spread` (a refresh changes the pick
  side, not the anchor line). A handful of challengers have their OWN
  dedicated refresh ledgers (see `scripts/lockday_verify.py`'s
  `DEDICATED_LEDGERS`); those are out of scope here and the challenger rows
  say so explicitly rather than silently reporting zero.
- **Calibration.** Brier score and reliability bins at
  `nfl_ats.reporting.calibration_table`'s existing bin width (10 bins),
  computed from each settled game's own recorded `home_cover_probability`
  (read back from the `recommendations.csv` card the ledger row points at --
  `forecast_artifact` for the active model, `source_artifact` for a
  challenger). A card that can no longer be found is excluded and the
  exclusion is reported (`games_with_recorded_probability` vs.
  `settled_games`), never silently dropped.
- **`classification`.** Always `unresolved_below_power` -- see below.
- **`registered_evidence`** (challengers only): the challenger's own
  `challengers.json` `status` and, when present, its registered
  `evidence.probability_positive` / `evidence.registry_verdict`. This is
  informational context from the pre-commitment declaration, never an input
  to this report's own classification.

## The binding invariant, encoded not just obeyed

Pasted into the module docstring verbatim, per AGENTS.md's requirement that
any code implementing this rule carry the full taxonomy:

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (2) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: report
> `probability_positive`, never the binary "contains zero".

`nfl_ats.weak_signals.CLASSIFICATIONS` has three admissible values. This
module NEVER emits the two terminal ones (`refuted_mechanism`,
`bounded_by_control`) on its own -- both require a registry-level judgement
(a predeclared closing ground, a split-half reliability check, or a proven
positive control) that a settled-week report does not perform and that only
`nfl-ats weak-signals record` / `nfl-ats rotation record-look` are allowed to
write. Every row this tool produces is therefore classified
`unresolved_below_power` regardless of which side of zero its paired-delta
interval falls on. `interval_crosses_zero` is reported alongside it for
transparency (`True`/`False`/`None` when there is not yet an interval to
read), but it never changes the classification -- that is the mechanical,
code-level encoding of the invariant, not a placeholder pending a smarter
rule. `settled_games` is always reported; no "games needed"/sample-size
target is ever computed anywhere in this tool.

## Reused interval machinery (nothing reimplemented)

Every interval and every `probability_positive` in this tool comes from
`nfl_ats.clv.week_blocked_bootstrap` (the same week-blocked percentile
bootstrap `nfl-ats prospective-score` uses), fed a small paired-delta
`metric_fn` local to this module. Block-count degeneracy is read with
`nfl_ats.estimation_variance.guard_block_count(..., on_degenerate="warn")` --
the same three-production-function convention the rest of the repo uses --
so a season with only a handful of settled weeks still gets a full report,
with `block_count_degenerate`/`block_count_message` naming exactly how
coarse the resampling is, rather than an omitted or suppressed interval.
Within-week game correlation is never estimated or padded (owner mandate):
the week is the whole bootstrap block, full stop.

## The live 2025 run

As of this writing, `artifacts/clv_ledger/decisions.parquet`,
`artifacts/prospective/challenger_decisions.parquet`, and
`artifacts/prospective/pick_revisions.parquet` do not exist in a fresh
checkout -- prospective evidence is a 2026+ mechanism
(`docs/prospective_evidence.md`), every challenger in `challengers.json` was
registered between 2026-08-16 and 2026-09-01, and the real Tuesday lock for
2026 Week 1 had not yet run as of the season-2025 read this doc was written
against. A `--season 2025` run therefore reports the active model and every
registered challenger with `games_on_card`, `games_recorded`, and
`settled_games` all `0`, `coverage_ratio` `None`, and `classification`
`unresolved_below_power` for every row -- which is the correct, literal
answer for a season no ledger has any rows for, not an error. Re-run the
command for the current numbers; this section deliberately does not
hard-code a table that will go stale the first week real rows land.

## ENG-33: closing-ground CANDIDATE detection (advisory only)

Two more fields per row, both purely advisory and both computed by
`nfl_ats.prospective_scorecard`, never by a registry write:

- **`closing_ground_candidate`**: `None`, `"wrong_sign_resolved"`,
  `"no_split_half_reliability"`, or `"positive_control_bound"` -- the exact
  strings `nfl_ats.weak_signals.CLOSING_GROUNDS` admits, reused directly
  rather than retyped. `wrong_sign_resolved` fires only when the row's WHOLE
  paired-delta week-blocked interval sits on the side opposite a predeclared
  sign, read from the challenger's own `evidence` in
  `artifacts/prospective/challengers.json` (a recorded `probability_positive`
  or `source_probability_positive` declares positive; a nonzero
  `effect_accuracy_points` / `source_effect_accuracy_points` /
  `paired_delta_points` declares its own sign; absent all of those, the field
  is `None` with reason `"no_predeclared_sign"` -- the honest answer for most
  live entries, which declare no single top-level numeric direction).
  `no_split_half_reliability` fires only when a split-half reliability
  computed on the entrant's own settled weeks has a bootstrap interval whose
  upper bound is at or below zero. `positive_control_bound` is always `None`
  here with reason `"no_positive_control_in_report"` -- this scorecard runs
  no positive control.
- **`next_admissible_action`**: one of the fixed six-item ENG-20 vocabulary
  (`run_unspent_window`, `run_reused_window_with_discount`,
  `test_on_production`, `run_candidate_sized_positive_control`,
  `record_pending_look`, `closed`) -- never `"wait"`. When a challenger's own
  cited weak signal (`evidence`'s `"registry/weak_signals.json:<name>"`
  references) already carries an admissible terminal closure, this is
  `closed`. When that signal's inferred family
  (`nfl_ats.weak_signals.signal_family`) is a declared rotation family, the
  decision delegates to the existing, tested
  `nfl_ats.research_queue.next_admissible_action` (its
  `test_on_top_of_production` / `run_positive_control` are translated to this
  report's `test_on_production` / `run_candidate_sized_positive_control`;
  the other four strings are identical). Otherwise: `record_pending_look` if
  the row already has settled shared prospective evidence (AGENTS.md:
  recording is the default action for a category-3 result), else
  `test_on_production` (the admissible next step is simply to keep it
  running as a live prospective challenger, which its `challengers.json`
  registration already does).

**Both fields are advisory only.** `classification` stays
`unresolved_below_power` on every row regardless of what either field says --
see the module docstring and `_closing_ground_candidate`/
`_next_admissible_action` in `nfl_ats/prospective_scorecard.py`. Neither
field is a verdict; only `nfl-ats weak-signals record` / `nfl-ats rotation
record-look` may act on one. This module reads
`registry/weak_signals.json` / `registry/rotation_registry.json` read-only
(`--registry-root` overrides the root for both, defaulting to the same
tracked `registry/` every other reader in this repo uses, honouring
`NFL_ATS_REGISTRY_DIR`) and never writes to either --
`tests/test_prospective_scorecard.py::test_registry_files_are_never_modified_by_a_scorecard_run`
asserts the byte content is unchanged after a run, and a real `--season 2025`
/ `--season 2026` run (2026-09-04, before the 2026 Week 1 lock) left both
files' mtimes and MD5s unchanged.

The Markdown table renders the two fields as `Closing-ground candidate`
(non-null values suffixed `(candidate)`) and `Next admissible action`.

The split-half reliability check has no existing week-level helper to reuse:
the two `split_half_reliability` implementations already in the repo
(`nfl_ats.durability_prior`, `nfl_ats.cfb_qb_dependence`) both key their two
halves on a *player* or *team-season* as the repeated-measure unit, neither
of which this per-week paired-accuracy-delta series has. Per ENG-33's own
fallback instruction, `_split_half_reliability` is a minimal Pearson
correlation instead: settled weeks are paired consecutively (week[0] with
week[1], week[2] with week[3], ...) -- weeks are the independent unit
throughout this module (within-week game correlation is treated as zero and
is never estimated or padded, per AGENTS.md/team memory) -- and the
correlation between the two positions across pairs is percentile-bootstrapped
for an interval.

**The live read (2026-09-04, before the 2026 Week 1 lock).** As documented
above, every ledger this module reads is still empty for both season 2025
and season 2026, so every row's `closing_ground_candidate` is `None` and
`next_admissible_action` is never `closed` in a real run today -- most rows
report `run_unspent_window` (their cited weak signal's inferred family
matches a declared rotation family with eligible blocks remaining;
`hc_year_one_fade_overlay` matches `docs/research_queue.md`'s own PER-07 row
exactly: "run_unspent_window -- hc_year_one_fade: draw the earliest eligible
close block") or `test_on_production` (no rotation family matched and no
prospective data yet); the active model row is always
`record_pending_look`. Re-run the command for the current numbers once real
ledger rows exist.

## Tests

`tests/test_prospective_scorecard.py` builds full-column synthetic ledgers
in `tmp_path` (using each real loader's actual on-disk column contract, not a
parallel one) and hand-computes every accuracy delta, flip count, and Brier
score it asserts on. It covers: coverage vs. the card, the paired delta and
`probability_positive` vs. the active model, the marginal effect restricted
to disagreement games, the refresh flip count (including "latest revision
wins" when a game was revised twice), calibration, the empty-ledger case (the
real 2025 shape), and an explicit unit test that an interval containing zero
-- and one that does not, on either side -- both classify
`unresolved_below_power`. A separate assertion scans the rendered Markdown,
JSON, and CSV output for the strings "failed", "rejected", "contains zero",
and "needs more" and asserts none of them ever appear.

ENG-33 adds unit tests on `_predeclared_sign`, `_referenced_signal_names`,
`_split_half_reliability` (a hand-built perfect-negative-correlation series
whose bootstrap interval upper bound is at or below zero, a perfect-positive
one that is not, and a too-few-pairs case), `_closing_ground_candidate`
(whole-interval-opposite-sign, interval-crosses-zero, zero-split-half, and
no-predeclared-sign cases -- each cross-checked against `_classification`
staying `unresolved_below_power` regardless), and `_next_admissible_action`
(active model, no-family/no-data, no-family/has-data, and a `closed` case
built from a real schema-valid `weak_signals.json` fixture, independently
re-verified against `weak_signals.TERMINAL_CLASSIFICATIONS` /
`CLOSING_GROUNDS` rather than trusted from the function's own output). A
language-guard test extends the forbidden-phrase checks above: `"closed"` may
appear as `next_admissible_action` only on a row whose cited signal the
fixture's registry actually closed admissibly; every other row is asserted
not to say `closed`. A full-pipeline test asserts `next_admissible_action` is
never `"wait"` on any row, and a byte-identity test asserts a scorecard run
never changes either registry file.
