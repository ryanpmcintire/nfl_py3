# Special-teams return top-quartile tilt overlay: a no-window-cost prospective challenger

Written 2026-09-01. Follows the `surface_switch_tilt_overlay` /
`interim_hc_first_game_tilt_overlay` / `pbp08_protection_mismatch_tilt_overlay`
precedent (`docs/surface_switch_tilt_overlay.md`, `docs/interim_coach_screen.md`,
`docs/pbp08_matchup_screen.md`) for wiring a pick-level, post-prediction
transform into the prospective challenger ledger at zero rotation-registry
window cost.

## The registry cell (read from `registry/weak_signals.json` before this module was built)

`special_teams_return_top_quartile` -- one of 8 predeclared cells in the
PBP-06 special-teams battery (`scripts/special_teams_screen.py`,
predeclaration `docs/special_teams_battery.md`, mined, uncorrected
multiplicity across the 8 cells). Flags teams whose PRIOR-season
`return_composite` (mean z of punt-return and kickoff-return yards) ranks in
the top quartile league-wide. Predicted POSITIVE on `team_covered` --
predeclared before any cover-rate sign was inspected.

| Grade | Population | Effect (accuracy pts) | 95% interval | `probability_positive` |
|---|---|---|---|---|
| Week-blocked (primary) | NFL REG 2009-2025, n=8,634 team-games | +0.4986 | [-0.0742, +1.0797] | 0.9547 |
| Season-blocked (secondary) | same | +0.4986 | [-0.0229, +1.0225] | 0.9690 |

`n_flag=2016`, `n_missing_required_data=496` (2009 games and any team's first
tracked season carry no trailing prior-season value and are reported as
missing, never dropped from the declared range).
Source: `artifacts/special_teams_battery/20260819T232856Z/results.json`.

**The interval crosses zero on both blockings. Per AGENTS.md, at this
evaluator's ~2-point resolution that is the EXPECTED shape for a real small
signal, never grounds to decline building a no-window-cost prospective
challenger.** Neither admissible closing ground applies -- the interval does
not sit entirely below zero (no `wrong_sign_resolved`), and no positive
control was run against this specific cell (no `positive_control_bound`) --
so it stays `unresolved_below_power`. `probability_positive` 0.9547 is far
above the 0.5 that makes playing it the favoured side of a forced pick
(AGENTS.md, "a promotion bar is not a decision bar").

## The reliability is low, and that is stated plainly, not hidden

Componentwise year-over-year Pearson reliability of the two legs that make up
`return_composite`:

| Leg | YoY Pearson r | 95% CI | n (team-season pairs) |
|---|---|---|---|
| `punt_return_yards` | +0.109 | [+0.019, +0.196] | 512 |
| `kickoff_return_yards` | +0.158 | [+0.073, +0.243] | 508 |

Both are **positive** and both intervals **exclude zero** -- the trait
persists across seasons, weakly. A low-but-positive, interval-excluding-zero
reliability attenuates a real effect toward zero; it does **not** refute the
mechanism, which under AGENTS.md's taxonomy would require zero split-half
reliability (a reliability whose interval includes, or sits at, zero). That
is not what was measured here. No promotion is implied by either number;
both travel with every use of this overlay, including the registration this
document describes.

## Battery-multiplicity caveat

`special_teams_return_top_quartile` is one of 8 cells (4 raw dimensions x
top/bottom quartile) predeclared together in the same battery. Its bottom-
quartile mirror (`special_teams_return_bottom_quartile`) and the other three
raw-dimension cells (`fg_oe`, `punt_net_yards`, and the four-dimension
`special_teams_composite_edge`) are correlated siblings sharing overlapping
windows and legs, not independent votes -- multiplicity across the battery is
uncorrected, exactly as `docs/pbp08_matchup_screen.md` documents for its own
four scheme/matchup cells.

## What this is not

The registry entry above spends no rotation-registry window -- it is a bias-
battery cell, not an opener-window confirmation run. This document and its
registration do not change that, and nothing here is an owner decision to
play the tilt on the real card. It is dual-tracked only.

## The construct, exactly as measured (ported, not redesigned)

Ported **verbatim** from `scripts/special_teams_screen.py`:

- `QUARTILE_TOP = 0.75` (`scripts/special_teams_screen.py:66`).
- `add_composites` (lines 126-145): pooled-sd z-score of each return leg's
  league-centered dimension (`sd = float(result[centered].std(ddof=1))`;
  `result[f"{dim}_z"] = result[centered] / sd if sd > 0 else np.nan`), then
  `return_composite_z = result[["punt_return_yards_z", "kickoff_return_yards_z"]].mean(axis=1)`
  (line 139 -- `pandas.DataFrame.mean` skips `NaN` by default, so a team
  missing one leg's centered value still gets a composite from the other leg
  alone).
- The top-quartile threshold is the `QUARTILE_TOP` quantile of
  `return_composite_z` over the WHOLE panel (`main()`, lines 322-333;
  "544-row 2009-2025 team-season panel"), never within-season.
- `_prior` (line 148): shift each team-season row's OWN `season` forward by
  exactly one before joining on `(team, season)`, so a row is only ever
  consulted as the PRIOR value for the season immediately after the one it
  describes.
- The league-centered dimensions the composite is built from
  (`punt_return_yards_centered`, `kickoff_return_yards_centered`) come from
  `scripts/special_teams_features.py::add_league_centered` (lines 351-358),
  itself built from `scripts/special_teams_features.py`'s season-by-season,
  never-persisted-in-full nflverse PBP aggregation (see that module's own
  data-source note).

Implemented in `src/nfl_ats/special_teams_return_tilt_overlay.py`'s
`return_composite_z_with_threshold` (the composite/threshold construction)
and `special_teams_return_flag_by_game` (the join), which reads the newest
local team-season snapshot
(`data/raw/special_teams/<snapshot>/team_season.parquet`) plus the newest
schedule snapshot, mirroring `pbp08_protection_mismatch_tilt_overlay`'s
stored-snapshot pattern.

**Reproduction check (measured 2026-09-01):** `return_composite_z_with_threshold`
run against the actual `data/raw/special_teams/20260819T232400Z/team_season.parquet`
snapshot reproduces the registry artifact's own stored threshold EXACTLY --
`0.4769479933229231` in both places
(`artifacts/special_teams_battery/20260819T232856Z/results.json`'s
`thresholds.return_composite_z.top`).

**Pregame-safe by construction.** The trait is a PRIOR-SEASON team-season
aggregate; a season's own row is NEVER used as that season's own prior, only
as the prior for the season immediately after it. The derivation never reads
`result`/`spread_line`/any outcome column at all -- team-season tables carry
no outcome columns to begin with. Two leakage regression tests
(`tests/test_special_teams_return_tilt_overlay.py`) prove this empirically:
mutating a team's CURRENT-season row never changes that same season's
already-computed flag, and a future season's team-season row never changes
an earlier season's already-computed flag.

**One accepted, stated dilution, mirroring `fg_oe`'s own documented
convention** (`scripts/special_teams_features.py` module docstring): the
top-quartile threshold is a quantile over the WHOLE available team-season
panel, so a team's own most-recent season contributes a small amount to the
very threshold its PRIOR season is compared against. With 17+ seasons and 32
teams pooled into one global quantile, one row's contribution is a small
fraction of the panel -- a materially smaller dilution ratio than `fg_oe`'s
own accepted season-local baseline (which pools only ~32 rows per season).

## The rule, exactly as built (parameter-free, frozen, REG season only)

```
home_flag = special_teams_return_flag_by_game(schedules, team_season).home_return_top_quartile
away_flag = ... .away_return_top_quartile
both_flagged = home_flag AND away_flag
home_pick = home_cover_probability >= 0.5

flip to home when: home_flag AND NOT both_flagged AND NOT home_pick
flip to away when: away_flag AND NOT both_flagged AND     home_pick
```

In plain language: **build the prior-season top-quartile `return_composite`
flag for both teams. If EXACTLY ONE team is flagged AND the active model's
own forced pick is NOT that team, flip the pick ONTO that team.** Both-
flagged games are NEVER touched -- mirroring
`interim_hc_first_game_tilt_overlay`'s `both_first_game_games` handling and
`coach_fade_overlay`'s `both_year_one_games` handling: a mutual case has no
measured direction to pick between and is reported separately, never
flipped. Missing prior-season data means `flag=False`, never an error.

Direction is PREDECLARED positive on `team_covered` -- back the elite return
unit -- so this overlay only ever flips ONTO the flagged side, never off it
(unlike the asymmetric-the-other-way `pbp08_protection_mismatch_tilt_overlay`,
which flips OFF a flagged offense).

Implemented in `src/nfl_ats/special_teams_return_tilt_overlay.py`:
`special_teams_return_flag_by_game`/`special_teams_return_flag_by_game_fail_open`
derive the flag; `apply_special_teams_return_tilt_overlay` applies the rule
at pick level; `overlay_disclosure_note` produces the plain-English
provenance sentence (not currently surfaced anywhere).

**FAIL-OPEN**, mirroring `pbp08_protection_mismatch_tilt_overlay` and
`interim_hc_first_game_tilt_overlay`: a missing
`data/raw/special_teams/*/team_season.parquet` snapshot yields zero flags and
a `RuntimeWarning`, never an exception. This overlay must never be able to
block a publish.

## The stacked-on-production back-test (a MINED-SEASONS read, context, not a gate)

`scripts/special_teams_return_stacked_backtest.py` applies this overlay's own
flip set on top of the active model's own opener-graded PRODUCTION picks --
`artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`, REG
2020-2025, 1,537 games, baseline 53.3599% on 1,503 scored games, graded under
the production probability rule -- and reports a paired
`nfl_ats.clv.week_blocked_bootstrap` delta (20,000 samples, seed 20260819).

| Grade | Candidate accuracy | Delta (accuracy pts) | 95% interval | `probability_positive` |
|---|---|---|---|---|
| Week-blocked (primary) | 53.6261% | +0.2661 | [-1.9920, +2.5743] | 0.5773 |
| Season-blocked (secondary) | 53.6261% | +0.2661 | [-1.1811, +1.8229] | 0.5877 |

`n_flipped=283` of 1,537 games (roughly a quarter of team-games league-wide,
matching the registry cell's `fraction_of_slate` of 0.2335 -- a large flip
count is a real, disclosed property of this rule, not a defect).
`n_both_flagged_untouched=74`. Result artifact:
`artifacts/special_teams_return_stacked/20260901T193949Z/results.json`.

**Both intervals cross zero and the point estimate is positive on both
blockings, so per AGENTS.md this is `unresolved_below_power`, not a closing
read.** It was recorded to `registry/weak_signals.json` under a NEW family,
`special_teams_return_stacked_on_production`
(`nfl-ats weak-signals record ... --classification unresolved_below_power`),
and the challenger is registered regardless -- an interval containing zero
is never grounds to decline.

## What is and is not wired in

- `src/nfl_ats/special_teams_return_tilt_overlay.py`: the transform
  (`apply_special_teams_return_tilt_overlay`), the signal reader
  (`special_teams_return_flag_by_game`/`_fail_open`), the disclosure
  sentence (`overlay_disclosure_note`, not currently surfaced anywhere), and
  the recorder (`record_special_teams_return_tilt_challenger_decisions`).
- `scripts/record_special_teams_return_challenger.py`: a standalone weekly
  recorder entry point, because this challenger's `weekly_recording_command`
  cannot be `nfl-ats publish-predictions --record-decisions` --
  `src/nfl_ats/cli.py` is off-limits to this build
  (`tests/test_cli.py::test_publish_challenger_result_map_covers_live_active_registry`
  asserts the CLI's challenger-result key map against the live registry, and
  wiring a new key into it is a separate integration pass). **Known gap:**
  CLI wiring into `nfl-ats publish-predictions --record-decisions` is the
  pending follow-up.
- `artifacts/prospective/challengers.json`: proposed entry, `challenger_id`
  `special_teams_return_tilt_overlay`, status `ACTIVE_PROSPECTIVE`.
- **Not wired anywhere:** there is no switch that applies the tilt to the
  published card. Playing this tilt for real is a separate owner decision
  this document does not make.
- **Tracked independently against the active model's own card, not stacked
  on the other overlays**, matching the existing pattern exactly.

## Week 1 2026 preview (dry-run only; no ledger row written)

Loaded `artifacts/active_ats_model.json` (model_id `d1f07d773475dc58`,
weekly forecast `margin_predictions/2026-week-01-20260824T120725Z`) and
applied the overlay against its 16-game Week 1 card (measured 2026-09-01).
Because the trait is a PRIOR-SEASON aggregate, it is fully known before
Week 1 and CAN fire in Week 1 -- and it does:

- **1 pick flipped:** `2026_01_DEN_KC` (DEN at KC). DEN's prior-season
  (2025) return composite is top-quartile (z=+0.666) and KC's is not; the
  model's own pick was on KC (`home_cover_probability` 0.5708, i.e. HOME).
  The overlay flips the pick to DEN.
- **2 both-flagged games, left untouched:** `2026_01_NE_SEA` (SEA z=+1.522,
  NE z=+1.358, both top-quartile) and `2026_01_NYJ_TEN` (TEN z=+1.269, NYJ
  z=+1.506, both top-quartile).
- **4 more exactly-one-flagged games where the model was ALREADY on the
  flagged side, so no flip was needed:** `2026_01_SF_LA` (SF flagged, model
  already picks SF), `2026_01_CLE_JAX` (JAX flagged, model already picks
  JAX), `2026_01_MIA_LV` (MIA flagged, model already picks MIA), and
  `2026_01_WAS_PHI` (WAS flagged, model already picks WAS).

## Both grades, from Week 1

`nfl-ats prospective-score` settles every `ACTIVE_PROSPECTIVE` challenger,
including this one, at both grades automatically once picks are recorded:
`decision_line` (primary) and `close_line` (secondary). No challenger-
specific scoring code was needed; this is the same generic machinery the
other overlay challengers already use (`docs/prospective_evidence.md`).

## Tests

`tests/test_special_teams_return_tilt_overlay.py`, mirroring
`tests/test_interim_hc_first_game_tilt_overlay.py`'s structure:

1. `return_composite_z_with_threshold`: reproduces an INDEPENDENT quantile
   computation on a controlled fixture (not just re-calling the function
   under test), and reproduces the live registry artifact's own stored
   threshold exactly when the real snapshot is present locally.
2. `special_teams_return_flag_by_game`: fires on the sole top-quartile
   prior-season side, fires on both sides of a mutual top-quartile game,
   is false when neither side qualifies, is false (never an error) with no
   prior-season row at all, raises on missing schedule columns, excludes
   non-REG games, and two leakage regression tests (a current-season row is
   never used as its own prior even when present in the panel; a future
   season's row never changes an earlier season's already-computed flag)
   plus an outcome-column-mutation-invariance test.
3. `apply_special_teams_return_tilt_overlay`: flips onto the sole flagged
   side, leaves a pick already on the flagged team untouched, never flips a
   mutual top-quartile game, has no effect outside the flagged population,
   treats missing prior data as no-flip/never-an-error, respects the
   REG-only gate, is a no-op when disabled, fails open with no snapshot, and
   changes only `home_cover_probability` on flipped rows.
4. `overlay_disclosure_note`: empty when nothing flipped, states the flip
   count and matchups when something did, and explicitly says "not applied
   to the published card".
5. `record_special_teams_return_tilt_challenger_decisions`: records the
   tilt's own arm (which can diverge from the active model's raw pick), is
   append-only and idempotent, fails open with no special-teams snapshot,
   refuses outside the recording lock window, refuses a fingerprint
   mismatch (an active-model promotion under the challenger's feet), and
   refuses an inactive registration.

## Decision left to the orchestrator

Nothing here decides anything -- it only starts a free evidence stream. Once
2026 accrues enough weeks, `nfl-ats prospective-score` reports this
challenger's `probability_positive` at both grades, paired against the
active model, exactly like the other overlay challengers. That number is
new, independent evidence about the TILT rule specifically -- it neither
replicates nor substitutes for the registry cell or the stacked-on-production
read above.
