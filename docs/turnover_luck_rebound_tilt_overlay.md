# Turnover-luck rebound tilt overlay: a no-window-cost prospective challenger

Written 2026-09-01. Follows the `surface_switch_tilt_overlay` /
`interim_hc_first_game_tilt_overlay` / `pbp08_protection_mismatch_tilt_overlay`
precedent (`docs/surface_switch_tilt_overlay.md`,
`docs/interim_coach_screen.md`, `docs/pbp08_matchup_screen.md`) for wiring a
pick-level, post-prediction transform into the prospective challenger ledger
at zero rotation-registry window cost.

## The registry cell, read before this module was built

`registry/weak_signals.json` key `close_game_luck_turnover_under_rebound`,
measured 2026-08-21 from `scripts/close_game_luck_screen.py`
(`artifacts/close_game_luck_screen/20260821T182234Z/results.json`, cell
`turnover_under_rebound`):

| Field | Value |
|---|---|
| Full-slate effect | **+0.4092 accuracy points** |
| Week-blocked 95% interval | `[-0.1526, +0.9692]` |
| `probability_positive` (week) | **0.92** |
| Season-blocked secondary 95% interval | `[+0.0297, +0.7949]` |
| `probability_positive` (season) | 0.981 |
| Sample | n=8,634 team-games, n_flag=2,036, 496 missing prior data |
| `sample_blocks` (week) | 294 |
| Seasons | 2009-2025 |
| Reliability (YoY Pearson, centered trait) | **0.1322**, 95% CI `[+0.0490, +0.2134]`, n=512 team-season pairs |
| Classification | `unresolved_below_power` |
| Registry note | "Correlated decomposition of turnover trait, not independent confirmation." |

## Direction check, stated up front

The flagged group covers **51.33%** against a **49.59%** field
(`subset_mean` / `complement_mean`, week-blocked, in the artifact's
`turnover_under_rebound` cell) -- exactly the REBOUND direction predeclared
before this cell was scored: a team that turned the ball over unluckily
last season tends not to repeat it, and its picks cover more often than the
field. This is a passed check, not an assumption.

## The interval crosses zero -- not grounds to decline

The week-blocked 95% interval is `[-0.1526, +0.9692]`. Per AGENTS.md, at
this evaluator's ~2-point resolution that is the EXPECTED shape for a
real-but-small signal, never grounds to reject, fail, or decline building a
no-window-cost prospective challenger. Neither admissible closing ground
applies here: the whole interval is not on the wrong side of zero (no
resolved wrong sign), and no positive control was run (no bound). So the
cell stays `unresolved_below_power`, and wiring it as a dual-tracked
challenger is an EV-positive play (`probability_positive` 0.92 > 0.5), not a
claim of a proven edge (AGENTS.md: "a promotion bar is not a decision bar").

## Reliability is low but positive -- what it does and does not imply

The underlying trait's year-over-year Pearson correlation (season-centered
`turnover_diff_per_game`) is **+0.1322**, 95% CI `[+0.0490, +0.2134]`,
n=512 team-season pairs -- entirely positive. This is a WEAK persistence: a
team's turnover luck this season only weakly predicts next season's, mostly
because turnovers really do regress hard toward the mean. A reliability this
low **attenuates** the measurable cover-rate effect (most, but not all, of a
team's turnover-luck signature washes out year over year) -- it does **not**
refute the mechanism. AGENTS.md's `no_split_half_reliability` closing ground
requires reliability to be exactly zero, which it is not; a low-but-positive
reliability is reported plainly here so the reader can judge how much
persistence to expect, not treated as a reason to decline.

## The dead mirror -- one asymmetric signal, not two

The sibling cell `turnover_over_fade` (top-quartile centered turnover
differential, predicted NEGATIVE/fade -- the SAME underlying trait's other
tail) reads full-slate **+0.0076 accuracy points**, `probability_positive`
**0.5008** -- a dead coin flip. So this is **one asymmetric signal**: the
bottom tail of the turnover-luck distribution moves; the top tail does not.
That is exactly why this overlay only ever flips a pick ONTO a bottom-
quartile team and never away from a top-quartile one -- there is no measured
direction for the fade case. The registry's own note on
`close_game_luck_turnover_under_rebound` already says this plainly:
"Correlated decomposition of turnover trait, not independent confirmation."
If either cell is ever pooled with another weak-signal input, neither should
be treated as independent confirmation of a second, different finding.

## The construct, transcribed VERBATIM (not re-derived) from `scripts/close_game_luck_screen.py`

- **Giveaways** (`build_giveaways_table`, lines 91-106): for REG-season
  plays only, `giveaways = interception + fumble_lost` per play, summed by
  `(game_id, posteam)`.
- **Takeaways** (`build_team_games`, lines 108-136): for each team-game,
  `takeaways` is the OPPONENT's `giveaways` in that same game.
- **Season aggregate** (`build_panel`, lines 139-163):
  `turnover_diff_per_game = (takeaways - giveaways) / games` per
  `(season, team)`.
- **"Centered"** (`build_panel`, lines 160-162): `league_mean` is that SAME
  season's mean `turnover_diff_per_game` across all teams
  (`panel.groupby("season")[trait].transform("mean")`), and
  `turnover_diff_per_game_centered = turnover_diff_per_game - league_mean`
  -- centering is PER-SEASON, never pooled across seasons.
- **Bottom-quartile cut** (`main`, line 369):
  `thresholds["turnover_q25"] = panel["turnover_diff_per_game_centered"].quantile(0.25)`
  -- a single POOLED quantile across the ENTIRE 2009-2025 panel (every
  team-season at once), measured at **-0.4026832217261905** and frozen here
  as a constant (AGENTS.md: "every overlay parameter must be the registry
  cell's own measured value, cited") -- never a per-season or expanding cut,
  and never re-derived or re-tuned by this module.
- **Prior-season lookup** (`_prior`, lines 193-198): the CENTERED value is
  shifted `season + 1` and joined back onto the following season's games by
  `team`, so a game in season *S* only ever sees the centered value computed
  from season *S-1*.

Implemented in `src/nfl_ats/turnover_luck_rebound_tilt_overlay.py`'s
`turnover_under_flag_by_game`, which ports the giveaways/team-game/panel
construction (turnover leg only -- `one_score_luck` and `takeaway_share`
belong to different, separately-registered cells) and reads from the newest
local schedule and play-by-play snapshots, never hand-typed.

## Why a prior-season aggregate is pregame-safe

The trait is a full PRIOR-season aggregate -- fully known before that
season's own Week 1 -- and the lookup only ever reads season *S-1* for a
game in season *S*. `turnover_under_flag_by_game` never reads the target
game's own outcome, `result`, or `spread_line` at all. Two leakage
regression tests in `tests/test_turnover_luck_rebound_tilt_overlay.py` prove
this empirically: the flag is unchanged when the CURRENT season's play-by-
play (including the target game's own events) is mutated, because the
function never even loads current-season plays into the panel the flag is
looked up from. A team with no observed prior season (an expansion team, or
any data gap) resolves to `False`, never an error.

## The rule, exactly as built (parameter-free, frozen, deliberately asymmetric)

```
home_flag, away_flag = turnover_under_flag_by_game(schedules, pbp)   # bottom-quartile prior-season centered turnover differential
both_flagged = home_flag AND away_flag
model_pick_home = home_cover_probability >= 0.5

flip to home when: home_flag AND NOT both_flagged AND NOT model_pick_home
flip to away when: away_flag AND NOT both_flagged AND     model_pick_home
```

In plain language: **when exactly one team is flagged AND the active
model's own forced pick is NOT that team, flip the pick ONTO the flagged
team.** REG season only (every measured read above was scored on regular-
season games). Both-flagged games are never touched -- no measured
direction for a mutual case, mirroring `coach_fade_overlay` /
`backup_qb_fade_overlay`'s clean-case handling -- and a pick already on the
flagged team needs no flip.

Implemented in `src/nfl_ats/turnover_luck_rebound_tilt_overlay.py`:
`turnover_under_flag_by_game` derives the flag; `apply_turnover_luck_rebound_tilt_overlay`
applies the rule at pick level; `overlay_disclosure_note` produces the
plain-English provenance sentence (not currently surfaced anywhere).

## The stacked-on-production read (mined-seasons, context not a gate)

`scripts/turnover_luck_rebound_stacked_backtest.py` layers this overlay on
top of the six already-registered `ACTIVE_PROSPECTIVE` overlays
(`coach_fade_overlay`, `injury_value_lost_tilt_overlay`,
`division_revenge_tilt_overlay`, `backup_qb_fade_overlay`,
`surface_switch_tilt_overlay`, `spread_gap_zone_fade_overlay`) OR-combined on
the frozen `artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`
baseline (1,537 REG games 2020-2025, production probability rule, baseline
53.36% on 1,503 scored games) -- the same "production proxy" diagnostic
`scripts/overlay_stack_backtest.py` already builds, NOT a literal replay of
`publishing.py`'s actual `four_overlay_composition`-composed card, which was
out of scope for this script's reuse list.

Measured 2026-09-01
(`artifacts/turnover_luck_rebound_stacked/20260901T195051Z/results.json`):
marginal delta of adding this tilt on top of that six-member stack is
**-0.865 accuracy points**, week-blocked 95% `[-2.170, +0.462]`,
`probability_positive` 0.0867; season-blocked secondary 95%
`[-1.484, -0.145]`, `probability_positive` 0.0050. n_flipped vs. the bare
baseline: 303 games (101 net-new beyond what the six-member stack already
flips, 202 already covered by it); 80 games left untouched as both-flagged.

**Verdict, per the binding closing-ground taxonomy.** Whole-interval-below-
zero on BOTH blockings is required for a resolved wrong sign; the week-
blocked interval's upper bound (+0.462) is positive, so that bar is not met
even though the season-blocked secondary is entirely negative. This reading
is recorded as `unresolved_below_power` in a NEW family
`close_game_luck_turnover_stacked_on_production` (registry key
`turnover_under_rebound_stacked_on_production`) -- **not** as a resolved
wrong sign, and the challenger registration is **not** declined on this
reading, per AGENTS.md: a negative point estimate whose interval crosses
zero is never grounds to reject.

## Why this construction, not a full retrained-model challenger

Same two reasons the sibling overlay docs give: (1) a full retrained
challenger dual-tracked against whatever model is currently active would
answer a different question than the isolated, prior-season-conditioned
construct measured above, and (2) the assigned pattern is the pick-level
tilt precedent, which touches zero training frames, zero feature profiles,
and zero stored model artifacts.

## What is and is not wired in

- `src/nfl_ats/turnover_luck_rebound_tilt_overlay.py`: the transform
  (`apply_turnover_luck_rebound_tilt_overlay`), the signal reader
  (`turnover_under_flag_by_game`), the disclosure sentence
  (`overlay_disclosure_note`, not currently surfaced anywhere), and the
  recorder (`record_turnover_luck_rebound_tilt_challenger_decisions`).
- `scripts/record_turnover_luck_rebound_challenger.py`: the standalone
  weekly recorder entry point. **This challenger's `weekly_recording_command`
  is NOT `nfl-ats publish-predictions --record-decisions`** -- wiring a
  seventh `try`/`except` block into `src/nfl_ats/cli.py` is off-limits to
  this build (a separate orchestrator integration pass owns that, so the
  live registry stays in sync with
  `tests/test_cli.py::test_publish_challenger_result_map_covers_live_active_registry`).
  Until that wiring lands, the standalone script is the weekly path.
- `artifacts/prospective/challengers.json`: registered as
  `turnover_luck_rebound_tilt_overlay`, status `ACTIVE_PROSPECTIVE`, `model`
  block a snapshot of the active configuration at registration time
  (fingerprint `bc77638d47e2748c`, verified against the live 2026 Week 1
  forecast artifact) for fingerprint-mismatch detection only.
- **Not wired anywhere:** there is no switch that applies the tilt to the
  published card. Playing this tilt for real is a separate owner decision
  this document does not make.
- **Tracked independently against the active model's own card, not stacked
  on the other overlays in production.** The stacked-on-production read
  above is a diagnostic backtest only, run outside the weekly recording
  path; the weekly recorder always transforms the SAME un-flipped active
  model card the other challengers read, exactly like every sibling
  overlay.

## Both grades, from Week 1

`nfl-ats prospective-score` settles every `ACTIVE_PROSPECTIVE` challenger,
including this one, at both grades automatically once picks are recorded:
`decision_line` (the spread the pick was actually made at -- primary) and
`close_line` (the same pick re-settled at the close -- secondary). No
challenger-specific scoring code was needed; this is the same generic
machinery the other overlay challengers already use
(`docs/prospective_evidence.md`).

**Week 1 2026 fires, as expected for a prior-season trait.** Applying the
overlay to `artifacts/margin_predictions/2026-week-01-20260824T120725Z`
(measured 2026-09-01): 8 of 32 team-slots are flagged (exactly 25%, matching
the registry's ~23.6% `fraction_of_slate`), but only **2 of 16 games**
actually flip -- `2026_01_CLE_JAX` (JAX -> CLE) and `2026_01_MIA_LV`
(MIA -> LV) -- because most other flagged teams were either already the
model's own pick (`2026_01_SF_LA`, `2026_01_GB_MIN`, `2026_01_WAS_PHI`,
`2026_01_DAL_NYG`) or part of the one both-flagged game
(`2026_01_NYJ_TEN`). This is exactly the intended shape: a prior-season
aggregate is fully known before Week 1, so it can and does fire immediately.

## Tests

`tests/test_turnover_luck_rebound_tilt_overlay.py`, mirroring
`tests/test_interim_hc_first_game_tilt_overlay.py`'s structure:

1. `turnover_under_flag_by_game`: the frozen threshold matches the screen's
   measured value; fires on the bottom-quartile prior-season teams and
   reproduces the screen's arithmetic on a controlled fixture; resolves to
   `False` (never raises) when a team has no prior-season data; raises on
   missing schedule/play-by-play columns; and TWO leakage regression tests
   proving the flag is unchanged by the current season's own turnover
   events, including the target game's own play-by-play.
2. `apply_turnover_luck_rebound_tilt_overlay`: flips onto the flagged team
   when not already picked, leaves a neutral game untouched, never flips a
   both-flagged game, leaves a pick already on the flagged team untouched,
   is a no-op when disabled, leaves postseason games untouched, and changes
   only `home_cover_probability` on flipped rows.
3. `overlay_disclosure_note`: empty when nothing flipped, states the flip
   count and matchups when something did, and explicitly says "not applied
   to the published card".
4. `record_turnover_luck_rebound_tilt_challenger_decisions`: records the
   tilt's own arm (which can diverge from the active model's raw pick), is
   append-only and idempotent, refuses outside the recording lock window,
   refuses a fingerprint mismatch (an active-model promotion under the
   challenger's feet), and refuses an inactive registration.

## Decision left to the orchestrator

Nothing here decides anything -- it only starts a free evidence stream. Once
2026 accrues enough weeks, `nfl-ats prospective-score` reports this
challenger's `probability_positive` at both grades, paired against the
active model, exactly like the other overlay challengers. That number is
new, independent evidence about the TILT rule specifically -- it neither
replicates nor substitutes for the registry cell above.
