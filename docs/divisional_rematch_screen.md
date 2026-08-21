# Within-season divisional rematch dynamics screen — predeclaration

Written **before** `scripts/divisional_rematch_screen.py` scored any cover-rate
outcome. Only population counts (n_flag sizes, threshold feasibility,
missing-data checks) were examined before this document was frozen — no cover
rate, gap, interval, or probability_positive for any cell was computed or
looked at beforehand. Method, population, blocking, seed, and predicted
directions are locked exactly as in `docs/primetime_cells_screen.md`.

## Binding closing-grounds taxonomy (verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. Every cell below is recorded regardless of sign.

## Overlap check (why these cells are new ground)

Checked **before** designing cells, per the session mandate:

- `bias_battery_division_revenge_game` and
  `bias_battery_division_revenge_game_opener` (**read** this session,
  `registry/weak_signals.json`; construct ported verbatim in
  `src/nfl_ats/division_revenge_tilt_overlay.py` and documented in
  `docs/division_revenge_tilt_overlay.md`): the UNPLIT revenge side — "2nd
  meeting this season vs. same opponent; team lost the 1st meeting" — backed
  in the rematch, no venue, margin, or timing conditioning. **The unsplit
  cell is deliberately NOT re-recorded here.** This screen's cells B1/B2/D1/D2
  are conditioning splits of that parent construct (first-meeting venue;
  rematch timing) and cell A conditions on first-meeting margin on the
  WINNER side — each a distinct flag, but all correlated with the parent.
  **Never pool any of them with the parent entries as independent.**
- `bias_battery_post_blowout_win_letdown` / `_loss_bounce` (**read** this
  session, `registry/weak_signals.json`): team's immediately preceding game
  ANY opponent, win/loss by >=17 raw points. Cell A is distinct by
  construction: the blowout must have come against THE SAME opponent in this
  season's FIRST meeting, the flag fires in the REMATCH only, the threshold
  is >=14 (frozen before scoring), and the direction claim is about fading
  the winner specifically. Correlated in spirit; not a subset.
- `era_trend_division_revenge_game` (**read** this session): season-trend
  drift of the unsplit revenge construct across eras. Cell D splits WITHIN
  season by rematch week — a different axis entirely.
- No other `rematch_*`, blowout-in-rematch, or revenge-venue-split name
  exists in `registry/weak_signals.json` (**measured**, name scan of all 341
  signals this session).

## Data source and leakage posture

- Newest snapshot `data/raw/20260817T235649Z/schedules.parquet` (**read**
  this session; same snapshot the primetime/body-clock screens use).
- Population: REG 2009-2025, pushes/missing-spread dropped via
  `nfl_ats.features.add_ats_outcomes`, one row per team-game (long table),
  canonicalized with `TEAM_ABBREVIATION_ALIASES`.
- Rematch identification: within `(team, opponent, season)`, games sorted by
  `gameday`; `meeting_rank >= 1` is a rematch; `first_margin`,
  `first_is_home`, `first_week`, `first_total` come from that group's rank-0
  row via a strict `gameday` ordering. **Pregame-safe by construction**: a
  rematch row's flags depend only on the outcome/venue/week of a STRICTLY
  EARLIER meeting — never the current game's own result, never a later
  meeting (same leakage posture as `division_revenge_side_by_game`, which
  carries two empirical leakage regression tests).
- Schedule/results facts only (`result`, `total`, `week`, home/away design).
  No line movement, no model outputs.

## Population diagnostics measured pre-freeze (counts only)

8,634 team-game rows (4,431 REG games, 114 pushes/missing dropped); 1,552
rematch team-game rows across 776 games; **0** rematch games with
`div_game != 1` (the meeting-count logic alone reproduces the divisional
framing, matching `docs/division_rematch_screen.md`'s sibling note in
`docs/division_revenge_tilt_overlay.md`); 12 rematch rows follow a tied
first meeting (no revenge side either way, excluded from every revenge flag
by construction). Cell counts: A n=269; B1 n=412; B2 n=358; D1 n=4; D2
n=628. D1's n=4 is known-small pre-freeze and recorded anyway per the
taxonomy (an underpowered cell is category 3, not a deletion).

## Derived quantities (frozen)

- `BLOWOUT_MARGIN = 14` (cell A), `EARLY_WEEK_MAX = 6`, `LATE_WEEK_MIN = 12`
  (cells D1/D2) — frozen before scoring.
- Cell C is DESCRIPTIVE ONLY: game-level OLS of rematch |margin| on
  first-meeting |margin| and rematch total on first-meeting total, plus
  Pearson r. **No ATS claim is made or recorded for cell C** (predeclared:
  no record line unless a future session predeclares one).

## Method (reused verbatim from `scripts/primetime_cells_screen.py` /
`scripts/nfl_bias_battery_screen.py`)

- Value = `team_covered`; subset-vs-complement full-slate-scaled effect:
  `(subset_cover − complement_cover) × 100 × fraction_of_slate`, population
  = full slate (complement = every non-flagged team-game), matching the
  battery convention so effects are comparable across screens.
- Week-blocked joint bootstrap primary (block = `season*100+week`),
  season-blocked secondary (block = `season`), same
  `block_bootstrap_two_group` algorithm.
- **20,000 samples, seed 20260821** (mandated).
- `probability_positive` = fraction of bootstrap draws favouring the
  predeclared direction (sign applied before the >0 test).
- Era splits (2009-2017 vs 2018-2025) scored for EVERY cell in the artifact.

## The 5 recorded cells (directions frozen before scoring)

| # | name | flag | predicted direction |
|---|---|---|---|
| 1 | `divisional_rematch_blowout_winner_fade` | rematch AND own first-meeting margin >= +14 (this side WON game 1 by 14+) | **−1** — film + motivation reversal: the loser fixes what broke, the winner coasts; fade the blowout winner in the rematch |
| 2 | `divisional_rematch_revenge_road_loser` | rematch AND lost game 1 AND game 1 was ON THE ROAD | **+1** — revenge spot with the venue flip coming home: the classic construction |
| 3 | `divisional_rematch_revenge_home_loser` | rematch AND lost game 1 AND game 1 was AT HOME | **+1** — weaker form of the same spot (no venue flip); direction frozen to match the parent construct's sign, NOT re-picked |
| 4 | `divisional_rematch_revenge_early_w1to6` | revenge side AND rematch week <= 6 | **+1** — grudge fresh, first-meeting film still current |
| 5 | `divisional_rematch_revenge_late_w12plus` | revenge side AND rematch week >= 12 | **+1** — playoff-seeding stakes amplify the spot |

Cells 2/3 partition the parent revenge flag by venue; cells 4/5 are timing
subsets of it. Their signs inherit the parent's frozen positive direction —
none is chosen after seeing outcomes.

## Recording commitment

Every cell above records to `registry/weak_signals.json` via
`nfl-ats weak-signals record` as `unresolved_below_power`, `league=nfl`,
`effect-units=accuracy_points`, seasons 2009-2025, regardless of interval
shape, with `probability_positive` reported (never "contains zero"). Cell C
records nothing. Exact command lines are returned by the session, not run
against the registry JSONs (which this task does not write).

## Results

All numbers below are **measured** this session from
`artifacts/divisional_rematch_screen/20260821T190026Z/results.json` (run log
stamped to `registry/experiments/divisional-rematch-screen/20260821T190026Z.json`),
produced by `scripts/divisional_rematch_screen.py` on snapshot
`data/raw/20260817T235649Z/schedules.parquet`, 20,000 samples, seed 20260821,
week-blocked primary. Every cell is category 3 `unresolved_below_power` per
the taxonomy above — none is closed, and no interval shape below was treated
as a rejection.

| # | cell | n | full-slate effect | week-blocked 95% | P+ | season-secondary P+ |
|---|---|---|---|---|---|---|
| 1 | `rematch_blowout_winner_fade` | 269 | **−0.0418 pts** | [−0.2106, +0.1274] | 0.2962 | 0.2431 |
| 2 | `rematch_revenge_road_loser` | 412 | **+0.0730 pts** | [−0.1438, +0.2865] | 0.7281 | 0.7611 |
| 3 | `rematch_revenge_home_loser` | 358 | **+0.1087 pts** | [−0.1183, +0.3386] | 0.8220 | 0.8044 |
| 4 | `rematch_revenge_early_w1to6` | 4 | **+0.0116 pts** | [−0.0232, +0.0232] | 0.7940 | 0.7884 |
| 5 | `rematch_revenge_late_w12plus` | 628 | **+0.0999 pts** | [−0.1732, +0.3723] | 0.7539 | 0.7752 |

Read-through (**inferred**, my reasoning — not evidence):

- Cell 1's point estimate sits against the predeclared fade direction (the
  blowout WINNER covered more often than the slate, raw gap +1.343 pts for
  the winner side) but only 29.6% of draws favour the fade — unresolved, not
  refuted: the scaled interval does not sit wholly on the wrong side of zero,
  so `wrong_sign_resolved` is inadmissible and nothing here closes the line.
- Cells 2/3 both lean the parent construct's way; the DESCRIPTIVE ordering
  (home-loser +0.1087 > road-loser +0.0730, overlapping intervals) runs
  OPPOSITE to the venue-flip intuition — noted, not adjudicated.
- Cells 4/5: the timing split shows no clean separation (late +0.0999 vs
  early +0.0116, but early has n=4 and a degenerate ±0.0232 band — its
  interval width reflects the count, not evidence of absence).
- Era splits are in the artifact for every cell; the largest era instability
  is cell 2 (+0.1641 in 2009-2017 vs −0.0253 in 2018-2025).

Cell C (**measured**, same artifact, descriptive only, NO ATS claim): across
776 rematch games, rematch |margin| on first-meeting |margin| slopes 0.0667
(Pearson r 0.0619); rematch total on first-meeting total slopes 0.0886
(r 0.0876). First-meeting scoring explains almost none of the rematch's —
strong regression toward the mean, which is the mechanism cell 1's fade
hypothesis was reaching for; the descriptive read does not support an ATS
claim and records nothing.

Correlation disclosure (binding): cells 2/3/4/5 partition or subset the
already-recorded unsplit revenge construct (`bias_battery_division_revenge_game`,
`_opener`, live as `division_revenge_tilt_overlay`); cell 1 is thematically
adjacent to `bias_battery_post_blowout_win_letdown`. None of these five may
be pooled with those entries as independent evidence.
