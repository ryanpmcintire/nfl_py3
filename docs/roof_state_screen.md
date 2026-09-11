# ENV-02 roof-state (open/closed) screen: history, prediction, production

Lane ENV-02. The surface half of this row (grass-modal visitor on turf) was
already closed out 2026-08-19 (`surface_switch_tilt_overlay`) and is not
repeated here. The roof half was previously screened 2026-08-20
(`docs/roof_decision_screen.md`, realized roof state only, no live source) and
audited for a live T-90 feed 2026-09-02 (`docs/roof_decision_sourcing.md`,
verdict: no defensible public per-game feed exists for any of the five
venues). This session does not repeat either of those; it builds the piece
that was still open -- a PREGAME PREDICTOR of roof state built only from
information visible at the pick deadline (forecast + each venue's own
history), which sidesteps the "no live T-90 source" blocker entirely because
it never needs to observe the actual T-90 decision.

Every claim below is tagged **measured** (run this session, command/path
given), **read** (a file opened this session, path/line given), or
**inferred** (reasoning, not evidence), per the binding `AGENTS.md` labeling
rule.

## Binding closing-grounds taxonomy (restated verbatim, governs every verdict below)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
(whole interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that
size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry hard-rejects inadmissible closures; if
a record command errors, the verdict is wrong, not the validator. Never use
95%/0.90 as a decision bar -- decide on expected value. Within-week game
correlation is ZERO by owner mandate.

Every cell in this document is `unresolved_below_power`. None is closed.

## Predeclaration

Before running the production screen (Part 3), this doc's tilt direction was
frozen while declaring the rotation family (`nfl-ats rotation declare --name
roof_state_predicted_open_fade_on_production`, 2026-09-11, before the window
was assigned or any production number was computed): **fade the home team
when the roof is predicted open at a venue that usually closes** -- this
session's assigned fallback per the task brief, since the prior screen's own
evidence (`docs/roof_decision_screen.md`) disagreed in sign between the
close grade (P+ 0.83 favouring "home covers worse when open") and the
2020-2025 opener grade (P+ 0.27, i.e. leaning the other way), which does not
count as a usable predeclared direction. **Measured** this session: every one
of the five retractable venues has a lifetime open rate under 25% (ARI 8.0%,
ATL 24.7%, DAL 6.5%, HOU 10.8%, IND 22.3% -- see Part 1), so "usually closes"
is satisfied for all five and the flag reduces in practice to
`predicted_open == True` for any of them; this is disclosed, not hidden.

## Population

**Retractable-roof teams, re-measured this session (not re-quoted from the
2026-08-20 screen):** `data/raw/20260908T162105Z/schedules.parquet` (**measured**,
`sorted(glob("data/raw/*/schedules.parquet"))[-1]`), grouped by `home_team`,
shows exactly five home teams that carry BOTH `roof=='open'` and
`roof=='closed'` rows: ARI, ATL, DAL, HOU, IND. LV (Allegiant) and LA/LAC
(SoFi) are always `dome`, never `open`/`closed`. **MIN is NOT retractable**
(the task's own open question) -- **measured**: every MIN home game is either
`dome` (Metrodome/U.S. Bank Stadium) or `outdoors` (the 2014-2015 TCF Bank
Stadium stint and one 2013/2024 international game); MIN never shows
`open`/`closed` in this data. ARI shows one stray `outdoors` row (a 2022
Azteca Stadium "home" game played in Mexico City) and ATL shows two
(2014/2021 Wembley/Tottenham "home" games); filtering to
`roof in {open, closed}` naturally drops these, since they were never played
at the retractable stadium itself.

Restricting to `game_type == REG`: **n=626 games, 2009-2025** (**measured**,
identical to the 2026-08-20 screen's own count, an independent
cross-check): ARI 137 (126 closed/11 open), ATL 73 (55/18, 2017+ only, when
Mercedes-Benz Stadium replaced the fixed-dome Georgia Dome), DAL 138
(129/9), HOU 139 (124/15), IND 139 (108/31). Total closed=542, open=84.

## Part 1 -- roof-state history

### Coverage by season and venue (measured)

Format `season:closed/open`, from `data/raw/20260908T162105Z/schedules.parquet`:

| venue | by season |
|---|---|
| ARI | 2009:7/1 2010:8/0 2011:8/0 2012:8/0 2013:8/0 2014:8/0 2015:7/1 2016:8/0 2017:7/1 2018:8/0 2019:8/0 2020:8/0 2021:5/3 2022:5/3 2023:6/2 2024:9/0 2025:8/0 |
| ATL | 2017:7/1 2018:4/4 2019:5/3 2020:8/0 2021:3/4 2022:6/3 2023:5/3 2024:9/0 2025:8/0 |
| DAL | 2009:5/3 2010:6/2 2011:8/0 2012:7/1 2013:8/0 2014:8/0 2015:8/0 2016:8/0 2017:8/0 2018:8/0 2019:8/0 2020:8/0 2021:7/1 2022:8/1 2023:7/1 2024:9/0 2025:8/0 |
| HOU | 2009:5/3 2010:4/4 2011:5/3 2012:8/0 2013:8/0 2014:8/0 2015:8/0 2016:8/0 2017:6/2 2018:8/0 2019:8/0 2020:8/0 2021:8/1 2022:8/0 2023:7/2 2024:8/0 2025:9/0 |
| IND | 2009:4/4 2010:7/1 2011:6/2 2012:5/3 2013:6/2 2014:5/3 2015:5/3 2016:6/2 2017:7/1 2018:8/0 2019:6/2 2020:8/0 2021:7/2 2022:5/3 2023:6/3 2024:8/0 2025:9/0 |

**Measured**: the field is populated for every one of these 626 games back
to 2009 -- no coverage gap, matching the 2026-08-20 screen's own resolution
of the "NEW Feb 2020" nflverse changelog note (it marks when the column was
exposed, not when the value existed). All five venues went to (near-)zero
opens in 2024-2025 (ARI 0/0, DAL 0/0, HOU 0/0, IND 0/0, ATL 0/0) -- **read**
from the table above, **inferred**: consistent with either a genuine recent
policy tightening across all five franchises or a small-sample coincidence;
not investigated further here, out of scope for this screen.

Lifetime open rate by venue (**measured**,
`artifacts/roof_state_screen/screen_results.json` `venue_lifetime_open_rate`):
ARI 8.03% (11/137), ATL 24.66% (18/73), DAL 6.52% (9/138), HOU 10.79%
(15/139), IND 22.30% (31/139). Every venue is a closed-majority venue.

### Home cover rate, open vs closed, with intervals (measured)

Two constructions, both within the five retractable venues only (open vs
closed AT THESE VENUES, not vs the whole league -- a cleaner isolation of the
open/closed contrast than the 2026-08-20 screen's whole-league complement,
per this task's literal framing):

**Close grade** (full 2009-2025 history, `game_features.parquet`'s own
`home_cover`, n=614 graded of 626): open cover **44.44%** vs closed cover
**46.90%**. Gap (open minus closed): **-2.4599 accuracy points**,
week-blocked 95% **[-14.138, +9.422]**, `probability_positive` (of "home
covers MORE when open") **0.339**. Season-blocked secondary: same point,
[-13.254, +10.224], P+ 0.348. n_open=81 (3 pushes dropped), n_closed=533,
277 week blocks.

**Opener grade** (Tuesday-line pairing intersected with this population,
2020-2025, n=227 graded of 236 paired): open cover **55.17%** vs closed
cover **44.44%**. Gap: **+10.728 accuracy points**, week-blocked 95%
**[-11.232, +32.830]**, P+ **0.838**. Season-blocked: [-20.970, +36.715],
P+ 0.768 (317/20,000 draws dropped -- thin season-block population, only
6 seasons). n_open=29, n_closed=198, 100 week blocks.

**The two grades disagree in sign**, reproducing the 2026-08-20 screen's
own finding on an independent, within-venue-only construction rather than
just re-quoting it: at close grade the lean is toward home covering WORSE
when open; at opener grade -- the grade this project's forced-pick pool
actually settles against -- the lean flips to home covering BETTER when
open. Both individual intervals cross zero. Per AGENTS.md, neither
resolves the other; recorded as two separate `unresolved_below_power`
entries (`roof_state_home_cover_open_vs_closed_close`,
`roof_state_home_cover_open_vs_closed_opener`), not averaged together or
picked as "the" answer.

### Market-pricing check (measured)

For every open-roof game, paired against that same home team's OTHER
(closed-roof) REG home games in the same season, comparing home-perspective
`spread_line` (close grade; the only line available for every game, not just
the 2020-2025 opener-paired subset): mean diff **+0.1443 spread points**
(open game's line vs. that team's own closed-game average), 95%
**[-1.317, +1.428]**, `probability_positive` (market already shades toward
home when open) **0.427**. n=84 open games, all had at least one eligible
control; 14 season blocks. **No detectable market repricing** at this
sample size -- consistent with the snow-game screen's own market-pricing
finding (`docs/roof_decision_screen.md`/`snow_game_home_prep` precedent):
books do not appear to move the number specifically for these venues'
open/closed state.

### Split-half reliability, odd vs even seasons (measured)

`is_open` is a per-game situational condition (this week's roof decision),
not a persistent per-team trait with its own year-over-year value, so a
formal split-half correlation coefficient does not apply the way it would to
a team trait (same disclosed caveat this project already uses for the
`snow_game_home_prep` family). Reported instead: the close-grade open-vs-
closed cover gap measured independently on disjoint season halves.

- **Odd seasons** (n=51 open games): gap **-10.571 accuracy points**, 95%
  **[-24.337, +3.633]**, P+ **0.072** (leans toward "home covers worse
  when open").
- **Even seasons** (n=30 open games): gap **+10.752 accuracy points**, 95%
  **[-9.185, +30.373]**, P+ **0.857** (leans the OPPOSITE way).

**The two halves disagree in sign.** Neither individual interval excludes
zero, so this is NOT a resolved `no_split_half_reliability` refutation (that
ground requires a demonstrated near-zero reliability, not two noisy halves
that each still cross zero) -- it is the honest, inconclusive read: this
construct's reliability across a season split is, at best, weak, and at
worst genuinely near zero. Recorded as two more `unresolved_below_power`
entries rather than pooled into one number, per the family-declared-before-
signs-seen discipline.

## Part 2 -- predicting roof state from deadline-visible information

The realized `roof` field is not knowable at pick time (the decision is
announced ~T-90 to kickoff, per `docs/roof_decision_sourcing.md`'s primary-
source audit). The only PLAYABLE form is a pregame prediction. Built here:
a per-season **walk-forward logistic regression**, refit every season using
STRICTLY PRIOR seasons only (expanding window, chronological -- no season
ever trains on its own or a later season's games), on:

- `forecast_temp_f` and `forecast_precip_prob_pct` at the
  `pool_decision` cutoff (`data/raw/forecast_archive/pool_decision_2009_2025/forecasts.parquet`,
  decision timestamp = min(kickoff, Sunday 16:00 ET), per
  `docs/forecast_archive_build.md`'s 2026-09-02 pool-decision build -- this
  archive covers the FULL 2009-2025 window, not just 2020-2025 as the prior
  roof screen's weather data did; **measured**, 100% `fetch_status=='ok'` for
  all 626 retractable-venue games).
- each venue's own forecast temperature, interacted separately per venue (so
  a venue where opening correlates with WARMER weather and one where it
  correlates with COLDER weather can each learn their own sign), plus five
  venue dummy intercepts.

**Measured** (`nfl-ats`-independent per-venue temperature correlations,
before any model was fit, informing why a venue interaction was used rather
than one global temperature coefficient): ARI's forecast temp correlates
NEGATIVELY with opening (-0.147 -- Phoenix opens when it's COOLER, closes
for heat), while ATL (+0.329) and IND (+0.325) correlate POSITIVELY (they
open in mild weather, close in cold) -- a single global slope would get at
least one of these backwards.

Season 2009 has no prior season to train on and falls back to a fixed
"always closed" default (32 games, disclosed as `cold_start` rows, not
blended into the learned-model accuracy figure below).

### Accuracy (measured)

**Raw classification accuracy is not the informative number here** because
of the ~87% closed base rate: a trivial "always predict closed" baseline
already scores 86.6% overall, and the learned model's raw accuracy (84.7%
full population, 85.7% on the 594 walk-forward-only games excluding the 2009
cold start) sits slightly BELOW that trivial baseline -- it trades a few
false positives (calling "open" when the roof stayed closed) for catching
some real opens, which a plain accuracy count penalizes.

**The properly-powered read is the correlation between predicted and
realized state** (point-biserial / phi coefficient for two binary
variables): **+0.2769**, week-blocked 95% **[0.1726, 0.3822]**,
`probability_positive` **1.000** -- fully excludes zero. This is a resolved,
real relationship: the predictor has genuine skill at calling roof state
ahead of the T-90 decision, using only forecast + venue history.

By-venue detail (recall of actual OPEN games; n games / n actual open /
n predicted open): ARI 137/11/0 (recall 0% -- the model never calls ARI
open, plausibly too few historical opens for the regularized fit to trust
the inverted temperature signal there), ATL 73/18/19 (recall 50.0%,
precision 47.4%), DAL 138/9/3 (recall 11.1%), HOU 139/15/11 (recall 33.3%),
IND 139/31/33 (recall 38.7%, precision 36.4%).

## Part 3 -- production screen

Rotation family `roof_state_predicted_open_fade_on_production` declared
opener-graded (`nfl-ats rotation declare`), assigned window **[2020, 2021]**
(`nfl-ats rotation assign` -- the same default-size opener window every
other ENV/weather lane drew tonight, exactly as this task's brief preferred).
Flag: `predicted_open` from Part 2's walk-forward model (all five venues
qualify as "usually closes," so no separate venue gate is needed on top of
the flag). Rule: **flips a home baseline pick to away when flagged; never
flips an away pick to home** (the literal predeclared direction).

**Measured**, `scripts/roof_state_screen.py production`,
`artifacts/roof_state_screen/production_results.json`:

- 456 paired opener games / 35 weeks, active model `d49194e04945a5e5`.
- 7 games in this window were flagged `predicted_open`; only **2** of those
  had a home baseline pick to fade (the other 5 were already away picks, so
  the flag is a no-op there by construction).
- Candidate accuracy **52.851%** vs baseline **53.289%**: **-0.4386
  accuracy points**.
- Week-blocked (primary) 95% **[-1.094, 0.000]** -- touches zero at its own
  upper bound. `probability_positive` **0.0634**.
- Season-blocked (secondary) 95% **[-0.4545, -0.4237]** -- fully negative,
  but this is a **2-flip DEGENERATE artifact**, not resolved statistical
  power: with only 2 flipped picks, the season-blocked resample has almost
  nothing to vary, exactly the same shape this repo's own prior roof screen
  (`docs/roof_decision_screen.md` Cell 2 opener-grade, n=3) explicitly
  declined to read as `wrong_sign_resolved`. Applying that same standard
  here: **no admissible closing ground applies.**

Recorded `unresolved_below_power` (`nfl-ats weak-signals record`) and
`unresolved` (`nfl-ats rotation record`, which also spends the assigned
[2020, 2021] window). At n=2 flipped picks this result says almost nothing
either way about whether the idea helps; it is not evidence against
continuing to carry the predictor, only evidence that this particular
2-season window contained too few genuinely flagged home picks to move the
needle.

## Registry entries recorded

All via `nfl-ats weak-signals record --league nfl --family roof_state
--category environment` (two cells deviate from `--effect-units
accuracy_points` because the underlying metric is not a forced-pick
accuracy gap, disclosed in each entry's own description/notes):

| name | effect | units | interval | P+ | classification |
|---|---|---|---|---|---|
| `roof_state_home_cover_open_vs_closed_close` | -2.4599 | accuracy_points | [-14.138, +9.422] | 0.339 | unresolved_below_power |
| `roof_state_home_cover_open_vs_closed_opener` | +10.728 | accuracy_points | [-11.232, +32.830] | 0.838 | unresolved_below_power |
| `roof_state_market_pricing_check` | +0.1443 | ats_points | [-1.317, +1.428] | 0.427 | unresolved_below_power |
| `roof_state_home_cover_gap_odd_seasons` | -10.571 | accuracy_points | [-24.337, +3.633] | 0.072 | unresolved_below_power |
| `roof_state_home_cover_gap_even_seasons` | +10.752 | accuracy_points | [-9.185, +30.373] | 0.857 | unresolved_below_power |
| `roof_state_prediction_skill_walk_forward` | +0.27688 | correlation | [+0.17257, +0.38222] | 1.000 | unresolved_below_power |
| `roof_state_predicted_open_fade_on_production` | -0.4386 | accuracy_points | [-1.094, 0.000] | 0.0634 | unresolved_below_power |

Rotation: `nfl-ats rotation record --name
roof_state_predicted_open_fade_on_production --verdict unresolved` spent the
[2020, 2021] window (`registry/rotation_registry.json`).

`roof_state_prediction_skill_walk_forward`'s interval fully excludes zero
(P+ 1.000) -- a resolved POSITIVE finding about the predictor's own skill,
not a candidate being closed; it is classified `unresolved_below_power`
only because the registry's terminal classifications describe closing a
line of work, and this predictor is being used (it feeds Part 3), not
closed.

## Files

- `scripts/roof_state_screen.py` -- `screen` (Part 1), `predict-roof`
  (Part 2), `production` (Part 3). Never writes to `registry/`.
- `artifacts/roof_state_screen/screen_results.json`,
  `predict_roof_results.json`, `production_results.json`,
  `roof_state_population.parquet`, `roof_prediction_table.parquet`,
  `production_paired.parquet` -- full outputs.
- `registry/weak_signals.json` -- the 7 `roof_state_*` entries above.
- `registry/rotation_registry.json` -- the spent
  `roof_state_predicted_open_fade_on_production` window.
