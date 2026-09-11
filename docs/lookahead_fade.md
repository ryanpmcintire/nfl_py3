# LEAD-44: CFB sandwich/lookahead fade, and its NFL analogue

Written before either construct below is scored. Frozen at this point:
mechanism, predeclared direction, population, encoding, and metric. Results
are appended under each construct's own "Results" heading after the one look
each takes.

## Closing-grounds taxonomy (binding, restated verbatim per AGENTS.md)

An interval or CI that contains zero is **NEVER** grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) **refuted mechanism** -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
**bounded by a positive control** -- the instrument was PROVEN able to detect
an effect that size and it was absent. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`,
report `probability_positive`, never the binary "contains zero". The
registry code hard-rejects inadmissible closures; if a record command errors,
the verdict is wrong, not the validator. A promotion threshold governs only
what the docs may CLAIM; it never governs which card is PLAYED, which is
expected value, full stop.

## Mechanism (ROADMAP LEAD-44, predeclared)

AP-ranked teams hosting an unranked opponent the week before a rivalry or
top-10 game play flat -- the classic lookahead/sandwich trap. **Predeclared
direction: FADE the ranked/favored host** in that sandwich spot.

## The data gap, and how this session resolves it differently from 2026-09-05

**Read**, `docs/cfb_lead_screens_wave1.md` section 4 and `ROADMAP.md`'s
LEAD-44 row: a 2026-09-05 pass found no local AP Top-25 poll table under
`data/cfb` (its thirteen directories -- `draft_picks`, `espn_betting`,
`lines`, `participants`, `pbp`, `portal`, `recruiting_players`,
`recruiting_teams`, `returning_production`, `rosters`, `schedules`,
`team_info`, `usage` -- carry no polls) or `data/raw`, and skipped the lead
under the no-fetch rule. **Measured this session**: that is still true --
no rankings/poll directory exists anywhere on disk. This session does not
fetch one. Instead, per this lane's own task brief, "ranked" is derived from
**pregame market-line strength**, a labelled, disclosed proxy, not a
replacement for a real AP table. Every number below inherits that proxy's
imprecision; this is flagged inline, not just here.

## Shared proxy construction (both leagues)

**Team power rating** (`market_power_rating`): for team T entering its game
in week w of a season, the trailing expanding mean of T's own
market-implied signed margin over T's STRICTLY PRIOR games in that same
season only (never carried across seasons, mirroring how AP rankings reset
each year). Per game, T's signed margin is the game's home-oriented spread
line if T was the home team, and its negation if T was away (positive means
the market favored T by that many points). A team's first game of a season
has no rating (NaN, excluded, never treated as 0 -- the project's standing
missingness convention). This is a deterministic transform of pregame market
lines and schedule identity, never of a score, so it carries zero outcome
leakage; it is, however, a coarser instrument than a real poll -- it cannot
distinguish "ranked because voters like this team" from "ranked because this
team has beaten a soft schedule," and the CFB leg in particular inherits
whatever conference-strength confound that implies. Disclosed, not fixed
here.

**"Ranked" / "top-X" cutoffs** are cross-sectional, computed separately for
each league-season-week from every team's rating available that week
(pooling both home- and away-perspective ratings from that week's games),
never from a fixed all-time cutoff. CFB uses two cutoffs (a top-quintile
"ranked" bar approximating AP Top 25 of ~130 FBS teams, and a stricter
top-8th-percentile "top-10-caliber" bar for the next-week-opponent check);
the NFL construct uses one top-quartile bar (top 8 of 32), matching this
lane's own task brief verbatim.

**Next-game lookahead** is read directly off each league's own schedule
identity (team, season, week order) -- never off that next game's own
market line or outcome, so identifying "next week is a marquee spot" cannot
leak anything about the CURRENT (sandwich) game's result. Where "next
opponent is ranked/top-quartile" is checked, it uses that opponent's OWN
trailing rating entering ITS next game (built from games strictly before
that next game, so still zero-leakage relative to the sandwich game itself).
A team with no further game that season (end of schedule) is scored as
having no lookahead risk (flag 0 / matched-control side), not excluded --
there genuinely is no sandwich trap when there is no next game.

## Construct 1: CFB sandwich/lookahead fade

**Population.** `data/processed/cfb_game_features.parquet`
(**read**, `docs/cfb_data.md` line ~105-130 for its build contract),
restricted to `nfl_ats.cfb_benchmark.CFB_CLEAN_CORE_SEASONS`
(2012-2019, 2021-2025) for the scored comparison; the full 2006-2025 table
feeds the trailing rating and the rivalry-pair identity (a schedule/history
fact, not a game outcome -- the same hindsight-identity argument
`docs/cfb_lead_screens_wave1.md`'s rivalry lead already uses for its
8-consecutive-season pair definition, reused verbatim here on this table's
own 2006-2025 span rather than re-derived from the raw 2001-2024 schedule
partitions).

**Grade.** CFB is close-graded (`spread_line`, the median-across-books close
proxy) -- **read**, `docs/cfb_lead_screens_wave1.md` line 48: "no verified
CFB opener exists." This settles no NFL play/no-play decision by itself.

**Eligible population (matched design).** Host is "ranked" (trailing rating
in the top quintile of that season-week's field) AND visitor is "unranked"
(trailing rating below that same quintile bar), both ratings defined
(not each team's season opener). Within this eligible set only:

- `cfb_lead44_sandwich_flag = 1` iff the host's own next scheduled game
  (any week, so a bye is skipped correctly, not miscounted as "no lookahead
  game") is EITHER against a rivalry-pair opponent (8-consecutive-season
  proxy) OR against an opponent whose own trailing rating (entering that
  next game) clears the stricter top-8th-percentile "top-10" bar.
- `cfb_lead44_sandwich_flag = 0` otherwise (including hosts with no further
  game that season).

**Metric.** Two pick rules scored on the SAME eligible population: Rule A
("always pick the ranked host to cover") and Rule B (identical to A except
flip the pick to the visitor whenever `cfb_lead44_sandwich_flag = 1`). The
predeclared direction implies Rule B beats Rule A. Per-game value =
`100 * (correct_B - correct_A)` (0 when flag=0; `100*(1 - 2*home_cover)` when
flag=1); accuracy_points. Also reported: raw home-cover rate for the
flag=1 and flag=0 subsets separately, each with its own week-blocked
interval.

**Uncertainty.** Week-blocked (season+week clusters) and season-blocked
block bootstrap, 2,000 resamples, seed 20260910, on the per-game value
column; `probability_positive` via
`nfl_ats.evidence_conventions.probability_positive_from_draws` (P(>0) +
0.5*P(==0)), never the binary "contains zero" read.

**Reliability.** This is a per-game situational construction (this week's
schedule position), not a persistent per-team trait -- the same category
the existing registry entry `bias_battery_sandwich_spot_opener` (**read**,
`registry/weak_signals.json`) already logs as "not_applicable ... a
per-game situational condition ... nothing to split-half." Classical
split-half reliability is therefore not directly applicable; the closest
analogous check reported here is an odd-season-vs-even-season split of the
same Rule B-minus-A delta on the eligible population, disclosed as a
directional-agreement check, not a trait reliability coefficient.

**No fetch, no NFL window spent.** This construct spends no rotation window
(CFB is free ground per rule 8 of `docs/rotation_registry.md`) and pools
with no other CFB family.

## Construct 2: NFL lookahead-favorite fade (deadline-visible analogue)

**Population.** The paired Tuesday-opener archive already scored against the
CURRENT active model this session -- **measured**,
`artifacts/opener_evaluation/20260910T211255Z/per_game.parquet` (1,537 REG
games, 2020-2025, model_id `d49194e04945a5e5`, matching
`artifacts/active_ats_model.json`'s active model, confirmed via
`nfl_ats.public_board.find_matching_opener_evaluation`). Home/away team
identity is read from each row's `game_id` (nflverse convention
`season_week_away_home`); division-game identity (`div_game`) is joined
from `data/raw/<latest>/schedules.parquet` by `game_id`.

**Eligible population.** Home team favored by 7+ points at the Tuesday
opener (`tue_open_home_spread >= 7.0`) -- exactly this lane's task brief,
no rating needed for the host side (the spread already establishes it).

- `nfl_lead44_lookahead_flag = 1` iff the host's next scheduled game in this
  same archive is EITHER a division game (`div_game = 1`) OR against an
  opponent whose trailing rating (entering that next game, same
  construction as the CFB leg) clears the top-quartile bar for that week's
  field.
- `nfl_lead44_lookahead_flag = 0` otherwise (including hosts with no further
  game in the archive that season).

**Metric 1 (matched cover rates).** Home-cover rate (`margin_vs_open > 0`)
for the flag=1 and flag=0 subsets of the eligible population, each with its
own week-blocked interval.

**Metric 2 (production screen).** Candidate = the active model's own opener
probability-rule pick (`pick_home_at_open_probability_rule`,
**read** from the per_game artifact) EXCEPT forced to the away side whenever
`nfl_lead44_lookahead_flag = 1`; baseline = the unmodified production pick.
Per-game value = `100 * (correct_candidate - correct_baseline)`,
accuracy_points, computed over the FULL 1,537-game archive (matching this
project's standing `*_on_production` convention -- e.g.
`road_fav_big_fade_on_production`, `division_dog_on_production` -- of
reporting the full-population effect of a pick override that only touches
the flagged subset). Reported per season and pooled.

**Rotation window.** Declared and assigned via `nfl-ats rotation
declare`/`assign` BEFORE this metric is scored, grade `opener`
(2020-2025 pool). Per this lane's task brief, the [2020, 2021] window is
preferred if the registry offers it. The registered confirmatory look uses
ONLY that assigned window; the same metric computed on the full 2020-2025
archive is reported alongside as disclosed, unregistered context (no
additional window spent), exactly the pattern MOD-07 and the public-claim
battery already use for a broader companion read.

**Reliability.** Same per-game-situational caveat as the CFB leg. An
odd-season/even-season split of the production-screen delta is reported on
the full 2020-2025 archive (the assigned window alone is too few seasons for
a meaningful split); disclosed as a directional-agreement check, not a trait
reliability coefficient.

## Decision rule (binding, restated)

Expected value, never a 0.90 promotion threshold, decides what would be
PLAYED. `probability_positive` is reported for every cell; an interval that
contains zero is `unresolved_below_power` and is recorded, not discarded.

## Construct 1 results (2026-09-11)

**Measured**, `scripts/lookahead_fade_screen.py --league cfb`, artifact
`artifacts/lookahead_fade_screen/20260911T031611Z/results.json`.

Eligible ranked-host/unranked-visitor population: **1,622 games**. Scored
(clean-core seasons only): **1,185**; **681 flagged** sandwich spots, **504**
matched no-lookahead controls. Rivalry-pair identity found **332 pairs**
across the 2006-2025 table.

Rule B (fade the host only in sandwich spots) minus Rule A (always back the
ranked host): **+1.112 accuracy points**, week-blocked 95%
**[-2.967, +5.005]**, `probability_positive` **0.7195** (178 week-blocks);
season-blocked **[-2.562, +4.697]**, P+ **0.7025** (13 season-blocks). Both
intervals contain zero -- `unresolved_below_power`, not a rejection.

Raw home-cover rate: flagged **49.03%** [45.55, 52.42] (n=671) vs matched
control **52.81%** [48.10, 57.38] (n=498) -- same direction as the Rule B
delta (ranked hosts cover less in sandwich spots), each interval its own
week-blocked read, not a separately bootstrapped joint gap.

Reliability (odd/even-season split of the Rule B-minus-A delta; per this
lane's caveat above, this is a directional-agreement check, not a trait
reliability coefficient): odd seasons **-0.485 pts** [-6.616, +5.355] P+
**0.455** (n=619); even seasons **+2.909 pts** [-2.072, +7.891] P+ **0.875**
(n=550). Both cross zero; the halves lean in different directions, which is
consistent with a real-but-small or absent effect at this sample size, not
with a resolved finding either way.

Recorded: `nfl-ats weak-signals record --name
cfb_lead44_sandwich_fade_on_benchmark --league cfb --classification
unresolved_below_power --effect-units accuracy_points --family lookahead_fade
--category schedule ...` (full command and output in the session report;
registry total after recording: 5,993 signals). No CFB rotation window is
spent (CFB is free ground, rule 8).

## Construct 2 results (2026-09-11)

**Measured**, `scripts/lookahead_fade_screen.py --league nfl`, artifact
`artifacts/lookahead_fade_screen/20260911T031622Z/results.json`, scored
against the active model (`d49194e04945a5e5`) via
`artifacts/opener_evaluation/20260910T211255Z/per_game.parquet`.

**Rotation.** Family `lookahead_fade_on_production` declared 2026-09-11
(grade `opener`, `acknowledges_mined_2018_2025=true`), assigned window
**[2020, 2021]** (the preferred window per this lane's task brief; the
registry offered it on the first `assign` call). 99 eligible games (home
favorite 7+ at the opener) in that window: 49 flagged (marquee next game),
50 matched control; 49 of the window's 456 forced picks changed under the
override.

**Registered look (window [2020, 2021], 456 games / 35 weeks).**
Production-screen delta: **+0.439 accuracy points**, week-blocked 95%
**[-0.673, +1.584]**, P+ **0.7625**. Season-blocked is degenerate at only 2
blocks (**[0.000, +0.909]**, lower bound touches exactly zero -- a lean, not
a resolution, the same standard this project's own `week1_dog_on_production`
entry uses for a zero-touching bound) and is not treated as resolving
anything. Per season: 2020 **+0.909 pts** [-1.310, +2.913] P+ 0.798 (n=220);
2021 **+0.000 pts** [-1.255, +1.245] P+ 0.517 (n=236).

Recorded via `nfl-ats rotation record --name lookahead_fade_on_production
--verdict unresolved --probability-positive 0.7625 ...` (window now spent)
and via `nfl-ats weak-signals record --name
nfl_lead44_lookahead_fade_on_production_window_2020_2021 --classification
unresolved_below_power ...` (registry total after recording: 5,994).

**Disclosed, unregistered companion read (full 2020-2025 archive, 1,537
games, no additional window spent).** Production-screen delta: **-0.200
pts**, week-blocked 95% **[-1.206, +0.800]**, P+ **0.344**; season-blocked
**[-0.969, +0.601]**, P+ **0.325**. Per season: 2020 +0.909 (P+ 0.798), 2021
+0.000 (P+ 0.517), 2022 +0.403 (P+ 0.689), 2023 -1.504 (P+ 0.131), 2024
-1.504 (P+ 0.187), 2025 +0.749 (P+ 0.708) -- the sign flips season to
season, consistent with a small-or-absent effect, not a stable one. Recorded
as `nfl_lead44_lookahead_fade_on_production_full_archive` (registry total
after recording: 5,995).

**A disclosed tension, not resolved.** The raw home-cover rate on the
eligible (matched) subset runs the OPPOSITE way from the predeclared FADE
mechanism, in both cuts: window flagged **56.25%** [44.996, 69.447] (n=48)
vs control **50.00%** [39.128, 60.467] (n=48); full archive flagged
**53.73%** [46.761, 61.070] (n=134) vs control **49.28%** [41.537, 56.391]
(n=138). Both CIs are wide (24-30 points) and heavily overlapping. This
does not contradict the production-screen delta leaning FOR the fade in the
registered window -- that metric nets the override against whatever the
active model's OWN baseline pick already was on each flagged game, not
against a naive "always pick home" rule, so the two framings can legitimately
diverge on a thin, noisy population. Neither framing resolves in either
direction; both are reported plainly rather than summarized into a single
verdict.

Reliability (odd/even-season split of the production-screen delta, full
archive -- the 2-season registered window is too small for this check):
odd seasons **-0.260 pts** [-1.565, +1.151] P+ 0.370 (n=769); even seasons
**-0.136 pts** [-1.745, +1.383] P+ 0.438 (n=734). Both cross zero and agree
loosely in the negative direction on this broader population -- not a
resolved finding either way, and not the registered window's own read.

## Decision, stated before any caveat

Nothing here clears an admissible closing ground (every interval crosses
zero, or touches it at exactly one boundary, which this project's own
precedent treats as a lean, not a resolution). Per the EV rule, none of
these cells is strong enough on its own to change what is played; all four
measured cells are recorded `unresolved_below_power` (CFB pooled,
NFL registered-window, NFL full-archive) or rotation-`unresolved`
(NFL registered window), not closed, and not discarded. The CFB leg's
sandwich-fade delta and cover-rate gap agree in direction (P+ 0.70-0.72);
the NFL leg's two framings disagree on direction at this sample size, which
is itself the finding worth carrying forward -- this construction does not
yet distinguish signal from noise on either league's data, and the honest
next step, if anyone chases this further, is more games under the SAME
frozen encoding, not a re-tuned threshold or proxy.
