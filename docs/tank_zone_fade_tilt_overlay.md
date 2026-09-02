# Tank-zone fade tilt overlay

A parameter-free, pick-level fade of the league's two worst teams in weeks
14-18, dual-tracked as a prospective challenger. **Nothing in this document is
wired into `publishing.py` or the production pick path** — no owner decision to
play it on the real card has been made.

- Module: `src/nfl_ats/tank_zone_fade_tilt_overlay.py`
- Tests: `tests/test_tank_zone_fade_tilt_overlay.py`
- Challenger id: `tank_zone_fade_tilt_overlay`
- Stacked back-test: `scripts/tank_zone_fade_stacked_backtest.py`
- Weekly recorder: `scripts/record_tank_zone_fade_challenger.py`

Provenance labels follow AGENTS.md: **measured** = run this session,
**read** = opened this session with a file:line, **inferred** = reasoning.

## 1. The rule, frozen in words

REG season, **weeks 14-18 only**. Build the tank-zone flag for both teams: a
team whose record places it in the league's **bottom two**, computed from games
in strictly prior weeks of the same season. Then:

> If **exactly one** of the two teams is flagged **and** the active model's own
> forced pick **is that team**, flip the pick to the other side.

Both-flagged games are never touched — there is no measured direction when both
sides carry the flag (the same clean-case handling `coach_fade_overlay` and
`interim_hc_first_game_tilt_overlay` use). The overlay never flips a pick
*toward* a tank-zone team. It never fires in weeks 1-13, which is why a Week 1
card can never be moved by it. Nothing else ever flips.

The rule has **no free parameters**. Both of its constants are the registry
cell's own flag definition, transcribed rather than chosen:

| Constant | Value | Source (read) |
|---|---|---|
| Week window | 14-18 | `scripts/motivation_ladder_screen.py:532-550` — `population["week"].between(14, 18)` inside the M4 cell |
| Tank-zone size | bottom **two** league-wide | `scripts/motivation_ladder_screen.py:164-165` — `league_ordered = sorted(DIVISIONS, key=lambda t: (tallies[t][0], -tallies[t][1], t))`, then `tank_zone = set(league_ordered[:2])` |

## 2. The evidence

Registry cell `motivation_ladder_tank_zone_wk14_18` (**read**,
`registry/weak_signals.json`), measured 2026-08-21 by
`scripts/motivation_ladder_screen.py`, predeclared in
`docs/motivation_ladder_screen.md` cell M4 *before* the screen scored anything.

Population: NFL REG close-graded slate 2009-2025, team-perspective long table,
weeks 13-18. n = 2,768 team-games, n_flag = 144 (5.20% of the slate).

| Blocking | Effect (accuracy points) | 95% interval | `probability_positive` | blocks |
|---|---|---|---|---|
| Week-blocked (primary) | **+0.3049** | [-0.0794, +0.6966] | **0.9334** | 90 |
| Season-blocked (secondary) | +0.3049 | [+0.0371, +0.6118] | 0.9856 | 17 |

Raw rates (**measured** this session, reproduced exactly from
`artifacts/motivation_ladder_screen/20260821T182643Z/results.json`, cell
`tank_zone_wk14_18`): flagged team-games covered **44.44%**; the complement
covered **50.30%**.

The primary interval crosses zero. Per AGENTS.md, at this evaluator's ~2-point
resolution that is the **expected** shape for a real-but-small signal and is
never grounds to decline building a no-window-cost prospective challenger.
Neither admissible closing ground applies — no resolved wrong sign, no
positive-control bound — so the cell stays `unresolved_below_power`. Building
this overlay is an EV-positive dual-tracked play (P+ 0.9334 > 0.5), not a claim
of a proven edge.

## 3. Registry-description correction — carry this forward

The registry entry's `description` field reads:

> "…leans OPPOSITE tank-fade prediction (tank teams over-cover)…"

and `docs/motivation_ladder_screen.md`'s M4 classification bullet repeats it
("leans opposite to prediction (tank-zone teams over-covered), and the
season-blocked secondary excludes zero on that opposite side").

**Both are wrong about the direction.** **Measured** from the artifact
(`artifacts/motivation_ladder_screen/20260821T182643Z/results.json`, cell
`tank_zone_wk14_18`):

- `sign_dir` = **-1** — the predeclared direction is NEGATIVE on
  `team_covered`, i.e. **fade** the tank-zone team.
- `subset_mean` = **0.4444444444444444**, `complement_mean` =
  **0.5030487804878049** — flagged teams **under**-covered by 5.86 raw points.
- `raw_gap_pts` = **+5.8604**, `full_slate_effect_pts` = **+0.30488**. Those are
  positive because the screen signs its output by the *prediction*: a positive
  effect means the predeclared direction was **confirmed**
  (`scripts/motivation_ladder_screen.py:355-361`, `full_slate_effect_pts =
  scale_subset_effect(raw_gap_fraction, sign=sign, …)`).

So the fade direction is the direction the data shows, and the season-blocked
secondary excludes zero on the **confirming** side, not on the opposite side.
The overlay uses the predeclared fade direction. The registry `description`
string is misleading and is flagged for correction through the CLI —
`registry/weak_signals.json` is never edited by hand.

## 4. Standings convention, and the one disclosed adaptation

**How the screen computes it (read).** `build_state_timeline`
(`scripts/motivation_ladder_screen.py:178-241`) walks each season's games in
`gameday` order and, for each distinct gameday, snapshots the standings
**before** processing any of that day's games
(`scripts/motivation_ladder_screen.py:192-199`). `compute_day_states` then ranks
the league (`:164-165`) by **wins ascending, losses descending, team code
ascending** and takes the first two. Ties in the ordering are broken exactly
that way: at equal wins, more losses ranks *worse*; then alphabetically. A tie
*game* (`result == 0`) increments neither wins nor losses (`:228-238`). The
tallies are built from the frame `load_schedules` has already filtered to
`home_cover.notna()` (`:92-98`), so a prior game that pushed against the spread
contributes nothing.

**The adaptation (disclosed, deliberate).** Gameday granularity is
point-in-time safe for a historical replay, but it is **not available at this
project's Tuesday recording lock**: under that convention a Sunday game's
snapshot already contains that same week's Thursday-night result. This live
overlay therefore computes the standings from every completed game in
**strictly prior weeks** of the same season — the repo's standard "prior games
only" convention (`coach_fade_overlay`, `backup_qb_fade_overlay`,
`division_revenge_tilt_overlay`, and the `climatology_deviation_disclosure`
precedent on `forecast_cold_visitor_tilt`) — and exactly what a Tuesday-lock
snapshot can see, since the current week's games carry no `result` yet.

**Measured cost of the adaptation**, on the registry cell's own population and
the very snapshot the screen ran against
(`data/raw/20260817T235649Z/schedules.parquet`):

| | Screen (gameday) | This overlay (prior weeks) |
|---|---|---|
| flagged team-games, weeks 14-18 | 144 | **143** |
| flagged cover rate | 44.44% | **44.76%** |
| complement cover rate | 50.30% | **50.29%** |
| agreement | — | **99.96%** (1 team-game differs, none added) |

**The registered +0.3049 / P+ 0.9334 figures therefore do not transfer exactly
to this live arm.** The 2026 prospective ledger accrues fresh, independent
evidence for *this* construction.

Two further verbatim-port notes:

- **Push-dropped tallies** are kept, because they are part of the construct
  that produced the measured numbers and are pregame-known for prior games. The
  same filter is what makes an unplayed game contribute nothing, which is
  exactly the behaviour a live Tuesday run needs.
- **League membership** is derived from the season's own schedule rather than
  the screen's hardcoded 32-team `DIVISIONS` map
  (`scripts/motivation_ladder_screen.py:53-66`). **Measured** this session: each
  season 2009-2025's team set equals `set(DIVISIONS)` exactly (32 teams), so the
  two are equivalent on every measured season; deriving it additionally lets the
  test fixture use synthetic team codes.

## 5. Pregame safety

`tank_zone_flag_by_game` never reads the flagged game's own `result` for its own
flag: the standings entering week *W* are built only from that season's
completed games in weeks strictly less than *W*. Three leakage regression tests
prove it empirically rather than by assertion
(`tests/test_tank_zone_fade_tilt_overlay.py`):

1. blanking every `result` from week 14 onward (a true Tuesday snapshot) leaves
   the week-14 flags byte-identical;
2. mutating a flagged game's own `result` and `spread_line` does not change its
   own flag;
3. inverting every week-15+ result does not change any flag in weeks ≤ 14.

Everything the rule consumes — the schedule, prior weeks' final scores, prior
weeks' spreads — is public well before the Tuesday lock. There is no live fetch
and no fail-open path is needed: the only input is the local schedules snapshot.

## 6. Stacked-on-production back-test (mined-seasons read, CONTEXT not a gate)

`scripts/tank_zone_fade_stacked_backtest.py`, artifact
`artifacts/tank_zone_fade_stacked/20260901T194008Z/results.json` (**measured**
this session).

Baseline archive: `artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`
— the active `weak_stack`/ridge-α-10 model's 1,537 REG games 2020-2025 graded at
the Tuesday **opener** under the production probability rule; 1,503 scored games
(34 pushes), 107 week blocks, 6 season blocks. Baseline accuracy **53.3599%**.

The incumbent is the **played** chain, not the bare model:
`nfl_ats.four_overlay_composition`'s frozen union policy
`overlay_union_coach_division_revenge_player_arrests_spread_gap_v1` (coach fade
107 flips, division-revenge tilt 151, player-arrests back-side 25, spread-gap
zone fade 195; union 427). Production accuracy on this archive: **55.4225%**.

The tank-zone fade adds **16** flips, of which **11** are not already flipped by
production (10 of those are scored). Result:

| | Accuracy |
|---|---|
| Bare baseline | 53.3599% |
| Production chain (incumbent) | **55.4225%** |
| Production + tank-zone fade | **55.4225%** |

| Blocking | Candidate − production | 95% interval | `probability_positive` |
|---|---|---|---|
| Week-blocked (primary) | **+0.0000 pts** | [-0.3979, +0.4537] | **0.4277** |
| Season-blocked | +0.0000 pts | [-0.5222, +0.5326] | 0.4011 |

20,000 bootstrap samples, seed **20260901**.

The point estimate is exactly zero for a concrete, **measured** reason: of the
10 scored games the overlay moves that production does not already move, the
model's original pick was right on exactly 5 and wrong on exactly 5, so the
flips cancel game-for-game. Reading `probability_positive` here needs one extra
fact: 12.33% of bootstrap draws are *exactly* zero (no moved game was
resampled), and those count as "not positive", so the 0.4277 figure is
depressed by ties, not by a negative lean — 42.77% of draws are positive,
44.90% negative, which excluding ties is 48.8%, a coin flip.

**Verdict.** The interval crosses zero at both predeclared blockings, so there
is no resolved wrong sign; no positive control was run for this composed form.
Neither admissible closing ground applies, so this is
`unresolved_below_power` and the challenger is registered. Recorded as
`motivation_ladder_tank_zone_fade_stacked_on_production` in family
`motivation_ladder_tank_zone_stacked_on_production`
(`nfl-ats weak-signals record`, registry at 636 signals after the write).

The thin population is a **power statement, not a defect**: weeks 14-18 are
~5% of the slate, only one side may carry the flag, and the model must already
be picking that side.

## 7. Expected firing rate

**Measured** on the current snapshot (`data/raw/20260824T115346Z`), REG weeks
14-18, 2009-2025:

- Games with **exactly one** tank-zone side: 140 total, mean **8.2 per season**
  (range 6-10). Games with **both** sides flagged: 2 in 17 seasons.
- Of the 54 such games in 2020-2025, the model actually picked the tank-zone
  side in **16** — so the overlay fires roughly **2.7 times per season**, of
  which about **1.8 per season** are games the production chain does not already
  flip.

**Week 1 2026: zero flips, by construction** (measured dry run against
`artifacts/margin_predictions/2026-week-01-20260824T120725Z/recommendations.csv`,
the active model `d1f07d773475dc58`'s own card): all 16 games are week 1, which
is outside the registered 14-18 window, so every probability is byte-identical
to the active card. The first opportunity for this challenger to differ from
the incumbent arrives in **week 14** of the 2026 season.

## 8. Independence

This overlay is dual-tracked only and is **never composed** with the other
prospective challengers. Like every sibling tilt, it transforms the *same*
un-overlaid base card independently and records its own arm; it never sees, and
is never seen by, another challenger's flips. The stacked back-test in section 6
is a diagnostic of what it would add on top of the played chain — it is not a
composition that anything records.
