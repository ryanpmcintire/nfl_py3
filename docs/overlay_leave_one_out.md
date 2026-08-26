# Does the played four-member overlay policy survive removing one member? (predeclared, 2026-08-26)

This document is written, and its design section committed, **before any output
of `scripts/overlay_leave_one_out.py` exists.** The Results section below is
appended after the run; nothing above it is edited afterward.

## Why this study, now

`docs/overlay_subset_holdout_v2.md` asked "should we ADD a member to the played
policy?" over ten candidates with a predeclared split. **Measured** by grepping
that document: it contains no leave-one-out, ablation, or drop-one arm. The
played policy's composition has only ever been tested by adding to it, never by
removing from it.

**Measured** this session
(`artifacts/reliability_map/20260826T112507Z/results.json`, built by
`scripts/reliability_map.py`): the feature `gap_division_revenge` has a
split-half reliability point estimate of **-0.136** (Spearman-Brown corrected;
underlying half-to-half Pearson r = -0.0635, 95% CI [-0.1384, +0.0127],
probability_positive 0.0505, n = 576 team-seasons).

**Binding closing-grounds taxonomy (verbatim, AGENTS.md).** An interval or CI
that contains zero is NEVER grounds to reject, fail, or close an experiment. At
this evaluator's ~2-point resolution, "contains zero" is the EXPECTED outcome
for a real small signal. Only two grounds ever close a line of work: (1)
refuted mechanism -- a RESOLVED wrong sign (whole interval on the wrong side of
zero) or zero split-half reliability; (2) bounded by a positive control proven
able to detect an effect that size. Everything else is `unresolved_below_power`:
record it with `nfl-ats weak-signals record`, report `probability_positive`,
never the binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the validator.

So the reliability reading above **closes nothing about `division_revenge_tilt`
and is not grounds to remove anything.** It is, however, the only played
member whose underlying feature has now measured below zero on repeatability --
a reason to MEASURE the removal decision directly, not to infer an answer from
a proxy statistic. That measurement is this document.

## What is actually played, confirmed by execution (not by reading a label)

Per `AGENTS.md`/`docs/overlay_subset_holdout_v2.md`'s "Correction recorded
first": verify what production does by running production's own code path.

**Measured** this session: `nfl_ats.clv.record_paper_decisions` was executed
against an isolated copy of the real active artifacts (same precedent as
`scripts/lockday_rehearsal.py`: an isolated artifacts root built by
`lockday_rehearsal.build_isolated_root`, a shadow data root with the real
player-arrests snapshot restamped fresh by `lockday_rehearsal.build_shadow_data_root`,
simulated lock instant 2026-09-08T16:00Z), called with
`require_fresh_arrest_overlay=True` -- production's own default, matching
`cli._cmd_publish_predictions`. Nothing was written to the real `artifacts/` or
`data/` trees; the isolated copy lived under the session scratchpad and is
discarded.

Recorder result on the live Week 1 2026 card (16 games):

```
decision_policy_id = overlay_union_coach_division_revenge_player_arrests_spread_gap_v1
coach_fade_flip_count = 1
division_revenge_flip_count = 0
player_arrests_flip_count = 0
spread_gap_zone_flip_count = 1
composed_overlay_flip_count = 2
```

The four live members, confirmed by execution:

- `coach_fade`
- `division_revenge_tilt`
- `player_arrests_back_side_policy`
- `spread_gap_zone_fade`

This matches `_FOUR_OVERLAY_POLICY_ID` (`nfl_ats.clv`) and
`docs/overlay_subset_holdout_v2.md`'s `PLAYED_UNION`.

## Design (predeclared)

Four variants only, each dropping exactly one member from the played
four-member union. Members are named with the overlay-name keys used by
`scripts/overlay_stack_backtest.py.OVERLAY_NAMES` /
`scripts/overlay_subset_holdout_v2.py.PLAYED_UNION`, which is the SAME flip-set
computation the confirmed production composition resolves to (`coach_fade` <->
`coach_fade_overlay`, `division_revenge_tilt` <-> `division_revenge_tilt_overlay`,
`player_arrests_back_side_policy` <-> `player_arrests_back_side_policy`,
`spread_gap_zone_fade` <-> `spread_gap_zone_fade_overlay`):

| variant | members retained (3 of 4) |
|---|---|
| drop `coach_fade` | division_revenge_tilt_overlay, player_arrests_back_side_policy, spread_gap_zone_fade_overlay |
| drop `division_revenge_tilt` | coach_fade_overlay, player_arrests_back_side_policy, spread_gap_zone_fade_overlay |
| drop `player_arrests_back_side_policy` | coach_fade_overlay, division_revenge_tilt_overlay, spread_gap_zone_fade_overlay |
| drop `spread_gap_zone_fade` | coach_fade_overlay, division_revenge_tilt_overlay, player_arrests_back_side_policy |

**Archive:** the same 1,537-game opener-evaluation archive
(`artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`), scored
against `correct_at_open_probability_rule`, that `overlay_subset_holdout_v2.py`
uses.

**Split:** the SAME predeclared decider as `overlay_subset_holdout_v2.py`:
**selection seasons 2020-2022, evaluation/holdout seasons 2023-2025.** The
evaluation half (2023-2025) is PRIMARY / decision-grade -- the opener grade
per `AGENTS.md` ("Grade the decision at the OPENER") -- reported once, not
chosen after peeking. The selection half is a secondary, exploratory look,
reported for completeness and for the shrinkage/rank-stability diagnostic; it
never changes which number is primary.

**Far less selection noise than the referenced study.** That study searched up
to 4,095 subsets for an argmax; this design tests exactly four PRE-SPECIFIED
variants, with no search and no subset selection -- there is no argmax to
overfit. The shrinkage/rank-stability diagnostics (OLS slope of holdout delta
on selection delta, Spearman rho) are still computed across these four points
for comparability with the referenced study's reporting convention, but are
explicitly flagged as under-powered at n=4 and not treated as decision-relevant
on their own -- four points estimate a slope and a rank correlation very
noisily, and that is disclosed rather than hidden.

**Primary metric, each variant:** the **paired per-game accuracy delta versus
the played four-member incumbent** -- `variant_correct - incumbent_correct` on
every game, not each arm compared independently to the raw model. A game only
differs between a variant and the incumbent when the dropped member fired
ALONE on it (no other retained member also fired there under the OR-union
rule); pairing removes the agreeing games from the noise rather than
re-adding them as zero-signal draws on both sides. Reported for the primary
(holdout) read: games changed, raw incumbent accuracy, raw variant accuracy,
delta in accuracy points, week-blocked AND season-blocked bootstrap intervals
(`nfl_ats.clv.week_blocked_bootstrap`, 20,000 samples, seed matching the
referenced study), and `probability_positive`. The same is reported for the
selection half as a secondary look.

**Sanity anchor:** the played union's full-archive accuracy is reproduced
against `docs/overlay_subset_holdout_v2.md`'s reported 55.4225% (vs the raw
model's 53.3599%) on 1,503 opener games, and this run states explicitly
whether it matches.

**Decision rule (binding, AGENTS.md).** The pool is forced picks: 285 cards
submitted regardless. `probability_positive` above 0.5 on the primary
(holdout) read favours DROPPING that member; below 0.5 favours KEEPING it. No
threshold gates the recommendation -- a predeclared bar may govern what gets
CLAIMED, never what gets PLAYED. This study makes no production change by
itself; it reports what the decision implies.

**Adaptation note.** `scripts/overlay_leave_one_out.py` adapts
`scripts/overlay_subset_holdout_v2.py` **by import**, not by rewriting the
harness: it imports `load_inputs`, `build_predictions_frame`, `run_overlays`,
`extra_flip_sets`, `union_delta`, `bootstrap`, `evaluate`, `PLAYED_UNION`,
`ARREST_MEMBER`, `SELECTION_SEASONS`, `EVALUATION_SEASONS`, `SAMPLES`, and
`SEED` directly from that module, so flip-set construction, the bootstrap
machinery, and the season split are identical to the referenced study, not a
re-implementation that could silently drift.

**Recording.** Every one of the four variants is recorded with
`nfl-ats weak-signals record`, classification `unresolved_below_power` unless
an admissible closing ground applies per the taxonomy above, regardless of
sign or magnitude.

**Attribution caveat (inherited from the referenced study).** Every member
overlay was registered on windows overlapping this archive, so this is an
honest estimate of the REMOVAL decision on this archive, not a fresh
confirmation of any component signal, and no rotation-registry window is
spent.

**Does not change the played card.** This is a measurement that informs an
owner decision ahead of the Week 1 2026-09-08 lock. Changing production is the
owner's call.

---

## Results

**Measured** by running `scripts/overlay_leave_one_out.py`
(`artifacts/overlay_leave_one_out/20260826T142312Z/result.json`).

### Sanity anchor: reproduced

Played four-member union, full 1,503-game opener archive: **55.4225%**
(0.5542248835662009) vs raw model **53.3599%** (0.5335994677312043), +2.0625
accuracy points. This matches `docs/overlay_subset_holdout_v2.md`'s reported
55.4225% / 53.3599% / 1,503 games exactly. Production's behavior in this
harness matches the referenced study's baseline.

### The four leave-one-out variants, opener grade, PRIMARY read (2023-2025 holdout)

Paired per-game delta = variant accuracy minus incumbent (played four-member
union) accuracy, on the same 799 opener-graded holdout games, week-blocked and
season-blocked (`nfl_ats.clv.week_blocked_bootstrap`, 20,000 samples each;
holdout has 54 distinct week-blocks and only 3 season-blocks, so the
season-blocked interval is comparatively coarse -- disclosed below per arm).

| variant (dropped member) | games changed | incumbent acc | variant acc | delta (pts) | week P+ | week 95% CI | season P+ | season 95% CI |
|---|---|---|---|---|---|---|---|---|
| drop `coach_fade` | 44 | 54.6934% | 53.9424% | **-0.7509** | 0.1672 | [-2.475, +0.878] | 0.2948 | [-3.384, +0.749] |
| drop `division_revenge_tilt` | 61 | 54.6934% | 54.8185% | **+0.1252** | 0.5288 | [-1.770, +1.990] | 0.5970 | [-1.873, +2.632] |
| drop `player_arrests_back_side_policy` | 9 | 54.6934% | 54.0676% | **-0.6258** | 0.0267 | [-1.372, +0.123] | 0.0000 | [-1.504, +0.000] |
| drop `spread_gap_zone_fade` | 70 | 54.6934% | 54.6934% | **0.0000** | 0.4924 | [-2.442, +2.244] | 0.3684 | [-3.007, +3.007] |

Every interval crosses (or, for the arrests arm's season-blocked read,
exactly touches) zero. Per the binding taxonomy above, **that closes nothing
about any of the four** -- it is the expected outcome at this evaluator's
resolution on n=799 games with 9-70 games actually changed per arm. What is
informative is the SIGN and `probability_positive`, reported below, not
whether the interval excludes zero.

Secondary look, selection half (2020-2022, NOT used to choose or to override
the primary read):

| variant (dropped member) | games changed | incumbent acc | variant acc | delta (pts) | week P+ | season P+ |
|---|---|---|---|---|---|---|
| drop `coach_fade` | 37 | 56.2500% | 56.1080% | -0.1420 | 0.3940 | 0.2908 |
| drop `division_revenge_tilt` | 60 | 56.2500% | 55.6818% | -0.5682 | 0.2312 | 0.2576 |
| drop `player_arrests_back_side_policy` | 8 | 56.2500% | 56.2500% | 0.0000 | 0.4480 | 0.2576 |
| drop `spread_gap_zone_fade` | 80 | 56.2500% | 55.1136% | -1.1364 | 0.1400 | 0.0000 |

Bonus, full 1,503-game archive (not primary, reported for completeness):

| variant (dropped member) | games changed | delta (pts) | week P+ | season P+ |
|---|---|---|---|---|
| drop `coach_fade` | 81 | -0.4657 | 0.1997 | 0.2251 |
| drop `division_revenge_tilt` | 121 | -0.1996 | 0.3626 | 0.3721 |
| drop `player_arrests_back_side_policy` | 17 | -0.3327 | 0.0993 | 0.1008 |
| drop `spread_gap_zone_fade` | 150 | -0.5323 | 0.2524 | 0.2246 |

### Shrinkage / rank-stability diagnostic (n=4, explicitly under-powered)

OLS slope of holdout delta on selection delta across the four variants:
**-0.694**. Spearman rho: **-0.600**. Both computed over only 4 points, as
predeclared -- this is not a usable shrinkage estimate (the referenced
12-member study used 4,095 subsets for the same statistic and still called
its own n large enough only to be a rough average). The negative sign here
mainly reflects that `drop_player_arrests_back_side_policy` and
`drop_spread_gap_zone_fade` swap which one looks worse between the two
halves (arrests: 0.000 selection -> -0.626 holdout; spread-gap: -1.136
selection -> 0.000 holdout) while `drop_division_revenge_tilt` flips sign
entirely (-0.568 selection -> +0.125 holdout). With 4 points this is reported,
not interpreted as a stable finding.

## Decision (stated before the caveats, per AGENTS.md)

**Three of four variants favour KEEPING the member** at the primary
(2023-2025, opener) grade: dropping `coach_fade` (P+ 0.167 week / 0.295
season), dropping `player_arrests_back_side_policy` (P+ 0.027 week / 0.000
season), and dropping `spread_gap_zone_fade` (P+ 0.492 week / 0.368 season,
a near coin flip on the week blocking but season-negative) all read
`probability_positive` below 0.5 on both blockings (spread-gap's week
blocking is within a hair of 0.5). Per the binding decision rule --
`probability_positive` above 0.5 favours the change, below 0.5 favours the
status quo -- none of these three clears the bar to drop. **The played policy
survives removal of `coach_fade`, `player_arrests_back_side_policy`, and
`spread_gap_zone_fade`.** That is itself informative: this is the first
removal test the policy has ever faced, and it holds up on three of its four
legs.

**One variant, `drop_division_revenge_tilt`, reads `probability_positive`
above 0.5 on BOTH blockings** (0.5288 week, 0.5970 season) at the primary
grade, with a positive point estimate (+0.1252 accuracy points on 799
holdout games). Per the project's own rule -- forced picks, no threshold
gates a play, `probability_positive` above 0.5 favours the change, and
declining a >50%-likely-better change is taking the worse side of the bet --
**this measurement favours dropping `division_revenge_tilt` from the played
policy.**

**The caveats, stated after the decision, not instead of it:** the effect is
tiny (+0.125 points), the interval is wide and crosses zero on both
blockings ([-1.77, +1.99] week, [-1.87, +2.63] season), the selection-half
read for the same arm was NEGATIVE (-0.568), and the full-archive bonus read
is also negative (-0.1996) -- so this is a marginal, P+-just-over-half read
on the ONE predeclared primary split, not a strong or consistent one. It
lines up directionally with the reliability-map result that motivated this
study (`gap_division_revenge` split-half reliability -0.136, P+ 0.0505,
measured this session) in the sense that both independently point away from
`division_revenge_tilt` rather than toward it, but the reliability
measurement is about the underlying FEATURE across team-seasons and this
measurement is about the OVERLAY's effect on picks -- related constructs,
not the same statistic, and neither is strong enough on its own to close
anything.

**Recommendation:** report this to the owner as a marginal, positive-EV-per-the-rule
signal to drop `division_revenge_tilt` from the played four-member policy --
worth a production decision, not worth overstating. The other three members
each showed a negative or coin-flip primary read and should stay. This
document does not change the played card; see "Do not change the played
card" below.

## Recorded

All four variants recorded via `nfl-ats weak-signals record`, classification
`unresolved_below_power` (no arm's interval sits entirely on one side of zero
on both blockings simultaneously with an admissible closing ground -- the
`player_arrests` season-blocked upper bound touches 0.0000 exactly but does
not clear it, so `wrong_sign_resolved` is not available even there). Effect
recorded is the PRIMARY (2023-2025 holdout) paired delta in
`accuracy_points`; week-blocked interval recorded as the primary interval,
season-blocked figures kept in notes. Family
`overlay_leave_one_out_2026_08_26` groups the four as correlated arms sharing
the same incumbent and archive, not independent votes.

## Do not change the played card

Per the task's own instruction and `AGENTS.md`: **this is a measurement, not
a production change.** No card was touched. The recommendation above is for
the owner's decision ahead of the 2026-09-08 Week 1 lock.

