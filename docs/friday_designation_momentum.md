# Friday designation momentum (LEAD-08)

Predeclared construct (ROADMAP.md LEAD-08, read): a player's practice
trajectory across Wed/Thu/Fri -- DNP-to-LP healing vs LP-to-DNP
deteriorating vs static -- predicts Sunday status and, through starter
availability, the cover result. Predeclared direction: FADE teams with
net-negative (deteriorating) trajectories.

## Binding verdict taxonomy (verbatim, per session instructions)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator.

## 1. The literal construct has zero coverage -- confirmed a third way

CX14 (2026-09-05, `docs/injury_trajectory_leads.md`) already found zero
covered team-games for LEAD-08 against `data/raw/nflcom_injuries` and
concluded "historical NFL.com rows have ... only one practice-status field,
not a stored Wednesday/Thursday/Friday revision sequence." This session
re-checked the question against the CANONICAL PER-03 source instead
(`data/raw/nflverse_injuries/20260910T203021Z/injuries.parquet`, the newest
of 25 capture stamps under that directory), which is the actual ingested
history PER-03 describes (76,784-class canonical rows; measured here at
90,891 rows once 2025/2026 and postseason are included).

**Measured** (`groupby(["season","game_type","week","team","gsis_id"]).size()`
over the full 2009-2026 archive, all game types): 90,887 of 90,889
player-week groups have exactly **one** row; only 2 groups have 2. **Read**
(`scripts/nflverse_injuries_ingest.py:126-134`, the ingest script's own
manifest note): "date_modified is 0-null for seasons 2011-2024 ... any
as-of/checkpoint feature keyed on date_modified will correctly resolve every
2025 row ... to missing" -- this source is documented, by its own author, as
one final-observation row per player-week, not a revision stream. **Read**
(the CX14-era module recovered from a stale agent worktree,
`.claude/worktrees/agent-a001b0c5640b0c82b/src/nfl_ats/injury_trajectory_features.py:92-110`):
its `build_flags` function tried to pair an "early" (Wed/Thu) row against a
"Friday" row for the same `gsis_id` within the same player-week and found the
join empty for LEAD-08, which is the necessary consequence of the row-count
fact above, not a proxy-filtering artifact.

**This is now confirmed three independent ways** (CX14's NFL.com read, the
ingest script's own manifest, and this session's direct row-count
measurement on the canonical PER-03 source): no player in the ingested
history ever has both an early-week and a Friday practice observation to
sequence. The literal Wed-to-Fri per-player trajectory class this row asks
for **cannot be built from any currently-ingested source in this repository**.
This is a data-availability fact about what was captured, not a claim that
the underlying biological effect is absent, and it is not treated as a
closing ground below.

## 2. The closest available proxy: revision-timing class

Since the source cannot show what a player's status was earlier in the week,
it can still show **when** the single stored row was last touched. Real NFL
practice-report data is revised as a player's status changes; if nothing
changes, the row's `date_modified` stays at its original (usually
Wednesday) value. **Measured** (`date_modified`'s Eastern weekday across the
full archive): Wed 6,166 / Thu 3,171 / Fri 64,972 / Sat 5,119 / Sun 64 / Mon
134 / Tue 192 -- the large Friday mass is exactly what "most reports get a
final touch on Friday" predicts, and the nontrivial Wed/Thu mass is exactly
what "a player's designation didn't move again after Wed/Thu" predicts.

Built (`scripts/friday_designation_momentum_screen.py`, `classify_revision_timing`):
- **static**: `date_modified` Eastern weekday is Mon/Tue/Wed -- the row was
  never revised again this week.
- **revised_late**: `date_modified` Eastern weekday is Thu/Fri/Sat/Sun --
  the row was touched again later in the week.

This proxy can say a revision happened; it cannot say whether the
underlying change was healing or deteriorating (the DNP-to-LP vs LP-to-DNP
distinction), because only the final value survives. **The predeclared
three-way construct (healing / deteriorating / static) is only ONE-THIRD
measurable** (static, via "never revised again"); healing and deteriorating
are not separable from this source. Reported honestly as a proxy for the
predeclared construct, not the construct itself.

## 3. Base rates and split-half reliability (Step 1 + Step 2)

**Measured** (`scripts/friday_designation_momentum_screen.py`, REG season
2010-2024 -- 2010 is the first season with any real `date_modified`
timestamps per the ingest manifest; 2025/2026 have none and are excluded
from this table): 76,554 player-weeks resolve both a practice level and a
revision class.

| Revision class | n | Sunday inactive rate |
|---|---:|---:|
| revised_late | 70,123 | 46.886% |
| static | 6,431 | 45.685% |

Conditioning on the player's own final practice level (the level the static
label already carries) still shows the same-direction gap in all three
levels:

| Final practice level | revised_late inactive rate | static inactive rate | gap |
|---|---:|---:|---:|
| DNP | 86.834% (n=19,558) | 72.539% (n=2,509) | +14.30pp |
| LP | 43.906% (n=18,066) | 38.816% (n=1,520) | +5.09pp |
| FP | 24.502% (n=32,499) | 21.982% (n=2,402) | +2.52pp |

"Inactive" here is a **snap-count proxy** (0 total snaps recorded for that
exact `game_id`/`gsis_id`, via `attach_snap_player_ids` +
`data/players/raw/20260817T184901Z/snap_counts.parquet`), not the official
Sunday inactive list. `data/players/inactives/` now holds real inactive-list
captures (absent when CX14 ran on 2026-09-05; 33 timestamped captures exist
as of this session), but those only cover the in-progress 2026 season and
cannot back-test 2010-2024, so they are not used as the historical ground
truth here.

**Split-half reliability** (team, odd-season mean vs even-season mean, the
2026-09-05 module's week-parity method re-keyed to season parity per this
session's brief), n=32 teams:

| Quantity | Correlation | Status |
|---|---:|---|
| revised_late absolute inactive rate | **0.6305** | measured |
| (revised_late minus static) gap | **0.3652** | measured |

Neither is zero or negative, so `no_split_half_reliability` (an admissible
closing ground) does not apply.

## 4. Team-game feature and production screen (Step 3)

**Value weights reused from the injury-value construct** (read,
`src/nfl_ats/players.py:1244-1269`, `_injury_features`): that function's
`severity x role_share` weighting (offense/defense snap share divided by
roster-spot count) is the piece reused here as **trailing snap share**
(`attach_snap_player_ids`, then `snap_share = max(offense_pct, defense_pct)`,
shifted one game back per player so no game uses its own outcome). The
construct's OTHER factor (`_player_value_rate`, an EPA-production-based
value rate with its own EWMA state) was **not** reconstructed for this
screen -- disclosed simplification, not hidden. `severity` is a **season-lagged
walk-forward** empirical rate (expanding window, strictly prior seasons only,
floored at 30 prior observations else a pooled fallback) -- the same
no-leakage principle `availability.py`'s `learned_unavailability` uses, keyed
on this construct's own class instead of report/practice status.

Team-game feature: for each team-game, sum over "starters" (trailing snap
share >= 0.5, the `HIGH_SNAP_SHARE_THRESHOLD` convention already used in
`src/nfl_ats/roster_availability_flag_features.py`) who appear on that
week's injury report, of `trailing_share x severity[season][class]`. A team
is **flagged** if it has at least one such starter in the `revised_late`
class (the predeclared LEAD-08 direction: FADE the team with the worse
trajectory signal). Flip rule (matching the CX14 LEAD-08/09/10/11
precedent): if exactly one side is flagged and PRODUCTION's current pick
already favors the flagged (fade-target) side, flip to the other side;
both-flagged, neither-flagged, or a flag that already disagrees with
production leaves the pick unchanged.

**Production** = `pick_home_at_open_probability_rule` from the opener
evaluation matching the active model (`find_matching_opener_evaluation`,
active model `d49194e04945a5e5`,
`artifacts/opener_evaluation/20260910T211255Z/per_game.parquet`, 1,537
games) -- this column already carries the served home-side offset (43 picks
changed vs the raw probability rule per that artifact's own metadata), so it
is the actual played pick, not a reconstruction.

**Rotation window**: declared family `friday_designation_momentum_on_production`,
`nfl-ats rotation assign --size 2` returned **[2020, 2021]** -- the same
window LEAD-62 used, as this row's brief preferred.

**Measured** (`artifacts/experiments/friday_designation_momentum/20260911T015641Z/results.json`,
20,000 week-blocked draws, seed 20260910):

| Population | Games | Picks changed | Production accuracy | Candidate accuracy | Effect (accuracy points) | 95% interval | probability_positive |
|---|---:|---:|---:|---:|---:|---|---:|
| Assigned window [2020,2021] | 456 | **20** | 53.2895% | 52.8509% | -0.4386 | [-1.7660, +0.8889] | 0.2679 |
| Full population (every graded season) [2020,2025] | 1,503 | **50** | 54.5576% | 54.5576% | 0.0000 | [-0.7984, +0.8005] | 0.502975 |

Positive control (candidate replaced by the realized `margin_vs_open`, same
[2020,2021] window): **+46.7105 accuracy points, 95% [+41.505, +52.116],
probability_positive 1.0** -- the harness is proven able to detect an effect
this size; the -0.4386 read above is a real (if unresolved) measurement, not
a blind instrument.

Per-season breakdown (full population), sign is not uniform -- this alone
rules out `wrong_sign_resolved` (which requires the WHOLE interval below
zero):

| Season | Effect | 95% interval | probability_positive |
|---|---:|---|---:|
| 2020 | +0.4545 | [-0.930, +1.887] | 0.7122 |
| 2021 | -1.2712 | [-3.376, +0.858] | 0.1232 |
| 2022 | +1.6129 | [-1.992, +4.925] | 0.8221 |
| 2023 | 0.0000 | [-1.498, +1.504] | 0.4986 |
| 2024 | -0.7519 | [-2.214, +0.760] | 0.1529 |
| 2025 | 0.0000 (zero coverage) | [0, 0] | 0.5000 |

2025 shows exactly zero because `date_modified` is null for every 2025 row
in the canonical source (per the ingest manifest quoted in section 1), so
the revision-timing proxy has no coverage that season -- a data gap, not a
measured null effect.

## 5. Verdict

Neither closing ground applies: the interval is not resolved to one side
(sign flips 2020 vs 2021 vs 2022, and the assigned-window interval
[-1.766, +0.889] straddles zero), and reliability is measured positive
(0.365-0.631), not zero. This is `unresolved_below_power` on both the
assigned window and the full-population read -- exactly the outcome AGENTS.md
says is EXPECTED for a real small signal at this evaluator's resolution, and
not grounds to close the line. Decision framing per the promotion-bar rule:
`probability_positive` 0.2679 on the assigned window means the CURRENT
expected-value read leans toward NOT adding this specific discrete fade rule
on top of production as constructed; it does not mean the underlying
mechanism is false, and the full-population read (P+ 0.503, effect exactly
0.0) is closer to a coin flip once 2025's zero-coverage season and the
2022 positive season are included.

## 6. What would resolve this further

- The revision-timing proxy is not the predeclared construct. If a raw
  source with true day-by-day (Wed/Thu/Fri) practice observations is ever
  ingested, the literal healing/deteriorating trajectory becomes
  measurable and should replace this proxy rather than extend it.
  `data/players/inactives/` (33 captures as of this session) will eventually
  support a real positive-control oracle once enough 2026 games accumulate;
  it does not yet.
- The full EPA-based `value_rate` factor from `_injury_value_features` was
  not reconstructed (snap-share-only weighting was used instead); doing so
  would be a closer, but not exact, reuse of the injury-value construct.

## 7. Commands and artifacts

- Screen: `.\.tools\uv.exe run --no-sync python scripts\friday_designation_momentum_screen.py --window-seasons 2020,2021`
- Rotation: `nfl-ats rotation declare --name friday_designation_momentum_on_production --grade opener --acknowledge-mined ...`;
  `nfl-ats rotation assign --name friday_designation_momentum_on_production --size 2` (returned [2020, 2021]);
  `nfl-ats rotation record --name friday_designation_momentum_on_production --verdict unresolved ...`
- Weak signals: `nfl-ats weak-signals record --name friday_designation_momentum_on_production_2020_2021 ...`;
  `nfl-ats weak-signals record --name friday_designation_momentum_on_production_full_population_2020_2025 ...`
- Artifact: `artifacts/experiments/friday_designation_momentum/20260911T015641Z/`
  (`results.json`, `paired_predictions.parquet`, `team_game_feature.parquet`).
