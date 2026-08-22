# Bye-week overvaluation screen (post-2011 CBA)

Written **before** `scripts/bye_overvaluation_screen.py` scores anything
(predeclaration frozen first, per project convention). Family: **bye-week
overvaluation after the 2011 CBA** — the hypothesis that the market still
prices bye rest at its historical magnitude while the true on-field bye
advantage has collapsed, leaving a systematic fade-the-bye edge.

## Literature lead (reported-in-docs; NOT verified or measured this session)

**Reported** (unverified): a 2026 Frontiers in Behavioral Economics paper using
state-space models reports the true bye advantage collapsing from **+2.2 points
before the 2011 CBA to +0.3 points after**, while the MARKET's spread
adjustment for the bye **rose from +0.39 to +0.97 points** — i.e. the market now
overvalues byes by ~0.6 points, and fading bye-advantaged teams is reported to
run ~52% since 2011. The same source carries the legacy, since-contradicted
claim of Sung & Tainsky (road favorites off bye covering ~73% ATS in their
window) as the era-hypothesis this screen tests in the modern era. None of
these numbers are independently verified here; they motivate direction only.

## Overlap audit (grep of `registry/weak_signals.json` signal names + docs, done before scoring)

| Existing signal | Why these cells are not a re-measure |
| --- | --- |
| `travel_rest_home_off_bye` / `travel_rest_away_off_bye` (`home_rest>=13` / `away_rest>=13`, absolute) | Disclosed overlap, distinct construct: those use an absolute rest-days threshold that conflates scheduled byes with MNF/primetime extra rest, and are UNconditioned on opponent bye status, market side, and era. Cells below require the strict >=12-day gap AND condition on market side and era. Correlated inputs; do not sign-test-pool together. |
| `venue_milestone_post_bye_home` / `venue_milestone_post_bye_road` (strict >=12-day gap, home/road conditioned, 2009-2025) | Closest precedent (its strict definition is reused verbatim), but unconditioned on opponent bye status, market side, or era. Cell (a)-post is a strict subset of `venue_milestone_post_bye_home` rows; disclosed correlation, no pooling. |
| `bias_battery_extra_rest_edge` (own rest − opponent rest >= 4 days, opener-grade 2020-2025) | Different construct entirely: relative-rest threshold with no bye identification, no era split, different population/grade. |
| `cfb_bias_battery_bye_week_rest_edge` | Different league. |
| `pick_conditioned_off_bye_fade_pre2018` | Model-pick-conditioned lead-gen replication, 2011-2017 close-grade walk-forward — different population, grading, and mechanism framing; conceptually adjacent to cell (d), disclosed. |
| `era_trend_extra_rest_edge` | Trend diagnostic of a different construct (relative rest >= 4 days). |

## Strict bye definition (disclosed choice)

A team is "off bye" in a game iff the gap to its immediately preceding game of
the same season is **>= 12 calendar days** — the exact precedent of
`scripts/venue_milestone_screen.py` (POST_BYE_GAP_DAYS=12), chosen for
cross-screen consistency; it excludes week-1 games (no prior game) and
primetime-only extra rest. Week-1 rows therefore carry flag=False and sit in
the complement.

## Spread convention (measured)

`schedules.parquet` `spread_line`: **positive = home team favored** — measured
this session on `data/raw/20260817T235649Z/schedules.parquet`: P(home wins
outright | spread_line >= +6) = 0.787 vs P(...| spread_line <= -6) = 0.217;
mean(result − spread_line) ≈ +0.04, consistent with `add_ats_outcomes`
(`ats_margin = result − spread_line`). Road favorite ⇔ `spread_line < 0`.

## Era boundary (frozen convention)

Post-CBA era = **seasons 2012-2025** ("post-2011"); pre-CBA control =
**seasons 2009-2011**, because the local snapshot starts at 2009 (measured:
snapshot min season 2009). 2011 is assigned to the pre-era conservatively: the
CBA was ratified August 2011, after schedules were built, so 2011 cannot show
CBA-era schedule-construction effects. The pre-era control is thin (3 seasons)
and is reported with that caveat.

## Point-in-time safety

Every flag is a **schedule fact**: gameday gaps within each team's own season
and the pre-release closing line `spread_line`. No scores, in-game actuals, or
weather enter any flag (scores enter only the outcome column, which is standard
for an ATS target). All cells are point-in-time safe by construction.

## Cells (directions frozen before scoring)

Outcome is game-level cover accuracy_points, full-slate scaled, unless noted.
Subset-vs-complement on the stated population; week-blocked primary bootstrap,
season-blocked secondary, 20,000 resamples, seed **20260821**.

1. **`bye_overval_home_edge_post2011`** — HOME team off strict bye AND opponent
   NOT off bye, seasons 2012-2025. Mechanism: market overprices the bye-holding
   side. Predicted direction: **NEGATIVE home_cover edge**.
2. **`bye_overval_home_edge_pre2011`** — identical flag, seasons 2009-2011.
   Era control: under the literature lead the true bye advantage was real
   (+2.2 pts reported) pre-CBA, so the same cell should be flat-or-positive.
   Predicted direction: **POSITIVE / null-difference vs cell 1**. Caveat: only
   3 seasons of blocks exist; season-blocked secondary will be DEGENERATE
   (below the 10-block floor) and is labeled as such.
3. **`bye_overval_road_fav_post2011`** — AWAY team off strict bye AND
   `spread_line < 0` (road favorite), seasons 2012-2025. Tests the dead 73%
   ATS claim of Sung & Tainsky (reported-in-docs) in the modern era. Under the
   overvaluation mechanism the bye-holding favorite fails to cover; under the
   surviving legacy claim it would cover. Predicted direction:
   **POSITIVE home_cover edge** (fade arm).
4. **`bye_overval_both_bye_sanity`** — BOTH teams off strict bye, full window
   2009-2025. Rest cancels but the market reportedly still prices one bye;
   instrument sanity cell. Predicted direction: **NULL (no directional claim;
   two-sided)**. If this cell shows a large one-sided interval it impeaches the
   instrument rather than confirming the family.
5. **`bye_overval_fade_full_slate_post2011`** — fade arm expressed full-slate:
   population = seasons 2012-2025 restricted to week-blocks containing at least
   one strictly-off-bye team anywhere in the league ("weeks with byes only");
   flag = exactly one of the two teams off strict bye; outcome column is the
   FADE-side cover indicator (home holds the bye edge ⇒ 1−home_cover, i.e.
   away covers; away holds the bye edge ⇒ home_cover; complement rows keep raw
   home_cover — disclosed asymmetry). Predicted direction: **POSITIVE**.

Five cells total (within the 4-6 gate). All directions frozen above before any
scoring run.

## Multiplicity and recording posture

Mined/predeclared 5-cell battery, uncorrected multiplicity: every scoreable
cell is predeclared to record `unresolved_below_power` regardless of interval
shape — per AGENTS.md an interval crossing zero is the EXPECTED shape for a
real small signal and is never grounds for rejection; report
`probability_positive`, never "contains zero". `wrong_sign_resolved` would
apply only if a whole week-blocked interval sat entirely below zero on the
wrong side of its frozen direction; no positive-control bound is run. Cells 4
(sanity null) records its two-sided read like the others. Measure-only: the
script never writes either registry JSON; recording happens via separate
explicit `nfl-ats weak-signals record` calls returned by the agent.

## Results (measured this session)

Run: `.\.tools\uv.exe run python scripts/bye_overvaluation_screen.py --output
artifacts\bye_overvaluation_screen\predeclared_run` on snapshot
`data/raw/20260817T235649Z/schedules.parquet`; scored population **4,317**
REG 2009-2025 games (pushes dropped by `add_ats_outcomes`); seed 20260821,
20,000 draws; artifact
`artifacts/bye_overvaluation_screen/predeclared_run/results.json`; registry
stamp `registry/experiments/bye-overvaluation-screen/predeclared_run.json`.
Measured enumeration: 245 home-edge games post-2011 / 58 pre-2011; 103
road-favorite-off-bye post-2011; 297 both-teams-off-bye; 170 bye week-blocks.

| Cell | n_flag | Full-slate effect (pts) | Week-blocked 95% CI | P+ | Season-blocked P+ | Frozen direction |
| --- | --- | --- | --- | --- | --- | --- |
| `bye_overval_home_edge_post2011` | 245 | −0.330 | [−0.756, +0.096] | 0.0637 | 0.0666 | − |
| `bye_overval_home_edge_pre2011` | 58 | +0.271 | [−0.743, +1.201] | 0.7070 | 0.7426 (3 blocks, DEGENERATE) | + / null-diff |
| `bye_overval_road_fav_post2011` | 103 | −0.013 | [−0.278, +0.257] | 0.4614 | 0.4731 | + |
| `bye_overval_both_bye_sanity` | 297 | −0.031 | [−0.435, +0.356] | 0.4424 | 0.4374 | null (two-sided) |
| `bye_overval_fade_full_slate_post2011` | 509 | +0.568 | [−0.408, +1.549] | 0.8704 | 0.8980 | + |

Reading (**measured** numbers, **inferred** interpretation):

- The era contrast the family predicted is present in point estimate and
  direction: home-edge cell leans negative post-CBA (P+ 0.0637 against the
  bye-holding home side) and positive-leaning pre-CBA (P+ 0.7070), i.e. the
  two eras lean opposite ways exactly as the overvaluation hypothesis
  requires — but neither interval is resolved on its own.
- The full-slate fade arm is the strongest cell at P+ 0.8704 (season-blocked
  0.8980), direction-consistent with pooling cells 1 and its road mirror;
  unresolved.
- The legacy Sung & Tainsky 73% ATS claim shows no modern residue: the
  road-favorite-off-bye cell sits almost exactly at null (effect −0.013,
  P+ 0.4614) against a frozen + that would have been supported by any
  surviving legacy effect.
- Instrument sanity passed: both-teams-off-bye reads null (P+ 0.4424),
  impeaching nothing.
- No week-blocked interval sits entirely below zero anywhere, so
  `wrong_sign_resolved` applies to NO cell; no positive-control bound was run.
  All five cells are category 3, `unresolved_below_power`, recorded via the
  exact `nfl-ats weak-signals record` commands returned by the agent (the
  script itself writes neither registry JSON). Per AGENTS.md these are
  reported as `probability_positive`, never as "contains zero".

## Correction 2026-08-22: cross-season bye-map bug fixed (measured)

Red-team audit (`docs/edge_audit_redteam.md`, claim 4) found
`build_bye_maps` sorted each team's games ACROSS seasons, so every season
opener inherited a >=12-day gap from the prior season's finale and was
misflagged "off bye". Fixed (measured this session): gaps are now computed
within `(team, season)` groups, so openers get no prior game and are never
off-bye; regression test `tests/test_bye_overvaluation_screen.py` pins this.
Re-run at seed 20260822 on the same snapshot
(`artifacts/bye_overvaluation_screen/post_fix_seed20260822/results.json`;
seed-matched control at the original seed 20260821 in
`post_fix_seed20260821/` isolates fix effect from resampling noise — both
runs agree). Old-vs-new, week-blocked primary:

| Cell | n_flag | Effect old→new (pts) | P+ old→new | Materially changed? |
| --- | --- | --- | --- | --- |
| `bye_overval_home_edge_post2011` | 245→238 | −0.330→−0.347 | 0.0637→0.0551 | No (<0.05 pts, ΔP+ 0.009) |
| `bye_overval_home_edge_pre2011` | 58→58 | +0.271→+0.271 | 0.7070→0.7098 | No (identical flag set) |
| `bye_overval_road_fav_post2011` | 103→102 | −0.013→−0.028 | 0.4614→0.4160 | Yes (ΔP+ 0.046 > 0.02) |
| `bye_overval_both_bye_sanity` | 297→48 | −0.031→+0.012 | 0.4424→0.5599 | Yes (n_flag collapsed 297→48; ΔP+ 0.117) |
| `bye_overval_fade_full_slate_post2011` | 509→498 | +0.568→+0.551 | 0.8705→0.8375 | Yes (ΔP+ 0.033 > 0.02) |

The red team predicted `both_bye_sanity` was affected (**measured**: correct —
the buggy map flagged all 32 openers as off-bye, manufacturing ~249 fake
both-off-bye games; the sanity cell's null read was an artifact of that
pollution, and it still reads null after the fix). The red team also called
the fade-full-slate cell "insulated" (**measured**: not fully — its n_flag
moved 509→498 and bye week-blocks 170→152 because opener misflags suppressed
genuine XOR edges in week-1-adjacent blocks); the corrected effect +0.5508 /
P+ 0.8375 matches the red team's independent within-season resample
(+0.5508, P+ 0.834) to the fourth decimal on the point estimate. No
classification changes: all five cells remain predeclared
`unresolved_below_power`; no interval shape supports any terminal ground.
Three materially changed cells need `nfl-ats weak-signals record --replace`
lines (returned by the agent; the script writes neither registry JSON).
Cells 1 and 2 are within bootstrap noise of their registered entries and are
NOT re-recorded.

## Provenance tags used in this document

- **measured**: run/read locally this session (command given).
- **read**: file opened just now (path given).
- **reported**: literature/paper claims quoted secondhand — NOT verified.
- **inferred**: reasoning or prediction, labelled as such.
