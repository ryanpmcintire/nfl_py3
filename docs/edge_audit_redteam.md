# Edge audit — red team (2026-08-22)

Adversarial re-attack of the four strongest recent edge claims using methods
DIFFERENT from each original screen: fresh seeds (base 20260822 vs the
screens' 20260821), different blocking, different estimators (season-blocked
lower-bound selection, leave-one-season-out CV, within-week permutation null,
leave-one-team/season-out, win-rate-stratified control, distance-controlled
LPM with week-clustered SEs, sham-bye placebo). Attribution on already-scored
archive data and fresh-seed resamples only; **no rotation-registry window was
spent**. Nothing here is a fresh confirmation of anything.

Artifact of record: `artifacts/edge_audit_redteam/20260822T040806Z/results.json`
(measured this session; run stamped to `registry/experiments/edge-audit-redteam/`).
Code: `scripts/edge_audit_redteam.py`. New cells carry the `redteam_` prefix;
recording is via explicit `nfl-ats weak-signals record` calls returned at the
bottom, not written by the script. Red-team decompositions are correlated with
their parent signals — never pool as independent.

## Claim 1 — Overlay composition holdout (~+1pt fair expectation; rho 0.72)

Independent rebuild (measured): 127-subset delta matrix rebuilt from
`artifacts/opener_evaluation/20260819T174244Z/per_game.parquet` via the same
overlay machinery with redteam seeds — Spearman rho 0.7207, shrinkage slope
0.6356, global-max subset +2.0625 pts: all three match the original artifact
(reported in `artifacts/overlay_selection_holdout/20260821T195512Z/result.json`;
that file I read, did not re-execute).

Attack results:

- **Season-blocked-only selection** (max season-blocked bootstrap LOWER bound
  instead of raw mean argmax). Forward picks the SAME subset as the original
  (coach_fade+division_revenge+spread_gap_zone), holdout +0.8761 pts,
  week-blocked P+ 0.6631. Reverse's conservative criterion collapses to
  arrest-only, holdout +0.2841 pts, P+ 0.6437 (season-blocked P+ 0.9624).
  The conservative estimator does NOT unlock more out-of-sample edge than the
  original read showed. Attack failed.
- **Leave-one-season-out CV of subset choice** (select argmax on five seasons,
  score the frozen choice on the held-out sixth; all six folds). Fold deltas:
  +2.73 (2020), +0.85 (2021), +1.21 (2022), −1.13 (2023), −4.51 (2024),
  +1.50 (2025). Pooled held-out delta over all 1,503 held-out games:
  **0.0000 pts**, week-blocked [−2.1462, +2.1348], P+ 0.4930. The selection
  PROCEDURE carries no measurable out-of-sample expectation on this archive;
  the fold numerators sum to exactly zero net flips (600+200+300−300−1200+400),
  verified arithmetically. This attack damaged the "~+1pt fair expectation"
  framing: the fair expectation of picking subsets out-of-sample here measures
  as a coin flip.
- **Within-week flip-shuffle permutation null for rank stability** (each
  member's flip indicator independently shuffled among games inside its week;
  400 draws, seed 20260837). Observed rho 0.7207 vs null mean 0.4801,
  sd 0.2938, q95 0.8534 → one-sided empirical p 0.2375. rho=0.72 does NOT
  survive this null: most of the rank stability is reproduced by weekly
  flip-count structure alone.

Verdict: **category 3, unresolved — with a warning**. No terminal ground is
admissible (the LOSO interval does not sit wholly below zero; no positive
control was run against it), so the line stays open per AGENTS.md, but the
composition-selection *expectation* claim should be downgraded from ~+1pt to
"unresolved, best direct estimate 0.00 pts, P+ 0.493" and the rho statistic
should not be quoted as independent evidence without its shuffle-null p-value.

## Claim 2 — NFL.com Friday out_count>=2 (−2.69 pts full-slate, P(direction) .976)

Independent reproduction (measured, seed 20260843, same flags rebuilt through
the screen's own point-in-time pipeline): raw gap −4.4945 pts, P+
(home-cover side) 0.0226 → P(direction negative) 0.9774; matches the original
artifact (`artifacts/nflcom_friday_designation_screen/20260821T224931Z/results.json`,
read) at −4.4945 / 0.024.

Attack results:

- **Leave-one-team-out**: dropping any one of 32 teams moves the gap only
  within [−5.553, −4.047] pts; 0/32 sign flips. **Leave-one-season-out**:
  2022 −5.555, 2023 −3.647, 2024 −4.445 — every season negative. Not
  concentrated anywhere. Attack failed.
- **"Bad teams have injured players" control** (prior within-season win rate,
  pregame-safe, quartile-stratified): flag rate falls monotonically across
  win-rate quartiles (65.8% → 64.8% → 59.4% → 52.9%), confirming flagged
  teams ARE worse — but the cover gap persists inside strata: −3.87 / −8.21 /
  −6.96 / +0.36 pts. Flag-weighted within-stratum adjusted gap −4.869 pts vs
  unadjusted −4.483 on the same subset: controlling for team quality makes the
  effect slightly STRONGER. Only the top-quartile stratum (n_flag=209) loses
  it. Attack failed.
- **Starter-proxy decomposition**: >=2 Outs among starter-caliber players
  (>=50% prior-week snap share): raw gap −12.386 pts, n=162 team-games,
  full-slate −1.2748 pts, scaled CI [−2.0550, −0.5122], P+(negative)
  0.9995. >=2 Outs with ZERO starter-caliber players: raw gap −0.616 pts,
  n=458, full-slate −0.1792, P+ 0.4119 — null. Yes, the starter proxy is
  driving it, and that is the mechanism-consistent direction (impact players,
  not depth bodies). Attack failed and returned supporting structure.

Verdict: **all three attacks failed to kill → SUPPORTING evidence**. Still
category 3 overall (small-n starters cell; three seasons only); new cells
recorded below. The concentration finding strengthens the mechanism case.

## Claim 3 — Night body-clock west-road (P(direction negative) ~0.92)

Independent reproduction (measured, seed 20260853): west-body-clock road,
kickoff >=20:00 ET, 2009–2025: raw home-cover gap −6.214 pts, P+ 0.0828 →
P(direction negative) 0.9172, n=119; matches
`artifacts/body_clock_night_screen/20260821T222542Z/results.json` (read).

Attack — is it just long-distance travel? Haversine join of away team's modal
home stadium to venue via `registry/stadium_coordinates.json`, then:

- **Distance-controlled LPM** among ALL night road games (n=817, 293
  week clusters): west-body-clock visitor coefficient **−10.7003 pts**
  (cluster SE 4.9337, z −2.17) holding travel distance constant; the distance
  coefficient itself is +3.43 pts/1000mi (z +1.16) — the wrong sign to explain
  the west gap. Distance terciles among night games: short (<501 mi) +1.11
  (n_west 29), mid (501–1086) −17.73, long (1086–2686) −14.42. Attack failed:
  the composite flag survives distance control.
- **Specificity warning**: within west-body-clock road teams ONLY (n=744,
  day+night), the night increment is −3.9272 pts (SE 5.0005, z −0.79);
  matched comparison at >=median road distance (807 mi): night −3.39 pts vs
  day (n_night 73 / n_day 504). The NIGHT-specific component beyond "west
  road team, far from home" is unresolved at this sample size — consistent
  with the night screen's own dose ladder but not separately established.

Verdict: **attack failed → SUPPORTING for the registered composite flag**
(`body_clock_night_west_road_ge2000et` stays unresolved_below_power), with an
explicit warning that the dose decomposition between "west road" and "night"
remains unresolved (new LPM cells recorded below).

## Claim 4 — Bye fade post-2011 (fade-full-slate P+ 0.870)

Instrument correction first (measured): `scripts/bye_overvaluation_screen.py`
`build_bye_maps` sorts each team's games ACROSS seasons, so every season
opener inherits a >=12-day gap from the prior season's finale and counts as
"off bye" (cross-season map flags opener team-games the within-season map
does not). The fade-full-slate cell is insulated from this bug (week-1 blocks
can never produce an XOR edge), and my fresh-seed cross-season-map resample
reproduces the original (+0.5514, P+ 0.843, n=509 vs reported +0.5680 /
0.87045 in `artifacts/bye_overvaluation_screen/predeclared_run/results.json`,
read). **The both-off-bye sanity cell is NOT insulated** — flag for the owner;
not adjudicated here.

Sham-bye placebo (within-season bye maps; each team's true strict-bye week
shifted +/-2 weeks within its season, random direction per team-season,
clipped; 100 draws, seeds 20261022+, identical cell construction):

- Real assignment (measured): effect +0.5508 pts, P+ 0.8340, n_flag 498 of
  2171 post-era bye-week-block games.
- Sham draws: mean effect −0.2083 pts, sd 0.3144, q2.5 −0.7565, q97.5
  +0.4434; observed +0.5508 exceeds the sham 97.5th percentile;
  P(placebo draw >= observed) = 0.01.

A real bye-rest mechanism should vanish under sham assignment and it does:
sham schedules produce a slightly NEGATIVE centered distribution while the
real schedule sits above 99% of sham draws. Verdict: **placebo attack failed
→ SUPPORTING** for the fade-full-slate cell's direction; still category 3
(uncorrected mined family, ~2.4-pt raw gap on 498 flagged games), recorded
below. Note the first placebo pass this session used the corrupted
cross-season bye map and produced garbage (sham n_flag ~24 vs 509) — caught
by inspecting flag counts before believing any number; the corrected result
above supersedes it and nothing from that pass is used.

## Verdict table

| # | Claim | Independent number | Attacks | Killed? | Verdict |
|---|-------|--------------------|---------|---------|---------|
| 1 | Overlay composition holdout | rho 0.7207, slope 0.6356 (rebuilt) | season-blocked-lower-bound selection: survived; LOSO-CV of subset choice: pooled 0.0000 pts, P+ 0.493 — damaged the expectation claim; within-week shuffle null: rho p=0.2375 — damaged the rho claim | Partially | unresolved-with-warning |
| 2 | NFL.com Friday out_count>=2 | raw gap −4.4945 pts, P(direction neg) 0.9774 | LOO-team, LOO-season, bad-team stratified control, starter decomposition — all survived | No | SUPPORTING (still category 3) |
| 3 | Night body-clock west-road | raw gap −6.214 pts, P(direction neg) 0.9172 | haversine distance control: survived (−10.70 pts, z −2.17 conditional on distance); night-specific increment unresolved (−3.93, z −0.79) | No | SUPPORTING with specificity warning |
| 4 | Bye fade post-2011 | fade-full-slate +0.5508 pts, P+ 0.8340 (within-season map) | sham-bye placebo x100: real beats 99% of shams | No | SUPPORTING (still category 3) |

No terminal classification was reached for any claim; nothing is closed.

## Record lines (run these verbatim; owner-executed, not written by the script)

```powershell
nfl-ats weak-signals record --name redteam_overlay_subset_loso_cv --league nfl --effect-units accuracy_points --effect 0.0000 --interval-low -2.1462105969148224 --interval-high 2.134792131113663 --probability-positive 0.4930 --sample-games 1503 --sample-blocks 107 --season-start 2020 --season-end 2025 --classification unresolved_below_power --source artifacts/edge_audit_redteam/20260822T040806Z/results.json --description "Red-team leave-one-season-out CV of the 127-subset overlay composition choice: select argmax-mean subset on five seasons, score frozen on the held-out sixth; pooled held-out paired delta vs unflipped baseline" --classification-evidence "Direct out-of-sample test of the selection procedure; interval shape is the expected small-signal form and no admissible closing ground applies (no resolved wrong sign, no positive-control bound)" --notes "WARNING: damages the '~+1pt fair expectation' framing - best direct estimate is a coin flip (P+ 0.493). Correlated with overlay_subset_* entries; never pool as independent. Week-blocked bootstrap seed 20260836."

nfl-ats weak-signals record --name redteam_nflcom_out2_starters_only --league nfl --effect-units accuracy_points --effect -1.2747875354107643 --interval-low -2.05502938632516 --interval-high -0.5121509875580299 --probability-positive 0.0005 --sample-games 1574 --sample-blocks 54 --season-start 2022 --season-end 2024 --classification unresolved_below_power --source artifacts/edge_audit_redteam/20260822T040806Z/results.json --description "Red-team decomposition of nflcom_friday_out_count_ge2: >=2 Out designations restricted to starter-caliber players (>=50% prior-week snap share); full-slate scaled team-game cover gap" --classification-evidence "Supporting decomposition of the parent signal (entire interval on the claimed negative side, P(negative)=0.9995); small n_flag=162 keeps it category 3, no terminal ground applies" --notes "Starter proxy IS driving the parent effect, mechanism-consistent (impact players, not depth). Subset/decomposition of nflcom_friday_out_count_ge2 - never pool as independent. Seed 20260844."

nfl-ats weak-signals record --name redteam_nflcom_out2_nonstarters_only --league nfl --effect-units accuracy_points --effect -0.17921146953404937 --interval-low -1.9873828160789495 --interval-high 1.6776375226744789 --probability-positive 0.4119 --sample-games 1574 --sample-blocks 54 --season-start 2022 --season-end 2024 --classification unresolved_below_power --source artifacts/edge_audit_redteam/20260822T040806Z/results.json --description "Red-team decomposition of nflcom_friday_out_count_ge2: >=2 Out designations with ZERO starter-caliber players; full-slate scaled team-game cover gap" --classification-evidence "Null decomposition arm (P+ 0.41); interval crosses zero which is never grounds to close; no admissible terminal ground applies" --notes "Null counterpart of redteam_nflcom_out2_starters_only. Decomposition - never pool as independent. Seed 20260845."

nfl-ats weak-signals record --name redteam_body_clock_night_west_road_distance_controlled_lpm --league nfl --effect-units accuracy_points --effect -10.700346933382911 --interval-low -20.37018304978242 --interval-high -1.0305108169833996 --probability-positive 0.0151 --sample-games 817 --sample-blocks 293 --season-start 2009 --season-end 2025 --classification unresolved_below_power --source artifacts/edge_audit_redteam/20260822T040806Z/results.json --description "Red-team distance control for body_clock_night_west_road_ge2000et: LPM of home_cover on west-body-clock visitor + haversine travel distance (per 1000mi) among night (>=20:00 ET) road games, week-clustered SEs; coefficient on the west flag" --classification-evidence "Attack FAILED to kill the parent: west-night gap survives holding travel distance constant (z=-2.17), distance coefficient has the wrong sign to confound; whole interval on claimed side but no terminal classification is warranted for a supporting control" --notes "Answers 'is west-road-at-night just long-distance road': NO at this resolution. Correlated with body_clock_night_west_road_ge2000et; never pool as independent. Normal-approximation P+ from clustered LPM, not block bootstrap."

nfl-ats weak-signals record --name redteam_body_clock_night_increment_within_west_road_lpm --league nfl --effect-units accuracy_points --effect -3.927223948341855 --interval-low -13.728004655944511 --interval-high 5.8735567592608025 --probability-positive 0.2161 --sample-games 744 --sample-blocks 289 --season-start 2009 --season-end 2025 --classification unresolved_below_power --source artifacts/edge_audit_redteam/20260822T040806Z/results.json --description "Red-team specificity split: LPM of home_cover on night kickoff (>=20:00 ET) + distance among WEST-body-clock road teams only (day+night), week-clustered SEs; coefficient on the night flag" --classification-evidence "Night-specific component beyond being a west road team is unresolved (z=-0.79); interval crosses zero which is never grounds to close; no admissible terminal ground applies" --notes "WARNING for claim-3 consumers: the dose decomposition between 'west road' and 'night' is not separately established; matched >=807mi night-vs-day gap is -3.39 pts. Correlated with body_clock_* family; never pool as independent."

nfl-ats weak-signals record --name redteam_bye_overval_fade_full_slate_withinseason_bye_map --league nfl --effect-units accuracy_points --effect 0.5508134037685816 --interval-low -0.5610627041254037 --interval-high 1.6260017138001357 --probability-positive 0.834 --sample-games 2171 --sample-blocks 179 --season-start 2012 --season-end 2025 --classification unresolved_below_power --source artifacts/edge_audit_redteam/20260822T040806Z/results.json --description "Fresh-seed resample of bye_overval_fade_full_slate_post2011 with a WITHIN-SEASON strict-bye map (the screen's build_bye_maps spans seasons and misflags every opener as off-bye; the fade cell is insulated from that bug)" --classification-evidence "Reproduction/stability cell for the parent signal under a corrected instrument; near-duplicate of bye_overval_fade_full_slate_post2011 (P+ 0.870 vs 0.834), no terminal ground applies" --notes "Correlated with bye_overval_fade_full_slate_post2011 - never pool as independent. Bootstrap 4000 samples, seed 20260863."

nfl-ats weak-signals record --name redteam_bye_fade_sham_placebo_null --league nfl --effect-units accuracy_points --effect -0.20828184079702614 --interval-low -0.7564979863012067 --interval-high 0.4434405980541774 --probability-positive 0.3510675 --sample-games 2171 --season-start 2012 --season-end 2025 --classification unresolved_below_power --source artifacts/edge_audit_redteam/20260822T040806Z/results.json --description "SHAM-bye placebo for the bye fade: 100 draws shifting every team's true strict-bye +/-2 weeks within its season, fade-full-slate cell rebuilt identically; distribution of draw-level effects" --classification-evidence "Placebo-validation cell, not a tradable signal: observed real-assignment effect +0.5508 exceeds 99% of sham draws (P(placebo >= observed)=0.01), so the placebo FAILED to kill the mechanism; recording the null distribution itself, classified unresolved_below_power because no registry class fits an instrument-validation cell" --notes "NEVER POOL - validation cell only. Sham mean -0.2083, sd 0.3144, q97.5 +0.4434. Supports bye_overval_fade_full_slate_post2011's mechanism reading."
```

Deliberately NOT recorded: the claim-1 season-blocked-only selection reads
(same frozen cells as existing `overlay_subset_holdout_*` entries), the
claim-2 out2-any reproduction (duplicate of registered
`nflcom_friday_out_count_ge2`), the claim-3 west-night reproduction
(duplicate of registered `body_clock_night_west_road_ge2000et`), and the
rank-stability permutation p-value (a null-test statistic, not an effect
cell; documented above).

## Gates

- `ruff format --check scripts/edge_audit_redteam.py` — clean (measured)
- `ruff check scripts/edge_audit_redteam.py` — clean (measured)
- `mypy src` — Success: no issues found in 101 source files (measured)
- `pytest --basetemp C:\Users\Ryan\AppData\Local\Temp\opencode\pt_redteam` —
  1707 passed, 1 failed: `tests/test_experiment_registry.py::
  test_every_script_writing_artifacts_json_uses_the_provenance_helper`
  flags `scripts/backfill_vegasinsider.py`, an UNTRACKED script that predates
  this session and is outside this task's ownership; this audit's script uses
  `write_experiment_artifact` and passes that check (measured)
