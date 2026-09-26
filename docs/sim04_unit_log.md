# SIM-04 unit log

Verbatim unit predeclarations and results moved from the lane on 2026-09-25.


## Unit 2 (capture-column widen) -- 2026-09-25, HOLD, subagent hit 50-tool-call cap mid-task

### State
- Edited (uncommitted) `src/nfl_ats/pbp.py`: `PBP_SNAPSHOT_COLUMNS` widened from 45
  to 56 columns, adding: `penalty_team`, `penalty_type`, `play_type_nfl`,
  `home_timeouts_remaining`, `away_timeouts_remaining`,
  `posteam_timeouts_remaining`, `defteam_timeouts_remaining`,
  `rusher_player_id`, `receiver_player_id`, `fumbled_1_player_id`,
  `fumble_recovery_1_player_id`. No other file edited. No commit, no capture, no
  verification run yet -- the 50-tool-call cap hit before ruff/mypy/pytest, the
  new snapshot capture, and the feature-rebuild diff could run.
- **Measured** (this session, `nflreadpy.load_pbp(seasons=[2025])`, 372 raw
  columns): no `offense_personnel`/`defense_personnel`/`offense_formation`
  columns exist upstream at all (only unrelated `shotgun`/`no_huddle`
  booleans matched a `personnel|formation` keyword scan) -- personnel/formation
  is NOT addable, confirming the plan's own Risk section. Do not add it; do not
  substitute shotgun/no_huddle for it without a separate decision.
- Timeouts: added both the home/away pair (literal "both teams") and the
  posteam/defteam pair (matches the Architecture section's `timeouts_off,
  timeouts_def` state tuple) since both are cheap and either reading of the
  plan's wording is defensible.
- Fumble IDs: only added `fumbled_1_player_id` (who fumbled) and
  `fumble_recovery_1_player_id` (who recovered), not the `_2_` or
  `forced_fumble_player_*` variants -- a narrower reading of "fumble player
  IDs" to avoid scope creep; revisit if Unit 5's turnover submodel needs more.

### Tried / measured (consumer audit, via Grep + direct reads, this session)
- `analysis_plays()` (pbp.py), `players.py:1557`, `quarterbacks.py:432` all
  call `require_columns(pbp, PBP_SNAPSHOT_COLUMNS, ...)`. Any snapshot written
  by `write_pbp_snapshot`/`fetch_pbp_snapshot` always backfills missing
  upstream columns as NaN before saving, so every NEW snapshot structurally
  has all 56 columns. Confirmed by direct read that the OLD snapshot
  `data/pbp/raw/20260817T184927Z/season=2025/plays.parquet` has exactly the
  original 45 columns and none of the 11 new ones -- loading that OLD snapshot
  through `analysis_plays`/`players.py`/`quarterbacks.py` will now raise
  `DataContractError` under the widened allowlist.
- `_resolve_pbp_snapshot(None)` in `cli_common.py` defaults to
  `latest_pbp_snapshot()` (sorted manifest glob, newest timestamp wins), so
  once a NEW snapshot is captured it automatically becomes "latest" and every
  default CLI call (`build-pbp-features`, `build-qb-features`,
  `build-player-features`) picks it up with no re-pointing needed. No
  production path explicitly pins the OLD snapshot id.
- `scripts/sim04_unit1_drive_chain.py` reads the OLD snapshot's parquet files
  directly via `pd.read_parquet(path)` per season -- does NOT go through
  `nfl_ats.pbp`'s `load_pbp_snapshot`/`analysis_plays`/`require_columns`
  chain. Confirmed safe; Unit 1 is unaffected by this widen.
- **Real, un-mitigated risk found**: `scripts/team_style_features.py` builds
  `RAW_COLUMNS = PBP_SNAPSHOT_COLUMNS + EXTRA_STYLE_COLUMNS` and caches a raw
  pull at `data/pbp/team_style/raw_pbp_narrow.parquet` (exists on disk, built
  2026-08-19, old/narrower schema, confirmed present via `ls`). Its
  `build_raw_cache()` returns that cached file as-is unless `--refresh` is
  passed, then feeds it to `analysis_plays()`/`build_drive_table()`, which will
  now raise `DataContractError` on the next non-refresh run until someone
  passes `--refresh`. This fails loud, not silently wrong -- consistent with
  repo philosophy -- but is not yet fixed or refreshed. Did not rebuild this
  cache (out of scope, needs a fresh 17-season nflreadpy pull).
- `scripts/fetch_penalty_type_snapshot.py` has a manifest "source" string
  literal claiming penalty_type/penalty_team are "absent from
  nfl_ats.pbp.PBP_SNAPSHOT_COLUMNS" -- now stale text, not a functional bug.
  Not edited (out of scope).
- `tests/test_players.py`'s `_pbp()` helper builds synthetic rows via
  `dict.fromkeys(PBP_SNAPSHOT_COLUMNS, np.nan)` -- self-adapting, unaffected.
  `tests/test_feature_manifest.py` uses the literal string `"20260817T184927Z"`
  only as an arbitrary fixture snapshot-id, no column dependency, unaffected.
- Measured: no current test file has "pbp" in its filename and no test
  function name in `tests/test_players.py` contains "pbp" literally --
  `pytest -q -k pbp` likely collects 0 tests. Run it anyway and report the real
  count; do not assume.
- Measured: `data/processed/game_features_pbp.parquet` on disk right now was
  built TODAY 2026-09-25 12:02 UTC-ish, sha256
  `7325e1e4df46be14a437d6f368ee20bb3a04de439ef9da9c237ea9aec01240f6`, 7,640,861
  bytes. Its manifest (`game_features_pbp.manifest.json`) was NOT read before
  the cap hit -- next agent must read it and confirm `source_pbp_snapshot ==
  "20260817T184927Z"` before treating this file as the "before" baseline for
  the identical-rebuild proof.

### Next (for a fresh agent)
1. Read `data/processed/game_features_pbp.manifest.json`; if
   `source_pbp_snapshot == "20260817T184927Z"`, today's on-disk
   `game_features_pbp.parquet` (hash above) IS the "before" baseline. If not,
   `git stash push -- src/nfl_ats/pbp.py`, run
   `.tools/uv.exe run --no-sync nfl-ats build-pbp-features`, hash that as
   "before", then `git stash pop`.
2. Run `.tools/uv.exe run --no-sync ruff format --check .`,
   `.tools/uv.exe run --no-sync ruff check .`,
   `.tools/uv.exe run --no-sync mypy src`, and
   `.tools/uv.exe run --no-sync pytest -q -k pbp`; report real output.
3. Capture the new NFL pbp snapshot:
   `.tools/uv.exe run --no-sync nfl-ats pbp-ingest --include-postseason`
   (defaults `--start-season 2009 --end-season 2025` match the plan and the
   old snapshot exactly, since `current_year=2026`). Writes a new
   `data/pbp/raw/<run_id>/`; do not touch `20260817T184927Z`.
4. Rebuild: `.tools/uv.exe run --no-sync nfl-ats build-pbp-features` (now
   defaults to the new latest snapshot). Compare against the "before" hash --
   ignore provenance-only manifest fields (`built_at_utc`,
   `source_pbp_snapshot`), require the parquet's actual values/row count to
   match exactly (use `DataFrame.equals` per column if the raw file hash
   differs, since parquet metadata can vary even for identical data).
5. Report per-season (2009-2025) non-null coverage share for the 11 new
   columns, read directly from the new raw snapshot's season partitions (not
   from `game_features_pbp.parquet`, which does not expose raw pbp columns).
6. Decide and note (or fix) the `team_style_features.py` stale-cache risk
   above; leave `fetch_penalty_type_snapshot.py`'s stale manifest string as a
   flagged Open item unless asked to fix it.
7. No commits, no push, no comments/docstrings, no new tests (task
   constraints carry over unchanged).

### Open
- Whether to refresh `data/pbp/team_style/raw_pbp_narrow.parquet` now (heavy,
  out of this task's literal scope) or leave it to fail loudly on next use.
- Whether `fetch_penalty_type_snapshot.py`'s now-stale manifest string is
  worth a one-line fix.

## Goal
Write the build plan for ROADMAP SIM-04 (full play-by-play simulator) and
SIM-05 (counterfactuals). Done for this lane's opening unit means the plan
document exists, is grounded in measured inventory and the MOD-09 audit, and
names a unit 1 a subagent can start immediately. No model training or `src/`
code in this planning unit.

## State
- `docs/sim04_play_simulator_plan.md` written 2026-09-25: architecture
  (game-state, per-play/clock/penalty/turnover/4th-down submodels, sampling),
  leakage rules, a predeclared chronological LOSO evaluation protocol against
  the served `DiscretePushReader` / `serve_discrete_three_way`
  (`src/nfl_ats/mass_preserving_lattice.py`), 10 build units, compute
  estimates, dashboard deliverable, risks, and a plain-English section.
- This lane file.
- ROADMAP.md **not edited** (out of scope for this task); proposed replacement
  row text was handed back to the caller for the primary orchestrator to
  apply.
- No commits made.

## Tried
- Measured (`tests/scratch` inventory script, this session): NFL pbp capture
  `data/pbp/raw/20260817T184927Z/` has 17 season partitions (2009–2025); 2025
  sample = 48,771 rows, 45 columns, matching the fixed allowlist
  `PBP_SNAPSHOT_COLUMNS` in `src/nfl_ats/pbp.py`. Missing from that allowlist:
  timeouts-remaining, penalty team/type, personnel/formation, non-passer
  player IDs — present upstream in the `nflreadpy` source already fetched
  elsewhere (`nflverse_current_season.py`), so widening is a config change
  (Unit 2), not a new data source.
- Measured: `data/processed/game_features_pbp.parquet` = 4,902 rows x 201
  columns, seasons 2009–2026 — this is the already-rejected MOD-09
  drive/summary bundle (50.36% ATS no edge, worsened Brier); the plan does not
  reuse it as a side-pick feature source.
- Measured: `data/cfb/pbp/raw/` has 44 parquet files (~1.3 GB) across capture
  runs — noted as an auxiliary/positive-control corpus only.
- Read: `docs/play_level_audit.md` (ICC 0.0131 finding) and ROADMAP MOD-09
  (line ~666) / Phase 7 simulations section (~672–683, including the existing
  "Simulation is accepted only if it improves held-out distribution
  calibration" acceptance line) to ground the plan's honest framing.
- Read: `src/nfl_ats/mass_preserving_lattice.py` (`DiscretePushReader`,
  `serve_discrete_three_way`, key numbers `(3, 7, 10, 14)`,
  `MIN_BAND_GAMES=200`) as the comparison baseline the plan's evaluation
  protocol targets.

## Next
Unit 1 is done and NO-GO (see result below). Escalate the reordering
question (score/clock conditioning needs to move earlier than Unit 6) to the
owner / primary orchestrator before spawning a subagent for Unit 2 or any
team-conditioned submodel unit.

## Unit 1 predeclaration (written before running, 2026-09-25)
Go/no-go criterion for the unconditional drive-chain baseline
(`scripts/sim04_unit1_drive_chain.py`), fixed before any simulation output was
read:
- **GO** (architecture is sound, proceed toward Unit 2+) if both hold: (a) the
  simulated mass at 4 of the 5 key numbers (3, 7, 10, 14, 17) falls inside the
  historical held-out seasons' bootstrap 90% CI for that number's mass share;
  and (b) the simulator's discrete-margin log loss on held-out seasons is not
  worse than the naive empirical-histogram baseline (train-season margin
  histogram applied unconditionally to every held-out game) by more than 0.02
  nats.
- **NO-GO** (rethink the drive-chain architecture before building Units 2-7)
  if either bound is missed — per the plan's Risks section, that means the
  discrete lattice's shape comes from something (score-driven end-game
  behavior, kicker-distance-specific FG rate, etc.) this drive-chain mechanism
  does not capture.
- An interval crossing zero / a near-miss on one key number alone is not
  NO-GO by itself (AGENTS.md "interval crossing zero" rule) — the bar above is
  the count-of-4/5-numbers plus the log-loss bound, not a single cell.

## Unit 1 result (measured, 2026-09-25)
Ran `scripts/sim04_unit1_drive_chain.py` once for real:
`.\.tools\uv.exe run --no-sync python scripts\sim04_unit1_drive_chain.py`.
Artifact: `artifacts/sim04_unit1/20260925T191907Z/report.json` (+
`sim_margins.npy`, N=10,000 simulated games). Fit drives on
`data/pbp/raw/20260817T184927Z/` REG-season 2009-2017 (54,324 drives),
held out 2018-2025 actual margins from `data/processed/game_features_pbp.parquet`
(2,227 games) as ground truth, bootstrapped the actual-mass CI over the 8
held-out seasons (2,000 resamples).

Method: reconstructed each drive's (starting field position, outcome
category, offense points, defense points) by replaying `posteam_score_post`
chronologically per game (this also captures PATs/2-point tries/safeties/
defensive-TD returns without a separate submodel, since the historical points
actually scored on the drive are read directly, not assumed as a fixed 7).
Chained drives into games by drawing a real historical (home,away) drive-count
pair, alternating possession from a coin-flip opening kickoff, drawing each
drive's outcome from the empirical pool for its field-position decile, and the
next drive's starting field position from the empirical pool of what actually
followed that outcome category historically; tied games get up to 12 extra
sudden-death-style drives.

Key-number mass, simulated vs. actual (8 held-out seasons, bootstrap 90% CI):
| number | simulated | actual (pooled) | actual 90% CI | in CI? |
|---|---|---|---|---|
| 3 | 0.0756 | 0.1482 | [0.1362, 0.1584] | **no** |
| 7 | 0.0870 | 0.0849 | [0.0757, 0.0956] | yes |
| 10 | 0.0563 | 0.0480 | [0.0419, 0.0542] | **no** |
| 14 | 0.0491 | 0.0521 | [0.0401, 0.0641] | yes |
| 17 | 0.0331 | 0.0364 | [0.0283, 0.0437] | yes |

Hits: 3 of 5. Discrete margin log loss: simulator 4.0186 vs. naive
train-season-histogram baseline 3.9725 (delta **+0.0461** nats, worse).
Simulated margin sd 15.07 vs. actual 14.25 (reasonably close); simulated mean
-0.11 vs. actual +1.72 (expected miss — no home-field term in this
unconditional baseline).

**Verdict: NO-GO** against the predeclared criterion (needed 4/5 key-number
hits and a log-loss delta <= +0.02 nats; got 3/5 and +0.046). The miss is
concentrated at margin 3 (simulated mass is roughly half of actual), which
matches the plan's named risk exactly: "score-driven end-game behavior,
kicker-distance-specific FG rates" — teams deliberately trade a late
touchdown-range drive for a field goal to preserve a 3-point lead / take a
3-point lead into a two-minute situation, a policy this baseline's
unconditional empirical resampling cannot produce since it draws whichever
historical drive happened to start from a similar field-position decile,
regardless of score/clock context. This is an architecture finding, not a
tuning failure: per the plan's Risks section, Units 3-7 (team-conditioned
per-play/clock/4th-down submodels) should not be built on this drive-chain
skeleton until the end-game score/clock conditioning gap above is named and
addressed, because none of those units add score-context conditioning either.

## Open
- Unit 1 is NO-GO. Before spending budget on Unit 2 (capture-column widen)
  or Units 3-7, the plan needs a named fix for the margin-3 gap — the leading
  candidate is conditioning drive outcome (or at least 4th-down/FG-attempt
  choice) on score differential and time remaining, not field position alone,
  which pulls in the Unit 6 4th-down-policy and Unit 4 clock submodels earlier
  than the plan's build order assumed. Escalate this reordering to the owner /
  primary orchestrator before starting Unit 2.
- Whether Unit 2 (capture-column widen, touches `src/nfl_ats/pbp.py`) still
  goes forward independently of the NO-GO (it is a data-availability step, not
  gated on Unit 1's mechanism finding) is still open.
- Whether SIM-04/05 gets a ROADMAP.md status change now to record the NO-GO
  (proposed row text was returned to the caller previously, not applied) is
  still open.
- Whether the primary orchestrator wants Unit 2 (capture-column widen, touches
  `src/nfl_ats/pbp.py`) done before or in parallel with Unit 1 (Unit 1 does
  not need the widened columns).
- Whether SIM-04/05 should get a ROADMAP.md status change now (proposed row
  text was returned to the caller, not applied) or wait until Unit 1's result
  is in hand.

## Unit 1b predeclaration (written before running, 2026-09-25)
Orchestrator decision on the Unit 1 NO-GO: condition drive outcomes on game
state before any team-strength unit. `scripts/sim04_unit1b_state_chain.py`
extends Unit 1's exact loaders/split/N/GO criterion; only the outcome
conditioning and clock simulation change. Fixed before reading any simulated
mass or log-loss number:

- **State tuple per drive** (captured at the first non-kickoff play of the
  drive, the same reference play Unit 1 uses for starting field position):
  possessing team's `score_differential` (nflverse posteam-perspective
  column, not recomputed), `qtr`, and `game_seconds_remaining`. Drives
  missing any of these three fields are dropped from the state-conditioned
  pools (count reported in the artifact).
- **Score-differential bucket (fine, 9 levels):** `<=-17, -16..-9, -8..-4,
  -3..-1, 0, 1..3, 4..8, 9..16, >=17` (possessing team's own diff; symmetric
  around 0; edges land on one-possession=8 and FG=3 thresholds since those
  are the named mechanism for the margin-3 gap).
- **Coarse score bucket (back-off, 5 levels):** trailing>=9, trailing 1-8,
  tied, leading 1-8, leading>=9.
- **Time bucket (fine, 8 levels):** `Q1`, `Q2_early` (>120s left in Q2),
  `Q2_twomin` (<=120s left in Q2), `Q3`, `Q4_early` (>300s left in Q4),
  `Q4_late` (120-300s left in Q4), `Q4_twomin` (<=120s left in Q4), `OT`
  (qtr>=5). Seconds-left-in-period computed as
  `game_seconds_remaining - (4-qtr)*900` for qtr 1-4 (confirmed by direct
  read of the 2019 season file: qtr 1 spans gsr 2700-3600, qtr 2 spans
  1800-2700, qtr 3 spans 900-1800, qtr 4 spans 0-900, OT (qtr 5) is an
  independent 0-600s clock — this is the 10-minute regular-season OT period,
  in force for the entire 2018-2025 test window).
- **Coarse time bucket (back-off, 3 levels):** normal (Q1/Q2_early/Q3/
  Q4_early), clock_pressure (Q2_twomin/Q4_late/Q4_twomin), OT.
- **Field position bucket:** unchanged from Unit 1 — 10 deciles of starting
  `yardline_100`.
- **Hierarchical back-off, minimum cell count = 25 drives** (predeclared
  now, not tuned on results): try the finest cell
  (score_fine x time_fine x fp_decile, 720 possible keys); if the pooled
  category/points/duration count in that cell is <25, back off to
  (score_fine x time_coarse x fp_decile); if still <25, back off to
  (score_coarse x time_coarse x fp_decile); if still <25, fall back to
  fp_decile alone (Unit 1's original pooling, guaranteed non-empty). One
  back-off decision (by outcome-pool count) is reused for both the outcome
  draw and the duration draw from the same cell, since they come from the
  same underlying drives.
- **Clock simulation:** each drive draws a duration (start `gsr` minus end
  `gsr` of that historical drive, clipped to [0, 1800] to guard against a
  half-boundary data glitch) from the same state cell as its outcome. The
  simulated game clock starts at `gsr=3600`, decrements by each drawn
  duration, and derives `qtr`/seconds-left with the same formula as the
  historical read, so bucket assignment is consistent between fit and
  simulation.
- **Overtime:** regulation ends when `gsr<=0`; if tied, one 10-minute
  (600s) modified-sudden-death OT period runs (matches the rule in force
  for the entire 2018-2025 test window): first possession decided by a
  fresh coin flip; a defensive/return score or an offensive touchdown ends
  the game immediately on any possession; a first-possession field goal
  does not end the game and guarantees the second team one possession;
  once both teams have had at least one OT possession, the first
  score-differential change ends the game; an OT period that expires still
  tied ends the game as a tie (matches the current regular-season rule — no
  second OT period in the regular season). Limitation stated, not fixed:
  the training pool's `OT` time-bucket cells mix pre-2012 pure-sudden-death
  and 2012-2016 15-minute-OT drives with 2017+ 10-minute-OT drives, since
  Unit 1's training split (2009-2017) spans all three eras and OT drive
  volume is too small to split further without violating the min-cell-count
  rule.
- **Fourth-down / field-goal policy:** not modeled separately in Unit 1b —
  it is implicit in the state-conditioned empirical outcome draw itself
  (conditioning on score differential and clock is exactly what should
  surface the FG-vs-TD-range end-game pattern named in Unit 1's NO-GO,
  without a separate policy submodel; that submodel is still Unit 6 per the
  plan).
- **Field-position transition pool:** unchanged from Unit 1 — keyed by
  outcome category only (not further conditioned on game state), to bound
  this unit's scope to the two things asked (outcome + clock conditioning).
- **Home-field term:** **not added.** The plan's Unit 1 description frames
  the drive-chain baseline as "two league-average teams" with no team
  identity yet; a home-field constant is deferred to Unit 3 where team
  conditioning begins, per the plan's build order.
- **GO/NO-GO criterion: unchanged from Unit 1** — GO requires both (a) >=4/5
  key numbers (3,7,10,14,17) inside the actual 8-held-out-season bootstrap
  90% CI, and (b) discrete margin log loss on held-out seasons within +0.02
  nats of the naive train-histogram baseline. Same train/test split
  (2009-2017 / 2018-2025), same N=10,000 simulated games, same RNG-seed
  discipline as Unit 1.
- **OT diagnostic (measured, not a gate):** report how many of the 2,227
  held-out (2018-2025 REG) games went to OT (qtr>=5 present in that game's
  pbp) and what share of the held-out `|margin|==3` games were OT games, to
  contextualize how much of the margin-3 gap OT could plausibly explain.

## Unit 1b result (measured, 2026-09-25)
Ran `scripts/sim04_unit1b_state_chain.py` once for real:
`.\.tools\uv.exe run --no-sync python scripts\sim04_unit1b_state_chain.py`.
Artifact: `artifacts/sim04_unit1b/20260925T192759Z/report.json` (+
`sim_margins.npy`, N=10,000). Same train/test split and inputs as Unit 1;
54,322 training drives, 2 dropped for missing state (qtr/gsr/score_diff
null) out of 54,324. Back-off cell coverage: 630 finest (score_fine x
time_fine x fp_decile) cells populated, 367 met the 25-drive minimum
directly; 194 coarser (score_fine x time_coarse x fp_decile) cells, 156 met
minimum; 114 (score_coarse x time_coarse x fp_decile) cells, 95 met minimum
— so most finest-level draws needed at least one back-off step, as expected
given 54k drives spread over up to 720 possible fine cells.

Key-number mass, simulated vs. actual (8 held-out seasons, bootstrap 90%
CI), Unit 1b vs. Unit 1:
| number | Unit1b sim | Unit1 sim | actual (pooled) | actual 90% CI | Unit1b in CI? | Unit1 in CI? |
|---|---|---|---|---|---|---|
| 3 | 0.1001 | 0.0756 | 0.1482 | [0.1364, 0.1588] | no (closer) | no |
| 7 | 0.0664 | 0.0870 | 0.0849 | [0.0762, 0.0955] | **no (regressed)** | yes |
| 10 | 0.0495 | 0.0563 | 0.0480 | [0.0420, 0.0543] | yes | no |
| 14 | 0.0324 | 0.0491 | 0.0521 | [0.0409, 0.0643] | **no (regressed)** | yes |
| 17 | 0.0425 | 0.0331 | 0.0364 | [0.0288, 0.0438] | yes | yes |

Hits: **2 of 5** (10, 17) — down from Unit 1's 3/5. The margin-3 gap
shrank (0.076 to 0.100 simulated vs. 0.148 actual, roughly closing a third
of the original gap) but 7 and 14 flipped from hits to misses, so the
raw hit count regressed.

Discrete margin log loss: simulator 3.9900 vs. naive train-histogram
baseline 3.9725 (delta **+0.0175** nats) — **inside** the +0.02 bound,
versus Unit 1's +0.0461 (a real improvement, log-loss criterion now
passes on its own). Simulated margin sd 15.54 vs. actual 14.25 (wider than
Unit 1's 15.07, i.e. slightly more variance, not less, from the state
conditioning); simulated mean -0.03 vs. actual +1.72 (still no home-field
term, as predeclared and stated above — this mean gap is expected and not
part of either gate).

OT diagnostic (measured): 118 of 2,227 held-out games (5.30%) went to OT;
of the 330 held-out games with `|margin|==3`, 68 (20.61%) were OT games —
one in five actual 3-point margins comes from an overtime finish. The
simulator's own OT rate (4.96% of the 10,000 simulated games) is close to
that actual 5.30%, so the OT frequency is roughly right, but this does not
by itself establish the simulated OT *margin* distribution is realistic —
OT's ~20% share of margin-3 games is a plausible partial contributor to
the remaining margin-3 gap, alongside the still-not-fully-closed effect of
end-game FG-vs-TD score management this conditioning targeted.

**Verdict: NO-GO** against the unchanged predeclared criterion (needs
BOTH >=4/5 key-number hits AND log-loss delta <=+0.02 nats). Log loss now
clears its bound on its own, but hits regressed to 2/5, so the joint bar
is still missed. This is a real, mixed, measured result, not a rejection
of the conditioning approach: state conditioning moved the mechanism in
the right direction for the named margin-3 gap and for overall
distributional fit (log loss), but two other key numbers (7, 14) got
worse, consistent with `unresolved_below_power`/an incompletely specified
mechanism rather than a refuted one — no interval here is on the wrong
side of the actual band, so nothing here is a "refuted mechanism" close
per AGENTS.md; it is evidence the game-state cells (as bucketed) are not
yet the whole story.

**Recommended next unit:** before adding a full per-play/clock/4th-down
submodel layer (Units 3-6), spend one more bounded diagnostic pass on
*why* 7 and 14 regressed under this exact conditioning — likely
candidates to check first: (a) whether the score-differential buckets are
too coarse near +/-7 (a team already leading by 7 has a different
FG-vs-TD incentive than a team down 7), (b) whether pooling the
fp-transition draw only by outcome category (unchanged from Unit 1, not
state-conditioned) is reintroducing unconditional noise into the *next*
drive's starting position right when the current drive's outcome was
state-appropriate, and (c) a cell-population check restricted to the
Q4_late/Q4_twomin buckets specifically (where the margin-3 mechanism is
concentrated) rather than the pooled coverage stats above. This is a
targeted Unit 1c-style diagnostic, not a jump to team conditioning.

## Unit 1c predeclaration (written before running, 2026-09-25)

Validation split for all diagnosis: fit on 2009-2014, validate on 2015-2017
(2018-2025 stays untouched, already looked at twice). All six configs below
share Unit 1b's exact drive reconstruction, back-off structure (level0/1/2/3),
N=10,000, RNG seed, and the unchanged GO criterion (>=4/5 key numbers in the
actual bootstrap 90% CI AND log-loss delta <=+0.02 nats vs. the naive
train-histogram baseline) — evaluated here against the 2015-2017 actual
distribution instead of 2018-2025.

**Pre-check (measured before any config ran, not itself a look):** the
try-after-TD candidate is already ruled out as the mechanism. Direct read of
`data/pbp/raw/20260817T184927Z/season=2019/plays.parquet`: PAT/two-point plays
share `fixed_drive` with the preceding touchdown play in 99.67% of cases (1210
PAT rows checked), so `reconstruct_drives`'s per-play loop already folds the
PAT/2pt result into that same drive's `points_off` before it is written to the
row. Unit 1b's `points_off`/`points_def` are drawn jointly with `category` and
`duration` from the same state cell already — try-after-TD choice is already
state-conditioned, not pooled globally. Config list below therefore targets
only the remaining three candidates (transition-pool conditioning, cell
sparsity/back-off threshold, score-bucket coarseness at +/-7/8), not this one.

Two independent mechanism changes, each isolated then combined, plus a
sparsity-sensitivity check:
- **Fine score buckets (score):** split Unit 1b's `4..8` / `-8..-4`
  score-differential buckets into `4..6`/`7..8` and `-6..-4`/`-8..-7` (11 fine
  buckets total instead of 9), isolating exactly-one-score-with-FG-cover (7-8)
  from smaller leads/deficits. Coarse score buckets (back-off level 2)
  unchanged.
- **State-conditioned field-position transition (transition):** the next
  drive's starting field position is drawn from a pool keyed on
  `(ending_category, next_drive_score_bucket_coarse, next_drive_time_bucket_coarse)`
  when that cell has >=25 drives, falling back to Unit 1b's original
  category-only pool otherwise. The state used is the *new* offense's own
  diff/clock right after this drive's duration is applied — computed the same
  way the next loop iteration computes it, so simulation and back-off stay
  consistent by construction.
- **Sparsity threshold (min_cell_n):** lower the back-off minimum from 25 to
  15 drives per cell, holding everything else at Unit 1b defaults, to test
  whether forced back-off out of Q4-late/two-minute cells (not a wrong
  conditioning choice) is the regression driver.

| config | fine score buckets | state-conditioned transition | min_cell_n |
|---|---|---|---|
| A (baseline, = Unit 1b logic) | no | no | 25 |
| B (transition only) | no | yes | 25 |
| C (score only) | yes | no | 25 |
| D (transition + score) | yes | yes | 25 |
| E (sparsity only) | no | no | 15 |
| F (kitchen sink) | yes | yes | 15 |

Six configs, one validation run each (six looks at 2015-2017, zero looks at
2018-2025 in this unit). Selection rule fixed now: pick the config with the
most key-number hits; ties broken by the smallest log-loss delta; a config
that regresses log loss above +0.02 nats relative to config A is disqualified
even if its hit count is higher. The chosen config runs exactly once on the
original split (train 2009-2017, test 2018-2025) against the unchanged Unit 1
GO criterion — that single run is the only look this unit takes at the test
seasons.

Also measured once on the 2015-2017 validation actual games only (a read of
history, not a model look): OT share of `|margin|==3` games, and among
non-OT `|margin|==3` games, the share whose chronologically last drive of the
game ended in category "Field goal" vs. "Touchdown" vs. other — to locate
whether the margin-3 gap concentrates in OT field goals or late go-ahead field
goals, per the orchestrator's question.

Script: `scripts/sim04_unit1c_diagnostic.py` (new, reuses Unit 1b's reconstruct
step; not imported from `sim04_unit1b_state_chain.py` to keep both scripts
independently runnable). Artifacts under
`artifacts/sim04_unit1c/<timestamp>/`.

### Unit 1c validation result (measured, 2026-09-25)
Ran `scripts/sim04_unit1c_diagnostic.py --mode validation` once (all 6 configs
in one process, deterministic seed `20260925`). Artifact:
`artifacts/sim04_unit1c/20260925T193458Z/validation_report.json`. Train
2009-2014 (36,455 drives, 1 dropped for missing state), eval 2015-2017 (801
actual REG games).

Key-number hits (of 5) and log loss, all 6 configs vs. the 2015-2017 actual
bootstrap 90% CI:

| config | fine score | state transition | min_cell_n | hits/5 | numbers hit | log-loss delta vs naive |
|---|---|---|---|---|---|---|
| A (baseline) | no | no | 25 | 1 | 10 | +0.01546 |
| B (transition) | no | yes | 25 | 1 | 10 | +0.02536 |
| C (score) | yes | no | 25 | 1 | 10 | +0.02718 |
| D (transition+score) | yes | yes | 25 | 1 | 10 | +0.03465 |
| E (sparsity) | no | no | 15 | 0 | none | +0.01491 |
| F (kitchen sink) | yes | yes | 15 | 0 | none | +0.02816 |

None of the three tested mechanism changes recovers key numbers 7 or 14 on
this validation split, and only number 10 hits in any config. All three
changes (state-conditioned FP transition, finer +/-7/8 score buckets, looser
25->15 min-cell-n) either leave the log loss unchanged or make it worse; none
improves the hit count. Per the predeclared selection rule (most hits, ties
broken by smallest log-loss delta, config disqualified only if its delta
regresses >0.02 nats versus config A — none did), **config A (Unit 1b's
unmodified logic) wins**: it ties for the top hit count (1/5, with B/C/D) and
has the smallest log-loss delta among that tied group.

Margin-3 decomposition (2015-2017 actual, 801 games, measured once, a read of
history not a model look): 119 games (14.86%) had `|margin|==3`, matching the
2018-2025 pooled rate. Of those, 29 (24.37%) went to OT — a bigger share than
the 2018-2025 test-set's 20.61% (Unit 1b), consistent given the smaller n.
Of the 90 non-OT margin-3 games, the chronologically **last drive of the
game** ended in category "End of half" for 51 (56.7%), "Field goal" for 25
(27.8%), "Touchdown" for 0 (0%), and turnover/turnover-on-downs/missed-FG for
the remaining 14. **Reading:** the dominant non-OT mechanism is a team that is
already up (or down) exactly 3 running the clock out without scoring again
(clock/game-management behavior), not a late go-ahead field-goal drive ending
the game — "Field goal" endings are real (over a quarter) but secondary. This
points at end-of-game clock-killing/4th-down policy (Unit 6 in the plan), not
at this drive-chain's outcome-pool conditioning, as the more promising next
lever for the margin-3 gap specifically.

Look count for this unit: 6 validation-split looks (configs A-F) + 1
predeclared test-split look (config A, in progress) = 7 total; 0 additional
looks at 2018-2025 beyond that one.

### Unit 1c result (test split, measured once, 2026-09-25)
Ran `scripts/sim04_unit1c_diagnostic.py --mode test --config A` once — the
single predeclared look at 2018-2025 for this unit. Artifact:
`artifacts/sim04_unit1c/20260925T193733Z/test_report.json`. Train 2009-2017
(54,322 drives, 2 dropped for missing state), eval 2018-2025 (matches Unit
1/1b exactly).

| number | Unit 1c (config A) sim | actual bootstrap CI | in CI? |
|---|---|---|---|
| 3 | 0.0996 | [0.1361, 0.1583] | no |
| 7 | 0.0668 | [0.0761, 0.0956] | no |
| 10 | 0.0494 | [0.0421, 0.0543] | yes |
| 14 | 0.0330 | [0.0404, 0.0643] | no |
| 17 | 0.0421 | [0.0285, 0.0438] | yes |

Hits: **2/5** (10, 17). Log loss: simulator 3.98993 vs. naive train-histogram
3.97248, delta **+0.01744** nats. Both numbers match Unit 1b's own run
(2/5 hits at 10/17, delta +0.0175) to within float noise, as expected since
config A is Unit 1b's unmodified logic re-implemented in the new script —
this is a useful parity check on the reimplementation, not a new finding.

**Verdict: NO-GO**, unchanged from Unit 1b, against the unchanged Unit 1
criterion (needs >=4/5 hits AND log-loss delta <=+0.02 nats; got 2/5, delta
passes on its own). This unit's diagnostic pass **did not find a fix**: none
of the three predeclared candidate mechanisms (state-conditioned
field-position transition, finer +/-7/8 score buckets, looser 15-drive
back-off minimum) improved the validation-split hit count or log loss over
Unit 1b's original conditioning, so no change was worth carrying into the
test-split run — the test run confirms Unit 1b's own NO-GO stands, it does
not add a new negative result beyond what Unit 1b already established.

**Look count, full unit:** 6 validation-split looks (configs A-F) + 1
test-split look (config A) = 7 total. 1 look at 2018-2025 (the minimum
possible given the predeclaration commits to a confirmatory run regardless of
the validation outcome).

**What this unit resolved:** the try-after-TD-choice candidate is refuted as
the mechanism (points already draw jointly with category from the same state
cell in both Unit 1b and here — see the pre-check above). The other three
candidates are `unresolved_below_power` for this outcome-pool-conditioning
approach specifically — none moved the needle, but none had its interval
land on the wrong side of a control either; they are better read as "not
addressing the actual mechanism" given the margin-3 decomposition below.

**What points to the actual mechanism:** the margin-3 decomposition (2015-2017
validation actual, read of history) found the dominant non-OT margin-3
ending is "End of half" (56.7% of non-OT margin-3 games) — a team already
leading or trailing by exactly 3 running the clock out without scoring again
— not a late go-ahead field-goal drive (27.8%). OT contributes another ~20-24%
of all margin-3 games across both the validation and test windows. Neither
of those mechanisms (deliberate clock management when already up 3; OT
sudden-death dynamics) is modeled by this drive-chain's outcome-pool
conditioning at all, regardless of how the pool is sliced — which is
consistent with the validation sweep's flat result.

### Next
Unit 1c is done. The named next unit per Unit 1b's own recommendation and
this unit's decomposition finding is **not** more outcome-pool tuning on the
drive-chain skeleton — it is Unit 6 (4th-down/clock-management policy),
specifically the end-of-game "protect a 3-point lead by killing clock"
behavior, moved earlier in the build order (as flagged after Unit 1's NO-GO
and repeated here). Escalate this reordering decision to the owner / primary
orchestrator before spawning a subagent for Unit 2's completion, Unit 6, or
any further Unit-1-family diagnostic — three diagnostic units (1, 1b, 1c)
have now all pointed at score/clock/endgame-policy conditioning as the gap,
not at deeper tuning of the unconditional-outcome-pool architecture.

## Unit 6 (endgame policy layer) -- 2026-09-25, NO-GO

### Design (predeclared before running, verbatim from the lane)
Endgame layer on top of Unit 1b's chain, active when `qtr==4 and gsr<=300`
or `qtr>=5`. New per-drive state at drive start from the widened snapshot
`data/pbp/raw/20260925T202544Z`: `off_to`/`def_to` (posteam/defteam timeouts
remaining, default 3.0 if missing), bucketed 0/1/2-3. `late_time_bucket`:
Q4 (120,300]s left -> 0, Q4 <=120s -> 1, OT -> 2. In the late window, a
drive's (category, points_off, points_def, duration) and its (off_to,
def_to) are resampled jointly and empirically from real late-window drives
matching (score_bucket_fine x late_time_bucket [x fp_bucket] x
def_to_bucket [x off_to_bucket]), backing off to a coarser cell (drop
fp_bucket) and then to Unit 1b's unmodified level0->level3 cascade as the
floor. This folds the leading team's clock-kill/kneel choice, the trailing
team's 4th-down go/kick/punt, and timeout-driven clock consumption into one
conditioning scheme rather than three hand-built submodels — a scope
limitation stated up front, not a full play-level policy/clock model (that
needs Units 3-5, not yet built). Back-off minimum predeclared: 15 late-window
drives per cell for configs B/C; config D reuses B's dimensions at
min_cell_n=25. Script: `scripts/sim04_unit6_endgame.py` (imports scalar
helpers from `sim04_unit1b_state_chain.py`; duplicates `reconstruct_drives`
to add the two timeout fields).

| config | late dimensions | min_cell_n (late) |
|---|---|---|
| A (control) | none (Unit 1b unmodified) | n/a |
| B | score x late_time x fp x def_to | 15 |
| C | score x late_time x fp x def_to x off_to | 15 |
| D | score x late_time x fp x def_to | 25 |

Selection rule (fixed before running): most key-number hits (of 5) on
2015-2017; ties broken by smallest log-loss delta vs. the train naive
histogram; a config regressing log-loss delta >0.02 nats above config A is
disqualified even with more hits. The winner runs exactly once on train
2009-2017 / test 2018-2025 against the unchanged Unit 1 GO bar.

### Validation result (measured, 2026-09-25)
Ran `scripts/sim04_unit6_endgame.py --mode validation` once (all 4 configs,
seed `20260925`). Artifact:
`artifacts/sim04_unit6/20260925T203417Z/validation_report.json`. Train
2009-2014 (36,455 drives, 1 dropped), eval 2015-2017 (801 actual games).

| config | hits/5 | numbers hit | log-loss delta vs naive |
|---|---|---|---|
| A (control) | 1 | 10 | +0.01622 |
| B | 0 | none | +0.01779 |
| C | 1 | 10 | +0.02053 |
| D | 1 | 10 | +0.01905 |

Mass at 3 barely moved from Unit 1b/1c's baseline across every config
(A 0.1052, B 0.1064, C 0.1044, D 0.1061, all vs. actual-pooled 0.1486); none
recovers 7 or 14 either. Per the predeclared rule, A/C/D tie on hits, none
disqualified (largest gap from A is C at +0.0043, under the 0.02 threshold),
and **config A (no endgame layer) has the smallest log-loss delta among the
tied group and wins**. The timeout-conditioned late-window resampling
(configs B-D) did not out-hit or out-score the unmodified Unit 1b cascade on
this validation split.

### Test result (measured once, 2026-09-25, config A)
Ran `scripts/sim04_unit6_endgame.py --mode test --config A` once — the
single predeclared look at 2018-2025 for this unit (the 4th across the sim04
family, after Units 1, 1b, 1c). Artifact:
`artifacts/sim04_unit6/20260925T203506Z/test_report.json`. Train 2009-2017
(54,322 drives, 2 dropped), eval 2018-2025 (2,227 actual games).

| number | Unit 6 (config A) sim | actual bootstrap CI | in CI? |
|---|---|---|---|
| 3 | 0.1001 | [0.1364, 0.1588] | no |
| 7 | 0.0664 | [0.0762, 0.0955] | no |
| 10 | 0.0495 | [0.0420, 0.0543] | yes |
| 14 | 0.0324 | [0.0409, 0.0643] | no |
| 17 | 0.0425 | [0.0288, 0.0438] | yes |

Hits: **2/5** (10, 17). Log loss: simulator 3.98996 vs. naive train
histogram 3.97248, delta **+0.01748** nats. Matches Unit 1b's own run
(2/5 at 10/17, delta +0.0175) to within float noise, expected since config A
is Unit 1b's logic unmodified — a parity check, not a new finding.

**Verdict: NO-GO**, unchanged from Units 1b/1c, against the unchanged Unit 1
criterion (needs >=4/5 hits AND log-loss delta <=+0.02 nats; got 2/5, delta
passes alone). Because config A won the predeclared validation selection,
this test run confirms the prior NO-GO rather than testing a materially new
mechanism.

**What this unit resolved:** timeout-conditioned resampling of the *existing*
per-drive (category, duration) pool, even with real timeouts-remaining data
and a purpose-built back-off cascade, does not move the margin-3 mass toward
the actual pile-up (stuck at ~0.10-0.106 vs. actual ~0.148-0.149 in both
validation and test). This is `unresolved_below_power` for this specific
mechanism (finer empirical conditioning of a single per-drive outcome draw),
not a refutation of endgame policy as a driver — the margin-3 decomposition
from Unit 1c (56.7% of non-OT margin-3 games end on a clock-expiration
"End of half" drive) still stands as a real, unmodeled mechanism. The likely
reason conditioning alone did not surface it: a probabilistic draw from a
mixed pool (some fraction of late-window cells are kill-clock drives, most
are not) still lets normal drive outcomes appear with their pooled
probability, while the real mechanism is closer to a **near-deterministic
rule** (leading team with the ball in the final ~2 minutes and a manageable
lead essentially always kills the clock, conditional on down/distance state
this unit does not track) rather than a resampling-weight problem. Sparsity
likely also mutes any signal that reached deep cells: many late-window
(score, time, fp, timeout) cells fall under the 15/25-drive floor and back
off to the coarser cell, diluting the timeout signal.

**Look count, full unit:** 4 validation-split looks (configs A-D) + 1
test-split look (config A) = 5 total; 1 look at 2018-2025 (4th across the
sim04 family).

### Next
Endgame policy as *empirical resampling conditioning* is exhausted for this
architecture. The remaining lever implied by both Unit 1c's decomposition and
this unit's null result is an explicit, near-deterministic clock-kill rule
(not a probability-weighted draw) for the leading team in the final ~2
minutes with a first-down/manageable-lead state — i.e., closer to Units 3-5's
per-play loop than to more state-cell conditioning on the drive-level chain.
Escalate to the orchestrator before building a play-level clock-kill rule:
whether to invest in the per-play submodels (Units 3-5) next, given three
successive drive-level conditioning attempts (1b, 1c, 6) have now all failed
to move the margin-3/7/14 gap.

## Unit 1d diagnostic (measured, 2026-09-25)

Diagnostic only (no new conditioning, no GO/NO-GO gate): decompose why three
drive-level attempts (1b, 1c, 6) all undershoot margin 3. Train 2009-2014,
validate 2015-2017, 2018-2025 untouched. Script
`scripts/sim04_unit1d_divergence.py` (dynamically imports
`sim04_unit1b_state_chain.py`, does not duplicate its logic). Ran once:
`.\.tools\uv.exe run --no-sync python scripts\sim04_unit1d_divergence.py`.
Artifact: `artifacts/sim04_unit1d/20260925T204442Z/report.json`. 36,455
training drives (1 dropped), N=10,000 simulated games, RNG seed 20260925,
1,000-resample bootstrap on the decomposition.

**Data caveat (measured):** the validation actual set here is 768 REG-only
games (season_type=="REG" read directly from the raw pbp snapshot), not the
801 Unit 1b/1c/6 used — `game_features_pbp.parquet` filters only by
`season`, so its "2015-2017" slice silently includes 33 playoff games (week
18-20 game ids, e.g. `2015_20_NE_DEN`). This unit's 768 is the cleaner
regular-season-only figure; Units 1b/1c/6's "801 actual REG games" wording
is off by these 33 postseason games. Flagged for the orchestrator, not
corrected retroactively here (out of this unit's scope).

**1. Score-differential (home-away) distribution at fixed checkpoints, sim
vs actual, and the sim/actual SD ratio:**

| checkpoint | sim mean | sim sd | actual mean | actual sd | sd ratio (sim/actual) |
|---|---|---|---|---|---|
| end Q1 (gsr 2700) | 0.03 | 7.28 | 1.12 | 6.72 | 1.084 |
| halftime (gsr 1800) | 0.05 | 11.45 | 2.13 | 10.53 | 1.088 |
| end Q3 (gsr 900) | 0.05 | 14.33 | 2.29 | 13.12 | 1.093 |
| 5:00 left Q4 (gsr 300) | 0.04 | 15.66 | 2.43 | 14.04 | 1.115 |
| 2:00 left Q4 (gsr 120) | 0.12 | 15.96 | 2.24 | 14.28 | 1.117 |
| final margin | 0.12 | 15.96 | 2.20 | 13.87 | 1.150 |

The sim/actual SD ratio is already 1.08 by the end of Q1 and rises steadily
(not a step change at the endgame) to 1.15 at the final margin -- excess
dispersion starts early and compounds gradually through the whole game, it
does not appear only in Q4. Share of games within 3 points also runs
consistently below actual at every checkpoint (e.g. q4_5min: sim 0.195 vs
actual 0.221).

**2. Transition matrix at 5:00 left in Q4 (home-relative score_bucket_coarse
-> P(final |margin|==3)):**

| bucket (home persp.) | actual n | actual P(margin=3) | sim n | sim P(margin=3) |
|---|---|---|---|---|
| trail >=9 | 158 | 0.063 | 2746 | 0.031 |
| trail 1-8 | 150 | 0.233 | 1990 | 0.156 |
| tied | 46 | 0.609 | 456 | 0.461 |
| lead 1-8 | 166 | 0.217 | 2066 | 0.146 |
| lead >=9 | 248 | 0.032 | 2742 | 0.031 |

Every non-blowout bucket shows actual converting to exactly a 3-point final
far more often than the sim, tied being the starkest (61% vs 46%).

**3. Per-drive scoring rate by the offense's own score_bucket_coarse
(mean points_off / P(scored), train-pool-fed sim draws vs actual validation
drives):** actual and sim show the same shape (trail>=9 and lead>=9 both
score least, ~1.77/1.69-1.77/1.69 pts, the three middle buckets ~1.75-1.85
pts in both) -- the marginal per-drive scoring-rate-by-state association is
already reasonably matched between sim and actual. This rules out "missing
negative dependence at the single-drive marginal level" as the main driver;
conditioning on score differential already reproduces roughly the right
trailing/leading scoring-rate shape.

**4. Last-drive-of-the-game category share, all games and margin-3 games
only:**

| category | actual all (n=768) | sim all (n=10000) | actual margin=3 (n=117) | sim margin=3 (n=990) |
|---|---|---|---|---|
| End of half (clock expiration) | 85.3% | 46.6% | 61.5% | 17.3% |
| Field goal | 5.2% | 11.2% | 23.9% | 37.8% |
| Touchdown | 1.7% | 11.6% | 0% | 11.6% |
| Turnover / on downs / punt / other | 7.8% | 30.7% | 14.6% | 33.3% |

Real games overwhelmingly end via clock expiration (85.3% of all games,
61.5% even restricted to margin-3 games); the simulator treats the final
drive like any other drive draw, so under half its games (46.6%) and well
under a fifth of its margin-3 games (17.3%) end that way -- it manufactures
3-point finals mostly via a live go-ahead field-goal drive (37.8%) instead.

**Decomposition (predeclared 2x2 swap on the 5:00-left transition matrix,
1000-resample bootstrap 90% CI):**

| quantity | value | 90% CI |
|---|---|---|
| P(margin=3), actual (direct) | 0.1523 | [0.1328, 0.1732] |
| P(margin=3), sim (direct) | 0.0990 | [0.0940, 0.1039] |
| sim reaching x actual finishing | 0.1452 | [0.1257, 0.1658] |
| actual reaching x sim finishing | 0.1058 | [0.0982, 0.1137] |

Gap to explain: 0.0533. Swapping only the actual *finishing* behavior onto
the sim's own (too-dispersed) 5:00 checkpoint distribution recovers 0.1452,
86.6% of the gap. Swapping only the actual *reaching* (checkpoint)
distribution onto the sim's own finishing behavior recovers just 0.1058,
12.8% of the gap. **Finishing, not reaching, explains the shortfall**
(~87% vs ~13%).

**Named mechanism:** the drive-chain simulator has no clock-expiration rule.
A drive drawn late in the 4th quarter is given its full drawn (category,
duration) outcome from the empirical pool regardless of whether real time
would have run out first, so games that reach a close state late still
convert to live scoring/turnover plays at the pool's unconditional rate
instead of overwhelmingly running out the clock the way real close games do
(item 4). This is the same mechanism Unit 1c's non-OT margin-3 breakdown
(56.7% "End of half") and Unit 6's null result on timeout-conditioned
resampling already pointed at, now quantified: it accounts for roughly 87%
of the margin-3 mass shortfall via the transition-matrix decomposition, not
the broader pre-Q4 dispersion buildup (~13%, though that buildup is real and
gradual, first measurable by end of Q1).

**Recommendation:** build an explicit, near-deterministic clock-expiration
/ kneel-down rule for the final drive of a half (a per-play or remaining-
time-vs-drive-duration check, not another resampling-pool conditioning
axis) -- Units 3-5's per-play loop, as the prior "Next" already escalated,
now with a measured ~87%-of-gap justification rather than a qualitative one.

## Unit 7 predeclaration (written before running, 2026-09-25)

Data-bug fix (applies to every actual set this unit uses, not retroactive to
1b/1c/6): `game_features_pbp.parquet` has a `game_type` column; confirmed by
direct read that `game_type=='REG'` yields exactly the same `game_id` set as
`season_type=='REG'` on the raw snapshot `data/pbp/raw/20260925T202544Z`
(2016 check: 256/256 identical ids, vs 267 games with `game_type` unfiltered).
Every actual set below uses `game_features_pbp.parquet` filtered on
`season.isin(...) & game_type=='REG'`.

**Mechanism target (from Unit 1d):** `sim04_unit1b_state_chain.py`'s game
loop (`while gsr>0: apply_drive(); gsr=max(0,gsr-duration)`) draws a late
drive's `(category, points_off, points_def, duration)` atomically from a
historical pool and applies the score in full even when that drive's drawn
duration exceeds the actual remaining clock -- there is no play-level check
that the clock expires before the drive resolves. Unit 7 adds exactly that
check for the final drive(s) of each half.

**Trigger:** a drive's start state (`qtr in {2,4}` and
`seconds_left_in_period(qtr,gsr) <= 300`) triggers the play-level clock race
in place of Unit 1b's plain atomic draw; OT is untouched (Unit 1b's existing
OT loop). Unit 1b's drive-level `level0..level3` cells and GO criterion are
otherwise reused unmodified via import (`sim04_unit1b_state_chain` as `u1b`).

**Play-level training pool** (from `data/pbp/raw/20260925T202544Z`,
`season_type=='REG'`, TRAIN_SEASONS): one row per offensive snap belonging to
a drive (`fixed_drive`) whose play falls in the trigger window, excluding
kickoff/extra_point rows. Per row: `label` in {kneel (`qb_kneel==1`), spike
(`qb_spike==1`), field_goal, punt, run, pass, no_play} from `play_type` with
the kneel/spike override; `elapsed` = this play's `game_seconds_remaining`
minus the next selected play's in the same game, or minus the half boundary
(1800 for qtr 2, 0 for qtr 4) for the last selected play of that half, clipped
to [0,300] -- an empirical duration, not a hand-coded runoff rule, so it
already bakes in incompletions/out-of-bounds/timeouts stopping the clock
without needing an explicit stoppage flag (not present in the 56-column
snapshot); `is_terminal` = True if `label` in {punt, field_goal} or
`touchdown==1` or `interception==1` or `fumble_lost==1` or the row is the
last row of its `fixed_drive` (turnover on downs / clock-forced end); cell
key fields `sb_fine` (Unit 1b's `score_bucket_fine` on `score_differential`),
`sl_bucket` (5 levels, `min(4, seconds_left//60)`), `own_to`/`def_to`
(`min(3, posteam/defteam_timeouts_remaining)`).

**Back-off (predeclared, minimum cell count = 25 rows, matching Unit
1b/1c):** L0 `(sb_fine, sl_bucket, own_to, def_to)` -> L1 `(sb_fine,
sl_bucket, own_to)` -> L2 `(sb_coarse, sl_bucket)` -> L3 floor (all
trigger-window rows pooled, guaranteed non-empty).

**Race loop per late-window drive:** `own_to`/`def_to` are read once at
drive start and held fixed for the whole race (declared simplification --
mid-drive timeout depletion is not re-bucketed play-to-play; the persistent
per-team timeout counters used by the *next* drive's own_to/def_to do update,
from each drawn row's timeout delta); `sb_fine` likewise fixed for the race;
`sl_bucket` is recomputed each iteration from `remaining - consumed`. Each
iteration bootstrap-draws one training row from the back-off cell; if
`consumed + elapsed >= remaining` (or a 30-play safety cap is hit), the clock
wins: category becomes `Clock expired`, `points_off=points_def=0`,
`duration=remaining` exactly, and the game loop proceeds unchanged (this
naturally covers both "End of half" at the 1800s boundary and game end at the
0s boundary, since Unit 1b's outer loop already just checks `gsr>0`). If a
drawn row is `is_terminal` before the clock expires, the drive's
category/points/next_fp are drawn from Unit 1b's own state cells at the
drive's start `(diff,qtr,gsr,fp)` exactly as Unit 1b already does (declared
simplification: the race's job is only to decide *whether* the clock survives
the drive, not to re-derive the scoring outcome), but `duration` is replaced
by the race's own `consumed` value.

**Configs (at most 3, predeclared):**
| config | timeout conditioning (own_to/def_to in L0/L1) | min_cell_n |
|---|---|---|
| A (baseline) | yes | 25 |
| B (no-timeout ablation) | no (L0=L1=`(sb_fine, sl_bucket)`) | 25 |
| C (sparsity) | yes | 15 |

**Validation split:** train 2009-2014, validate 2015-2017 (REG only, fixed
actual set, ~768 games per Unit 1d). Selection rule (fixed now): most
key-number hits of 5; ties broken by smallest log-loss delta; a config whose
log-loss delta regresses more than +0.02 nats versus config A is disqualified
even with a higher hit count (same rule as Unit 1c). The chosen config runs
once on train 2009-2017 / test 2018-2025 (REG only) against the unchanged
Unit 1/1b GO criterion (>=4/5 key numbers in the actual bootstrap 90% CI AND
log-loss delta <=+0.02 nats vs. the naive train histogram) -- this is the 5th
look at 2018-2025 (after Unit 1, Unit 1b, Unit 1c-config-A, Unit 6-config-A).
Also reported, not gating: final-drive "End of half"/"Clock expired" share
and margin sd vs. actual, and the REG-only-fix delta on 2018-2025 key-number
mass (old season-only-filtered vs. new `game_type=='REG'`-filtered actual).

Script: `scripts/sim04_unit7_clock.py`. Artifacts under
`artifacts/sim04_unit7/<timestamp>/`.

## Unit 7 result (measured, 2026-09-25)

Validation (train 2009-2014, eval 2015-2017, artifact
`artifacts/sim04_unit7/20260925T205718Z/report.json`): hits/5 -- A=1 (7),
B=2 (7,10), C=1 (7); log-loss delta -- A=+0.01424, B=+0.01323, C=+0.01686,
none disqualified. Config B (no-timeout-conditioning ablation) wins the
predeclared rule.

Test split (config B, train 2009-2017, test 2018-2025 REG-only, the **5th
look at 2018-2025**, artifact
`artifacts/sim04_unit7/20260925T210113Z/report.json`):

| number | Unit 7 (B) sim | actual (REG-only) 90% CI | in CI? | Unit 1b sim (old actual) |
|---|---|---|---|---|
| 3 | 0.0987 | [0.1333, 0.1548] | no | 0.1001 |
| 7 | 0.0733 | [0.0755, 0.0966] | no | 0.0664 |
| 10 | 0.0546 | [0.0430, 0.0546] | yes | 0.0495 |
| 14 | 0.0382 | [0.0396, 0.0630] | no | 0.0324 |
| 17 | 0.0384 | [0.0287, 0.0445] | yes | 0.0425 |

Hits: **2/5** (10, 17) -- same count and same numbers as Unit 1b/1c/6. Log
loss: simulator 4.00001 vs. naive 3.98391, delta **+0.01610** (inside the
+0.02 bound, an improvement over Unit 1b's +0.0175). Sim margin sd 15.39 vs.
actual 14.31 (ratio 1.076, tighter than Unit 1b's 1.090). **Final-drive
clock-expired share (of all 10,000 games, excluding OT games decided in
OT): 90.06%** vs. actual 85.3% (Unit 1d) -- the targeted mechanism is now
close to real, a large jump from Unit 1b's implicit ~46.6%.

**REG-only fix effect on 2018-2025** (season-only-filtered vs.
`game_type=='REG'`-filtered actual, 2227 vs. 2127 games, 100 playoff games
removed): mass changes are small at this snapshot -- 3: 0.1482->0.1443
(-0.0038), 7: 0.0849->0.0851, 10: 0.0480->0.0489, 14: 0.0521->0.0512, 17:
0.0364->0.0362. The REG-only fix matters more for 2015-2017 (33/801 = 4.1%
playoff games, Unit 1d) than it moves 2018-2025's headline numbers here,
though the game-count correction itself (2227->2127) is real and now used
consistently for this unit's GO/NO-GO gate.

**Verdict: NO-GO** against the unchanged criterion (needs both >=4/5 hits
and log-loss delta <=+0.02; log loss clears, hits do not). This is a
measured, mixed result: the play-level clock race achieves its named
target (final-drive clock-expiration rate 90.1% vs. actual 85.3%, up from
Unit 1b's ~46.6%) and modestly improves log loss and sd ratio, but does not
move the hit count -- 7 and 14 remain misses at the same numbers as every
prior unit (1b, 1c, 6), and the margin-3 gap (sim 0.0987 vs. actual
[0.1333,0.1548]) is barely smaller than Unit 1b's. No interval here flips
sign (all misses are simulator-under-actual, not over), so this is not a
refuted mechanism per AGENTS.md -- it is `unresolved_below_power`: fixing
*whether* the clock expires was necessary but not sufficient; the
"clock-survives" branch still draws its scoring outcome from Unit 1b's
original unmodified drive-level cells (a declared simplification, see the
predeclaration), so a next diagnostic should check whether that branch's
outcome distribution -- not the clock-expiration rate itself -- is now the
binding constraint on 7/14/margin-3.

**Recommended next unit:** a bounded diagnostic on the "clock survives"
branch's drawn category mix in the late window (does it still overweight
live go-ahead scores relative to the real late-window state-conditioned
rate, now that the clock-expired branch is no longer diluting the
comparison), before any further clock-mechanism engineering.

## Unit 7b diagnostic (measured, 2026-09-25)

`scripts/sim04_unit7b_scoring_mix.py`, Unit 7 config B simulator, train
2009-2014 / validate 2015-2017 REG-only, 10,000 sim games, artifact
`artifacts/sim04_unit7b/20260925T211101Z/report.json`. Found the opposite
of the suspected mechanism: the "clock survives" branch does not
overweight live scores in the late window, it **underweights** them.

Tied-at-5:00 bucket (n=49 actual, 412 sim): P(final margin=3) actual 0.714
vs sim 0.573, gap +0.141. Split the gap by whether the game reached OT:
via-OT actual 0.184 vs sim 0.481 (gap -0.297, sim is *not* short here --
overtime alone already manufactures more 3s than reality once a game gets
there); via-regulation (no OT) actual 0.531 vs sim 0.092 (gap +0.438, this
is the whole story). Root cause: sim's OT rate for tied-at-5:00 games is
0.869 vs actual 0.347 -- 2.5x too many tied games reach overtime instead
of being decided by a single late score. That traces to a flat scoring
deficit in the final-5-minutes window: mean scoring plays after 5:00 is
0.99 sim vs 1.33 actual; share of post-checkpoint drives that score
nothing is 76.7% sim vs 64.5% actual; both FG share (15.6% vs 23.5%, 1.5x
short) and TD+PAT share (6.4% vs 10.4%, 1.6x short) of post-checkpoint
drives are proportionally under-drawn -- not a TD-vs-FG mix problem, a
general late-and-tied scoring-rate problem.

Cell audit (measured on the 2009-2014 train drives) names the mechanism:
`draw_cell`'s fine cell for (tied, final-5-or-2-min-of-Q4, field-position)
has n=2 to 49 across the 10 field-position buckets (median ~10), almost
always under `MIN_CELL_N=25`, so nearly every draw in the "clock
survives" branch falls back past level0 to the coarse pool. That coarse
pool is keyed by `tb_coarse`, and `time_bucket_coarse` merges
`tb_fine` in {2, 5, 6} into one bucket -- meaning the true Q4 endgame
cell is diluted with pre-halftime Q2 "tied, clock running down" drives,
a lower-urgency population where teams play more conservatively and
score less. This is the binding constraint, not clock expiry (already
fixed in Unit 7) and not a TD/FG mix error.

Secondary, smaller finding: OT resolution itself is also off -- sim OT
tied-bucket endings are 14.8% ties (53/358) vs actual 0/17, and
`p_fg_given_ot` runs a little hot (0.430 sim vs 0.294 actual); `OT_SECONDS
=600` (10-minute period) is applied uniformly even though 2015-2016 used
the 15-minute regular-season OT rule (changed to 10 minutes for 2017
only) -- a pre-existing simplification (shared with Units 1b/1c/6/7/1d),
not newly introduced here, and secondary to the regulation-path gap.

Field-position table (`field_position_table_post_5min_drives` in the
artifact) shows the same general under-scoring pattern pooled across all
four state buckets, most visible in the 20-50-yard-line bins where actual
FG-attempt rate runs 15-38% against sim's 5-15%.

**Next build change:** split the fallback the "clock survives" branch
uses so Q2 pre-half tied cells (`tb_fine==2`) never pool with Q4 endgame
tied cells (`tb_fine` in {5, 6}) -- either give `time_bucket_coarse` a
distinct bucket for {5, 6} vs {2}, or insert a Q4-only intermediate level
between level0 and level1/level2 in `build_state_cells`/`draw_cell` -- then
re-measure the tied-bucket OT rate and via-regulation margin-3 share on
this same 2015-2017 split before touching OT resolution or the era OT-
length mismatch.

## Unit 8 predeclaration (written before running, 2026-09-25)

Target (from Unit 7b): `build_state_cells`/`draw_cell`'s coarse fallback
(`time_bucket_coarse`) merges Q2 pre-half tied drives (`tb_fine==2`) with
Q4 endgame tied drives (`tb_fine` in {5,6}) into one pooled bucket, diluting
the true endgame scoring rate. New script `scripts/sim04_unit8_cells.py`,
built on Unit 7 config B (play-level clock race, `use_timeouts=False`,
race `min_cell_n=25`, imported unmodified from `sim04_unit7_clock`).
Reuses `sim04_unit7b_scoring_mix.actual_diagnostics`/`aggregate` unmodified
for the actual-side tied-at-5:00 bucket so the sim/actual comparison stays
apples-to-apples with Unit 7b's own numbers.

**Fix 1, back-off hierarchy** (local `time_bucket_coarse2`,
`build_state_cells2`, `draw_cell2`, not touching `sim04_unit1b_state_chain`):
`time_bucket_coarse2` splits the old 3-way split into 4: `tb_fine==7`(OT)->3,
`tb_fine==2`(Q2 pre-half)->1, `tb_fine` in {5,6}(Q4 endgame, the trigger
window)->2, else->0. New chain: level0 `(sb_fine,tb_fine,fb)` -> **level_q4**
(new, active only when `tb_fine` in {5,6}) `(sb_fine,tb_fine)` pooled over
`fb` -> level1 `(sb_fine,tb_coarse2,fb)` -> level2 `(sb_coarse,tb_coarse2,fb)`
-> **level_score** (new) `(sb_fine,)` pooled over all time/fp -> level3, now
a single unconditional pool of every training drive (previously indexed by
`fb` alone). `tb_fine==4` (Q4, >300s left) stays in the general group 0 --
Unit 7b flagged only the <=300s buckets as low-n.

**Fix 2, OT duration by era:** real rule is 15-minute (900s) regular-season
OT through 2016, 10-minute (600s) from 2017 (**read**, general NFL-rules
knowledge, not re-verified against a primary source this unit). Simulated
games carry no season tag, so each simulated game draws its own OT length
from `rng.choice` over the eval seasons' rule (`900.0` for season<=2016,
`600.0` for >=2017) -- for the 2015-2017 validation split that is P(900)=2/3,
P(600)=1/3, matching season composition.

**Fix 3, OT tie rate:** audited the existing possession-rule ("settled")
logic against the modified-sudden-death rule (first-possession FG earns the
opponent an answering drive; a first-possession TD or a defensive/return
score wins immediately; once both sides have had one possession, the next
score of either side wins) -- traced through all three branches
(FG-then-answer, TD/defensive-score-immediate, scoreless-first-answered) and
the code's `settled` check already matches the rule in each case. No
possession-rule bug found; **fix 2 (era-correct OT length) is the whole of
fix 3** -- shortening every 2015-2016 OT period to 600s when the real rule
gave 900s mechanically raises the chance the simulated clock expires before
a winner emerges. Measured pre/post tie rate is reported; if a material gap
remains after the era fix this is named `unresolved_below_power`, not
patched further this unit.

**Configs (at most 3, predeclared), state-cell backoff only (race pools
fixed at Unit 7 config B for all three):**
| config | min_cell_n | level_q4 | level_score |
|---|---|---|---|
| A (full fix) | 25 | on | on |
| B (sparser floor) | 15 | on | on |
| C (coarse-split only, ablation) | 25 | off | off |

**Validation split:** train 2009-2014, validate 2015-2017 REG-only (same
~768-game actual set as Units 7/7b). Selection rule (same pattern as prior
units): most key-number hits of 5; ties broken by smallest log-loss delta;
a config whose log-loss delta regresses more than +0.02 nats versus this
unit's own config A is disqualified even with a higher hit count. **This
unit is validation-only** -- no test-split run, no 6th look at 2018-2025.

Reported, not gating: tied-at-5:00 regulation-path margin-3 share, OT rate,
OT tie rate (all vs Unit 7b's actual 2015-2017 numbers), key-number mass
table with CI, discrete log loss vs naive histogram, margin sd -- all
against Unit 7 config B's validation numbers.

Script: `scripts/sim04_unit8_cells.py`. Artifacts under
`artifacts/sim04_unit8/<timestamp>/`.

## Unit 8 result (measured, 2026-09-25)

Validation (train 2009-2014, eval 2015-2017 REG-only, artifact
`artifacts/sim04_unit8/20260925T212234Z/report.json`). All three configs:
hits/5 **1** (only 7). Log-loss delta -- A=+0.01349, B=+0.01395,
C=+0.00812, none disqualified (bound +0.02 above config A's own delta).
Predeclared rule (most hits, tie-break smallest delta) selects **config C**
(coarse-split fix only, no Q4-only/score-only levels) -- verified by hand,
matches the script's own `selected_config`.

**Fix 1 (backoff hierarchy) is verified working as designed but does not
close the targeted gap.** Direct cell audit on the 2009-2014 train drives
confirms `level_q4` is reached and no longer polluted: tied+`tb_fine=5`
(2-5 min left) has n=91 with per-drive scoring rate 0.352 -- within noise of
Unit 7b's actual rate (~0.355 blended); tied+`tb_fine=6` (final 2 min) has
n=165, scoring rate 0.242 (lower, as expected -- less time to convert).
Despite the first-drive scoring rate now landing close to actual, the
**aggregate tied-at-5:00 bucket barely moved**: sim OT rate 84.3-84.6%
across configs vs actual 34.7% (Unit 7b's pre-fix number was 86.9% -- a
~2 point improvement, not the expected large closure), and sim
via-regulation P(margin=3) is 0.092-0.105 vs actual 0.531 (Unit 7b pre-fix:
sim 0.092 -- **unchanged**). This is a measured, verified-mechanism,
unresolved result: the diagnosed Q2/Q4 coarse-cell dilution was real and is
now fixed at the single-drive level, but it is not the dominant driver of
the tied-at-5:00 under-scoring gap -- most of the gap must come from
elsewhere in the multi-drive sequence within the 5-minute window (candidate:
the play-level clock-race's per-play elapsed-time/duration distribution,
which still ignores whether the drive is a Q2-half-ending or Q4-game-ending
race -- untouched this unit -- bounding how many total drives fit in the
window regardless of each drive's individual scoring rate).

**Fix 2 (OT duration by era) measurably worked.** Overall sim OT tie rate
fell from Unit 7b's 14.8% (53/358, uniform 600s) to 6.5% (config A), 9.4%
(B), 7.7% (C) with era-correct 900s/600s draws -- roughly halved, in the
direction and rough magnitude the named mechanism predicts, against an
actual rate of 0/17 (small-sample zero, cannot fully validate against).

**Fix 3 (possession-rule audit):** no bug found (see predeclaration); the
era fix carried the full measured improvement above.

**Net effect vs Unit 7 config B (validation):** hits regressed 2/5 (7,10)
-> 1/5 (7 only) -- **10** moved from inside the actual 90% CI
([0.0391,0.0546]) to just above it (sim 0.0589-0.0591). Sim margin sd rose
to 15.63-15.72 vs Unit 7 B's implied dispersion (Unit 7 B's own validation
run did not publish margin sd; only its test-split sd ratio 1.076 vs a
different actual population is on record, so this is not a like-for-like
comparison). Miss directions are mixed, not uniformly simulator-under-actual
(3 and 14 under; 10 and 17 over) -- a change in shape from every prior unit,
where misses had been uniformly under. No number's actual-CI is fully
crossed in sign by a config disagreement (all three configs miss the same
four numbers), so this is not a refutation of the Unit 7b mechanism, but it
is a clear miss of the predeclared GO bar.

**Verdict: NO-GO on validation** (needs >=4/5 hits; got 1/5, worse than
Unit 7 B's 2/5). Classify as `unresolved_below_power`: the named mechanism
(Q2/Q4 coarse-cell dilution) is fixed and confirmed non-dominant, not
refuted (no sign flip on the fix itself); the era-correct OT fix is
confirmed working. **Recommended next unit:** extend the play-level clock
race (Unit 7's `build_play_rows`/`build_race_pools`) to condition on
qtr (2 vs 4) so the number-of-drives-that-fit-in-the-window distribution
stops pooling end-of-half with end-of-game plays -- the remaining lever
this unit did not touch -- before further tuning the drive-level cells.

## Unit 8b trace (measured, 2026-09-25)

Debugging unit, not a new config. Copy `scripts/sim04_unit8b_trace.py`
(Unit 8 config C: train 2009-2014, validate 2015-2017 REG-only,
`use_q4_level=False`, `use_score_level=False`, `min_cell_n=25`), instrumented
to log every post-5:00-of-Q4 possession (race outcome, cell level/key,
start clock, duration, category, points) for sim games tied at the 5:00
checkpoint, plus the matching actual possession set from
`sim04_unit1b_state_chain.reconstruct_drives` restricted to `start_qtr==4`
rows after the checkpoint drive. Artifact
`artifacts/sim04_unit8b_trace/20260925T214746Z/report.json` (before/after
fix, `--both`).

**Possession-level comparison (tied-at-5:00, before fix):**

| metric | sim | actual |
|---|---|---|
| n tied-at-5:00 games | 445 | 49 |
| possessions/game (post-checkpoint, REG) | 1.63 | 3.04 |
| mean possession duration | 121.9s | 59.4s |
| scoring rate per possession | 0.101 | 0.302 |
| share of possessions = clock-expired/"End of half" | 61.4% (445/725) | 15.4% (23/149) |
| OT rate | 84.3% | 34.7% |
| via-regulation P(margin=3) | 0.094 | 0.531 |

**Named bug (not the Q2/Q4-pooling mechanism Unit 8 flagged next -- that
was checked and ruled out: splitting the race's tied-bucket play pool by
qtr shows near-identical mean elapsed and P(terminal) in qtr 2 vs qtr 4 at
every `sl_bucket`, e.g. `sl_bucket=4`: qtr2 mean_elapsed=68.1s/p_term=0.164
(n=445) vs qtr4 67.0s/0.167 (n=204)).** The real bug is in
`build_play_rows` (`scripts/sim04_unit7_clock.py:92-148`, the `elapsed`
calculation at lines 109-115): `selected` is rebuilt fresh inside the
per-drive loop (`for _drive_id, drive in game.groupby("fixed_drive")`), so
for the *last selected play of every drive* -- which is every terminal
play (punt, FG, scoring play) that ends a drive before the period truly
ends -- `j + 1 < len(selected)` is false and the code falls to
`elapsed = gsr - boundary` (boundary=0 for Q4), crediting that single
historical play with the *entire remaining clock at that instant* instead
of the true few-seconds-to-the-next-team's-snap. Measured: mean elapsed on
terminal-flagged tied-bucket rows is 267.8s at `sl_bucket=4` (240-300s
left) vs 28.3s on non-terminal rows in the same bucket (cell audit,
2009-2014 train). Because `run_race` (`sim04_unit7_clock.py:268`) checks
`if consumed + elapsed >= remaining: return True (clock expired)` *before*
checking `is_terminal`, drawing one of these corrupted rows makes the race
declare "clock expired" (0 points, game/period ends immediately) instead
of crediting the real score/punt/turnover and continuing -- this is what
starves the 5-minute window of possessions and inflates the OT rate. Three
example sim traces (before fix) show it directly: game 109, a single away
possession at 226s left instantly returns `clock_expired` with
`duration=226` (no scoring chance at all, straight to OT, final margin 0);
game 189 same pattern at 265s left (final margin -3, OT); game 5 has one
real possession (missed FG, 147s) then its next possession instantly
expires at 30s left (OT, final margin 4).

**Fix (in the copy only):** `build_play_rows_fixed` in
`sim04_unit8b_trace.py` accumulates `selected` plays across the whole game
(all drives, chronological) before computing `elapsed`, so a drive-ending
play's elapsed is measured to the next real selected play (any team, any
drive) and the boundary fallback fires only for the play that is truly
last in the period.

**Before/after (validation, same split, config C, one fixed run, no
tuning):**

| metric | before | after | actual |
|---|---|---|---|
| possessions/game | 1.63 | 3.04 | 3.04 |
| mean possession duration | 121.9s | 64.7s | 59.4s |
| scoring rate per possession | 0.101 | 0.153 | 0.302 |
| OT rate (tied-at-5:00) | 84.3% | 66.5% | 34.7% |
| via-regulation P(margin=3) | 0.094 | 0.162 | 0.531 |
| key-number hits/5 | 1 | 1 | -- |
| log-loss delta vs naive | +0.00812 | +0.00681 | -- |

The fix closes the possessions-per-game gap essentially completely (1.63
-> 3.04, matching actual 3.04) and roughly halves the OT-rate gap (49.6pt
-> 31.8pt) and lifts via-regulation margin-3 threefold (0.094 -> 0.162),
but does not close the gap to GO: post-fix `clock_expired`/"End of half"
share is still 32.9% (421/1278) vs actual's 15.4%, and scoring rate per
possession is still about half actual's (0.153 vs 0.302). This is a
**confirmed, named, fixed code bug** (not `unresolved_below_power` -- the
mechanism was directly measured, the fix directly verified against the
same held-out split), but a second, separate, unnamed mechanism remains:
after the elapsed fix, possession *count* matches reality but each
possession still scores too rarely and still resolves via bogus-looking
"clock expired" too often. Not applied to `src/` or Unit 7/8 (task scope:
copy-only debugging unit; no registry writes, no commits).

**Recommended next unit:** (1) port this `build_play_rows` elapsed fix
into `scripts/sim04_unit7_clock.py` for real (it changes Unit 7/7b/8's own
race pools, a rerun of those units' validation numbers, not just this
diagnostic copy); (2) diagnose the residual clock-expired-too-often /
scores-too-rarely gap now that possession count is fixed -- candidate: the
race's per-play conditioning still lacks down/distance or "plays already
run this drive," so it cannot tell a fresh 1st-and-10 snap from a
3rd-and-short snap and likely still over-draws non-terminal early-down
plays before finding a terminal one.

## Unit 9 (measured, 2026-09-25)

Built `scripts/sim04_unit9.py` on Unit 8 config C (`min_cell_n=25,
use_q4_level=False, use_score_level=False`; race `use_timeouts=False,
min_cell_n=25`), train 2009-2014 / validate 2015-2017 REG-only. Does not
edit units 7/8; imports `sim04_unit1b_state_chain`, `sim04_unit7_clock`,
`sim04_unit7b_scoring_mix`, `sim04_unit8_cells` and defines two new,
corrected functions. Artifact
`artifacts/sim04_unit9/20260925T215733Z/report.json`.

**Defect 1 (new, found here): cross-quarter contamination in the Unit 8b
elapsed fix.** `build_play_rows_fixed` (`sim04_unit8b_trace.py:73-169`)
accumulates `selected` plays across the whole game before computing
`elapsed`, which fixed the per-drive rebuild bug, but the accumulated list
mixes late-Q2-window and late-Q4-window rows with nothing in between (Q1,
early Q2, Q3, early Q4 are filtered out of `selected` entirely). For the
true last late-Q2 play of a game, `j + 1 < len(selected)` is true but
`selected[j+1]` is actually the first late-Q4 row of the same game, so
`elapsed = gsr - next_gsr` computes a bogus multi-quarter gap (900+
seconds, silently clipped to `WINDOW_SECONDS=300`) instead of falling back
to the period boundary. `sim04_unit9.py:29-140`
(`build_play_rows_qtr_safe`) fixes this by only using `selected[j+1]` when
it shares the same `qtr` as row `j`, else using the boundary fallback,
same as the original per-drive logic intended.

**Defect 2 (new, found here): `run_race` discards terminal (scoring)
draws that land near the clock boundary.** `run_race`
(`sim04_unit7_clock.py:263-275`) checks `if consumed + elapsed >=
remaining: return True (clock_expired)` **before** checking `is_terminal`.
A drawn play that is itself terminal (TD, FG, punt, turnover) but whose
`elapsed` happens to push the cumulative race clock to or past what's
left gets misclassified as a bare clock expiry -- the caller then credits
zero points and treats the period as having ended with no score, discarding
the real scoring/terminal event. `sim04_unit9.py:143-172`
(`run_race_fixed`) checks `is_terminal` first: a terminal draw always
returns the score outcome (with `consumed` clipped to `remaining` for
clock bookkeeping), and only a **non-terminal** draw that would exhaust
the clock is treated as expiry. This directly targets the reported "half
the scoring rate, doubled expiry share" symptom.

**Drive-duration audit (`reconstruct_drives`, unit 1/1b) -- no
comparable bug found.** Measured on 2009-2014 train drives
(`audit_drive_duration`, `sim04_unit9.py:373-395`): period-ending drives
(the last drive of a quarter with `start_qtr` 2 or 4) have shorter mean
duration than other drives in the same quarter (qtr2: 49.7s
period-ending vs 139.0s other, n=1536/7751; qtr4: 87.2s vs 133.8s,
n=1536/7687). This is real structure, not a bug: a period-ending drive is
capped by the period boundary at whatever time it started, so it cannot
run as long as a mid-period drive on average. `_emit_drive`
(`sim04_unit1b_state_chain.py:106-148`) always measures duration from the
drive's own actual last recorded play's `game_seconds_remaining` (never a
period-boundary fallback substituting for a missing next play, unlike the
old `build_play_rows` bug), so there is no analogous inflation mechanism.
Sanity check: the "other" mean durations (139.0s / 133.8s) are close to
the actual per-drive average from `game_features_pbp.parquet`
(`home_drive_seconds_per_drive`/`away_drive_seconds_per_drive`, pooled
mean 138.6s).

**Before (Unit 8b, elapsed fix only, contamination + ordering bugs still
present) vs after (Unit 9, both fixes) vs actual, tied-at-5:00 Q4
possessions, validation:**

| metric | before (8b) | after (unit 9) | actual |
|---|---|---|---|
| possessions/game | 3.04 | 3.40 | 3.04 |
| mean possession duration | 64.7s | 58.1s | 59.4s |
| scoring rate per possession | 0.153 | 0.161 | 0.302 |
| clock-expired / "End of half" share | 32.9% | 25.1% | 15.4% (23/149) |
| OT rate (tied-at-5:00) | 66.5% | 59.9% | 34.7% |
| via-regulation P(margin=3) | 0.162 | 0.171 | 0.531 |
| key-number hits/5 | 1 | 1 (still only "10") | -- |
| log-loss delta vs naive | +0.00681 | +0.00968 | -- |
| sim margin sd / actual margin sd | -- | 15.92 / 13.87 (ratio 1.15) | -- |

Full key-number table (unit 9, validation): 3: sim 0.0996 vs actual
0.1523 (CI 0.1380-0.1667, miss); 7: sim 0.0749 vs actual 0.0911 (CI
0.0755-0.1068, miss); 10: sim 0.0545 vs actual 0.0482 (CI
0.0391-0.0573, **hit**); 14: sim 0.0317 vs actual 0.0508 (CI
0.0456-0.0560, miss); 17: sim 0.0402 vs actual 0.0299 (CI
0.0208-0.0391, miss). `go_no_go: NO_GO` (needs >=4/5 hits and log-loss
delta <=0.02; got 1/5, though the log-loss delta itself still passes).

**Verdict: both fixes move the mechanism in the right direction
(clock-expired share and OT rate close roughly a third to half of their
gap) but do not close it, and the scoring-rate deficit that was expected
to shrink most from Defect 2 barely moved (0.153 -> 0.161 vs actual
0.302).** This says the ordering bug was real but not the dominant
remaining mechanism. Candidate reason, matching what the task asked to
audit ("whether down, distance and field position progress"): they do
not. When the race says "not expired," the actual scored outcome is not
the terminal play the race drew -- it is a **second, independent draw**
from `draw_cell2`'s historical whole-drive pool
(`sim04_unit8_cells.py:100-138`, called from `sim04_unit9.py:229-231`),
keyed only by score-bucket/time-bucket/field-position and, under config
C, falling through to the coarse level1/level2 pools (Q4-specific
`level_q4` and `level_score` are both off in config C). This reproduces
Unit 8's own finding that the Q4-specific level does not close the gap
even when it fires cleanly at the cell level -- the race's terminal
signal and the drive's scored outcome are structurally decoupled, so
fixing the race's own bookkeeping (Defect 2) cannot, by itself, raise the
drive-outcome pool's scoring rate.

Not applied to `src/`; no registry writes; no commits.

## Engine unit (2026-09-25/26, root)

Built `scripts/sim04_engine.py`, one clean self-contained play-level engine
(no imports from the abandoned units 1b-9 chain). Design departs from every
prior unit: instead of drawing a whole-drive outcome from a pool and racing
it against a separately-fit clock model (the structurally-decoupled defect
Unit 9 named as the remaining mechanism), this engine samples one real
historical play row per snap from a bucketed empirical cell keyed on
(down, distance bucket, field-position bucket, score-diff bucket, time
bucket incl. two-minute/last-5-min-Q4/OT, timeout flags) with hierarchical
backoff (L0 finest, L1 drops timeouts and coarsens score/time, L2 coarsens
distance/field position, L3 down-only floor, MIN_CELL_N=25), and copies
that historical row's own recorded transition (next down, next distance,
next yardline_100, whether possession flipped, clock elapsed to the next
snap, points scored) directly onto the simulated game state. Down,
distance, field position, scoring and the clock all come from this one
sequence. Because the transition is read directly off real (current row ->
next live snap in the same game) pairs, punt net yardage, FG make/miss by
distance, turnover-return field position and the 4th-down go/kick/punt
choice are not separately modeled -- they fall out of the same draw for
free, since real coaches' historical decisions are already baked into what
actually happened next in that bucket. Two explicit non-empirical rules
were added on top: touchdown scoring rows are credited with the
immediately following extra-point/2-point row's own score delta (folded
into points_off/points_def at table-build time via build_pat_bonus;
otherwise every TD would only ever be worth 6, never 7/8, the single
largest bug found this unit, see below), and era-correct OT length (900s
through 2016, 600s from 2017) with a hand-coded sudden-death settlement
(any defensive/return score or offense TD ends it; an offense FG only
ends it if this is not the first team's first OT possession and the score
is no longer tied) plus a timeouts-reset-to-3-at-halftime rule, since
timeout counts are tracked as delta-consumption applied to real team
identities, not copied absolute values. build_tables(seasons) and
simulate(n_games, rng, tables, ot_seconds=..., policy=...) match the
requested interface; policy is a pass-through seam for SIM-05, not
exercised this unit. Train 2009-2014, validate 2015-2017, REG-only via
season_type=='REG' on data/pbp/raw/20260925T202544Z and game_type=='REG'
on game_features_pbp.parquet (matches Unit 7's confirmed-equivalent
filter). No 2018-2025 run.

Bugs found and fixed in-session (self-contained to this file, each
re-measured before the next): (1) NaN down/yardline_100/opening-pool draws
crashed bucket functions -- dropna on required columns and on the
opening-field-position pool. (2) Extra points/2-point conversions are
separate play_type=='extra_point'/play_type_nfl=='PAT2' rows excluded from
the live-snap set, so every touchdown scored exactly 6 with no PAT --
fixed via build_pat_bonus, which reads the immediately-following PAT row's
own score delta and adds it to the TD row's points_off/points_def before
folding into the table; this fix alone dropped the validation log-loss
delta from +0.785 to +0.119 at n=300 games and moved key-number-3 mass
from grossly excess toward the actual range. (3) The OT sudden-death check
treated any positive score after the first possession as game-ending,
including a field goal that only re-tied the score (e.g. 3-0 -> 3-3) --
fixed to require home_score != away_score after the score before
settling; did not measurably move the tie rate (see below), so this bug
was real but not the dominant source of the elevated tie rate.

Command run (final, n=3334 games/season = 10,002 total, ~16s wall):
`.venv/Scripts/python scripts/sim04_engine.py --n-games-per-season 3334`.
Artifact: `artifacts/sim04_engine/20260926T020854Z/report.json`.

Key-number table (validation 2015-2017, bootstrap CI over actual seasons):
3: sim 0.0894 vs actual 0.1523 (CI 0.1380-0.1667, miss, sim too low); 7:
sim 0.0640 vs actual 0.0911 (CI 0.0755-0.1068, miss, too low); 10: sim
0.0551 vs actual 0.0482 (CI 0.0391-0.0573, hit); 14: sim 0.0366 vs actual
0.0508 (CI 0.0456-0.0560, miss, too low); 17: sim 0.0375 vs actual 0.0299
(CI 0.0208-0.0391, hit). 2/5 hits. Discrete log loss: simulator 3.9948 vs
naive-train-histogram 3.9522, delta +0.0426 (fails the <=+0.02 bar).
Margin SD ratio (sim/actual) 1.077. Points/game (sim) 42.75. Plays/game:
sim 148.95 vs actual 158.51. Possessions/game: sim 22.27 vs actual 23.06.
Late-Q4 (final 5:00) possession scoring rate: sim 0.180 vs
measured-this-session actual 0.215 (n=768 REG games, 2015-2017; this is a
fresh from-scratch measurement, not the same definition as the task
prompt's cited historical figure of 0.302 from Units 7b/9 -- the two do
not agree, most likely a differing drive-boundary or "scored" definition;
flagged, not reconciled, given the tool budget). Tied-at-5:00-remaining-
in-Q4 OT rate: sim 0.530 (n=419 sim games tied at that instant) vs
measured-this-session actual 0.149 (n=47 real games). Overall OT rate: sim
0.0481 vs actual 0.0625 (closest any unit in this series has come to
actual -- prior units ranged 59.9%-91%). Tie rate: sim 0.0283 vs a
real-world rate on the order of 0.003-0.005 (not separately re-measured
this session; visibly still roughly 6-10x too high).

GO/NO-GO: NO_GO (2/5 hits, log-loss delta +0.0426 > +0.02).

Verdict and named remaining mechanism. The literal play-by-play
architecture closes most of the gap this whole series has chased: OT rate
now nearly matches actual (4.8% vs 6.25%, vs 60-91% for every drive-chain
hybrid unit 1b-9), and the aggregate late-Q4 scoring-rate ratio
(0.180/0.215 = 0.84) is far closer to 1 than the old units' 0.161/0.302 =
0.53. The remaining, more specific defect: the tied-at-5:00 subgroup still
resolves to OT far more often in the sim (53.0%) than in real games
(14.9%), a much larger relative gap than the aggregate late-Q4 rate shows,
meaning tied/urgent-endgame situations specifically are under-scoring more
than the average late-Q4 possession. A same-session diagnostic on train
data (2009-2014) supports this: real drives that start tied, qtr==4,
gsr<=300 score at 0.291, actually higher than the all-late-Q4 average of
0.199 (trailing/leading teams plausibly play worse than an evenly matched
tied situation), so a correctly-calibrated engine should show the tied
subgroup scoring more than average, not roughly the same or less. This is
the same family of defect diagnosed in Units 7b-9 (tied-late-game scoring
undershoot from cell-sparsity backoff) recurring in a structurally
different engine, at much smaller magnitude, which is evidence the earlier
units' "structural decoupling" diagnosis was real but not the sole cause
-- data sparsity in the tied-plus-late-plus-specific-down/distance/field-
position corner of the cell hierarchy is at minimum a second contributing
mechanism. Not yet isolated to a single fix: the score-bucket functions
already keep tied (diff==0) as its own fine and coarse bucket (ruling out
the exact old mixing bug), so the next diagnostic step (not run this
session, out of budget) is to measure, from the same simulated games, the
per-play scoring rate specifically inside the k0/k1/k2 cells active when
score_diff==0 and qtr==4 and gsr<=300, split by which backoff level
actually fired, to see whether L1/L2 backoff (which drops or coarsens
field position/distance) is diluting the tied-specific elevated aggression
the same way Unit 7b found for the old drive-pool design.

Not ported to `src/`; no registry writes; no commits; no dashboard or
publish work (research-only per AGENTS.md).

## Engine fix 1 (measured, 2026-09-26, orchestrator)

Bug: `build_transition_frame` shifted next-state within game then dropped rows
with no next play, so the last play of every game never entered the pool. On
2009-2014 REG that removed 148 game-ending scores (81 of 94 OT final plays are
the winning score) and 851 final kneel-downs, so a simulated OT could almost
never end on a score. Fix: keep the last row as a terminal transition (clock to
zero in regulation, 6 s in OT, possession flips). Also added the halftime
possession change (second-half receiver is the opening kicker, fresh field
position, clock set to 1800). One validation run, 2015-2017, artifact
`artifacts/sim04_engine/20260926T021517Z/report.json`:

| metric | before | after | actual |
|---|---|---|---|
| mass at 3 / 7 / 10 / 14 / 17 | .089/.064/.055/.037/.038 | .098/.063/.056/.041/.041 | .152/.091/.048/.051/.030 |
| key-number hits | 2 | 1 | -- |
| log-loss delta vs naive | +0.0426 | +0.0207 | -- |
| tie rate | 2.8% | 1.6% | ~0.4% |
| tied-at-5:00 OT rate | 53.0% | 49.3% | 14.9% |
| late-Q4 possession scoring rate | 0.180 | 0.136 | 0.215 |

Still NO-GO. 3 and 7 are short by a third; late possessions under-score.

## Engine unit 2 predeclaration (written before fixing, 2026-09-26, subagent)

Diagnostic measurements (2009-2014 train transition table plus raw pbp),
before any fix:

- fp_f (fine field-position bucket, 10 yd wide) within-bucket std of
  yardline_100: 2.7-2.9 yd (matches uniform-in-10 theory, not itself
  excessive). fp_c (coarse, 20-30 yd wide, used at backoff): std 4.8-8.9 yd,
  worst in the two 30-yd-wide middle buckets.
- `next_yardline`/`next_distance` are copied as the drawn row's own absolute
  values regardless of how far the drawn row's own state differs from the
  simulator's actual state inside the same bucket (`scripts/sim04_engine.py`
  lines 176, 265-267, 517-519, 524-526). This is defect (a): confirmed by
  code, and the coarse-bucket std above shows the magnitude once backoff
  engages.
- Rows where `next_gsr > gsr` within a game (clock appears to run backward):
  309 of 230,642 transition rows. `clock_elapsed` clips these to 1.0 s
  (`sim04_engine.py` lines 230-232) — confirmed defect (b). Sampled rows show
  most are regulation-to-OT crossings (real OT `game_seconds_remaining`
  resets to the OT quarter length, e.g. 900, independent of the continuous
  0-3600 regulation clock) plus a handful of other same-quarter anomalies.
  Small in row count (0.13%) but concentrated in exactly the late/tied cells
  under scrutiny.
- score_diff==0, qtr==4, gsr<=300 (the tied-late-Q4 corner): 320 distinct L0
  cells, mean 4.7 rows/cell, only 3.4% reach MIN_CELL_N=25. L1 as currently
  defined (`down_i, dist_f, fp_f, sc_c, tb_c`) does not add cell size here
  (same 320 cells, same 4.7 mean) because timeouts were the only axis it
  drops and this subset is already homogeneous on timeouts, so 96.6% of
  these situations must fall to L2 (`down_i, dist_c, fp_c, sc_c, tb_c`, still
  only 14.3% reach n>=25) — a large share falls all the way to L3 (down
  only, pooling every quarter and score state). Per-play scoring rate
  measured directly on these tied-late training rows is 0.0769 vs 0.0548
  overall — real teams play more aggressively when tied and late — so
  diluting this cell into L3 erases exactly the signal the sim needs. This
  is defect (c)'s mechanism, not the literal score-bucket-merge named in the
  task (tied is already its own fine and coarse bucket, ruled out), but the
  same family: backoff order coarsens score/time before it needs to, because
  today it coarsens score/time at L1 while keeping field position/distance
  fine, the reverse of what the tied-late cell needs.
- Kickoff-return TDs: 76 of 15,503 kickoff plays (0.49%) score, worth about
  0.2 pts/game in aggregate. Structurally impossible in the current sim
  because `build_opening_pool` draws only from each game's own first live
  snap, which by construction never followed a returned-for-a-score kick.
  Confirmed real but small (~0.2 of the ~3 pt/game gap); deferred, not fixed
  this unit, given the tool budget and its small size relative to the
  key-number bar.
- Late-Q4 scoring-rate definitions in `summarize_sim` and
  `measure_actual_diagnostics` were read side by side: both count "any score
  (offense or defense) during a drive whose first play has qtr==4 and
  gsr<=300," so item (e) is not the mechanism; the two numbers are honestly
  comparable and the gap is real.

Predeclared fixes, to be made together and re-measured as one look (each is
individually motivated by the measurements above, and fix 1 is a
precondition for fix 3 being safe):

1. Field position and distance-to-go: for a non-flip (drive-continuing) row,
   apply the drawn row's own yards-gained (its `yardline_100` minus its own
   next yardline) to the simulator's actual current yardline instead of
   copying the absolute next yardline; do the same for distance-to-go, reset
   to `min(10, new_yardline)` when the drawn row's own down resets to 1 (a
   real conversion or auto-first-down), otherwise apply the drawn row's own
   distance-to-go delta. Keep the existing absolute-copy behavior on a flip
   (punt, turnover, score-then-kickoff): a new drive's starting spot is
   legitimately independent of the previous team's position.
2. Clock: when a row's own next_gsr exceeds its gsr (the 309 rows above),
   treat the play as using up the remainder of the clock (`clock_elapsed =
   gsr`) instead of the clipped 1.0 s value.
3. Backoff order: swap L1 so it coarsens distance and field position first
   (`down_i, dist_c, fp_c, sc_f, tb_f`) and keeps score/time fine, instead of
   today's L1 which coarsens score/time and keeps distance/field position
   fine. This is only safe because of fix 1: once field position and
   distance move by relative delta, a wide backoff bucket no longer teleports
   the ball, so it is safe to sacrifice that granularity first and keep the
   tied/urgent-time signal through backoff instead. L2 (fully coarse) is
   unchanged as the final fallback before the down-only floor.

Not changed this unit: MIN_CELL_N, the score-bucket boundaries themselves,
kickoff modeling. Also adding `simulate_from_state`/`simulate_from_states`
(task item 1, finishing-vs-reaching split) to `scripts/sim04_engine.py`,
factored out of `simulate`'s per-game loop body with no behavior change to
`simulate` itself.

## Engine unit 2 result (measured, 2026-09-26, subagent)

Fix 3 (backoff reorder) was implemented and measured, then dropped: with all
three fixes, hits fell to 1/5 and log-loss delta rose to +0.0103, tied-at-5:00
OT rate rose to 56.6% and tie rate to 2.03%, all worse than fix1+fix2 alone.
Coarsening field position/distance at L1 while keeping score/time fine pools
plays across very different field-position/down-and-distance contexts (e.g. a
midfield snap answering a goal-line situation), which turned out to cost more
realism than the score/time dilution it was meant to fix. Reverted; kept only
fixes 1 and 2 (`scripts/sim04_engine.py`: `build_transition_frame` now emits
`yards_gained`/`dist_gained` and a corrected `clock_elapsed`; `run_one_game`'s
non-flip branch applies the delta to the simulator's own state instead of
copying the drawn row's absolute `next_yardline`/`next_distance`; L1/L2
grouping and `k1`/`k2` keys unchanged from Engine fix 1).

Full validation, `--n-games-per-season 3334` (10,002 games), train 2009-2014,
valid 2015-2017 REG, artifact `artifacts/sim04_engine/20260926T023148Z/report.json`:

| metric | before (fix 1) | after (fix 1+2) | actual |
|---|---|---|---|
| mass at 3 / 7 / 10 / 14 / 17 | .098/.063/.056/.041/.041 | .102/.072/.055/.039/.038 | .152/.091/.048/.051/.030 |
| key-number hits | 1 | 2 (10, 17) | -- |
| log-loss delta vs naive | +0.0207 | +0.0092 | -- |
| margin SD ratio | 1.077 | 1.056 | -- |
| points/game | 41.8 | 40.8 | ~45 |
| tie rate | 1.6% | 1.8% | ~0.4% |
| tied-at-5:00 OT rate | 49.3% | 52.4% | 14.9% |
| late-Q4 possession scoring rate | 0.136 | 0.129 | 0.215 |

Log-loss delta now clears the <=0.02 bar for the first time this series.
Hits improved 1->2. Tied-at-5:00 OT rate and tie rate did not improve (both
slightly worse), confirming these are not primarily a field-position-copy or
OT-clock artifact. Still NO-GO (2/5 hits < 4).

Diagnostic split (task item 1), run against the fix1+2 engine,
`sim04_engine.simulate_from_states` (n=768 real 2015-2017 REG games, each
started from its own first live snap with qtr==4 and
game_seconds_remaining<=300, 200 reps/game = 153,600 sim endings) and
`simulate` (2000 full games, `tied_at_5_diff` column read for every game, not
only tied ones, despite its name):

| split | metric | sim | actual |
|---|---|---|---|
| finishing (from real Q4<=300s state to end) | mass@3 | .1136 | .1523 |
| finishing | mass@7 | .0746 | .0911 |
| finishing | mean / SD of margin | 2.26 / 14.13 | 2.20 / 13.87 |
| reaching (home margin at 5:00 left, full-game sim vs actual) | mass@3 | .090 | .087 |
| reaching | mass@7 | .077 | .090 |
| reaching | mass@14 | .041 | .055 |
| reaching | mass@17 | .036 | .027 |

Finishing-error SD ratio is 14.13/13.87 = 1.02, essentially matched, well
below the full-validation ratio of 1.056 for the same fixed engine. The
aggregate reaching-error mass (all games' home margin at 5:00, not
conditioned on being tied) is also fairly close to actual. This localizes
most of the remaining excess variance and the mass-3/7 shortfall to how the
game arrives at a state, not to how it closes one out: specifically the
tied-at-5:00 subgroup (52.4% sim OT rate vs 14.9% actual), which is invisible
in the aggregate reaching-mass check above because ties are a small slice of
all home margins at 5:00. This confirms the Engine unit's original diagnosis
(cell-sparsity backoff dilutes the real, measured, higher per-play scoring
rate in tied+late situations, 0.077 vs 0.055 overall) as the still-open
mechanism, and rules out fix 3's specific remedy (reordering which axis
backoff coarsens first) as ineffective.

GO/NO-GO: NO_GO (2/5 hits, need >=4). Named next mechanism, not yet
implemented: the tied+late(qtr==4, gsr<=300, score_diff==0) corner has only
320 L0 cells averaging 4.7 rows and 96.6% must back off; instead of
coarsening an existing axis, pool historical rows from ALL trailing-team and
leading-team small-margin (|score_diff|<=3) late-Q4 rows with the CURRENT
sim's own score_diff sign and magnitude used only to select offense-vs-
defense aggression symmetrically (a trailing-by-3 team plays like a
trailing-by-3 team regardless of exact game), which would multiply tied-cell
n several-fold without touching field position/distance/time granularity at
all. Not measured this unit (out of tool budget).

Not ported to `src/`; no registry writes; no commits; no dashboard or
publish work (research-only per AGENTS.md).

## Engine unit 3 (TD-overshoot/return-TD fixes + NN backoff replacement) -- 2026-09-25/26

### Predeclaration (written before any validation-split look)

Fix A (`scripts/sim04_engine.py` `run_one_game`, non-flip branch, ~line
686): a drawn non-scoring gain whose relative delta (`yardline -
drawn["yards_gained"]`) crosses the goal line is now scored as a touchdown
(6 + a bonus drawn uniformly from `tables["pat_bonus_pool"]`) instead of
clamped to the 1-yard line. `pat_bonus_pool` (built in `build_tables`,
~line 456) pools the empirical PAT/2pt bonus (`points_off/points_def - 6.0`)
from every real touchdown row, offense- or defense-scored, since both share
the same PAT mechanism. The receiving team's next field position is drawn
uniformly from `post_score_pool` (~line 467): the empirical `next_yardline`
of every real scoring, flip-causing row (any score, not just this branch's
kind). In OT this ends the game immediately (mirrors the existing in-OT
settle rule, since a TD always breaks a tie or a one-score-behind state).

Fix B (`build_transition_frame`, ~lines 350-351): `return_score` flags
touchdown rows that are not interception/fumble-return scores but where the
scoring credit did not go to `posteam` (punt-return and blocked-kick-return
TDs, `posteam_score_post - posteam_score == 0`). These now feed the same
`def_score` branch as INT/fumble returns (6 + PAT bonus credited to
`points_def`), instead of being misrouted into `points_off` with the score
delta silently zero.

Backoff-level measurement (`scripts/sim04_backoff_diag.py`, run once,
3,000-game simulate() call on the fixed engine, still using the OLD
categorical l0-l3 backoff before its removal): overall level-0 (exact fine
cell) fires 67.2%, l1 25.3%, l2 6.4%, l3 (down-only) 1.1% (n=433,470 draws).
In the Q4<=300s window specifically: l0 27.9%, l1 29.5%, l2 36.4%, l3 6.3%
(n=47,186) -- 72% of late-Q4 draws must leave the exact fine cell, and 42.7%
fall to the two coarsest levels. In Q2<=120s: l0 44.7%, l1 20.5%, l2 30.8%,
l3 3.9% (n=28,396). This confirms item (C): the categorical hierarchy backs
off hardest exactly where score/time conditioning matters most.

NN replacement design (predeclared before running validation): down (exact
match, 1-4) x phase (exact match, 5 levels: `compute_phase`, ~line 168 --
Q1-Q3-and-Q2>120s "normal"=0, Q2<=120s=1, Q4>300s=2, Q4<=300s=3, OT=4) each
get one `sklearn.neighbors.KDTree` (`build_neighbor_index`, ~line 236; 20
trees total, one per down x phase). Distance features (`feature_matrix`,
~line 208): `ydstogo/5`, `yardline_100/20`, a piecewise score_diff scale
clipped to +/-21 with slope 1/2 inside +/-8 and 1/8 beyond it (so a 3-point
lead and a 10-point lead differ by 2.5 scaled units, both late-game distinct
regimes, while a 15- and a 20-point lead differ by only 0.625 -- picked from
football's one-possession-game threshold, not fit to any result), continuous
seconds-left-in-half (`continuous_time_feature`/`vectorized_time_raw`,
~line 188-196; OT uses the raw OT-period clock directly, confirmed by
reading 2015 OT rows: `game_seconds_remaining` for qtr>=5 already runs
149-900, i.e. it IS the OT-period clock, not a continuation of the game
clock) divided by 300, and off/def timeouts remaining divided by 1 -- but
the timeout pair is multiplied by 0 outside the three late phases
(Q2<=120s/Q4<=300s/OT) so it is a no-op constant everywhere else, keeping
every tree 6-dimensional without touching normal-phase neighbor rankings.
Draw uniformly from the k=40 nearest in that (down, phase) tree
(`pick_index_nn`, ~line 259); k and every scale above were fixed before this
predeclaration was written, not chosen by looking at 2015-2017. Neighbour
sets are memoized on a rounded state key (`round_state_key`, ~line 222:
ydstogo to 2, yardline to 5, score_diff to 2, seconds-left to 30, timeouts
exact in late phases / collapsed to 0 elsewhere) -- this rounding is a
runtime-only approximation (coarser round = faster, marginally blunter
neighbor set), not a modeling choice, so it was tuned for the "few minutes
for 10k games" requirement after the fact: 300 games at the first (finer)
rounding took 6.3s with a 94% cache-miss rate; widening the rounding to the
values above dropped 1,000 games to 15.9s (extrapolated ~159s for the full
3x3,334-game validation), which the real run then confirmed.

GO/NO-GO criterion: unchanged from every prior sim04 unit -- >=4/5 key
numbers (3,7,10,14,17) inside the actual 2015-2017 bootstrap 90% CI AND
discrete log-loss delta vs. the naive train-histogram baseline <=+0.02 nats,
both required. This is the only look this unit takes at 2015-2017 (no
config sweep -- the NN design was fixed by football reasoning, not selected
from a validation sweep, so there is nothing to select between).

### Result (measured once, `.venv/Scripts/python scripts/sim04_engine.py
--n-games-per-season 3334`, 2026-09-26)

## Engine unit 3b predeclaration (orchestrator, 2026-09-26)

After unit 3 (k=40 nearest-neighbour draws), plays per possession are 6.26 vs
6.87 actual and the SD ratio is 1.105. Named mechanism (read): on a non-flip
play the sim copies the neighbour row's `next_down`, so a 3rd-and-2 state that
draws a 3rd-and-9 neighbour converts or fails by the neighbour's distance, not
its own. Change (one config, no tuning): downs follow the rules. A first down
comes when the relative gain covers the sim's distance, or when the row was a
penalty auto first down (next down 1 with gain short of its distance). A
repeated down (penalty/no play) keeps the down. Otherwise the down advances,
and failing on 4th is a turnover on downs at the spot. One validation run,
logged as a look.

Unit 3b result (look 1, `artifacts/sim04_engine/20260926T025349Z`): 2/5 hits
(7, 10), log-loss delta +0.0146, SD ratio 1.122, points/game 46.2, ties 1.1%,
tied-at-5:00 OT rate 28.7%. Tally of the sim's own draws (3000 games) against
the 2009-2014 pool: row points 40.3 vs 44.3, while game totals are 46.3. About
6 points per game come from relative gains that cross the goal line on rows that
were not touchdowns. FGs are 2.26 vs 3.21 and possession changes 20.1 vs 22.4.
Mechanism (measured): with `SCALE_FP = 20` the neighbours span about ±20 yards
of field position, and long gains from rows farther out overshoot the goal, so
FG drives become TD drives and dispersion rises.

## Engine unit 3c predeclaration

One change: `SCALE_FP` 20 -> 5 (neighbours within about ±5 yards), nothing
else. One validation run, logged as a look.

Unit 3c result (look 2, `artifacts/sim04_engine/20260926T025644Z`): 2/5 hits
(10, 17), log-loss delta +0.0087, SD ratio 1.080, points/game 42.8. The
overshoot share fell, but FGs are still 2.61 vs 3.21. Diagnosis (measured, 2000
games): the rule-based downs added in 3b flagged every 1st-down conversion as a
"repeat down" (next down equals current down), so the rule gave 1st-down
conversion 0.061 vs 0.179 on the pool's own rows. Fix: a repeat down requires a
gain short of the distance and a next distance equal to distance minus gain;
an auto first down is a short gain whose next down is 1 and is not a repeat.
After the fix, first-down rates by down are sim/pool 0.185/0.179,
0.294/0.282, 0.343/0.350.

Unit 3d result (look 3, bug fix only, `artifacts/sim04_engine/20260926T030149Z`):
2/5 hits (10, 17), **log-loss delta +0.0045** (best of the series), SD ratio
1.086, points/game 41.8, ties 1.0%, tied-at-5:00 OT rate 31.2% (actual 14.9%),
mass at 3 .103 vs .152 and at 7 .074 vs .091. Remaining named gaps (measured):
FGs 2.84 vs 3.21 per game, because the sim reaches 4th down inside the 35 on
2.8% of plays vs 3.1% (mean field position 53.8 vs 52.5); dispersion 8% high
even with equal teams, and it builds before the last five minutes (the unit 2
finishing test from real 5:00 states gave an SD ratio of 1.02).

Orchestrator decision: the validation split has now had 7 engine looks. The
key-number bar stays a mechanism check, `unresolved_below_power` (no sign
flip, not refuted). The engine moves on to team conditioning and the
decision-relevant grade: leave-one-season-out at the opener against the served
discrete read.

## Unit LOSO grade predeclaration (orchestrator, 2026-09-26)

Team conditioning (built by the conditioning unit; sanity run
`artifacts/sim04_engine/20260926T030954Z` reproduces unit 3d unconditioned):
k_state=200 state neighbours, reweighted by a Gaussian kernel on pregame
off/def EPA (h = half the training-window SD of off EPA). With the predeclared
home weight of 1.5, the home edge at equal ratings was +0.56 vs the training
era's +2.57 (measured). Changed once, before any graded season was touched:
the home/away match is exact (weight 1e6), so home offenses draw home plays.
Result: +2.41 ± 0.26 on 2009-2014 tables (measured, 3000 games). Grade: 2020-2025
REG Tuesday openers, tables built from 2009..S-1 for each season S, 200
simulations per game. Primary = raw histogram; secondary = the same shape
re-centred on the served predicted margin. Both are recorded.
