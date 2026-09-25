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
