# lead59-archive-battery

## Goal

LEAD-59: let the 2009-2014 officials archive reach the crew traits, then
re-run the officials archive battery on production with the archive on and
report every arm with `probability_positive`. Done when the battery stages
run clean on the fixed builders and the numbers are recorded through
`nfl-ats weak-signals record` by the coordinator.

## State

Src fix done, uncommitted (opencode lane D, `big-pickle`, verified by the
coordinator 20:15 ET): `src/nfl_ats/officials_flag_features.py` (crew
traits from crew rows, penalty left-join with NaN, rookie floor from the
population), `src/nfl_ats/crew_tilt_refresh_overlay.py` (drop referee-games
absent from the penalty-type snapshot). Measured no-op with the archive
off: crew-tilt trait tables and all five builders bit-identical.
`scripts/officials_archive_battery_eval.py` restored from `b7ed31d~1`
(comments stripped). Battery re-run is opencode lane G (`mimo-v2.5-free`,
relaunched 20:32 ET after nemotron quit with no output): brief
`tests/scratch/lanes/laneG_brief.md`, report `laneG_report.md`; stages
traits, opener, served-proxy, era, composition, card; never `record`.

## Tried

- A literal inner join in `_referee_name_season` (what the doc asked)
  would drop 287 modern referee-games whose zero holding count is genuine
  and shift the holding trait sum 461.65 -> 508.70; the lane implemented the
  narrower filter instead.

Lane G done 20:44 ET; the coordinator regenerated the record commands
(`--stage record`), ran all 23 with `--replace` (measured: 23 `recorded`
lines), and wrote the results into ROADMAP LEAD-59 and the tail of
`docs/officials_archive_battery.md`. Uncommitted.

## Next

Commit. Then decide whether the penalty-type trait binning with the archive
on needs a NaN filter (diagnostic only today).

## 2009-2014 sweep probe 2026-09-17 (measured)

Bounded probe of `scripts/officials_wayback_sweep.py` (subagent, defaults
untouched, first 20 2009 REG games as the predeclared set, killed by its
25-minute cap during game 8): 7 attempted, 42 rows, 6 officials-block hits.
Throttle present but passable: 5 of 7 games drew CDX backoff (900 s total,
60/120/240 s schedules), no final 403. Rate ~21-23 games/h, extrapolating to
~51-102 h wall for all 1,536 REG 2009-2014 games. Implication: the full sweep
is not a foreground task; it needs a background scheduler job spread over
weeks, or a sampled archive. Probe wrote only under ignored `data/raw/`
(`officials_pfr_wayback/probe20260917Tb20/`, no parquet — killed before final
write); no tracked file touched.

## Open

Whether the `n_censored_2015_debut` key should be renamed now that the
floor is population-derived (cosmetic).
