# Revision-safe historical quarterback depth identities

PER-04 uses the official nflverse annual depth-chart releases without
pretending their legacy weekly rows have publication timestamps. The retained
2009–2024 source has `season`, `week`, and depth rank but no revision or
observation time. Consequently a week-W row is never admitted to a week-W
prediction.

## Conservative chronology

`canonicalize_historical_depth_charts` joins every source season/week to the
canonical schedule and defines:

- `observed_at_utc`: null, because the legacy source does not provide it;
- `effective_at_utc`: one microsecond after the final kickoff in that source
  season/week; and
- `provenance_mode`: `legacy_prior_week`.

The PER-02 selector uses `effective_at_utc` for all visibility and age checks.
Timestamped 2025-forward revisions use their real observation time as both
observed and effective time. Thus the two source regimes share one selector
and cannot use later revisions.

The historical parser keeps only the offensive `QB` depth position. The
official 2009–2024 files contain 19 duplicated team-week/player identities;
some assign one player two ranks. A player cannot be both QB1 and QB2, so the
canonical archive retains the best listed rank once and marks
`source_role_conflict=true`. It never manufactures a second player. Missing
week chronology, missing effective time, an observation after its effective
time, stale state, a missing QB2, and manifest hash drift all fail closed or
remain explicit missing coverage.

## Measured archive and coverage

Measured on 2026-09-02 from
`data/quarterbacks/depth/historical/20260902T225115Z/manifest.json`, the
immutable snapshot contains 22,962 QB rows across all 32 teams and every
requested season from 2009 through 2024. The parquet SHA-256 is
`9ee4ce3457139053d621caaee0a9d33928ec3a4424f52c5ad7801035e73c8d81`.
Each season contains 17–19 regular-season source weeks; 99.06% of source
team-weeks contain at least two distinct quarterback identities.

Measured against `data/processed/game_features.parquet` at the default
kickoff-minus-24-hours decision time with the existing 14-day age ceiling:

- QB1 identity covers 8,140 of 8,690 historical team-games (93.67%);
- named QB2 identity covers 8,065 of 8,690 (92.81%);
- both teams have QB1 for 4,068 games;
- both teams have QB2 for 3,994 games; and
- all 510 Week-1 team-games remain intentionally uncovered because no earlier
  in-season weekly observation exists.

Weeks 2 onward are 99%+ covered in the first measured weeks, and each season's
named-QB2 coverage is 91.20%–93.63%. This is adequate historical identity
coverage for feature construction while preserving an honest missing Week-1
regime. It is not evidence of ATS value and no outcome experiment was run.

## Rebuild

The archive is gitignored and reproducible from the official annual releases:

```powershell
.\.tools\uv.exe run nfl-ats depth-history-ingest `
  --start-season 2009 --end-season 2024 `
  --features data/processed/game_features.parquet
```

Build the PER-02 features from a specific historical snapshot explicitly:

```powershell
.\.tools\uv.exe run nfl-ats build-qb-features `
  --depth-root data/quarterbacks/depth/historical `
  --depth-snapshot 20260902T225115Z
```

The output manifest records the immutable PBP, player, and depth snapshot IDs,
the availability source and optional rate-file hash, feature version, decision
parameters, and coverage counts. No active model profile includes the
`quarterback_depth` family.
