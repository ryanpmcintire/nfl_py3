# Share data/ and artifacts/ via private GitHub repo release assets

## Goal
Publish data/ and artifacts/ (gitignored) as release assets on a separate PRIVATE repo so cloud sessions can run the simulator and the owner can share with a friend.

## State
- Size inventory measured (python scandir, scratchpad inv.txt): no single file over 1 GB (so none over 2 GB). 450 top-level entries.
- data/ approx 33.7 GB: raw 24.2 GB (109k files; snapshots dated 20260812..20261013 plus ~40 source folders), processed 4.87 GB, cfb 1.68 GB, market 1.23 GB, players 0.99 GB, pbp 0.30 GB, environment_recovery 13 MB, quarterbacks 6 MB, splash/lock_failures/tmp/uv_cache/scheduler_* each under 1 MB.
- artifacts/ approx 21.7 GB: mod25e3 15.5 GB (24k files), rehearsal_lockday 2.2 GB, sim09 0.98 GB, margins 0.87 GB, mod25e 0.51 GB, cache 0.13 GB, rest small (mod25c 33 MB, mod25d 50 MB).
- Simulator-minimum inputs (from cloud-offload.md): pbp snapshots 20260925T202544Z and 20260929T191306Z, schedules 20260915T170343Z and 20260908T162105Z, game_features_pbp.parquet, data/raw/officials, officials_pfr_wayback, data/players/participation 20260813T131635Z, artifacts/sim09/f2/real_fit.parquet, mod25c/policies2_2009_2017.npz, mod25d/ep_model.joblib, about 45 artifacts/mod25e3/*/fit.json. Estimated under 1 GB (not yet sized exactly).
- Provisional exclusions: data/uv_cache, data/tmp, data/scheduler_*, data/*_log.txt, data/lock_failures, data/splash (pool data, other players), artifacts/rehearsal_lockday and artifacts/backups pending review.
- Third-party paid or scraped (license review before sharing with a friend): data/raw/odds_api_props, beat_reporters, prediction_markets, sbr_odds, vegasinsider, public_betting*, sagarin, arctic_shift (reddit), gdelt; data/market.

## Tried
- Secret and sensitivity content scan (scratchpad scan.py: key-name regexes and email regex over text files under 3 MB, plus suspicious filenames, plus per-subfolder sizes of raw/market/processed) launched as background task b7pkzzze4; output buffered to scratchpad scan.txt, not yet finished when this was written. game_features.parquet embeds Splash decision lines (non-public).

## Next
1. Read scan.txt (scratchpad F--Repos-nfl-py3/.../scratchpad/scan.txt); verify no secret values; list hits by path and key name only.
2. Size the minimum bundle exactly and the full bundle after exclusions.
3. Plan tar.zst parts at most 1.9 GB each (full bundle about 50 GB is roughly 30 parts after compression; minimum bundle 1 to 2 parts).

## Open
Whether to ship third-party scraped/paid sets to a friend; whether mod25e3 intermediate sim outputs (15 GB) are needed or only fit.json.

## Shelved 2026-10-18 (owner)
Full data set is ~55 GB (data 33.7, artifacts 21.7); hosting it costs money, so the owner shelved cloud offload and data sharing. Nothing was uploaded or committed; bundle build and secret scan were stopped, the seed-99 instrumentation output was deleted. If revived: the simulator-only bundle is <1 GB (estimate, not built).
