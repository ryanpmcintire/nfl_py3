# Cloud offload of MOD-25e 3-seed validations

## Goal
Run MOD-25e 3-seed simulator validations in a cloud session holding only a git clone. Decision: inputs rebuilt from public nflverse with repo scripts; nothing committed or uploaded.

## State (trace only, nothing rebuilt)
- Whole `artifacts/**` and `data/**` are gitignored (only `artifacts/.gitkeep`, `artifacts/prospective/challengers.json` tracked). tests/scratch is ignored too.
- crH import closure is ~80 scripts (all tracked). Paths read: pbp snapshot 20260925T202544Z (sim04_engine, mod25_generator, mod25e_adj/kick/late/rtd/revert_sim), 20260929T191306Z (mod25_mechanisms, kfit, kstate, sim09_dynamics), schedules 20260915T170343Z (mod25d_variance) and 20260908T162105Z (budget, sim09_dynamics), game_features_pbp.parquet, data/raw/officials + officials_pfr_wayback (sim09_f3), data/players/participation 20260813T131635Z, artifacts/sim09/f2/real_fit.parquet, artifacts/mod25c/policies2_2009_2017.npz, artifacts/mod25d/ep_model.joblib, ~45 artifacts/mod25e3/*/fit.json.
- Rebuildable from nflverse: pbp snapshot (`nfl-ats pbp-ingest --start 2009 --end 2025`, nflreadpy), schedules (`fetch-data`), game_features_pbp (`build-features` then `build-pbp-features`).
- NOT byte-identical: game_features_pbp.parquet was rebuilt 2026-10-18 from pbp snapshot 20261013T220620Z (not 20260925), and game_features.parquet embeds Splash decision lines (non-public, 2026 only). Sim reads only `home/away_off/def_epa_per_play` for REG games. nflverse revises history; only 33 pbp columns kept (manifest).
- Chain of fit.json files is the real blocker: each came from an earlier experiment (`fit()` in mod25e_<name>.py, tracked), several need prior sim outputs or tests/scratch data.
- geo/fit.json: `mod25e_geo.py fit` uses cache tests/scratch/e115g (outer.pkl, inner_2009..2017.pkl, built by the same script, line 155-200). ldr: `mod25e_ldr.py fit`. rg/fit.json: tests/scratch/e162/rg_fit.py reads rg_pool.npz and gp_meta.npz, produced by tests/scratch/e162/geo_phase.py and rg_pool.py (sim runs). All hard-code `F:/Repos/nfl_py3`.
- Scoring: scripts/mod25e_e100.py imports mod25e_budget/e96/analysis (tracked, schedules 20260908T162105Z); tests/scratch/e114/tgt.py and e155/build3.py (+ e131 helpers) are untracked, hard-code F:/ paths, read sim play_*.parquet.
- Hardcoded Windows paths: mod25d_variance NV_DIR and 9 `--cache` defaults, mod25e_rulestr/sim09_f3/sim09_urgency NV (C:/Users/Ryan/.../scratchpad/nv, purpose unknown), smoke_gs.sh `cd /f/...`, .venv/Scripts/python.exe.

## Tried
Static trace only.

## Next
1. Decide: bundle artifacts (fit.jsons, ep_model.joblib, mod25_generator, f2 real_fit, policies npz; est. <200 MB) as private cloud storage, since a clean rebuild reproduces experiments, not these exact fits. Owner rule says no upload, so a rebuild needs: pbp-ingest -> fetch-data -> build-features -> build-pbp-features -> `mod25_generator real`, `fit` -> mod25d ep_model fit -> each mod25e_* `fit` in flag order -> geo_phase sims -> rg_fit -> geo/ldr fit. Estimate: data 1-2 h; fit chain several hours to days plus fixing hardcoded paths.
2. Move e162/e114/e155/e131 scripts into tracked scripts/ with REPO-relative paths.
3. Replace NV/--cache C: defaults; pin snapshot ids via env.

## Open
Which fit steps need sim outputs; what NV dir holds; whether 2009-2017 EPA ratings change between 20260925 and 20261013 snapshots.

## Decision 2026-10-18 (owner)
Owner approved bundling the data (release assets on a separate private GitHub repo, or a private data repo) and also wants to share the full data set with a friend. Rebuild-from-public is dropped (hours-days of fit-chain reconstruction, not byte-identical). Plan: separate private repo holding release assets (tar.zst parts <= 1.9 GB), excluding secrets/junk/other people's data; inventory + secret scan in docs/lanes/data-share.md first. Then move the e114/e155/e162/e131 scoring scripts to tracked paths with REPO-relative paths, fix the hardcoded NV/cache paths, and test one 3-seed in the cloud v local.

## Shelved 2026-10-18 (owner)
Full data set is ~55 GB (data 33.7, artifacts 21.7); hosting it costs money, so the owner shelved cloud offload and data sharing. Nothing was uploaded or committed; bundle build and secret scan were stopped, the seed-99 instrumentation output was deleted. If revived: the simulator-only bundle is <1 GB (estimate, not built).
