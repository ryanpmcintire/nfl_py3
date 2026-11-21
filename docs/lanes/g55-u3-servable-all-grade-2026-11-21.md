# g55-u3-servable-all-grade-2026-11-21

## Goal
The one declared G55-U3 grade (goal-55-backwards-2026-11-21.md "G55-U3 declaration" + root pin): S_ALL = P2-L harness with servable O1 (U1 gameday actives), O2 (U2 expected snap share rebuilt from U1 actives, no stand-in), O3 (projected QB INA on the actives list), known_at_deadline games only. Model alone at the Tuesday opener, 1,503 decisive 2020-25 games, looks all/quiet/moved = 3, family g55_deadline_availability, 50-draw within-season matched null.

## State (2026-11-21 14:47 DONE, all measured)
- Scripts: scripts/g55_u3_servable_all.py (modes build/grade/null/summarize; ruff clean); scripts/g55_u2_expected_snaps.py gained `--no-stand-in` (NA active = not active). U2 rebuilt to artifacts/g55_u3/u2_rebuilt/ (active rows only on U1 known 2020-25 team-games; span 2, fraction 0.338 every fold).
- Leak guard: synthetic team-games by season and slot equal the known counts (build_summary.json); 0 on SNF/MNF/unknown games; pre-2020 has none (U1 script has hardcoded SEASONS 2020-26, no season argument). Harness is walk-forward weekly (read clv.py:2184-2185), so 2026 rows never train 2020-25. O3 nonzero 101 games.
- Base 802-701 (declared record) but probabilities differ from the 11-11 artifact (max 0.133, median 0.003, 46 pick flips; quiet 437-400 v 439-398): served features rebuilt today 13:38. Paired arm-v-base uses today's inputs for both.
- S_ALL all 809-694 v 802-701: +0.47 pts, season-block [-0.77, 1.47] P+ .785 (week-block P+ .73); log loss 0.69904 v 0.69678 (worse); Brier 0.25218 v 0.25164 (worse); changed picks 95, arm 51-44. Quiet 450-387 v 437-400 +1.55 [0.39, 2.41] P+ .99; moved 355-298 v 361-292 -0.92 [-2.68, 0.76] P+ .13. Per season (arm v base): 114-106/119-101, 128-108/130-106, 138-110/133-115, 136-130/132-134, 150-116/148-118, 143-124/140-127.
- Null (50 draws, 14:35-14:47): all wins pct 87.0 (null mean 798.0, range 780-835), log loss pct 62; quiet 89.0 / 42; moved 65.0 / 54.
- Serving test FAILED on all three legs (809 < 827; null pct 87 < 95; log loss worse). Recorded 3 cells unresolved_below_power, family g55_deadline_availability, shard registry/weak_signals/g55_deadline_availability.json (new, untracked); pool loads (artifacts/g55_u3/pool.log).
- Artifacts: artifacts/g55_u3/ results.json, per_game.parquet, build_summary.json, batch.json, null/draw_000..049.json.

## Tried
- Build 1: active players with no U2 row (6,837, all 2026; U2 has no 2026 snaps) got sev2 = 1. Fixed before any grade: sev2 0 when no forecast.

## Next
1. Root: verify per_game.parquet record, commit scripts + shard, move this lane to done/. Per the declaration, the arm is not re-tuned on these games.
2. ROADMAP G55 update: S_ALL 809-694 (+0.47 P+ .785, null pct 87), serving test failed; quiet +1.55 P+ .99 is one of 3 looks, not a finding.

## Open
- O3 reads the pin literally (status INA); non-ACT non-INA projected QBs 3-10 a season and not-listed 6-11 are left at share 1.
