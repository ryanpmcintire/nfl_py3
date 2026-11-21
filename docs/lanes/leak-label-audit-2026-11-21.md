# Leak and label audit 2026-11-21 (LEAD-75 pregame check, ENG-54 unit 4)

## Goal
LEAD-75: row-level pregame check of the pass-rate input to the paired-wind interim look. ENG-54 next unit, first part: opener-transfer cover signs and forward-log base labels audited for silent outcome defaults and name joins; re-record only cells whose number changes.

## State
Done 2026-11-21 10:20 (date). Scripts: scripts/lead75_leak_rows.py (artifacts/lead75/leak_rows.json); scripts/eng54_unit4_defaults.py (artifacts/eng54/label_audit_unit4.json). Details in docs/lanes/lead75.md and docs/lanes/eng-54-outcome-label-default-audit.md (Unit 4). All measured unless marked.
- LEAD-75: 53 paired rows, 48 games, 106 team-sides: 0 use pbp from a game on or after the Tuesday capture, 0 from the game itself or later; Tuesday-cut rebuild changes 0 of 48 games (max 0.0). No leak. Interim reproduces at +49.35 (recorded +46.26 from an unversioned script) on the same 7 games; the gap is not the pass-rate input.
- ENG-54 opener-transfer: 0 outcome defaults. College name join keeps 3110 of 3110 games, 0 missing points, 0 of 3054 label mismatches; NFL 0 of 1503. Feature default: unit3.py:176 sets open_move 0 on 2231 of 3734 extended rows (all 2011-2019), so the v3 all-graded cell carries a zero move term for the SBR era; 2020-2025 cells unaffected.
- ENG-54 forward logs: mod24 "served" side equals the pool side on 115 of 115; armb model side differs from pool on 29 of 70, draftcap 11 of 29 (14 of 43 without a served frame, left empty); no registry cell reads them. Six FTN forward cells say "vs served" but the base is the locked raw card (read); numbers unchanged.
- Records (audit unit): none. Repair unit (same day, below) re-recorded 8 cells.

## Tried
Truncation rebuild at each week's Tuesday capture; deadline-leg and margin alternatives for the interim gap (one leg per game; ats_margin equals result minus spread_line, slope unchanged).

## Next
ROADMAP LEAD-75 update: 2026-11-21 row-level check (measured): 0 of 106 team-sides use pbp on or after the Tuesday capture; Tuesday-cut rebuild changes 0 of 48 paired games; no leak. Interim reproduces at +49.35 from versioned code (recorded +46.26). Forward only; next look 2027-01-10 with scripts/lead75_leak_rows.py pairing.
ROADMAP ENG-54 update: 2026-11-21 unit 4 (measured): no silent outcome default in the opener-transfer chain (0 of 3110 name-join losses, 0 label mismatches college and NFL); unit3.py:176 zero-fills open_move on all 2231 SBR-era rows (v3 all-graded cell only); forward-log served labels move no recorded number; six FTN forward cell descriptions say served for a raw-card base. Next unit: repair unit3.py:176 and replace the v3 all-graded cells; fix the six FTN descriptions; then the served-base re-records (mod42, mod43, mod44, mod47, lead76_rule_prior, mkt27).

## Open
Extended NFL population has no result column, so its home_covered was not re-derived against the feature opener (outcome label check covered the 1503-row fit population only). v3 2020-2025 subset cell: the repair moves it (below); not re-recorded on the root's instruction, batch ready.

## Repair 2026-11-21
10:13 (date). Measured unless marked; scripts and logs: scripts/opener_error_transfer_unit3.py (SBR move source), scripts/lead75_interim_bootstrap.py (artifacts/lead75/interim_wk7_bootstrap.json), tests/scratch/eng54/{unit3_baseline_repro,unit3_repaired,pool_repair}.log, batches tests/scratch/eng54/batch_*.json built by build_repair_batch.py.
- ENG-54 v3: unmodified script reproduces the recorded cells exactly (artifact 20261121T151040Z). Repair: open_move for 2011-2019 now SBR close minus open from data/processed/sbr_odds.parquet (proxy opener equals that file's open on 2231 of 2231 rows; 0 moves missing in any season; nothing defaulted). Artifact 20261121T151245Z. all_graded +0.9586 [+0.2711, +1.6511] P+ 0.9945 -> +0.6184 [-0.3038, +1.4233] P+ 0.9121, n 3234, decisive 124 games 72-52 (was 179, 105-74); re-recorded, unresolved_below_power kept (no closing ground either way). 2020_2025_subset also moves, because the LOSO plus fit trains on SBR-era rows: +0.6653 [-0.3199, +1.4966] P+ 0.9191 -> +0.5988 [+0.1289, +0.9804] P+ 0.9944, decisive 19 games 14-5 (was 42, 26-16); NOT re-recorded (root instruction premised it unchanged); ready: tests/scratch/eng54/batch_v3_2020_2025_NOT_RECORDED.json.
- LEAD-75: lead75_paired_wind_interim_wk7 replaced: +46.26 [-496.9, +184.3] P+ 0.64 SE 180 (unversioned) -> +49.35 [-513.05, +186.22] P+ 0.652 SE 176.1, same 7 games, game-level bootstrap with replacement, 4000 draws, seed 20261121, 95% percentile, 1 all-zero draw dropped. Declared sign negative; interval spans zero, so unresolved_below_power kept.
- FTN: six cells re-recorded, description only (raw-card base), numbers identical (git diff: only description lines changed); base-tags CSV already tags them raw_model.
- `weak-signals pool --league nfl --effect-units accuracy_points` exit 0. Changed shards: lead75_paired_wind, mod10_graph_blitz, mod23_unit12_ftn_scheme, opener_error_transfer_v3.

ROADMAP ENG-54 update: 2026-11-21 repair (measured): opener_error_transfer_unit3.py now takes the SBR-era open move from SBR close minus open (2231 of 2231 rows, none defaulted); v3 all-graded cell re-recorded +0.62 [-0.30, +1.42] P+ 0.912 (was +0.96 [+0.27, +1.65] P+ 0.995), unresolved_below_power; the 2020-2025 subset also moves to +0.60 [+0.13, +0.98] P+ 0.994 through the shared LOSO fit, re-record pending orchestrator; six FTN forward cells relabelled raw-card base, numbers unchanged. Next: served-base re-records (mod42, mod43, mod44, mod47, lead76_rule_prior, mkt27).
ROADMAP LEAD-75 update: 2026-11-21 (measured): interim cell replaced from versioned code (scripts/lead75_leak_rows.py + scripts/lead75_interim_bootstrap.py): +49.35 [-513, +186] P+ 0.652 on the same 7 Week 7 outdoor MEX games (was +46.26 [-497, +184] P+ 0.64, unversioned), unresolved_below_power; forward only, next look 2027-01-10.
ROADMAP MOD-10 and MOD-23 update (FTN forward 2026 W1-7 cells; row ids inferred from the family names, ROADMAP grep for forward2026_ftn not run, tool cap): 2026-11-21 description correction only: the six forward cells score against the locked raw card, not the served probability; numbers unchanged.
