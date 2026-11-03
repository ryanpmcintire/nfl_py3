# MKT-27

## Goal
Total-line level as a margin-width input to the discrete lattice (ROADMAP MKT-27).

## State
Unit 1 done 2026-11-03 ~17:10 (date). Script scripts/mkt27_total_lattice.py (stages describe|typical|grade|evaluate), outputs artifacts/mkt27/.
- read: served lattice does NOT use the total. mass_preserving_lattice.py _POOL_COLUMNS (line ~160) has no total; band_read (:112) selects pool by |pool_line - line| only. total_line appears only in margin_variance.py (CFB variance features, :33).
- measured 2011-25, 3919 games, total terciles within season, stratified by season x spread bin (9 looks, permutation null shuffling within cells; total = game_features total_line, a closing proxy): top minus bottom tercile mass at 3 +0.04pp, 7 +0.99pp [-1.16,3.16], 10 -0.71, 14 -0.64, 17 -0.74 [-2.15,0.62], any key -1.06, abs ATS residual +0.18 pts [-0.44,0.81], push rate -0.90pp [-2.18,0.41] P+ .085 (perm p .15). All perm p >= 0.15. No shape move resolved.
- measured typical pairs (pool 2017-25): total-conditioned cover prob differs from served by -4.6pp (home +10, total 51) to +3.1pp (+7, total 38); n 205-374 per read, so noisy.
- measured 2020-25 opener grade (Tuesday consensus total, 1527 games, 1493 non-push; corr Tuesday vs close total .947): decisive games (pick differs) tercile 153: served right 79, total right 74; median split 128: served 69, total 59. Raw log loss gain vs served: tercile -0.00069 [-0.0029,+0.0016] P+ .28; median -0.00155 P+ .03. LOSO add-one over calibrated served: tercile -0.00073 [-0.00140,-0.00007] P+ .013, delta coef + 1/6; median -0.00007 [-0.0019,+0.0018] P+ .48, coef + 0/6. Intercept-only LOSO log loss .6933 vs served raw .6955.
- Records: 5 cells family mkt27_total_lattice, unresolved_below_power.
## Tried
Tercile and median-split band conditioning (pool filtered to same total group, band widened to 200 games). Not tried: smooth kernel on total, key-number mass tilt by total, per-line reads.
## Next
Unit 2 option: pooled-residual width by total is MOD-45's tempo lane; if continued, a kernel-weighted (not hard-split) total weight keeps n near 200+ and cuts the dilution that likely drives the negative graded result. Not a closure: negative is not wrong_sign_resolved against a power-bounded control.
## Open
Served replication in the script re-derives the point as line + residual_at_open_served; mean abs diff 0.026 vs file column home_cover_probability_at_open_discrete_uncorrected, so the paired comparison is valid (same point) but absolute levels are not the served numbers. Pool totals are closing totals (no Tuesday totals before 2020).
