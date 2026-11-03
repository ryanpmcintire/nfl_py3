# MOD-10 graph model: player-state reliability

## Goal
Row MOD-10 (ROADMAP:691): "Player/team matchup graph only after player state is reliable". No numeric DoD in the row; gate is player-state reliability.

## State
- Source of .27-.44 (read): docs/latent_ratings_on_production.md 8.2 (RAPM offense SB +0.296, defense +0.268, odd v even seasons, >=500 plays both halves), docs/st_player_ratings.md (ST SB 0.436), docs/unit_apm_ratings.md (0.236-0.394). Nonzero, so no_split_half_reliability closing ground unavailable.
- Prior lever already tested (read): hierarchical unit shrinkage k=500, docs/hierarchical_apm.md. Not repeated.
- Named cause (inferred): per-player coefficient is one of 22 collinear terms on a noisy EPA label; team feature is team-wide across 5 seasons, so team-season quality drift is absorbed by player terms on the same roster.
- Unit run: scripts/mod10_player_reliability_teamseason.py (team-season dummies, alpha grid {1000,3000}); output artifacts/mod10_graph_reliability/result.json.

## Tried
Measured (artifacts/mod10_graph_reliability/result.json, same 2016-2025 odd v even gate, 760 off / 808 def players, 4 looks): baseline reproduces SB .296/.268. Team-season dummies: .158 [.029,.267] / .168; alpha 3000 raises to .396 [.299,.478] / .365 [.263,.450] (still no team-season, alpha chosen on the same halves so in-sample, gap unmeasured). Team-season + alpha 3000: .193/.195. Reading (inferred): about half of RAPM reliability is team-season quality, not player-specific; shrinkage lifts it to the ST level (.436) but not past.

## Update 2026-11-03 (unit 2, measured: artifacts/mod10_graph_reliability/lopo_wide.json, scripts/mod10_player_reliability_lopo.py)
Repair: P+ recomputed from 2000 bootstrap draws: team-season a1000 0.992 (1984/2000), alpha3000 1.000; both re-recorded.
Leave-one-pair-out: alpha chosen on one disjoint split (S1 2016-19, S2 2022-25; 2020-21 unused, declared), scored on the other, 2-season halves, 7-alpha grid x 3 designs = 21 looks. Selected alpha was the grid top in every arm (30000 on the 5-grid, 300000 on the 7-grid), so in-sample best = selected, gap 0, optimum not bracketed.
OOS SB (mean of two directions) offense/defense: plain .552/.376; team-season dummies .553/.351; per-season fits snap-weighted .588/.393 (P+ 1.0, bounds ~.44-.68 offense). On the 5-grid (max 30000): plain .277/.232, per-season .399/.149.
Reading (inferred): at alpha >= 100x default the coefficient is near an on-field EPA sum; team-season dummies are shrunk by the same alpha so they stop removing team quality, hence ts arm == plain. So the offense .55-.59 is NOT shown player-specific. Gate .436 cleared by offense only, defense not, and not shown net of team quality.

## Next
Not cleared; no graph-term spec yet. Next cause: team quality is penalised with the player terms. Unit: residualise EPA on unpenalised team-season (and opponent) means, then ridge on players with the alpha grid extended past the optimum (bracket it), same S1/S2 cross-selection, snap-weighted per-season state. Spec the graph term only if offense and defense OOS both exceed .436 net of that.

## Open
Row has no numeric gate; PER-09 .436 used as gate. Defense stays below it.

## Update 2026-11-03 (unit 3, measured: artifacts/mod10_graph_reliability/resid.json, scripts/mod10_player_reliability_resid.py)
Design: EPA residualised on unpenalised offense+defense team-season means (lsqr OLS per fit set), player-only ridge on residual, 11-alpha grid (0.01x-1000x default, half-decade), splits S1 2016-19, S2 2020-23, S3 2024-25 (2020-21 now in: participation coverage is full, 11 players on every play all seasons), alpha chosen on the other two splits, 2 designs x 11 x 3 = 66 looks, 2000 draws.
OOS SB offense/defense, per-season snap-weighted: S1 -.05/-.32; S2 .12 [-.10,.30]/.15 [-.06,.32] (P+ .87/.92); S3 .09/-.17. Pooled-halves: S1 -.03/-.19, S2 .09/.09, S3 .09/-.17. P(SB>.436) = 0.000 in every cell but one (.001). Selected alpha landed at grid index 0-1 (bottom) in four of six arms, 5-7 in the rest: unstable, not a clean interior optimum; in-sample gap up to +.36.
Reading (inferred): the .55-.59 earlier was team-season quality; net of it, player state is near zero to ~.1, far below .436. Recorded in family mod10_player_state_reliability (offense, defense, S2 per-season, unresolved_below_power). Not refuted (S2 P+ .87-.92 positive).
Gate NOT cleared; no graph-term spec. Next cause (inferred): within-team-season identification is thin (rotation only), 2-season halves too short for ~40 snaps-per-game labels; test with longer halves (full 4-season interleaved) or position-group aggregation, or lineup-level not player-level state.

## Update 2026-11-03 (unit 4, FTN player traits; in progress)
Script scripts/mod10_ftn_player_traits.py (ruff clean) written, not yet verified by output. FTN has no player ids: joined to pbp snapshot data/pbp/raw/20261027T220410Z (receiver_player_id, passer_player_id) on nflverse_game_id/play_id, REG season 2022-25. Declared before looking: receiver min per half (odd v even weeks) 15 catchable targets for drop rate, 25 for created/contested (per catchable target); QB min 100 dropbacks per half; team context removed by leave-player-out team-season-half rate (receivers only; QB cannot be separated from team); stability = season t v t+1, min 2x half volume, raw rate; 2000 bootstrap draws by player; 9 trait variants x 6 cells = 54 looks.
Launched in background (task bj7o93nbh, started from the prior agent); writes artifacts/mod10_ftn_traits/result.json on completion. If absent, rerun once: uv run --no-sync python scripts/mod10_ftn_player_traits.py.
Next: read result.json, compare sb_pooled and stability to .436, record via weak-signals record --batch (family mod10_ftn_player_traits, effect-units correlation), write gate verdict here. Note split-half is within-season so not out of sample; cross-season stability is the out-of-sample read.

## Result unit 4 (measured: artifacts/mod10_ftn_traits/result.json; 9 variants recorded, family mod10_ftn_player_traits, unresolved_below_power)
Pooled 2022-25 within-season odd v even SB [95%] / cross-season stability t v t+1 (Spearman, raw): receiver drop rate .01/.09 (null); created reception .45 [.29,.57] (team-demeaned .50 [.36,.61]) / .53 [.39,.64]; contested catch .60 [.50,.69] (demeaned .65 [.54,.73]) / .42 [.25,.55]; QB catchable-ball rate .68 [.55,.76] / .35 [.05,.58]; QB interception-worthy .47 [.23,.65] / .24; QB fault sack .54 [.37,.67] / .04.
Gate .436: cleared in-season by contested (P>gate 1.0), QB catchable (.999), created reception demeaned (.82); cross-season stability clears only created reception (.53). Stability n 173 receivers / 77 QBs. Caveat: QB catchable rate partly measures receivers and scheme; QB team context not removable.
Verdict: FTN supplies reliable player-attributable state for receiver created-reception and contested-catch (and QB ball placement), but these are skills, not team strength; they are descriptive node state, unproven for the served probability. Graph term spec (not run): one fitted term = lineup-weighted receiver created-reception/contested shrunk rates (prior seasons only) in the served logit, leave-one-season-out, scored on the opener grade vs market and model-only baselines. MOD-10 stays gated on the team-level player-state test, which remains ~0 (unit 3).
