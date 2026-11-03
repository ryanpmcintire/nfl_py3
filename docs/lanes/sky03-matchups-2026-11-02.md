# Lane: SKY-03 matchup interactions

## Goal
Pair-level matchup effect beyond main effects, from pbp: (A) pass protection (pressure = sack or QB hit per dropback) x opposing pass rush; (B) receivers x coverage (EPA per non-sack dropback). One fitted term on the four-term probability, LOSO 2020-2025, opener.

## State
Done 2026-11-02. scripts/sky03_matchups.py; artifacts/sky03/ (reliability.json, fitted_term.json, record_batch.json, team_games.parquet cache). Recorded 6 cells, family sky03_matchups, unresolved_below_power.
Traits: EB-shrunk as-of team rates (current season prior weeks plus previous season), shrink constant from odd/even variance components of earlier seasons. Interaction c fit on all earlier games (expanding), x = expected dropbacks * (cA*zA + cB*zB*non-sack share), home minus away.
Reliability (odd v even weeks, 2016-2025, team-level interaction slopes, 32 teams): A offense r 0.07 [-0.27,0.40] p+ 0.66; A defense 0.15 [-0.25,0.54] p+ 0.78; B offense 0.34 [-0.08,0.64] p+ 0.95; B defense 0.27 [-0.11,0.57] p+ 0.93. Not zero.
Term (1,503 games): OOS Brier gain -0.00043 [-0.00088,+0.00003] p+ 0.03; log loss -0.00088; accuracy -0.67 pts [-1.53,+0.13]; decisive 38, term 14-24. In-sample +0.00002 p+ 0.56; gap +0.00045. Fold coefficients -0.081,-0.109,-0.030,+0.017,-0.033,+0.021 (unstable, mostly negative). c_A about -5, c_B about -1 EPA per unit product (mechanism sign correct for pooled c).

## Tried
Single summed term, two units. Not tried: separate A/B terms, play-level pressure/air-yards data (not in on-disk pbp subset), player-level OL/CB ratings (MOD-22/23 territory).

## Next
Split A and B as separate declared looks only if owner wants; richer pressure data (PFR/NGS pressure, coverage) would raise trait reliability.

## Open
ROADMAP SKY-03 row not updated (orchestrator). Reliability is not zero, so no closing ground applies.

## Update 2026-11-03
Planted positive control (scripts/sky03_positive_control.py, 200 reps/size, LOSO 2020-2025, 1,503 games): planted logit coefficient on x; detected = OOS Brier gain P+ >= 0.95. Coef 0.13 (4x observed |0.033|): 1% detected; 0.26: 13.5%; 0.526: 93%; 1.05: 100%. Null false-detect 0.5%. Control cannot bound the observed size (about 16x below detectable), so no positive_control_bound; sign of observed coef negative but interval spans zero, so no wrong_sign_resolved. Stays unresolved_below_power. Artifacts: artifacts/sky03/positive_control.json, positive_control_large.json. Re-recorded 2026-11-03 as cell sky03_positive_control_detectable_size in family sky03_positive_control, category control, effect 0.004455 brier_improvement (mean OOS gain at b0 0.526, 93 pct detection), no probability_positive; old cell removed from sky03_matchups.
Next: richer pressure/coverage trait source (earliest when PFR/NGS data is on disk); skill vs selection vs script separation still undone.

## Update 2026-11-03 PFR/NGS traits
Data: data/raw/pfr_advstats/20261103T125324Z (pass+def weekly 2018-2025, manifest.json); NGS from data/raw/ngs. Script scripts/sky03_pfr_traits.py, artifacts/sky03/pfr_traits_rel.json, pfr_record_batch.json.
Trait split-half r (odd v even weeks, team-season demeaned, 2018-2025, offense/defense): pbp pressure .62/.34; PFR pressure .49/.25; pbp EPA non-sack .49/.19; NGS separation .43/.32; NGS time to throw .75/.17. Interaction slope r (2020-2025): A pbp .02/-.07 -> PFR .24/-.16; B pbp .23/.40 -> NGS separation -.37/.15.
Verdict: no material rise (PFR pressure is below the pbp sack+hit rate; separation only helps defense side), so no refit. 18 reliability looks recorded, family sky03_trait_reliability. Not a closing ground.
Next: PFR def file (hurries/coverage per target) as defender-level traits, or pbp-based pressure with sack+hit+hurry; refit only if slope reliability clearly rises.

## Update 2026-11-03 defender traits and declared pair
Script scripts/sky03_def_traits.py, artifacts/sky03/def_traits/rel.json. pbp has no hurry column, so the combined rate is pbp sack+qb_hit plus PFR def-file hurries per dropback. PFR def file has no snap counts; coverage traits use targets as the denominator.
Trait split-half r (offense/defense, 2018-2025): sack+hit+hurry .54/.26; PFR def pressures .53/.26; yards per target allowed .44/.19; completions per target allowed .34/.28.
Interaction slope r (offense/defense, 2020-2025): A hit+hurry .-07/.04; A oP x PFR pressures -.13/.06; B oN x yards/target .01/.30; B oN x completions/target .10/.32.
DECLARED before any outcome look: B_oN_x_completions_per_target_allowed (highest min over both sides, .10/.32). Single interaction term, refit once, LOSO 2020-2025, opener.
Refit (one look, artifacts/sky03/def_traits/term.json, 1,503 games): OOS Brier gain -0.00029 [-0.00098,+0.00043] P+ 0.199; log loss -0.00057 P+ 0.212; accuracy -0.80 pts [-1.73,0.00]; decisive 42, term 15-27. In-sample +0.00007, gap +0.00036. Fold coefs -0.236,-0.106,-0.115,-0.007,-0.105,-0.103 (all negative, stable-ish). Model-only Brier 0.2448, market 0.25. Recorded: sky03_matchups cell sky03_def_cov_term_oos_brier, 16 reliability cells in sky03_trait_reliability. unresolved_below_power.
Positive control (scripts/sky03_def_positive_control.py, def_traits/positive_control.json, 200 reps): planted b 0.105 (1x) 0% detected; 0.21: 1.5%; 0.42: 23.5%; 0.84 (8x observed): 98.5%, mean gain 0.0047. Detectable size about 8x observed (was 16x).
Next: the all-negative coefficient hints at wrong sign; sign check on a larger sample before any wrong_sign_resolved.

## Update 2026-11-03 sign check
Served OOS opener probabilities exist only for 2020-2025 (per_game.parquet; no 2018-2019 anywhere in artifacts/), so no refit on 2018-2025. Instead pooled coefficient of the declared term, scripts/sky03_def_sign.py, artifacts/sky03/def_traits/sign.json: -0.105; season-block bootstrap (6 blocks, 4000 draws) [-0.460,+0.055], P(neg) 0.874; week-block [-0.383,+0.182] 0.754; game [-0.380,+0.182] 0.774. Interval includes zero, so wrong_sign_resolved not met. Recorded sky03_def_cov_coef_sign_season_block unresolved_below_power (units field "correlation" used as nearest to a coefficient; P+ 0.1265).
Next: closed only by wrong_sign_resolved on more seasons (2026 forward) or a control bound; no further looks now.

## Update 2026-11-03 gap audit (no new run)
Closing grounds checked: wrong_sign_resolved needs whole interval negative; sign.json season-block [-0.460,+0.055] P(neg) .874, week-block [-0.383,+0.182], so not met. positive_control_bound needs a control detecting the observed size; control detects ~8x (0.84 at 98.5%), observed 0.105, not met. Row stays unresolved_below_power; nothing recorded this round.
Remaining to the row goal (separate skill from opponent selection and game script): not done. Only sample-increasing paths are forward 2026 rows (frozen term artifacts/forward_terms/sky03/frozen_coefficients.json, no_2026_outcomes true) and 2018-2019 served OOS probabilities (absent). Next: after Week 9 lock, score the frozen term forward; separation unit (team-week random effects on pair EPA, scripts/sky03_matchup.py, 1 core) is the only unit that moves the stated goal.

## Update 2026-11-03 separation unit (declared before outcomes)
Script scripts/sky03_matchup.py (reuses sky03_matchups reliability/game_x/fitted_term on separated rows). Separation: as-of crossed offense and defense effects (alternating shrinkage, k from the lane's variance components) absorb opponent selection; game-level script (mean win probability on the team's dropbacks) absorbs script via a fitted slope; the pair residual is outcome minus mu minus off minus def minus script. Team-week random effects are not identified separately from the pair at one row per team-game, so game-level script plus crossed effects stand in; stated deviation.
Arms/looks (5): reliability of separated pair slope, odd v even weeks, 2016-2025, A off, A def, B off, B def (4); one summed term (A+B, as the prior unit), LOSO 2020-2025, opener, base per_game.parquet (1). Term fit only if any reliability r > 0. Family sky03_matchup_separated.
Result (artifacts/sky03/separated/{rel,term,record_batch}.json; 5 cells recorded, sky03_matchup_separated, unresolved_below_power): slope split-half r A off -0.04 [-0.37,0.27] P+ .39; A def 0.49 [0.11,0.75] .99; B off 0.61 [0.39,0.79] 1.0; B def 0.31 [0.01,0.54] .98 (prior unseparated .07/.15/.34/.27). Script slope gamma .64 (epa) .57 (epa_ns) per unit win prob.
Term (1,503 games, decisive first): decisive 82, term 38-44. OOS Brier gain +0.00019 [-0.00053,+0.00091] P+ .67; log loss +0.00038 P+ .68; accuracy -0.40 pts [-1.60,+0.80] P+ .24. In-sample +0.00027 P+ .75, gap +0.00008. Fold coefs .23,.20,.29,.27,.22,.33 (all positive, stable). Season OOS gains + in 4 of 6.
Remaining: separated pair is reliable and sign now stable, but the decision effect is still not resolved and accuracy does not improve; no positive control run at this size. Next: forward 2026 scoring of this term; planted control for detectable size.

## Update 2026-11-03 separated control and freeze (in flight)
Frozen: artifacts/forward_terms/sky03_separated/frozen_coefficients.json (cA -2.369, cB -0.882, b 0.258 logit/pt, fit through 2025, no_2026_outcomes true), made by scripts/sky03_separated_control.py --stage freeze.
Control running: --stage control, 200 reps, mult 0,1,2,4,8 of 0.258, log artifacts/sky03/separated/control.log, result positive_control.json. Not yet recorded (family sky03_positive_control, ids _separated).
forward_terms_week.py change needed (not made): add term "sky03_separated" to TERMS; sky03_week variant that builds the as-of rows with sky03_matchup.build_rows (needs per-game win-probability script column from 2026 pbp wp), computes zA=oP*dP, zB=oN*dN, pts=db_exp*(cA*zA+cB*zB*ns_share) with frozen cA,cB, x=sum of sign*pts per game, term b from frozen file.
Control result (200 reps, LOSO 2020-25, detect = OOS Brier P+ >= .95; null 0%): 1x (0.258) 2.0%, 2x 13.5%, 4x 88.5%, 8x 100%. Smallest size at power .8: 4x observed. Observed size not detectable, so no positive_control_bound; sign positive and stable, no wrong_sign_resolved. Stays unresolved_below_power. 4 cells recorded, family sky03_positive_control, ids sky03_positive_control_detect_{1,2,4,8}x_separated. Freeze file written (see above). Remaining: forward_terms_week.py change by orchestrator; 2026 forward scoring.

## Update 2026-11-03 forward_terms_week sky03_separated term (done)
scripts/forward_terms_week.py now has term sky03_separated (sky03_separated_frozen verifies coefficient_hash, no_2026_outcomes; sky03_separated_week builds as-of rows via sky03_matchup.build_rows with 2026 prior-game mean wp column, week-W rows as NaN-outcome placeholders; pts=db_exp*(cA*zA+cB*zB*ns_share), sign-summed, times b in combine). Added --dry (no write, no kickoff filter, schedule fallback card with p=0.5 when the served card is absent). Dry Week 9 (pbp through week 7): 15 games, mean 0.0486, sd 0.3342. Other six terms' row hashes identical before/after (tests/scratch/ftw_before.txt, ftw_after.txt). ruff clean.
Next: after lock root runs: .venv/Scripts/python.exe scripts/forward_terms_week.py --week 9 --term sky03_separated --no-fetch ; then score forward after settle.
