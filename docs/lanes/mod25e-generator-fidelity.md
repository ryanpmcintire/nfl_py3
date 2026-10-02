# mod25e-generator-fidelity

## Goal
Keep improving the MOD-25 generator (owner 2026-10-02). Gate failures to
attack: margin SD (band 13.97-14.87), non-strength variance (real 178),
late-season R2 (band .132-.166), held-out shape (mass 3 real .144). Parent
docs/lanes/mod25-synthetic-data.md; prior docs/lanes/mod25d-variance-source.md.
Rules: nothing targets mass at 3/7/10/14/17; aggregates (pts/g, margin SD,
variance) are checks only. Owner 2026-10-02: NO compensating constants; every
parameter is a named mechanism fitted to its own real behaviour.

## State (measured 2026-10-02, gate @ .9 held out 2018-25)
| variant | SD | nonstr var | r2 w10-18 | mass3 | pts/g |
|---|---|---|---|---|---|
| crk06 = no bias + 1-yd dist matching (E1) | 15.41 | 213 | .122 | .114 | 41.9 |
| crf4 = crk06 + 4th-down bug fix (E4) | 15.16 | 205 | .111 | .118 | 42.5 |
| crf4m = crf4 + gain 6.5 + scale 1.0 (E7, 3 seeds @1.0) | 16.16 | 216 | .197 | .108 | 42.5 |
| crf4 3 seeds (E5) | 15.16 +-.12 | 207 +-2.4 | .112 +-.005 | n/a | 42.0 |
Code: scripts/mod25d_variance.py (DV variants, gate, downdiag), mod25e_budget.py
(variance budget), mod25e_scorestate.py (css/css2), mod25e_deficit.py.
Artifacts: artifacts/mod25d/gate_e1*, gate_e4a.json, downdiag_e1*/e4a;
artifacts/mod25e/budget.txt, deficit_e4*; artifacts/mod25e3/.

## Tried (measured)
- E2 budget (crj vs real): margin var 250 vs 205; strength 56 vs 52 (not the
  excess); noise 194 vs 152. Excess = cross-quarter covariance +16 vs -12
  (real leads mean-revert; Q4 slope -.073 real vs -.030) and drives/g 22.6 vs
  21.3 (~13). Per-drive scoring variance matches.
- E1: yard_bias .75 (MOD-25a/b hack) removed; dist cache rounding 2 yd ->
  1 yd plus distance kernel: 3rd&1 run share .66 (real .72), conv .61 (.66).
  Residual: 3rd 10+ share .232 vs .19; 3rd<=1 share .073 vs .11.
- E3: play choice, tempo, 4th-go by lead already match. Gap is drive outcomes:
  leader TD .269 vs .230 (Q3), punts .352 vs .405, trailer 2-min drives 3.0 vs
  4.1 plays. css2 halves cross-quarter cov (+6.8) but gate not improved; one
  sim stream only.
- E4: engine bug turned drawn 4th-down successes into turnovers on downs when
  sampled gain < sim distance (0.93 spurious/g; 4th conv .416 vs .487). Fixed
  in crf4. Field position, ypp, FG, turnovers match. Remaining pts gap 42.5 vs
  ~44-46: first-down move-on 0.7-1.3 pts low, 20+ yd share 5.25 vs 5.62,
  drives/g +0.8.

## E5-E7b (merge, knobs, gain; all measured, details in artifacts/mod25e3, artifacts/mod25e)
- E5 cf4s merge (3 seeds): xq cov improves (-12 real, +7.5 crf4, +4.1 cf4s); strength RE and r2 worsen (>2 sd); margin var, noise, Q4 slope, drives unchanged. Policy alone does not fix drives/g (+2.5) or Q4 slope.
- E6 knob audit (mod25e_knobs.py): real targets yard_gain 6.5 yd/EPA, def_sign +1, pace .04; strength scale has no real target (var_mu 1.0); structural knobs (inner/outer/stime, K 40, h .5, dkern .6) have no single target.
- E7/E7b: gain 6.5 with scale 1.0 double counts (margin SD 16.2, r2 .197). Derived on mechanism targets only: yard_gain 0, scale 1.0 (kernel carries strength). 3 seeds: SD 15.2-15.4, r2 .094-.131, mass3 .117, same as crf4.

## E9 (late R2 gap; mod25e_r2gap.py, artifacts/mod25e3/r2gap)
- Hindsight LOO season-strength R2 weeks 10-18: real .201 (CI .177-.228) vs sim .136; sim latent R2 .223 flat. Feature and drift are not the cause. Late gap .063: ~30% noise excess, ~16% strength spread, ~55% EPA-to-margin slope on persistent strength 35.9 vs 32.6 (unresolved).
- Marginal slopes: LOO-season-net EPA/play slope 35.9 real vs 32.6 sim; game EPA/play slope 22.9 vs 24.2; ratio 1.57 vs 1.35.

## E10 (drive-class gain; mod25e_gain.py, artifacts/mod25e3/gain)
- Joint regression of margin on P (season-mean net EPA incl. the game, total EPA units) and T (game minus P), all weeks, real 2018-25 vs sim: total P/T .78/.77 real vs .81/.80 sim, ratio 1.011 vs 1.005. E10 flagged garbage time (+.04), post-turnover (+.02), red-zone TD (-.05/-.04) and transient var +15%, but compared against 2018-25 while the sim pool is 2009-17.

## E11 (era check + RZ fp resolution; mod25e_rz.py comp/search/e5/rzval, artifacts/mod25e3/rz, e5_crzhr_s11)
- Reconcile: E9 ratio is a ratio of marginal slopes on a LOO season mean (noisy, attenuated) vs game EPA; E10 is a joint P,T fit on in-sample season mean; the joint fit shows no conversion-gain gap, so the 1.57 vs 1.35 reflects regressor noise, not engine gain (inferred).
- Pool era 2009-17 (measured, 2304 real games, boot over seasons): garbage time +.016/+.005 (pp .87/.65), post-turnover P -.007 T +.011 (pp .22/.94), RZ-TD P -.028 [-.062,+.008] pp .06, T +.001; transient var ratio 1.153. 2018-25: RZ-TD -.046/-.036 (pp .01/.00), garbage +.04 earlier. So most RZ-TD T gap and garbage gap are era drift; P RZ finishing and +15% transient variance remain.
- Drive level: real RZ TD rate .555 (2009-17) vs .592 (2018-25); sim crzhk .521 (se ~.004): -3.4 pp vs pool era unresolved.
- Mechanism tried: RZ fp rounding (cache bin 5 yd). Held-out likelihood grid {5,3,2,1}: val J 3.0/3.0023/3.0004/2.9996, chose 1 (exact); test vs 5: joint-key LL +.0105 pp .95, yards CRPS +.0045 pp .99, pass LL -.0009 pp .04. SIM_FAST byte-equal (1 world x 3 seasons). e5 s11 crzhr vs crzhk: RZ TD|RZ .517 vs .521, margin SD 15.07 vs 15.08, r2 w10-18 .097 vs .107, strength RE 45.1 vs 47.9: no effect, not adopted as a fix.
- Next: RZ gap is not field-position resolution; test RZ neighbour weighting by team strength (h_rz by held-out RZ likelihood) and goal-to-go distance/4th-down RZ go rate; transient variance +15% target.

## E12 (residual decomposition; scripts/mod25e_resid.py, artifacts/mod25e3/resid/resid.json, run.log)
- Measured: margin ~ a+b*game EPA diff (run/pass EPA). Resid var pool 2009-17 real 28.4 vs sim 26.4 (d -2.0 [-4.2,-0.02], pp .03); 2018-25 real 22.3 vs sim 26.4 (d +4.1 [+2.6,+6.1]). The E9 "30.5 vs 24.7" gap is an era artifact: the sim matches 2009-17 or runs lower, so no residual excess in the pool era under this regressor.
- Components sum exactly (check 1e-14). 2009-17 sim-real cov: td6 -1.45 [-3.1,+.2], fg_luck -.17, pat -.16, ret_kick -.18 (sim has none; pp 0), ret_fum var +.48 pp .95, ret_punt var +.31 pp .89, ot -.17; 2018-25 excess sits in return TDs (int var +2.4, punt cov +1.3, fum var +1.3, all pp 1.00) and fg_exp cov +1.8.
- EPA credit (read of chain2_epa vs real): TD plays 2.65 sim vs 2.61 real, turnover plays -4.57 vs -4.52: EPA is credited the same way; no scoring/special-teams decoupling found.
- Mechanism: none carries a pool-era excess; sim lacks kickoffs (no kick-return TDs) and over-produces defensive/return TDs vs 2018-25.
- Next: re-point E9 residual to same regressor/era; transient +15% is not in points-vs-EPA, look at variance of E itself.

## E13 (dual-era; scripts/mod25e_era.py, artifacts/mod25e3/era/era.txt, era.json)
- Gate and budget now carry a pool-era target (real 2011-17 gate, 2009-17 budget; held-out 2018-25 unchanged), re-scored from existing e5.json, no sim rerun. crzhk 3 seeds vs pool: SD 15.13 vs 14.63 (+.50), nonstr 208 vs 189, r2 w10-18 .104 vs .146, w5-9 .092 vs .116, mass3 .120 vs .141 (m14 .032 vs .048), strength RE 47.6 vs 58.0, noise 181 vs 165, xq cov +4.1 vs -6.4, q4 slope -.040 vs -.060, drives .14 over. crf4/crzk/u4g/f2 same shape; crI margin var 221 (matches) but pts 39.7 and strength 45.5.
- Remaining vs pool era: strength RE -10, noise +16, late r2 -.04, xq cov +10, mass3/14 low. Margin var and drives/g are matched, so those two were era artifacts.
- Era drift 2009-17 to 2018-25 (measured): SD 14.63 to 14.31, strength 58.0 to 52.3, noise 165 to 152, pts/g 45.2 to 45.8, r2 w1-4 .052 to .105, w10-18 .146 to .150, xq -6.4 to -12.0, drives 22.7 to 21.3, mass3 .141 to .144. Play side: 4th-go .124 to .191 (p 1.00), RZ drive TD .569 to .605 (p 1.00), penalty/scrimmage play and pass/explosive rates flat (CI spans 0), scrimmage plays/g -1.7.
- Design, no build: production pool = all seasons but the held-out one (leave-season-out), era-weighted by held-out play likelihood (weight or rolling N chosen by that likelihood, not by aggregates); conditioning on season for 4th-go and RZ finishing so era-drifting behaviours are not averaged. Gate target set always matches the pool. Frozen MOD-25 distilled challenger was trained on the biased crj world (yard_bias, 4th-down bug), must be rebuilt from the fixed generator on the production pool; its prior results are void.

## E14 (channels; scripts/mod25e_channels.py, artifacts/mod25e3/channels/channels.txt, run.log; crzhk s11 sim, 6528 games)
- Measured, split-half true team-season margin variance V_M (season bootstrap): real 2009-17 28.8 [24.9,32.4], 2018-25 26.6, sim 25.5 [22.2,28.9] (sim<real pp .89). Scrimmage EPA net explains 28.4 of 28.8 real (persistent residual .4 [-.2,.9], 1.4%), sim 25.5 of 25.5: no persistent margin component outside scrimmage EPA in either, so the hypothesis is not supported.
- Channels beyond EPA are tiny and unresolved: real ST EPA incr 0.65 [.22,1.08] pt2 (punt .77, kick .53), field-position start .27, fg_exp .15, punt net .26; each rel .2-.5. Sim lacks persistence: punt net rel -.01 vs .33 (pp .00), return TD pts rel .05 vs .30 (pp .01), fp start true var 1.9 vs 5.0 yd2 (pp .00), fg_exp var .19 vs .40 (pp .04); sim kickoff/ST EPA absent.
- Likely carrier of the strength gap: scrim net true var 35.1 vs 37.6 (pp .33) with off/def variances matched (24.3/8.1 vs 24.0/7.8); true off-def correlation sim .095 [-.03,.24] vs real .213 [-.00,.39] (2018-25 .231 [.05,.44]); pp .16. Inferred: generator draws off and def latents ~independent; real are positively correlated.
- Design (no build): fit a joint (off,def) team-season latent covariance from real play kernels (cross-fit EPA effects), plus a team ST latent (punt/kick net, return, kicker FG over expected) with real reliabilities and spreads; ST value is ~1-2 pt2 of 28.8, so expect <=1 of the 10 RE gap.

## E15 (joint off/def latent; scripts/mod25e_cov.py fit|sim|e5|simfeat|channels, artifacts/mod25e3/cov, era_cov/era.txt)
- Read: generator drew levels, carry and weekly drift with ONE shared corr (season-mean corr of off, dfn, -.124 raw, noise-diluted); fixed here by cross-lag covariance (lags 1-6, 2009-17, season bootstrap): corr -.228 in dfn units = +.228 in scrim_def orientation [-.49,-.009] pp neg .98, matches split-half .213.
- Variant crzhc = crzhk + joint cov (levels, carry, stationary drift cov); SIM_FAST byte-equal (1 world x 1 season). 4x8 sim true rho .304 [.19,.43] (old .095), net var 40.7 (real 37.6, old 35.1), V_M 27.9 vs 28.8.
- e5 2 seeds vs pool 2009-17 (crzhk 3 seeds): strength RE 55.0 vs 47.6 (real 58.0; held 52.3), noise 178 vs 181 (165), late r2 .127 vs .104 (.146), xq cov +8.2 vs +4.1 (-6.4; held -12.0), SD 15.26 vs 15.13, mass3 .117 vs .120.
- Not fixed: noise +13 and xq cov +14 vs pool; rho overshoots slightly; lagged season-mean cov model/obs .56/.39 (carry untested).
