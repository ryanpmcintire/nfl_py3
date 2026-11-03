# SIM-18

## Goal
Lead-protection clock burn: run rate and seconds per snap when leading in the fourth quarter, per coach, and the drive count it removes (ROADMAP SIM-18).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: Leading teams burn clock, shrinking the number of remaining possessions and capping the trailing team; drive count is the mechanism that changes the margin distribution. Enters SIM-10 as a team-conditioned remaining-drive count; the pregame side gets a favorite-margin term in the line-conditional distribution.

## Tried
Novelty check only. Nearest: MOD-28 (margin compression trait), SIM-09 Tier B, SIM-10; grep 'prevent' found only a capture note. Differs: Clock and drive-count mechanism, not a margin-level trait.

## Next
First unit: Measure remaining drives and seconds per snap by lead size and coach, reliability with permutation null, compare with MOD-28 compression on the same games.

## Open
Data: Local play-by-play (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.

## Unit 1 result (2026-11-03, measurement only; script scripts/sim18_lead_protection.py, output artifacts/sim18/unit1.json)
Data: nflverse REG 2009-2025 2nd half, 282,218 run/pass/kneel plays, 50,280 drives. Bins declared: time 5-min (0-1800 s left), lead 1-8 / 9-16 / 17+ (one score, two score, more); LOSO degree 0-4 polynomial in lead per time bin. Looks: 37 curve cells, 12 LOSO fits, 8 reliability tests, 36 drive cells = 93. Not recorded in weak-signals: no admissible effect unit (descriptive curve and reliabilities).
**Real curve (measured).** Leader sec/play (lead 1-8, by time bin 1500-1800 down to 0-300): 33.9, 33.8, 33.4, 34.7, 35.8, 22.6; lead 17+: 34.4, 35.0, 35.4, 37.7, 39.4, 33.3. Trailer lead 17+: 31.8, 30.5, 28.2, 26.8, 24.6, 18.4. Run share leader 1-8: .47 .45 .45 .48 .51 .57; 17+: .50 .51 .55 .60 .72 .62 (last bin includes kneels, 20-27% of snaps). LOSO degree 1-4 (mostly 1-3); slope in lead ~+0.1 s/play per point at 300-900 s. Mean 30.5 s/play tied.
**Real remaining drives (leader/trailer, start of drive, t 600-900 s left):** lead 1-8 2.95/2.85; 17+ 2.65/2.55; leader drives last 191 s vs trailer 187 (1-8), 200 vs 156 (17+): lead protection acts on drive length (+44 s), not drive count.
**Persistence (HC spell | OC spell), residual after LOSO league cell means, split-half SB / season t vs t+1 r (sign-flip null q95):** sec/play .70 / .47 (null .33/.09) and .67 / .51 (.29/.12); run share .80 / .45 and .76 / .44; all perm p = 0/2000. Caveat: residuals not adjusted for team quality, so talent-driven lead-keeping is inside the trait (inferred).
**Simulator gap (measured vs artifacts/sim10/instrument/instr_s11, 3 workers, seed 11; no dflt run logs plays).** sim10_drive.py never simulates plays: it resamples a historical drive j with its real elapsed time el[j] (lines 1366-1440, el defined 225-233), so seconds/play, run share and play clock do not exist in the engine and cannot be tabulated. At drive level the sim reproduces the lead effect: leader-ball drive length 17+ at 600-900 s is 195 vs real 200, trailer 152 vs 156; remaining drives within +0.15 of real in every cell (e.g. 1-8 at 1500-1800: 5.70/5.70 vs 5.55/5.48; sim symmetry leader=trailer where real leader is +0.07-0.2). Gap = sim leader drives 3-4% short in 1-8/9-16 bins (176 vs 191, 180 vs 192 at 600-900 s) and sim leader-minus-trailer remaining drives ~0 vs real +0.1-0.2 early; no per-team trait.
**Named mechanism to close it (not implemented; Tier B chain on sim10_drive.py).** Coordinator-keyed (OC spell) tempo trait: per-OC lead-state clock multiplier on el[j] (and a matching run-share term) estimated as the LOSO shrunken OC residual of sec/play (reliability .67-.76 above) in lead>0 states, applied in Engine drive loop at `secs -= self.el[j]` (line ~1450) with the same lead-state key as the draw; because drive count follows from sum of el, remaining drives change through el, no separate drive-count knob. Pair with halftime-adjustment OC key (SIM-17). Grade by gates (q4_slope) and m3/final margin compression, not by tempo fit.
Next: unit 2 = fit shrunken OC multiplier LOSO, test it predicts held-out leader drive length and final-margin compression (MOD-28) on the same games.

## Unit 2 result (2026-11-03 ~22:00 UTC; measurement only; scripts/sim18_unit2_multiplier.py, artifacts/sim18/unit2.json)
Looks ~35 (8 reliability, 16 multiplier, 3 pick arms, 6 compression). Recorded 5 unresolved cells (family sim18_lead_protection_unit2).
- Quality adjustment (LOSO OLS on prior-game off/def EPA ratings, own and opponent): R2 of sec/play residual .003, run share .024. Reliability survives: OC sec/play split-half .692 vs raw .689, season r .508 vs .506; HC .697/.471 vs .693/.466; run share OC .787/.380 vs .807/.441 (86-97% survives). Trait is not team quality.
- Held-out multiplier (OC spell, drive-length ratio vs LOSO cell mean): all leads, shrunk, other seasons: slope .43 [.23,.64], corr .183 [.103,.262], but drive MSE gain vs constant -0.0003 (P+ .27); lead 1-8 slope .62 [.32,.95], corr .178, MSE gain +.0006 (P+ .79); leads 9-16 and 17+ no signal (K 520-580 drives, corr .00-.03). Prior-season-only: slope -.05, corr -.02 (half-spell reliability .2 too low). Unshrunk worse than constant.
- Pick term (2020-25 opener, 1503 games): served 800-703, LL .69216; +run_adj 814-689, dLL +.00009 [-.0026,.0026] P+ .53, flips 318 W166-L152, coef negative all 6 folds (-.06 to -.14); +sec_adj 770-733, dLL -.0011 P+ .03, flips 140 W55-L85; +sec_raw dLL -.0011. Intercept LL .69328.
- MOD-28 compression (leader-signed final minus Q4-start lead, leader-minus-trailer sec trait): 2012-25 n 3455 corr -.066, slope -.51 pt/sec [-.77,-.24]; 2020-25 corr -.027 P+ .15. Slower-pace leaders lose slightly more lead (inferred: opposite of a protection mechanism).
Next: unit 3 = engine-side lead-1-8 OC multiplier prototype after Tier B chain frees sim10_drive.py; grade by q4_slope/m3, not tempo fit.
