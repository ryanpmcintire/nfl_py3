# Coach familiarity: head coach facing a recent former employer (college first)

## Goal
Outside-leads candidate 3. Screen a declared college family on the XLG-14 add-one harness (unit 3 `read_candidate`, as unit 6), record every cell, declare (not run) one NFL confirmation per survivor. Zero crossing closes nothing; one fitted probability selects the side; LOSO; every cell a look; no magic numbers.

## Declaration (2026-11-16 15:57 EST, before coaching data was fetched or any outcome joined)
- Family `coach_familiarity_family_v1`, category market. Mechanism: a head coach facing a school he was head coach of recently knows its schemes, terminology and many of its players (he recruited them); the opener prices team strength, not that familiarity. Hypothesis: the familiar side covers more, POSITIVE.
- Novelty (**read**, grep of registry/weak_signals and ROADMAP for former.team, revenge, coach.famil, former.employer, coaching.tree): only team-level division revenge (NFL served member) and the CFB rivalry home-dog flag; no coach-to-former-employer term anywhere.
- Data: CFBD `/coaches` (head coaches by school-season; key from the environment, never printed), raw JSON under `data/external/coaches/` (ignored). School names map to CFBD team ids through `data/cfb/team_info`. CFBD has no coordinators, so the college family is head coach only; the coordinator version is NFL-only and needs a staff source (Open).
- Flag for team t in season s against opponent o: t's head coach in s was head coach of o in any season s-N..s-1, with N = 4 because college eligibility lasts four seasons, so after four seasons none of the players he coached there remain (redshirts aside). Known preseason. A school-season listing more than one head coach gets 0 (the game-time coach is not knowable preseason from season-level rows). Term = home flag minus away flag.
- Candidates: (1) `hc_former_employer_gap` as above. (2) `hc_former_employer_resid`: the same term residualised (OLS over all games, no outcome) on home-minus-away own ATS-to-date against the opener (mean over the team's earlier same-season games in the frame, 0 when none), the second look the u6 own-margin confound asks for. Both hypothesis POSITIVE; rule two-sided.
- Harness: unit 3 `read_candidate` (offset add-one per SD, no intercept, LOSO), 200 within-season shuffles, seed 36 stream 14 (new). Survivor: held-out log-loss gain above the 95th percentile of its own null AND equal coefficient sign in both halves. Looks: 2 gating, 6 recorded cells, 4 half coefficients, controls. Detectable (**read**, power lane): .08-.11 per SD four times in five; a sparse flag will usually not show.

## State
Done 2026-11-16 16:00 EST. Script `scripts/coach_familiarity.py` (`fetch`, `screen`); raw CFBD JSON `data/external/coaches/coaches_<year>.json` 2000-25 (FBS head coaches only, 116-161 per year; `.gitignore *`); source entry `cfbd_coaches` in `config/source_policies.json`. Outputs `artifacts/coach_familiarity/` (screen.log, summary.json/txt, weak_signals_batch.json, record.log, pool.log).
Build (**measured**, `screen` exit 0): 3,399 coach-school-season rows, 0 unmatched schools; 515 of 3,066 single-coach team-seasons have a former employer within 4 seasons, but only **22 of 8,925** frame games are flagged (home 9, away 13, never both; 0-4 per season). r(gap, own ATS to date) .007. As-of check on ATS-to-date: 5 cutoffs, 512 team-games, max diff 0.0.
Screen (**measured**; 13 seasons; null ll 5-95 about [-.00017,+.00011]):

| Candidate (+) | Nonzero; changed; record | Acc pts [95%] P+ (pct) | LL gain [95%] P+ (pct) | Gap | Coef/SD; folds; halves |
|---|---|---|---|---|---|
| hc_former_employer_gap | 22; 12; 7-5 | +0.02 [-0.05,+0.10] .72 (65.8) | -.00008 [-.00031,+.00014] .24 (55.0) | +.00012 | +.019; 13/13; +.065 / -.008 |
| hc_former_employer_resid | 22 effective; 16; 9-7 | +0.02 [-0.07,+0.11] .67 (68.2) | -.00008 [-.00031,+.00012] .22 (50.5) | +.00012 | +.019; 13/13; +.064 / -.008 |

Controls: base 50.48% / .69400; constant 50.72% / .69311; base+intercept 50.40% / .69403. No survivor (ll percentile 55 / 50.5, halves disagree). Residualising changes nothing (r .007), so the own-margin confound is absent here. **Inferred**: with 22 flagged games the college screen cannot see any plausible effect; it is a non-test, not a negative. Nothing closed.
Recorded: 6 cells `coach_familiarity_cfb_<candidate>_<units>` (cfb, market, unresolved_below_power) in `registry/weak_signals/coach_familiarity_family_v1.json`; record exit 0 (6); `weak-signals pool --league nfl --effect-units accuracy_points` exit 0.
NFL confirmation: none declared (no survivor, per the survivor rule).

## Tried
- Head-coach former-employer flag, N=4, college, as declared; one screen; no re-cut.

## Next
- The college flag is too sparse; any further look is a NEW declared family, not a re-cut. Candidate: NFL-native family using nflverse schedule `home_coach`/`away_coach` plus coordinator histories (staff source needed), where coaches move between division rivals; declare before joining outcomes, MOD-36 opener frame.

## Open
- Coordinator familiarity has no structured source on disk or in CFBD.
- `reliability` unrecorded on the 6 cells; nothing committed.
