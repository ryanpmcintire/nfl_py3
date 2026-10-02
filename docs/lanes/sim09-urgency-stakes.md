# sim09-urgency-stakes

## Goal
Model situational human behaviour in the generator (owner 2026-10-02): urgency
when trailing (especially behind at the half), conservatism when leading,
late-season stakes (playoff race, eliminated, clinched, rest). Effect size
varies by coach and players. ROADMAP SIM-09. Generator lane:
docs/lanes/mod25e-generator-fidelity.md. Rules: AGENTS.md "No magic numbers
anywhere": every parameter is fitted to real behaviour; margin shape,
margin SD and R-squared are held-out checks only.

## State
2026-10-02 opened. Known from E3: run/pass share, tempo and 4th-down go by
lead x time match real already; the gap is drive outcomes (Q3 leader TD
.269 vs .230, trailer final-2-min drives 3.0 vs 4.1 plays). So urgency must
be measured in what changes inside a play type: depth, risk, defence.
Data: full nflverse pbp 2009-17 (372 cols: air_yards, pass_length, shotgun,
no_huddle, xpass, home_coach/away_coach) at
C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad/nv/;
schedules have home_coach/away_coach.
U1, U2 done 2026-10-02; U3 next.

## Tried
U1 (scripts/sim09_urgency.py, artifacts/sim09/*.csv; 288,770 plays 2009-17, z=lead/sqrt(min/60+1), 181 looks): leader Q3 lead 7: shotgun -3.5pp, air -0.22, plays/drive -0.19, TD/drive -.011. Trailer final 2 min trail 7: shotgun +15pp, no-huddle +6pp, INT +0.5pp, drive yds/play +.18. Halftime per 10 behind: drive TD +.030. Coach slope SD real but modest (shotgun .0102 rel .62, split-half .33). Weeks 13-18 vs 1-12: no clear difference.
U2 (scripts/sim09_stakes.py, stakes_*): standings MC rebuild (field overlap 93.5%). 120 looks, effects small: leverage x trail no-huddle -.022 (p+ .003), INT -.005; eliminated 4th-down go +2.2pp; game level wk10+ stake-high side beats expectation ~+1 pt (held +0.8, p+ .78), unstable.

B1/C1/D3 (scripts/sim09_dynamics.py, artifacts/sim09/dyn_*; fit 2009-17, held 2018-25; B2 fit 2016-20 held 2021-25; 61 looks counted across both eras: b1 21, b2 12, c1 4, d3 24; epa/success/ypp offence-side, game-cluster OLS, strength+z+down/dist/field+clock controls).
B1: cum defensive snaps beyond clock +0.086 epa/100 snaps (p+ .96) held +0.052 (.83); prior drive length +0.015 epa/10 plays (.96, held .99), ypp +.18/+.15; no-huddle faced +.034 epa (.98, 2009-17 only); drive position -0.02/-0.03 (p+ .07/.01). Small, same sign both eras.
B2: starter leaves: defence epa allowed -.034 (p+ .04) held 0.000 (.51); offence non-QB -.020/-.021; QB exit n=53/75 unresolved (+.015, -.045). No fatigue-style drop.
C1: staff 2H-1H adjustment tau .019/.026 epa, reliability .10/.18, split-half r off .04 def -.18, held-out slope on shrunk -.39 (se .80): no staff skill detected, unresolved below power (34 coaches).
D3: after own score +.016/+.033 epa (p+ .99/1.0); after 4th-down stop -.046/-.022 (p+ .002/.05); turnover, missed FG, own 20+ play, allowed score ~0. Momentum ~0 to slightly positive after scoring.

F1 (scripts/sim09_units.py; data/processed/sim09_unit_ratings.parquet; artifacts/sim09/units_*). 22 metrics, rolling 3-season ridge off+def per team-week, pregame asserted, hl/tau tuned 2017-21.
Split-half SB off/def (carryover off): epa_all .76/.54 (.42); cpoe .68/.42; qbrun .87 (.58); press_net .73/.52; sack_r .66/.25; int .25; zone off .61 def .31; man off .39 def 0; deep .2; run_net .48/.38; run_light/stack, fg ~0; sep .62/.35; yac .47/.16.
Cov: pass off units load on one factor (epa_pass-zone .90, man .80, cpoe .74); run_light-run_net .73.
Held-out 2022-25 incremental (36 looks): all |R2 gain|<.008; run_net +.0073 on run EPA (p+ .94); ALL-ridge negative. Interactions (24 looks) unresolved; yac x yac allowed +.006 (p+ .94).
A (scripts/sim09_personnel.py, artifacts/sim09/pers_*; terms declared in pers_declared.json before signs; fit 2016-21 held 2022-25; NH fit 2013-15 held 2016-17; shell terms 2018-21 / 2022-24; 77 looks, 4 families; cluster-by-game LPM/OLS, ratings std per SD, EB shrink by family tau2).
Family tau2 (std units) ~0: A1 0, A3 3e-7, A4 4e-4, A5 3e-5: almost all interaction variance is noise.
A1 rookie x two-minute EPA +.062/+.076 (p+ .87/.88 fit/held, same sign); QB experience x no-huddle/two-minute sack, INT, pre-snap penalty: signs flip, none replicate (NH pspen x exp +.004/+.0004).
A3 OL x rush in obvious passing: none replicate (OPD3 PP x pressure -.007 fit, +.010 held; TRL PP x EPA -.016/-.037 but PPxRUSH EPA +.021/-.015). Unresolved below power.
A4 deep x two-high: comp -3.6pp/-6.1pp (p+ .004/.013), INT +2.2pp/+1.6pp (p+ .999/.89), EPA +.139/+.054; deep x pass-offence EPA +.062/+.030 (p+ .997/.71); WR separation x deep flips sign in held (EPA +.035 vs -.070).
A5 leader runs: box/run-block x lead flips sign on success, EPA, clock; fd x box +.003/+.010 only same-sign. Unresolved below power; no closure.
Yes: only A4 shell and A4 pass-offence carry usable numbers. No: matchup-conditioned weighting for A1/A3/A5 supported.
S1 (scripts/sim09_shell.py; artifacts/sim09/shell_*; policy shell_policy.joblib; 2016-25; coverage only on pass plays 2018+, box/personnel 2016+; 28 policy+outcome looks, 16 decomp looks): HGB policy fit/held (log loss vs state-blind, held 2022-25): two .6533 v .6759, man .6267 v .6636, DB count .7070 v .8831, box 1.1668 v 1.3187 (state adds .003-.079 beyond strength). Offence trailing late (defence protecting lead) final 2 min: light box .88/.87, dime .41/.34, two-high .44/.50 vs early .57/.66, .14/.10, .30/.38; leader offence faces heavier boxes. Pass: light box TD -.020/-.021 per play, two-high expl -.7/-2.4pp, yds -.06/-.58; runs into light box +.6-.8 yds. Mediation (early-game ref, drive TD): trailerLast2 gap -.126/-.123 fit/held, posture mediates 84%/87% (boot 90% CI .74-.99/.74-1.1); leader Q3/Q4 drives: posture raises TD (+.019/+.015; -.069 total Q4, direct -.117), so it does NOT explain the sim leader TD excess. Defence posture is observed with offence formation (selection, inferred).
U3a (scripts/sim09_u3a.py, artifacts/sim09/u3a/*.csv, analyze.log; fixed code col lg[:,6], idx yards match 1.0; 6 worlds x 8 seasons, 1.48M plays, game-bootstrap CI). Sim reproduces: shotgun/no-huddle urgency slopes (diff <=.009pp), INT/sack/fum/deep, leader Q3 TD (.229 vs .230, p+ .42), 3rd&1 (.672 v .667), D3 after own score/turnover (fit). Too weak: leader air -.14 v -.22 (p+ .99), leader ypp/risk (-.13 v -.20 risk, p+ 1.0), second-half air +.36 v +.55, risk +.09 v +.21, prior_len ypp +.10 v +.18, drive-position epa 0 v -.02/-.03, held own-score +.011 v +.033. Wrong: trailer final-2-min drives -0.46 plays v -0.08 (p+ 0), TD -.029 v -.012, Q4 trailer TD .200 v .234, Q4 leader TD .121 v .132, epa trailer -.032 v -.014; no-huddle faced overshoot ypp +.70 v +.22; 4th-down conv .522 v .498, 3rd .394 v .400, 2nd .330 v .324; drives/g 23.35 v 22.67. Halftime-deficit TD carry +.035 v +.030 ok (p+ .84).

U3b (scripts/sim09_posture.py, artifacts/sim09/u3b/*; variant crzp = crz + posture). Policy refit on 2016-17 only (box, DB|box HGB on score/clock/down/home/timeouts + linear team tilt, OOF log loss box 1.388 blind -> 1.268 state -> 1.264 +team; DB|box .964 -> .661); seasons 2018+ never touched (old shell_policy.joblib fit 2016-21 leaks, not used). Pool tags: 63,776 of 346,552 pool rows (18.4%; 98.9% of 2016-17 scrimmage plays; 2009-15 and kicks neutral). Mechanism: neighbour weight x pi(posture|sim state, team)/pi(posture|play state, team), box x DB|box joint, qtr read from caller frame. 6x8 sim vs crz, same seed: posture mix among tagged draws already matches real in every state under crz (trailer final-2 light box .488 crz / .479 crzp / .479 real; leader final-2 .031/.028/.033), so reweighting moves nothing: trailer final-2 plays/drive -.455 (crz -.460, real -.082), TD -.027 (-.029, -.012), Q4 trailer TD .2009 (.2001, real .2341), Q4 leader TD .118 (.121 vs .132), drives/g 23.35. Gate/budget seed 11 vs e5 crf4 (s11 SD 15.23, nonstr 209.6, r2 .107, margin var 231.9, strength 49.6, noise 182.9, Q4 slope -.044, drives 23.86): run in flight since 13:34 (3 workers, ~100 min), result lands in artifacts/mod25e3/e5_crzp_s11/e5.json, log artifacts/sim09/u3b/e5_crzp_11.log (last line 'done crzp 11'). Conclusion (inferred): posture is already imported by the state kernel; the trailer late-drive gap is not missing posture in the drawn plays. Unresolved below power: nothing closed.
U3d (scripts/sim09_u3d.py; variant crzk = crz + tfix; artifacts/sim09/u3d/anatomy_*.csv, play_crzk, u3d_analyze.log). Anatomy real v crz, trailer drives starting in last 2 min of Q4: snaps/drive 4.62 v 3.53, drives/game 0.46 v 0.73, ending on downs .117 v .214, last play at down 1-3 in 64% of sim downs-endings (real 1%). Mechanism: pool rows that are the last row of a game or of Q2 carry an artifact transition (game_last forced flip, next_down 1, own 25; Q2-last row flip = Q3 first-play posteam), so a drawn end-of-game or end-of-half kneel/spike/incomplete flips possession mid-clock and fdnb reads next_down 1 as a conversion. Fix install_tfix: for those rows with no score and no turnover set flip False and next_down NaN (engine then derives the continuation from yards; 4th down flips by rule). 2717 rows repaired. crzk v real: plays/drive trailer final2 -.199 (crz -.46, real -.082), drives/g 22.78 v 22.67, Q4 trailer TD .217 v .234, Q4 leader .127 v .131, 4th-down conv .511 v .498, Q2 trailer drive snaps 4.14 v 4.15. Remaining: trailer TD/drive -.0195 v -.0119, Q2 window drives/game .64 v .75. Caution: u3a to_pbp maps code 0 to pass but engine code 0 is run, 1 pass.
U3d e5 status: seeds 11,12,13 of crzk running sequentially in background (3 workers); s11 e5.json landed at artifacts/mod25e3/e5_crzk_s11/e5.json (not yet compared to crf4 s11: SD 15.23, nonstr 209.6, r2 .107, margin var 231.9, strength 49.6, noise 182.9, Q4 slope -.044, drives 23.86); s12, s13 land in e5_crzk_s12/e5_crzk_s13, logs artifacts/sim09/u3d/e5_crzk_1*.log. Compare each to artifacts/mod25e3/e5_crf4_s1*/e5.json.

## Next
2026-10-02: fixed run/pass code swap in sim09_u3a.py to_pbp (engine 0=run,
1=pass): every U3a/U3c/U3d analyze table so far is suspect for run/pass
split metrics; rerun analyze for crz (u3a_play), crzh (u3c/play_crzh),
crzk (u3d/play_crzk). U3d found pool rows at game/half end with fabricated
flips (crzk tfix, scripts/sim09_u3d.py) - real bug fix; e5 seeds for crzk
and crzh running (artifacts/mod25e3/e5_crz{k,h}_s11-13). Next: merge
crzk+crzh -> crzhk and gate.
U3c IN FLIGHT: crzh (crz + exact half split, Q3 phase 5 pool; min pool
7,799, fallback 0) in scripts/sim09_half.py; chain artifacts/sim09/u3c/
run_all.sh writes alldone.txt when done. Then compare analyze.log with
u3a/analyze.log and e5_crzh_s11-13 with e5_crf4_s11-13 (artifacts/mod25e3).
U3b done: posture reweighting (crzp, scripts/sim09_posture.py) changes
nothing; the kernel already carries real posture mix. crzp seed 11 e5 may
still be running (artifacts/mod25e3/e5_crzp_s11/e5.json); read it, run
seeds 12-13 only if it differs from crz. Structural defect found: the
engine's time feature folds halves (time_raw = seconds left in half; phase
0 shared by Q1 and Q3), so second-half behaviour is diluted by first-half
neighbours (U1 2H air +.55 real vs +.36 sim). U3c: add half to the state
match. U3d: trailer final-2 drive anatomy (drive-end reasons, sec/snap by
outcome, timeouts, out of bounds) and fix.
U3b done: posture adds nothing because the kernel already carries it. Remaining
trailer final-2 gap is in drive structure, not drawn-play posture: investigate
(1) per-play clock/elapsed used in the final 2 min vs real (plays/drive -.46 v
-.08 while no-huddle and shotgun slopes match), (2) the engine's time feature
has no half (time_raw = gsr-1800 in Q1/Q2, gsr in Q3/Q4, so Q1 and Q3 states
share neighbours; only Q2-end and Q4 phases flag), (3) drive-end rule when the
clock expires inside a drive. Then second-half level, prior-drive fatigue,
no-huddle-faced overshoot, 4th-down conv, stakes, A4. Seeds 12 and 13 of crzp
e5 (python scripts/sim09_posture.py e5 --variant crzp --seed 12 --workers 3,
about 75 min each) if the seed-11 gate/budget differs from crf4.

## Open
Full-column pbp for 2018-25 is not downloaded (needed for held-out checks).
