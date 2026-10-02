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

## Next
All measurement units done (U1, U2, F1, B1/B2/C1/D3, A, S1). U3 builds on
the E5/E7 base variant (docs/lanes/mod25e-generator-fidelity.md). Step 1:
run the U1/S1/B1 estimators on the sim's own plays to see what it already
reproduces (suspect the team kernel dilutes score-state matching). Step 2,
add only what is missing, each fitted: (a) defensive posture policy
(shell_policy.joblib) with neighbour draws restricted to the sampled
posture, which explains 84-87% of the trailer final-2-min drive gap;
(b) prior-drive-length fatigue state (B1); (c) urgency/conservatism
responses (U1); (d) stakes shifts weeks 10+ (U2); (e) deep x two-high and
deep x pass-factor (A4). Leader Q3/Q4 TD excess is NOT explained by posture:
look at leader offence behaviour and the yard shift (E7). Then U4 held-out
gate plus budget.

## Open
Full-column pbp for 2018-25 is not downloaded (needed for held-out checks).
