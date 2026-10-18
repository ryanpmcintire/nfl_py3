# mod25e E174 serial covariance split (strength / game-level / carry-over)

## Goal
On base ...g2rtgw13gx2faldrg s11-13 v real 2009-17, split the excess within-game cross-quarter covariance (E168) into season-rating strength, possession carry-over, and a game-level remainder; explain the H/A negative correlation. Offline.

## State (measured 2026-10-18; tests/scratch/e174/an.py, an.txt, an.csv; 300 game boots, 90%; ratings = leave-game-out team-season mean margin, fixed across boots; ~40 stats, one family)
- 2*sum of quarter cross-covs (VarM minus sum Var m_q): real -11.4, sim +5.1, gap +16.5 [+6.0,+28.2] P1.00.
- (a) strength (regress m_q on S): explains 28.3 real v 29.8 sim, gap +1.4 [-2.7,+6.1]; gap after S +15.1 [+6.3,+23.6]. Strength is not the carrier. VarS sim 88.8 v real 79.9 (+8.8 [+4.8,+12.8]; mostly rating noise, inferred).
- (c) carry-over (first possession of the quarter: offence side and start yl): explains only +2.0 [+0.2,+4.2] of the gap; gap after S+carry +13.1 [+4.2,+21.3]. m1m2 gap is absorbed (+.28), so Q1->Q2 is carry-over.
- (b) game-level remainder carries about 13 of 16.5: m2m3 +2.3 [+.5,+4.4] (across halftime, so cannot be carry-over), m3m4 +1.8 [-.1,+3.6], m2m4 +1.6, m1m4 +1.1 (CI span 0).
- Generator (read): per-game latents are only AR(1) season-level dev + backup-QB (mod25_generator.py:232-247, cov_fit.json; E85: latents 14% of realised sim form, off phi .9987 = pure season level) and STF special-teams game latent (stf.py:243-247). No per-game tempo draw; rest is engine path (pool draw).
- H/A: Cov(H,A) -7.4 real v -12.6 sim (-5.2 [-9.2,-1.2]); after team points ratings -1.1 v -3.8 (-2.6 [-5.9,+.4]); after ratings+N -1.3 v -4.2. Pace is not it: Var(N) sim lower (-.87 [-1.4,-.3]), Cov(H,N)+Cov(A,N) per N equal (.237 v .254). About half the gap is rating-driven (VarS higher), the rest unexplained (P .07-.08).

## E174b channel attribution (measured 2026-10-18; tests/scratch/e174/ch.py an2.py an2.txt; real 2009-17 v sim base s11-13 possession frames from e131 ch.py; 300 game boots 90%; ~110 cells one family; reproduces real -11.40 sim +5.10 gap +16.5 [+7.0,+26.9])
Identity: 2*sum xq cov = sum over drives of (centred margin after prior quarter x offence-signed net points), split into start-yl value (real mean net pts by 10-yd start bin) and conversion (points minus that value), by leader/trailer and previous-drive end (prev end: score/punt/other inferred from points and last play).
- Start-yl channel total -1.4 [-4.8,+2.4]; conversion channel +17.9 [+9.9,+26.3]. Conversion: trailer +16.3 [+8.9,+24.3], leader +1.6 [-3.8,+6.4]. Trailer conversion gap by window: Q2 +5.9 [+2.8,+8.4], Q3 +4.9 [+1.2,+9.6], Q4 gsr 300-900 +5.4 [+1.2,+10.5], Q4 last 5 min +0.0 [-3.0,+2.8]. Leader conversion: Q2 -4.1, Q3 -1.9, Q4 300-900 +4.5 [+1.1,+7.8], last 5 +3.1 [+0.6,+5.4]. Start yl: leader after score -5.3 [-7.8,-3.3] (last 5 min -4.6 [-7.0,-2.3]); trailer Q4 300-900 +3.2 [+1.0,+5.6].
- Last 5 min nets about 0 (lead conv +3.1, lead yl -4.6, trail conv 0, trail yl +1.5). Q4 300-900 about +11 and Q2+Q3 about +11 (components, not jointly bootstrapped).
- Descriptives sim-real: pts/drive trail 1-7 -.12 [-.18,-.07], 8-16 -.16 [-.23,-.09]; lead 1-7 -.11; other-end (turnover/downs) rate trail 8-16 -1.3pp, 17+ -2.4pp [-4.0,-1.0] (sim trailers turn over less); trail 17+ run share +2.0pp [+1.2,+2.7]; plays/drive trail 8-16 -.12; drives/game equal (CIs span 0); trailer start yl after score +.4 (8-16). yds/play -2.2 is the TD yards=0 frame artifact (E160b), ignore.

## Mapping to named mechanisms (inferred from effect windows above plus lane effect sizes)
STBK: acts on leader starts after trailer scores (late); sign is WRONG for this metric (closing -4.6..-5.3 raises gap), fidelity fix only. RGT, FD5, GSTOP2: all act in the last 5 min where the net window is 0; expected closure 0 +-3 each (FD5 FG effect -.035 FG/game too small). RG/LDR in base. SAFEG, return-TD geometry: start-yl channel, total -1.4: negligible. Unaddressed: trailer conversion deficit Q2-Q4 300-900 (+16 [+9,+24]); leader over-conversion Q4 300-900 (+4.5). Candidate real-behaviour evidence: sim trailers turn over less, pass less when far behind, score less per drive at the same start (no mechanism named yet; E88 start-shift channel is only the trailer yl +3.2 piece).
Ranking of queued runs by expected gap closure: all near 0; RGT/GSTOP2/FD5 about 0 (upper |3|), STBK negative; none above 3 of 16.5.

## Next
Name the mid-game trailer behaviour mechanism: sim v real trailer play mix, 4th-down go and turnover rate per drive in Q2-Q4 gsr>300 by margin (LOSO fit class ratios by role). unresolved_below_power until weak-signals record (not run). Do not use ADJ2.

## Open
Joint CI for windows; prev-end type is inferred. Note: importing e131 ch.py re-ran its main and overwrote tests/scratch/e131/Rd.pkl and ch.txt with the full-base label.
