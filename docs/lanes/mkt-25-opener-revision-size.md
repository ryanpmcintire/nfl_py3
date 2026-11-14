# MKT-25

## Goal
Opener revision size: this week's opener minus the team strength implied by last week's close, adjusted for known injuries (ROADMAP MKT-25). Does the market over- or under-revise?

## State
Unit 1 (2026-11-03, scripts/mkt25_opener_revision.py, tests/scratch/mkt25_out.txt): trailing 17-week rating on closing lines; R = opener minus implied, sd 3.88; n 1325, 2020-25 weeks 2+. R slope on prior cover margin +0.049 [.038,.060] P+ 1.0 (market follows last cover). LOSO terms on the at-open model logit: raw R dLL -.00029 P+ .29 flips 71-58, coefs positive every fold; cover-adjusted -.00010 P+ .45. 5 cells unresolved, family mkt25_opener_revision.

Unit 2 (2026-11-03, scripts/mkt25_unit2.py, tests/scratch/mkt25_unit2_out.txt): n 1415, 17 cells unresolved, family mkt25_opener_revision_unit2. Slope after cover .066, after non-cover .085, diff -.018 P+ .04; blowout diff -.023 [-.040,-.006] P+ .004; windows 8/12/17 slope .052/.054/.053; every term dLL negative (P+ .01 to .21).

Unit 3 (declared 2026-11-14 17:13 ET before any 2013-19 result was joined; done 17:19 ET; scripts/mkt25_unit3.py stages proxy/replicate/artifact, tests/scratch/mkt25_unit3_out.txt, artifacts/mkt25/record_batch_unit3.json). All **measured** unless marked.
- Line: 2013-19 has no timestamped opener on disk, only the SBR Open column (data/processed/sbr_odds.parquet: no timestamp, no book), with the at-open model logit from artifacts/extended_fit_population/20260923T205910Z. On the 2020-21 overlap (n 466) it sits 1.37 pts mean abs from the true Tuesday opener (exact 17.0%, within a point 60.3%, r .951, home side +0.32). It is not a close (1.83 from the capture close, 2.03 from SBR's close; path position .08 where 0 is opener and 1 close), but it is farther from the true opener than the close is (1.13). 2013-19: 1.40 from SBR's close.
- Structure replicates: R slope on prior cover margin +0.0472, season-block [.0293,.0650], P+ 1.0, 7/7 seasons, n 1517, beside +0.0492 [.0329,.0677] 6/6. Era difference -.0020 [-.027,.022] P+ .45; pooled 2013-25 +0.0481 [.0353,.0608], n 2842.
- Outcome against the line does not replicate the fixed follow direction. Logit coef of R per point beside the model logit: -0.0294 [-.0472,-.0129] P+ .000, 0/7 seasons positive (per SD -.105), beside +0.0096 [-.0108,.0245] P+ .84, 4/6. Follow side vs the line: 729-788 (-1.94 pts [-3.86,-0.30], P+ .007, 2/7 seasons above half) beside 673-652 (+0.79, P+ .87, 3/6). Within-season permutation p two-sided .035 (coef) and .149 (record); one-sided for follow .98 and .94. Frozen 2020-25 fit applied unchanged: dLL -.00125 [-.00196,-.00063], flips 56-62, 0/7. LOSO terms inside 2013-19 gain by fading (raw R +.00122, week-block P+ .81, flips 234-210, coefs negative every fold). Intercept-only base gives the same coef (-.0299); the model logit does not beat an intercept there (-.00065, P+ .32). Pooled coef -.0105 [-.0235,.0021] P+ .06; era difference -.039 [-.063,-.014].
- Eras are commensurable in construction, not in line type. Same-game bridge 2020-21 (n 362): slope with the true opener .0552, with the SBR open .0115, difference -.0437 [-.055,-.033]; 2019 alone is .003. **Inferred**: from 2019 the SBR Open is a line set before the prior game, so the overlap years do not describe 2013-18.
- Post hoc (2 looks, after the sign was seen), 2020-21 n 352: the part of the SBR open that differs from the true opener earns -0.123 per point [-.256,-.009] when graded at the SBR line. A line error fades itself. An independent error of sd 1.77 pts would produce the whole 2013-19 coef; the gap to archived opening boards 2009-16 is 0.68 mean abs (**read**, done/opener-population-backfill.md:93), which gives about a third (**inferred**).
- Size beside power: MOD-36 detects .17 logit per SD at 1,503 games against its matched null, scaled by sqrt(1503/n): .169 at 1517, .181 at 1325, .124 pooled. Measured per SD -.105, +.037, -.039.
- 28 cells recorded unresolved_below_power, family mkt25_opener_revision_unit3 (26 declared looks plus 2 post hoc); pool exit 0. Nothing closed, nothing served.

## Tried
Novelty check only before unit 1 (nearest MOD-21, MKT-22, MKT-23). Unit 3: SBR close used as the proxy era's close (same source as its open); sign-rule cells recorded as rate_pp, not accuracy_points, because a proxy-line record is not commensurable with the pooled reference.

## Next
1. Declare a forward log before week 11 openers are seen: revision R at the true Tuesday opener, the follow side, result against the opener; direction follow (the only commensurable evidence), no served term.
2. Rerun the 2013-16 outcome coef on games where the SBR open agrees with the archived opening board (docs/sbr_opener_provenance.md names the file); if the fade vanishes there it is line error.
3. The revision slope stands as a market-structure finding in both eras.

## Open
Whether `wrong_sign_resolved` applies to the 2013-19 outcome cells: the season-block interval is wholly negative, but the line is a proxy with a named mechanical bias of that sign and the true-opener era reads positive, so they are recorded unresolved; root to verify. Season-block intervals rest on 7 and 6 blocks. Zero crossing closes nothing; one fitted probability selects the side; every cell is a look; graded at the opener with intercept-only control.

- Root adjudication 2026-11-14 17:20 ET: the 2013-19 outcome coefficient (-0.0294 [-0.0472, -0.0129]) is NOT recorded as a resolved wrong sign. That era's line is the archive Open column, which sits 1.37 points from a true Tuesday opener where both exist and whose own error earns -0.123 per point, so the negative sign measures the proxy's error, not over- or under-revision at a true opener. The eras are not commensurable in line type; nothing is pooled and nothing is closed. The same caution applies to any pre-2020 opener-graded cell, including the era split in the MKT-24 slot reliability unit (its odd-even split within era is the cleaner read).
