# MKT-25

## Goal
Opener revision size: this week's opener minus the team strength implied by last week's close, adjusted for known injuries (ROADMAP MKT-25).

## State
Unit 1 done 2026-11-03 16:37 local. scripts/mkt25_opener_revision.py (output tests/scratch/mkt25_out.txt, batch artifacts/mkt25/record_batch.json). Rating: trailing 17-week LS on closing spread_line with fitted home field (mean 1.475, sd 0.20), in-window resid sd 3.03; R = opener minus implied, sd 3.88 (corr with close-minus-implied 0.93). n 1325 games 2020-25 weeks 2+. Measured: R slope on prior cover margin diff +0.049 CI[.038,.060] P+ 1.0 (halves .046/.052; market follows last cover, both hinge sides); prior line move +0.047 P+ .84 (halves -.046/.123); injury-news change -0.041 P+ .17; injury level -0.32 CI[-.43,-.22]. LOSO terms on served logit (9 looks): raw R dLL -.00029 P+ .29 flips 71-58; cover-adjusted e dLL -.00010 P+ .45 flips 91-93; +move+injury adjusted -.00007 P+ .48 flips 94-95; coefs positive every fold; served vs intercept +.00112. 5 cells recorded unresolved_below_power, family mkt25_opener_revision. Nothing closed. Team-level revision is not identified (16 games per week, perfect matching), so game-level only.

## Tried
Novelty check only. Nearest: MOD-21 (strength from closing lines), MKT-22, MKT-23; grep 'revision' found only pick-revision ledger text. Differs: Revision of the opener versus the previous close.

## Next
Unit 2: 2013-19 openers if archived; position-weighted injury news (QB, OL); split revision by sign of prior cover to test asymmetry; rating window alternatives declared as a family.

## Open
Data: Local lines archive and injury captures (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
