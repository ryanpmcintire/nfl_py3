# Contract year: team snap share from players in the last year of their deal (NFL)

## Goal
Outside-leads candidate 4. Players in the final year of a contract may play with more effort; a team with a larger share of snaps from such players may beat the line. One term, NFL only, screened against the close, one served look at the opener only for a survivor. Zero crossing closes nothing; one fitted probability selects the side; LOSO; every cell a look; no magic numbers.

## Declaration (2026-11-16 16:01 EST, before contracts were downloaded or any outcome joined)
- Family `contract_year_family_v1`, category market. Hypothesis POSITIVE (home share minus away share up -> home covers more).
- Novelty (**read**): grep `contract.year|walk.year|load_contracts` over src/scripts/docs: only a literature-based findings card (`findings_content.py:847`, no registry fingerprint for a measured screen), scout v4 row 13, MOD-07 exclusion (no data). No measured contract-year term anywhere.
- Data: nflverse `historical_contracts` release (OverTheCap-derived), snap counts on disk `data/players/raw/<latest>/snap_counts.parquet` (2013-25). Player join pfr id -> gsis/otc via contracts or nflverse players table.
- Walk-year flag for season s: the latest contract signed in a calendar year before s (year_signed < s) has last season year_signed + years - 1 == s. Uses only deals signed before the season's calendar year (no in-season extension leaks); year-s offseason re-signings are noise, not leakage.
- Term per team-game: offensive plus defensive snaps by flagged players over all offensive plus defensive snaps, summed over the team's strictly earlier regular-season games of the same season; 0 when none. Game term = home minus away.
- Screen: regular-season games 2013-25 graded against the close (`spread_line`, pushes dropped), base = fair market probability 0.5 (offset logit 0), one offset-logistic coefficient per SD, no intercept, LOSO; 200 within-season shuffles, seed 36 stream 42. Survivor: held-out log-loss gain above its null 95th percentile AND same coefficient sign in both season halves. Looks: 1 gating, 3 recorded cells, 2 half coefficients.
- Served look (only if survivor): ONE MOD-36 opener read 2020-25 on the served out-of-season probability, pushes kept per the confirmation harness, declared here before running.

## State
Script `scripts/contract_year.py` (`build`, `screen`, `served`); raw `data/external/contracts/` (historical_contracts.parquet, players.parquet; `.gitignore *`); source entry `nflverse_contracts` in `config/source_policies.json`. Outputs `artifacts/contract_year/`.
Build (**measured**, exit 0): 52,615 contracts, 20,633 player-seasons with a prior-signed deal, 6,814 REG team-games 2013-25; snaps matched to a contract 75% (2013) rising to 88-89% (2018+); walk-year snap share 14-24% by season; term SD .078; as-of check 5 cutoffs max diff 0.0.
Screen (**measured**, `screen` exit 0, `screen.log`): 3,322 REG games 2013-25 vs close (pushes dropped; recomputed cover agrees with `home_cover` 100%), 3,117 with both teams' prior snaps (week 1 = 0). Changed 1,566, record 784-782. Acc +0.06 pts [-2.14,+2.41] pp .526 (null pct 24.5). LL gain -.000201 [-.000485,+.000089] pp .084, **null pct 49.0** (null 5-95 [-.00047,+.00035]). Coef per SD -0.014, same sign in 12/13 folds; halves +0.041 (2013-19) / -0.071 (2020-25) disagree. In-sample minus held-out ll +.000226. **No survivor**; served look not run, per the declared rule.
Recorded: 3 cells `contract_year_nfl_close_contract_year_snap_share_gap_<units>` (nfl, market, unresolved_below_power) in `registry/weak_signals/contract_year_family_v1.json`; record exit 0 (3); `weak-signals pool --league nfl --effect-units accuracy_points` exit 0 (`pool.log`). Nothing committed.

## Tried
- Net walk-year snap share, as declared; one screen; no re-cut. Fold coefficients are small and mostly negative but the interval spans both sides: unresolved_below_power, nothing closed (**inferred**: the halves disagree, so no stable sign either way).

## Next
- Nothing queued from this family. Any further look (position-weighted share, starters only, second-half-of-season only, a rookie-deal vs veteran split) is a NEW declared family, not a re-cut, and should be screened before any served look.

## Open
- Contract match covers 75-89% of snaps; year-s offseason re-signings are counted as walk years (noise, not leakage). `reliability` unrecorded.
- The literature-based findings card (`findings_content.py:847`) says "our own screen agrees"; this is now the first measured screen; the card text is the orchestrator's call.
