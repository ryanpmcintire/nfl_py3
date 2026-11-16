# Preseason favourite underpriced (goal-65 wave), 2026-11-16

## Goal
One declared confirmation look: the market under-adjusts toward preseason quality, so teams rated well preseason (win totals, Super Bowl futures) are underpriced against the line later in the season. Script scripts/favourite_underpriced.py, artifacts artifacts/favourite_underpriced/.

## Provenance (read)
Hypothesis SUGGESTED BY EARLIER RESULTS, so this look is the family's confirmation, not a discovery: futures_anchor college gap coefficient positive 13/13 folds against its declared fade (futures-anchor-2026-11-16.md); NFL team popularity leaned positive (done/team-mispricing-2026-11-16.md). Conflicting prior read: win_totals_anchor weeks 7-12 at the opener fitted the near-identical wt_drift term NEGATIVE, 6/6 folds, -0.079/SD (done/win-totals-2026-11-16.md), i.e. that look said preseason-favoured teams that slipped stay overpriced. Prior related cells: 6 + 4 + 4 + 3 in four families.

## Declaration (stamped 2026-11-16T22:40:21Z, before any outcome joined)
- Family favourite_underpriced_2026_11_16. Term pre_gap(team, s, w) = preseason_pct - opener_pct. preseason_pct = percentile rank among the season's teams of the mean of two percentile ranks (win total from data/external/win_totals, Super Bowl implied probability "prior to Wk 1" from data/external/preseason_futures); equal weights, none fitted (the two read the same quantity, corr 0.875 measured in win-totals lane); one source missing uses the other. opener_pct = percentile rank of a least-squares rating (team columns plus home-field column, neutral 0, minimum-norm) fitted to season-s REG openers of weeks < w (opener = Tuesday capture else SBR open, hindsight_audit_team.tuesday_quotes). Game term = home minus away, 0 when either team unrated. Franchise codes OAK/SD/STL -> LV/LAC/LA.
- Expected sign POSITIVE on home cover (back the preseason-favoured side).
- Population: REG weeks 3-18, all games.
- Screen (look 1, gating): REG 2011-2025 against the close, pushes dropped; OLS slope of home cover on the standardised term, accuracy points per SD in the declared direction; 200 within-season shuffles; 2000 season-block bootstraps for interval and probability_positive. Survivor = slope above the null 95th AND positive in both halves 2011-18 and 2019-25. Context: per-season slopes, back-preseason-side record on nonzero games.
- Served look (look 2, only for a survivor): opener 2020-25, weeks 3-18, pushes kept at 0.5, QW reference served probability (scripts/v4_candidate.py helpers); one offset-logistic coefficient, no intercept, LOSO; gating = held-out log-loss gain null percentile above 95 of 200 within-season shuffles AND positive coefficient on 2020-22 and on 2023-25. Cells: accuracy, log loss, Brier.
- Looks: at most 2 gating reads, 1 screen cell + 3 served cells.

## State (done 2026-11-16 17:42 EST; measured, artifacts/favourite_underpriced/screen.json, screen.log exit 0)
- Screen vs close, REG 2011-25 weeks 3-18: 3,350 non-push games (90 pushes dropped), all rated, 3,225 nonzero term; opener coverage 3,335; truncation check 12 games max diff 0.0. Term SD 0.234 rank units.
- pre_gap: -0.32 accuracy pts per SD in the declared direction [-1.62, +0.76], probability_positive 0.30, null percentile 36.5 of 200 (p95 +1.42, p05 -1.34); halves 2011-18 -0.22, 2019-25 -0.45 (both opposite the declared sign); 4/15 seasons declared-positive; back-preseason-side record 1608-1617. Term vs same-week close-minus-opener move corr +0.010.
- Not a survivor (null pct 36.5 < 95, halves negative), so the declared served look at the opener 2020-25 on QW was NOT run. Looks used: 1 gating read, 1 cell.
- Recorded (measured): 1 cell favourite_underpriced_pre_gap_close_screen, unresolved_below_power (interval crosses zero, halves agree so split-half reliability is not zero, no positive control); `weak-signals record --batch` exit 0 (recorded 1); `pool --league nfl --effect-units accuracy_points` exit 0.
- Reading (inferred): the confirmation look did not reproduce the college futures lean; the NFL point estimate sits with the win_totals_anchor sign (slight lean against the preseason-favoured side), both well inside the null.

## Tried
- One run, exit 0; no reruns, no alternate terms.

## Next
- Nothing in this family: the declared confirmation is spent. Any new preseason-prior idea needs a new mechanism and a new declaration, counted against the five prior families (futures_anchor, team_mispricing, win_totals_drift, win_totals_anchor, this one). Move lane to done/ at orchestrator review.

## Open
- None.
