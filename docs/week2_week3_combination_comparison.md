# Compare the good week with the bad week

**Measured, September 28, 2026:** the final published Week 2 card went 14-2;
the raw model would have gone 11-5 at the same pool lines. Week 3 through Sunday
was 5-10 versus raw 9-6, with one Monday game pending. The adjustments added
three wins in Week 2 and cost four in Week 3. Across those two completed sets,
served picks were 19-12 versus raw 20-11: a one-win difference.

| Week | Raw model | Published card | Net wins from adjustments |
| --- | --- | --- | --- |
| 2 | 11-5 | 14-2 | +3 |
| 3, through Sunday | 9-6 | 5-10 | -4 |

**Measured:** Week 2 changes DET to BUF, GB to NYJ, LAC to LV, and CHI to MIN
turned losses into wins. TEN to PHI turned a win into a loss. All four Week 3
changes, BAL to DAL, PIT to CIN, WAS to SEA, and NYG to TEN, turned wins into
losses. Across nine disagreements, four helped and five hurt. The descriptive
Wilson 95% interval for the share that helped is 18.9%-73.3%; games are not a
new independent validation cohort.

**Measured:** the Sunday forecast metadata uses the same ridge recipe, feature
profile, discrete probability policy and seven counted signals in both weeks.
The fitted coefficients changed only slightly. Applying the Week 2 coefficients
to the six already-verified Week 3 input replays changes zero sides; the largest
probability change is 0.000968 percentage points. Those six replays cover three
games at two publication times, not the entire Week 3 card.

**Read:** the scoped changes between commits `5fb88a2` and `dd6c140` in
`card_view.py` concern presentation and nomination wording. The change in
`sharp_book_movement_features.py` accepts self-timed quotes in the legacy
movement path; the Sunday-inclusive policy used by these coefficient artifacts
already accepted those quotes. This is a scoped version comparison, not a
claim that all repository code was unchanged.

**Inferred:** a diagnosis that counts only the four Week 3 harms gives an
incomplete account. The same adjustment system contributed to the excellent
Week 2. These results leave the longer-run value unsettled; they do not justify
changing the served recipe solely in response to this week's result. The
untouched future comparison remains in place. No signal is closed here.

## Evidence and checks

The comparison command was:
`uv run --no-sync --offline python .tmp/compare_live_weeks.py` with the repository
uv executable and `.tmp/uv-cache`. It saved the per-game comparison to
`.tmp/week2-week3-contribution.csv` and the aggregate and old-weight replay to
`.tmp/week2-week3-comparison.json`. The input published grades are in
`.tmp/2026-three-week-published-grade.csv`; they use the saved public final
scores in `.tmp/week03-nflverse-scores-latest.csv`.

Every raw forecast precedes its game's pool deadline and uses the exact
published pool spread. The home cover margin is home score minus away score
minus the stored home-favorite-positive spread. A separate check using the
latest forecast before publication instead of before the deadline produced
identical raw sides for all 32 games. The 15 completed Week 3 raw sides also
matched the earlier per-game trace. Published picks, rather than the discrepant
paper settlement ledger, define what was served.

Forecast directories under `artifacts/margin_predictions/`:

- Week 2 Thursday: `2026-week-02-20260917T210914Z`.
- Week 2 Sunday: `2026-week-02-20260920T160851Z`.
- Week 3 Thursday: `2026-week-03-20260924T210918Z`.
- Week 3 Sunday: `2026-week-03-20260927T161013Z`.

Coefficient artifacts under `artifacts/pick_probability/`:
`20260920T161202Z` and `20260927T161329Z`. The original verified Week 3 replay
inputs are in `.tmp/week03-probability-trace.json`.

Week 1's published 9-7 result is context only; this adjustment comparison is
limited to Weeks 2 and 3. No prediction code, live card, study enrollment,
scheduler, or dashboard changed during this audit.
