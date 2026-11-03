# mod24-unit-next-2026-11-03

## Goal
One MOD-24 unit on labels per game: more graded training games for the served base ridge. Parent docs/lanes/accuracy-ceiling-theory.md.

## State
DECLARED before any look (2026-11-03). Lever: add postseason games (WC/DIV/CON/SB, 199 rows 2009-2025, ats_margin and features already in game_features_weak_stack.parquet) as training rows. Served training, the opener harness and every ridge arm drop them via regular_season_rows (read: src/nfl_ats/modeling.py:22, clv.py:2104).
Novelty (read): LEAD-87 borrowed playoff games only for the four-term news-response coefficient (Brier -0.0000036), not the margin ridge; acc-u5/u6, mod23 u10 robust, pre2009, FTN luck all train on REG only. Grep of ROADMAP/lanes/registry finds no playoff-rows-in-base-ridge arm.
Design: arm = clv.opener_pick_evaluation with clv.regular_season_rows replaced by identity in script scope (no src edit), served alpha 10, weak_stack 88 columns, market_residual, gaussian_median, expanding past-only refit weekly. Base = same code with REG only. Scored on decisive REG games 2020-25 at the opener; paired, season-week block bootstrap 4000. No fitted parameter, so no LOSO choice; include/exclude fixed here. Looks: 3 metrics (acc, log loss, Brier) = 3 cells, family mod24_postseason_rows, plus per-season.
Mechanism: +4% labels (~12/season) from the best teams in real games; playoff teams are selected so rows are not exchangeable with REG; a null is expected, this tests it.

## Tried
RESULT (measured; artifacts/mod24_postseason_rows/report.json; scripts/mod24_postseason_rows.py run): same 1503 decisive 2020-25 games, base 802-701 reproduced, arm 811-692, 87 flips. Acc +0.60 pts [-0.53,1.73] P+ .84; LL gain +0.00026 [-0.0011,0.0016] P+ .64; Brier +0.00015 P+ .67. Per season acc +0.9,+1.3,+0.4,-0.8,+0.4,+1.5 (5 of 6 positive). 3 cells recorded, family mod24_postseason_rows, unresolved_below_power. Looks 3. In-sample not run (no fitted choice).

## Next
Replicate on SBR-proxy 2011-19 (more graded games) and score 2026 forward; frozen include-rule, no tuning. Not closed.

## Open
Postseason rows differ in rest/neutral site; effect size on 4% extra labels is likely below evaluator resolution (unresolved_below_power expected).

## Replication 2011-19 (DECLARED before the look)
Population: REG decisive games 2011-2019 with an SBR open line (mod24_u1.proxy_population, base 1142-1089, n 2231). Grade: mod24_u1.proxy_eval (SBR open, discrete read plus home-side offset, expanding weekly refit, served alpha 10, weak_stack 88 cols); arm = same with postseason rows kept in training only (population stays REG). Frozen rule from the 2020-25 look, no tuning, no fitted choice. Looks 3 (accuracy, log loss, Brier) pooled; per-season shown, not extra cells. Bootstrap season-week blocks. Family mod24_postseason_rows, cells suffixed _2011_2019. Run: scripts/mod24_postseason_rows.py replicate_2011_2019.
IN FLIGHT: background job (bash task b0zm08zgm) running `scripts/mod24_postseason_rows.py replicate_2011_2019`, log artifacts/mod24_postseason_rows/replicate.log, writes report_2011_2019.json (acc/ll/brier d with P+, per-season, flips). Next: read report_2011_2019.json, append numbers here, record 3 cells (_2011_2019 suffix, family mod24_postseason_rows, unresolved_below_power) via weak-signals record --batch. Code done (ruff clean; added __main__ guard to scripts/mod24_u1.py).
FINDING (read, src/nfl_ats/margin.py:890): fit_margin_model itself calls regular_season_rows, so patching only clv.regular_season_rows (the 2020-25 run) did NOT put postseason rows in the ridge fit; the first 2011-19 run with the same patch gave arm == base exactly (1142-1089, 0 flips). The 2020-25 +0.60 therefore came from other clv paths (population/pool), not training rows: needs a corrected re-score with margin.regular_season_rows also patched. 2011-19 rerun now patches both (background blopvk70a, log replicate2.log). Cached parquets arm/base_2011_2019 in artifacts/mod24_postseason_rows/.
RESULT 2011-19 (measured; report_2011_2019.json; postseason rows in fit via clv+margin patch; season-week blocks, 153): arm 1139-1092 v base 1142-1089, acc -0.13 pts [-1.30,+1.08] P+ .39; LL gain +0.00146 [-0.00079,+0.00367] P+ .90; Brier gain +0.00076 [-0.00027,+0.00178] P+ .93. 145 flips, decisive (flipped) games arm 71 v base 74. Per season d_acc -0.8,-0.4,+2.5,-2.4,-2.0,-2.1,+1.2,+3.7,-0.8 (3 of 9 positive). 3 cells recorded (re-recorded --replace after fixing a sign slip), unresolved_below_power. Looks 3. No fitted parameter, so no in-sample/out-of-sample split.
CAVEAT: the 2020-25 result used only the clv patch, which per margin.py:890 left the ridge fit REG-only; its +0.60 is not a training-rows effect. Next: re-score 2020-25 with margin.regular_season_rows patched too (new cells, _2020_2025_fit) and supersede the old cells; accuracy mixed, probability scores lean positive.

## Correction in progress (2026-11-03)
Forward script checked (read, scripts/mod24_postseason_forward.py:38-58): it fits make_margin_estimator directly on a train mask, bypassing fit_margin_model, so postseason rows DO enter the ridge there; the forward cells are valid, no re-freeze needed.
Candidate mechanism for the clv-only 87 flips (inferred, unproven): clv.py:2104 lets postseason weeks into paired scoring, so archive_stream (clv.py ~2250) gains postseason rows that feed prior_rows_before/fit_home_side_offsets (clv.py:2227), shifting the home-side offset on later weeks. Not training rows.
IN FLIGHT: `scripts/mod24_postseason_rows.py refit_2020_2025` (log artifacts/mod24_postseason_rows/refit.log, report_2020_2025_fit.json): base, true arm (clv+margin patched), clv-only, clv-only with postseason dropped from the offset stream. Next: read report, record true-arm cells with --replace (names mod24_postseason_rows_acc/log_loss/brier), note mechanism result here.
PIDs of the running refit: 19620 (launcher), 8960 (python). base_2020_2025.parquet cached; arms cache as they finish. Rerunning the same command resumes from cache.

## Root adjudication 2026-11-03 12:21 (measured, report_2020_2025_fit.json)
- 2020-25 true arm (clv + margin patches) = clv-only exactly: 811-692 v 802-701, 87 flips, +0.60 [-0.53, 1.73] P+ .84, log loss P+ .64. The 2020-25 path trains on the clv.py:2104 frame, so margin.py:890 does not apply there; the original cells are correct and stand.
- Mechanism check: clv-only with postseason rows dropped from the offset stream 810-693 (86 flips, +0.53 P+ .81); the home-side offset explains 1 flip, so the effect is the training rows.
- Evidence across populations: 2020-25 accuracy +0.60 P+ .84; 2011-19 accuracy -0.13 P+ .39, log loss P+ .90, Brier P+ .93; 2026 W1-7 forward 60-42 v 60-42, log loss P+ .75. Unresolved_below_power; probabilistic grades lean positive in all three, accuracy does not replicate. Not served.
