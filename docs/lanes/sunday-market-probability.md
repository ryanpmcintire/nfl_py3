# sunday-market-probability

## Goal

Measure one fixed offline structural arm: include Sunday morning leading-book spread movement in the same four-term fitted cover probability, against the current Sunday-midnight cutoff, before any serving decision.

## State

Measured and activated 2026-09-20 by the root owner at `artifacts/pick_probability/20260920T152908Z`; the old coefficient artifact remains available for rollback. Read-only verification loaded active pointer, metadata, coefficients, and model, and all four name `leader_median_through_sunday_prekick_v1`; 1,503 fit games and 799 market exposures. `scripts/sunday_market_probability_eval.py` reads the frozen discrete population at `artifacts/confidence_ranking_audit/20260920_aligned/population.parquet` and historical quotes at `artifacts/sharp_book_weighted_movement/spread_quotes.parquet`. It writes predictions, coefficients, reliability, paired week-block intervals, and the frozen 799-row `market_move.parquet` to `artifacts/sunday_market_probability/20260920_fixed/`. Six unresolved metric/protocol cells were recorded by `nfl-ats weak-signals record --batch artifacts/sunday_market_probability/20260920_fixed/weak_signals_batch.json` under `sunday_market_probability_fixed_v1`. Sunday extraction and pointer/coefficient/metadata version binding are in `sharp_book_movement_features.py`, `pick_probability.py`, `pick_probability_fit.py`, `card_view.py`, and the CLI. The omitted CLI version inherits the validated active artifact version, including weekly-run's existing argv.

## Tried

Measured with `uv run --no-sync python scripts/sunday_market_probability_eval.py`: 799 incumbent move exposures; 797 reproduced exactly from cached quotes. Two 2025 games differ by 0.5 point between cache reconstruction and the frozen incumbent, so both stay unchanged in the challenger. Among 797 comparable exposures, 232 cumulative leader moves change using observations before the earlier of 12:45 p.m. ET Sunday and kickoff. All other features and 1,503 population rows remain fixed. This is one declared structural arm; the artifact counts three protocols, three metrics, five reliability bands per arm, and 32 coefficient fits. Earlier feature choice reused these seasons, so this is not untouched outer validation.

Leave-one-season-out: Sunday versus current Brier 0.244839 versus 0.245388, improvement +0.000549, week-block 95% [-0.000298, +0.001464], `probability_positive=0.892`; log loss 0.682776 versus 0.683952; 863 versus 859 correct, with 22 versus 18 on 40 decisive games. Chronological 2022-25: Brier +0.000403 [-0.000529, +0.001430], `probability_positive=0.782`; log loss 0.682781 versus 0.683638; 603 versus 602 correct, with 12 versus 11 on 23 decisive games. In-sample Brier improvement is +0.000472, slightly below the leave-one-season-out estimate. Sunday move coefficients are positive in all six leave-one-season-out folds (0.189–0.246); chronological 2022-23 weights are zero because 2020-22 have no incumbent move exposure, then 0.210 and 0.189. The `summary.json` reliability tables show a shortfall in the strongest band for both arms. The result is `unresolved_below_power`, not a rejection or serve decision.

Live versioned exposure replay: with genuine OddsGap and Bovada private captures retrieved before `as_of`, 15/15 remaining Week 2 games have three leaders, and eight median moves differ from the Sunday-blackout input. At `as_of=2026-09-20T15:10Z`, before either private retrieval, zero games use those quotes. Eligibility requires actual retrieval before cutoff; source scan orders quotes within a book, so an older OddsGap Bovada scan does not supersede later direct Bovada. Historical 797/797 comparable exposures had all three leader books; one-book inference remains an extrapolation. Targeted Ruff format/check, mypy on `src`, and `scripts/strip_comments.py --check` passed before the final version helper edit; root owns complete final gates.

## Next

Root owns the real refresh/publication path, complete verification gates, dashboard publication and push. Inspect `artifacts/active_pick_probability.json` and this lane on a fresh session before any further fit; the normal weekly fit now inherits the active Sunday feature version.

## Open

The two cache-versus-incumbent 2025 quote differences remain held at their incumbent values in the fixed comparison and training artifact. The Sunday improvement is unresolved below power; promotion is a mechanism and owner decision, not a claim of a stable edge.
