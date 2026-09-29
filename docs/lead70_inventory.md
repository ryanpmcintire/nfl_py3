# LEAD-70 source inventory

**Measured:** metadata and margin predictions only; no outcomes selected these sources.
The gate needs three distinct challenger configurations with all six seasons and
strictly pregame training cutoffs. Market/classifier outputs, college football, the
served configuration, and duplicate margin vectors cannot satisfy the gate.

| Inventory count | Value |
| --- | --- |
| college_files | 10 |
| common_games | 1503 |
| duplicate_margin_streams | 2 |
| files | 135 |
| missing_required_columns | 35 |
| no_target_games | 2 |
| not_margin_regression | 246 |
| qualifying_streams | 118 |
| served_configuration | 78 |
| unique_qualifying_models | 15 |

## Selected challenger streams

| Column | File | Configuration | Games | Minimum cutoff lag (days) | SHA256 |
| --- | --- | --- | --- | --- | --- |
| challenger_0 | artifacts/availability_experiments/20260813T133345Z/predictions.parquet | {"availability_method": "fixed", "feature_profile": "player_value", "method": "market_residual", "model_name": "ridge", "ridge_alpha": 10.0} | 1503 | 2 | c633ffc6a7e845d1542f97323d55e422cfefbe98b1496ee0acd6905ae768e964 |
| challenger_1 | artifacts/availability_experiments/20260813T133345Z/predictions.parquet | {"availability_method": "learned", "feature_profile": "player_value", "method": "market_residual", "model_name": "ridge", "ridge_alpha": 10.0} | 1503 | 2 | c633ffc6a7e845d1542f97323d55e422cfefbe98b1496ee0acd6905ae768e964 |
| challenger_2 | artifacts/player_experiments/20260813T122348Z/predictions.parquet | {"availability_method": null, "feature_profile": "base", "method": "market_residual", "model_name": "ridge", "ridge_alpha": null} | 1503 | 2 | a58b52ac07b342da1bd50d7e4d8adfe1e890cd0e1a0bb02401c9c784a0e96468 |
| challenger_3 | artifacts/margins/20260818T000122Z/predictions.parquet | {"availability_method": null, "feature_profile": "player", "method": "fair_margin", "model_name": "ridge", "ridge_alpha": 10.0} | 1503 | 2 | 5850a70fab43358e4ad73e38b11cb9774752daf2500abdb00f10beed01f940c4 |
| challenger_4 | artifacts/margins/20260818T000122Z/predictions.parquet | {"availability_method": null, "feature_profile": "player", "method": "market_residual", "model_name": "ridge", "ridge_alpha": 10.0} | 1503 | 2 | 5850a70fab43358e4ad73e38b11cb9774752daf2500abdb00f10beed01f940c4 |
| challenger_5 | artifacts/player_experiments/20260813T122348Z/predictions.parquet | {"availability_method": null, "feature_profile": "player", "method": "market_residual", "model_name": "ridge", "ridge_alpha": null} | 1503 | 2 | a58b52ac07b342da1bd50d7e4d8adfe1e890cd0e1a0bb02401c9c784a0e96468 |
| challenger_6 | artifacts/player_experiments/20260813T122348Z/predictions.parquet | {"availability_method": null, "feature_profile": "player_continuity", "method": "market_residual", "model_name": "ridge", "ridge_alpha": null} | 1503 | 2 | a58b52ac07b342da1bd50d7e4d8adfe1e890cd0e1a0bb02401c9c784a0e96468 |
| challenger_7 | artifacts/player_experiments/20260813T122348Z/predictions.parquet | {"availability_method": null, "feature_profile": "player_injuries", "method": "market_residual", "model_name": "ridge", "ridge_alpha": null} | 1503 | 2 | a58b52ac07b342da1bd50d7e4d8adfe1e890cd0e1a0bb02401c9c784a0e96468 |
| challenger_8 | artifacts/player_experiments/20260813T122348Z/predictions.parquet | {"availability_method": null, "feature_profile": "player_injuries_continuity", "method": "market_residual", "model_name": "ridge", "ridge_alpha": null} | 1503 | 2 | a58b52ac07b342da1bd50d7e4d8adfe1e890cd0e1a0bb02401c9c784a0e96468 |
| challenger_9 | artifacts/player_experiments/20260813T122348Z/predictions.parquet | {"availability_method": null, "feature_profile": "player_injury_value", "method": "market_residual", "model_name": "ridge", "ridge_alpha": null} | 1503 | 2 | a58b52ac07b342da1bd50d7e4d8adfe1e890cd0e1a0bb02401c9c784a0e96468 |
| challenger_10 | artifacts/participation_experiments/20260813T132030Z/predictions.parquet | {"availability_method": null, "feature_profile": "player_participation", "method": "market_residual", "model_name": "ridge", "ridge_alpha": 10.0} | 1503 | 2 | 56f768901fc3d2769287808f7a6781a59f4b93c71727a93cc82233a6f0aa770d |
| challenger_11 | artifacts/player_experiments/20260813T122348Z/predictions.parquet | {"availability_method": null, "feature_profile": "player_qb", "method": "market_residual", "model_name": "ridge", "ridge_alpha": null} | 1503 | 2 | a58b52ac07b342da1bd50d7e4d8adfe1e890cd0e1a0bb02401c9c784a0e96468 |
| challenger_12 | artifacts/player_experiments/20260813T122348Z/predictions.parquet | {"availability_method": null, "feature_profile": "player_qb_continuity", "method": "market_residual", "model_name": "ridge", "ridge_alpha": null} | 1503 | 2 | a58b52ac07b342da1bd50d7e4d8adfe1e890cd0e1a0bb02401c9c784a0e96468 |
| challenger_13 | artifacts/player_experiments/20260813T122348Z/predictions.parquet | {"availability_method": null, "feature_profile": "player_qb_injuries", "method": "market_residual", "model_name": "ridge", "ridge_alpha": null} | 1503 | 2 | a58b52ac07b342da1bd50d7e4d8adfe1e890cd0e1a0bb02401c9c784a0e96468 |
| challenger_16 | artifacts/margins/20260929T192312Z/predictions.parquet | {"availability_method": null, "feature_profile": "weak_stack", "method": "fair_margin", "model_name": "ridge", "ridge_alpha": 10.0} | 1503 | 2 | 93a0ad9e446e152eeebe4de34043d7b7bcca0cc1a3899ef8d77a358e12ab1fa3 |

## File inventory

| File | Disposition | Qualifying streams before deduplication |
| --- | --- | --- |
| artifacts/availability_experiments/20260813T133345Z/predictions.parquet | {"qualifying_streams": 2} | 2 |
| artifacts/backtests/20260812T101321Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/backtests/20260812T101634Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/backtests/20260812T111305Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/backtests/20260812T111822Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/backtests/20260812T125726Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/backtests/20260812T130504Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/backtests/20260812T145820Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/backtests/20260818T012001Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/cfb_benchmark/20260816T174400Z/predictions.parquet | college football; excluded | 0 |
| artifacts/cfb_benchmark/20260818T115149Z/predictions.parquet | college football; excluded | 0 |
| artifacts/cfb_james_stein_unit/20260818T213139Z/predictions.parquet | college football; excluded | 0 |
| artifacts/cfb_role_experiments/20260817T110002Z/predictions.parquet | college football; excluded | 0 |
| artifacts/cfb_role_experiments/20260817T110541Z/predictions.parquet | college football; excluded | 0 |
| artifacts/cfb_value_weighted_continuity/20260818T211758Z/predictions.parquet | college football; excluded | 0 |
| artifacts/cfb_variance_experiments/20260817T112146Z/predictions.parquet | college football; excluded | 0 |
| artifacts/era_weighting_cfb_screen/20260819T235500Z/predictions.parquet | college football; excluded | 0 |
| artifacts/era_weighting_nfl_screen/20260820T000500Z/predictions.parquet | missing train_max_gameday | 0 |
| artifacts/era_weighting_opener_read/20260820T002230Z/predictions.parquet | missing gameday, predicted_margin, train_max_gameday | 0 |
| artifacts/era_weighting_promotion_look/20260821T174655Z/predictions.parquet | missing gameday, predicted_margin, train_max_gameday | 0 |
| artifacts/era_weighting_promotion_look/20260821T174753Z/predictions.parquet | missing gameday, predicted_margin, train_max_gameday | 0 |
| artifacts/experiments/20260812T111352Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/experiments/20260812T111752Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/experiments/20260812T130732Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/experiments/20260812T143322Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/experiments/20260812T174728Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/experiments/20260812T175708Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/experiments/20260812T181705Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/experiments/20260812T183318Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/experiments/20260812T202616Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/independent_historical_validation/20260928_nested/calibration_diagnostic/predictions.parquet | missing predicted_margin, train_max_gameday | 0 |
| artifacts/independent_historical_validation/20260928_nested/predictions.parquet | missing predicted_margin, train_max_gameday | 0 |
| artifacts/margins/20260816T184528Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 2} | 2 |
| artifacts/margins/20260817T163540Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 2} | 2 |
| artifacts/margins/20260817T200603Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 2} | 2 |
| artifacts/margins/20260818T000122Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 2} | 2 |
| artifacts/margins/20260818T012407Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260820T004830Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260820T004951Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260824T110926Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260824T120013Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260903T141756Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260903T143251Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260905T124253Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260905T131238Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260905T133348Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260906T160812Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260907T120627Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260907T122310Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260907T151555Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260907T160814Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260907T164310Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260907T232637Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260908T011655Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260908T124423Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260908T162837Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260908T170903Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260908T183858Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260908T190735Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260908T192425Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260908T194146Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260908T195212Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260908T200240Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260908T200538Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260908T201934Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260909T182523Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260910T005406Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260910T154130Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260910T155650Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260910T161359Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260910T165054Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260910T210809Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260911T160857Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260911T164740Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260911T164743Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260911T164746Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260911T173646Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260911T183459Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260912T135818Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260912T160802Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260912T173846Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260913T113502Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260913T133822Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260913T160755Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260914T160810Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260915T171205Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260915T183932Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260916T160841Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260917T005413Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260917T161058Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260917T210805Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260918T160820Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260920T133847Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260920T150819Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260920T160800Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260921T160806Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260922T183842Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260923T003019Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260923T005937Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260923T160903Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260924T005342Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260924T161104Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260924T210830Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260925T160830Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260926T161310Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260926T163946Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260926T165526Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260926T173914Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260927T134152Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260927T160927Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260928T160842Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260929T183944Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/margins/20260929T192312Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/nested_evaluations/20260812T155915Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/nested_evaluations/20260812T161623Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/nested_evaluations/20260812T162320Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/nested_evaluations/20260812T175132Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/nested_evaluations/20260812T202910Z/predictions.parquet | missing predicted_margin | 0 |
| artifacts/offseason_retention_cfb_permetric/20260818T211604Z/predictions.parquet | college football; excluded | 0 |
| artifacts/paired_opener_evaluation/20260927-frozen-config/predictions.parquet | missing gameday, predicted_margin, train_max_gameday | 0 |
| artifacts/participation_experiments/20260813T132030Z/predictions.parquet | {"qualifying_streams": 2} | 2 |
| artifacts/pbp_replication/20260816T142340Z/predictions.parquet | no target NFL games | 0 |
| artifacts/per13_durability_stage1/20260901T190549Z/predictions.parquet | missing game_id, gameday, predicted_margin, train_max_gameday | 0 |
| artifacts/per13_durability_stage1/20260901T190609Z/predictions.parquet | missing game_id, gameday, predicted_margin, train_max_gameday | 0 |
| artifacts/player_experiments/20260813T120248Z/predictions.parquet | {"qualifying_streams": 8} | 8 |
| artifacts/player_experiments/20260813T121419Z/predictions.parquet | {"qualifying_streams": 5} | 5 |
| artifacts/player_experiments/20260813T122110Z/predictions.parquet | {"qualifying_streams": 5} | 5 |
| artifacts/player_experiments/20260813T122348Z/predictions.parquet | {"qualifying_streams": 10} | 10 |
| artifacts/qb_continuity_replication/20260816T143913Z/predictions.parquet | no target NFL games | 0 |
| artifacts/qb_dependence_cfb/20260818T214601Z/predictions.parquet | college football; excluded | 0 |
| artifacts/rehearsal_lockday/sim/artifacts/margins/20260903T143251Z/predictions.parquet | {"not_margin_regression": 3, "qualifying_streams": 1, "served_configuration": 1} | 1 |
| artifacts/research/laneO/predictions.parquet | missing predicted_margin, train_max_gameday | 0 |
| artifacts/sunday_market_probability/20260920_fixed/predictions.parquet | missing gameday, predicted_margin, train_max_gameday | 0 |
| artifacts/totals_backtest/20260901T184010Z/predictions.parquet | missing predicted_margin, train_max_gameday | 0 |
| artifacts/totals_backtest_wave2/wp18run/positive_control/predictions.parquet | missing predicted_margin, train_max_gameday | 0 |
