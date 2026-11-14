import argparse
import csv
import glob
import json
from collections import Counter
from pathlib import Path

CUTOFF = "2026-11-03"
NO_BASE_UNITS = {
    "correlation",
    "epa_per_play",
    "elapsed_seconds_bias",
    "payout_first_pp",
    "rate_pp",
}

FAMILY_BASE = [
    ("eng50", "served_card", "scripts/eng50_online_stacker.py:91", "read"),
    ("eng51", "raw_model", "tests/scratch/eng51/lead66_unit2_open.py:65", "read"),
    ("env02", "served_card", "scripts/env02_roof_fitted_term.py:27", "read"),
    ("lead28", "raw_model", "scripts/per07_first_game_fitted_term.py:26", "read"),
    ("lead104", "not_applicable", "scripts/lead104_revision_path.py:31", "read"),
    ("lead106", "not_applicable", "scripts/lead106_injury_language.py:288", "inferred"),
    ("lead109", "not_applicable", "scripts/lead109_position_weights.py:15", "inferred"),
    ("lead110", "not_applicable", "scripts/lead110_absorption.py", "inferred"),
    ("lead75", "not_applicable", "scripts/lead75_unit1.py", "inferred"),
    ("mkt21", "served_card", "scripts/mkt21_line_path.py:114", "read"),
    ("mkt22", "not_applicable", "artifacts/mkt22/cells.json", "inferred"),
    ("mkt23", "served_card", "scripts/mkt23_unit2_term.py:55", "read"),
    ("mkt24", "raw_model", "scripts/mkt24_slot_residual.py:63", "read"),
    (
        "mkt25",
        "raw_model",
        "scripts/mkt25_opener_revision.py:103;scripts/mkt25_unit2.py:135",
        "read",
    ),
    ("mkt26", "served_card", "scripts/mkt26_move_dispersion.py:19", "read"),
    ("mkt27", "raw_model", "scripts/mkt27_total_lattice.py:251", "read"),
    ("mkt28", "served_card", "tests/scratch/mkt28/served.py:7", "read"),
    ("mod10", "raw_model", "scripts/mod10_graph_term.py:172", "inferred"),
    ("mod23", "raw_model", "scripts/mod23_unit1.py:157;scripts/mod23_unit10b.py:22", "read"),
    ("mod24", "raw_model", "scripts/mod24_u1.py:239;scripts/mod24_finer_grade.py:36", "read"),
    ("mod25", "raw_model", "scripts/mod25_pipeline.py:408", "read"),
    ("mod27", "served_card", "scripts/mod27_fenced_repairs_replay.py:200", "read"),
    ("mod28", "raw_model", "scripts/mod28_coasting_trait.py:216", "read"),
    ("mod29", "raw_model", "scripts/per07_first_game_fitted_term.py:26", "read"),
    ("mod31", "not_applicable", "scripts/mod31_wind_elasticity.py", "inferred"),
    ("mod32", "raw_model", "scripts/mod32_coach_aggression_shape.py:192", "read"),
    ("mod33", "raw_model", "scripts/mod33_prior_week_exits.py:41", "read"),
    ("mod34", "raw_model", "scripts/mod34_residualised_addone.py:21", "read"),
    ("mod35", "raw_model", "scripts/mod35_apparent_temp.py:54", "read"),
    ("mod36", "raw_model", "scripts/mod36_noise_floor.py:21", "read"),
    ("mod37", "raw_model", "scripts/mod37_step3.py:183", "read"),
    ("mod38", "not_applicable", "docs/lanes/mod-38-mod38-kicker-wind.md", "inferred"),
    ("mod39", "raw_model", "scripts/mod39_leverage.py:95", "read"),
    ("mod40", "raw_model", "scripts/mod40_ngs_fit.py:13", "read"),
    ("mod41_residual_load_u6", "served_card", "scripts/mod41_unit6_served_base.py:19", "read"),
    ("mod41_cumulative_workload_u2", "raw_model", "scripts/mod41_unit2_components.py:38", "read"),
    ("mod41", "raw_model", "scripts/per07_first_game_fitted_term.py:26", "read"),
    ("mod42", "raw_model", "scripts/mod42_ol_continuity.py:105", "read"),
    ("mod43", "raw_model", "scripts/mod43_qb_learning_curve.py:219", "read"),
    ("mod44", "raw_model", "scripts/mod44_coach_tenure.py:49", "read"),
    ("mod45", "raw_model", "scripts/mod45_tempo_spread.py:350", "read"),
    ("mod46", "raw_model", "scripts/mod46_favourite_skew.py:627", "read"),
    ("mod47_global", "served_card", "scripts/mod47_unit2_global_calibration.py:64", "read"),
    ("mod47", "raw_model", "scripts/mod47_short_week_slope.py:35", "read"),
    ("mod49", "served_card", "scripts/mod49_construction.py:144", "read"),
    ("mod50", "served_card", "scripts/mod50_early_carryover.py:68", "read"),
    ("mod51", "served_card", "tests/scratch/mod51/served_delta.py:19", "read"),
    ("per11", "not_applicable", "artifacts/per11_arrival_rampup/results.json", "inferred"),
    ("per12", "not_applicable", "scripts/per12_qb_return.py", "inferred"),
    ("per13", "not_applicable", "artifacts/per13_locked_seed_rest/cells.json", "inferred"),
    ("per14", "not_applicable", "artifacts/per14/firings_event_study.json", "inferred"),
    ("per15", "raw_model", "scripts/per15_return_ramp.py:224", "read"),
    ("pool", "not_applicable", "scripts/pool05_ownership_v2.py", "inferred"),
    ("sim07", "not_applicable", "artifacts/sim07/ftn_state_summary.json", "inferred"),
    ("sim08", "raw_model", "scripts/sim08_evaluate.py:299", "read"),
    ("sim09", "not_applicable", "scripts/sim09_shell.py", "inferred"),
    ("sim12", "not_applicable", "docs/lanes/sim-12-sim12-ingame-exits.md", "inferred"),
    ("sim13", "not_applicable", "artifacts/sim13/cells_unit2.csv", "inferred"),
    ("sim14", "not_applicable", "docs/lanes/sim-14-sim14-coach-aggression.md", "inferred"),
    ("sim15", "not_applicable", "artifacts/sim15/cells.csv", "inferred"),
    ("sim16", "raw_model", "scripts/sim16_attrition.py:140", "read"),
    ("sim17", "raw_model", "scripts/sim17_unit3_heldout.py:260", "read"),
    ("sim18", "raw_model", "scripts/sim18_unit2_multiplier.py:505", "read"),
    ("sky", "served_card", "scripts/sky04_sharpness.py:41", "read"),
    ("xlg10", "cfb_base", "scripts/xlg10_addone2.py", "inferred"),
    ("xlg11", "cfb_base", "scripts/xlg11_cfb_centered_u2.py:23", "read"),
    ("xlg12", "cfb_base", "scripts/xlg12_predict.py", "inferred"),
    ("xlg13", "not_applicable", "docs/lanes/xlg-13-xlg13-screen-transfer.md", "inferred"),
    ("xlg14", "cfb_base", "scripts/xlg14_family_screen.py:97", "read"),
    ("xlg15", "not_applicable", "artifacts/xlg15/report.json", "inferred"),
]


NEW_RULES = [
    ("eng49_forward_ledger_vs_card", "served_card", "scripts/forward_ledger.py:129", "read"),
    (
        "eng62_",
        "served_card",
        "scripts/eng62_reference_system_arms.py:413;scripts/eng62_tuesday_reference_oos.py:232",
        "read",
    ),
    ("flag_gate_evaluation", "served_card", "scripts/flag_gate_evaluation.py:545", "read"),
    ("flag_set_freeze", "served_card", "docs/lanes/flag-gate-evaluation-2026-11-14.md", "inferred"),
    ("flip_attribution_audit", "served_card", "scripts/flip_attribution_audit.py:278", "inferred"),
    ("lead100", "not_applicable", "scripts/lead100_analyze.py", "inferred"),
    ("lead101", "not_applicable", "scripts/lead101_delta_read.py", "inferred"),
    ("lead102", "not_applicable", "scripts/lead102_blinded_read.py", "inferred"),
    ("lead104_revision_path", "raw_model", "scripts/lead104_revision_path.py:188;201", "inferred"),
    (
        "lead106",
        "unknown",
        "scripts/lead106_forward_score.py names no probability column",
        "inferred",
    ),
    (
        "lead110_market_absorption",
        "served_card",
        "scripts/lead110_market_absorption.py:432",
        "read",
    ),
    (
        "lead99_forward_decision_read",
        "unknown",
        "scripts/lead99_forward_decision_read.py:116,176 reads recommendations probability",
        "inferred",
    ),
    ("mod22_qb_input_repair", "raw_model", "scripts/mod22_qb_input_repair.py:584", "read"),
    (
        "mod25_distillation_walkforward",
        "served_card",
        "scripts/mod25_wfj_served_freeze.py",
        "read",
    ),
    ("mod36_addone_noise_floor", "served_card", "scripts/mod36_addone_noise_floor.py:19", "read"),
    (
        "mod36_metric_power",
        "unknown",
        "scripts/mod36_metric_power.py:308 mixes served and intercept-only comparators",
        "inferred",
    ),
    ("mod39_playoff_leverage", "served_card", "scripts/mod39_freeze_levdiff_served.py:23", "read"),
    (
        "mod47_short_week_side_term",
        "served_card",
        "scripts/mod47_short_week_side_term.py:218",
        "read",
    ),
    ("mod52_", "served_card", "scripts/mod52_stability.py:87", "read"),
    (
        "mod57_uncorrected_error_model",
        "raw_model",
        "scripts/mod57_p2c_uncorrected_error_model.py",
        "inferred",
    ),
    (
        "mod57_availability_ceiling",
        "raw_model",
        "scripts/mod57_p2l_availability_ceiling.py",
        "inferred",
    ),
    (
        "mod57_how_wrong_is_the_line",
        "raw_model",
        "scripts/mod57_p2o_how_wrong_is_the_line.py",
        "inferred",
    ),
    (
        "opener_error_transfer",
        "unknown",
        "scripts/opener_error_transfer_eval.py grades its own fitted base",
        "inferred",
    ),
    ("per14", "not_applicable", "scripts/per14_mover_portability.py", "inferred"),
    (
        "per16_habit_availability_addone",
        "served_card",
        "scripts/per16_unit4_availability_addone.py:18",
        "read",
    ),
    (
        "sim30_margin_distribution_challenger",
        "raw_model",
        "scripts/sim30_margin_mass.py",
        "inferred",
    ),
    ("xlg09_unit6", "unknown", "scripts/xlg09_unit6.py grades its own fitted base", "inferred"),
    ("xlg_early_overreaction_cfb", "cfb_base", "scripts/xlg_early_overreaction_cfb.py", "inferred"),
    (
        "xlg_harness_power",
        "unknown",
        "docs/lanes/xlg-harness-power-2026-11-14.md mixes harnesses",
        "inferred",
    ),
    ("mod07_joint_stack", "served_card", "scripts/mod07_joint_stack.py:51", "read"),
    ("mod53_penalty_grid_edge", "raw_model", "scripts/mod53_penalty_grid.py:15", "inferred"),
    ("mod54_mirrored_rows", "raw_model", "scripts/mod54_mirrored_rows.py:15", "inferred"),
    (
        "mod57_confirmed_moves",
        "served_card",
        "scripts/mod57_p2j_confirmed_moves.py:12;31",
        "inferred",
    ),
    (
        "mod57_deadline_fair_line",
        "served_card",
        "scripts/mod57_u3_deadline_fair_line.py:247;265",
        "read",
    ),
    (
        "mod57_play_level_matchup_model",
        "served_card",
        "scripts/mod57_p2h_play_level_matchup_model.py:347",
        "inferred",
    ),
    (
        "mod57_quiet_game_slopes",
        "served_card",
        "scripts/mod57_u4_quiet_game_slopes.py:87",
        "inferred",
    ),
    (
        "mod57_returning_share_stack_term",
        "served_card",
        "scripts/mod57_p3g_returning_share_term.py BASE four-term",
        "inferred",
    ),
    (
        "mod57_offset_term_era_placebo",
        "served_card",
        "tests/scratch/mod57/p3b/run.py:54",
        "read",
    ),
    ("mod57_pool_source_book", "served_card", "tests/scratch/mod57/u8_run.py:26;28", "read"),
    (
        "mod57_signals_on_quiet_games",
        "unknown",
        "scripts/mod57_u4_quiet_game_slopes.py slope cells",
        "inferred",
    ),
    (
        "mod57_future_market_label",
        "unknown",
        "scripts/mod57_p4a_future_market_label.py accuracy comparator not read",
        "inferred",
    ),
    (
        "mod57_margin_error_vs_line",
        "unknown",
        "scripts/mod57_p2n_margin_error_vs_line.py:258 ats_points comparator not read",
        "inferred",
    ),
    ("mod57_team_change_gap_frame", "not_applicable", "scripts/mod57_p3_gap_frame.py:281", "read"),
    ("sim25_pool_stratum", "not_applicable", "scripts/sim25_pool_stratum.py", "inferred"),
    ("sim28_timeouts", "not_applicable", "scripts/sim28_timeouts.py", "inferred"),
    ("sim29_lineup", "not_applicable", "scripts/sim29_lineup_expectation.py", "inferred"),
]

NOT_APPLICABLE_UNITS = {
    "mod36_metric_power": {"rps_improvement"},
    "mod57_signals_on_quiet_games": {"slope_difference"},
    "mod57_future_market_label": {"slope_difference"},
    "mod57_margin_error_vs_line": {"mae_improvement", "slope_difference"},
    "lead110_market_absorption": {"slope_difference"},
    "mod22_qb_input_repair": {"mae_improvement"},
}

AUDIT_NOTE = "corrected 2026-11-14 from the label audit"
AUDIT_EXCLUDED_SHARDS = {"mkt28_served_base_retest.json"}
RAW_ONLY_SHARDS = {
    "base_model_recency_v1.json": "base_model_recency.py",
    "lead76_rule_prior.json": "lead76_rule_prior.py",
    "mkt24_slot_opener_residual.json": "mkt24_slot_residual.py",
    "mkt25_opener_revision.json": "mkt25_opener_revision.py",
    "mkt25_opener_revision_unit2.json": "mkt25_unit2.py",
    "mkt25_opener_revision_unit3.json": "mkt25_unit3.py",
    "mkt27_total_lattice.json": "mkt27_total_lattice.py",
    "mod22_qb_input_repair.json": "mod22_qb_input_repair.py",
    "mod24_finer_grade.json": "mod24_finer_grade.py",
    "mod24_pre2009_extension.json": "mod24_pre2009_extension.py",
    "mod42_ol_continuity.json": "mod42_ol_continuity.py",
    "mod43_qb_learning_curve.json": "mod43_qb_learning_curve.py",
    "mod44_coach_tenure.json": "mod44_coach_tenure.py",
    "mod47_short_week_slope.json": "mod47_short_week_slope.py",
    "per15_return_ramp.json": "per15_return_ramp.py",
    "line_move_target_v1.json": "line_move_target.py",
    "refresh_timing_policy.json": "refresh_timing_policy.py",
    "sharp_book_weighted_movement.json": "sharp_book_weighted_movement.py",
    "sim04_play_simulator.json": "sim04_loso.py",
    "sim30_margin_distribution_challenger.json": "sim30_margin_mass.py",
    "spread_size_calibration.json": "spread_size_calibration.py",
}
RAW_ONLY_FAMILIES = {
    "mod40": "mod40_ngs_fit.py",
    "mod41_cumulative_workload_u2": "mod41_unit2_components.py",
    "mod45": "mod45_tempo_spread.py",
    "mod46": "mod46_favourite_skew.py",
    "mod57_uncorrected_error_model": "mod57_p2c_uncorrected_error_model.py",
    "mod57_availability_ceiling": "mod57_p2l_availability_ceiling.py",
    "mod57_how_wrong_is_the_line": "mod57_p2o_how_wrong_is_the_line.py",
    "sim08": "sim08_evaluate.py",
}


def tag(family, name, units):
    match = next(
        (row for row in NEW_RULES + FAMILY_BASE if family.startswith(row[0])),
        None,
    )
    if match is None:
        if units in NO_BASE_UNITS:
            return "not_applicable", "unit carries no pick-probability base", "inferred"
        return "unknown", "no family rule and no script evidence", "unread"
    prefix, base, evidence, confidence = match
    lowered = name.lower()
    if units in NO_BASE_UNITS or units in NOT_APPLICABLE_UNITS.get(prefix, ()):
        return "not_applicable", evidence, confidence
    if "cfb" in lowered and base in {"raw_model", "served_card"}:
        return "cfb_base", evidence, "inferred"
    if family.startswith("mod47_global"):
        if "served" in lowered:
            return "served_card", evidence, confidence
        return "raw_model", "scripts/mod47_unit2_global_calibration.py:69", confidence
    return base, evidence, confidence


def audit_correction(shard, family, units, base):
    if base != "served_card" or units in NO_BASE_UNITS:
        return None
    if shard in AUDIT_EXCLUDED_SHARDS:
        return None
    if shard in RAW_ONLY_SHARDS:
        return RAW_ONLY_SHARDS[shard]
    for prefix, script in RAW_ONLY_FAMILIES.items():
        if family.startswith(prefix):
            return script
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="registry/weak_signals_base_tags.csv")
    args = parser.parse_args()
    rows = []
    for path in sorted(glob.glob("registry/weak_signals/*.json")):
        payload = json.loads(Path(path).read_text(encoding="utf8"))
        for name, signal in payload.get("signals", {}).items():
            if signal.get("recorded_at", "") < CUTOFF:
                continue
            base, evidence, confidence = tag(signal["family"], name, signal["effect_units"])
            note = ""
            script = audit_correction(
                Path(path).name, signal["family"], signal["effect_units"], base
            )
            if script:
                base, evidence, confidence = "raw_model", f"scripts/{script}", "read"
                note = AUDIT_NOTE
            rows.append(
                [name, signal["family"], signal["effect_units"], base, evidence, confidence, note]
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["signal", "family", "effect_units", "base", "evidence", "confidence", "note"]
        )
        writer.writerows(rows)
    print(len(rows), dict(Counter(r[3] for r in rows)), dict(Counter(r[5] for r in rows)))


main()
