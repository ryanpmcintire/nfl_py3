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


def tag(family, name, units):
    match = next((row for row in FAMILY_BASE if family.startswith(row[0])), None)
    if match is None:
        return "unread", "", "unread"
    _prefix, base, evidence, confidence = match
    lowered = name.lower()
    if units in NO_BASE_UNITS:
        return "not_applicable", evidence, confidence
    if "cfb" in lowered and base in {"raw_model", "served_card"}:
        return "cfb_base", evidence, "inferred"
    if family.startswith("mod47_global"):
        if "served" in lowered:
            return "served_card", evidence, confidence
        return "raw_model", "scripts/mod47_unit2_global_calibration.py:69", confidence
    return base, evidence, confidence


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
            rows.append(
                [name, signal["family"], signal["effect_units"], base, evidence, confidence]
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signal", "family", "effect_units", "base", "evidence", "confidence"])
        writer.writerows(rows)
    print(len(rows), dict(Counter(r[3] for r in rows)), dict(Counter(r[5] for r in rows)))


main()
