"""Record the combined overlay-stack backtest result to the weak-signal registry.

Reads the JSON artifact produced by ``scripts/overlay_stack_backtest.py`` and
calls ``nfl_ats.cli.main`` in-process with ``weak-signals record`` -- the same
CLI path ``nfl-ats weak-signals record`` invokes -- so the registry's own
validators (``nfl_ats.weak_signals.validate_closure``) run exactly as they
would from the command line. Every number passed to the recorder (effect,
interval, probability_positive, sample_games, sample_blocks, season range) is
read straight out of the artifact JSON; nothing numeric is hand-typed here.

Only categorical/methodological fields are literal in this script (league,
effect_units, classification, name, description template) -- these are
judgment calls about how to file the result, not measurements that could
drift from the computed artifact.

The PRIMARY reading is the week-blocked bootstrap (107 blocks -- matches this
project's convention of listing week-blocked before season-blocked, and is
the finer-grained blocking; the combined-stack week-blocked interval crosses
zero). The season-blocked reading (only 6 blocks -- one per season 2020-2025)
is coarser and, in this run, happens to exclude zero entirely; it is recorded
in ``notes``, not used as the registry's ``interval``/``probability_positive``,
because six blocks is too few for that blocking scheme to be the primary
read and AGENTS.md's binding rule is that the case for closing a line of work
must clear the bar on the interval actually used, not on whichever blocking
looks more decisive.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/overlay_stack_backtest_record.py
    .\\.tools\\uv.exe run --no-sync python scripts/overlay_stack_backtest_record.py \\
        --artifact artifacts/overlay_stack_backtest/20260819T191534Z/result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl_ats.cli import main as cli_main

DEFAULT_ROOT = Path("artifacts/overlay_stack_backtest")
SIGNAL_NAME = "overlay_stack_combined_opener_2020_2025"


def latest_artifact(root: Path) -> Path:
    candidates = sorted(root.glob("*/result.json"))
    if not candidates:
        raise FileNotFoundError(f"No overlay_stack_backtest result.json found under {root}")
    return candidates[-1]


def build_description(payload: dict) -> str:
    flips = payload["overlay_flip_counts"]
    overlap = payload["overlap"]
    marginal = payload["marginal_contributions"]
    lines = [
        "Joint (OR-combined: flip if ANY overlay's independent condition fires "
        "against the SAME baseline pick) application of all six live "
        "pick-flipping ACTIVE_PROSPECTIVE overlay challengers atop the active "
        f"model's own opener-graded picks (production probability rule), "
        f"{payload['seasons'][0]}-{payload['seasons'][1]}, {payload['n_games']} REG games "
        f"({payload['n_scored_games']} non-push, {payload['n_pushes']} pushes excluded). "
        "Baseline accuracy "
        f"{payload['combined_stack']['week_blocked']['baseline_accuracy']['estimate'] * 100:.2f}%, "
        "combined-stack accuracy "
        f"{payload['combined_stack']['week_blocked']['combined_accuracy']['estimate'] * 100:.2f}%. "
        f"Trigger counts: coach_fade {flips['coach_fade_overlay']}, "
        f"injury_value_lost_tilt {flips['injury_value_lost_tilt_overlay']}, "
        f"division_revenge_tilt {flips['division_revenge_tilt_overlay']}, "
        f"backup_qb_fade {flips['backup_qb_fade_overlay']}, "
        f"surface_switch_tilt {flips['surface_switch_tilt_overlay']}, "
        f"spread_gap_zone_fade {flips['spread_gap_zone_fade_overlay']} "
        f"(sum of solo flips {overlap['sum_of_solo_flip_counts']}, union "
        f"{overlap['games_touched_by_at_least_one_overlay']} games, "
        f"{overlap['games_touched_by_multiple_overlays']} touched by 2+ overlays). "
        "Every overlay's flip equals the complement of the SAME unflipped baseline "
        "pick (verified programmatically), so overlap is redundant coverage, never "
        "a contradictory recommendation -- zero direction conflicts are possible by "
        "construction and zero were found. Per-overlay marginal (leave-one-out) "
        "week-blocked deltas: "
        + "; ".join(
            f"{name} {marginal[name]['week_blocked']['marginal_delta']['estimate'] * 100:+.2f}pts "
            f"P+{marginal[name]['week_blocked']['marginal_delta']['probability_positive']:.3f}"
            for name in sorted(marginal)
        )
        + ". Excludes mod07_weak_signal_stack (IS the baseline model itself, not an "
        "overlay on top of it) and best_pick_nomination_v2 (never changes "
        "home_cover_probability, only which already-picked game gets the Best-Pick "
        "marker)."
    ]
    return " ".join(lines)


def build_classification_evidence(payload: dict) -> str:
    week = payload["combined_stack"]["week_blocked"]["combined_minus_baseline"]
    season = payload["combined_stack"]["season_blocked"]["combined_minus_baseline"]
    return (
        "unresolved_below_power, not a terminal closure: the PRIMARY (week-blocked, "
        f"{payload['week_block_count']} blocks) interval "
        f"[{week['lower'] * 100:+.3f}, {week['upper'] * 100:+.3f}] crosses zero "
        f"(P+ {week['probability_positive']:.4f} that the combined stack beats the "
        "baseline), so wrong_sign_resolved is inadmissible per AGENTS.md ('the wrong "
        "sign is a lean, not a resolution' unless the whole interval sits below zero). "
        "The season-blocked reading "
        f"[{season['lower'] * 100:+.3f}, {season['upper'] * 100:+.3f}] (P+ "
        f"{season['probability_positive']:.4f}) DOES sit entirely below zero, but it is "
        f"built from only {payload['season_block_count']} blocks (one per season) and is "
        "reported here as a secondary, coarser read, not as grounds for a terminal "
        "verdict on its own. No positive control was run. This is also explicitly a "
        "diagnostic on already-looked-at windows (several of the six overlays were "
        "themselves screened or tuned on spans this archive re-touches), continuous "
        "evidence rather than a fresh confirmation, so it would not license a terminal "
        "closure even if the primary interval had excluded zero."
    )


def build_notes(payload: dict) -> str:
    season = payload["combined_stack"]["season_blocked"]["combined_minus_baseline"]
    return (
        f"Artifact: {payload.get('_artifact_path', '')}. Source opener-evaluation "
        f"archive: {payload['source_artifact']} (sha256 "
        f"{payload['source_artifact_sha256'][:16]}...). Schedule snapshot: "
        f"{payload['schedule_snapshot']}. Player feature table: "
        f"{payload['player_feature_table']}. "
        f"Season-blocked (secondary, {payload['season_block_count']} blocks): "
        f"{season['estimate'] * 100:+.3f} pts "
        f"[{season['lower'] * 100:+.3f}, {season['upper'] * 100:+.3f}] "
        f"P+ {season['probability_positive']:.4f} (excludes zero at this coarser "
        "blocking -- reported, not used as the registry interval). "
        f"{payload['predeclaration_note']} Grading rule: {payload['grading_rule']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)

    artifact_path = args.artifact or latest_artifact(DEFAULT_ROOT)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["_artifact_path"] = str(artifact_path)

    week_delta = payload["combined_stack"]["week_blocked"]["combined_minus_baseline"]

    record_argv = [
        "weak-signals",
        "record",
        "--name",
        SIGNAL_NAME,
        "--description",
        build_description(payload),
        "--source",
        str(artifact_path),
        "--effect",
        str(week_delta["estimate"] * 100.0),
        "--effect-units",
        "accuracy_points",
        "--classification",
        "unresolved_below_power",
        "--league",
        "nfl",
        "--season-start",
        str(payload["seasons"][0]),
        "--season-end",
        str(payload["seasons"][1]),
        "--interval-low",
        str(week_delta["lower"] * 100.0),
        "--interval-high",
        str(week_delta["upper"] * 100.0),
        "--probability-positive",
        str(week_delta["probability_positive"]),
        "--sample-games",
        str(payload["n_scored_games"]),
        "--sample-blocks",
        str(payload["week_block_count"]),
        "--classification-evidence",
        build_classification_evidence(payload),
        "--notes",
        build_notes(payload),
    ]
    if args.replace:
        record_argv.append("--replace")

    return cli_main(record_argv)


if __name__ == "__main__":
    raise SystemExit(main())
