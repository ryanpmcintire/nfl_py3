"""PER-14 annual unit-prior opener screen and original missing-source inventory."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from on_production_opener_confirmation import null_distribution, summarize

from nfl_ats import margin
from nfl_ats.clv import (
    CLOSE_LABEL_PRIORITY,
    HISTORICAL_CAPTURE_KIND,
    build_pairing_table,
    close_reference_table,
    pick_correct,
)
from nfl_ats.constants import FEATURE_SETS
from nfl_ats.io import run_id
from nfl_ats.modeling import regular_season_rows
from nfl_ats.provenance import (
    artifact_provenance,
    sha256_file,
    stamp_sidecar,
    write_experiment_artifact,
    write_stamped_artifact,
)
from nfl_ats.unit_prior_features import UNIT_PRIOR_COLUMNS, attach_unit_prior_features

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/experiments/unit_prior_features"
PROFILE = "weak_stack_unit_prior_cx20"
SEASONS = tuple(range(2020, 2026))
SEED = 20260902


@contextmanager
def candidate_profile():
    """Append exactly two columns locally and restore all profile definitions."""
    names = (f"football_{PROFILE}", f"full_{PROFILE}")
    with (
        patch.dict(
            FEATURE_SETS,
            {
                names[0]: FEATURE_SETS["football_weak_stack"] + UNIT_PRIOR_COLUMNS,
                names[1]: FEATURE_SETS["full_weak_stack"] + UNIT_PRIOR_COLUMNS,
            },
        ),
        patch.dict(margin._MARGIN_PROFILE_FEATURE_SETS, {PROFILE: names}),
        patch.object(margin, "MARGIN_FEATURE_PROFILES", (*margin.MARGIN_FEATURE_PROFILES, PROFILE)),
    ):
        assert margin.margin_feature_columns("market_residual", PROFILE) == (
            *margin.margin_feature_columns("market_residual", "weak_stack"),
            *UNIT_PRIOR_COLUMNS,
        )
        yield


def oracle_features(games, ratings):
    """Diagnostic ONLY: join same-season finals without the availability check."""
    result = games.copy()
    for unit, column in zip(("OFF_OL", "OFF_SKILL"), UNIT_PRIOR_COLUMNS, strict=True):
        lookup = ratings.loc[ratings.unit.eq(unit)].set_index(["season", "team"])["rating"]
        values = {}
        for side in ("home", "away"):
            keys = pd.MultiIndex.from_arrays([games.season, games[f"{side}_team"]])
            values[side] = lookup.reindex(keys).to_numpy(float)
        result[column] = values["home"] - values["away"]
    return result


def run_folds(base, candidate, oracle, lines, probability_method):
    """One fit per held-out season; training and calibration precede season start."""
    rows, audit = [], []
    for season in SEASONS:
        eligible = candidate.loc[candidate.season.eq(season)].dropna(
            subset=list(UNIT_PRIOR_COLUMNS)
        )
        group = eligible.merge(lines, on="game_id", validate="one_to_one")
        group = group.loc[
            group.result.notna() & (group.result - group.tue_open_home_spread).ne(0)
        ].copy()
        if group.empty:
            continue
        cutoff = base.loc[base.season.eq(season), "gameday"].min()
        training_mask = base.season.lt(season) & base.gameday.lt(cutoff) & base.result.notna()
        group["spread_line"] = group.tue_open_home_spread
        scored = group[["game_id", "season", "week"]].copy()
        scored["margin_vs_open"] = group.result - group.tue_open_home_spread
        for label, source, profile in (
            ("baseline", base, "weak_stack"),
            ("candidate", candidate, PROFILE),
            ("oracle", oracle, PROFILE),
        ):
            training = source.loc[training_mask].copy()
            scoring = group.copy()
            if label == "oracle":
                scoring = scoring.drop(columns=list(UNIT_PRIOR_COLUMNS)).merge(
                    oracle[["game_id", *UNIT_PRIOR_COLUMNS]], on="game_id", validate="one_to_one"
                )
            if label != "baseline" and scoring[list(UNIT_PRIOR_COLUMNS)].isna().any().any():
                raise ValueError(f"{label} unavailable on the common paired population")
            model = margin.fit_margin_model(
                training,
                target="market_residual",
                model_name="ridge",
                ridge_alpha=10.0,
                feature_profile=profile,
            )
            prediction = model.predict(scoring, probability_method=probability_method)
            probability = prediction.home_cover_probability.to_numpy()
            scored[f"{label}_probability"] = probability
            scored[f"{label}_pick_home_pr"] = probability >= 0.5
            scored[f"{label}_correct_open_pr"] = pick_correct(
                scored[f"{label}_pick_home_pr"], scored.margin_vs_open
            )
            audit.append(
                {
                    "season": season,
                    "arm": label,
                    "training_rows": len(training),
                    "training_max_season": int(training.season.max()),
                    "training_max_gameday": str(training.gameday.max()),
                    "test_min_gameday": str(cutoff),
                    "paired_games": len(scored),
                    "distribution_rows": model.distribution_rows,
                }
            )
        rows.append(scored)
        print(f"held-out season={season} paired_games={len(scored)}", flush=True)
    if not rows:
        raise ValueError("No eligible paired opener games")
    return pd.concat(rows, ignore_index=True), audit


def screen(args):
    output = args.output or OUTPUT / run_id()
    output.mkdir(parents=True, exist_ok=False)
    active_path = ROOT / "artifacts/active_ats_model.json"
    active = json.loads(active_path.read_text())
    if (
        active["feature_profile"],
        active["regressor"],
        active["ridge_alpha"],
        active["method"],
        active["calibration_method"],
    ) != ("weak_stack", "ridge", 10.0, "market_residual", "none"):
        raise ValueError("Active recipe differs from the predeclared production baseline")
    if sha256_file(args.features) != active["feature_table_sha256"]:
        raise ValueError("Feature table differs from the active production table")
    ratings = pd.read_parquet(args.ratings)
    base = regular_season_rows(pd.read_parquet(args.features)).reset_index(drop=True)
    base["gameday"] = pd.to_datetime(base.gameday)
    # Conservative pregame boundary for reconstructed annual priors, not a
    # fabricated historical quote timestamp. S-1 ended months earlier.
    base["prediction_timestamp"] = pd.to_datetime(base.gameday, utc=True) - pd.Timedelta(days=7)
    candidate = attach_unit_prior_features(base, ratings)
    oracle = oracle_features(base, ratings)
    pairing = build_pairing_table(
        args.market_root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=base,
    )
    close = close_reference_table(pairing, base)
    lines = (
        pairing.loc[pairing.decision_label.eq("tue_open"), ["game_id", "home_spread"]]
        .rename(columns={"home_spread": "tue_open_home_spread"})
        .merge(close[["game_id"]], on="game_id", validate="one_to_one")
    )
    with candidate_profile():
        paired, audit = run_folds(base, candidate, oracle, lines, active["probability_method"])
    results = {}
    for label in ("candidate", "oracle"):
        result = summarize(
            paired, "baseline_correct_open_pr", f"{label}_correct_open_pr", 20_000, SEED
        )
        result["effect_accuracy_points"] = result["delta_accuracy"] * 100
        result["ci95_accuracy_points"] = [v * 100 for v in result["week_blocked_ci95"]]
        result["probability_positive"] = result["week_blocked_probability_positive"]
        result["per_season"] = [
            {
                "season": int(s),
                "games": len(g),
                "baseline_correct": int(g.baseline_correct_open_pr.sum()),
                "candidate_correct": int(g[f"{label}_correct_open_pr"].sum()),
                "effect_accuracy_points": float(
                    (g[f"{label}_correct_open_pr"] - g.baseline_correct_open_pr).mean() * 100
                ),
            }
            for s, g in paired.groupby("season")
        ]
        null_frame = paired.copy()
        if label == "oracle":
            null_frame["candidate_pick_home_pr"] = paired.oracle_pick_home_pr
        result["permutation_null"] = null_distribution(
            null_frame, probability_rule=True, permutations=1000, seed=SEED
        )
        results[label] = result
    config = {
        "predeclaration": "docs/unit_prior_features.md",
        "held_out_seasons": list(SEASONS),
        "bootstrap_samples": 20_000,
        "permutations": 1000,
        "seed": SEED,
        "probability_method": active["probability_method"],
        "model_id": active["model_id"],
        "ratings_sha256": sha256_file(args.ratings),
        "active_sha256": sha256_file(active_path),
        "ratings_path": str(args.ratings),
        "profile": "weak_stack + exactly OL/skill differences",
    }
    payload = {
        "configuration": config,
        "results": results,
        "fold_audit": audit,
        "continuity": {
            "unknown_games": len(paired),
            "stayed_stayed": 0,
            "any_changed": 0,
            "reason": (
                "weekly roster source has no publication timestamp; "
                "no dated pregame membership supplied"
            ),
        },
        "limitations": [
            "reused historical windows; exploratory selection discount",
            "only spread replaced by opener; inherited close-era covariates",
            "season-held-out refits, distinct from weekly-refit published baseline",
            "source availability reconstructed as March 1; not archived historical vintages",
            "unit priors measure residual player quality, not a measured continuity effect",
        ],
        "provenance": artifact_provenance(config, args.features, project_root=ROOT),
    }
    for name, frame in (("paired_predictions", paired), ("candidate_features", candidate)):
        path = output / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        stamp_sidecar(
            path, {"sha256": sha256_file(path), "configuration": config}, project_root=ROOT
        )
    write_experiment_artifact(
        output,
        "results.json",
        payload,
        command="unit-prior-features",
        metrics=results,
        project_root=ROOT,
        registry_root=output / "experiment_registry",
        notes="Exploratory season-held-out PER-14; curated effect recorded separately by CLI.",
    )
    print(json.dumps(payload, indent=2))
    print(f"wrote {output / 'results.json'}")


def inventory_main() -> None:
    files = sorted((ROOT / "artifacts/unit_apm").rglob("*"))
    inventory = []
    for path in files:
        if not path.is_file():
            continue
        entry: dict[str, object] = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            entry["keys"] = sorted(data)
            entry["source_seasons"] = (
                data.get("provenance", {}).get("configuration", {}).get("seasons", [])
            )
            entry["unit_summary_keys"] = {
                unit: sorted(summary) for unit, summary in data.get("units", {}).items()
            }
        inventory.append(entry)
    # Fail closed if a producer adds a new layout: this audited fallback must
    # not silently label a newly available coefficient table as missing.
    expected_keys = {"elapsed_seconds", "provenance", "units", "unmapped_positions"}
    expected_summary = {
        "min_plays_per_half",
        "n",
        "spearman_brown_pearson",
        "split_half_pearson",
        "split_half_spearman",
        "unit",
    }
    for entry in inventory:
        if set(entry.get("keys", [])) != expected_keys:
            raise ValueError(f"New unit artifact layout; inspect before screening: {entry['path']}")
        for keys in entry["unit_summary_keys"].values():
            if set(keys) != expected_summary:
                raise ValueError("New unit summary fields; inspect for annual ratings")
    payload = {
        "status": "missing_season_final_unit_ratings",
        "inventory": inventory,
        "files": len(inventory),
        "annual_rating_rows": 0,
        "annual_rating_seasons": [],
        "membership_rows_in_unit_artifacts": 0,
        "paired_games_scored": 0,
        "effect_accuracy_points": None,
        "probability_positive": None,
        "missing": [
            "season/team/OFF_OL final rating and availability timestamp",
            "season/team/OFF_SKILL final rating and availability timestamp",
            "annual odd/even coefficient estimates for team-unit reliability",
            "unit personnel membership and dated offseason continuity in these artifacts",
        ],
        "scope": "artifacts/unit_apm; raw roster/participation reconstruction not attempted",
        "predeclaration": "docs/unit_prior_features.md",
        "registry_measurement": "none: no measured accuracy-point effect",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_stamped_artifact(payload, OUTPUT / "inventory.json", project_root=ROOT)
    print(json.dumps(payload, indent=2))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument(
        "--features", type=Path, default=ROOT / "data/processed/game_features_weak_stack.parquet"
    )
    parser.add_argument("--market-root", type=Path, default=ROOT / "data/market/raw")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.ratings is None:
        inventory_main()
    else:
        screen(args)


if __name__ == "__main__":
    main()
