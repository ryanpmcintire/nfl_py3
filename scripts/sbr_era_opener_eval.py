"""MEASURE-ONLY: era-stratified grading of the frozen PRODUCTION model at
SBR-derived openers, seasons 2011-2021.

Predeclared in ``docs/sbr_opener_evaluation.md`` BEFORE this script produced
any era-level number. Read that document for the full predeclaration
(blindness disclosure, era scheme, bootstrap spec, registry plan); this
docstring does not repeat it.

Reuses ``scripts/proxy_opener_replication.py``'s ``build_population`` and
``sbr_proxy_pick_evaluation`` **by import, unmodified** -- a single
weekly-refit walk-forward scoring pass over 2009-2021 REG games with an SBR
Open, so Eras A (2011-2014) and B (2015-2019) are provably the same scored
games as the already-recorded ``proxy_opener_production_rule_2009_2019``
registry entry (a re-slice, not a re-measurement); Era C (2020-2021, full
SBR-matched population) is the only genuinely new SBR-graded population here.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/sbr_era_opener_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from proxy_opener_replication import (  # noqa: E402
    FEATURE_PROFILE,
    REGRESSOR,
    RIDGE_ALPHA,
    build_population,
    per_season_table,
    sbr_proxy_pick_evaluation,
)

from nfl_ats.clv import week_blocked_bootstrap  # noqa: E402
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

# Full scoring-pass population range: wide enough that the walk-forward
# warm-up floor itself (not a manual cutoff) is what drops 2009-2010 -- the
# same mechanism docs/proxy_opener_replication.md already used for 2009-2019,
# extended two seasons to reach SBR's archive ceiling (2021-22 is the last
# populated season, docs/sbr_odds_archive.md).
POP_SEASON_START = 2009
POP_SEASON_END = 2021

# Era scheme, predeclared in docs/sbr_opener_evaluation.md section 4.
ERAS: tuple[tuple[str, int, int], ...] = (
    ("era_2011_2014", 2011, 2014),
    ("era_2015_2019", 2015, 2019),
    ("era_2020_2021", 2020, 2021),
)
POOLED_START, POOLED_END = 2011, 2021

# Task-specified seed for THIS evaluation (docs/sbr_opener_evaluation.md
# section 5 discloses this differs from the project's older standing
# opener-bootstrap seed 20260817 used elsewhere).
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819


def _vs_coin_flip_metric_fn(col: str) -> Any:
    def metric_fn(frame: pd.DataFrame) -> dict[str, float]:
        return {"accuracy_minus_50": float(frame[col].mean()) - 0.5}

    return metric_fn


def _bootstrap(
    frame: pd.DataFrame, col: str, block: str, samples: int, seed: int
) -> dict[str, float]:
    return (
        week_blocked_bootstrap(
            frame, _vs_coin_flip_metric_fn(col), block=block, samples=samples, seed=seed
        )
        .iloc[0]
        .to_dict()
    )


def _rule_block(
    subset: pd.DataFrame, col: str, n_seasons: int, samples: int, seed: int
) -> dict[str, Any]:
    return {
        "absolute_accuracy": float(subset[col].mean()),
        "week_blocked": _bootstrap(subset, col, "week", samples, seed),
        "season_blocked": (
            _bootstrap(subset, col, "season", samples, seed) if n_seasons >= 2 else None
        ),
    }


def slice_result(
    scored: pd.DataFrame, start: int, end: int, samples: int, seed: int
) -> dict[str, Any]:
    subset = scored.loc[scored["season"].between(start, end)]
    n_seasons = subset["season"].nunique()
    n_weeks = subset.groupby(["season", "week"]).ngroups
    return {
        "season_start": start,
        "season_end": end,
        "games": len(subset),
        "seasons_present": sorted(int(s) for s in subset["season"].unique()),
        "n_weeks": int(n_weeks),
        "sign_rule": _rule_block(subset, "correct_at_open_proxy", n_seasons, samples, seed),
        "probability_rule": _rule_block(
            subset, "correct_at_open_proxy_pr", n_seasons, samples, seed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=REPO / "data/processed/game_features_weak_stack.parquet"
    )
    parser.add_argument("--sbr-odds", type=Path, default=REPO / "data/processed/sbr_odds.parquet")
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    features = pd.read_parquet(args.features)
    sbr_odds = pd.read_parquet(args.sbr_odds)

    print(f"=== Population: REG games {POP_SEASON_START}-{POP_SEASON_END} with an SBR Open ===")
    population = build_population(features, sbr_odds, POP_SEASON_START, POP_SEASON_END)
    print(f"population games: {len(population)}")

    scored, skipped = sbr_proxy_pick_evaluation(
        features=features, population=population, min_train_games=args.min_train_games
    )
    print(f"scored games: {len(scored)}, skipped weeks: {len(skipped)}")

    era_results = {
        name: slice_result(scored, start, end, args.samples, args.seed) for name, start, end in ERAS
    }
    pooled_result = slice_result(scored, POOLED_START, POOLED_END, args.samples, args.seed)

    # Cross-check: era A + B pooled should reproduce the already-recorded
    # proxy_opener_production_rule_2009_2019 registry entry's 2011-2019
    # population exactly (same script logic, same games).
    ab_cross_check = slice_result(scored, 2011, 2019, args.samples, args.seed)

    per_season = per_season_table(scored, POP_SEASON_START, POP_SEASON_END)

    report = {
        "config": {
            "feature_profile": FEATURE_PROFILE,
            "regressor": REGRESSOR,
            "ridge_alpha": RIDGE_ALPHA,
            "min_train_games": args.min_train_games,
            "bootstrap_samples": args.samples,
            "bootstrap_seed": args.seed,
        },
        "population_games": len(population),
        "population_ambiguous_open_games": int(population["open_ambiguous"].sum()),
        "scored_games": len(scored),
        "scored_seasons": sorted(int(s) for s in scored["season"].unique()),
        "skipped_weeks": skipped,
        "per_season": per_season,
        "eras": era_results,
        "pooled_2011_2021": pooled_result,
        "cross_check_2011_2019": ab_cross_check,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "per_season"}, indent=2, default=str))

    out_dir = args.out_dir
    if out_dir is None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = REPO / "artifacts" / "sbr_era_opener_eval" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = dict(report)
    metadata["generated_at_utc"] = datetime.now(UTC).isoformat()
    metadata["features_path"] = str(args.features)
    metadata["sbr_odds_path"] = str(args.sbr_odds)
    configuration = {
        "command": "sbr-era-opener-eval",
        "features": str(args.features),
        "sbr_odds": str(args.sbr_odds),
        "min_train_games": args.min_train_games,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
    }
    metadata["provenance"] = artifact_provenance(configuration, args.features, project_root=REPO)
    write_experiment_artifact(
        out_dir,
        "summary.json",
        metadata,
        command="sbr-era-opener-eval",
        metrics=metadata,
        notes=(
            "Measure-only era-stratified grading of the frozen production model at "
            "SBR-derived openers, 2011-2021; Eras A/B re-slice the already-recorded "
            "proxy_opener_production_rule_2009_2019 population, Era C is new."
        ),
    )
    scored.to_parquet(out_dir / "scored.parquet")

    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
