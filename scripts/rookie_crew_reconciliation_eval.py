"""Score the reconciled rookie-crew rule predeclared in docs/rookie_crew_reconciliation.md."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

SEED = 20260821
SAMPLES = 20_000
ARM = "rookie_archive_era_floor"
ERAS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("2020_2025", (2020, 2021, 2022, 2023, 2024, 2025)),
    ("2020_2021", (2020, 2021)),
    ("2022_2023", (2022, 2023)),
    ("2024_2025", (2024, 2025)),
)


def load_script(repo: Path, name: str) -> Any:
    """Import a sibling script by file location."""
    spec = importlib.util.spec_from_file_location(name, repo / "scripts" / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def served_model_config(metadata: dict[str, Any], profile: str) -> dict[str, Any]:
    """The served evaluation's own model configuration with the profile swapped."""
    served = metadata["active_model_config"]
    return {
        "feature_profile": profile,
        "regressor": served["regressor"],
        "ridge_alpha": served["ridge_alpha"],
        "target": served["target"],
        "calibration_method": served["calibration_method"],
        "probability_method": served["probability_method"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--stamp", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    bat = load_script(repo, "officials_archive_battery_eval")
    mod18 = load_script(repo, "spread_regime_opener_eval")

    from nfl_ats.clv import opener_pick_evaluation
    from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
    from nfl_ats.officials_flag_features import ROOKIE_CREW_UNDERDOG_COLUMN
    from nfl_ats.pbp_coaching_traits import (
        build_odd_even_halves,
        build_season_to_season_pairs,
        paired_split_half_reliability,
    )
    from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact

    out = repo / "artifacts/rookie_crew_reconciliation" / args.stamp
    out.mkdir(parents=True, exist_ok=True)

    artifact = bat.served_artifact_id()
    served_dir = repo / "artifacts/opener_evaluation" / artifact
    served = pd.read_parquet(served_dir / "per_game.parquet")
    metadata = json.loads((served_dir / "metadata.json").read_text(encoding="utf-8"))

    schedule = bat.schedules()
    base_features = pd.read_parquet(bat.FEATURES)
    proxy = bat.flag_columns(schedule, close_proxy=True)[ARM]
    shipped = bat.flag_columns(schedule, close_proxy=False)[ARM]

    def score(features: pd.DataFrame, profile: str) -> pd.DataFrame:
        scored = opener_pick_evaluation(
            bat.MARKET_ROOT,
            features,
            active_model_config=served_model_config(metadata, profile),
            min_train_games=DEFAULT_MIN_TRAIN_GAMES,
        )
        return scored.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    with threadpool_limits(limits=1):
        baseline = score(base_features, "weak_stack")
        gate = bat.replay_gate(baseline, served, artifact)
        candidate = score(
            bat.attach(base_features, ROOKIE_CREW_UNDERDOG_COLUMN, proxy),
            bat.ARM_PROFILE[ARM],
        )
        opener_twin = score(
            bat.attach(base_features, ROOKIE_CREW_UNDERDOG_COLUMN, shipped),
            bat.ARM_PROFILE[ARM],
        )

    for name, frame in (("candidate", candidate), ("opener_twin", opener_twin)):
        if not frame["game_id"].equals(baseline["game_id"]):
            raise ValueError(f"{name} scored a different game set than production")

    truth = baseline.loc[:, ["game_id", "season", "week", "margin_vs_open"]]
    base_pick = baseline["pick_home_at_open_probability_rule"].to_numpy()
    cand_pick = candidate["pick_home_at_open_probability_rule"].to_numpy()
    twin_pick = opener_twin["pick_home_at_open_probability_rule"].to_numpy()

    card_frame = served.copy()
    order = candidate.set_index("game_id")
    card_frame["home_cover_probability_at_open"] = order.loc[
        card_frame["game_id"], "home_cover_probability_at_open"
    ].to_numpy()
    card_frame["pick_home_at_open_probability_rule"] = order.loc[
        card_frame["game_id"], "pick_home_at_open_probability_rule"
    ].to_numpy()
    card_frame["correct_at_open_probability_rule"] = (
        card_frame["pick_home_at_open_probability_rule"]
        .eq(card_frame["margin_vs_open"].gt(0))
        .astype(float)
        .where(card_frame["margin_vs_open"].ne(0))
    )
    card_dir = out / "opener_evaluation" / "rookie_crew_reconciled"
    card_dir.mkdir(parents=True, exist_ok=True)
    card_frame.to_parquet(card_dir / "per_game.parquet", index=False)
    stamp_sidecar(card_dir / "per_game.parquet")
    card_metadata = dict(metadata)
    card_metadata.update(
        research_arm="rookie_crew_reconciled",
        incumbent_model_id=metadata.get("active_model_id"),
        active_model_id="research_rookie_crew_reconciled",
        research_scope="Opener columns refit with the archive rookie flag; close columns retained.",
    )
    card_metadata["active_model_config"] = {
        **metadata["active_model_config"],
        "model_id": "research_rookie_crew_reconciled",
        "feature_profile": bat.ARM_PROFILE[ARM],
    }
    write_stamped_artifact(card_metadata, card_dir / "metadata.json")

    incumbent_four, incumbent_three = mod18.composed_picks(served, served_dir / "per_game.parquet")
    candidate_four, candidate_three = mod18.composed_picks(
        card_frame, card_dir / "per_game.parquet"
    )
    served_truth = served.loc[:, ["game_id", "season", "week", "margin_vs_open"]]

    def paired(frame: pd.DataFrame, cand: np.ndarray, base: np.ndarray) -> dict[str, Any]:
        return bat.paired_accuracy(frame, cand, base, samples=SAMPLES, seed=SEED)

    windows: dict[str, Any] = {}
    for label, seasons in ERAS:
        mask = truth["season"].isin(list(seasons)).to_numpy()
        served_mask = served_truth["season"].isin(list(seasons)).to_numpy()
        windows[label] = {
            "standalone_vs_raw_model": paired(
                truth.loc[mask].reset_index(drop=True), cand_pick[mask], base_pick[mask]
            ),
            "proxyline_vs_openerline_twin": paired(
                truth.loc[mask].reset_index(drop=True), cand_pick[mask], twin_pick[mask]
            ),
            "marginal_on_played_card": paired(
                served_truth.loc[served_mask].reset_index(drop=True),
                candidate_three[served_mask],
                incumbent_three[served_mask],
            ),
            "marginal_on_four_member_union": paired(
                served_truth.loc[served_mask].reset_index(drop=True),
                candidate_four[served_mask],
                incumbent_four[served_mask],
            ),
        }

    lines = bat.opener_line_frame(schedule, close_proxy=True)
    games = bat.tenure_table(bat.referee_game_table(include_archive=True, schedule=schedule))
    results = pd.read_parquet(bat.FEATURES, columns=["game_id", "result"])
    trait = (
        games.loc[:, ["game_id", "official_name", "season", "week", "prior_seasons_experience"]]
        .merge(lines, on="game_id", how="inner")
        .merge(results, on="game_id", how="inner")
        .dropna(subset=["tue_open_home_spread", "result"])
    )
    trait["margin_vs_line"] = trait["result"] - trait["tue_open_home_spread"]
    check = served.loc[:, ["game_id", "margin_vs_open"]].merge(
        trait.loc[:, ["game_id", "margin_vs_line"]], on="game_id", how="inner"
    )
    identity_gap = float((check["margin_vs_open"] - check["margin_vs_line"]).abs().max())
    trait = trait.loc[trait["tue_open_home_spread"].ne(0.0) & trait["margin_vs_line"].ne(0.0)]
    home_dog = trait["tue_open_home_spread"].lt(0.0)
    trait = trait.assign(
        underdog_covered=np.where(
            home_dog, trait["margin_vs_line"].gt(0.0), trait["margin_vs_line"].lt(0.0)
        ).astype(float)
    ).rename(columns={"official_name": "team"})

    season_trait = (
        trait.groupby(["team", "season"], as_index=False)
        .agg(underdog_covered=("underdog_covered", "mean"), games=("underdog_covered", "size"))
        .loc[lambda f: f["games"] >= 8]
    )
    season_pairs = build_season_to_season_pairs(season_trait, "underdog_covered")
    odd_even = build_odd_even_halves(trait, "underdog_covered", min_per_half=3)
    reliability = {
        "identity_gap_vs_served_margin": identity_gap,
        "trait_games": len(trait),
        "referee_seasons": len(season_trait),
        "season_to_season": paired_split_half_reliability(
            season_pairs,
            metric="referee_season_underdog_cover_rate",
            method="season-to-season pairs of the same referee, season-blocked bootstrap",
            seed=SEED,
            spearman_brown=False,
        ),
        "within_season_odd_even": paired_split_half_reliability(
            odd_even,
            metric="referee_season_underdog_cover_rate",
            method="odd/even week halves within a referee-season, season-blocked bootstrap",
            seed=SEED,
            spearman_brown=True,
        ),
    }

    flag_counts = (
        proxy.drop_duplicates("game_id")
        .merge(baseline.loc[:, ["game_id", "season"]], on="game_id", how="inner")
        .groupby("season")[ROOKIE_CREW_UNDERDOG_COLUMN]
        .apply(lambda s: int(s.ne(0.0).sum()))
        .to_dict()
    )

    result = {
        "predeclaration": "docs/rookie_crew_reconciliation.md",
        "arm": ARM,
        "served_artifact": artifact,
        "seed": SEED,
        "bootstrap_samples": SAMPLES,
        "replay_gate": gate,
        "flagged_games_by_season_graded_window": {str(k): v for k, v in flag_counts.items()},
        "windows": windows,
        "reliability": reliability,
    }
    write_stamped_artifact(bat.jsonable(result), out / "results.json")
    print(json.dumps(bat.jsonable(result), indent=1, default=str))


if __name__ == "__main__":
    main()
