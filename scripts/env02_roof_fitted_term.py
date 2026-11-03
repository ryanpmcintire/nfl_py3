from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
for sub in ("src", "scripts"):
    if str(REPO / sub) not in sys.path:
        sys.path.insert(0, str(REPO / sub))

import roof_state_screen as rs  # noqa: E402
from line_move_yardstick_paired_eval import (  # noqa: E402
    _design,
    _standardise,
    block_bootstrap,
    cell_stats,
)
from line_move_yardstick_paired_eval import _predict as _predict_features  # noqa: E402
from surface_switch_fitted_term import EXTENDED_POPULATION_PATH, loso_fit  # noqa: E402

from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, _fit_logit  # noqa: E402

AWARE = REPO / "data/raw/forecast_archive/pool_decision_2009_2025_aware"
NEW_TERM = "roof_open_probability_pit"
SEED = 20261103
DRAWS = 2000
EPS = 1e-12


def roof_open_probability(archive: Path, manifest: Path) -> pd.DataFrame:
    table = rs.build_prediction_table(forecast_archive=archive, forecast_manifest=manifest)
    table = table.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    table[NEW_TERM] = 0.0
    for season in sorted(table["season"].unique()):
        train = table.loc[table["season"] < season]
        if train.empty or train["is_open"].nunique() < 2:
            continue
        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit(rs._venue_features(train), train["is_open"].astype(int).to_numpy())
        mask = table["season"].eq(season)
        table.loc[mask, NEW_TERM] = model.predict_proba(rs._venue_features(table.loc[mask]))[:, 1]
    return table[["game_id", NEW_TERM]]


def insample_fit(population: pd.DataFrame, features: tuple[str, ...]) -> pd.Series:
    means, stds = _standardise(population, features)
    beta = _fit_logit(
        _design(population, features, means, stds),
        population["home_covered"].astype(float).to_numpy(),
        FIT_RIDGE,
    )
    return _predict_features(population, features, beta, means, stds)


def losses(p: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    clipped = p.clip(EPS, 1 - EPS)
    log_loss = -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
    return log_loss, (p - y) ** 2


def improvement_cells(frame: pd.DataFrame, base: str, plus: str, label: str) -> dict:
    y = frame["home_covered"].astype(float)
    base_ll, base_br = losses(frame[base], y)
    plus_ll, plus_br = losses(frame[plus], y)
    work = frame.copy()
    work["d_logloss"] = base_ll - plus_ll
    work["d_brier"] = base_br - plus_br
    work["d_accuracy"] = (frame[plus].ge(0.5).astype(float).eq(y).astype(float)) - (
        frame[base].ge(0.5).astype(float).eq(y).astype(float)
    )
    out = {"label": label, "games": len(work)}
    for column in ("d_logloss", "d_brier"):
        point, low, high, pp = block_bootstrap(work, ["season"], column, DRAWS, SEED)
        out[column] = {
            "mean_improvement": point,
            "season_block_low": low,
            "season_block_high": high,
            "probability_positive": pp,
        }
    out["d_accuracy_cell"] = cell_stats(label + "_accuracy", work, "d_accuracy", SEED, DRAWS)
    return out


def main() -> int:
    flags = roof_open_probability(AWARE / "forecasts.parquet", AWARE / "manifest.json")
    population = pd.read_parquet(EXTENDED_POPULATION_PATH)
    population["game_id"] = population["game_id"].astype(str)
    flags["game_id"] = flags["game_id"].astype(str)
    population = population.merge(flags, on="game_id", how="left")
    population[NEW_TERM] = population[NEW_TERM].fillna(0.0)
    plus_features = (*FIT_FEATURES, NEW_TERM)

    base_p, base_betas = loso_fit(population, FIT_FEATURES)
    plus_p, plus_betas = loso_fit(population, plus_features)
    model_only_p, _ = loso_fit(population, ("model_logit",))
    frame = population.copy()
    frame["market_p"] = 0.5
    frame["model_only_p"] = model_only_p
    frame["base_p"] = base_p
    frame["plus_p"] = plus_p
    frame = frame.loc[frame["base_p"].notna() & frame["plus_p"].notna()].reset_index(drop=True)
    frame["base_is_p"] = insample_fit(frame, FIT_FEATURES)
    frame["plus_is_p"] = insample_fit(frame, plus_features)

    y = frame["home_covered"].astype(float)
    levels = {}
    for name, column in (
        ("market_half", "market_p"),
        ("model_only", "model_only_p"),
        ("four_term_oos", "base_p"),
        ("plus_roof_oos", "plus_p"),
        ("four_term_insample", "base_is_p"),
        ("plus_roof_insample", "plus_is_p"),
    ):
        ll, br = losses(frame[column], y)
        levels[name] = {"log_loss": float(ll.mean()), "brier": float(br.mean())}

    oos = improvement_cells(frame, "base_p", "plus_p", "roof_plus_vs_four_term_oos")
    vs_market = improvement_cells(frame, "market_p", "plus_p", "roof_plus_vs_market_oos")
    vs_model_only = improvement_cells(
        frame, "model_only_p", "plus_p", "roof_plus_vs_model_only_oos"
    )
    insample = improvement_cells(frame, "base_is_p", "plus_is_p", "roof_plus_vs_four_term_insample")
    gap = {
        "logloss_insample_minus_oos": insample["d_logloss"]["mean_improvement"]
        - oos["d_logloss"]["mean_improvement"],
        "brier_insample_minus_oos": insample["d_brier"]["mean_improvement"]
        - oos["d_brier"]["mean_improvement"],
    }
    roof_betas = np.array([fold[NEW_TERM] for fold in plus_betas.values()])
    flagged = frame.loc[frame[NEW_TERM].gt(0.0)]
    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "archive": str(AWARE),
        "term": NEW_TERM,
        "games": len(frame),
        "seasons": sorted(int(s) for s in frame["season"].unique()),
        "term_nonzero_games": len(flagged),
        "looks": 4,
        "levels": levels,
        "oos_vs_four_term": oos,
        "oos_vs_market": vs_market,
        "oos_vs_model_only": vs_model_only,
        "insample_vs_four_term": insample,
        "gap": gap,
        "fold_roof_coefficients": {k: v[NEW_TERM] for k, v in plus_betas.items()},
        "fold_roof_beta_mean": float(roof_betas.mean()),
        "fold_roof_beta_sd": float(roof_betas.std(ddof=1)),
        "fold_roof_beta_positive": int((roof_betas > 0).sum()),
        "fold_count": len(roof_betas),
        "four_term_fold_betas_example": base_betas[sorted(base_betas)[0]],
    }
    out_dir = (
        REPO / "artifacts" / "env02_roof_fitted_term" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(out_dir)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("oos_vs_market", "oos_vs_model_only", "insample_vs_four_term")
            },
            indent=1,
            default=str,
        )
    )
    for key in ("oos_vs_market", "oos_vs_model_only", "insample_vs_four_term"):
        r = result[key]
        print(key, r["d_logloss"], r["d_brier"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
