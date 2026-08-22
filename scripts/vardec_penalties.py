"""Variance decomposition of penalties in NFL game margins, 2009-2025 REG.

Question: how much of the variance in the final margin do penalties explain,
and what fraction of that is forecastable ex ante?

Constructs:

- Penalty-EPA swing (per team-game): sum of ``epa`` over every regular-season
  snapshot row where ``penalty == 1``, ``play == 1``, ``epa`` is present, and
  ``posteam == team``. This is the net expected-points swing penalties produced
  on that team's offensive snaps -- its own offensive fouls (negative EPA) plus
  defensive fouls by the opponent (positive EPA). It is deliberately the SAME
  mixed signal the ``penalty_discipline`` reconstruction documents: the narrowed
  local snapshot carries no ``penalty_team``, so the committing side is not
  knowable play-by-play; the benefiting side (the posteam) is.

- Exclusions, explicit: rows with ``penalty == 1`` and ``play == 0`` are dropped.
  Those are the no-play fouls -- dead-ball fouls and offsetting double fouls
  replaying the down -- and they carry no EPA in nflverse, so they cannot enter
  an EPA swing by construction. Declined fouls CANNOT be separated from accepted
  ones in the narrowed snapshot (no description or accepted flag survives
  ingestion), so they stay inside the kept rows with the actual play's EPA;
  this inflates swing magnitude slightly and is reported, not hidden.

- Counterfactuals: (a) league-mean swap -- every game's realized penalty
  differential replaced by zero (league-average penalty behavior in the same
  situations nets to nothing between the two sides); the margin moves by the
  realized differential, so delta-sd equals sd(differential). (b) season-lagged
  swap -- each team-game swing predicted from its PRIOR-season swing-per-snap
  rate times this game's snaps; the differential is swapped for the predicted
  one and the resulting margin movement measured.

- Type mix: the widened officials snapshot
  (data/raw/officials/20260820T112517Z/game_penalty_types.parquet,
  2015-2025 only, built by scripts/fetch_penalty_type_snapshot.py) supplies
  home/away-attributed counts by penalty_type; team-season Offensive Holding
  and Defensive Holding rates per snap get year-over-year reliabilities as
  forecastability anchors next to the known crew anchors (+0.3226 OH,
  +0.2702 DH, docs/penalty_crew_tendencies.md section 3) and the known team
  overall-rate anchor (~+0.26, reproduced here rather than trusted).

Uncertainty: week-blocked bootstrap (block id = season*100 + week, the repo
convention), 2,000 resamples, joint resampling of every statistic's blocks.

Writes artifacts/vardec_pen/results.json only. Research measurement, not a
feature family; nothing in src/ changes, so no leakage-regression test is due.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

PBP_RAW_ROOT = REPO / "data/pbp/raw"
FEATURES_PATH = REPO / "data/processed/game_features.parquet"
TYPE_SNAPSHOT_PATH = REPO / "data/raw/officials/20260820T112517Z/game_penalty_types.parquet"
TEAM_SNAPSHOT_PATH = REPO / "data/raw/officials/20260819T190537Z/game_penalties.parquet"
OUT_DIR = REPO / "artifacts/vardec_pen"

BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20260822

KNOWN_TEAM_RATE_RELIABILITY = 0.261
KNOWN_CREW_OH_RELIABILITY = 0.3226
KNOWN_CREW_DH_RELIABILITY = 0.2702


def _canonical(team: pd.Series) -> pd.Series:
    return team.map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def load_and_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    snapshot = latest_pbp_snapshot(PBP_RAW_ROOT)
    plays = load_pbp_snapshot(snapshot, include_postseason=False)
    plays["posteam"] = _canonical(plays["posteam"].astype("string").fillna(""))
    penalty_flag = pd.to_numeric(plays["penalty"], errors="coerce").fillna(0.0)
    play_flag = pd.to_numeric(plays["play"], errors="coerce").fillna(0.0)
    epa = pd.to_numeric(plays["epa"], errors="coerce")
    kept_mask = (penalty_flag == 1.0) & (play_flag == 1.0) & epa.notna() & plays["posteam"].ne("")
    audit = {
        "snapshot_id": snapshot.snapshot_id,
        "total_reg_plays": len(plays),
        "penalty_flag_rows": int((penalty_flag == 1.0).sum()),
        "excluded_no_play_penalties": int(((penalty_flag == 1.0) & (play_flag != 1.0)).sum()),
        "excluded_play_penalty_missing_epa_or_posteam": int(
            ((penalty_flag == 1.0) & (play_flag == 1.0) & ~kept_mask).sum()
        ),
        "kept_penalty_epa_rows": int(kept_mask.sum()),
    }
    kept = plays.loc[kept_mask].copy()
    kept["epa_f"] = epa.loc[kept_mask]
    kept["yards_f"] = pd.to_numeric(plays.loc[kept_mask, "penalty_yards"], errors="coerce").fillna(
        0.0
    )
    return plays, kept, audit


def build_team_game_table(plays: pd.DataFrame, kept: pd.DataFrame) -> pd.DataFrame:
    snaps = (
        plays.loc[plays["posteam"].ne("")]
        .groupby(["game_id", "season", "week", "posteam"], sort=False)
        .agg(snaps=("play_id", "size"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    agg = (
        kept.groupby(["game_id", "season", "week", "posteam"], sort=False)
        .agg(
            swing=("epa_f", "sum"),
            pen_count=("epa_f", "size"),
            pen_yards=("yards_f", "sum"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    table = snaps.merge(agg, on=["game_id", "season", "week", "team"], how="left")
    table[["swing", "pen_count", "pen_yards"]] = table[["swing", "pen_count", "pen_yards"]].fillna(
        0.0
    )
    return table


def build_game_frame(table: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    reg = features.loc[features["game_type"] == "REG"].copy()
    reg["home_team"] = _canonical(reg["home_team"])
    reg["away_team"] = _canonical(reg["away_team"])
    reg["margin"] = reg["home_score"] - reg["away_score"]
    reg = reg.loc[
        reg["margin"].notna(),
        ["game_id", "season", "week", "home_team", "away_team", "margin", "ats_margin"],
    ].reset_index(drop=True)

    value_cols = ["swing", "pen_count", "pen_yards", "snaps", "swing_pred"]
    home_side = table.rename(columns={"team": "home_team", **{c: f"{c}_home" for c in value_cols}})
    away_side = table.rename(columns={"team": "away_team", **{c: f"{c}_away" for c in value_cols}})
    out = reg.merge(home_side.drop(columns=["season", "week"]), on=["game_id", "home_team"])
    out = out.merge(away_side.drop(columns=["season", "week"]), on=["game_id", "away_team"])
    if len(out) != len(reg):
        raise RuntimeError("home/away join lost games")
    out["swing_diff"] = out["swing_home"] - out["swing_away"]
    out["swing_pred_diff"] = out["swing_pred_home"] - out["swing_pred_away"]
    out["count_diff"] = out["pen_count_home"] - out["pen_count_away"]
    out["yards_diff"] = out["pen_yards_home"] - out["pen_yards_away"]
    out["week_block"] = out["season"].astype(int) * 100 + out["week"].astype(int)
    return out


def year_over_year_reliability(rate: pd.DataFrame) -> tuple[float, int]:
    ordered = rate.sort_values(["team", "season"]).copy()
    ordered["next_rate"] = ordered.groupby("team")["rate"].shift(-1)
    ordered["next_season"] = ordered.groupby("team")["season"].shift(-1)
    pairs = ordered.loc[ordered["next_season"] == ordered["season"] + 1]
    return float(pairs["rate"].corr(pairs["next_rate"])), len(pairs)


def overall_rate_reliability(plays: pd.DataFrame) -> tuple[pd.DataFrame, float, int]:
    sub = plays.loc[plays["posteam"].ne("")].copy()
    sub["penalty_num"] = pd.to_numeric(sub["penalty"], errors="coerce").fillna(0.0)
    grouped = sub.groupby(["season", "posteam"], sort=False).agg(
        plays_n=("penalty_num", "size"), penalties=("penalty_num", "sum")
    )
    grouped["rate"] = grouped["penalties"] / grouped["plays_n"]
    rate = grouped.reset_index().rename(columns={"posteam": "team"})
    reliability, n_pairs = year_over_year_reliability(rate)
    return rate, reliability, n_pairs


def add_lagged_predictions(table: pd.DataFrame) -> pd.DataFrame:
    season = (
        table.groupby(["team", "season"], sort=False)
        .agg(total=("swing", "sum"), snaps=("snaps", "sum"))
        .reset_index()
    )
    season["rate"] = season["total"] / season["snaps"]
    ordered = season.sort_values(["team", "season"]).copy()
    ordered["prev_rate"] = ordered.groupby("team")["rate"].shift(1)
    ordered["prev_season"] = ordered.groupby("team")["season"].shift(1)
    lagged = ordered.loc[ordered["season"] - ordered["prev_season"] == 1][
        ["team", "season", "prev_rate"]
    ]
    out = table.merge(lagged, on=["team", "season"], how="left")
    out["swing_pred"] = out["prev_rate"] * out["snaps"]
    return out


class BlockBootstrapper:
    """Joint week-blocked resampling of several game-level statistics."""

    def __init__(self, blocks: np.ndarray) -> None:
        self._blocks, self._inverse = np.unique(blocks, return_inverse=True)
        self._inverse = np.asarray(self._inverse).reshape(-1)
        self._n_blocks = len(self._blocks)

    def _block_sums(self, values: np.ndarray) -> np.ndarray:
        return np.bincount(self._inverse, weights=values, minlength=self._n_blocks)

    def run(
        self,
        series: dict[str, np.ndarray],
        statistic: Any,
        samples: int = BOOTSTRAP_SAMPLES,
        seed: int = BOOTSTRAP_SEED,
    ) -> np.ndarray:
        sums = {name: self._block_sums(np.asarray(values)) for name, values in series.items()}
        counts = np.bincount(self._inverse, minlength=self._n_blocks).astype(np.float64)
        rng = np.random.default_rng(seed)
        drawn = rng.multinomial(
            self._n_blocks, np.full(self._n_blocks, 1.0 / self._n_blocks), size=samples
        ).astype(np.float64)
        out = np.empty(samples, dtype=np.float64)
        for i in range(samples):
            weights = drawn[i]
            scaled = {name: weights @ values for name, values in sums.items()}
            scaled["_n"] = weights @ counts
            out[i] = statistic(scaled)
        return out


def variance_share(scaled: dict[str, float]) -> float:
    n = scaled["_n"]
    var_m = scaled["m2"] / n - (scaled["m"] / n) ** 2
    var_d = scaled["d2"] / n - (scaled["d"] / n) ** 2
    if var_m <= 0:
        return float("nan")
    return float(var_d / var_m)


def squared_correlation(scaled: dict[str, float]) -> float:
    n = scaled["_n"]
    ex = scaled["d"] / n
    ey = scaled["m"] / n
    vx = scaled["d2"] / n - ex**2
    vy = scaled["m2"] / n - ey**2
    cxy = scaled["dm"] / n - ex * ey
    denom = vx * vy
    if denom <= 0:
        return float("nan")
    return float((cxy * cxy) / denom)


def make_sd_statistic(residual_name: str) -> Any:
    def statistic(scaled: dict[str, float]) -> float:
        n = scaled["_n"]
        mean = scaled[residual_name] / n
        var = scaled[f"{residual_name}2"] / n - mean**2
        return float(np.sqrt(max(var, 0.0)))

    return statistic


def interval(draws: np.ndarray, confidence: float = 0.95) -> dict[str, float]:
    tail = (1.0 - confidence) / 2.0
    return {
        "point": float(np.mean(draws)),
        "lower": float(np.quantile(draws, tail)),
        "upper": float(np.quantile(draws, 1.0 - tail)),
        "probability_positive": float(np.mean(draws > 0.0)),
        "samples": len(draws),
    }


def type_rate_reliabilities(
    plays: pd.DataFrame,
) -> dict[str, Any]:
    types = pd.read_parquet(TYPE_SNAPSHOT_PATH)
    types = types.loc[types["season_type"] == "REG"].copy()
    attribution = pd.read_parquet(TEAM_SNAPSHOT_PATH)
    attribution = attribution.loc[attribution["season_type"] == "REG"][
        ["game_id", "home_team", "away_team"]
    ].drop_duplicates("game_id")
    attribution["home_team"] = _canonical(attribution["home_team"])
    attribution["away_team"] = _canonical(attribution["away_team"])
    types = types.merge(attribution, on="game_id", validate="many_to_one")

    snap_sub = plays.loc[plays["posteam"].ne("")]
    snaps = snap_sub.groupby(["season", "posteam"], sort=False).agg(snaps=("play_id", "size"))

    results: dict[str, Any] = {}
    wanted = {"offensive_holding": "Offensive Holding", "defensive_holding": "Defensive Holding"}
    available = set(types["penalty_type"].unique())
    for key, label in wanted.items():
        if label not in available:
            results[key] = {"present": False}
            continue
        subset = types.loc[types["penalty_type"] == label]
        home_rows = subset[["season", "home_team", "penalties_on_home"]].rename(
            columns={"home_team": "team", "penalties_on_home": "calls"}
        )
        away_rows = subset[["season", "away_team", "penalties_on_away"]].rename(
            columns={"away_team": "team", "penalties_on_away": "calls"}
        )
        long = pd.concat([home_rows, away_rows], ignore_index=True)
        totals = long.groupby(["season", "team"], sort=False)["calls"].sum().reset_index()
        totals = totals.merge(snaps.reset_index().rename(columns={"posteam": "team"}))
        totals["rate"] = totals["calls"] / totals["snaps"]
        reliability, n_pairs = year_over_year_reliability(totals)
        results[key] = {
            "label": label,
            "reliability": reliability,
            "pairs": n_pairs,
            "known_crew_anchor": KNOWN_CREW_OH_RELIABILITY
            if key == "offensive_holding"
            else KNOWN_CREW_DH_RELIABILITY,
        }
    results["games_in_type_snapshot_reg"] = int(types["game_id"].nunique())
    results["seasons_in_type_snapshot"] = sorted(int(s) for s in types["season"].unique())
    return results


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    print("=== Step 1: load snapshot, exclusion audit ===")
    plays, kept, audit = load_and_audit()
    results["exclusion_audit"] = audit
    for name, value in audit.items():
        print(f"{name}: {value}")

    print("\n=== Step 2: team-game aggregates and game frame ===")
    table = build_team_game_table(plays, kept)
    table = add_lagged_predictions(table)
    features = pd.read_parquet(FEATURES_PATH)
    games = build_game_frame(table, features)
    ats_games = games.loc[games["ats_margin"].notna()].copy()
    print(f"games={len(games)} (with ATS margin: {len(ats_games)})")

    print("\n=== Step 3: headline decomposition ===")
    d = games["swing_diff"].to_numpy(dtype=np.float64)
    m = games["margin"].to_numpy(dtype=np.float64)
    headline = {
        "sd_swing_diff_pts": float(np.std(d, ddof=1)),
        "mean_abs_swing_diff_pts": float(np.mean(np.abs(d))),
        "sd_margin_pts": float(np.std(m, ddof=1)),
        "variance_share_gross": float(np.var(d, ddof=1) / np.var(m, ddof=1)),
        "correlation_with_margin": float(np.corrcoef(d, m)[0, 1]),
    }
    headline["variance_share_regression"] = headline["correlation_with_margin"] ** 2
    results["headline"] = headline
    print(json.dumps(headline, indent=2))

    bootstrapper = BlockBootstrapper(games["week_block"].to_numpy())
    base_series = {"d": d, "m": m, "d2": d * d, "m2": m * m, "dm": d * m}
    share_draws = bootstrapper.run(base_series, variance_share)
    r2_draws = bootstrapper.run(base_series, squared_correlation)
    results["bootstrap_variance_share_gross"] = interval(share_draws)
    results["bootstrap_r2_margin_on_swing"] = interval(r2_draws)

    league_mean_delta_sd = float(np.std(d, ddof=1))
    results["league_mean_swap"] = {
        "delta_sd_pts": league_mean_delta_sd,
        "note": "swap realized differential to zero; delta equals the realized differential",
    }
    lm_series = dict(base_series)
    lm_series["e"] = d
    lm_series["e2"] = d * d
    results["bootstrap_league_mean_swap_delta_sd"] = interval(
        bootstrapper.run(lm_series, make_sd_statistic("e"))
    )

    print("\n=== Step 4: season-lagged counterfactual swap ===")
    covered = games["swing_pred_diff"].notna().to_numpy()
    dpred = games["swing_pred_diff"].fillna(0.0).to_numpy(dtype=np.float64)
    resid = np.where(covered, d - np.where(covered, dpred, 0.0), np.nan)
    lag_corr = float(np.corrcoef(d[covered], dpred[covered])[0, 1])
    lag = {
        "covered_games": int(covered.sum()),
        "correlation_lagged_prediction": lag_corr,
        "r_squared_lagged_prediction": float(lag_corr**2),
        "delta_sd_lagged_swap_pts": float(np.nanstd(resid, ddof=1)),
        "forecastable_fraction_of_swing_variance": float(
            1.0 - np.nanvar(resid, ddof=1) / np.var(d[covered], ddof=1)
        ),
    }
    lag["forecastable_share_of_total_margin_variance"] = (
        lag["forecastable_fraction_of_swing_variance"] * headline["variance_share_gross"]
    )
    dc = d[covered]
    pc = dpred[covered]
    slope = float(np.cov(dc, pc, ddof=1)[0, 1] / np.var(pc, ddof=1))
    intercept = float(np.mean(dc) - slope * np.mean(pc))
    fitted_resid = dc - (intercept + slope * pc)
    lag["fitted_slope"] = slope
    lag["delta_sd_lagged_swap_fitted_pts"] = float(np.std(fitted_resid, ddof=1))
    lag["forecastable_fraction_of_swing_variance_in_sample_ols"] = float(
        1.0 - np.var(fitted_resid, ddof=1) / np.var(dc, ddof=1)
    )
    lag["scale_note"] = (
        "the raw-rate swap's residual EXCEEDS the realized swing sd because unshrunk "
        "lagged rates are mis-scaled for game-level use (optimal slope "
        f"{slope:.3f}, regression-to-mean); the fitted-swap figures above are the "
        "honest forecastable fraction and carry in-sample OLS optimism"
    )
    results["lagged_swap"] = lag
    print(json.dumps(lag, indent=2))

    lag_series = {
        "e": np.nan_to_num(resid),
        "e2": np.nan_to_num(resid) ** 2,
    }
    results["bootstrap_lagged_swap_delta_sd"] = interval(
        bootstrapper.run({**base_series, **lag_series}, make_sd_statistic("e"))
    )

    print("\n=== Step 5: count and yards differentials (secondary) ===")
    secondary: dict[str, Any] = {}
    for name, col in (("count_diff", "count_diff"), ("yards_diff", "yards_diff")):
        x = games[col].to_numpy(dtype=np.float64)
        r = float(np.corrcoef(x, m)[0, 1])
        entry = {
            "sd": float(np.std(x, ddof=1)),
            "correlation_with_margin": r,
            "r2": r * r,
        }
        series = {
            "d": x,
            "d2": x * x,
            "m": m,
            "m2": m * m,
            "dm": x * m,
        }
        entry["bootstrap_r2"] = interval(bootstrapper.run(series, squared_correlation))
        secondary[name] = entry
        print(name, json.dumps(entry))
    results["secondary_differentials"] = secondary

    print("\n=== Step 6: ATS-margin variant ===")
    da = ats_games["swing_diff"].to_numpy(dtype=np.float64)
    ma = ats_games["ats_margin"].to_numpy(dtype=np.float64)
    ats = {
        "games": len(ats_games),
        "correlation_with_ats_margin": float(np.corrcoef(da, ma)[0, 1]),
        "variance_share_gross_vs_ats": float(np.var(da, ddof=1) / np.var(ma, ddof=1)),
    }
    ats["r2_vs_ats"] = ats["correlation_with_ats_margin"] ** 2
    results["ats_variant"] = ats
    print(json.dumps(ats, indent=2))

    print("\n=== Step 7: reliability anchors ===")
    _, team_rel, team_pairs = overall_rate_reliability(plays)
    swing_season = table.groupby(["team", "season"], sort=False).agg(
        total=("swing", "sum"), snaps=("snaps", "sum")
    )
    swing_season["rate"] = swing_season["total"] / swing_season["snaps"]
    swing_rate = swing_season.reset_index()
    swing_rel, swing_pairs = year_over_year_reliability(swing_rate)
    anchors = {
        "team_overall_rate": {
            "measured_reliability": team_rel,
            "pairs": team_pairs,
            "known_anchor": KNOWN_TEAM_RATE_RELIABILITY,
        },
        "team_swing_per_snap": {
            "measured_reliability": swing_rel,
            "pairs": swing_pairs,
        },
        "type_rates": type_rate_reliabilities(plays),
    }
    results["reliability_anchors"] = anchors
    print(json.dumps(anchors, indent=2))

    results["elapsed_seconds"] = time.time() - started
    configuration = {
        "command": "python scripts/vardec_penalties.py",
        "pbp_raw_root": str(PBP_RAW_ROOT),
        "type_snapshot": str(TYPE_SNAPSHOT_PATH),
        "team_snapshot": str(TEAM_SNAPSHOT_PATH),
        "features": str(FEATURES_PATH),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    results["provenance"] = artifact_provenance(configuration, FEATURES_PATH, project_root=REPO)
    write_experiment_artifact(
        OUT_DIR,
        "results.json",
        results,
        command=configuration["command"],
        metrics=results,
        notes=(
            "Measure-only penalty variance decomposition; no feature-family "
            "change in src/, no weak-signal registry writes. The experiment "
            "stamp is rooted under artifacts/vardec_pen/experiment_registry so "
            "the shared registry/experiments/ tree stays untouched."
        ),
        project_root=REPO,
        registry_root=OUT_DIR / "experiment_registry",
    )
    print(f"\nwrote {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
