"""Variance decomposition of turnovers in NFL game margins, 2009-2025 REG.

Question: how much of the variance in the final margin do turnovers explain
(turnover-EPA swing around the line), what part is persistent team trait
versus single-game luck, and how much of the fumble slice is recovery luck?

Constructs:

- Turnover-EPA swing (per team-game): sum of ``epa`` over snapshot rows with
  ``interception == 1`` or ``fumble_lost == 1``, ``play == 1``, EPA present,
  ``posteam == team`` -- the expected-points cost of that team's giveaways on
  its offensive snaps. Defensive return touchdowns are inside the realized
  EPA of these rows (measured: touchdown-flagged turnover rows average about
  -8.1 EPA versus about -4.0 for non-scoring interceptions), so the folk
  "pick-six / scoop-and-score" channel is included by construction.

- Counterfactual swaps: (a) league-mean swap -- every game's realized turnover
  differential replaced by zero; delta-sd equals sd(differential). (b)
  season-lagged swap -- each team-game swing predicted from its PRIOR-season
  giveaway-EPA-per-snap rate times this game's snaps; the trait component.
  The gap between the two deltas plus the fitted-swap forecastable fraction
  is the luck-versus-trait split.

- Fumble-recovery luck: the narrowed local snapshot carries no ``fumble``
  (all-fumbles) column and no recovery attribution, so a literal 50/50
  recovery reassignment with EP recomputation is impossible here. Proxy:
  fit OLS of lost-fumble EPA on pre-play situation (down, yardline_100,
  ydstogo, quarter, score state) across ALL lost fumbles, recompute every
  team-game fumble swing from fitted situational expectations, and treat
  realized-minus-fitted as the play-realization residual -- of which recovery
  assignment and return outcome are the dominant pieces. The partial R^2 of
  that residual against margin is reported as the recovery-luck slice, an
  upper-bound-flavored proxy whose limits are stated in
  docs/vardec_turnovers.md.

Uncertainty: week-blocked bootstrap (block id = season*100 + week), 2,000
resamples, joint resampling of every statistic's blocks.

Writes artifacts/vardec_to/results.json only. Research measurement, not a
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
OUT_DIR = REPO / "artifacts/vardec_to"

BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20260822

KNOWN_TEAM_TO_RATE_RELIABILITY = 0.13


def _canonical(team: pd.Series) -> pd.Series:
    return team.map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def load_and_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    snapshot = latest_pbp_snapshot(PBP_RAW_ROOT)
    plays = load_pbp_snapshot(snapshot, include_postseason=False)
    plays["posteam"] = _canonical(plays["posteam"].astype("string").fillna(""))
    inter = pd.to_numeric(plays["interception"], errors="coerce").fillna(0.0)
    flost = pd.to_numeric(plays["fumble_lost"], errors="coerce").fillna(0.0)
    play_flag = pd.to_numeric(plays["play"], errors="coerce").fillna(0.0)
    epa = pd.to_numeric(plays["epa"], errors="coerce")
    to_mask = (inter == 1.0) | (flost == 1.0)
    kept_mask = to_mask & (play_flag == 1.0) & epa.notna() & plays["posteam"].ne("")
    audit = {
        "snapshot_id": snapshot.snapshot_id,
        "total_reg_plays": len(plays),
        "turnover_flag_rows": int(to_mask.sum()),
        "interception_rows": int((inter == 1.0).sum()),
        "fumble_lost_rows": int((flost == 1.0).sum()),
        "excluded_no_play_turnovers": int((to_mask & (play_flag != 1.0)).sum()),
        "excluded_play_turnover_missing_epa_or_posteam": int(
            (to_mask & (play_flag == 1.0) & ~kept_mask).sum()
        ),
        "kept_turnover_rows": int(kept_mask.sum()),
    }
    kept = plays.loc[kept_mask].copy()
    kept["epa_f"] = epa.loc[kept_mask]
    kept["is_int"] = inter.loc[kept_mask]
    kept["is_flost"] = flost.loc[kept_mask]
    kept["td_flag"] = pd.to_numeric(plays.loc[kept_mask, "touchdown"], errors="coerce").fillna(0.0)
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
            to_count=("epa_f", "size"),
            int_epa=("epa_f", lambda s: float((kept.loc[s.index, "is_int"] * s).sum())),
            fum_epa=("epa_f", lambda s: float((kept.loc[s.index, "is_flost"] * s).sum())),
            fum_count=("is_flost", "sum"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    table = snaps.merge(agg, on=["game_id", "season", "week", "team"], how="left")
    fill_cols = ["swing", "to_count", "int_epa", "fum_epa", "fum_count"]
    table[fill_cols] = table[fill_cols].fillna(0.0)
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

    value_cols = [
        "swing",
        "to_count",
        "int_epa",
        "fum_epa",
        "fum_count",
        "snaps",
        "swing_pred",
        "fum_epa_neutral",
        "swing_neutral",
    ]
    home_side = table.rename(columns={"team": "home_team", **{c: f"{c}_home" for c in value_cols}})
    away_side = table.rename(columns={"team": "away_team", **{c: f"{c}_away" for c in value_cols}})
    out = reg.merge(home_side.drop(columns=["season", "week"]), on=["game_id", "home_team"])
    out = out.merge(away_side.drop(columns=["season", "week"]), on=["game_id", "away_team"])
    if len(out) != len(reg):
        raise RuntimeError("home/away join lost games")
    out["swing_diff"] = out["swing_home"] - out["swing_away"]
    out["int_diff"] = out["int_epa_home"] - out["int_epa_away"]
    out["fum_diff"] = out["fum_epa_home"] - out["fum_epa_away"]
    out["fum_neutral_diff"] = out["fum_epa_neutral_home"] - out["fum_epa_neutral_away"]
    out["fum_realization_resid_diff"] = out["fum_diff"] - out["fum_neutral_diff"]
    out["swing_neutral_diff"] = out["swing_neutral_home"] - out["swing_neutral_away"]
    out["count_diff"] = out["to_count_home"] - out["to_count_away"]
    out["swing_pred_diff"] = out["swing_pred_home"] - out["swing_pred_away"]
    out["week_block"] = out["season"].astype(int) * 100 + out["week"].astype(int)
    return out


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


def fit_situational_expectation(kept: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    fum = kept.loc[kept["is_flost"] == 1.0].copy()
    n = len(fum)
    down = pd.to_numeric(fum["down"], errors="coerce").fillna(0.0)
    onehot_down = np.column_stack([(down == d).to_numpy(dtype=np.float64) for d in (2, 3, 4)])
    X = np.column_stack(
        [
            np.ones(n),
            onehot_down,
            pd.to_numeric(fum["yardline_100"], errors="coerce").fillna(50.0).to_numpy() / 100.0,
            pd.to_numeric(fum["ydstogo"], errors="coerce").fillna(10.0).to_numpy() / 10.0,
            pd.to_numeric(fum["qtr"], errors="coerce").fillna(1.0).to_numpy() / 4.0,
            pd.to_numeric(fum["score_differential"], errors="coerce").fillna(0.0).to_numpy() / 21.0,
        ]
    )
    y = fum["epa_f"].to_numpy(dtype=np.float64)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = pd.Series(X @ beta, index=fum.index)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    info = {
        "fumble_rows_used": n,
        "in_sample_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "coefficients": {
            "intercept": float(beta[0]),
            "down_2": float(beta[1]),
            "down_3": float(beta[2]),
            "down_4": float(beta[3]),
            "yardline_100_per100": float(beta[4]),
            "ydstogo_per10": float(beta[5]),
            "qtr_per4": float(beta[6]),
            "score_diff_per21": float(beta[7]),
        },
    }
    return fitted, info


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


def year_over_year_reliability(rate: pd.DataFrame) -> tuple[float, int]:
    ordered = rate.sort_values(["team", "season"]).copy()
    ordered["next_rate"] = ordered.groupby("team")["rate"].shift(-1)
    ordered["next_season"] = ordered.groupby("team")["season"].shift(-1)
    pairs = ordered.loc[ordered["next_season"] == ordered["season"] + 1]
    return float(pairs["rate"].corr(pairs["next_rate"])), len(pairs)


def reliability_anchors(plays: pd.DataFrame, kept: pd.DataFrame) -> dict[str, Any]:
    sub = plays.loc[plays["posteam"].ne("")].copy()
    games_per_team_season = (
        sub.groupby(["season", "posteam"], sort=False)["game_id"]
        .nunique()
        .rename("games")
        .reset_index()
    )
    to_num = pd.to_numeric(sub["interception"], errors="coerce").fillna(0.0) + pd.to_numeric(
        sub["fumble_lost"], errors="coerce"
    ).fillna(0.0)
    tos = (
        to_num.groupby([sub["season"], sub["posteam"]])
        .sum()
        .rename("tos")
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    snaps = (
        sub.groupby(["season", "posteam"], sort=False)
        .agg(snaps=("play_id", "size"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    base = games_per_team_season.rename(columns={"posteam": "team"}).merge(
        tos, on=["season", "team"]
    )
    base["to_rate_pg"] = base["tos"] / base["games"]
    rel_pg, pairs_pg = year_over_year_reliability(
        base[["team", "season", "to_rate_pg"]].rename(columns={"to_rate_pg": "rate"})
    )

    per_snap = base.merge(snaps, on=["season", "team"])
    per_snap["to_rate_ps"] = per_snap["tos"] / per_snap["snaps"]
    rel_ps, pairs_ps = year_over_year_reliability(
        per_snap[["team", "season", "to_rate_ps"]].rename(columns={"to_rate_ps": "rate"})
    )

    swing_season = (
        kept.groupby(["season", "posteam"], sort=False)
        .agg(total=("epa_f", "sum"))
        .reset_index()
        .rename(columns={"posteam": "team"})
        .merge(snaps, on=["season", "team"])
    )
    swing_season["rate"] = swing_season["total"] / swing_season["snaps"]
    rel_swing, pairs_swing = year_over_year_reliability(swing_season[["team", "season", "rate"]])

    fum_season = (
        kept.loc[kept["is_flost"] == 1.0]
        .groupby(["season", "posteam"], sort=False)
        .agg(fum_lost=("is_flost", "sum"))
        .reset_index()
        .rename(columns={"posteam": "team"})
        .merge(games_per_team_season.rename(columns={"posteam": "team"}), on=["season", "team"])
    )
    fum_season["fum_lost_pg"] = fum_season["fum_lost"] / fum_season["games"]
    rel_fum, pairs_fum = year_over_year_reliability(
        fum_season[["team", "season", "fum_lost_pg"]].rename(columns={"fum_lost_pg": "rate"})
    )

    take = (
        kept.assign(team=_canonical(kept["defteam"].astype("string").fillna("")))
        .groupby(["season", "team"], sort=False)
        .agg(takeaways=("epa_f", "size"))
        .reset_index()
    )
    net = base.merge(take, on=["season", "team"], how="left")
    net[["takeaways"]] = net[["takeaways"]].fillna(0.0)
    net["net_to_pg"] = (net["takeaways"] - net["tos"]) / net["games"]
    rel_net, pairs_net = year_over_year_reliability(
        net[["team", "season", "net_to_pg"]].rename(columns={"net_to_pg": "rate"})
    )

    return {
        "team_turnover_rate_per_game": {
            "reliability": rel_pg,
            "pairs": pairs_pg,
            "known_anchor": KNOWN_TEAM_TO_RATE_RELIABILITY,
        },
        "team_turnover_rate_per_snap": {"reliability": rel_ps, "pairs": pairs_ps},
        "team_turnover_epa_per_snap": {"reliability": rel_swing, "pairs": pairs_swing},
        "team_fumbles_lost_per_game": {"reliability": rel_fum, "pairs": pairs_fum},
        "team_net_turnover_margin_per_game": {"reliability": rel_net, "pairs": pairs_net},
    }


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    print("=== Step 1: load snapshot, exclusion audit ===")
    plays, kept, audit = load_and_audit()
    results["exclusion_audit"] = audit
    for name, value in audit.items():
        print(f"{name}: {value}")

    print("\n=== Step 2: return-TD content of turnover EPA ===")
    int_td = kept.loc[(kept["is_int"] == 1.0) & (kept["td_flag"] == 1.0), "epa_f"]
    int_notd = kept.loc[(kept["is_int"] == 1.0) & (kept["td_flag"] == 0.0), "epa_f"]
    fum_td = kept.loc[(kept["is_flost"] == 1.0) & (kept["td_flag"] == 1.0), "epa_f"]
    fum_notd = kept.loc[(kept["is_flost"] == 1.0) & (kept["td_flag"] == 0.0), "epa_f"]
    ret_td = {
        "interceptions_returned_for_td": len(int_td),
        "interception_rows_total": int((kept["is_int"] == 1.0).sum()),
        "mean_epa_pick_six": float(int_td.mean()) if len(int_td) else float("nan"),
        "mean_epa_interception_no_td": float(int_notd.mean()),
        "fumbles_lost_returned_for_td": len(fum_td),
        "fumble_lost_rows_total": int((kept["is_flost"] == 1.0).sum()),
        "mean_epa_defensive_fumble_td": float(fum_td.mean()) if len(fum_td) else float("nan"),
        "mean_epa_fumble_lost_no_td": float(fum_notd.mean()),
    }
    results["return_td_content"] = ret_td
    print(json.dumps(ret_td, indent=2))

    print("\n=== Step 3: team-game aggregates, game frame, lagged predictions ===")
    table = build_team_game_table(plays, kept)
    fitted, sit_info = fit_situational_expectation(kept)
    results["situational_fit_fumbles"] = sit_info
    kept = kept.assign(fum_epa_neutral=fitted)
    neutral = kept.groupby(["game_id", "season", "week", "posteam"], sort=False).agg(
        fum_epa_neutral=("fum_epa_neutral", "sum")
    )
    neutral = neutral.reset_index().rename(columns={"posteam": "team"})
    table = table.merge(neutral, on=["game_id", "season", "week", "team"], how="left")
    table["fum_epa_neutral"] = table["fum_epa_neutral"].fillna(0.0)
    table["swing_neutral"] = table["swing"] - table["fum_epa"] + table["fum_epa_neutral"]
    table = add_lagged_predictions(table)
    features = pd.read_parquet(FEATURES_PATH)
    games = build_game_frame(table, features)
    ats_games = games.loc[games["ats_margin"].notna()].copy()
    print(f"games={len(games)} (with ATS margin: {len(ats_games)})")

    print("\n=== Step 4: headline decomposition ===")
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
    results["bootstrap_variance_share_gross"] = interval(
        bootstrapper.run(base_series, variance_share)
    )
    results["bootstrap_r2_margin_on_swing"] = interval(
        bootstrapper.run(base_series, squared_correlation)
    )

    results["league_mean_swap"] = {
        "delta_sd_pts": float(np.std(d, ddof=1)),
        "note": "swap realized differential to zero; delta equals the realized differential",
    }
    lm_series = dict(base_series)
    lm_series["e"] = d
    lm_series["e2"] = d * d
    results["bootstrap_league_mean_swap_delta_sd"] = interval(
        bootstrapper.run(lm_series, make_sd_statistic("e"))
    )

    print("\n=== Step 5: component split (interceptions vs fumbles) ===")
    secondary: dict[str, Any] = {}
    for col, label in (
        ("int_diff", "interception_epa"),
        ("fum_diff", "fumble_lost_epa"),
        ("count_diff", "to_count"),
    ):
        x = games[col].to_numpy(dtype=np.float64)
        r = float(np.corrcoef(x, m)[0, 1])
        unit = "pts" if col != "count_diff" else "count"
        entry = {
            "label": label,
            f"sd_{unit}": float(np.std(x, ddof=1)),
            "correlation_with_margin": r,
            "r2": r * r,
            "gross_variance_share": float(np.var(x, ddof=1) / np.var(m, ddof=1)),
        }
        series = {"d": x, "d2": x * x, "m": m, "m2": m * m, "dm": x * m}
        entry["bootstrap_r2"] = interval(bootstrapper.run(series, squared_correlation))
        secondary[col] = entry
        print(col, json.dumps(entry))
    results["component_split"] = secondary

    print("\n=== Step 6: recovery-luck slice (situation-neutralized fumble EPA) ===")
    fn = games["fum_neutral_diff"].to_numpy(dtype=np.float64)
    fr = games["fum_realization_resid_diff"].to_numpy(dtype=np.float64)
    a = games["int_diff"].to_numpy(dtype=np.float64)
    fd = games["fum_diff"].to_numpy(dtype=np.float64)
    r2_fum_realized_alone = float(np.corrcoef(fd, m)[0, 1]) ** 2
    r2_fum_neutral_alone = float(np.corrcoef(fn, m)[0, 1]) ** 2

    def two_pred_r2(x1: np.ndarray, x2: np.ndarray) -> float:
        X = np.column_stack([np.ones(len(m)), x1, x2])
        beta, *_ = np.linalg.lstsq(X, m, rcond=None)
        resid_y = m - X @ beta
        return float(1.0 - np.var(resid_y, ddof=0) / np.var(m, ddof=0))

    r2_int_plus_resid = two_pred_r2(a, fd)
    r2_int_plus_neutral = two_pred_r2(a, fn)

    def ols_r2_statistic(scaled: dict[str, float]) -> float:
        n = scaled["_n"]
        my = scaled["m"] / n
        mx = scaled["x"] / n
        ma = scaled["a"] / n
        vxx = scaled["x2"] / n - mx * mx
        vaa = scaled["a2"] / n - ma * ma
        cxa = scaled["xa"] / n - mx * ma
        cyx = scaled["xm"] / n - mx * my
        cya = scaled["am"] / n - ma * my
        vy = scaled["m2"] / n - my * my
        det = vxx * vaa - cxa * cxa
        if det == 0 or vy <= 0:
            return float("nan")
        bx = (vaa * cyx - cxa * cya) / det
        ba = (vxx * cya - cxa * cyx) / det
        return float((bx * cyx + ba * cya) / vy)

    def component_series(x: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "m": m,
            "m2": m * m,
            "a": a,
            "a2": a * a,
            "x": x,
            "x2": x * x,
            "xa": x * a,
            "xm": x * m,
            "am": a * m,
        }

    full_draws = bootstrapper.run(component_series(fd), ols_r2_statistic)
    neutral_draws = bootstrapper.run(component_series(fn), ols_r2_statistic)
    a_r2_alone = float(np.corrcoef(a, m)[0, 1]) ** 2
    recovery = {
        "method_limit": (
            "no fumble or recovery columns survive ingestion; 50/50 recovery cannot be "
            "assigned play-by-play locally. Proxy: realized-minus-situation-fitted "
            "lost-fumble EPA residual; recovery assignment and return outcome dominate "
            "that residual"
        ),
        "sd_fumble_neutral_diff_pts": float(np.std(fn, ddof=1)),
        "sd_fumble_realization_resid_pts": float(np.std(fr, ddof=1)),
        "correlation_neutral_diff_with_margin": float(np.corrcoef(fn, m)[0, 1]),
        "r2_margin_on_neutral_diff_alone": r2_fum_neutral_alone,
        "r2_margin_on_realized_fumble_diff_alone": r2_fum_realized_alone,
        "r2_model_int_plus_neutral_fumble": r2_int_plus_neutral,
        "r2_model_int_plus_realization_resid": r2_int_plus_resid,
        "fumble_increment_over_int_neutral_r2_points": r2_int_plus_neutral - a_r2_alone,
        "recovery_luck_slice_r2_points": r2_int_plus_resid - r2_int_plus_neutral,
        "recovery_luck_gross_variance_share": float(np.var(fr, ddof=1) / np.var(m, ddof=1)),
        "correlation_realization_resid_with_margin": float(np.corrcoef(fr, m)[0, 1]),
        "covariance_neutral_x_residual": float(
            np.cov(np.column_stack([fn, fr]), rowvar=False, ddof=1)[0, 1]
        ),
    }
    results["recovery_luck_slice"] = recovery
    print(json.dumps(recovery, indent=2))
    results["bootstrap_recovery_luck_slice_r2_points"] = interval(full_draws - neutral_draws)
    rec_series = {
        "d": fr,
        "d2": fr * fr,
        "m": m,
        "m2": m * m,
        "dm": fr * m,
    }
    results["bootstrap_recovery_luck_gross_share"] = interval(
        bootstrapper.run(rec_series, variance_share)
    )

    print("\n=== Step 7: season-lagged counterfactual swap ===")
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
        "if the raw-rate swap residual EXCEEDS the realized swing sd, unshrunk "
        "lagged rates are mis-scaled for game-level use (regression-to-the-mean); "
        "the fitted-swap figures are the honest forecastable fraction and carry "
        "in-sample OLS optimism"
    )
    results["lagged_swap"] = lag
    print(json.dumps(lag, indent=2))
    lag_series = {"e": np.nan_to_num(resid), "e2": np.nan_to_num(resid) ** 2}
    results["bootstrap_lagged_swap_delta_sd"] = interval(
        bootstrapper.run({**base_series, **lag_series}, make_sd_statistic("e"))
    )

    print("\n=== Step 8: ATS-margin variant ===")
    da = ats_games["swing_diff"].to_numpy(dtype=np.float64)
    ma = ats_games["ats_margin"].to_numpy(dtype=np.float64)
    fa = games.loc[games["ats_margin"].notna(), "fum_realization_resid_diff"].to_numpy(
        dtype=np.float64
    )
    ats = {
        "games": len(ats_games),
        "correlation_with_ats_margin": float(np.corrcoef(da, ma)[0, 1]),
        "variance_share_gross_vs_ats": float(np.var(da, ddof=1) / np.var(ma, ddof=1)),
        "r2_vs_ats": float(np.corrcoef(da, ma)[0, 1] ** 2),
        "recovery_luck_gross_share_vs_ats": float(np.var(fa, ddof=1) / np.var(ma, ddof=1)),
        "recovery_luck_correlation_vs_ats": float(np.corrcoef(fa, ma)[0, 1]),
    }
    results["ats_variant"] = ats
    print(json.dumps(ats, indent=2))

    print("\n=== Step 9: reliability anchors ===")
    anchors = reliability_anchors(plays, kept)
    results["reliability_anchors"] = anchors
    print(json.dumps(anchors, indent=2))

    results["elapsed_seconds"] = time.time() - started
    configuration = {
        "command": "python scripts/vardec_turnovers.py",
        "pbp_raw_root": str(PBP_RAW_ROOT),
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
            "Measure-only turnover variance decomposition; no feature-family "
            "change in src/, no weak-signal registry writes. The experiment "
            "stamp is rooted under artifacts/vardec_to/experiment_registry so "
            "the shared registry/experiments/ tree stays untouched."
        ),
        project_root=REPO,
        registry_root=OUT_DIR / "experiment_registry",
    )
    print(f"\nwrote {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
