"""Mine the never-explored halftime/2H fields in the ingested SBR odds archive.

Source: data/raw/sbr_odds/<snapshot>/nfl-odds-<slug>.html (fetched 2026-08-19,
see docs/sbr_odds_archive.md). Every season table carries a ``2H`` column that
``scripts/ingest_sbr_odds.py`` parsed but dropped, plus per-quarter scores
(``1st/2nd/3rd/4th``) that were also dropped. This script re-parses the raw
snapshot with the SAME conventions as the ingest script (parsing helpers are
imported from it, not re-derived), keeps the 2H line (interleaved spread/total
across a game's two rows, smaller magnitude = spread on the favored team's row)
and the quarter scores, joins to ``data/processed/sbr_odds.parquet`` on
(season, away_rot, home_rot) to inherit game_id/week, and answers three
questions:

(a) Info value: does the 2H line embed information about the FULL-game outcome
    beyond the pregame close? Leave-one-season-out CV logistic forced-pick
    accuracy of P(home covers vs close) under M0 [close] / M1 [+realized H1
    margin] / M2 [+2H line]. M0->M2 sizes the late-information pie; M1->M2 is
    the live market's increment over the already-realized halftime score.
(b) Descriptive: 2H line vs realized 2H margin (bias, residual SD, MAE,
    key-number hit rate, favorite cover rate).
(c) Era trend: per-season residual SD and scale-free sharpness
    SD(residual)/SD(actual 2H margin); has live sharpness increased?

ATS-relevant results are emitted as weak-signal RECORD LINES on stdout and in
artifacts/sbr_2h/records.txt. Nothing is written to registry/.

Usage:
    .\\.tools\\uv.exe run python scripts/sbr_halftime_mining.py
    .\\.tools\\uv.exe run python scripts/sbr_halftime_mining.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ingest_sbr_odds import (  # noqa: E402
    RAW_ROW_COLUMNS,
    SEASON_SLUGS,
    _disambiguate_spread_total,
    _parse_date,
    _to_number,
)

DEFAULT_SNAPSHOT = REPO_ROOT / "data" / "raw" / "sbr_odds" / "20260819T192226Z"
DEFAULT_PARQUET = REPO_ROOT / "data" / "processed" / "sbr_odds.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "sbr_2h"

SUSPECT_2H_TOTAL_MAX = 40.0
SUSPECT_2H_TOTAL_MIN = 10.0
RNG_SEED = 20260822
N_BOOTSTRAP = 200


def parse_snapshot_2h(snapshot_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for slug in SEASON_SLUGS:
        path = snapshot_dir / f"nfl-odds-{slug}.html"
        if not path.is_file():
            raise FileNotFoundError(path)
        season = int(slug.split("-")[0])
        records: list[dict[str, Any]] = []
        lines = extract_rows(path)
        if len(lines) % 2 != 0:
            raise ValueError(f"{slug}: odd row count {len(lines)}")
        for i in range(0, len(lines), 2):
            away, home = lines[i], lines[i + 1]
            away_2h_raw, home_2h_raw = away[12].strip(), home[12].strip()
            both_pk = away_2h_raw.lower() == "pk" and home_2h_raw.lower() == "pk"
            away_2h = _to_number(away_2h_raw)
            home_2h = _to_number(home_2h_raw)
            mags = sorted((abs(away_2h), abs(home_2h)))
            home_favored_2h = abs(home_2h) <= abs(away_2h)
            close_spread, close_total, _ = _disambiguate_spread_total(
                _to_number(away[10]), _to_number(home[10])
            )
            records.append(
                {
                    "season": season,
                    "sbr_date_raw": away[0],
                    "game_date": _parse_date(away[0], season),
                    "away_team_raw": away[3],
                    "home_team_raw": home[3],
                    "away_rot": int(away[1]),
                    "home_rot": int(home[1]),
                    "away_score": float(away[8]),
                    "home_score": float(home[8]),
                    "h1_margin_home": (float(home[4]) + float(home[5]))
                    - (float(away[4]) + float(away[5])),
                    "h1_pts_total": (float(home[4]) + float(home[5]))
                    + (float(away[4]) + float(away[5])),
                    "m2_margin_home": (float(home[6]) + float(home[7]))
                    - (float(away[6]) + float(away[7])),
                    "close_home_spread_raw": close_spread,
                    "close_total_raw": close_total,
                    "two_h_home_spread": mags[0] if home_favored_2h else -mags[0],
                    "two_h_total": mags[1],
                    "two_h_both_pk": both_pk,
                }
            )
        frames.append(pd.DataFrame.from_records(records))
    return pd.concat(frames, ignore_index=True)


def extract_rows(path: Path) -> list[list[str]]:
    from html.parser import HTMLParser

    class _TableParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.rows: list[list[str]] = []
            self._cur_row: list[str] | None = None
            self._cur_cell: list[str] | None = None

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag == "tr":
                self._cur_row = []
            elif tag == "td":
                self._cur_cell = []

        def handle_endtag(self, tag: str) -> None:
            if tag == "tr" and self._cur_row is not None:
                if len(self._cur_row) == len(RAW_ROW_COLUMNS):
                    self.rows.append(self._cur_row)
                self._cur_row = None
            elif tag == "td" and self._cur_cell is not None:
                assert self._cur_row is not None
                self._cur_row.append("".join(self._cur_cell).strip())
                self._cur_cell = None

        def handle_data(self, data: str) -> None:
            if self._cur_cell is not None:
                self._cur_cell.append(data)

    parser = _TableParser()
    parser.feed(path.read_text(encoding="utf-8"))
    header = next((r for r in parser.rows if r == RAW_ROW_COLUMNS), None)
    if header is None:
        raise ValueError(f"header row not found in {path}")
    start = parser.rows.index(header) + 1
    return [r for r in parser.rows[start:] if len(r) == len(RAW_ROW_COLUMNS)]


def join_processed(mined: pd.DataFrame, parquet_path: Path) -> pd.DataFrame:
    processed = pd.read_parquet(
        parquet_path,
        columns=[
            "season",
            "game_date",
            "away_rot",
            "home_rot",
            "game_id",
            "close_home_spread",
        ],
    )
    key = ["season", "game_date", "away_rot", "home_rot"]
    if processed.duplicated(subset=key).any():
        raise ValueError("duplicate join keys in processed parquet")
    merged = mined.merge(processed, on=key, how="left")
    matched = merged.loc[merged["game_id"].notna()]
    max_close_diff = float(
        (matched["close_home_spread_raw"] - matched["close_home_spread"]).abs().max()
    )
    if max_close_diff > 1e-9:
        raise ValueError(f"close spread mismatch vs processed parquet: {max_close_diff}")
    n_unmatched = int(merged["game_id"].isna().sum())
    expected_unmatched = int((merged["season"] < 2009).sum())
    if n_unmatched != expected_unmatched:
        raise ValueError(
            f"unexpected unmatched rows: {n_unmatched} != pre-2009 {expected_unmatched}"
        )
    return merged


def build_analysis_frame(mined: pd.DataFrame) -> pd.DataFrame:
    df = mined.copy()
    bad_total = (df["two_h_total"] > SUSPECT_2H_TOTAL_MAX) | (
        df["two_h_total"] < SUSPECT_2H_TOTAL_MIN
    )
    df["suspect_two_h"] = df["two_h_both_pk"] | bad_total
    df["suspect_reason"] = np.where(
        df["two_h_both_pk"],
        "both_pk",
        np.where(bad_total, "implausible_total", ""),
    )
    df["full_margin"] = df["home_score"] - df["away_score"]
    df["cover_margin_full"] = df["full_margin"] - df["close_home_spread"]
    df["y_cover_home"] = (df["cover_margin_full"] > 0).astype(int)
    df["two_h_residual"] = df["m2_margin_home"] - df["two_h_home_spread"]
    df["two_h_actual_total"] = df["away_score"] + df["home_score"] - df["h1_pts_total"]
    df["total_residual"] = df["two_h_actual_total"] - df["two_h_total"]
    df["week_cluster"] = df["game_id"].fillna(df["season"].astype(str) + "_" + df["sbr_date_raw"])
    return df


def _cv_probs(X: np.ndarray, y: np.ndarray, seasons: np.ndarray) -> np.ndarray:
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    probs = np.empty(len(y), dtype=float)
    for season in np.unique(seasons):
        train = seasons != season
        test = ~train
        model.fit(X[train], y[train])
        probs[test] = model.predict_proba(X[test])[:, 1]
    return probs


def _metrics(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    picks = (probs > 0.5).astype(int)
    eps = 1e-12
    return {
        "accuracy": float((picks == y).mean()),
        "log_loss": float(-(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)).mean()),
    }


def loyo_cv(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["y_cover_home"].to_numpy()
    seasons = df["season"].to_numpy()
    return _metrics(_cv_probs(X, y, seasons), y)


def week_cluster_bootstrap_deltas(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, Any]:
    X0 = df[["close_home_spread"]].to_numpy(dtype=float)
    X1 = df[["close_home_spread", "h1_margin_home"]].to_numpy(dtype=float)
    X2 = df[["close_home_spread", "h1_margin_home", "two_h_home_spread"]].to_numpy(dtype=float)
    y = df["y_cover_home"].to_numpy()
    seasons = df["season"].to_numpy()
    clusters = df["week_cluster"].to_numpy()
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    groups = [np.where(inverse == i)[0] for i in range(len(unique_clusters))]

    d_total: list[float] = []
    d_market: list[float] = []
    ll_base: list[float] = []
    ll_oracle: list[float] = []
    for _ in range(N_BOOTSTRAP):
        selected = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in selected])
        p0 = _cv_probs(X0[idx], y[idx], seasons[idx])
        p1 = _cv_probs(X1[idx], y[idx], seasons[idx])
        p2 = _cv_probs(X2[idx], y[idx], seasons[idx])
        m0, m2 = _metrics(p0, y[idx]), _metrics(p2, y[idx])
        m1 = _metrics(p1, y[idx])
        d_total.append(m2["accuracy"] - m0["accuracy"])
        d_market.append(m2["accuracy"] - m1["accuracy"])
        ll_base.append(m0["log_loss"])
        ll_oracle.append(m2["log_loss"])

    def summarize(deltas: list[float]) -> dict[str, float]:
        arr = np.asarray(deltas)
        return {
            "delta_accuracy_mean_boot": float(arr.mean()),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "probability_positive": float((arr > 0).mean()),
        }

    return {
        "M2_minus_M0": summarize(d_total),
        "M2_minus_M1": summarize(d_market),
        "log_loss_M0_point": float(np.mean(ll_base)),
        "log_loss_M2_point": float(np.mean(ll_oracle)),
    }


def linear_r2_increment(df: pd.DataFrame, cols_base: list[str], col_add: str) -> dict[str, float]:
    def design(cols: list[str]) -> np.ndarray:
        x = df[cols].to_numpy(dtype=float)
        return np.column_stack([np.ones(len(x)), x])

    y = df["full_margin"].to_numpy(dtype=float)

    def fit(cols: list[str]) -> tuple[float, np.ndarray]:
        x = design(cols)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot, beta

    r2_base, _ = fit(cols_base)
    cols_full = [*cols_base, col_add]
    r2_full, beta_full = fit(cols_full)
    x = design(cols_full)
    resid = y - x @ beta_full
    dof = len(y) - x.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(x.T @ x)
    se = float(np.sqrt(cov[-1, -1]))
    coef = float(beta_full[-1])
    return {
        "r2_base": float(r2_base),
        "r2_full": float(r2_full),
        "r2_increment": float(r2_full - r2_base),
        "coef_two_h_pts_per_pt": coef,
        "se_two_h": se,
        "ci_low": float(coef - 1.96 * se),
        "ci_high": float(coef + 1.96 * se),
    }


def sharpness_descriptive(df: pd.DataFrame) -> dict[str, Any]:
    e = df["two_h_residual"].to_numpy()
    spread = df["two_h_home_spread"].to_numpy()
    fav_covers = np.where(spread > 0, e > 0, e < 0)
    nonpush = e != 0
    buckets: dict[str, float] = {}
    for label, mask in {
        "abs_line_0_to_2.5": np.abs(spread) <= 2.5,
        "abs_line_3_to_6.5": (np.abs(spread) > 2.5) & (np.abs(spread) <= 6.5),
        "abs_line_7_plus": np.abs(spread) > 6.5,
    }.items():
        sub_e = e[mask]
        spread[mask]
        fav_margin = np.abs(sub_e)
        buckets[label] = float(fav_margin.mean()) if mask.any() else float("nan")
    return {
        "n_games": len(df),
        "bias_pts": float(e.mean()),
        "residual_sd_pts": float(e.std(ddof=1)),
        "mae_pts": float(np.abs(e).mean()),
        "share_within_3pts": float((np.abs(e) <= 3).mean()),
        "actual_margin_sd_pts": float(df["m2_margin_home"].std(ddof=1)),
        "scale_free_sharpness": float(e.std(ddof=1) / df["m2_margin_home"].std(ddof=1)),
        "favorite_cover_rate_nonpush": float(fav_covers[nonpush].mean()),
        "n_pushes": int((e == 0).sum()),
        "total_bias_pts": float(df["total_residual"].mean()),
        "total_residual_sd_pts": float(df["total_residual"].std(ddof=1)),
        "fav_mean_abs_error_by_bucket": buckets,
    }


def era_trend(df: pd.DataFrame) -> dict[str, Any]:
    per_season: list[dict[str, Any]] = []
    for season, group in df.groupby("season"):
        e = group["two_h_residual"]
        sd_act = float(group["m2_margin_home"].std(ddof=1))
        per_season.append(
            {
                "season": int(season),
                "n_games": len(group),
                "bias_pts": float(e.mean()),
                "residual_sd_pts": float(e.std(ddof=1)),
                "mae_pts": float(e.abs().mean()),
                "actual_sd_pts": sd_act,
                "scale_free_sharpness": float(e.std(ddof=1) / sd_act),
            }
        )
    table = pd.DataFrame(per_season)
    eras: dict[str, dict[str, float]] = {}
    for label, (lo, hi) in {
        "2007-2011": (2007, 2011),
        "2012-2016": (2012, 2016),
        "2017-2021": (2017, 2021),
    }.items():
        sub = df.loc[(df["season"] >= lo) & (df["season"] <= hi)]
        stats = sharpness_descriptive(sub)
        eras[label] = {
            "n_games": stats["n_games"],
            "bias_pts": stats["bias_pts"],
            "residual_sd_pts": stats["residual_sd_pts"],
            "scale_free_sharpness": stats["scale_free_sharpness"],
            "favorite_cover_rate_nonpush": stats["favorite_cover_rate_nonpush"],
        }
    trend: dict[str, dict[str, float]] = {}
    x = np.column_stack([np.ones(len(table)), table["season"].to_numpy(dtype=float)])
    for target in ("residual_sd_pts", "scale_free_sharpness", "bias_pts"):
        yv = table[target].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(x, yv, rcond=None)
        resid = yv - x @ beta
        sigma2 = float(resid @ resid) / (len(yv) - 2)
        cov = sigma2 * np.linalg.inv(x.T @ x)
        se = float(np.sqrt(cov[1, 1]))
        trend[target] = {
            "slope_per_decade": float(beta[1] * 10.0),
            "se_slope_per_decade": float(se * 10.0),
        }
    return {"per_season": table.to_dict("records"), "eras": eras, "trend": trend}


def run_analyses(df: pd.DataFrame) -> dict[str, Any]:
    usable = df.loc[~df["suspect_two_h"]].copy().reset_index(drop=True)
    rng = np.random.default_rng(RNG_SEED)
    cv0 = loyo_cv(usable, ["close_home_spread"])
    cv1 = loyo_cv(usable, ["close_home_spread", "h1_margin_home"])
    cv2 = loyo_cv(usable, ["close_home_spread", "h1_margin_home", "two_h_home_spread"])
    boots = week_cluster_bootstrap_deltas(usable, rng)
    lin_vs_pregame = linear_r2_increment(usable, ["close_home_spread"], "two_h_home_spread")
    lin_m2_target = linear_r2_increment(
        usable.assign(full_margin=usable["m2_margin_home"]),
        ["close_home_spread"],
        "two_h_home_spread",
    )
    lin_oracle = linear_r2_increment(
        usable, ["close_home_spread", "h1_margin_home"], "two_h_home_spread"
    )
    baseline_favorite_acc = float(
        (
            (usable["full_margin"] - usable["close_home_spread"]).gt(0)
            == usable["close_home_spread"].gt(0)
        ).mean()
    )
    return {
        "n_usable": len(usable),
        "n_excluded_suspect": int(df["suspect_two_h"].sum()),
        "excluded_detail": (
            df.loc[
                df["suspect_two_h"],
                [
                    "season",
                    "sbr_date_raw",
                    "away_team_raw",
                    "home_team_raw",
                    "two_h_home_spread",
                    "two_h_total",
                    "suspect_reason",
                ],
            ].to_dict("records")
        ),
        "baseline_pick_close_favorite_accuracy": baseline_favorite_acc,
        "cv": {
            "M0_pregame_only": cv0,
            "M1_plus_realized_h1": cv1,
            "M2_plus_2h_line": cv2,
            "delta_accuracy_points_M2_minus_M0": float((cv2["accuracy"] - cv0["accuracy"]) * 100.0),
            "delta_accuracy_points_M2_minus_M1": float((cv2["accuracy"] - cv1["accuracy"]) * 100.0),
        },
        "bootstrap": boots,
        "linear_full_margin_vs_pregame": lin_vs_pregame,
        "linear_m2_margin_vs_pregame": lin_m2_target,
        "linear_full_margin_oracle": lin_oracle,
        "sharpness_overall": sharpness_descriptive(usable),
        "era": era_trend(usable),
    }


def record_lines(results: dict[str, Any]) -> list[str]:
    common = (
        "classification=unresolved_below_power league=nfl "
        "season_start=2007 season_end=2021 "
        f"sample_games={results['n_usable']} source=artifacts/sbr_2h/analysis.json"
    )

    def fmt(block: dict[str, float]) -> str:
        return (
            f"interval=[{block['ci_low'] * 100.0:+.3f}, {block['ci_high'] * 100.0:+.3f}] "
            f"probability_positive={block['probability_positive']:.3f}"
        )

    d0 = results["bootstrap"]["M2_minus_M0"]
    d1 = results["bootstrap"]["M2_minus_M1"]
    sh = results["sharpness_overall"]
    lin_m2 = results["linear_m2_margin_vs_pregame"]
    bias_se = sh["residual_sd_pts"] / math.sqrt(sh["n_games"])
    bias_ppos = 0.5 * (1.0 + math.erf(sh["bias_pts"] / bias_se / math.sqrt(2.0)))

    eff0 = d0["delta_accuracy_mean_boot"] * 100.0
    eff1 = d1["delta_accuracy_mean_boot"] * 100.0
    bias_lo = sh["bias_pts"] - 1.96 * bias_se
    bias_hi = sh["bias_pts"] + 1.96 * bias_se

    return [
        f"RECORD name=sbr_2h_halftime_oracle_vs_pregame effect={eff0:+.3f} "
        f"effect_units=accuracy_points {common} {fmt(d0)} "
        f"description='LOYO-CV forced-pick accuracy gain from adding halftime info "
        f"(realized H1 score + SBR 2H line) to the pregame close, full-game cover vs close' "
        f"notes='oracle sizing of the late-information pie; ties to movement channel +1.72; "
        f"the 2H line does not exist at Tuesday lock so this cannot transfer directly'",
        f"RECORD name=sbr_2h_live_market_beyond_realized_score effect={eff1:+.3f} "
        f"effect_units=accuracy_points {common} {fmt(d1)} "
        f"description='LOYO-CV forced-pick accuracy gain from adding the SBR 2H line on top of "
        f"the realized halftime score (live-market increment alone)' "
        f"notes='oracle sizing only; not a pickable pregame feature'",
        f"RECORD name=sbr_2h_line_efficiency_bias effect={sh['bias_pts']:+.4f} "
        f"effect_units=ats_points {common} "
        f"interval=[{bias_lo:+.4f}, {bias_hi:+.4f}] "
        f"probability_positive={bias_ppos:.3f} "
        f"description='mean realized 2H margin minus SBR 2H line (home convention); "
        f"live-market side bias' "
        f"notes='descriptive book-sharpness diagnostic'",
        f"RECORD name=sbr_2h_line_remaining_half_info_slope "
        f"effect={lin_m2['coef_two_h_pts_per_pt']:+.4f} "
        f"effect_units=ats_points {common} "
        f"interval=[{lin_m2['ci_low']:+.4f}, {lin_m2['ci_high']:+.4f}] "
        f"description='OLS coefficient on the SBR 2H line for the REMAINING half margin, "
        f"m2 ~ close + 2H line (R2 {lin_m2['r2_base']:.4f} -> {lin_m2['r2_full']:.4f}); "
        f"the direct info-value-of-the-live-line number' "
        f"notes='measured; each point of live line moves expected remaining "
        f"margin by this many points'",
    ]


def self_test() -> None:
    away = ["906", "461", "V", "TestA", "0", "3", "8", "10", "21", "48.5", "51", "300", "27.5"]
    home = ["906", "462", "H", "TestB", "0", "14", "7", "7", "28", "2.5", "7", "-365", "3"]
    h1 = (float(home[4]) + float(home[5])) - (float(away[4]) + float(away[5]))
    m2 = (float(home[6]) + float(home[7])) - (float(away[6]) + float(away[7]))
    assert h1 == 11.0 and m2 == -4.0
    close_spread, close_total, _ = _disambiguate_spread_total(
        _to_number(away[10]), _to_number(home[10])
    )
    assert close_spread == 7.0 and close_total == 51.0
    away_2h, home_2h = _to_number(away[12]), _to_number(home[12])
    mags = sorted((abs(away_2h), abs(home_2h)))
    assert mags == [3.0, 27.5]
    spread = mags[0] if abs(home_2h) <= abs(away_2h) else -mags[0]
    assert spread == 3.0
    assert _parse_date("906", 2015) == pd.Timestamp(year=2015, month=9, day=6)
    assert _parse_date("123", 2015) == pd.Timestamp(year=2016, month=1, day=23)

    rng = np.random.default_rng(7)
    n = 80
    frame = pd.DataFrame(
        {
            "season": np.repeat([2009, 2010], n // 2),
            "close_home_spread": rng.normal(0, 5, n),
            "h1_margin_home": rng.normal(0, 10, n),
            "two_h_home_spread": rng.normal(0, 4, n),
            "y_cover_home": rng.integers(0, 2, n),
        }
    )
    frame["week_cluster"] = np.arange(n)
    frame["suspect_two_h"] = False
    frame["full_margin"] = rng.normal(0, 13, n)
    preds = _cv_probs(
        frame[["close_home_spread"]].to_numpy(float),
        frame["y_cover_home"].to_numpy(),
        frame["season"].to_numpy(),
    )
    assert len(preds) == n and ((preds > 0) & (preds < 1)).all()
    cv = loyo_cv(frame, ["close_home_spread", "h1_margin_home"])
    assert 0.0 <= cv["accuracy"] <= 1.0
    lin = linear_r2_increment(frame, ["close_home_spread"], "two_h_home_spread")
    assert np.isfinite(lin["coef_two_h_pts_per_pt"]) and np.isfinite(lin["se_two_h"])
    sh = sharpness_descriptive(
        frame.assign(
            two_h_residual=rng.normal(0, 7, n),
            m2_margin_home=rng.normal(0, 14, n),
            total_residual=rng.normal(0, 5, n),
        )
    )
    assert sh["n_games"] == n and np.isfinite(sh["residual_sd_pts"])
    print("self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    mined = parse_snapshot_2h(args.snapshot)
    joined = join_processed(mined, args.parquet)
    df = build_analysis_frame(joined)
    results = run_analyses(df)
    lines = record_lines(results)

    inventory = {
        "source_snapshot": str(args.snapshot),
        "seasons": sorted(int(s) for s in mined["season"].unique()),
        "n_games_parsed": len(mined),
        "coverage_per_season": {
            str(int(s)): int(g) for s, g in mined.groupby("season").size().items()
        },
        "fields_extracted": {
            "two_h_home_spread": (
                "smaller-magnitude 2H cell; positive = home favored (2nd-half spread)"
            ),
            "two_h_total": "larger-magnitude 2H cell (2nd-half total)",
            "h1_margin_home": "(1st+2nd home) minus (1st+2nd away)",
            "m2_margin_home": "(3rd+4th home) minus (3rd+4th away), realized second-half margin",
            "two_h_both_pk": "both 2H cells literally 'pk' (unusable)",
            "suspect_two_h": "both_pk or 2H total outside [10, 40]",
        },
        "excluded_games": results["excluded_detail"],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "field_inventory.json").write_text(
        json.dumps(inventory, indent=2, default=str), encoding="utf-8"
    )
    serializable = json.loads(json.dumps(results, default=str))
    (args.out_dir / "analysis.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    (args.out_dir / "records.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"n_parsed={len(mined)} n_usable={results['n_usable']} "
        f"n_excluded={results['n_excluded_suspect']}"
    )
    print("\n=== (a) info value: LOYO-CV P(home covers vs close) ===")
    print(json.dumps(results["cv"], indent=2))
    print(
        "baseline pick-close-favorite accuracy:",
        results["baseline_pick_close_favorite_accuracy"],
    )
    print("bootstrap:", json.dumps(results["bootstrap"], indent=2))
    print(
        "linear vs pregame (target=full margin):",
        json.dumps(results["linear_full_margin_vs_pregame"], indent=2),
    )
    print(
        "linear vs pregame (target=2H margin):",
        json.dumps(results["linear_m2_margin_vs_pregame"], indent=2),
    )
    print("linear oracle:", json.dumps(results["linear_full_margin_oracle"], indent=2))
    print("\n=== (b) sharpness overall ===")
    print(json.dumps(results["sharpness_overall"], indent=2))
    print("\n=== (c) era ===")
    print(json.dumps(results["era"]["eras"], indent=2))
    print("trend:", json.dumps(results["era"]["trend"], indent=2))
    print("\n=== record lines ===")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
