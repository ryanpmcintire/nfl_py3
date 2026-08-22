"""When does ATS-residual variance materialize? A quarter-of-origin
decomposition of ``ats_margin`` (= ``result`` - ``spread_line``).

Population: REG 2009-2025, newest ``data/raw/*/schedules.parquet`` snapshot and
the newest local PBP snapshot under ``data/pbp/raw`` (regular-season scope),
joined on ``game_id``. Pushes are kept; only games missing a result or a line,
or absent from PBP, are dropped.

Method:
- Per game, map every play's ``score_differential`` (posteam perspective) to a
  home-perspective differential and read it at the LAST play of quarters 1-4
  plus any overtime. Quarter deltas d1..d4(+OT) sum to the final margin by
  construction; the market line is a pregame constant, so all ATS-residual
  variance lives in those deltas.
- Variance share of quarter q = Cov(d_q, R) / Var(R), R = ``ats_margin``;
  shares sum to 1. Absolute share = E|d_q| / sum_k E|d_k|.
- Per-quarter home-perspective EPA swings; Q4 EPA swing correlated against
  pregame-only observables with week-blocked bootstrap intervals.
- Lead-change volatility: count of win-probability lead flips per game (the
  in-game WP favourite switching between consecutive plays with defined WP)
  plus one-score-finish rates (|margin| <= 8) by |line| bucket.
- Halftime-recoverable bound: OLS of R on [1, first-half differential, line],
  solved per week-blocked bootstrap draw from per-block Gram statistics;
  recoverable variance share = 1 - SSR/TSS. Upper-bound proxy for what a
  PERFECT in-game model starting at halftime could remove, because it
  conditions on the true halftime state rather than an estimate of it.

MINED FAMILY DISCLOSURE: the five Q4-EPA-swing correlates were chosen after
the data landscape was known, so roughly one spurious 95% interval excluding
zero is expected by chance alone. Intervals are reported regardless of shape;
an interval containing zero is never grounds for closing a line.

Writes JSON to ``artifacts/vardec_late/<UTC timestamp>/results.json`` and a
summary table to stdout. Measure-only; nothing recorded automatically.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

SEASON_START = 2009
SEASON_END = 2025
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_SEED = 20260822
PRIMETIME_ET = "20:00"
HIGH_WIND_MPH = 13.0
ONE_SCORE_MARGIN = 8
SPREAD_BUCKETS = (
    ("line_abs_le_3", -np.inf, 3.0),
    ("line_abs_3to7", 3.0, 7.0),
    ("line_abs_7to10", 7.0, 10.0),
    ("line_abs_gt_10", 10.0, np.inf),
)
QUARTER_KEYS = ("q1", "q2", "q3", "q4", "ot")

TEAM_ALIASES = {
    "LV": ("OAK",),
    "OAK": ("LV",),
    "LAC": ("SD",),
    "SD": ("LAC",),
    "LA": ("STL", "LAR"),
    "LAR": ("LA", "STL"),
    "STL": ("LA", "LAR"),
}


def resolve_franchise(posteam: pd.Series, home: pd.Series, away: pd.Series) -> pd.Series:
    """Map modern/historical franchise codes onto the schedule-side spelling."""

    direct = posteam.eq(home) | posteam.eq(away)
    resolved = posteam.where(direct)
    for idx in resolved.index[resolved.isna()]:
        team = posteam.at[idx]
        candidates = TEAM_ALIASES.get(team, ()) if isinstance(team, str) else ()
        h, a = home.at[idx], away.at[idx]
        for alt in candidates:
            if alt in (h, a):
                resolved.at[idx] = alt
                break
    return resolved


def _latest_schedules() -> Path:
    candidates = sorted((REPO / "data/raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


DEFAULT_SCHEDULES = _latest_schedules()


def load_population(schedules_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(schedules_path)
    keep = [
        c
        for c in [
            "game_id",
            "season",
            "week",
            "gametime",
            "game_type",
            "home_team",
            "away_team",
            "result",
            "spread_line",
            "roof",
            "wind",
            "away_rest",
            "home_rest",
        ]
        if c in raw.columns
    ]
    df = raw.loc[:, keep].copy()
    df = df.loc[df["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)

    df = add_ats_outcomes(df)
    df = df.loc[
        df["ats_margin"].notna() & df["spread_line"].notna() & df["result"].notna()
    ].reset_index(drop=True)

    outdoor = df["roof"].isin(["outdoors", "open"])
    wind = pd.to_numeric(df.get("wind"), errors="coerce")
    df["outdoor_wind_known"] = outdoor & wind.notna()
    df["is_high_wind"] = (outdoor & wind.notna() & (wind >= HIGH_WIND_MPH)).fillna(False)
    df["is_primetime"] = (df["gametime"].astype("string").fillna("") >= PRIMETIME_ET).fillna(False)
    df["is_dome_or_closed"] = df["roof"].isin(["dome", "closed"])
    df["abs_spread"] = df["spread_line"].abs()
    df["rest_diff"] = pd.to_numeric(df.get("home_rest"), errors="coerce") - pd.to_numeric(
        df.get("away_rest"), errors="coerce"
    )
    df["week_block"] = df["season"] * 100 + df["week"]
    return df


def load_plays(pbp_root: Path, game_ids: set[str]) -> pd.DataFrame:
    plays = load_pbp_snapshot(latest_pbp_snapshot(pbp_root))
    plays = plays.loc[plays["game_id"].isin(game_ids)].copy()
    for column in ("play_id", "qtr", "score_differential", "epa", "wp"):
        plays[column] = pd.to_numeric(plays[column], errors="coerce")
    return plays.sort_values(["game_id", "play_id"]).reset_index(drop=True)


def build_game_table(df: pd.DataFrame, plays: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    p = plays.drop(columns=["home_team", "away_team"]).merge(
        df[["game_id", "home_team", "away_team"]], on="game_id", how="inner"
    )
    p["posteam"] = resolve_franchise(p["posteam"], p["home_team"], p["away_team"]).fillna(
        p["posteam"]
    )
    sign = np.where(
        p["posteam"].eq(p["home_team"]),
        1.0,
        np.where(p["posteam"].eq(p["away_team"]), -1.0, np.nan),
    )
    p = p.assign(sign=sign)
    p["home_diff"] = p["score_differential"] * p["sign"]
    p["home_epa"] = p["epa"] * p["sign"]
    p["home_diff_ff"] = p.groupby("game_id")["home_diff"].ffill()

    scored = p.dropna(subset=["home_diff"])
    boundaries = scored.groupby(["game_id", "qtr"]).tail(1)
    pivot = boundaries.pivot(index="game_id", columns="qtr", values="home_diff_ff")
    for q in range(1, 5):
        if q not in pivot.columns:
            pivot[q] = np.nan
    pivot = pivot.sort_index().ffill(axis=1)
    parts: dict[str, pd.Series] = {f"s{q}": pivot[q] for q in range(1, 5)}
    ot_last = pivot[5] if 5 in pivot.columns else pd.Series(np.nan, index=pivot.index)
    parts["ot"] = ot_last.fillna(pivot[4]) - pivot[4]

    epa_wide = (
        p.dropna(subset=["home_epa"]).groupby(["game_id", "qtr"])["home_epa"].sum().unstack("qtr")
    )
    for q in range(1, 6):
        if q not in epa_wide.columns:
            epa_wide[q] = 0.0
    epa_wide = epa_wide.sort_index()
    for q, key in ((1, "epa_q1"), (2, "epa_q2"), (3, "epa_q3"), (4, "epa_q4")):
        parts[key] = epa_wide[q].reindex(pivot.index).fillna(0.0)

    wp = p.dropna(subset=["wp", "sign"])
    wp = wp.assign(home_wp=np.where(wp["sign"] == 1.0, wp["wp"], 1.0 - wp["wp"]))
    wp = wp.assign(leader=np.sign(wp["home_wp"] - 0.5))
    wp = wp.loc[wp["leader"].ne(0.0)]
    flipped = wp["leader"].ne(wp.groupby("game_id")["leader"].shift())
    wp = wp.assign(flipped=flipped.astype(float))
    flips = wp.dropna(subset=["flipped"]).groupby("game_id")["flipped"].sum()
    parts["wp_flips"] = flips.reindex(pivot.index).fillna(0.0)
    parts["has_wp"] = wp.groupby("game_id").size().reindex(pivot.index).notna()

    g = pd.DataFrame(parts)
    g["final_margin"] = g["s4"] + g["ot"]
    g = g.join(
        df.set_index("game_id")[
            [
                "ats_margin",
                "result",
                "spread_line",
                "abs_spread",
                "wind",
                "outdoor_wind_known",
                "is_high_wind",
                "is_primetime",
                "is_dome_or_closed",
                "rest_diff",
                "week_block",
                "season",
            ]
        ],
        how="inner",
    )
    g["q1"] = g["s1"]
    g["q2"] = g["s2"] - g["s1"]
    g["q3"] = g["s3"] - g["s2"]
    g["q4"] = g["s4"] - g["s3"]

    mismatch = (g["final_margin"] - g["result"]).abs()
    reconciliation = {
        "n_games_scored_schedule": len(df),
        "n_games_with_pbp_boundaries": len(g),
        "coverage_ratio": float(len(g) / len(df)),
        "max_abs_margin_mismatch_points": float(mismatch.max()),
        "share_margin_within_half_point": float((mismatch <= 0.5).mean()),
    }
    return g.reset_index(drop=True), reconciliation


def quarter_table(g: pd.DataFrame) -> list[dict[str, Any]]:
    residual = g["ats_margin"].to_numpy(dtype=float)
    total_var = float(np.var(residual, ddof=1))
    deltas = np.column_stack([g[k].to_numpy(dtype=float) for k in QUARTER_KEYS])
    total_mean_abs = float(np.abs(deltas).sum(axis=1).mean())
    rows: list[dict[str, Any]] = []
    running = np.zeros(len(g))
    raw_covs = [float(np.cov(deltas[:, i], residual, ddof=1)[0, 1]) for i in range(deltas.shape[1])]
    cov_total = sum(raw_covs)
    for i, key in enumerate(QUARTER_KEYS):
        delta = deltas[:, i]
        running = running + delta
        cov_share = float(np.cov(delta, residual, ddof=1)[0, 1]) / total_var
        cum_share = float(np.cov(running, residual, ddof=1)[0, 1]) / total_var
        rows.append(
            {
                "quarter": key,
                "mean_home_persp_points": float(np.mean(delta)),
                "sd_points": float(np.std(delta, ddof=1)),
                "variance_share_cov_with_residual": cov_share,
                "variance_share_normalized": raw_covs[i] / cov_total,
                "cumulative_variance_share_through_quarter": cum_share,
                "absolute_contribution_share": float(np.abs(delta).mean()) / total_mean_abs,
                "corr_with_residual": float(np.corrcoef(delta, residual)[0, 1]),
            }
        )
    return rows


def blocked_corr_ci(
    a: np.ndarray,
    b: np.ndarray,
    blocks: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    codes = pd.factorize(blocks[finite])[0]
    n_blocks = int(codes.max()) + 1
    sums_a = np.bincount(codes, weights=a, minlength=n_blocks)
    sums_b = np.bincount(codes, weights=b, minlength=n_blocks)
    counts = np.bincount(codes, minlength=n_blocks).astype(float)
    sq_a = np.bincount(codes, weights=a * a, minlength=n_blocks)
    sq_b = np.bincount(codes, weights=b * b, minlength=n_blocks)
    prod = np.bincount(codes, weights=a * b, minlength=n_blocks)
    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=samples).astype(float)
    n = drawn @ counts
    sa = drawn @ sums_a
    sb = drawn @ sums_b
    saa = drawn @ sq_a
    sbb = drawn @ sq_b
    sab = drawn @ prod
    cov = sab / n - (sa / n) * (sb / n)
    va = saa / n - (sa / n) ** 2
    vb = sbb / n - (sb / n) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(va * vb)
    r = r[np.isfinite(r)]
    lower, upper = np.quantile(r, [0.025, 0.975])
    return {
        "pearson_r": float(np.corrcoef(a, b)[0, 1]),
        "ci95_week_blocked": [float(lower), float(upper)],
        "n": len(a),
        "probability_negative": float(np.mean(r < 0)),
        "bootstrap_samples": samples,
    }


def q4_epa_correlates(g: pd.DataFrame, samples: int, seed: int) -> dict[str, dict[str, Any]]:
    swing = g["epa_q4"].to_numpy(dtype=float)
    blocks = g["week_block"].to_numpy()
    wind = (
        pd.to_numeric(g["wind"], errors="coerce")
        .where(g["outdoor_wind_known"])
        .to_numpy(dtype=float)
    )
    return {
        "wind_mph_outdoor_known": blocked_corr_ci(
            wind, swing, blocks, samples=samples, seed=seed + 11
        ),
        "rest_diff_home_minus_away": blocked_corr_ci(
            g["rest_diff"].to_numpy(dtype=float), swing, blocks, samples=samples, seed=seed + 12
        ),
        "primetime_slot": blocked_corr_ci(
            g["is_primetime"].to_numpy(dtype=float), swing, blocks, samples=samples, seed=seed + 13
        ),
        "abs_spread": blocked_corr_ci(
            g["abs_spread"].to_numpy(dtype=float), swing, blocks, samples=samples, seed=seed + 14
        ),
        "dome_or_closed": blocked_corr_ci(
            g["is_dome_or_closed"].to_numpy(dtype=float),
            swing,
            blocks,
            samples=samples,
            seed=seed + 15,
        ),
    }


def ols_recoverable(
    x: np.ndarray, y: np.ndarray, blocks: np.ndarray, *, samples: int, seed: int
) -> dict[str, Any]:
    design = np.column_stack([np.ones(len(y)), x])
    codes = pd.factorize(blocks)[0]
    n_blocks = int(codes.max()) + 1
    k = design.shape[1]

    def block_mats(weights: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
        if weights is not None:
            d = design * weights[:, None]
            yy = y * weights
        else:
            d, yy = design, y
        grams = np.zeros((n_blocks, k, k))
        rhs = np.zeros((n_blocks, k))
        y_sq = np.zeros(n_blocks)
        y_sum = np.zeros(n_blocks)
        n_sum = np.bincount(codes, minlength=n_blocks).astype(float)
        for j in range(n_blocks):
            m = codes == j
            db, yb = d[m], yy[m]
            grams[j] = db.T @ db
            rhs[j] = db.T @ yb
            y_sq[j] = float(yb @ yb)
            y_sum[j] = float(yb.sum())
        return grams, rhs, y_sq, y_sum, n_sum

    grams, rhs, y_sq, y_sum, n_sum = block_mats()
    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=samples).astype(float)
    gt = np.tensordot(drawn, grams, axes=(1, 0))
    ht = drawn @ rhs
    ysqt = drawn @ y_sq
    yst = drawn @ y_sum
    nt = drawn @ n_sum
    betas = np.linalg.solve(gt, ht[:, :, None])[:, :, 0]
    ssr = (
        ysqt - 2.0 * np.einsum("sk,sk->s", betas, ht) + np.einsum("sk,skl,sl->s", betas, gt, betas)
    )
    tss = ysqt - yst**2 / np.maximum(nt, 1.0)
    draws = 1.0 - ssr / tss
    lower, upper = np.quantile(draws[np.isfinite(draws)], [0.025, 0.975])
    gram_full = grams.sum(axis=0)
    rhs_full = rhs.sum(axis=0)
    beta_full = np.linalg.solve(gram_full, rhs_full)
    n_full = n_sum.sum()
    ssr_full = (
        float(y_sq.sum())
        - 2.0 * float(beta_full @ rhs_full)
        + float(beta_full @ gram_full @ beta_full)
    )
    tss_full = float(y_sq.sum()) - float(y_sum.sum()) ** 2 / n_full
    return {
        "recoverable_variance_share": 1.0 - ssr_full / tss_full,
        "ci95_week_blocked": [float(lower), float(upper)],
        "conditional_residual_sd_points": float(np.sqrt(ssr[0] / nt[0])),
        "unconditional_residual_sd_points": float(np.sqrt(tss[0] / nt[0])),
        "coefficients_intercept_halftime_line": [float(v) for v in beta_full],
        "probability_share_below_one_quarter": float(np.mean(draws < 0.25)),
        "bootstrap_samples": samples,
    }


def volatility_and_one_score(g: pd.DataFrame) -> dict[str, Any]:
    has_wp = g["has_wp"].to_numpy()
    flips_all = g["wp_flips"].to_numpy(dtype=float)
    one_score = (g["result"].abs() <= ONE_SCORE_MARGIN).astype(float).to_numpy()
    buckets: list[dict[str, Any]] = []
    for name, low, high in SPREAD_BUCKETS:
        mask = (g["abs_spread"].to_numpy(dtype=float) > low) & (
            g["abs_spread"].to_numpy(dtype=float) <= high
        )
        buckets.append(
            {
                "bucket": name,
                "n_games": int(mask.sum()),
                "one_score_finish_rate": float(one_score[mask].mean()),
                "mean_wp_flips": float(flips_all[mask & has_wp].mean()),
                "share_games_with_3plus_flips": float((flips_all[mask & has_wp] >= 3).mean()),
            }
        )
    return {
        "one_score_margin_points": ONE_SCORE_MARGIN,
        "overall_one_score_rate": float(one_score.mean()),
        "mean_wp_flips_per_game": float(flips_all[has_wp].mean()),
        "median_wp_flips": float(np.median(flips_all[has_wp])),
        "share_games_with_3plus_flips": float((flips_all[has_wp] >= 3).mean()),
        "games_without_usable_wp": int((~has_wp).sum()),
        "by_line_bucket": buckets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=DEFAULT_SCHEDULES)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "vardec_late" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} ===")
    df = load_population(args.schedules)
    print(f"scheduled REG games with lines: {len(df)}")

    pbp_root = REPO / "data/pbp/raw"
    print(f"=== loading PBP snapshot from {pbp_root} ===")
    plays = load_plays(pbp_root, set(df["game_id"]))
    print(f"plays matched to population games: {len(plays)}")

    g, recon = build_game_table(df, plays)
    baseline_sd = float(g["ats_margin"].std(ddof=1))
    gates = {
        "coverage_ratio_ge_0p95": bool(recon["coverage_ratio"] >= 0.95),
        "margin_reconciliation_share_ge_0p99": bool(
            recon["share_margin_within_half_point"] >= 0.99
        ),
    }

    table = quarter_table(g)
    shares = [row["variance_share_cov_with_residual"] for row in table]
    gates["covariance_shares_sum_to_one"] = bool(abs(sum(shares) - 1.0) <= 0.02)
    line_resid_corr = float(
        np.corrcoef(g["spread_line"].to_numpy(dtype=float), g["ats_margin"].to_numpy(dtype=float))[
            0, 1
        ]
    )

    x_half = np.column_stack(
        [g["s2"].to_numpy(dtype=float), g["spread_line"].to_numpy(dtype=float)]
    )
    recovery = ols_recoverable(
        x_half,
        g["ats_margin"].to_numpy(dtype=float),
        g["week_block"].to_numpy(),
        samples=args.samples,
        seed=args.seed,
    )

    correlates = q4_epa_correlates(g, samples=args.samples, seed=args.seed)
    volatility = volatility_and_one_score(g)

    print(f"\nREG {SEASON_START}-{SEASON_END}: {len(g)} games, ATS-residual sd = {baseline_sd:.3f}")
    print("\nquarter-of-origin decomposition (shares of ATS-residual variance):")
    for row in table:
        share = row["variance_share_cov_with_residual"] * 100
        print(
            f"  {row['quarter']:>3s}  mean={row['mean_home_persp_points']:+.3f} "
            f"sd={row['sd_points']:6.3f}  var_share={share:6.2f}% "
            f"cum={row['cumulative_variance_share_through_quarter'] * 100:6.2f}% "
            f"abs_share={row['absolute_contribution_share'] * 100:6.2f}%"
        )
    print(f"\nQ4 alone variance share: {table[3]['variance_share_cov_with_residual'] * 100:.2f}%")
    print(
        f"raw covariance shares sum to {sum(shares) * 100:.2f}% "
        f"(corr(line, residual) = {line_resid_corr:+.4f}); normalized shares close to 100%"
    )
    lo = recovery["ci95_week_blocked"][0] * 100
    hi = recovery["ci95_week_blocked"][1] * 100
    print(
        f"halftime-recoverable variance share (R ~ 1 + H1 diff + line): "
        f"{recovery['recoverable_variance_share'] * 100:.2f}% [{lo:.2f}, {hi:.2f}]"
    )
    print(
        f"residual sd {recovery['unconditional_residual_sd_points']:.3f} -> "
        f"{recovery['conditional_residual_sd_points']:.3f} given true halftime state + line"
    )
    print("\nQ4 EPA swing vs pregame observables (mined family, no correction):")
    for name, stat in correlates.items():
        print(
            f"  {name:28s} r={stat['pearson_r']:+.4f} "
            f"[{stat['ci95_week_blocked'][0]:+.4f}, {stat['ci95_week_blocked'][1]:+.4f}] "
            f"P(neg)={stat['probability_negative']:.3f}"
        )
    print(f"\ngates: {gates}")

    payload = {
        "schema": 1,
        "generated_at_utc": timestamp,
        "schedules_path": str(args.schedules),
        "population": {
            "game_type": "REG",
            "season_start": SEASON_START,
            "season_end": SEASON_END,
            "n_games": len(g),
            "baseline_ats_residual_sd_points": baseline_sd,
        },
        "reconciliation": recon,
        "method": {
            "variance_share_definition": "Cov(d_q, R)/Var(R); the line is a pregame constant",
            "halftime_bound": "OLS R ~ 1 + true first-half differential + line; 1 - SSR/TSS",
            "bootstrap": "week-block multinomial, percentile 95%",
            "multiplicity_disclosure": (
                "five mined Q4-EPA-swing correlates; roughly one spurious 95% exclusion "
                "expected by chance; no correction applied"
            ),
        },
        "quarter_table": table,
        "q4_alone_variance_share": table[3]["variance_share_cov_with_residual"],
        "covariance_share_closure_note": {
            "raw_share_sum": float(sum(shares)),
            "corr_line_residual": line_resid_corr,
            "explanation": (
                "quarter deltas sum to the final margin, not to R minus a constant, "
                "because Cov(spread_line, ats_margin) > 0; normalized shares close to 100%"
            ),
        },
        "halftime_recoverable": recovery,
        "q4_epa_swing_correlates": correlates,
        "volatility_and_one_score": volatility,
        "gates": gates,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    configuration = {
        "schedules_path": str(args.schedules),
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "samples": args.samples,
        "seed": args.seed,
    }
    payload["provenance"] = artifact_provenance(configuration, args.schedules, project_root=REPO)
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="vardec-lategame",
        metrics={
            "baseline_ats_residual_sd_points": baseline_sd,
            "q4_alone_variance_share": table[3]["variance_share_cov_with_residual"],
            "halftime_recoverable_variance_share": recovery["recoverable_variance_share"],
            "coverage_ratio": recon["coverage_ratio"],
        },
        notes=(
            "Measure-only quarter-of-origin variance decomposition; mined correlate "
            "family disclosed; nothing recorded automatically."
        ),
        source="scripts/vardec_lategame.py",
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
