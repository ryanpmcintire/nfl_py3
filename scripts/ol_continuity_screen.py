"""Offensive-line continuity screen (4 predeclared cells).

Predeclaration: docs/ol_continuity_screen.md, frozen before this script was
run against any cover outcome. Reliability-first: YoY persistence and
within-season split-half on the OL-continuity trait are measured before any
cell exists. Binding taxonomy owned verbatim per AGENTS.md: an interval that
crosses zero is NEVER a closing ground; only refuted mechanism (resolved wrong
sign or no split-half reliability) or bounded-by-positive-control closes a
line; everything else is unresolved_below_power, recorded with
probability_positive. This script only measures; it records nothing to the
weak-signal registry.

Method copied algorithm-identical from scripts/redzone_reversion_screen.py
(block_bootstrap_two_group / summarize / score_cell / year_over_year_pairs /
bootstrap_pearson_ci).

Writes artifacts/ol_continuity_screen/<UTC stamp>/results.json. It does NOT
write the experiment registry; the caller registers via
`nfl-ats weak-signals record`.
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

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.experiment_runner import scale_subset_effect  # noqa: E402
from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

SEASON_START = 2013
SEASON_END = 2025
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260821
EARLY_WEEK_MAX = 8
STARTER_OFFENSE_PCT = 0.50
OL_POSITIONS = frozenset({"C", "G", "T", "OG", "OT", "LG", "RG", "LT", "RT", "OL"})
ACUTE_OVERHAUL_MAX = 0.4
LEAKAGE_CHECK_ROWS = 250

SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "gameday",
    "game_type",
    "home_team",
    "away_team",
    "result",
    "spread_line",
]


def _latest_snap_counts() -> Path:
    candidates = sorted((REPO / "data/players/raw").glob("*/snap_counts.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/players/raw/*/snap_counts.parquet snapshot found")
    return candidates[-1]


def _latest_schedules() -> Path:
    candidates = sorted((REPO / "data/raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


def normalize_reg(raw: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in SCHEDULE_COLUMNS if c in raw.columns]
    df = raw.loc[:, available].copy()
    df = df.loc[df["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)
    df["home_team"] = df["home_team"].replace(TEAM_ABBREVIATION_ALIASES)
    df["away_team"] = df["away_team"].replace(TEAM_ABBREVIATION_ALIASES)
    return df


def load_schedules(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    full_reg = normalize_reg(raw)
    df = add_ats_outcomes(full_reg.copy())
    n_before_push_drop = len(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)
    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = n_before_push_drop - len(df)
    df.attrs["full_reg"] = full_reg
    df["week_block"] = df["season"] * 100 + df["week"]
    return df


def _is_ol(position: pd.Series) -> pd.Series:
    parts = position.str.split("/")
    return parts.apply(lambda ps: any(p in OL_POSITIONS for p in ps))


def build_starters(snaps: pd.DataFrame) -> pd.DataFrame:
    df = snaps.copy()
    df = df.loc[df["game_type"].eq("REG")].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].copy()
    df["team"] = df["team"].astype("string").replace(TEAM_ABBREVIATION_ALIASES)
    df["position"] = df["position"].astype("string")
    df["offense_pct"] = pd.to_numeric(df["offense_pct"], errors="coerce")

    ol = df.loc[
        _is_ol(df["position"]), ["game_id", "season", "week", "team", "player", "offense_pct"]
    ].copy()
    ol = ol.loc[ol["offense_pct"] >= STARTER_OFFENSE_PCT]
    starters = (
        ol.groupby(["game_id", "season", "week", "team"], sort=False)["player"]
        .agg(lambda s: frozenset(s.astype("string")))
        .rename("starters")
        .reset_index()
    )
    return starters


def build_team_games(schedules: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    sides = []
    for is_home in (True, False):
        team_col = "home_team" if is_home else "away_team"
        side = pd.DataFrame(
            {
                "game_id": schedules["game_id"],
                "season": schedules["season"],
                "week": schedules["week"],
                "gameday": schedules["gameday"],
                "week_block": schedules.get("week_block", np.nan),
                "team": schedules[team_col],
                "team_covered": (
                    schedules["home_cover"] if is_home else 1.0 - schedules["home_cover"]
                )
                if "home_cover" in schedules
                else np.nan,
            }
        )
        sides.append(side)
    team_games = pd.concat(sides, ignore_index=True)
    team_games = team_games.merge(starters, on=["game_id", "season", "week", "team"], how="left")
    return team_games.sort_values(["team", "season", "week"]).reset_index(drop=True)


def continuity_from_starters(team_games: pd.DataFrame) -> pd.Series:
    def pair_value(row: pd.Series) -> float | None:
        cur = row["starters"]
        prev = row["prev_starters"]
        if not isinstance(cur, frozenset) or not isinstance(prev, frozenset):
            return None
        return len(cur & prev) / 5.0

    shifted = team_games.groupby(["team", "season"], sort=False)["starters"].shift(1)
    work = team_games.assign(prev_starters=shifted)
    values = work.apply(pair_value, axis=1)
    return pd.Series(values, index=team_games.index, dtype="float64")


def add_traits(team_games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    df = team_games.copy()
    df["ol_continuity"] = continuity_from_starters(df)

    group = df.groupby(["team", "season"], sort=False)["ol_continuity"]
    df["ol_recent"] = group.shift(1)
    trailing = pd.concat([group.shift(1), group.shift(2)], axis=1)
    df["ol_trailing2"] = trailing.mean(axis=1, skipna=True).where(trailing.notna().any(axis=1))

    season_panel = (
        df.dropna(subset=["ol_continuity"])
        .groupby(["team", "season"], sort=False)["ol_continuity"]
        .mean()
        .rename("ol_season_mean")
        .reset_index()
    )
    league_mean = season_panel.groupby("season")["ol_season_mean"].transform("mean")
    season_panel["ol_continuity_centered"] = season_panel["ol_season_mean"] - league_mean
    prior = season_panel[["team", "season", "ol_continuity_centered"]].copy()
    prior["season"] = prior["season"] + 1
    prior = prior.rename(columns={"ol_continuity_centered": "prior_ol_centered"})
    df = df.merge(prior, on=["team", "season"], how="left")

    thresholds = {
        "trailing2_q25": float(df["ol_trailing2"].quantile(0.25)),
        "trailing2_q75": float(df["ol_trailing2"].quantile(0.75)),
        "prior_centered_q25": float(season_panel["ol_continuity_centered"].quantile(0.25)),
    }
    return df, thresholds


def year_over_year_pairs(panel: pd.DataFrame, metric: str) -> pd.DataFrame:
    left = panel[["team", "season", metric]].rename(columns={metric: "value_t"})
    right = panel[["team", "season", metric]].rename(columns={metric: "value_t1"})
    right["season"] = right["season"] - 1
    pairs = left.merge(right, on=["team", "season"], how="inner")
    return pairs.dropna(subset=["value_t", "value_t1"]).reset_index(drop=True)


def bootstrap_pearson_ci(x: np.ndarray, y: np.ndarray, *, samples: int, seed: int) -> list[float]:
    n = len(x)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        xi, yi = x[idx], y[idx]
        if np.std(xi) == 0 or np.std(yi) == 0:
            draws[i] = np.nan
            continue
        draws[i] = float(np.corrcoef(xi, yi)[0, 1])
    valid = draws[~np.isnan(draws)]
    lower, upper = np.quantile(valid, [0.025, 0.975])
    return [float(lower), float(upper)]


def reliability_table(team_games: pd.DataFrame) -> dict[str, Any]:
    panel = (
        team_games.dropna(subset=["ol_continuity"])
        .groupby(["team", "season"], sort=False)["ol_continuity"]
        .mean()
        .rename("ol_season_mean")
        .reset_index()
    )
    league_mean = panel.groupby("season")["ol_season_mean"].transform("mean")
    panel["ol_continuity_centered"] = panel["ol_season_mean"] - league_mean

    pairs = year_over_year_pairs(panel, "ol_continuity_centered")
    x = pairs["value_t"].to_numpy(dtype=float)
    y = pairs["value_t1"].to_numpy(dtype=float)
    yoy_pearson = float(np.corrcoef(x, y)[0, 1])
    yoy_spearman = float(pairs["value_t"].rank().corr(pairs["value_t1"].rank()))
    yoy_ci = bootstrap_pearson_ci(x, y, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)

    weeks = team_games.dropna(subset=["ol_continuity"]).copy()
    weeks["is_odd"] = weeks["week"] % 2 == 1
    halves = (
        weeks.groupby(["team", "season", "is_odd"], sort=False)["ol_continuity"]
        .mean()
        .unstack("is_odd")
    )
    halves = halves.rename(columns={True: "odd", False: "even"}).dropna()
    sx = halves["odd"].to_numpy(dtype=float)
    sy = halves["even"].to_numpy(dtype=float)
    split_pearson = float(np.corrcoef(sx, sy)[0, 1])
    split_ci = bootstrap_pearson_ci(sx, sy, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)

    return {
        "ol_continuity": {
            "yoy_n_pairs": len(pairs),
            "yoy_pearson": yoy_pearson,
            "yoy_pearson_ci95": yoy_ci,
            "yoy_spearman": yoy_spearman,
            "split_half_n_pairs": len(halves),
            "split_half_pearson": split_pearson,
            "split_half_pearson_ci95": split_ci,
            "excluded_on_reliability": bool(yoy_ci[1] <= 0.0 and split_ci[1] <= 0.0),
        }
    }


def block_bootstrap_two_group(
    df: pd.DataFrame,
    *,
    flag_col: str,
    value_col: str,
    block_col: str,
    samples: int,
    seed: int,
) -> np.ndarray:
    blocks, block_index = np.unique(df[block_col].to_numpy(), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    block_count = len(blocks)
    values = df[value_col].to_numpy(dtype=np.float64)
    flag = df[flag_col].to_numpy(dtype=bool)

    sums: dict[bool, np.ndarray] = {}
    counts: dict[bool, np.ndarray] = {}
    for group in (True, False):
        mask = flag == group
        sums[group] = np.bincount(
            block_index[mask], weights=values[mask], minlength=block_count
        ).astype(np.float64)
        counts[group] = np.bincount(block_index[mask], minlength=block_count).astype(np.float64)

    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)
    subset_count = drawn @ counts[True]
    complement_count = drawn @ counts[False]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_subset = (drawn @ sums[True]) / subset_count
        mean_complement = (drawn @ sums[False]) / complement_count
    gap = (mean_subset - mean_complement) * 100.0
    valid = (subset_count > 0) & (complement_count > 0)
    return gap[valid]


def summarize(
    df: pd.DataFrame,
    *,
    flag: pd.Series,
    value_col: str,
    block_col: str,
    sign: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    n_total = len(df)
    n_flag = int(flag.sum())
    n_complement = n_total - n_flag
    if n_flag == 0 or n_complement == 0:
        return {
            "n_total": n_total,
            "n_flag": n_flag,
            "n_complement": n_complement,
            "insufficient_data": True,
        }

    work = df.copy()
    work["_flag"] = flag.to_numpy()
    subset_mean = float(work.loc[work["_flag"], value_col].mean())
    complement_mean = float(work.loc[~work["_flag"], value_col].mean())
    raw_gap_fraction = subset_mean - complement_mean
    fraction_of_slate = n_flag / n_total
    full_slate_effect_pts = scale_subset_effect(
        raw_gap_fraction, sign=sign, fraction_of_slate=fraction_of_slate
    )

    draws = block_bootstrap_two_group(
        work, flag_col="_flag", value_col=value_col, block_col=block_col, samples=samples, seed=seed
    )
    dropped = samples - len(draws)
    signed_draws = sign * draws
    scaled_draws = signed_draws * fraction_of_slate
    lower, upper = (
        np.quantile(scaled_draws, [0.025, 0.975]) if len(scaled_draws) else (np.nan, np.nan)
    )

    return {
        "n_total": n_total,
        "n_flag": n_flag,
        "n_complement": n_complement,
        "n_blocks": int(work[block_col].nunique()),
        "subset_mean": subset_mean,
        "complement_mean": complement_mean,
        "raw_gap_pts": sign * raw_gap_fraction * 100.0,
        "fraction_of_slate": fraction_of_slate,
        "full_slate_effect_pts": full_slate_effect_pts,
        "ci95_scaled": [float(lower), float(upper)],
        "probability_positive": float(np.mean(scaled_draws > 0)) if len(scaled_draws) else np.nan,
        "bootstrap_samples": samples,
        "dropped_draws": int(dropped),
        "insufficient_data": False,
    }


def score_cell(
    df: pd.DataFrame,
    name: str,
    *,
    flag: pd.Series,
    missing_mask: pd.Series,
    value_col: str,
    sign: int,
    description: str,
    reliability_note: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    flag = flag.fillna(False).astype(bool)
    missing_mask = missing_mask.fillna(False).astype(bool)
    week_blocked = summarize(
        df,
        flag=flag,
        value_col=value_col,
        block_col="week_block",
        sign=sign,
        samples=samples,
        seed=seed,
    )
    season_blocked = summarize(
        df,
        flag=flag,
        value_col=value_col,
        block_col="season",
        sign=sign,
        samples=samples,
        seed=seed,
    )
    return {
        "name": name,
        "description": description,
        "sign_dir": sign,
        "reliability_note": reliability_note,
        "n_flag": int(flag.sum()),
        "n_missing_required_data": int(missing_mask.sum()),
        "week_blocked": week_blocked,
        "season_blocked_secondary": season_blocked,
    }


def leakage_check(team_games: pd.DataFrame, starters: pd.DataFrame) -> dict[str, Any]:
    sample = team_games.dropna(subset=["ol_trailing2"]).sample(
        n=min(LEAKAGE_CHECK_ROWS, int(team_games["ol_trailing2"].notna().sum())),
        random_state=BOOTSTRAP_SEED,
    )
    mismatches = 0
    checked = 0
    failures: list[dict[str, Any]] = []
    for row in sample.itertuples(index=False):
        prior_starters = starters.loc[
            (starters["team"] == row.team)
            & (
                (starters["season"] < row.season)
                | ((starters["season"] == row.season) & (starters["week"] < row.week))
            )
        ]
        same_season = prior_starters.loc[prior_starters["season"] == row.season]
        same_season = same_season.sort_values("week")
        recent = same_season.tail(2)
        values = []
        for _, game in recent.iterrows():
            match = team_games.loc[
                (team_games["team"] == row.team)
                & (team_games["season"] == row.season)
                & (team_games["week"] == game["week"])
            ]
            if len(match) == 1 and isinstance(match.iloc[0]["starters"], frozenset):
                prev_match = same_season.loc[same_season["week"] < game["week"]]
                if len(prev_match) >= 1:
                    prev = prev_match.iloc[-1]["starters"]
                    cur = match.iloc[0]["starters"]
                    if isinstance(prev, frozenset):
                        values.append(len(cur & prev) / 5.0)
        recomputed = float(np.mean(values)) if values else np.nan
        target = float(row.ol_trailing2)
        ok = bool(np.isnan(target)) if np.isnan(recomputed) else abs(recomputed - target) < 1e-9
        if not ok:
            mismatches += 1
            if len(failures) < 5:
                failures.append(
                    {
                        "team": row.team,
                        "season": int(row.season),
                        "week": int(row.week),
                        "pipeline": target,
                        "recomputed": recomputed,
                    }
                )
        checked += 1
    return {
        "rows_checked": checked,
        "mismatches": mismatches,
        "passed": mismatches == 0,
        "example_failures": failures,
        "method": (
            "recomputed trailing-2 continuity from strictly-prior games only "
            "(season<target or same season week<target) and compared to pipeline flags"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--snap-counts", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    started = time.time()
    schedules_path = args.schedules or _latest_schedules()
    snaps_path = args.snap_counts or _latest_snap_counts()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "ol_continuity_screen" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {schedules_path} ===")
    schedules = load_schedules(schedules_path)
    print(f"REG {SEASON_START}-{SEASON_END} close-graded games: {len(schedules)}")

    print(f"=== loading snap counts {snaps_path} ===")
    snaps = pd.read_parquet(snaps_path)
    starters = build_starters(snaps)
    print(f"team-games with OL starter sets: {len(starters)}")

    team_games_full = build_team_games(schedules.attrs["full_reg"], starters)
    team_games_full, thresholds = add_traits(team_games_full)
    trait_cols = [
        "game_id",
        "team",
        "ol_continuity",
        "ol_recent",
        "ol_trailing2",
        "prior_ol_centered",
    ]
    team_games = build_team_games(schedules, starters)
    team_games = team_games.drop(columns=["starters"]).merge(
        team_games_full[trait_cols], on=["game_id", "team"], how="left"
    )
    defined = int(team_games["ol_trailing2"].notna().sum())
    print(f"team-game rows: {len(team_games)}; trailing2 defined: {defined}")

    reliabilities = reliability_table(team_games_full)
    rel = reliabilities["ol_continuity"]
    print("\n=== reliability (measured before scoring) ===")
    print(
        f"  YoY Pearson {rel['yoy_pearson']:+.3f} "
        f"[{rel['yoy_pearson_ci95'][0]:+.3f}, {rel['yoy_pearson_ci95'][1]:+.3f}] "
        f"Spearman {rel['yoy_spearman']:+.3f} n={rel['yoy_n_pairs']}"
    )
    print(
        f"  split-half Pearson {rel['split_half_pearson']:+.3f} "
        f"[{rel['split_half_pearson_ci95'][0]:+.3f}, {rel['split_half_pearson_ci95'][1]:+.3f}] "
        f"n={rel['split_half_n_pairs']} excluded={rel['excluded_on_reliability']}"
    )

    print("\n=== pooled thresholds ===")
    for key, value in thresholds.items():
        print(f"  {key}: {value:.4f}")

    leak = leakage_check(team_games_full, starters)
    print(f"\n=== leakage self-check: passed={leak['passed']} ({leak['rows_checked']} rows) ===")

    rel_note = (
        f"YoY Pearson {rel['yoy_pearson']:+.3f}, 95% CI "
        f"[{rel['yoy_pearson_ci95'][0]:+.3f},{rel['yoy_pearson_ci95'][1]:+.3f}], "
        f"n={rel['yoy_n_pairs']}; split-half Pearson {rel['split_half_pearson']:+.3f}, "
        f"95% CI [{rel['split_half_pearson_ci95'][0]:+.3f},"
        f"{rel['split_half_pearson_ci95'][1]:+.3f}], n={rel['split_half_n_pairs']}"
    )

    cells: list[dict[str, Any]] = []

    cells.append(
        score_cell(
            team_games,
            "ol_low_continuity_fade",
            flag=team_games["ol_trailing2"] <= thresholds["trailing2_q25"],
            missing_mask=team_games["ol_trailing2"].isna(),
            value_col="team_covered",
            sign=-1,
            description=(
                "Trailing-2-game OL continuity <= pooled Q25 vs the field. Predicted NEGATIVE "
                "on team_covered (docs/ol_continuity_screen.md C1)."
            ),
            reliability_note=rel_note,
            samples=args.samples,
            seed=args.seed,
        )
    )

    cells.append(
        score_cell(
            team_games,
            "ol_high_continuity_back",
            flag=team_games["ol_trailing2"] >= thresholds["trailing2_q75"],
            missing_mask=team_games["ol_trailing2"].isna(),
            value_col="team_covered",
            sign=1,
            description=(
                "Trailing-2-game OL continuity >= pooled Q75 vs the field. Predicted POSITIVE "
                "on team_covered (docs/ol_continuity_screen.md C2)."
            ),
            reliability_note=rel_note,
            samples=args.samples,
            seed=args.seed,
        )
    )

    cells.append(
        score_cell(
            team_games,
            "ol_acute_overhaul_fade",
            flag=team_games["ol_recent"] <= ACUTE_OVERHAUL_MAX,
            missing_mask=team_games["ol_recent"].isna(),
            value_col="team_covered",
            sign=-1,
            description=(
                f"Most recent game OL continuity <= {ACUTE_OVERHAUL_MAX:.1f} (<=2 of 5 returning "
                "starters). Predicted NEGATIVE on team_covered "
                "(docs/ol_continuity_screen.md C3)."
            ),
            reliability_note=rel_note,
            samples=args.samples,
            seed=args.seed,
        )
    )

    early = team_games.loc[team_games["week"] <= EARLY_WEEK_MAX].reset_index(drop=True)
    cells.append(
        score_cell(
            early,
            "ol_prior_season_weak_early_fade",
            flag=early["prior_ol_centered"] <= thresholds["prior_centered_q25"],
            missing_mask=early["prior_ol_centered"].isna(),
            value_col="team_covered",
            sign=-1,
            description=(
                f"Weeks 1-{EARLY_WEEK_MAX} only; prior-season mean OL continuity centered "
                "<= pooled Q25. Predicted NEGATIVE on team_covered "
                "(docs/ol_continuity_screen.md C4)."
            ),
            reliability_note=rel_note,
            samples=args.samples,
            seed=args.seed,
        )
    )

    print("\n=== results ===")
    for cell in cells:
        wb = cell["week_blocked"]
        sb = cell["season_blocked_secondary"]
        print(f"\n--- {cell['name']} ---")
        if wb.get("insufficient_data"):
            print("  insufficient data")
            continue
        print(
            f"  n_flag={cell['n_flag']} n_total={wb['n_total']} "
            f"n_missing_required_data={cell['n_missing_required_data']}"
        )
        print(
            f"  full_slate_effect={wb['full_slate_effect_pts']:+.4f}pts week-blocked 95% "
            f"[{wb['ci95_scaled'][0]:+.4f}, {wb['ci95_scaled'][1]:+.4f}] "
            f"P+={wb['probability_positive']:.4f}"
        )
        if not sb.get("insufficient_data"):
            print(
                f"  [season-blocked secondary] 95% [{sb['ci95_scaled'][0]:+.4f}, "
                f"{sb['ci95_scaled'][1]:+.4f}] P+={sb['probability_positive']:.4f}"
            )

    configuration = {
        "command": "ol-continuity-screen",
        "snap_counts_snapshot": str(snaps_path),
        "schedules": str(schedules_path),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "starter_offense_pct": STARTER_OFFENSE_PCT,
        "acute_overhaul_max": ACUTE_OVERHAUL_MAX,
        "early_week_max": EARLY_WEEK_MAX,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "n_reg_games": len(schedules),
        "n_team_game_rows": len(team_games),
        "snap_counts_snapshot": str(snaps_path),
        "reliabilities": reliabilities,
        "thresholds": thresholds,
        "leakage_check": leak,
        "predeclaration": (
            "docs/ol_continuity_screen.md (frozen before this script scored anything)"
        ),
        "results": cells,
        "provenance": artifact_provenance(configuration, schedules_path, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="ol-continuity-screen",
        metrics=payload,
        notes=(
            "Offensive-line continuity battery (4 predeclared cells); every cell recorded "
            "regardless of sign, per AGENTS.md binding taxonomy."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
