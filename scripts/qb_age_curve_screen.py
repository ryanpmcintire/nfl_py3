"""Quarterback age/experience trajectory screen (4 predeclared cells).

Predeclaration: docs/qb_age_curve_screen.md, frozen before this script was run
against any cover outcome. The experience axis is career start count (age and
blitz rate are not available locally; both substitutions are disclosed in the
doc). Binding taxonomy owned verbatim per AGENTS.md: an interval that crosses
zero is NEVER a closing ground; only refuted mechanism (resolved wrong sign or
no split-half reliability) or bounded-by-positive-control closes a line;
everything else is unresolved_below_power, recorded with probability_positive.
This script only measures; it records nothing to the weak-signal registry.

Method copied algorithm-identical from scripts/redzone_reversion_screen.py
(block_bootstrap_two_group / summarize / score_cell / bootstrap_pearson_ci).

Writes artifacts/qb_age_curve_screen/<UTC stamp>/results.json and the
experiment-registry row under registry/experiments/qb-age-curve-screen/.
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
from nfl_ats.pbp import analysis_plays, latest_pbp_snapshot, load_pbp_snapshot  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

SEASON_START = 2009
SEASON_END = 2025
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260821
FIRST_YEAR_MAX_STARTS = 6
VETERAN_MIN_STARTS = 200
ROOKIE_LATE_WEEK_MIN = 9
VETERAN_LATE_WEEK_MIN = 13
SPLIT_HALF_MIN_GAMES = 4

SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "home_team",
    "away_team",
    "result",
    "spread_line",
]


def _latest_schedules() -> Path:
    candidates = sorted((REPO / "data/raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


def load_schedules(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    available = [c for c in SCHEDULE_COLUMNS if c in raw.columns]
    df = raw.loc[:, available].copy()
    df = df.loc[df["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)
    df["home_team"] = df["home_team"].replace(TEAM_ABBREVIATION_ALIASES)
    df["away_team"] = df["away_team"].replace(TEAM_ABBREVIATION_ALIASES)
    df = add_ats_outcomes(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)
    df["week_block"] = df["season"] * 100 + df["week"]
    return df


def _alias_team(values: pd.Series) -> pd.Series:
    return values.astype("string").replace(TEAM_ABBREVIATION_ALIASES)


def build_start_table(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = pbp.copy()
    for column in ("pass_attempt", "play_id", "season", "week"):
        plays[column] = pd.to_numeric(plays[column], errors="coerce")
    if "cpoe" in plays.columns:
        plays["cpoe"] = pd.to_numeric(plays["cpoe"], errors="coerce")
    else:
        plays["cpoe"] = np.nan
    pass_plays = plays.loc[
        plays["pass_attempt"].fillna(0).eq(1) & plays["passer_player_id"].notna()
    ].copy()
    pass_plays["posteam"] = _alias_team(pass_plays["posteam"])
    counts = (
        pass_plays.groupby(["game_id", "posteam", "passer_player_id"], sort=False)
        .agg(
            attempts=("play_id", "size"),
            first_play=("play_id", "min"),
            passer_name=("passer_player_name", "first"),
            season=("season", "first"),
            week=("week", "first"),
            cpoe_mean=("cpoe", "mean"),
        )
        .reset_index()
    )
    counts = counts.sort_values(
        ["game_id", "posteam", "attempts", "first_play"],
        ascending=[True, True, False, True],
    )
    starters = counts.drop_duplicates(["game_id", "posteam"], keep="first")
    starters = starters.rename(
        columns={
            "posteam": "team",
            "passer_player_id": "qb_id",
            "passer_name": "qb_name",
        }
    )
    starters["season"] = starters["season"].astype(int)
    starters["week"] = starters["week"].astype(int)
    starters = starters.loc[starters["season"].between(SEASON_START, SEASON_END)]
    return compute_career_axes(starters).reset_index(drop=True)


def compute_career_axes(starters: pd.DataFrame) -> pd.DataFrame:
    ordered = starters.sort_values(["qb_id", "season", "week"]).copy()
    ordered["prior_total"] = ordered.groupby("qb_id").cumcount()
    ordered["prior_same_season"] = ordered.groupby(["qb_id", "season"]).cumcount()
    ordered["career_starts_entering"] = ordered["prior_total"] - ordered["prior_same_season"]
    ordered["first_start_season"] = ordered.groupby("qb_id")["season"].transform("min")
    return ordered


def _pressure_panel_from_plays(plays: pd.DataFrame) -> pd.DataFrame:
    plays = plays.loc[plays["season_type"].eq("REG")].copy()
    plays["season"] = pd.to_numeric(plays["season"], errors="raise").astype(int)
    plays = plays.loc[plays["season"].between(SEASON_START, SEASON_END)].copy()
    plays["posteam"] = _alias_team(plays["posteam"])
    plays["defteam"] = _alias_team(plays["defteam"])
    for column in ("qb_dropback", "sack", "qb_hit"):
        plays[column] = pd.to_numeric(plays[column], errors="coerce").fillna(0)
    panel = (
        plays.groupby(["season", "defteam"], sort=False)
        .agg(
            dropbacks=("qb_dropback", "sum"),
            sacks=("sack", "sum"),
            qb_hits=("qb_hit", "sum"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    panel["pressure_rate_allowed"] = (panel["sacks"] + panel["qb_hits"]) / panel[
        "dropbacks"
    ].replace(0, np.nan)
    league_mean = panel.groupby("season")["pressure_rate_allowed"].transform("mean")
    panel["pressure_rate_centered"] = panel["pressure_rate_allowed"] - league_mean
    return panel.sort_values(["season", "team"]).reset_index(drop=True)


def build_pressure_panel(pbp: pd.DataFrame) -> pd.DataFrame:
    return _pressure_panel_from_plays(analysis_plays(pbp))


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


def pearson_report(pairs: pd.DataFrame) -> dict[str, Any]:
    x = pairs["value_t"].to_numpy(dtype=float)
    y = pairs["value_t1"].to_numpy(dtype=float)
    ci = bootstrap_pearson_ci(x, y, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
    return {
        "n_pairs": len(pairs),
        "yoy_pearson": float(np.corrcoef(x, y)[0, 1]),
        "ci95": ci,
        "excluded_on_reliability": bool(ci[1] <= 0.0),
    }


def rookie_split_half(starters: pd.DataFrame) -> dict[str, Any]:
    rookies = starters.loc[starters["career_starts_entering"] <= FIRST_YEAR_MAX_STARTS].copy()
    rookies = rookies.dropna(subset=["cpoe_mean"])
    rookies = rookies.sort_values(["team", "season", "week"]).reset_index(drop=True)
    rookies["game_index"] = rookies.groupby(["team", "season"]).cumcount()
    sizes = rookies.groupby(["team", "season"]).size()
    keep = sizes[sizes >= SPLIT_HALF_MIN_GAMES].index
    rookies = rookies.set_index(["team", "season"]).loc[keep].reset_index()
    rookies["parity"] = rookies["game_index"] % 2
    pivoted = rookies.pivot_table(
        index=["team", "season"], columns="parity", values="cpoe_mean", aggfunc="mean"
    ).dropna()
    pivoted.columns = ["even", "odd"]
    report = pearson_report(
        pd.DataFrame(
            {
                "value_t": pivoted["even"].reset_index(drop=True),
                "value_t1": pivoted["odd"].reset_index(drop=True),
            }
        )
    )
    report["construct"] = "odd-vs-even game-index split of first-year-starter game CPOE"
    report["min_games_per_team_season"] = SPLIT_HALF_MIN_GAMES
    return report


def build_long_table(
    schedules: pd.DataFrame, starters: pd.DataFrame, pressure_panel: pd.DataFrame
) -> pd.DataFrame:
    starter_cols = [
        "game_id",
        "team",
        "qb_id",
        "qb_name",
        "career_starts_entering",
        "first_start_season",
    ]
    prior_pressure = pressure_panel[["team", "season", "pressure_rate_centered"]].copy()
    prior_pressure["season"] = prior_pressure["season"] + 1
    prior_pressure = prior_pressure.rename(columns={"pressure_rate_centered": "prior_pressure"})
    sides = []
    for is_home in (True, False):
        team_col = "home_team" if is_home else "away_team"
        side = pd.DataFrame(
            {
                "game_id": schedules["game_id"],
                "season": schedules["season"],
                "week": schedules["week"],
                "week_block": schedules["week_block"],
                "team": schedules[team_col],
                "team_covered": (
                    schedules["home_cover"] if is_home else 1.0 - schedules["home_cover"]
                ),
            }
        )
        side = side.merge(starters[starter_cols], on=["game_id", "team"], how="left")
        side = side.merge(prior_pressure, on=["team", "season"], how="left")
        sides.append(side)
    return pd.concat(sides, ignore_index=True).reset_index(drop=True)


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
    fraction_denominator: int,
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
    fraction_of_slate = n_flag / fraction_denominator
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
        "n_total_population": n_total,
        "n_flag": n_flag,
        "n_complement_in_population": n_complement,
        "n_blocks": int(work[block_col].nunique()),
        "subset_mean": subset_mean,
        "complement_mean": complement_mean,
        "raw_gap_pts": sign * raw_gap_fraction * 100.0,
        "fraction_of_full_slate": fraction_of_slate,
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
    fraction_denominator: int,
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
        fraction_denominator=fraction_denominator,
    )
    season_blocked = summarize(
        df,
        flag=flag,
        value_col=value_col,
        block_col="season",
        sign=sign,
        samples=samples,
        seed=seed,
        fraction_denominator=fraction_denominator,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    started = time.time()
    schedules_path = args.schedules or _latest_schedules()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "qb_age_curve_screen" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {schedules_path} ===")
    schedules = load_schedules(schedules_path)
    print(f"REG {SEASON_START}-{SEASON_END} close-graded games: {len(schedules)}")

    snapshot = latest_pbp_snapshot(REPO / "data/pbp/raw")
    print(f"=== loading PBP snapshot {snapshot.snapshot_id} ===")
    pbp = load_pbp_snapshot(snapshot)

    print("=== identifying starting QB per team-game ===")
    starters = build_start_table(pbp)
    print(f"start-identified team-games: {len(starters)}")

    print("=== building defensive pressure panel ===")
    pressure_panel = build_pressure_panel(pbp)
    print(f"defense-season rows: {len(pressure_panel)}")

    pressure_rel = pearson_report(year_over_year_pairs(pressure_panel, "pressure_rate_centered"))
    split_half = rookie_split_half(starters)
    print("\n=== reliability (measured before scoring) ===")
    print(
        f"  pressure_rate_allowed YoY Pearson {pressure_rel['yoy_pearson']:+.3f} "
        f"[{pressure_rel['ci95'][0]:+.3f}, {pressure_rel['ci95'][1]:+.3f}] "
        f"n={pressure_rel['n_pairs']} excluded={pressure_rel['excluded_on_reliability']}"
    )
    print(
        f"  rookie CPOE split-half Pearson {split_half['yoy_pearson']:+.3f} "
        f"[{split_half['ci95'][0]:+.3f}, {split_half['ci95'][1]:+.3f}] "
        f"n={split_half['n_pairs']} excluded={split_half['excluded_on_reliability']}"
    )

    long_df = build_long_table(schedules, starters, pressure_panel)
    full_slate_n = len(long_df)
    threshold_q75 = float(pressure_panel["pressure_rate_centered"].quantile(0.75))
    print(f"\nprior-pressure Q75 threshold: {threshold_q75:.4f}")

    long_df["is_first_year"] = long_df["career_starts_entering"] <= FIRST_YEAR_MAX_STARTS
    long_df["is_veteran"] = long_df["career_starts_entering"] >= VETERAN_MIN_STARTS
    long_df["is_second_year"] = long_df["first_start_season"] == long_df["season"] - 1

    pressure_note = (
        f"pressure-rate YoY Pearson {pressure_rel['yoy_pearson']:+.3f}, 95% CI "
        f"[{pressure_rel['ci95'][0]:+.3f},{pressure_rel['ci95'][1]:+.3f}], "
        f"n={pressure_rel['n_pairs']} defense-season pairs"
    )
    form_note = (
        f"rookie CPOE odd/even split-half Pearson {split_half['yoy_pearson']:+.3f}, "
        f"95% CI [{split_half['ci95'][0]:+.3f},{split_half['ci95'][1]:+.3f}], "
        f"n={split_half['n_pairs']} team-seasons"
    )

    rookies = long_df.loc[long_df["is_first_year"]].reset_index(drop=True)
    veterans = long_df.loc[long_df["is_veteran"]].reset_index(drop=True)

    cells: list[dict[str, Any]] = []

    cells.append(
        score_cell(
            rookies,
            "rookie_late_improvement",
            flag=rookies["week"] >= ROOKIE_LATE_WEEK_MIN,
            missing_mask=pd.Series(False, index=rookies.index),
            value_col="team_covered",
            sign=1,
            description=(
                "First-year starters (<=6 career starts entering the season), weeks 9+ vs "
                "weeks 1-8 within the same population. Predicted POSITIVE on team_covered "
                "(within-season improvement; docs/qb_age_curve_screen.md C1)."
            ),
            reliability_note=form_note,
            samples=args.samples,
            seed=args.seed,
            fraction_denominator=full_slate_n,
        )
    )

    cells.append(
        score_cell(
            veterans,
            "veteran_late_fade",
            flag=veterans["week"] >= VETERAN_LATE_WEEK_MIN,
            missing_mask=pd.Series(False, index=veterans.index),
            value_col="team_covered",
            sign=-1,
            description=(
                "Very-high-experience starters (>=200 career starts entering the season), "
                "weeks 13+ vs weeks 1-12 within the same population. Predicted NEGATIVE on "
                "team_covered (late fade; docs/qb_age_curve_screen.md C2)."
            ),
            reliability_note=form_note,
            samples=args.samples,
            seed=args.seed,
            fraction_denominator=full_slate_n,
        )
    )

    cells.append(
        score_cell(
            rookies,
            "rookie_vs_pressure",
            flag=rookies["prior_pressure"] >= threshold_q75,
            missing_mask=rookies["prior_pressure"].isna(),
            value_col="team_covered",
            sign=-1,
            description=(
                "First-year starters facing a defense whose PRIOR-season centered pressure "
                f"rate allowed is >= Q75 ({threshold_q75:.4f}; blitz rate unavailable, "
                "pressure proxy disclosed). Predicted NEGATIVE on team_covered "
                "(docs/qb_age_curve_screen.md C3)."
            ),
            reliability_note=pressure_note,
            samples=args.samples,
            seed=args.seed,
            fraction_denominator=full_slate_n,
        )
    )

    cells.append(
        score_cell(
            long_df,
            "second_year_jump",
            flag=long_df["is_second_year"],
            missing_mask=long_df["career_starts_entering"].isna(),
            value_col="team_covered",
            sign=1,
            description=(
                "Starters whose FIRST career start occurred last season (year-2 starters) vs "
                "the rest of the slate. Predicted POSITIVE on team_covered (second-year jump; "
                "docs/qb_age_curve_screen.md C4)."
            ),
            reliability_note=form_note,
            samples=args.samples,
            seed=args.seed,
            fraction_denominator=full_slate_n,
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
            f"  n_flag={cell['n_flag']} population={wb['n_total_population']} "
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
        "command": "qb-age-curve-screen",
        "pbp_snapshot": snapshot.snapshot_id,
        "schedules": str(schedules_path),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "first_year_max_starts": FIRST_YEAR_MAX_STARTS,
        "veteran_min_starts": VETERAN_MIN_STARTS,
        "rookie_late_week_min": ROOKIE_LATE_WEEK_MIN,
        "veteran_late_week_min": VETERAN_LATE_WEEK_MIN,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "n_reg_games": len(schedules),
        "pbp_snapshot": snapshot.snapshot_id,
        "n_start_identified_team_games": len(starters),
        "reliabilities": {
            "pressure_rate_allowed": pressure_rel,
            "rookie_cpoe_split_half": split_half,
        },
        "thresholds": {"prior_pressure_q75": threshold_q75},
        "predeclaration": (
            "docs/qb_age_curve_screen.md (frozen before this script scored anything)"
        ),
        "results": cells,
        "provenance": artifact_provenance(configuration, schedules_path, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="qb-age-curve-screen",
        metrics=payload,
        notes=(
            "QB age/experience trajectory battery (4 predeclared cells); every cell recorded "
            "regardless of sign, per AGENTS.md binding taxonomy."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
