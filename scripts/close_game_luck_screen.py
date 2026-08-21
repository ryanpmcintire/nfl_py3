"""Close-game and turnover LUCK regression screen (6 predeclared cells).

Predeclaration: docs/close_game_luck_screen.md, frozen before this script was
run against any cover outcome. Reliability-first: year-over-year Pearson and
Spearman on season-centered team-season luck traits are measured before any
cell exists. Binding taxonomy owned verbatim per AGENTS.md: an interval that
crosses zero is NEVER a closing ground; only refuted mechanism (resolved wrong
sign or no reliability) or bounded-by-positive-control closes a line;
everything else is unresolved_below_power, recorded with probability_positive.
This script only measures; it records nothing to the weak-signal registry.

Method copied algorithm-identical from scripts/redzone_reversion_screen.py
(block_bootstrap_two_group / summarize / score_cell / year_over_year_pairs /
bootstrap_pearson_ci), which itself copies scripts/team_style_screen.py.

Writes artifacts/close_game_luck_screen/<UTC stamp>/results.json and the
experiment-registry row under registry/experiments/close-game-luck-screen/.
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
from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

SEASON_START = 2009
SEASON_END = 2025
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260821
EARLY_WEEK_MAX = 8
ONE_SCORE_MARGIN_MAX = 8
LUCK_TRAITS = ("one_score_luck", "turnover_diff_per_game", "takeaway_share")

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
    n_before_push_drop = len(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)
    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = n_before_push_drop - len(df)
    df["week_block"] = df["season"] * 100 + df["week"]
    return df


def _alias_team(values: pd.Series) -> pd.Series:
    return values.astype("string").replace(TEAM_ABBREVIATION_ALIASES)


def build_giveaways_table(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = pbp.loc[pbp["season_type"].eq("REG")].copy()
    plays["season"] = pd.to_numeric(plays["season"], errors="raise").astype(int)
    plays = plays.loc[plays["season"].between(SEASON_START, SEASON_END)].copy()
    plays["posteam"] = _alias_team(plays["posteam"])
    plays["_turnover"] = pd.to_numeric(plays["interception"], errors="coerce").fillna(
        0.0
    ) + pd.to_numeric(plays["fumble_lost"], errors="coerce").fillna(0.0)
    giveaways = (
        plays.dropna(subset=["posteam"])
        .groupby(["game_id", "posteam"], sort=False)["_turnover"]
        .sum()
        .reset_index(name="giveaways")
    )
    return giveaways


def build_team_games(schedules: pd.DataFrame, giveaways: pd.DataFrame) -> pd.DataFrame:
    sides = []
    own = giveaways.rename(columns={"posteam": "team"})
    against = giveaways.rename(columns={"posteam": "opponent"})
    for is_home in (True, False):
        team_col = "home_team" if is_home else "away_team"
        opp_col = "away_team" if is_home else "home_team"
        side = pd.DataFrame(
            {
                "game_id": schedules["game_id"],
                "season": schedules["season"],
                "week": schedules["week"],
                "week_block": schedules["week_block"],
                "team": schedules[team_col],
                "opponent": schedules[opp_col],
                "margin_for": schedules["result"] if is_home else -schedules["result"],
            }
        )
        side = side.merge(own, on=["game_id", "team"], how="left")
        side = side.merge(against, on=["game_id", "opponent"], how="left", suffixes=("", "_opp"))
        side = side.rename(columns={"giveaways_opp": "takeaways"})
        sides.append(side)
    team_games = pd.concat(sides, ignore_index=True).reset_index(drop=True)
    team_games["giveaways"] = team_games["giveaways"].fillna(0.0)
    team_games["takeaways"] = team_games["takeaways"].fillna(0.0)
    team_games["win"] = team_games["margin_for"] > 0
    team_games["one_score"] = team_games["margin_for"].abs() <= ONE_SCORE_MARGIN_MAX
    team_games["one_score_win"] = team_games["win"] & team_games["one_score"]
    return team_games


def build_panel(team_games: pd.DataFrame) -> pd.DataFrame:
    panel = (
        team_games.groupby(["season", "team"], sort=False)
        .agg(
            games=("win", "size"),
            wins=("win", "sum"),
            one_score_games=("one_score", "sum"),
            one_score_wins=("one_score_win", "sum"),
            giveaways=("giveaways", "sum"),
            takeaways=("takeaways", "sum"),
        )
        .reset_index()
    )
    panel["one_score_luck"] = (
        panel["one_score_wins"] / panel["one_score_games"].replace(0, np.nan)
    ) - (panel["wins"] / panel["games"].replace(0, np.nan))
    panel["turnover_diff_per_game"] = (panel["takeaways"] - panel["giveaways"]) / panel[
        "games"
    ].replace(0, np.nan)
    total_turnovers = (panel["takeaways"] + panel["giveaways"]).replace(0, np.nan)
    panel["takeaway_share"] = panel["takeaways"] / total_turnovers
    for trait in LUCK_TRAITS:
        league_mean = panel.groupby("season")[trait].transform("mean")
        panel[f"{trait}_centered"] = panel[trait] - league_mean
    return panel.sort_values(["season", "team"]).reset_index(drop=True)


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


def reliability_table(panel: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for trait in LUCK_TRAITS:
        pairs = year_over_year_pairs(panel, f"{trait}_centered")
        x = pairs["value_t"].to_numpy(dtype=float)
        y = pairs["value_t1"].to_numpy(dtype=float)
        pearson = float(np.corrcoef(x, y)[0, 1])
        spearman = float(pairs["value_t"].rank().corr(pairs["value_t1"].rank()))
        ci = bootstrap_pearson_ci(x, y, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
        report[trait] = {
            "n_pairs": len(pairs),
            "yoy_pearson": pearson,
            "yoy_pearson_ci95": ci,
            "yoy_spearman": spearman,
            "excluded_on_reliability": bool(ci[1] <= 0.0),
        }
    return report


def _prior(panel: pd.DataFrame, trait: str) -> pd.DataFrame:
    shifted = panel[["team", "season", f"{trait}_centered"]].copy()
    shifted["season"] = shifted["season"] + 1
    shifted = shifted.rename(
        columns={f"{trait}_centered": f"prior_{trait}_centered", "team": "team"}
    )
    return shifted


def build_long_table(schedules: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    priors = [_prior(panel, trait) for trait in LUCK_TRAITS]
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
        for prior in priors:
            side = side.merge(prior, on=["team", "season"], how="left")
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
    output_dir: Path = args.output or (REPO / "artifacts" / "close_game_luck_screen" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {schedules_path} ===")
    schedules = load_schedules(schedules_path)
    print(f"REG {SEASON_START}-{SEASON_END} close-graded games: {len(schedules)}")

    snapshot = latest_pbp_snapshot(REPO / "data/pbp/raw")
    print(f"=== loading PBP snapshot {snapshot.snapshot_id} ===")
    pbp = load_pbp_snapshot(snapshot)
    giveaways = build_giveaways_table(pbp)
    team_games = build_team_games(schedules, giveaways)
    panel = build_panel(team_games)
    print(f"giveaway rows: {len(giveaways)}; team-season panel rows: {len(panel)}")

    reliabilities = reliability_table(panel)
    print("\n=== year-over-year reliability (centered traits) ===")
    for trait, row in reliabilities.items():
        print(
            f"  {trait}: Pearson {row['yoy_pearson']:+.3f} "
            f"[{row['yoy_pearson_ci95'][0]:+.3f}, {row['yoy_pearson_ci95'][1]:+.3f}] "
            f"Spearman {row['yoy_spearman']:+.3f} n={row['n_pairs']} "
            f"excluded={row['excluded_on_reliability']}"
        )

    thresholds = {
        "luck_q75": float(panel["one_score_luck_centered"].quantile(0.75)),
        "luck_q25": float(panel["one_score_luck_centered"].quantile(0.25)),
        "turnover_q75": float(panel["turnover_diff_per_game_centered"].quantile(0.75)),
        "turnover_q25": float(panel["turnover_diff_per_game_centered"].quantile(0.25)),
        "takeaway_share_q75": float(panel["takeaway_share_centered"].quantile(0.75)),
    }
    print("\n=== pooled-panel quartile thresholds ===")
    for key, value in thresholds.items():
        print(f"  {key}: {value:.4f}")

    long_df = build_long_table(schedules, panel)

    def note(trait: str) -> str:
        row = reliabilities[trait]
        return (
            f"YoY Pearson {row['yoy_pearson']:+.3f}, 95% CI "
            f"[{row['yoy_pearson_ci95'][0]:+.3f},{row['yoy_pearson_ci95'][1]:+.3f}], "
            f"n={row['n_pairs']} team-season pairs"
        )

    cells: list[dict[str, Any]] = []

    cells.append(
        score_cell(
            long_df,
            "one_score_over_fade",
            flag=long_df["prior_one_score_luck_centered"] >= thresholds["luck_q75"],
            missing_mask=long_df["prior_one_score_luck_centered"].isna(),
            value_col="team_covered",
            sign=-1,
            description=(
                "Prior-season top-quartile centered one-score luck (one-score win rate "
                "minus overall win rate) vs the field. Predicted NEGATIVE on team_covered "
                "(close-game-luck fade; docs/close_game_luck_screen.md L1)."
            ),
            reliability_note=note("one_score_luck"),
            samples=args.samples,
            seed=args.seed,
        )
    )

    cells.append(
        score_cell(
            long_df,
            "one_score_under_rebound",
            flag=long_df["prior_one_score_luck_centered"] <= thresholds["luck_q25"],
            missing_mask=long_df["prior_one_score_luck_centered"].isna(),
            value_col="team_covered",
            sign=1,
            description=(
                "Prior-season bottom-quartile centered one-score luck vs the field. "
                "Predicted POSITIVE on team_covered (rebound; "
                "docs/close_game_luck_screen.md L2)."
            ),
            reliability_note=note("one_score_luck"),
            samples=args.samples,
            seed=args.seed,
        )
    )

    cells.append(
        score_cell(
            long_df,
            "turnover_over_fade",
            flag=long_df["prior_turnover_diff_per_game_centered"] >= thresholds["turnover_q75"],
            missing_mask=long_df["prior_turnover_diff_per_game_centered"].isna(),
            value_col="team_covered",
            sign=-1,
            description=(
                "Prior-season top-quartile centered turnover differential per game vs the "
                "field. Predicted NEGATIVE on team_covered (turnover-regression fade; "
                "docs/close_game_luck_screen.md L3)."
            ),
            reliability_note=note("turnover_diff_per_game"),
            samples=args.samples,
            seed=args.seed,
        )
    )

    cells.append(
        score_cell(
            long_df,
            "turnover_under_rebound",
            flag=long_df["prior_turnover_diff_per_game_centered"] <= thresholds["turnover_q25"],
            missing_mask=long_df["prior_turnover_diff_per_game_centered"].isna(),
            value_col="team_covered",
            sign=1,
            description=(
                "Prior-season bottom-quartile centered turnover differential per game vs "
                "the field. Predicted POSITIVE on team_covered (rebound; "
                "docs/close_game_luck_screen.md L4)."
            ),
            reliability_note=note("turnover_diff_per_game"),
            samples=args.samples,
            seed=args.seed,
        )
    )

    cells.append(
        score_cell(
            long_df,
            "takeaway_share_extreme_fade",
            flag=long_df["prior_takeaway_share_centered"] >= thresholds["takeaway_share_q75"],
            missing_mask=long_df["prior_takeaway_share_centered"].isna(),
            value_col="team_covered",
            sign=-1,
            description=(
                "Prior-season top-quartile centered takeaway share (takeaways over total "
                "turnovers in the team's games; disclosed computable stand-in for pure "
                "fumble-recovery rate, which the local PBP snapshot cannot compute). "
                "Predicted NEGATIVE on team_covered (recovery-luck fade; "
                "docs/close_game_luck_screen.md L5)."
            ),
            reliability_note=note("takeaway_share"),
            samples=args.samples,
            seed=args.seed,
        )
    )

    early = long_df.loc[long_df["week"] <= EARLY_WEEK_MAX].reset_index(drop=True)
    cells.append(
        score_cell(
            early,
            "early_season_luck_fade",
            flag=(early["prior_one_score_luck_centered"] >= thresholds["luck_q75"])
            | (early["prior_turnover_diff_per_game_centered"] >= thresholds["turnover_q75"]),
            missing_mask=(
                early["prior_one_score_luck_centered"].isna()
                & early["prior_turnover_diff_per_game_centered"].isna()
            ),
            value_col="team_covered",
            sign=-1,
            description=(
                f"Weeks 1-{EARLY_WEEK_MAX} only; prior-season top-quartile on EITHER "
                "centered one-score luck OR centered turnover differential per game. "
                "Predicted NEGATIVE on team_covered (docs/close_game_luck_screen.md L6)."
            ),
            reliability_note=f"{note('one_score_luck')}; {note('turnover_diff_per_game')}",
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
        "command": "close-game-luck-screen",
        "pbp_snapshot": snapshot.snapshot_id,
        "schedules": str(schedules_path),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "early_week_max": EARLY_WEEK_MAX,
        "one_score_margin_max": ONE_SCORE_MARGIN_MAX,
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
        "reliabilities": reliabilities,
        "thresholds": thresholds,
        "predeclaration": (
            "docs/close_game_luck_screen.md (frozen before this script scored anything)"
        ),
        "results": cells,
        "provenance": artifact_provenance(configuration, schedules_path, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="close-game-luck-screen",
        metrics=payload,
        notes=(
            "Close-game/turnover LUCK regression battery (6 predeclared cells); every cell "
            "recorded regardless of sign, per AGENTS.md binding taxonomy."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
