"""PBP-08 team "personality"/style screen: 5 predeclared cells (2 identity, 3
matchup) against NFL REG close-grade cover outcomes, 2009-2025.

**Predeclaration**: ``docs/team_style.md``, written and frozen before this
script was run against any cover outcome. Do not add, remove, or redefine a
cell here without updating that document first -- that is precisely the
discipline AGENTS.md exists to protect.

**Binding taxonomy (owned verbatim, per AGENTS.md/CLAUDE.md)**: an interval
or CI that contains zero is NEVER grounds to reject, fail, or close an
experiment -- at this evaluator's ~2-point resolution "contains zero" is the
EXPECTED outcome for a real small signal. Only two closing grounds: (1)
refuted mechanism -- RESOLVED wrong sign (whole interval on the wrong side
of zero) or zero split-half reliability; (2) bounded by a positive control
proven able to detect an effect that size. Everything else is
``unresolved_below_power``: record with ``probability_positive``, never
"contains zero". If a record command errors, the verdict is wrong, not the
validator. Every cell below is recorded to the registry regardless of sign
or interval shape -- this script does not decide anything, it only measures.

**Method** (read both, reused/imported per the module docstrings there):
- ``block_bootstrap_two_group`` is algorithm-identical to
  ``scripts/nfl_weather_battery_screen.py``/``scripts/nfl_bias_battery_screen.py``
  (vectorized joint multinomial block resample, dropped-draw accounting).
- ``nfl_ats.experiment_runner.scale_subset_effect`` is IMPORTED directly
  (not reimplemented) for the full-slate-scaled point estimate.
- The team-perspective long table (one row per team per game, ``team_covered``)
  for the two identity cells reuses the exact pattern documented in
  ``scripts/nfl_bias_battery_screen.py::build_long_table``.

Population: NFL REG, close grade (``schedules.parquet`` ``spread_line``),
full local history 2009-2025. Style inputs are prior-FULL-SEASON, so 2009
games carry no trailing style and are reported as missing, not dropped from
the declared range. 20,000 bootstrap samples, seed 20260819, week-blocked
primary / season-blocked secondary.

Writes ``artifacts/team_style_screen/<UTC timestamp>/results.json``.
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
sys.path.append(str(REPO / "scripts"))

from _common import block_bootstrap_two_group, latest_schedules  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))

from team_style_features import (  # noqa: E402
    STYLE_DIMENSIONS,
    TEAM_SEASON_FACED_PATH,
    TEAM_SEASON_PATH,
)

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.experiment_runner import scale_subset_effect  # noqa: E402
from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

SEASON_START = 2009
SEASON_END = 2025
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819
OUTDOOR_ROOFS = frozenset({"outdoors", "open"})
HIGH_WIND_MPH = 15.0
QUARTILE = 0.75

# All 9 clear PER-07's own +0.320 "genuinely reliable" bar or sit with a 95%
# CI entirely above zero (docs/team_style.md reliability table); zero
# dimensions are excluded on reliability grounds.
RELIABLE_DIMENSIONS = STYLE_DIMENSIONS

SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "home_team",
    "away_team",
    "result",
    "spread_line",
    "roof",
    "surface",
    "temp",
    "wind",
]


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

    df = add_ats_outcomes(df)  # ats_margin, home_cover (reused verbatim)
    n_before_push_drop = len(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)
    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = n_before_push_drop - len(df)

    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")
    df["spread_line"] = pd.to_numeric(df["spread_line"], errors="coerce")
    df["outdoor"] = df["roof"].isin(OUTDOOR_ROOFS)
    df["week_block"] = df["season"] * 100 + df["week"]
    # Verified convention (scripts/nfl_weather_battery_screen.py comment,
    # cross-checked against add_ats_outcomes's ats_margin=result-spread_line):
    # spread_line > 0 -> HOME favored; spread_line < 0 -> AWAY favored.
    df["dog_cover"] = np.select(
        [df["spread_line"] < 0.0, df["spread_line"] > 0.0],
        [df["home_cover"], 1.0 - df["home_cover"]],
        default=np.nan,
    )
    return df


def add_identity_distance(team_season: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Z-score each reliable dimension (pooled sd across the 544-row panel)
    and return the L2 norm as ``identity_l2_distance``, plus the sds used.
    """

    result = team_season.copy()
    sds: dict[str, float] = {}
    z_cols = []
    for dim in RELIABLE_DIMENSIONS:
        centered = f"{dim}_centered"
        sd = float(result[centered].std(ddof=1))
        sds[dim] = sd
        z_col = f"{dim}_z"
        result[z_col] = result[centered] / sd if sd > 0 else np.nan
        z_cols.append(z_col)
    result["identity_l2_distance"] = np.sqrt((result[z_cols] ** 2).sum(axis=1))
    return result, sds


def _prior(table: pd.DataFrame, columns: list[str], rename_team: str) -> pd.DataFrame:
    """Shift a (season, team) table forward one season and rename ``team``
    to the join key for a specific side, so joining on (side_team, season)
    pulls the PRIOR season's value onto that season's game.
    """

    shifted = table[["team", "season", *columns]].copy()
    shifted["season"] = shifted["season"] + 1
    rename = {c: f"prior_{c}" for c in columns}
    rename["team"] = rename_team
    return shifted.rename(columns=rename)


def build_long_table(schedules: pd.DataFrame, team_season_l2: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, side); reuses the exact pattern documented in
    ``scripts/nfl_bias_battery_screen.py::build_long_table``.
    """

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
        prior = _prior(
            team_season_l2, ["short_pass_share_centered", "identity_l2_distance"], "team"
        )
        side = side.merge(prior, on=["team", "season"], how="left")
        sides.append(side)
    return pd.concat(sides, ignore_index=True).reset_index(drop=True)


def build_game_table(
    schedules: pd.DataFrame, team_season: pd.DataFrame, faced: pd.DataFrame
) -> pd.DataFrame:
    game = schedules.copy()

    prior_away_short = _prior(team_season, ["short_pass_share_centered"], "away_team")
    game = game.merge(prior_away_short, on=["away_team", "season"], how="left")

    prior_home_def = _prior(faced, ["shotgun_rate_faced_centered"], "home_team")
    game = game.merge(prior_home_def, on=["home_team", "season"], how="left")

    prior_home_pace = _prior(team_season, ["seconds_per_play_pace_centered"], "home_team")
    prior_home_pace = prior_home_pace.rename(
        columns={"prior_seconds_per_play_pace_centered": "home_prior_pace_centered"}
    )
    game = game.merge(prior_home_pace, on=["home_team", "season"], how="left")

    prior_away_pace = _prior(team_season, ["seconds_per_play_pace_centered"], "away_team")
    prior_away_pace = prior_away_pace.rename(
        columns={"prior_seconds_per_play_pace_centered": "away_prior_pace_centered"}
    )
    game = game.merge(prior_away_pace, on=["away_team", "season"], how="left")
    game["pace_diff_abs"] = (
        game["home_prior_pace_centered"] - game["away_prior_pace_centered"]
    ).abs()

    prior_away_deep = _prior(team_season, ["deep_share_centered"], "away_team")
    game = game.merge(prior_away_deep, on=["away_team", "season"], how="left")

    return game


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
    schedules_path = args.schedules or latest_schedules()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "team_style_screen" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {schedules_path} ===")
    schedules = load_schedules(schedules_path)
    print(f"REG {SEASON_START}-{SEASON_END} close-graded games: {len(schedules)}")

    team_season = pd.read_parquet(TEAM_SEASON_PATH)
    faced = pd.read_parquet(TEAM_SEASON_FACED_PATH)
    team_season_l2, _identity_sds = add_identity_distance(team_season)

    thresholds = {
        "short_pass_share_centered": float(
            team_season_l2["short_pass_share_centered"].quantile(QUARTILE)
        ),
        "deep_share_centered": float(team_season_l2["deep_share_centered"].quantile(QUARTILE)),
        "shotgun_rate_faced_centered": float(
            faced["shotgun_rate_faced_centered"].quantile(QUARTILE)
        ),
        "identity_l2_distance": float(team_season_l2["identity_l2_distance"].quantile(QUARTILE)),
    }
    print("=== quartile thresholds (0.75, 544-row 2009-2025 team-season panel) ===")
    for key, value in thresholds.items():
        print(f"  {key}: {value:.4f}")

    long_df = build_long_table(schedules, team_season_l2)
    game = build_game_table(schedules, team_season, faced)
    pace_threshold = float(game["pace_diff_abs"].quantile(QUARTILE))
    print(f"  pace_diff_abs (game-level): {pace_threshold:.4f}")

    reliability_notes = {
        "short_pass_share": (
            "YoY Pearson r +0.408, 95% CI [+0.324,+0.486], n=512 team-season pairs"
        ),
        "deep_share": "YoY Pearson r +0.306, 95% CI [+0.226,+0.382], n=512 team-season pairs",
        "seconds_per_play_pace": (
            "YoY Pearson r +0.489, 95% CI [+0.405,+0.567], n=512 team-season pairs"
        ),
        "shotgun_rate_faced": (
            "YoY Pearson r +0.278, 95% CI [+0.200,+0.353], n=512 team-season pairs"
        ),
        "identity_l2_distance": (
            "composite of 9 dimensions, each individually YoY-reliable "
            "(range +0.306 to +0.653, all 95% CI above zero); see docs/team_style.md"
        ),
    }

    cells: list[dict[str, Any]] = []

    # A1: distinct identity vs field (unsigned)
    flag_a1 = long_df["prior_identity_l2_distance"] >= thresholds["identity_l2_distance"]
    missing_a1 = long_df["prior_identity_l2_distance"].isna()
    cells.append(
        score_cell(
            long_df,
            "team_style_distinct_identity",
            flag=flag_a1,
            missing_mask=missing_a1,
            value_col="team_covered",
            sign=1,
            description=(
                "Top-quartile prior-season L2 distance (z-scored, 9 reliable dims) from "
                "league-mean style vs the field. UNSIGNED (docs/team_style.md)."
            ),
            reliability_note=reliability_notes["identity_l2_distance"],
            samples=args.samples,
            seed=args.seed,
        )
    )

    # A2: short-game identity vs field (unsigned)
    flag_a2 = long_df["prior_short_pass_share_centered"] >= thresholds["short_pass_share_centered"]
    missing_a2 = long_df["prior_short_pass_share_centered"].isna()
    cells.append(
        score_cell(
            long_df,
            "team_style_short_game_identity",
            flag=flag_a2,
            missing_mask=missing_a2,
            value_col="team_covered",
            sign=1,
            description=(
                "Top-quartile prior-season centered short_pass_share (air_yards<=5 share) vs "
                "the field. UNSIGNED (docs/team_style.md)."
            ),
            reliability_note=reliability_notes["short_pass_share"],
            samples=args.samples,
            seed=args.seed,
        )
    )

    # B1: short-game offense (away) vs pressure-style defense (home)
    flag_b1 = (
        game["prior_short_pass_share_centered"] >= thresholds["short_pass_share_centered"]
    ) & (game["prior_shotgun_rate_faced_centered"] >= thresholds["shotgun_rate_faced_centered"])
    missing_b1 = (
        game["prior_short_pass_share_centered"].isna()
        | game["prior_shotgun_rate_faced_centered"].isna()
    )
    cells.append(
        score_cell(
            game,
            "team_style_short_game_vs_pressure_defense",
            flag=flag_b1,
            missing_mask=missing_b1,
            value_col="home_cover",
            sign=-1,
            description=(
                "AWAY team prior-season top-quartile short_pass_share AND HOME defense "
                "prior-season top-quartile shotgun_rate_faced (pressure-style proxy). "
                "Predicted NEGATIVE on home_cover (docs/team_style.md)."
            ),
            reliability_note=(
                f"offense: {reliability_notes['short_pass_share']}; "
                f"defense proxy: {reliability_notes['shotgun_rate_faced']}"
            ),
            samples=args.samples,
            seed=args.seed,
        )
    )

    # B2: pace mismatch -> underdog cover. dog_cover is undefined for true
    # pick'ems (spread_line==0) -- analogous to a push, this is a POPULATION
    # restriction for this cell's value column, not just a flag-input gap, so
    # those rows are dropped from the population passed to score_cell (not
    # merely flagged False with a NaN value, which would corrupt the
    # bootstrap's weighted sums into NaN across every draw).
    game_b2 = game.loc[game["dog_cover"].notna()].reset_index(drop=True)
    n_pickem_dropped = int(game["dog_cover"].isna().sum())
    flag_b2 = game_b2["pace_diff_abs"] >= pace_threshold
    missing_b2 = game_b2["pace_diff_abs"].isna()
    print(f"  [B2] dropped {n_pickem_dropped} true pick'ems (dog_cover undefined)")
    cells.append(
        score_cell(
            game_b2,
            "team_style_pace_mismatch_dog_cover",
            flag=flag_b2,
            missing_mask=missing_b2,
            value_col="dog_cover",
            sign=1,
            description=(
                "Top-quartile |home-away prior-season centered seconds_per_play_pace| vs the "
                "field. Predicted POSITIVE on dog_cover (variance mechanism: fewer possessions "
                "favour the dog; docs/team_style.md)."
            ),
            reliability_note=reliability_notes["seconds_per_play_pace"],
            samples=args.samples,
            seed=args.seed,
        )
    )

    # B3: deep-ball offense (away) outdoors in high wind
    flag_b3 = (
        game["outdoor"]
        & (game["wind"] >= HIGH_WIND_MPH)
        & (game["prior_deep_share_centered"] >= thresholds["deep_share_centered"])
    )
    missing_b3 = (
        game["wind"].isna() | game["roof"].isna() | game["prior_deep_share_centered"].isna()
    )
    cells.append(
        score_cell(
            game,
            "team_style_deep_ball_outdoor_wind",
            flag=flag_b3,
            missing_mask=missing_b3,
            value_col="home_cover",
            sign=1,
            description=(
                "AWAY team prior-season top-quartile deep_share AND outdoor/open roof AND "
                f"wind>={HIGH_WIND_MPH}mph. Predicted POSITIVE on home_cover (docs/team_style.md). "
                "Actual-weather mechanism screen (game-time actuals, not pregame forecasts) -- "
                "upper bound for a forecast-time feature, matching the weather battery's own "
                "leakage caveat."
            ),
            reliability_note=reliability_notes["deep_share"],
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
        "command": "team-style-screen",
        "schedules": str(schedules_path),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "quartile": QUARTILE,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "n_reg_games": len(schedules),
        "thresholds": thresholds,
        "pace_diff_abs_threshold": pace_threshold,
        "predeclaration": "docs/team_style.md (frozen before this script scored anything)",
        "results": cells,
        "provenance": artifact_provenance(configuration, schedules_path, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="team-style-screen",
        metrics=payload,
        notes=(
            "PBP-08 team style/personality battery (5 predeclared cells); every cell recorded "
            "to registry/weak_signals.json regardless of sign, per AGENTS.md."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
