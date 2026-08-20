"""PBP-06 special-teams screen: 8 predeclared cells (4 traits x top/bottom
quartile) against NFL REG close-grade cover outcomes, 2009-2025.

**Predeclaration**: ``docs/special_teams_battery.md``, written and frozen
before this script was run against any cover outcome. Do not add, remove, or
redefine a cell here without updating that document first.

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

**Method** (read both, reused/imported per module docstrings there):
- ``block_bootstrap_two_group`` is algorithm-identical to
  ``scripts/nfl_weather_battery_screen.py`` / ``scripts/team_style_screen.py``.
- ``nfl_ats.experiment_runner.scale_subset_effect`` is IMPORTED directly
  (not reimplemented) for the full-slate-scaled point estimate.
- The team-perspective long table (one row per team per game,
  ``team_covered``) reuses the exact pattern documented in
  ``scripts/team_style_screen.py::build_long_table`` /
  ``scripts/nfl_bias_battery_screen.py::build_long_table``.

Population: NFL REG, close grade (``schedules.parquet`` ``spread_line``),
full local history 2009-2025. Special-teams inputs are prior-FULL-SEASON, so
2009 games carry no trailing value and are reported as missing, not dropped
from the declared range. 20,000 bootstrap samples, seed 20260819,
week-blocked primary / season-blocked secondary.

Writes ``artifacts/special_teams_battery/<UTC timestamp>/results.json``.
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

SEASON_START = 2009
SEASON_END = 2025
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819
QUARTILE_TOP = 0.75
QUARTILE_BOTTOM = 0.25

RAW_DIMENSIONS = ("fg_oe", "punt_net_yards", "punt_return_yards", "kickoff_return_yards")

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

RELIABILITY_NOTES = {
    "fg_oe": "YoY Pearson r +0.065, 95% CI [-0.022,+0.153], n=512 team-season pairs",
    "punt_net_yards": "YoY Pearson r +0.313, 95% CI [+0.233,+0.391], n=512 team-season pairs",
    "return_composite": (
        "componentwise YoY Pearson r: punt_return +0.109 [+0.019,+0.196] n=512, "
        "kickoff_return +0.158 [+0.073,+0.243] n=508; docs/special_teams_battery.md"
    ),
    "special_teams_composite_edge": (
        "weakest-link of 4 kept dimensions: fg_oe YoY r +0.065 [-0.022,+0.153]; "
        "docs/special_teams_battery.md"
    ),
}


def _latest_schedules() -> Path:
    candidates = sorted((REPO / "data/raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


def _latest_team_season() -> Path:
    candidates = sorted((REPO / "data" / "raw" / "special_teams").glob("*/team_season.parquet"))
    if not candidates:
        raise FileNotFoundError(
            "no data/raw/special_teams/*/team_season.parquet -- run "
            "scripts/special_teams_features.py first"
        )
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

    df = add_ats_outcomes(df)  # ats_margin, home_cover (reused verbatim)
    n_before_push_drop = len(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)
    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = n_before_push_drop - len(df)
    df["week_block"] = df["season"] * 100 + df["week"]
    return df


def add_composites(team_season: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Z-score each raw centered dimension (pooled sd, 544-row panel) and add
    the two composites (returns; overall special-teams edge).
    """

    result = team_season.copy()
    sds: dict[str, float] = {}
    for dim in RAW_DIMENSIONS:
        centered = f"{dim}_centered"
        sd = float(result[centered].std(ddof=1))
        sds[dim] = sd
        result[f"{dim}_z"] = result[centered] / sd if sd > 0 else np.nan

    result["return_composite_z"] = result[["punt_return_yards_z", "kickoff_return_yards_z"]].mean(
        axis=1
    )
    result["special_teams_composite_edge_z"] = result[
        ["fg_oe_z", "punt_net_yards_z", "punt_return_yards_z", "kickoff_return_yards_z"]
    ].mean(axis=1)
    return result, sds


def _prior(table: pd.DataFrame, columns: list[str], rename_team: str) -> pd.DataFrame:
    """Shift a (season, team) table forward one season and rename ``team`` to
    the join key for a specific side, so joining on (side_team, season) pulls
    the PRIOR season's value onto that season's game.
    """

    shifted = table[["team", "season", *columns]].copy()
    shifted["season"] = shifted["season"] + 1
    rename = {c: f"prior_{c}" for c in columns}
    rename["team"] = rename_team
    return shifted.rename(columns=rename)


def build_long_table(schedules: pd.DataFrame, team_season: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, side); reuses the exact pattern documented in
    ``scripts/team_style_screen.py::build_long_table`` /
    ``scripts/nfl_bias_battery_screen.py::build_long_table``.
    """

    trait_cols = [
        "fg_oe_z",
        "punt_net_yards_z",
        "return_composite_z",
        "special_teams_composite_edge_z",
    ]
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
        prior = _prior(team_season, trait_cols, "team")
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
    """Vectorized joint block bootstrap of ``100*(subset_mean-complement_mean)``.

    Algorithm-identical to ``scripts/nfl_weather_battery_screen.py`` /
    ``scripts/team_style_screen.py``.
    """

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
        value_col="team_covered",
        block_col="week_block",
        sign=sign,
        samples=samples,
        seed=seed,
    )
    season_blocked = summarize(
        df,
        flag=flag,
        value_col="team_covered",
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
    parser.add_argument("--team-season", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    started = time.time()
    schedules_path = args.schedules or _latest_schedules()
    team_season_path = args.team_season or _latest_team_season()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "special_teams_battery" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {schedules_path} ===")
    schedules = load_schedules(schedules_path)
    print(f"REG {SEASON_START}-{SEASON_END} close-graded games: {len(schedules)}")

    print(f"=== loading {team_season_path} ===")
    team_season_raw = pd.read_parquet(team_season_path)
    team_season, sds = add_composites(team_season_raw)

    thresholds: dict[str, dict[str, float]] = {}
    for col in (
        "fg_oe_z",
        "punt_net_yards_z",
        "return_composite_z",
        "special_teams_composite_edge_z",
    ):
        thresholds[col] = {
            "top": float(team_season[col].quantile(QUARTILE_TOP)),
            "bottom": float(team_season[col].quantile(QUARTILE_BOTTOM)),
        }
    print(
        "=== quartile thresholds (top=0.75, bottom=0.25, 544-row 2009-2025 team-season panel) ==="
    )
    for key, value in thresholds.items():
        print(f"  {key}: top={value['top']:.4f} bottom={value['bottom']:.4f}")

    long_df = build_long_table(schedules, team_season)

    cell_specs = [
        (
            "special_teams_fg_kicker_top_quartile",
            "prior_fg_oe_z",
            "top",
            1,
            "Top-quartile teams by prior-season field-goal makes-over-distance-expected "
            "(fg_oe) vs the field. Predicted POSITIVE on team_covered "
            "(docs/special_teams_battery.md).",
            RELIABILITY_NOTES["fg_oe"],
        ),
        (
            "special_teams_fg_kicker_bottom_quartile",
            "prior_fg_oe_z",
            "bottom",
            -1,
            "Bottom-quartile teams by prior-season fg_oe vs the field. Predicted NEGATIVE on "
            "team_covered (docs/special_teams_battery.md).",
            RELIABILITY_NOTES["fg_oe"],
        ),
        (
            "special_teams_punt_net_top_quartile",
            "prior_punt_net_yards_z",
            "top",
            1,
            "Top-quartile teams by prior-season net punt yards vs the field. Predicted POSITIVE "
            "on team_covered (docs/special_teams_battery.md).",
            RELIABILITY_NOTES["punt_net_yards"],
        ),
        (
            "special_teams_punt_net_bottom_quartile",
            "prior_punt_net_yards_z",
            "bottom",
            -1,
            "Bottom-quartile teams by prior-season net punt yards vs the field. Predicted "
            "NEGATIVE on team_covered (docs/special_teams_battery.md).",
            RELIABILITY_NOTES["punt_net_yards"],
        ),
        (
            "special_teams_return_top_quartile",
            "prior_return_composite_z",
            "top",
            1,
            "Top-quartile teams by prior-season return_composite (mean z of punt-return and "
            "kickoff-return yards) vs the field. Predicted POSITIVE on team_covered "
            "(docs/special_teams_battery.md).",
            RELIABILITY_NOTES["return_composite"],
        ),
        (
            "special_teams_return_bottom_quartile",
            "prior_return_composite_z",
            "bottom",
            -1,
            "Bottom-quartile teams by prior-season return_composite vs the field. Predicted "
            "NEGATIVE on team_covered (docs/special_teams_battery.md).",
            RELIABILITY_NOTES["return_composite"],
        ),
        (
            "special_teams_composite_edge_top_quartile",
            "prior_special_teams_composite_edge_z",
            "top",
            1,
            "Top-quartile teams by prior-season special_teams_composite_edge (mean z of all "
            "four raw dimensions) vs the field -- the field-position-above-expectation cell. "
            "Predicted POSITIVE on team_covered (docs/special_teams_battery.md).",
            RELIABILITY_NOTES["special_teams_composite_edge"],
        ),
        (
            "special_teams_composite_edge_bottom_quartile",
            "prior_special_teams_composite_edge_z",
            "bottom",
            -1,
            "Bottom-quartile teams by prior-season special_teams_composite_edge vs the field. "
            "Predicted NEGATIVE on team_covered (docs/special_teams_battery.md).",
            RELIABILITY_NOTES["special_teams_composite_edge"],
        ),
    ]

    threshold_key_map = {
        "prior_fg_oe_z": "fg_oe_z",
        "prior_punt_net_yards_z": "punt_net_yards_z",
        "prior_return_composite_z": "return_composite_z",
        "prior_special_teams_composite_edge_z": "special_teams_composite_edge_z",
    }

    cells: list[dict[str, Any]] = []
    for name, col, quartile, sign, description, reliability_note in cell_specs:
        threshold_col = threshold_key_map[col]
        cutoff = thresholds[threshold_col][quartile]
        flag = long_df[col] >= cutoff if quartile == "top" else long_df[col] <= cutoff
        missing = long_df[col].isna()
        cells.append(
            score_cell(
                long_df,
                name,
                flag=flag,
                missing_mask=missing,
                sign=sign,
                description=description,
                reliability_note=reliability_note,
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
        "command": "special-teams-screen",
        "schedules": str(schedules_path),
        "team_season": str(team_season_path),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "quartile_top": QUARTILE_TOP,
        "quartile_bottom": QUARTILE_BOTTOM,
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
        "pooled_sd": sds,
        "predeclaration": (
            "docs/special_teams_battery.md (frozen before this script scored anything)"
        ),
        "results": cells,
        "provenance": artifact_provenance(configuration, schedules_path, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="special-teams-screen",
        metrics=payload,
        notes=(
            "PBP-06 special-teams battery (8 predeclared cells); every cell recorded to "
            "registry/weak_signals.json regardless of sign, per AGENTS.md."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
