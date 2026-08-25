"""PBP-08 scheme/matchup interaction screen: coarse pricing of conditional
propensities. Lagged strictly-prior PBP shape features (prior-4-game windows):
offense pass rate over expected and pressure-rate allowed; defense coverage-
EPA allowed on free-rush dropbacks and pressure-generation rate. Four frozen
quartile-interaction cells (two candidate mismatches, two bottom-vs-bottom
mirror controls expected null) plus an era split of the strongest candidate,
on REG 2009-2025 NFL team-games, week-blocked bootstrap primary (20k draws,
seed 20260824), season-blocked secondary, full-slate-scaled accuracy_points
effects. Predeclaration frozen in ``docs/pbp08_matchup_screen.md`` before any
cover rate was computed; split-half reliability anchors are computed and
printed FIRST. Measure-only: never writes registry JSON; stamps a run log to
``registry/experiments/pbp08-matchup-screen/``.
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

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.pbp import analysis_plays, load_pbp_snapshot, snapshot_from_root  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260824
SEASON_START = 2009
SEASON_END = 2025
ERA_SPLITS: tuple[tuple[str, int, int], ...] = (
    ("2009_2017", 2009, 2017),
    ("2018_2025", 2018, 2025),
)
WINDOW_GAMES = 4
MIN_WINDOW_OBS = 3
MIN_QUANTILE_POOL = 200
WINDOW_TRAITS = ("off_pass_oe_w", "off_press_allow_w", "cov_epa_free_w", "press_gen_w")
GAME_TRAITS = ("off_pass_oe_g", "off_press_allow_g", "cov_epa_free_g", "press_gen_g")
WINDOW_OF_GAME_TRAIT = dict(zip(GAME_TRAITS, WINDOW_TRAITS, strict=True))


def _latest_pbp_snapshot() -> Path:
    candidates = sorted((REPO / "data/pbp/raw").glob("*/manifest.json"))
    if not candidates:
        raise FileNotFoundError("no data/pbp/raw/*/manifest.json snapshot found")
    return candidates[-1].parent


def default_schedules() -> Path:
    """Resolve lazily so importing this module never requires local data."""
    return latest_schedules()


DEFAULT_PBP_SNAPSHOT = _latest_pbp_snapshot()

SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
    "result",
    "spread_line",
]


def load_population(schedules_path: Path) -> pd.DataFrame:
    available = [c for c in SCHEDULE_COLUMNS if c in pd.read_parquet(schedules_path).columns]
    df = pd.read_parquet(schedules_path, columns=available)
    df = df.loc[df["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)
    df = add_ats_outcomes(df)
    n_before_push_drop = len(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)
    for column in ("home_team", "away_team"):
        df[column] = df[column].replace(TEAM_ABBREVIATION_ALIASES)
    df["gameday"] = pd.to_datetime(df["gameday"], errors="raise")
    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = n_before_push_drop - len(df)
    return df


def build_game_trait_tables(pbp_snapshot_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshot = snapshot_from_root(pbp_snapshot_path)
    plays_raw = load_pbp_snapshot(snapshot)
    plays = analysis_plays(plays_raw)
    plays = plays.loc[plays["competitive_play"]].copy()
    plays["posteam"] = plays["posteam"].replace(TEAM_ABBREVIATION_ALIASES)
    plays["defteam"] = plays["defteam"].replace(TEAM_ABBREVIATION_ALIASES)
    for column in ("qb_dropback", "sack", "qb_hit", "pass_oe", "epa"):
        plays[column] = pd.to_numeric(plays[column], errors="coerce")
    dropbacks = plays.loc[plays["qb_dropback"].fillna(0).eq(1)].copy()
    dropbacks["pressure"] = (
        dropbacks["sack"].fillna(0).eq(1) | dropbacks["qb_hit"].fillna(0).eq(1)
    ).astype(float)
    dropbacks["free_rush"] = 1.0 - dropbacks["pressure"]

    def _rate(group: pd.DataFrame, numerator: str) -> float:
        total = float(group.shape[0])
        return float(group[numerator].sum() / total) if total > 0 else np.nan

    offense_rows = []
    for (game_id, team), group in dropbacks.groupby(["game_id", "posteam"], sort=False):
        offense_rows.append(
            {
                "game_id": game_id,
                "team": team,
                "off_pass_oe_g": float(group["pass_oe"].mean()),
                "off_press_allow_g": _rate(group, "pressure"),
            }
        )
    defense_rows = []
    for (game_id, team), group in dropbacks.groupby(["game_id", "defteam"], sort=False):
        free = group.loc[group["free_rush"].eq(1.0)]
        defense_rows.append(
            {
                "game_id": game_id,
                "team": team,
                "cov_epa_free_g": float(free["epa"].mean()) if len(free) else np.nan,
                "press_gen_g": _rate(group, "pressure"),
            }
        )
    return pd.DataFrame(offense_rows), pd.DataFrame(defense_rows)


def build_long_table(
    df: pd.DataFrame, offense: pd.DataFrame, defense: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for is_home in (True, False):
        side = pd.DataFrame(
            {
                "game_id": df["game_id"],
                "season": df["season"],
                "week": df["week"],
                "gameday": df["gameday"],
                "team": df["home_team"] if is_home else df["away_team"],
                "opponent": df["away_team"] if is_home else df["home_team"],
                "is_home": is_home,
                "team_covered": df["home_cover"] if is_home else 1.0 - df["home_cover"],
            }
        )
        rows.append(side)
    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df.merge(offense, on=["game_id", "team"], how="left", validate="many_to_one")
    long_df = long_df.merge(defense, on=["game_id", "team"], how="left", validate="many_to_one")
    long_df["week_block"] = long_df["season"] * 100 + long_df["week"]
    long_df = long_df.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    for trait in GAME_TRAITS:
        values = pd.to_numeric(long_df[trait], errors="coerce")
        long_df[WINDOW_OF_GAME_TRAIT[trait]] = values.groupby(long_df["team"]).transform(
            lambda s: s.shift(1).rolling(WINDOW_GAMES, min_periods=MIN_WINDOW_OBS).mean()
        )
    long_df["has_window"] = long_df[list(WINDOW_TRAITS)].notna().all(axis=1)
    return long_df


def split_half_reliability(long_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    anchors: dict[str, dict[str, float]] = {}
    for trait in GAME_TRAITS:
        values = pd.to_numeric(long_df[trait], errors="coerce")
        shifted = {
            lag: values.groupby(long_df["team"]).shift(lag) for lag in range(1, WINDOW_GAMES + 1)
        }
        recent = (shifted[1] + shifted[2]) / 2.0
        older = (shifted[3] + shifted[4]) / 2.0
        pair = pd.DataFrame({"recent": recent, "older": older}).dropna()
        n = len(pair)
        if n < 30:
            anchors[trait] = {"n": float(n), "r": np.nan}
            continue
        r = float(np.corrcoef(pair["recent"].to_numpy(), pair["older"].to_numpy())[0, 1])
        anchors[trait] = {"n": float(n), "r": r}
    return anchors


def expanding_quartile_flags(values: pd.Series, blocks: pd.Series) -> np.ndarray:
    v = values.to_numpy(dtype=np.float64)
    b = blocks.to_numpy()
    sort_order = np.argsort(b, kind="stable")
    v_sorted = v[sort_order]
    b_sorted = b[sort_order]
    flags_sorted = np.full(len(v), np.int8(-1))
    pool: list[np.ndarray] = []
    start = 0
    n = len(v)
    while start < n:
        end = start
        while end < n and b_sorted[end] == b_sorted[start]:
            end += 1
        if pool:
            pooled = np.concatenate(pool)
            if len(pooled) >= MIN_QUANTILE_POOL:
                q25, q75 = np.quantile(pooled, [0.25, 0.75])
                segment = v_sorted[start:end]
                assigned = ~np.isnan(segment)
                seg_flags = np.where(
                    segment <= q25, np.int8(0), np.where(segment >= q75, np.int8(2), np.int8(1))
                )
                flags_sorted[start:end] = np.where(assigned, seg_flags, np.int8(-1))
        segment_values = v_sorted[start:end]
        segment_values = segment_values[~np.isnan(segment_values)]
        if len(segment_values):
            pool.append(segment_values)
        start = end
    result = np.empty(n, dtype=np.int8)
    result[sort_order] = flags_sorted
    return result


def attach_quartile_flags(long_df: pd.DataFrame, trait: str) -> pd.Series:
    flags = expanding_quartile_flags(long_df[trait], long_df["week_block"])
    return pd.Series(flags, index=long_df.index, name=f"{trait}_q")


def build_cells(long_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    work = long_df.copy()
    for trait in WINDOW_TRAITS:
        work[f"{trait}_q"] = attach_quartile_flags(work, trait)

    paired_traits = ("cov_epa_free_w", "press_gen_w")
    pair_flags = work[["game_id", "opponent"] + [f"{t}_q" for t in paired_traits]].copy()
    pair_flags = pair_flags.rename(columns={"opponent": "team"})
    pair_flags = pair_flags.rename(columns={f"{t}_q": f"opp_{t}_q" for t in paired_traits})
    work = work.merge(pair_flags, on=["game_id", "team"], how="left", validate="many_to_one")

    complete = (
        work["has_window"]
        & (work[[f"{t}_q" for t in WINDOW_TRAITS]] >= 0).all(axis=1)
        & (work[[f"opp_{t}_q" for t in paired_traits]] >= 0).all(axis=1)
    )
    work["eligible"] = complete.fillna(False).astype(bool)

    off_pass_hi = work["off_pass_oe_w_q"].eq(2)
    off_pass_lo = work["off_pass_oe_w_q"].eq(0)
    press_allow_hi = work["off_press_allow_w_q"].eq(2)
    press_allow_lo = work["off_press_allow_w_q"].eq(0)
    cov_bad = work["opp_cov_epa_free_w_q"].eq(2)
    cov_good = work["opp_cov_epa_free_w_q"].eq(0)
    press_gen_hi = work["opp_press_gen_w_q"].eq(2)
    press_gen_lo = work["opp_press_gen_w_q"].eq(0)

    specs: list[tuple[str, pd.Series, int, str]] = [
        (
            "pbp08_pass_mismatch",
            off_pass_hi & cov_bad,
            1,
            "Top-quartile prior-4-game pass-OVE offense facing bottom-quartile "
            "(worst) free-rush coverage-EPA defense -> back the passing side "
            "(sign +1)",
        ),
        (
            "pbp08_protection_mismatch",
            press_allow_hi & press_gen_hi,
            -1,
            "Top-quartile prior-4-game pressure-rate-allowed offense facing "
            "top-quartile pressure-generating defense -> back the defense side "
            "(sign -1)",
        ),
        (
            "pbp08_pass_mirror_null",
            off_pass_lo & cov_good,
            1,
            "Mirror control, bottom-vs-bottom on the pass axes: bottom-quartile "
            "pass-OVE offense vs bottom-quartile coverage-EPA-allowed defense "
            "(i.e., best coverage) -> expected NULL, read two-sided",
        ),
        (
            "pbp08_protection_mirror_null",
            press_allow_lo & press_gen_lo,
            1,
            "Mirror control, bottom-vs-bottom on the protection axes: "
            "bottom-quartile pressure-allowed offense vs bottom-quartile "
            "pressure-generating defense -> expected NULL, read two-sided",
        ),
    ]

    cells: dict[str, dict[str, Any]] = {}
    for name, flag, sign, description in specs:
        cells[name] = {
            "flag": flag.fillna(False).astype(bool),
            "sign": sign,
            "description": description,
        }
    assert len(cells) == 4, f"expected 4 predeclared cells, got {len(cells)}"
    return work, cells


def summarize(
    df: pd.DataFrame,
    *,
    flag: pd.Series,
    sign: int,
    block_col: str,
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
    subset_cover = float(work.loc[work["_flag"], "team_covered"].mean())
    complement_cover = float(work.loc[~work["_flag"], "team_covered"].mean())
    raw_gap_pts = (subset_cover - complement_cover) * 100.0
    fraction_of_slate = n_flag / n_total
    full_slate_effect_pts = sign * raw_gap_pts * fraction_of_slate

    draws = block_bootstrap_two_group(
        work,
        flag_col="_flag",
        value_col="team_covered",
        block_col=block_col,
        samples=samples,
        seed=seed,
    )
    signed_draws = sign * draws
    dropped = samples - len(draws)
    scaled_draws = signed_draws * fraction_of_slate
    lower, upper = (
        np.quantile(scaled_draws, [0.025, 0.975]) if len(scaled_draws) else (np.nan, np.nan)
    )

    return {
        "n_total": n_total,
        "n_flag": n_flag,
        "n_complement": n_complement,
        "n_blocks": int(work[block_col].nunique()),
        "subset_cover": subset_cover,
        "complement_cover": complement_cover,
        "raw_gap_pts": raw_gap_pts,
        "fraction_of_slate": fraction_of_slate,
        "full_slate_effect_pts": full_slate_effect_pts,
        "ci95_scaled": [float(lower), float(upper)],
        "probability_positive": float(np.mean(signed_draws > 0)) if len(signed_draws) else np.nan,
        "bootstrap_samples": samples,
        "dropped_draws": int(dropped),
        "insufficient_data": False,
    }


def score_cell(
    eligible_df: pd.DataFrame,
    name: str,
    spec: dict[str, Any],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    population = eligible_df.reset_index(drop=True)
    flag = spec["flag"].loc[eligible_df.index].reset_index(drop=True)

    week_blocked = summarize(
        population, flag=flag, sign=spec["sign"], block_col="week_block", samples=samples, seed=seed
    )
    season_blocked = summarize(
        population, flag=flag, sign=spec["sign"], block_col="season", samples=samples, seed=seed
    )
    era_results = {}
    for era_label, start, end in ERA_SPLITS:
        era_mask = population["season"].between(start, end).fillna(False)
        era_results[era_label] = summarize(
            population.loc[era_mask].reset_index(drop=True),
            flag=flag.loc[era_mask].reset_index(drop=True),
            sign=spec["sign"],
            block_col="week_block",
            samples=samples,
            seed=seed,
        )
    return {
        "name": name,
        "sign_dir": spec["sign"],
        "description": spec["description"],
        "n_flag": int(flag.sum()),
        "week_blocked": week_blocked,
        "season_blocked_secondary": season_blocked,
        "era_split": era_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--pbp-snapshot", type=Path, default=DEFAULT_PBP_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.schedules is None:
        args.schedules = default_schedules()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "pbp08_matchup" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} ===")
    schedules = load_population(args.schedules)
    print(
        f"REG {SEASON_START}-{SEASON_END} games: {schedules.attrs['n_before_push_drop']}, "
        f"pushes/missing dropped: {schedules.attrs['pushes_or_missing']}"
    )
    print(f"=== loading pbp snapshot {args.pbp_snapshot} ===")
    offense, defense = build_game_trait_tables(args.pbp_snapshot)
    print(f"pbp offense team-games: {len(offense)}, defense team-games: {len(defense)}")
    long_df = build_long_table(schedules, offense, defense)
    print(f"team-game rows (pushes dropped): {len(long_df)}")
    print(f"rows with complete strictly-prior window: {int(long_df['has_window'].sum())}")

    print("\n=== split-half reliability anchors (computed FIRST, predeclared) ===")
    reliability = split_half_reliability(long_df)
    for trait, anchor in reliability.items():
        print(f"  {trait}: r={anchor['r']:.4f} (n={anchor['n']:.0f})")

    work, cells = build_cells(long_df)
    eligible = work.loc[work["eligible"]].copy()
    long_df["eligible"] = work["eligible"]
    print(f"\neligible scored population (window + expanding quartiles): {len(eligible)}")

    results = []
    for name, spec in cells.items():
        print(f"\n=== {name} ===")
        cell = score_cell(eligible, name, spec, samples=args.samples, seed=args.seed)
        results.append(cell)
        wb = cell["week_blocked"]
        sb = cell["season_blocked_secondary"]
        if wb.get("insufficient_data"):
            print("  insufficient data (empty subset or complement)")
            continue
        print(
            f"  n_flag={cell['n_flag']} n_total={wb['n_total']} "
            f"subset_cover={wb['subset_cover']:.4f} complement_cover={wb['complement_cover']:.4f}"
        )
        print(
            f"  raw_gap={wb['raw_gap_pts']:+.3f}pts frac_of_slate={wb['fraction_of_slate']:.4f} "
            f"full_slate_effect={wb['full_slate_effect_pts']:+.4f}pts"
        )
        print(
            f"  week-blocked 95% [{wb['ci95_scaled'][0]:+.4f}, {wb['ci95_scaled'][1]:+.4f}] "
            f"P+={wb['probability_positive']:.4f} n_week_blocks={wb['n_blocks']}"
        )
        if not sb.get("insufficient_data"):
            print(
                f"  [season-blocked secondary] 95% [{sb['ci95_scaled'][0]:+.4f}, "
                f"{sb['ci95_scaled'][1]:+.4f}] P+={sb['probability_positive']:.4f} "
                f"n_seasons={sb['n_blocks']}"
            )

    candidates = [
        c for c in results if c["name"] in ("pbp08_pass_mismatch", "pbp08_protection_mismatch")
    ]
    scored_candidates = [c for c in candidates if not c["week_blocked"].get("insufficient_data")]
    ranked = sorted(
        scored_candidates,
        key=lambda c: abs(c["week_blocked"]["full_slate_effect_pts"]),
        reverse=True,
    )
    strongest = ranked[0]["name"] if ranked else None
    strongest_sign = next((c["sign_dir"] for c in results if c["name"] == strongest), 1)

    era_cells = []
    if strongest is not None:
        source = next(c for c in results if c["name"] == strongest)
        print(f"\n=== item d: era splits of strongest candidate ({strongest}) ===")
        for era_label, start, end in ERA_SPLITS:
            era = source["era_split"][era_label]
            era_cells.append(
                {
                    "name": f"{strongest}_era_{era_label}",
                    "parent": strongest,
                    "sign_dir": strongest_sign,
                    "season_start": start,
                    "season_end": end,
                    "week_blocked": era,
                }
            )
            if era.get("insufficient_data"):
                print(f"  [{era_label}] insufficient data")
                continue
            print(
                f"  [{era_label}] n_flag={era['n_flag']} "
                f"full_slate_effect={era['full_slate_effect_pts']:+.4f}pts "
                f"95% [{era['ci95_scaled'][0]:+.4f}, {era['ci95_scaled'][1]:+.4f}] "
                f"P+={era['probability_positive']:.4f}"
            )

    configuration = {
        "command": "pbp08-matchup-screen",
        "schedules": str(args.schedules),
        "pbp_snapshot": str(args.pbp_snapshot),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "window_games": WINDOW_GAMES,
        "min_window_obs": MIN_WINDOW_OBS,
        "min_quantile_pool": MIN_QUANTILE_POOL,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
    }
    payload = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "n_reg_games_before_push_drop": schedules.attrs["n_before_push_drop"],
        "n_pushes_or_missing_dropped": schedules.attrs["pushes_or_missing"],
        "n_team_game_rows": len(long_df),
        "n_rows_complete_window": int(long_df["has_window"].sum()),
        "n_eligible_scored": len(eligible),
        "split_half_reliability_anchors": reliability,
        "predeclaration": "docs/pbp08_matchup_screen.md (frozen before scoring)",
        "results": results,
        "strongest_cell": strongest,
        "era_cells_item_d": era_cells,
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="pbp08-matchup-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation screen (PBP-08 scheme/matchup interactions): "
            "4 frozen quartile-interaction cells + strongest candidate's 2 era splits; "
            "mined family disclosed in docs/pbp08_matchup_screen.md; every cell recorded "
            "unresolved_below_power via returned nfl-ats weak-signals record lines "
            "unless an admissible terminal ground applies; never writes registry JSON."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
