"""Arctic Shift subreddit fan-reaction ATS battery: 5 predeclared cells against
the spread on NFL REG games, week-blocked bootstrap (season-blocked
secondary), full-slate scaled, seeded and deterministic.

**Predeclaration**: ``docs/arctic_shift_ats_battery.md``, written and frozen
before this script was run against any cover outcome. Do not add, remove, or
redefine a cell here without updating that document first. This is the
separate ATS-battery decision ``docs/arctic_shift_gate.md`` explicitly left
open after its shared-variance leg failed against Wikipedia pageviews
(construct overlap is not an admissible AGENTS.md closing ground).

**Binding taxonomy (owned verbatim, per AGENTS.md/CLAUDE.md)**: an interval
or CI that contains zero is NEVER grounds to reject, fail, or close an
experiment -- at this evaluator's ~2-point resolution "contains zero" is the
EXPECTED outcome for a real small signal. Only two closing grounds: (1)
refuted mechanism -- RESOLVED wrong sign (whole interval on the wrong side
of zero) or zero split-half reliability; (2) bounded by a positive control
proven able to detect an effect that size. Everything else is
``unresolved_below_power``: record with ``probability_positive``, never
"contains zero". If a record command errors, the verdict is wrong, not the
validator. This script does not decide anything, it only measures; every
cell is recorded to the registry regardless of sign or interval shape via
the separate ``scripts/arctic_shift_battery_record.py``.

**Point-in-time-safe as-of construction** (``docs/arctic_shift_ats_battery.md``
section 3): per team, in that team's own chronological game order, a
Tuesday-ending 7-day window (byte-identical convention to
``scripts/attention_battery_screen.py`` / ``scripts/fluview_battery_screen.
py`` / ``scripts/arctic_shift_gate.py``) sums daily Arctic Shift post+comment
counts; a trailing baseline (mean/std over the team's own STRICTLY PRIOR
``TRAILING_WINDOW_GAMES`` games this season, ``shift(1)`` before the rolling
window) turns that window sum, and separately the window's comment/post
ratio, into z-scores. Both quantities are point-in-time-safe by
construction: the window itself only sums days at or before the Tuesday
decision cutoff, and the baseline never includes the current window.

Method reused verbatim from ``scripts/fluview_battery_screen.py`` /
``scripts/nfl_weather_battery_screen.py``: the same
``block_bootstrap_two_group`` joint week-blocked bootstrap, the same
full-slate effect scaling via ``nfl_ats.experiment_runner.scale_subset_effect``
(imported, not reimplemented), the same ``probability_positive`` definition,
and ``nfl_ats.cfb_qb_dependence.split_half_reliability`` (imported) for the
predeclared reliability check (section 6) -- run TWICE (volume, ratio),
since the ratio construct is a structurally distinct trait from the gate's
own volume-only year-over-year figure.

Writes JSON to ``artifacts/arctic_shift_battery/<UTC timestamp>/results.json``
and prints a summary table to stdout.
"""

from __future__ import annotations

import argparse
import json
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
from arctic_shift_battery_fetch import SUBREDDITS_ALL  # noqa: E402
from attention_battery_screen import TRAILING_MIN_GAMES, TRAILING_WINDOW_GAMES  # noqa: E402

from nfl_ats.cfb_qb_dependence import split_half_reliability  # noqa: E402
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.experiment_runner import scale_subset_effect  # noqa: E402
from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260826
SEASON_START = 2009
SEASON_END = 2026

# docs/arctic_shift_ats_battery.md section 4 -- reused from
# scripts/attention_battery_screen.py's hot_team_fade/away_hot precedent
# threshold rather than a fresh unexamined pick, for BOTH the volume and the
# ratio spike constructs.
SPIKE_THRESHOLD = 2.0

DESCRIPTION_SUFFIX = (
    " Predeclared docs/arctic_shift_ats_battery.md; mined battery, "
    "uncorrected multiplicity; Arctic Shift subreddit post+comment volume, "
    "window ends Tuesday of game week (point-in-time safe)."
)


def _canonical(team: pd.Series) -> pd.Series:
    return team.map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


# ---------------------------------------------------------------------------
# 1. Raw daily counts
# ---------------------------------------------------------------------------


def load_subreddit_daily_counts(raw_dir: Path) -> dict[str, dict[str, pd.Series]]:
    """Return {team: {'posts': daily series, 'comments': daily series}} for
    every team whose widened fetch (``scripts/arctic_shift_battery_fetch.py``)
    produced both files with at least one day of data. A team missing either
    file, or with an empty series, is silently EXCLUDED here (not zero-filled)
    -- excluded teams simply never get a computable baseline downstream, so
    every game touching them is dropped from every cell via the existing
    ``has_baseline`` mechanism, not defaulted to "not elevated"."""

    out: dict[str, dict[str, pd.Series]] = {}
    skipped: list[str] = []
    for team, subreddit in SUBREDDITS_ALL.items():
        series_by_kind: dict[str, pd.Series] = {}
        ok = True
        for kind in ("posts", "comments"):
            path = raw_dir / f"{subreddit}_{kind}_timeseries_full.json"
            if not path.is_file():
                ok = False
                break
            body = json.loads(path.read_text(encoding="utf-8"))
            items = body.get("data") or []
            if not items:
                ok = False
                break
            dates = pd.to_datetime([int(it["date"]) for it in items], unit="s")
            counts = pd.Series([float(it["value"]) for it in items], index=dates, dtype="float64")
            series_by_kind[kind] = counts.groupby(counts.index).sum()
        if not ok:
            skipped.append(team)
            continue
        out[team] = series_by_kind
    if skipped:
        print(f"  skipped (no usable fetch): {sorted(skipped)}")
    return out


def window_sum(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if series.empty:
        return 0.0
    idx = series.index
    mask = (idx >= start) & (idx <= end)
    return float(series.loc[mask].sum())


# ---------------------------------------------------------------------------
# 2. Game / team-game table construction
# ---------------------------------------------------------------------------


def load_games(schedules_path: Path) -> pd.DataFrame:
    schedules = pd.read_parquet(schedules_path)
    games = schedules.loc[
        (schedules["game_type"] == "REG")
        & (schedules["season"] >= SEASON_START)
        & (schedules["season"] <= SEASON_END)
    ].copy()
    games["home_team"] = _canonical(games["home_team"])
    games["away_team"] = _canonical(games["away_team"])
    games["gameday"] = pd.to_datetime(games["gameday"], errors="raise")
    games["spread_line"] = pd.to_numeric(games["spread_line"], errors="coerce")
    games = games.loc[games["spread_line"].notna()].copy()
    games = add_ats_outcomes(games)
    games = games.loc[games["home_cover"].notna()].copy()
    games["week"] = games["week"].astype(int)
    games["season"] = games["season"].astype(int)
    games["week_block"] = games["season"] * 100 + games["week"]
    return games.reset_index(drop=True)


def build_team_game_long(
    games: pd.DataFrame, team_daily: dict[str, dict[str, pd.Series]]
) -> pd.DataFrame:
    """One row per (game, side): team, gameday, window volume/ratio + trailing z."""

    sides = []
    for is_home, team_col in ((True, "home_team"), (False, "away_team")):
        side = pd.DataFrame(
            {
                "game_id": games["game_id"],
                "season": games["season"],
                "week": games["week"],
                "gameday": games["gameday"],
                "team": games[team_col],
                "is_home": is_home,
            }
        )
        sides.append(side)
    long_df = pd.concat(sides, ignore_index=True)

    weekday = long_df["gameday"].dt.weekday  # Monday=0 ... Sunday=6, Tuesday=1
    tuesday_offset = (weekday - 1) % 7
    window_end = long_df["gameday"] - pd.to_timedelta(tuesday_offset, unit="D")
    window_start = window_end - pd.Timedelta(days=6)
    long_df["window_start"] = window_start
    long_df["window_end"] = window_end

    window_posts = np.full(len(long_df), np.nan, dtype="float64")
    window_comments = np.full(len(long_df), np.nan, dtype="float64")
    for team, group in long_df.groupby("team", sort=False):
        daily = team_daily.get(team)
        if daily is None:
            continue
        idx = group.index.to_numpy()
        window_posts[idx] = [
            window_sum(daily["posts"], ws, we)
            for ws, we in zip(group["window_start"], group["window_end"], strict=True)
        ]
        window_comments[idx] = [
            window_sum(daily["comments"], ws, we)
            for ws, we in zip(group["window_start"], group["window_end"], strict=True)
        ]
    long_df["window_posts"] = window_posts
    long_df["window_comments"] = window_comments
    long_df["window_volume"] = long_df["window_posts"] + long_df["window_comments"]
    with np.errstate(invalid="ignore", divide="ignore"):
        long_df["comment_post_ratio"] = np.where(
            long_df["window_posts"] > 0,
            long_df["window_comments"] / long_df["window_posts"],
            np.nan,
        )

    # Trailing baseline reset PER (team, season) -- identical convention to
    # scripts/attention_battery_screen.py's build_team_game_long (first
    # TRAILING_MIN_GAMES-1 games of a team's season structurally have no
    # baseline; disclosed, not corrected).
    long_df = long_df.sort_values(["team", "season", "gameday"]).reset_index(drop=True)
    for metric in ("window_volume", "comment_post_ratio"):
        grouped = long_df.groupby(["team", "season"], sort=False)[metric]
        trailing_mean = grouped.transform(
            lambda s: (
                s.shift(1)
                .rolling(window=TRAILING_WINDOW_GAMES, min_periods=TRAILING_MIN_GAMES)
                .mean()
            )
        )
        trailing_std = grouped.transform(
            lambda s: (
                s.shift(1)
                .rolling(window=TRAILING_WINDOW_GAMES, min_periods=TRAILING_MIN_GAMES)
                .std()
            )
        )
        has_baseline = trailing_mean.notna() & trailing_std.notna() & (trailing_std > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (long_df[metric] - trailing_mean) / trailing_std
        z = z.where(has_baseline)
        suffix = "volume" if metric == "window_volume" else "ratio"
        long_df[f"{suffix}_z"] = z
        long_df[f"has_baseline_{suffix}"] = has_baseline.fillna(False)

    return long_df


def attach_game_level(games: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    home_side = long_df.loc[long_df["is_home"]].set_index("game_id")
    away_side = long_df.loc[~long_df["is_home"]].set_index("game_id")

    out = games.set_index("game_id").copy()
    for prefix, side in (("home", home_side), ("away", away_side)):
        out[f"{prefix}_window_volume"] = side["window_volume"]
        out[f"{prefix}_comment_post_ratio"] = side["comment_post_ratio"]
        out[f"{prefix}_volume_z"] = side["volume_z"]
        out[f"{prefix}_ratio_z"] = side["ratio_z"]
        out[f"{prefix}_has_baseline_volume"] = side["has_baseline_volume"]
        out[f"{prefix}_has_baseline_ratio"] = side["has_baseline_ratio"]

    return out.reset_index()


# ---------------------------------------------------------------------------
# 3. Cells (docs/arctic_shift_ats_battery.md section 5)
# ---------------------------------------------------------------------------


def build_cells(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}

    def add(name: str, population: pd.Series, flag: pd.Series, description: str) -> None:
        cells[name] = {
            "population": population.fillna(False).astype(bool),
            "flag": flag.fillna(False).astype(bool),
            "description": description,
        }

    add(
        "reddit_home_spike_fade",
        df["home_has_baseline_volume"],
        df["home_volume_z"] >= SPIKE_THRESHOLD,
        f"Home team subreddit volume_z >= {SPIKE_THRESHOLD:g} (own-baseline post+comment "
        "spike) vs not, response home_cover. Predicted sign: NEGATIVE." + DESCRIPTION_SUFFIX,
    )
    add(
        "reddit_away_spike_value",
        df["away_has_baseline_volume"],
        df["away_volume_z"] >= SPIKE_THRESHOLD,
        f"Away team subreddit volume_z >= {SPIKE_THRESHOLD:g} vs not, response home_cover. "
        "Predicted sign: POSITIVE." + DESCRIPTION_SUFFIX,
    )

    home_spike = df["home_volume_z"] >= SPIKE_THRESHOLD
    away_spike = df["away_volume_z"] >= SPIKE_THRESHOLD
    diff_population = (
        df["home_has_baseline_volume"] & df["away_has_baseline_volume"] & (home_spike != away_spike)
    )
    add(
        "reddit_spike_gap_home_worse",
        diff_population,
        home_spike & ~away_spike,
        "Restricted to games where exactly one side spikes (volume_z >= "
        f"{SPIKE_THRESHOLD:g}, home XOR away); subset = home spikes & away not, complement "
        "= away spikes & home not, response home_cover. Own-baseline-normalized "
        "asymmetry cell (structural fanbase-size gap cancels by construction). "
        "Predicted sign: NEGATIVE." + DESCRIPTION_SUFFIX,
    )

    add(
        "reddit_home_comment_ratio_elevated",
        df["home_has_baseline_ratio"],
        df["home_ratio_z"] >= SPIKE_THRESHOLD,
        f"Home team comment-to-post ratio_z >= {SPIKE_THRESHOLD:g} (own-baseline elevated "
        "argument/anxiety proxy) vs not, response home_cover. Predicted sign: NEGATIVE."
        + DESCRIPTION_SUFFIX,
    )
    add(
        "reddit_away_comment_ratio_elevated",
        df["away_has_baseline_ratio"],
        df["away_ratio_z"] >= SPIKE_THRESHOLD,
        f"Away team comment-to-post ratio_z >= {SPIKE_THRESHOLD:g} vs not, response "
        "home_cover. Predicted sign: POSITIVE." + DESCRIPTION_SUFFIX,
    )

    expected = 5
    assert len(cells) == expected, f"expected {expected} predeclared cells, got {len(cells)}"
    return cells


# ---------------------------------------------------------------------------
# 4. Bootstrap (algorithm-identical to fluview_battery_screen.py)
# ---------------------------------------------------------------------------


def summarize(
    df: pd.DataFrame, *, flag: pd.Series, block_col: str, samples: int, seed: int
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
    subset_cover = float(work.loc[work["_flag"], "home_cover"].mean())
    complement_cover = float(work.loc[~work["_flag"], "home_cover"].mean())
    raw_gap_pts = (subset_cover - complement_cover) * 100.0
    fraction_of_slate = n_flag / n_total
    sign = 1 if raw_gap_pts >= 0 else -1
    full_slate_effect_pts = scale_subset_effect(
        abs(raw_gap_pts) / 100.0, sign=sign, fraction_of_slate=fraction_of_slate
    )

    draws = block_bootstrap_two_group(
        work,
        flag_col="_flag",
        value_col="home_cover",
        block_col=block_col,
        samples=samples,
        seed=seed,
    )
    dropped = samples - len(draws)
    scaled_draws = draws * fraction_of_slate
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
        "probability_positive": float(np.mean(draws > 0)) if len(draws) else np.nan,
        "bootstrap_samples": samples,
        "dropped_draws": int(dropped),
        "insufficient_data": False,
    }


def score_cell(
    df: pd.DataFrame, name: str, spec: dict[str, Any], *, samples: int, seed: int
) -> dict[str, Any]:
    population = spec["population"]
    flag = spec["flag"]
    scored = df.loc[population].reset_index(drop=True)
    scored_flag = flag.loc[population].reset_index(drop=True)

    week_blocked = summarize(
        scored, flag=scored_flag, block_col="week_block", samples=samples, seed=seed
    )
    season_blocked = summarize(
        scored, flag=scored_flag, block_col="season", samples=samples, seed=seed
    )

    return {
        "name": name,
        "description": spec["description"],
        "n_population": int(population.sum()),
        "n_excluded_missing": int((~population).sum()),
        "n_flag": int(scored_flag.sum()),
        "week_blocked": week_blocked,
        "season_blocked_secondary": season_blocked,
    }


# ---------------------------------------------------------------------------
# 5. Reliability check (docs/arctic_shift_ats_battery.md section 6)
# ---------------------------------------------------------------------------


def compute_reliability(long_df: pd.DataFrame, metric: str, *, seed: int) -> dict[str, Any]:
    panel = long_df.rename(columns={"team": "team_id"})
    return split_half_reliability(panel, metric, seed=seed)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=REPO / "data" / "raw" / "arctic_shift")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    schedules_path = args.schedules or latest_schedules()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "arctic_shift_battery" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading Arctic Shift daily counts from {args.raw_dir} ===")
    team_daily = load_subreddit_daily_counts(args.raw_dir)
    print(f"  {len(team_daily)}/{len(SUBREDDITS_ALL)} teams with usable fetched data")

    print(f"\n=== loading {schedules_path} ===")
    games = load_games(schedules_path)
    print(f"REG {SEASON_START}-{SEASON_END} games with spread_line + result: {len(games)}")

    long_df = build_team_game_long(games, team_daily)
    n_no_baseline_volume = int((~long_df["has_baseline_volume"]).sum())
    n_no_baseline_ratio = int((~long_df["has_baseline_ratio"]).sum())
    print(
        f"team-game rows: {len(long_df)}; no volume baseline: {n_no_baseline_volume}; "
        f"no ratio baseline: {n_no_baseline_ratio}"
    )
    coverage_by_season = long_df.groupby("season")["has_baseline_volume"].mean().round(4).to_dict()
    print("volume-baseline coverage by season:")
    for season, cov in sorted(coverage_by_season.items()):
        print(f"  {season}: {cov:.1%}")

    game_df = attach_game_level(games, long_df)

    print("\n=== reliability check (section 6) ===")
    reliability_volume = compute_reliability(long_df, "window_volume", seed=args.seed)
    reliability_ratio = compute_reliability(long_df, "comment_post_ratio", seed=args.seed + 1)
    print(
        f"  volume: n_team_seasons={reliability_volume['n_team_seasons']} "
        f"pearson_r={reliability_volume['pearson_r']:.4f} "
        f"spearman_brown={reliability_volume['spearman_brown_full_length_reliability']:.4f} "
        f"P+={reliability_volume['probability_positive']:.4f}"
    )
    print(
        f"  ratio: n_team_seasons={reliability_ratio['n_team_seasons']} "
        f"pearson_r={reliability_ratio['pearson_r']:.4f} "
        f"spearman_brown={reliability_ratio['spearman_brown_full_length_reliability']:.4f} "
        f"P+={reliability_ratio['probability_positive']:.4f}"
    )

    cells = build_cells(game_df)
    results = []
    for name, spec in cells.items():
        print(f"\n=== {name} ===")
        cell = score_cell(game_df, name, spec, samples=args.samples, seed=args.seed)
        results.append(cell)
        wb = cell["week_blocked"]
        sb = cell["season_blocked_secondary"]
        print(
            f"  n_population={cell['n_population']} "
            f"n_excluded_missing={cell['n_excluded_missing']} n_flag={cell['n_flag']}"
        )
        if wb.get("insufficient_data"):
            print("  insufficient data (empty subset or complement)")
            continue
        print(
            f"  subset_cover={wb['subset_cover']:.4f} "
            f"complement_cover={wb['complement_cover']:.4f} "
            f"raw_gap={wb['raw_gap_pts']:+.3f}pts frac_of_slate={wb['fraction_of_slate']:.4f}"
        )
        print(
            f"  full_slate_effect={wb['full_slate_effect_pts']:+.4f}pts "
            f"week-blocked 95% [{wb['ci95_scaled'][0]:+.4f}, {wb['ci95_scaled'][1]:+.4f}] "
            f"P+={wb['probability_positive']:.4f} n_week_blocks={wb['n_blocks']}"
        )
        if not sb.get("insufficient_data"):
            print(
                f"  [season-blocked secondary] 95% [{sb['ci95_scaled'][0]:+.4f}, "
                f"{sb['ci95_scaled'][1]:+.4f}] P+={sb['probability_positive']:.4f} "
                f"n_seasons={sb['n_blocks']}"
            )

    ranked = sorted(
        (r for r in results if not r["week_blocked"].get("insufficient_data")),
        key=lambda r: abs(r["week_blocked"]["full_slate_effect_pts"]),
        reverse=True,
    )
    print("\n=== ranked by |full-slate effect|, week-blocked primary ===")
    for rank, cell in enumerate(ranked, start=1):
        wb = cell["week_blocked"]
        print(
            f"{rank}. {cell['name']:<36} {wb['full_slate_effect_pts']:+.4f}pts "
            f"P+={wb['probability_positive']:.4f} n_flag={cell['n_flag']}"
        )

    configuration = {
        "command": "arctic-shift-battery-screen",
        "schedules": str(schedules_path),
        "raw_dir": str(args.raw_dir),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "spike_threshold": SPIKE_THRESHOLD,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "n_cells": len(cells),
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "n_teams_with_data": len(team_daily),
        "n_teams_total": len(SUBREDDITS_ALL),
        "teams_with_data": sorted(team_daily),
        "n_reg_games": len(games),
        "n_team_game_rows": len(long_df),
        "n_no_baseline_volume": n_no_baseline_volume,
        "n_no_baseline_ratio": n_no_baseline_ratio,
        "coverage_by_season_volume": {str(k): v for k, v in coverage_by_season.items()},
        "reliability_volume": reliability_volume,
        "reliability_ratio": reliability_ratio,
        "predeclaration": "docs/arctic_shift_ats_battery.md (frozen before scoring)",
        "results": results,
        "ranked_by_abs_full_slate_effect": [cell["name"] for cell in ranked],
        "provenance": artifact_provenance(configuration, schedules_path, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="arctic-shift-battery-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation battery (5 predeclared cells, Arctic Shift "
            "subreddit fan-reaction volume/ratio); mined family, every cell predeclared "
            "to record via a separate scripts/arctic_shift_battery_record.py call "
            "regardless of interval shape (AGENTS.md)."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
