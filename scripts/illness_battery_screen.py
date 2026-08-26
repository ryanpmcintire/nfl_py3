"""Illness-designation battery: 5 predeclared cells against the spread on
NFL REG 2010-2024 (excluding COVID-era 2020, scored separately), week-blocked
bootstrap (season-blocked secondary), full-slate scaled, seeded and
deterministic.

**Predeclaration**: ``docs/illness_battery.md``, written and frozen before
this script was run against any cover outcome. Do not add, remove, or
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
validator. This script does not decide anything, it only measures; every
cell is recorded to the registry regardless of sign or interval shape via
the separate ``scripts/illness_battery_record.py``.

**Point-in-time-safe as-of construction** (``docs/illness_battery.md``
section 3, read it): the decision cutoff for each game is
``nfl_ats.pick_refresh.pick_deadline(kickoff, sunday_lock)`` (imported, not
reimplemented) -- the project's own binding per-game pick deadline,
``min(that game's own kickoff, that week's Sunday 16:00 ET)``. Per (season,
week, team, gsis_id) entity, only report rows with ``date_modified <=
cutoff`` are visible; the as-of state is the LATEST such row. A team-week
with zero visible rows resolves to MISSING, never a zero illness count.

Method reused verbatim from ``scripts/fluview_battery_screen.py``: the same
``block_bootstrap_two_group`` joint week-blocked bootstrap
(``scripts/_common.py``), the same full-slate effect scaling via
``nfl_ats.experiment_runner.scale_subset_effect`` (imported, not
reimplemented), the same ``probability_positive`` definition, and
``nfl_ats.cfb_qb_dependence.split_half_reliability`` (imported) for the
predeclared reliability check (section 6).

Writes JSON to ``artifacts/illness_battery/<UTC timestamp>/results.json``
and prints a summary table to stdout.
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

from _common import block_bootstrap_two_group  # noqa: E402

from nfl_ats.cfb_qb_dependence import split_half_reliability  # noqa: E402
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.experiment_runner import scale_subset_effect  # noqa: E402
from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260826
SEASON_START = 2010
SEASON_END = 2024
COVID_SEASON = 2020
ILLNESS_COUNT_THRESHOLD = 2
NOT_EXPECTED_TO_PLAY = frozenset({"Out", "Doubtful"})

ILLNESS_TEXT_COLS = (
    "report_primary_injury",
    "report_secondary_injury",
    "practice_primary_injury",
    "practice_secondary_injury",
)


def _latest(glob_pattern: str, label: str) -> Path:
    candidates = sorted(REPO.glob(glob_pattern))
    if not candidates:
        raise FileNotFoundError(f"no {label} found matching {glob_pattern!r}")
    return candidates[-1]


def _latest_schedules() -> Path:
    return _latest("data/raw/*/schedules.parquet", "schedules.parquet snapshot")


def _latest_nflverse_injuries() -> Path:
    return _latest("data/raw/nflverse_injuries/*/injuries.parquet", "nflverse_injuries snapshot")


def default_schedules() -> Path:
    """Resolve lazily so importing this module never requires local data."""
    return _latest_schedules()


def default_injuries() -> Path:
    return _latest_nflverse_injuries()


# ---------------------------------------------------------------------------
# 1. Decision cutoff (docs/illness_battery.md section 3)
# ---------------------------------------------------------------------------


def _kickoff_utc(games: pd.DataFrame) -> pd.Series:
    """Duplicated (not imported) from ``nfl_ats.features._kickoff_utc`` -- an
    internal, underscore-prefixed helper this repo's own convention is to
    duplicate rather than import across module boundaries (see
    ``nfl_ats.transaction_wire_features`` / ``scripts/qb_backup_news_visibility.py``,
    both of which state the same reasoning). Identical arithmetic."""

    date_text = pd.to_datetime(games["gameday"], errors="coerce").dt.strftime("%Y-%m-%d")
    time_text = games["gametime"].astype("string")
    local = pd.to_datetime(date_text + " " + time_text, errors="coerce")
    return local.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")


def attach_cutoffs(games: pd.DataFrame) -> pd.DataFrame:
    """Per-game decision cutoff = ``pick_deadline(kickoff, sunday_lock)``,
    both imported from ``nfl_ats.pick_refresh`` -- the project's own binding
    per-game pick-lock rule, computed per (season, week) from that week's own
    kickoffs (mode Tue..Mon cycle Sunday)."""

    games = games.copy()
    games["kickoff_utc"] = _kickoff_utc(games)
    cutoff_parts: list[pd.Series] = []
    for _, group in games.groupby(["season", "week"]):
        sunday_lock = sunday_pick_lock(group["kickoff_utc"])
        cutoff_parts.append(
            group["kickoff_utc"].map(lambda k, sl=sunday_lock: pick_deadline(k, sl))
        )
    games["cutoff_date"] = pd.concat(cutoff_parts).reindex(games.index)
    return games


# ---------------------------------------------------------------------------
# 2. Illness flag + as-of team-week resolution (docs/illness_battery.md sec 3)
# ---------------------------------------------------------------------------


def add_is_illness(frame: pd.DataFrame) -> pd.DataFrame:
    """Vectorized illness flag: any of the four reason columns, lower-cased,
    contains the substring "illness" (docs/illness_battery.md section 1 --
    measured variant strings: "illness", "illness (non-covid)", "knee
    illness", "medical illness", "non-football illness", and several
    two-body-part combinations on the secondary columns)."""

    frame = frame.copy()
    flag = pd.Series(False, index=frame.index)
    for col in ILLNESS_TEXT_COLS:
        if col not in frame.columns:
            continue
        flag = flag | frame[col].astype("string").str.lower().str.contains(
            "illness", na=False, regex=False
        )
    frame["is_illness"] = flag
    return frame


def load_injuries(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    df = raw.loc[raw["game_type"].astype(str) == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df["team"] = df["team"].replace(TEAM_ABBREVIATION_ALIASES).astype("string")
    df["date_modified"] = pd.to_datetime(df["date_modified"], errors="coerce", utc=True)
    df = add_is_illness(df)
    return df


def build_team_week_cutoffs(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team, cutoff_date) -- both the home and
    away side of every REG game share their game's own cutoff (it is the
    same game). A team playing more than once in the same numbered week
    never happens in the REG season, so this is a clean 1:1 mapping."""

    home = games[["season", "week", "home_team", "cutoff_date"]].rename(
        columns={"home_team": "team"}
    )
    away = games[["season", "week", "away_team", "cutoff_date"]].rename(
        columns={"away_team": "team"}
    )
    return pd.concat([home, away], ignore_index=True).drop_duplicates()


def resolve_asof_team_week(injuries: pd.DataFrame, team_week_cutoffs: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, team, gsis_id) entity: keep only rows with
    ``date_modified <= cutoff`` (a NaT ``date_modified`` never satisfies this
    -- it is dropped, matching the point-in-time-recoverable-floor treatment
    of the 2009/2025 gap), then take each entity's LATEST surviving revision.
    Aggregate to team-week ``illness_count``/``active_illness_count``.

    A (season, week, team) with ZERO surviving rows is simply ABSENT from
    the returned frame -- missing, not a zero count. This is the function
    the leakage regression test (tests/test_illness_battery_leakage.py)
    exercises directly.
    """

    merged = injuries.merge(team_week_cutoffs, on=["season", "week", "team"], how="inner")
    # Defensive re-coercion to a common tz-aware dtype: a naive-vs-aware
    # dtype mismatch on either side (e.g. an all-NaT column with no explicit
    # utc=True upstream) raises TypeError on comparison rather than the
    # intended "never visible" False -- fail safe by normalizing, not by
    # letting a dtype quirk silently change which rows are visible.
    merged["date_modified"] = pd.to_datetime(merged["date_modified"], utc=True, errors="coerce")
    merged["cutoff_date"] = pd.to_datetime(merged["cutoff_date"], utc=True, errors="coerce")
    visible = merged.loc[merged["date_modified"] <= merged["cutoff_date"]].copy()
    columns = ["season", "week", "team", "illness_count", "active_illness_count"]
    if visible.empty:
        return pd.DataFrame(columns=columns)

    visible["is_active_illness"] = visible["is_illness"] & ~visible["report_status"].isin(
        NOT_EXPECTED_TO_PLAY
    )
    visible = visible.sort_values(["season", "week", "team", "gsis_id", "date_modified"])
    as_of = visible.drop_duplicates(subset=["season", "week", "team", "gsis_id"], keep="last")
    agg = as_of.groupby(["season", "week", "team"], as_index=False).agg(
        illness_count=("is_illness", "sum"),
        active_illness_count=("is_active_illness", "sum"),
    )
    return agg[columns]


# ---------------------------------------------------------------------------
# 3. Population + feature construction
# ---------------------------------------------------------------------------


def load_schedules(path: Path, *, season_start: int, season_end: int) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    df = raw.loc[raw["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df["home_team"] = df["home_team"].replace(TEAM_ABBREVIATION_ALIASES)
    df["away_team"] = df["away_team"].replace(TEAM_ABBREVIATION_ALIASES)
    df = df.loc[df["season"].between(season_start, season_end)]
    n_before_push_drop = len(df)

    df = add_ats_outcomes(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)
    pushes_or_missing = n_before_push_drop - len(df)

    df["gameday"] = df["gameday"].astype("string")
    df["week_block"] = df["season"] * 100 + df["week"]

    df = attach_cutoffs(df)

    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = pushes_or_missing
    return df


def attach_team_week_features(df: pd.DataFrame, team_week_agg: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        team_week_agg.rename(
            columns={
                "team": "home_team",
                "illness_count": "home_illness_count",
                "active_illness_count": "home_active_illness_count",
            }
        ),
        on=["season", "week", "home_team"],
        how="left",
    )
    df = df.merge(
        team_week_agg.rename(
            columns={
                "team": "away_team",
                "illness_count": "away_illness_count",
                "active_illness_count": "away_active_illness_count",
            }
        ),
        on=["season", "week", "away_team"],
        how="left",
    )
    df["home_missing"] = df["home_illness_count"].isna()
    df["away_missing"] = df["away_illness_count"].isna()
    df["home_ge2"] = np.where(
        df["home_missing"], False, df["home_illness_count"] >= ILLNESS_COUNT_THRESHOLD
    )
    df["away_ge2"] = np.where(
        df["away_missing"], False, df["away_illness_count"] >= ILLNESS_COUNT_THRESHOLD
    )
    df["home_active_ge1"] = np.where(
        df["home_missing"], False, df["home_active_illness_count"] >= 1
    )
    df["away_active_ge1"] = np.where(
        df["away_missing"], False, df["away_active_illness_count"] >= 1
    )
    return df


# ---------------------------------------------------------------------------
# 4. Cells (docs/illness_battery.md section 5)
# ---------------------------------------------------------------------------


def build_cells(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}

    def add(name: str, population: pd.Series, flag: pd.Series, description: str) -> None:
        cells[name] = {
            "population": population.fillna(False).astype(bool),
            "flag": flag.fillna(False).astype(bool),
            "description": description,
        }

    everyone = pd.Series(True, index=df.index)

    add(
        "illness_home_ge2",
        everyone & ~df["home_missing"],
        df["home_ge2"],
        "Home team illness_count >= 2 (as of the game's own pick-deadline cutoff) vs. < 2, "
        "response home_cover. Predicted sign: NEGATIVE.",
    )
    add(
        "illness_away_ge2",
        everyone & ~df["away_missing"],
        df["away_ge2"],
        "Away team illness_count >= 2 (as of cutoff) vs. < 2, response home_cover. "
        "Predicted sign: POSITIVE.",
    )

    diff_population = ~df["home_missing"] & ~df["away_missing"] & (df["home_ge2"] != df["away_ge2"])
    add(
        "illness_differential_home_worse",
        diff_population,
        df["home_ge2"] & ~df["away_ge2"],
        "Restricted to games where exactly one side has illness_count >= 2 (home XOR away); "
        "subset = home>=2 & away<2, complement = away>=2 & home<2, response home_cover. "
        "Predicted sign: NEGATIVE.",
    )

    add(
        "illness_home_active_ge1",
        everyone & ~df["home_missing"],
        df["home_active_ge1"],
        "Home team active_illness_count >= 1 (illness-flagged AND report_status not in "
        "{Out, Doubtful} as of cutoff -- expected to play through it) vs. 0, response "
        "home_cover. Predicted sign: NEGATIVE.",
    )
    add(
        "illness_away_active_ge1",
        everyone & ~df["away_missing"],
        df["away_active_ge1"],
        "Away team active_illness_count >= 1 vs. 0, response home_cover. Predicted sign: POSITIVE.",
    )

    expected = 5
    assert len(cells) == expected, f"expected {expected} predeclared cells, got {len(cells)}"
    return cells


# ---------------------------------------------------------------------------
# 5. Bootstrap (algorithm-identical to fluview_battery_screen.py)
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

    n_blocks = int(work[block_col].nunique())
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
        "n_blocks": n_blocks,
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
        "degenerate_single_block": n_blocks < 2,
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


def score_all_cells(
    df: pd.DataFrame, *, samples: int, seed: int
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    cells = build_cells(df)
    results = []
    for name, spec in cells.items():
        results.append(score_cell(df, name, spec, samples=samples, seed=seed))
    return cells, results


# ---------------------------------------------------------------------------
# 6. Reliability check (docs/illness_battery.md section 6)
# ---------------------------------------------------------------------------


def build_team_week_panel(df: pd.DataFrame, team_week_agg: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, season, week): a genuinely team-specific panel
    (unlike FluView's shared-by-state panel) -- every REG team-week in the
    scored population, whether the team was home or away that week, with its
    own as-of ``illness_count`` (NaN where missing, dropped by
    ``split_half_reliability`` itself)."""

    in_scope_weeks = df[["season", "week"]].drop_duplicates()
    panel = team_week_agg.merge(in_scope_weeks, on=["season", "week"], how="inner")
    panel = panel.rename(columns={"team": "team_id"})
    return panel[["team_id", "season", "week", "illness_count"]]


def compute_reliability(panel: pd.DataFrame) -> dict[str, Any]:
    long = panel.dropna(subset=["illness_count"]).copy()
    long["illness_count"] = long["illness_count"].astype(float)
    return split_half_reliability(long, "illness_count", seed=BOOTSTRAP_SEED)


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------


def _print_cell(cell: dict[str, Any]) -> None:
    wb = cell["week_blocked"]
    sb = cell["season_blocked_secondary"]
    print(
        f"  n_population={cell['n_population']} "
        f"n_excluded_missing={cell['n_excluded_missing']} n_flag={cell['n_flag']}"
    )
    if wb.get("insufficient_data"):
        print("  insufficient data (empty subset or complement)")
        return
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--injuries", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.schedules is None:
        args.schedules = default_schedules()
    if args.injuries is None:
        args.injuries = default_injuries()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "illness_battery" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} ===")
    # Full 2010-2024 range loaded once (primary population excludes 2020
    # below; the 2020 stratum is scored separately from this same load).
    df_all = load_schedules(args.schedules, season_start=SEASON_START, season_end=SEASON_END)
    # .attrs does not survive the .merge() calls below, so capture these now.
    n_before_push_drop = df_all.attrs["n_before_push_drop"]
    pushes_or_missing = df_all.attrs["pushes_or_missing"]
    print(
        f"REG {SEASON_START}-{SEASON_END} games: {n_before_push_drop}, "
        f"pushes/missing dropped: {pushes_or_missing}, "
        f"scored population: {len(df_all)}"
    )

    print(f"\n=== loading {args.injuries} ===")
    injuries = load_injuries(args.injuries)
    print(f"nflverse injuries REG rows: {len(injuries)}")

    team_week_cutoffs = build_team_week_cutoffs(df_all)
    team_week_agg = resolve_asof_team_week(injuries, team_week_cutoffs)
    print(f"team-weeks resolved as-of cutoff (non-missing): {len(team_week_agg)}")

    df_all = attach_team_week_features(df_all, team_week_agg)

    n_home_missing = int(df_all["home_missing"].sum())
    n_away_missing = int(df_all["away_missing"].sum())
    print(
        f"\nmissingness: home_missing={n_home_missing}/{len(df_all)} "
        f"({n_home_missing / len(df_all):.1%}), away_missing={n_away_missing}/{len(df_all)} "
        f"({n_away_missing / len(df_all):.1%})"
    )
    coverage_by_season = (1.0 - df_all.groupby("season")["home_missing"].mean()).to_dict()
    print("coverage (fraction with non-missing home_illness_count) by season:")
    for season, cov in sorted(coverage_by_season.items()):
        print(f"  {season}: {cov:.1%}")

    # Primary population: excludes the COVID-era 2020 stratum.
    df_primary = df_all.loc[df_all["season"] != COVID_SEASON].reset_index(drop=True)

    panel = build_team_week_panel(df_primary, team_week_agg)
    print("\n=== reliability check (section 6), primary population ===")
    reliability = compute_reliability(panel)
    print(
        f"  n_team_seasons={reliability['n_team_seasons']} "
        f"pearson_r={reliability['pearson_r']:.4f} "
        f"ci95={reliability['pearson_r_ci95']} "
        f"spearman_brown={reliability['spearman_brown_full_length_reliability']:.4f} "
        f"P+={reliability['probability_positive']:.4f}"
    )

    print("\n=== primary cells (2010-2024 excl. 2020) ===")
    _cells_primary, results_primary = score_all_cells(
        df_primary, samples=args.samples, seed=args.seed
    )
    for cell in results_primary:
        print(f"\n=== {cell['name']} ===")
        _print_cell(cell)

    ranked = sorted(
        (r for r in results_primary if not r["week_blocked"].get("insufficient_data")),
        key=lambda r: abs(r["week_blocked"]["full_slate_effect_pts"]),
        reverse=True,
    )
    print("\n=== ranked by |full-slate effect|, week-blocked primary ===")
    for rank, cell in enumerate(ranked, start=1):
        wb = cell["week_blocked"]
        print(
            f"{rank}. {cell['name']:<32} {wb['full_slate_effect_pts']:+.4f}pts "
            f"P+={wb['probability_positive']:.4f} n_flag={cell['n_flag']}"
        )

    print("\n=== supplementary: 2020 (COVID-era) stratum, NOT pooled ===")
    df_2020 = df_all.loc[df_all["season"] == COVID_SEASON].reset_index(drop=True)
    if len(df_2020):
        _cells_2020, results_2020 = score_all_cells(df_2020, samples=args.samples, seed=args.seed)
        for cell in results_2020:
            print(f"\n=== {cell['name']} (2020 only) ===")
            _print_cell(cell)
    else:
        results_2020 = []
        print("  no 2020 games in the loaded schedule range")

    configuration = {
        "command": "illness-battery-screen",
        "schedules": str(args.schedules),
        "injuries": str(args.injuries),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "covid_season_excluded": COVID_SEASON,
        "illness_count_threshold": ILLNESS_COUNT_THRESHOLD,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "n_cells": len(results_primary),
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "covid_season_excluded": COVID_SEASON,
        "n_reg_games_before_push_drop": n_before_push_drop,
        "n_pushes_or_missing_dropped": pushes_or_missing,
        "n_scored_population_all_seasons": len(df_all),
        "n_scored_population_primary": len(df_primary),
        "n_home_missing": n_home_missing,
        "n_away_missing": n_away_missing,
        "coverage_by_season": {str(k): v for k, v in coverage_by_season.items()},
        "reliability": reliability,
        "predeclaration": "docs/illness_battery.md (frozen before scoring)",
        "results": results_primary,
        "ranked_by_abs_full_slate_effect": [cell["name"] for cell in ranked],
        "stratum_2020_covid_era_supplementary_not_pooled": results_2020,
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="illness-battery-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation battery (5 predeclared cells, nflverse illness "
            "designation, docs/illness_battery.md); mined family, every cell predeclared to "
            "record via a separate scripts/illness_battery_record.py call regardless of "
            "interval shape (AGENTS.md). 2020 stratum scored separately, NOT pooled, NOT "
            "recorded to the registry as its own entries."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
