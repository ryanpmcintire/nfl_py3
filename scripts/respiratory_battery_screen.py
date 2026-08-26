"""Total-respiratory-illness home-market battery: 5 predeclared cells against
the spread on NFL REG 2022-2026 ``location == "Home"`` games, week-blocked
bootstrap (season-blocked secondary), full-slate scaled, seeded and
deterministic.

**Predeclaration**: ``docs/respiratory_battery.md``, written and frozen
before this script was run against any cover outcome. Do not add, remove,
or redefine a cell here without updating that document first.

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
the separate ``scripts/respiratory_battery_record.py``.

**Point-in-time-safe as-of construction** (``docs/respiratory_battery.md``
section 3, read it): NSSP's ``covidcast`` rows carry an ``issue`` in
``YYYYWW`` epiweek format, not a calendar ``release_date`` the way FluView's
raw rows do. This script converts ``issue`` to a conservative calendar
release date -- the SATURDAY ending that epiweek, via ``epiweek_to_release_
date`` (the algebraic inverse of ``fluview_battery_screen.cdc_epiweek``,
reusing that module's hoisted ``_week_start``/``_epi_year_week1_start``
helpers rather than re-deriving the epiweek calendar) -- then reshapes each
of the three per-pathogen NSSP signals into FluView's own raw-row column
schema (``region``, ``epiweek``, ``issue``, ``release_date``, ``ili``) and
hands each one, UNCHANGED, to ``fluview_battery_screen.build_checkpoint_
tables`` / ``asof_lookup`` (imported, not reimplemented). The three
per-pathogen AS-OF values are then summed into ``respiratory_total`` -- only
when all three are non-missing, never a partial sum (docs/respiratory_
battery.md section 3).

Method reused verbatim from ``scripts/fluview_battery_screen.py`` /
``scripts/nfl_weather_battery_screen.py``: the same ``block_bootstrap_two_
group`` joint week-blocked bootstrap (``scripts/_common.py``), the same
full-slate effect scaling via ``nfl_ats.experiment_runner.scale_subset_
effect`` (imported, not reimplemented), the same ``probability_positive``
definition, the same reused ``PEAK_WEEKS`` constant (docs/respiratory_
battery.md section 4 -- NOT re-derived from NSSP's own national series),
and ``nfl_ats.cfb_qb_dependence.split_half_reliability`` (imported) for the
predeclared reliability check (section 6).

Writes JSON to ``artifacts/respiratory_battery/<UTC timestamp>/results.json``
and prints a summary table to stdout.
"""

from __future__ import annotations

import argparse
import datetime as _dt
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
from fluview_battery_ingest import STATE_BY_TEAM  # noqa: E402
from fluview_battery_screen import (  # noqa: E402
    PEAK_WEEKS,
    _epi_year_week1_start,
    asof_lookup,
    build_checkpoint_tables,
    cdc_epiweek,
)

from nfl_ats.cfb_qb_dependence import split_half_reliability  # noqa: E402
from nfl_ats.experiment_runner import scale_subset_effect  # noqa: E402
from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260826
SEASON_START = 2022
SEASON_END = 2026
DECILE_THRESHOLD = 0.90

# docs/respiratory_battery.md section 1 -- measured NSSP state-level floor;
# the three per-pathogen signals still updating as of ingest.
PATHOGEN_SIGNALS = ("pct_ed_visits_covid", "pct_ed_visits_influenza", "pct_ed_visits_rsv")


def epiweek_to_release_date(epiweek: int) -> pd.Timestamp:
    """Inverse of ``fluview_battery_screen.cdc_epiweek``: the SATURDAY ending
    the given ``YYYYWW`` epiweek (conservative "no earlier than" release-date
    anchor -- docs/respiratory_battery.md section 3 explains why the last
    day of the week, not NSSP's documented Friday-morning cadence, is used).
    """

    year, week_num = divmod(int(epiweek), 100)
    y1_start = _epi_year_week1_start(year)
    week_start_date = y1_start + _dt.timedelta(days=(week_num - 1) * 7)
    release_date = week_start_date + _dt.timedelta(days=6)
    return pd.Timestamp(release_date)


# ---------------------------------------------------------------------------
# 1. As-of checkpoint construction (docs/respiratory_battery.md section 3)
# ---------------------------------------------------------------------------


def _to_fluview_shape(raw: pd.DataFrame) -> pd.DataFrame:
    """Reshape one pathogen signal's raw NSSP rows into FluView's own raw-row
    column schema (``region``, ``epiweek``, ``issue``, ``release_date``,
    ``ili``) so ``fluview_battery_screen.build_checkpoint_tables`` /
    ``asof_lookup`` apply UNCHANGED (docs/respiratory_battery.md section 3).
    """

    shaped = raw.rename(columns={"time_value": "epiweek", "value": "ili"})[
        ["region", "epiweek", "issue", "ili"]
    ].copy()
    # Round-trip through a string, then pd.to_datetime -- measured this
    # session: schedules.parquet's own ``gameday`` column is stored as
    # plain strings, so ``load_schedules``' ``cutoff_date`` gets whatever
    # datetime64 resolution ``pd.to_datetime`` infers FROM A STRING
    # (``datetime64[us]`` on this session's pandas). A Series built
    # directly via ``.apply()`` over per-row Timestamp OBJECTS infers a
    # DIFFERENT resolution and merge_asof then refuses to compare them
    # (``pandas.errors.MergeError: incompatible merge keys``). Routing
    # through the same string-parsing path ``cutoff_date`` and FluView's
    # own ingest-parsed ``release_date`` both use guarantees a matching
    # resolution by construction rather than by hardcoding a specific unit
    # that could silently drift with a pandas/parquet version change.
    shaped["release_date"] = pd.to_datetime(
        shaped["issue"].apply(epiweek_to_release_date).astype(str), errors="raise"
    )
    return shaped


def build_pathogen_checkpoints(respiratory: pd.DataFrame, signal: str) -> dict[str, pd.DataFrame]:
    shaped = _to_fluview_shape(respiratory.loc[respiratory["pathogen_signal"] == signal])
    return build_checkpoint_tables(shaped)


def attach_asof_pathogen(
    df: pd.DataFrame, checkpoints: dict[str, pd.DataFrame], *, side: str, out_col: str
) -> pd.DataFrame:
    """Attach one pathogen's AS-OF value for ``home``/``away`` (``side``) by
    (state, cutoff_date) lookup, built once per unique pair -- same
    de-duplication strategy as ``fluview_battery_screen.attach_asof_ili``."""

    state_col = f"{side}_state"
    pairs = df[[state_col, "cutoff_date"]].rename(columns={state_col: "state"}).drop_duplicates()

    results = []
    for state, group in pairs.groupby("state"):
        checkpoint = checkpoints.get(state)
        if checkpoint is None or checkpoint.empty:
            merged = group.copy()
            merged["known_ili"] = np.nan
        else:
            looked_up = asof_lookup(checkpoint, group["cutoff_date"])
            merged = group.reset_index(drop=True).copy()
            merged["known_ili"] = looked_up["known_ili"].to_numpy()
        results.append(merged)
    lookup_table = pd.concat(results, ignore_index=True).drop_duplicates(
        subset=["state", "cutoff_date"]
    )

    df = df.merge(
        lookup_table.rename(columns={"state": state_col, "known_ili": out_col}),
        on=[state_col, "cutoff_date"],
        how="left",
    )
    return df


# ---------------------------------------------------------------------------
# 2. Population + feature construction
# ---------------------------------------------------------------------------


def _latest(glob_pattern: str, label: str) -> Path:
    candidates = sorted(REPO.glob(glob_pattern))
    if not candidates:
        raise FileNotFoundError(f"no {label} found matching {glob_pattern!r}")
    return candidates[-1]


def _latest_schedules() -> Path:
    return _latest("data/raw/*/schedules.parquet", "schedules.parquet snapshot")


def _latest_respiratory() -> Path:
    return _latest(
        "data/raw/respiratory/*/respiratory_raw.parquet", "respiratory_raw.parquet snapshot"
    )


def default_schedules() -> Path:
    """Resolve lazily so importing this module never requires local data."""
    return _latest_schedules()


def default_respiratory() -> Path:
    return _latest_respiratory()


def load_schedules(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    df = raw.loc[raw["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)]
    df = df.loc[df["location"] == "Home"].reset_index(drop=True)
    n_before_push_drop = len(df)

    df = add_ats_outcomes(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)
    pushes_or_missing = n_before_push_drop - len(df)

    df["gameday"] = pd.to_datetime(df["gameday"], errors="raise")
    weekday = df["gameday"].dt.weekday  # Monday=0 ... Sunday=6, Tuesday=1
    tuesday_offset = (weekday - 1) % 7
    df["cutoff_date"] = df["gameday"] - pd.to_timedelta(tuesday_offset, unit="D")
    df["week_block"] = df["season"] * 100 + df["week"]

    df["home_state"] = df["home_team"].map(STATE_BY_TEAM)
    df["away_state"] = df["away_team"].map(STATE_BY_TEAM)
    unmapped = df.loc[df["home_state"].isna() | df["away_state"].isna()]
    if len(unmapped):
        raise SystemExit(
            f"{len(unmapped)} games have a home/away team not in STATE_BY_TEAM: "
            f"{sorted(set(unmapped['home_team']) | set(unmapped['away_team']))}"
        )

    # docs/respiratory_battery.md section 4 -- REUSED unchanged from
    # fluview_battery_screen.PEAK_WEEKS, not re-derived from NSSP's own
    # national series.
    game_epiweek_of_year = df["gameday"].apply(lambda d: cdc_epiweek(d) % 100)
    df["is_peak_week"] = game_epiweek_of_year.isin(PEAK_WEEKS)

    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = pushes_or_missing
    return df


def attach_asof_respiratory(df: pd.DataFrame, respiratory: pd.DataFrame) -> pd.DataFrame:
    # Checkpoints built ONCE per signal (not per side) -- home and away both
    # look up the same per-state, per-signal checkpoint table.
    checkpoints_by_signal = {
        signal: build_pathogen_checkpoints(respiratory, signal) for signal in PATHOGEN_SIGNALS
    }
    for side in ("home", "away"):
        total = None
        for signal in PATHOGEN_SIGNALS:
            col = f"{side}_{signal}"
            df = attach_asof_pathogen(df, checkpoints_by_signal[signal], side=side, out_col=col)
            # Plain ``+`` (not ``.add(..., fill_value=...)``): pandas NaN
            # propagates through addition by default, which is exactly the
            # any-missing-poisons-the-sum rule docs/respiratory_battery.md
            # section 3 requires -- a missing pathogen value must never be
            # silently treated as 0.
            total = df[col] if total is None else total + df[col]
        df[f"{side}_respiratory_total"] = total
    return df


def build_state_week_panel(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (state, season, week): the AS-OF ``respiratory_total``
    value shared by every game whose home OR away side maps to that state
    that week. Used for the frozen per-state top-decile threshold (section
    3) and the reliability check (section 6)."""

    home_side = df[
        ["home_state", "season", "week", "cutoff_date", "home_respiratory_total"]
    ].rename(columns={"home_state": "state", "home_respiratory_total": "ili"})
    away_side = df[
        ["away_state", "season", "week", "cutoff_date", "away_respiratory_total"]
    ].rename(columns={"away_state": "state", "away_respiratory_total": "ili"})
    panel = pd.concat([home_side, away_side], ignore_index=True).drop_duplicates(
        subset=["state", "season", "week"]
    )
    return panel


def compute_state_thresholds(panel: pd.DataFrame) -> dict[str, float]:
    """Per-state 90th percentile of that state's own AS-OF value panel
    (docs/respiratory_battery.md section 3 -- frozen once, not re-derived)."""

    thresholds: dict[str, float] = {}
    for state, group in panel.groupby("state"):
        values = group["ili"].dropna()
        if len(values) >= 10:  # floor for a stable decile estimate
            thresholds[state] = float(values.quantile(DECILE_THRESHOLD))
    return thresholds


def attach_elevated_flags(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["home_threshold"] = df["home_state"].map(thresholds)
    df["away_threshold"] = df["away_state"].map(thresholds)
    df["home_missing"] = df["home_respiratory_total"].isna() | df["home_threshold"].isna()
    df["away_missing"] = df["away_respiratory_total"].isna() | df["away_threshold"].isna()
    df["home_elevated"] = np.where(
        df["home_missing"], False, df["home_respiratory_total"] >= df["home_threshold"]
    )
    df["away_elevated"] = np.where(
        df["away_missing"], False, df["away_respiratory_total"] >= df["away_threshold"]
    )
    return df


# ---------------------------------------------------------------------------
# 3. Cells (docs/respiratory_battery.md section 5)
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
        "respiratory_home_market_elevated",
        everyone & ~df["home_missing"],
        df["home_elevated"],
        "Home team's own state AS-OF covid+flu+rsv ED-visit-% sum in that state's own top "
        "decile (of its full as-of panel) vs. not, response home_cover. Predicted sign: "
        "NEGATIVE.",
    )
    add(
        "respiratory_away_market_elevated",
        everyone & ~df["away_missing"],
        df["away_elevated"],
        "Away team's own state AS-OF covid+flu+rsv ED-visit-% sum in that state's own top "
        "decile vs. not, response home_cover. Predicted sign: POSITIVE.",
    )

    diff_population = (
        ~df["home_missing"] & ~df["away_missing"] & (df["home_elevated"] != df["away_elevated"])
    )
    add(
        "respiratory_differential_home_worse",
        diff_population,
        df["home_elevated"] & ~df["away_elevated"],
        "Restricted to games where exactly one side is elevated (home XOR away); subset "
        "= home elevated & away not, complement = away elevated & home not, response "
        "home_cover. Predicted sign: NEGATIVE.",
    )

    peak_population = df["is_peak_week"] & ~df["home_missing"]
    add(
        "respiratory_peak_home_elevated",
        peak_population,
        df["home_elevated"],
        "Restricted to REUSED predeclared late-season peak weeks (fluview_battery_screen."
        "PEAK_WEEKS, not re-derived); home team's own state AS-OF respiratory sum elevated "
        "vs. not, response home_cover. Predicted sign: NEGATIVE.",
    )
    peak_population_away = df["is_peak_week"] & ~df["away_missing"]
    add(
        "respiratory_peak_away_elevated",
        peak_population_away,
        df["away_elevated"],
        "Restricted to REUSED predeclared late-season peak weeks; away team's own state "
        "AS-OF respiratory sum elevated vs. not, response home_cover. Predicted sign: "
        "POSITIVE.",
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
# 5. Reliability check (docs/respiratory_battery.md section 6)
# ---------------------------------------------------------------------------


def compute_reliability(panel: pd.DataFrame) -> dict[str, Any]:
    long = panel.dropna(subset=["ili"]).copy()
    long["team_id"] = long["state"]
    return split_half_reliability(long, "ili", seed=BOOTSTRAP_SEED)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--respiratory", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.schedules is None:
        args.schedules = default_schedules()
    if args.respiratory is None:
        args.respiratory = default_respiratory()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "respiratory_battery" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} ===")
    df = load_schedules(args.schedules)
    print(
        f"REG {SEASON_START}-{SEASON_END} 'Home'-location games: {df.attrs['n_before_push_drop']}, "
        f"pushes/missing dropped: {df.attrs['pushes_or_missing']}, scored population: {len(df)}"
    )

    print(f"\n=== loading {args.respiratory} ===")
    respiratory = pd.read_parquet(args.respiratory)
    print(
        f"respiratory rows: {len(respiratory)}, signals: "
        f"{sorted(respiratory['pathogen_signal'].unique())}, "
        f"states: {sorted(respiratory['region'].unique())}"
    )

    print("\n=== attaching as-of respiratory_total (covid+flu+rsv) ===")
    df = attach_asof_respiratory(df, respiratory)

    panel = build_state_week_panel(df)
    thresholds = compute_state_thresholds(panel)
    print(f"\n=== per-state top-decile ({DECILE_THRESHOLD}) thresholds ===")
    for state, thr in sorted(thresholds.items()):
        print(f"  {state}: {thr:.4f}")

    df = attach_elevated_flags(df, thresholds)

    n_home_missing = int(df["home_missing"].sum())
    n_away_missing = int(df["away_missing"].sum())
    print(
        f"\nmissingness: home_missing={n_home_missing}/{len(df)} "
        f"({n_home_missing / len(df):.1%}), away_missing={n_away_missing}/{len(df)} "
        f"({n_away_missing / len(df):.1%})"
    )
    coverage_by_season = (1.0 - df.groupby("season")["home_missing"].mean()).to_dict()
    print("coverage (fraction with non-missing home_respiratory_total) by season:")
    for season, cov in sorted(coverage_by_season.items()):
        print(f"  {season}: {cov:.1%}")

    print("\n=== reliability check (section 6) ===")
    reliability = compute_reliability(panel)
    print(
        f"  n_state_seasons={reliability['n_team_seasons']} "
        f"pearson_r={reliability['pearson_r']:.4f} "
        f"ci95={reliability['pearson_r_ci95']} "
        f"spearman_brown={reliability['spearman_brown_full_length_reliability']:.4f} "
        f"P+={reliability['probability_positive']:.4f}"
    )

    cells = build_cells(df)
    results = []
    for name, spec in cells.items():
        print(f"\n=== {name} ===")
        cell = score_cell(df, name, spec, samples=args.samples, seed=args.seed)
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
        "command": "respiratory-battery-screen",
        "schedules": str(args.schedules),
        "respiratory": str(args.respiratory),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "decile_threshold": DECILE_THRESHOLD,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "n_cells": len(cells),
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "n_reg_home_games_before_push_drop": df.attrs["n_before_push_drop"],
        "n_pushes_or_missing_dropped": df.attrs["pushes_or_missing"],
        "n_scored_population": len(df),
        "n_home_missing": n_home_missing,
        "n_away_missing": n_away_missing,
        "coverage_by_season": {str(k): v for k, v in coverage_by_season.items()},
        "state_thresholds": thresholds,
        "reliability": reliability,
        "predeclaration": "docs/respiratory_battery.md (frozen before scoring)",
        "results": results,
        "ranked_by_abs_full_slate_effect": [cell["name"] for cell in ranked],
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="respiratory-battery-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation battery (5 predeclared cells, NSSP covid+flu+rsv "
            "ED-visit-%% sum, extending fluview_battery from influenza-only to total "
            "respiratory illness); mined family, every cell predeclared to record via a "
            "separate scripts/respiratory_battery_record.py call regardless of interval "
            "shape (AGENTS.md)."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
