"""FluView home-market illness battery: 5 predeclared cells against the
spread on NFL REG 2010-2025 ``location == "Home"`` games, week-blocked
bootstrap (season-blocked secondary), full-slate scaled, seeded and
deterministic.

**Predeclaration**: ``docs/fluview_battery.md``, written and frozen before
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
the separate ``scripts/fluview_battery_record.py``.

**Point-in-time-safe as-of construction** (``docs/fluview_battery.md``
section 3, read it): per state, sort the full multi-issue history (from
``scripts/fluview_battery_ingest.py``'s snapshot) by ``release_date``
ascending and carry forward the running-max ``epiweek`` seen so far -- a
monotone-in-``known_epiweek`` checkpoint table. A game's as-of value is
``merge_asof`` of the game's decision-cutoff Tuesday against that table,
direction="backward". If no checkpoint row has ``release_date <=`` the
cutoff, the value is missing, not a leaked final value -- this holds by
construction for nearly all of 2010-2017 (measured, section 1), which is
expected and reported, not corrected.

Method reused verbatim from ``scripts/nfl_weather_battery_screen.py`` /
``scripts/team_style_screen.py``: the same ``block_bootstrap_two_group``
joint week-blocked bootstrap, the same full-slate effect scaling via
``nfl_ats.experiment_runner.scale_subset_effect`` (imported, not
reimplemented), the same ``probability_positive`` definition, and
``nfl_ats.cfb_qb_dependence.split_half_reliability`` (imported) for the
predeclared reliability check (section 6).

Writes JSON to ``artifacts/fluview_battery/<UTC timestamp>/results.json``
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

from nfl_ats.cfb_qb_dependence import split_half_reliability  # noqa: E402
from nfl_ats.experiment_runner import scale_subset_effect  # noqa: E402
from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819
SEASON_START = 2010
SEASON_END = 2025
DECILE_THRESHOLD = 0.90

# docs/fluview_battery.md section 4 -- measured, predictor-distribution-only,
# frozen before any cover-rate sign was examined.
PEAK_WEEKS = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 51, 52, 53})


def _week_start(x: _dt.date) -> _dt.date:
    """Sunday on/before ``x`` -- the start of ``x``'s CDC MMWR week.

    Hoisted to module level (2026-08-26, pure code motion, no behavior
    change) from a ``cdc_epiweek``-local closure so
    ``scripts/respiratory_battery_screen.py`` can reuse the exact same
    calendar arithmetic for its epiweek -> release-date inverse, rather
    than re-deriving it.
    """

    dow_sun0 = (x.weekday() + 1) % 7  # Sun=0, Mon=1, ..., Sat=6
    return x - _dt.timedelta(days=dow_sun0)


def _epi_year_week1_start(year: int) -> _dt.date:
    """Start date (a Sunday) of epi-week 1 of ``year`` -- hoisted alongside
    ``_week_start``, same reason."""

    jan1 = _dt.date(year, 1, 1)
    wk_start = _week_start(jan1)
    days_in_new_year = 7 - (jan1 - wk_start).days
    if days_in_new_year >= 4:
        return wk_start
    return wk_start + _dt.timedelta(days=7)


def cdc_epiweek(date: pd.Timestamp) -> int:
    """CDC MMWR epiweek (YYYYWW) for a calendar date. Sunday-start weeks; a
    year's week 1 is the first week with >=4 days in that calendar year --
    the standard CDC/MMWR definition, matching the ``epiweek`` field Delphi
    itself returns. Deliberately NOT ``pandas``' ``isocalendar()`` (ISO 8601
    weeks are Monday-start with a Thursday rule) -- validated this session
    against a live Delphi row (``ca`` epiweek 201840, ``release_date``
    2018-10-12, a Friday 6 days after this function's computed week-end of
    Saturday 2018-10-06) to confirm alignment with Delphi's own numbering.
    """

    d = date.date() if hasattr(date, "date") else date

    ws = _week_start(d)
    for cand_year in (d.year - 1, d.year, d.year + 1):
        y1_start = _epi_year_week1_start(cand_year)
        y2_start = _epi_year_week1_start(cand_year + 1)
        if y1_start <= ws < y2_start:
            week_num = (ws - y1_start).days // 7 + 1
            return cand_year * 100 + week_num
    raise ValueError(f"could not resolve epiweek for {date}")


def _latest(glob_pattern: str, label: str) -> Path:
    candidates = sorted(REPO.glob(glob_pattern))
    if not candidates:
        raise FileNotFoundError(f"no {label} found matching {glob_pattern!r}")
    return candidates[-1]


def _latest_schedules() -> Path:
    return _latest("data/raw/*/schedules.parquet", "schedules.parquet snapshot")


def _latest_fluview() -> Path:
    return _latest("data/raw/fluview/*/fluview_raw.parquet", "fluview_raw.parquet snapshot")


def default_schedules() -> Path:
    """Resolve lazily so importing this module never requires local data."""
    return _latest_schedules()


DEFAULT_FLUVIEW = _latest_fluview()


# ---------------------------------------------------------------------------
# 1. As-of checkpoint construction (docs/fluview_battery.md section 3)
# ---------------------------------------------------------------------------


def build_checkpoint_tables(fluview: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-state monotone-in-``known_epiweek`` checkpoint table, indexed by
    real calendar ``release_date``, ready for ``merge_asof``."""

    tables: dict[str, pd.DataFrame] = {}
    for region, group in fluview.groupby("region"):
        # Measured this session: Delphi returns release_date=null for EVERY
        # row of the "ny" region (verified live, not a parsing artifact) --
        # an upstream data gap, not something this script can recover from.
        # Rows with no release_date carry no point-in-time information, so
        # they are dropped here; a region left with zero rows falls through
        # to attach_asof_ili's existing empty-checkpoint fallback (known_ili
        # = NaN for every game mapped to that state, i.e. genuinely missing,
        # not leaked and not silently defaulted).
        group = group.dropna(subset=["release_date"])
        if group.empty:
            tables[region] = pd.DataFrame(columns=["release_date", "known_epiweek", "known_ili"])
            continue
        # Collapse same-release_date rows to the max epiweek released at that
        # instant (the freshest content available at that exact checkpoint):
        # sort ascending by epiweek within each release_date, keep the last.
        collapsed = (
            group.sort_values(["release_date", "epiweek"])
            .drop_duplicates(subset="release_date", keep="last")
            .sort_values("release_date")
            .reset_index(drop=True)
        )
        # Running max epiweek -- carries forward whichever (epiweek, ili) pair
        # is freshest as release_date advances (revisions to OLD epiweeks
        # that arrive late must never override a newer epiweek already known).
        running_max_idx = collapsed["epiweek"].cummax()
        is_new_max = collapsed["epiweek"] >= running_max_idx.shift(1).fillna(-1)
        checkpoint = collapsed.loc[is_new_max, ["release_date", "epiweek", "ili"]].reset_index(
            drop=True
        )
        checkpoint = checkpoint.rename(columns={"epiweek": "known_epiweek", "ili": "known_ili"})
        tables[region] = checkpoint
    return tables


def asof_lookup(checkpoint: pd.DataFrame, cutoff_dates: pd.Series) -> pd.DataFrame:
    """merge_asof cutoff_dates against a state's checkpoint table."""

    left = pd.DataFrame({"cutoff_date": pd.to_datetime(cutoff_dates.to_numpy())}).sort_values(
        "cutoff_date"
    )
    left["_orig_order"] = left.index
    right = checkpoint.sort_values("release_date")
    merged = pd.merge_asof(
        left,
        right,
        left_on="cutoff_date",
        right_on="release_date",
        direction="backward",
    )
    return merged.sort_values("_orig_order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Population + feature construction
# ---------------------------------------------------------------------------


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

    # National-mean-week-of-year classification for the peak-week cells
    # (docs/fluview_battery.md section 4) -- derived from the game's own
    # gameday via the CDC epiweek convention (matching how PEAK_WEEKS was
    # itself measured from Delphi's "nat" series), no point-in-time concern.
    game_epiweek_of_year = df["gameday"].apply(lambda d: cdc_epiweek(d) % 100)
    df["is_peak_week"] = game_epiweek_of_year.isin(PEAK_WEEKS)

    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = pushes_or_missing
    return df


def attach_asof_ili(df: pd.DataFrame, checkpoints: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach ``home_ili``/``away_ili`` as-of values by (state, cutoff_date)
    lookup, built once per unique (state, cutoff_date) pair to avoid
    re-running merge_asof per game row."""

    pairs = pd.concat(
        [
            df[["home_state", "cutoff_date"]].rename(columns={"home_state": "state"}),
            df[["away_state", "cutoff_date"]].rename(columns={"away_state": "state"}),
        ],
        ignore_index=True,
    ).drop_duplicates()

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
        lookup_table.rename(columns={"state": "home_state", "known_ili": "home_ili"}),
        on=["home_state", "cutoff_date"],
        how="left",
    )
    df = df.merge(
        lookup_table.rename(columns={"state": "away_state", "known_ili": "away_ili"}),
        on=["away_state", "cutoff_date"],
        how="left",
    )
    return df


def build_state_week_panel(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (state, season, week): the as-of ILI value shared by
    every game whose home OR away side maps to that state that week. Used
    for the frozen per-state top-decile threshold (section 3) and the
    reliability check (section 6)."""

    home_side = df[["home_state", "season", "week", "cutoff_date", "home_ili"]].rename(
        columns={"home_state": "state", "home_ili": "ili"}
    )
    away_side = df[["away_state", "season", "week", "cutoff_date", "away_ili"]].rename(
        columns={"away_state": "state", "away_ili": "ili"}
    )
    panel = pd.concat([home_side, away_side], ignore_index=True).drop_duplicates(
        subset=["state", "season", "week"]
    )
    return panel


def compute_state_thresholds(panel: pd.DataFrame) -> dict[str, float]:
    """Per-state 90th percentile of that state's own AS-OF value panel
    (docs/fluview_battery.md section 3 -- frozen once, not re-derived)."""

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
    df["home_missing"] = df["home_ili"].isna() | df["home_threshold"].isna()
    df["away_missing"] = df["away_ili"].isna() | df["away_threshold"].isna()
    df["home_elevated"] = np.where(
        df["home_missing"], False, df["home_ili"] >= df["home_threshold"]
    )
    df["away_elevated"] = np.where(
        df["away_missing"], False, df["away_ili"] >= df["away_threshold"]
    )
    return df


# ---------------------------------------------------------------------------
# 3. Cells (docs/fluview_battery.md section 5)
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
        "fluview_home_market_elevated",
        everyone & ~df["home_missing"],
        df["home_elevated"],
        "Home team's own state AS-OF ILI in that state's own top decile (of its full "
        "as-of panel) vs. not, response home_cover. Predicted sign: NEGATIVE.",
    )
    add(
        "fluview_away_market_elevated",
        everyone & ~df["away_missing"],
        df["away_elevated"],
        "Away team's own state AS-OF ILI in that state's own top decile vs. not, "
        "response home_cover. Predicted sign: POSITIVE.",
    )

    diff_population = (
        ~df["home_missing"] & ~df["away_missing"] & (df["home_elevated"] != df["away_elevated"])
    )
    add(
        "fluview_differential_home_worse",
        diff_population,
        df["home_elevated"] & ~df["away_elevated"],
        "Restricted to games where exactly one side is elevated (home XOR away); subset "
        "= home elevated & away not, complement = away elevated & home not, response "
        "home_cover. Predicted sign: NEGATIVE.",
    )

    peak_population = df["is_peak_week"] & ~df["home_missing"]
    add(
        "fluview_peak_home_elevated",
        peak_population,
        df["home_elevated"],
        "Restricted to predeclared late-season peak weeks (national ILI top-quartile "
        "weeks, measured predictor-only); home team's own state AS-OF ILI elevated vs. "
        "not, response home_cover. Predicted sign: NEGATIVE.",
    )
    peak_population_away = df["is_peak_week"] & ~df["away_missing"]
    add(
        "fluview_peak_away_elevated",
        peak_population_away,
        df["away_elevated"],
        "Restricted to predeclared late-season peak weeks; away team's own state AS-OF "
        "ILI elevated vs. not, response home_cover. Predicted sign: POSITIVE.",
    )

    expected = 5
    assert len(cells) == expected, f"expected {expected} predeclared cells, got {len(cells)}"
    return cells


# ---------------------------------------------------------------------------
# 4. Bootstrap (algorithm-identical to nfl_weather_battery_screen.py)
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
# 5. Reliability check (docs/fluview_battery.md section 6)
# ---------------------------------------------------------------------------


def compute_reliability(panel: pd.DataFrame) -> dict[str, Any]:
    long = panel.dropna(subset=["ili"]).copy()
    long["team_id"] = long["state"]
    # "week" is already the NFL week number carried through from the
    # schedules join -- reused directly for the odd/even parity split
    # (same convention as every other split-half precedent in the repo).
    return split_half_reliability(long, "ili", seed=BOOTSTRAP_SEED)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--fluview", type=Path, default=DEFAULT_FLUVIEW)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.schedules is None:
        args.schedules = default_schedules()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "fluview_battery" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} ===")
    df = load_schedules(args.schedules)
    print(
        f"REG {SEASON_START}-{SEASON_END} 'Home'-location games: {df.attrs['n_before_push_drop']}, "
        f"pushes/missing dropped: {df.attrs['pushes_or_missing']}, scored population: {len(df)}"
    )

    print(f"\n=== loading {args.fluview} ===")
    fluview = pd.read_parquet(args.fluview)
    print(f"fluview rows: {len(fluview)}, regions: {sorted(fluview['region'].unique())}")

    print("\n=== building per-state as-of checkpoint tables ===")
    checkpoints = build_checkpoint_tables(fluview.loc[fluview["region"] != "nat"])
    for state, cp in sorted(checkpoints.items()):
        print(f"  {state}: {len(cp)} checkpoints, earliest release {cp['release_date'].min()}")

    print("\n=== attaching as-of ILI values ===")
    df = attach_asof_ili(df, checkpoints)

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
    print("coverage (fraction with non-missing home_ili) by season:")
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
        "command": "fluview-battery-screen",
        "schedules": str(args.schedules),
        "fluview": str(args.fluview),
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
        "predeclaration": "docs/fluview_battery.md (frozen before scoring)",
        "results": results,
        "ranked_by_abs_full_slate_effect": [cell["name"] for cell in ranked],
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="fluview-battery-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation battery (5 predeclared cells, FluView state ILI); "
            "mined family, every cell predeclared to record via a separate "
            "scripts/fluview_battery_record.py call regardless of interval shape (AGENTS.md)."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
