"""Daylight-saving TRANSITION battery: does the clock-change shock move ATS
outcomes, not merely need correct UTC-offset arithmetic?

**Predeclaration**: ``docs/dst_transition_battery.md``, written and frozen
before this script was run against any cover-rate outcome. Do not add,
remove, or redefine a cell here without updating that document first.

**Binding taxonomy (owned verbatim, per AGENTS.md/CLAUDE.md)**: an interval
or CI that contains zero is NEVER grounds to reject, fail, or close an
experiment -- at this evaluator's ~2-point resolution "contains zero" is the
EXPECTED outcome for a real small signal. Only two closing grounds: (1)
refuted mechanism -- RESOLVED wrong sign (whole interval on the wrong side
of the predicted direction) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is ``unresolved_below_power``: record with ``probability_positive``, never
"contains zero". If a record command errors, the verdict is wrong, not the
validator. This script does not decide anything, it only measures; every
cell is recorded to the registry regardless of sign or interval shape via
the separate ``scripts/record_dst_transition_battery.py``.

**Transition dates are MEASURED, not hardcoded** (``docs/dst_transition_battery.md``
section 1): scan ``America/New_York``'s UTC offset via stdlib ``zoneinfo``
day-by-day across Oct 1 - Dec 1 (fall) / Feb 1 - May 1 (spring) for each
season and record the first date the offset changes. This is the exact same
mechanism ``docs/travel_rest_battery.md`` already uses for
``tz_delta_eastbound`` -- this battery is the first to test the transition
itself, not just use it as plumbing.

**Population and machinery are reused, not rewritten**:
``nfl_travel_rest_battery_screen.load_population`` (imported directly)
supplies the REG 2009-2025 population, ``home_cover``
(``nfl_ats.features.add_ats_outcomes``), ``week_block``, and the already-
validated ``tz_delta_eastbound`` column reused verbatim for cell D5. Effect
scaling uses ``nfl_ats.experiment_runner.scale_subset_effect`` (imported,
not reimplemented); bootstrap uses ``_common.block_bootstrap_two_group``
(the same joint week-blocked / season-blocked engine every prior battery in
this family uses).

**Measure-only.** Never writes ``registry/weak_signals.json``. Writes a
provenance-stamped run log via ``write_experiment_artifact``.

Additionally reports, per cell and per blocking, ``n_flag_blocks``: the
count of DISTINCT blocks that contain at least one flagged row -- the
number that actually governs a once-or-twice-per-season flag's resampling
variability, as opposed to ``n_blocks`` (every block in the whole restricted
population, subset and complement together), which is much larger and can
be misleadingly reassuring on its own (``docs/dst_transition_battery.md``
section 5).

Writes JSON to ``artifacts/dst_transition_battery/<UTC timestamp>/results.json``
and prints a summary table to stdout.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

sys.path.append(str(REPO / "scripts"))

from _common import block_bootstrap_two_group  # noqa: E402
from nfl_travel_rest_battery_screen import (  # noqa: E402
    DEFAULT_COORDS_PATH,
    load_coords,
    load_population,
)

from nfl_ats.experiment_runner import scale_subset_effect  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260826
SEASON_START = 2009
SEASON_END = 2025
EASTBOUND_HOURS = 2.0  # matches travel_rest_eastbound_multizone's own threshold
WINDOW_DAYS = 7  # [0, 6] inclusive -- a true calendar week starting at the transition
PLACEBO_OFFSET_DAYS = 21  # 3 weeks before the real transition, zero overlap (measured)


def _latest_schedules() -> Path:
    candidates = sorted((REPO / "data/raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


DEFAULT_SCHEDULES = _latest_schedules()


# ---------------------------------------------------------------------------
# 1. Transition dates (measured via zoneinfo, docs/dst_transition_battery.md sec 1)
# ---------------------------------------------------------------------------


def _scan_transition(year: int, start: date, end: date) -> date:
    """First date in ``[start, end)`` where America/New_York's noon UTC
    offset differs from the previous day's -- a DST transition, found by
    direct observation, not by assuming which week/day it falls on."""

    ny = ZoneInfo("America/New_York")
    d = start
    prev = datetime(d.year, d.month, d.day, 12, tzinfo=ny).utcoffset()
    while d < end:
        d2 = d + timedelta(days=1)
        cur = datetime(d2.year, d2.month, d2.day, 12, tzinfo=ny).utcoffset()
        if cur != prev:
            if d2.weekday() != 6:
                raise AssertionError(f"transition on {d2} is not a Sunday for year {year}")
            return d2
        prev = cur
        d = d2
    raise RuntimeError(f"no DST transition found in [{start}, {end}) for year {year}")


def fall_transition_date(year: int) -> date:
    return _scan_transition(year, date(year, 10, 1), date(year, 12, 1))


def spring_transition_date(year: int) -> date:
    return _scan_transition(year, date(year, 2, 1), date(year, 5, 1))


def phoenix_observes_dst(year: int) -> bool:
    """True if America/Phoenix's UTC offset ever changes in ``year`` (it
    does not, every year -- checked programmatically, not assumed)."""

    az = ZoneInfo("America/Phoenix")
    summer = datetime(year, 7, 1, 12, tzinfo=az).utcoffset()
    winter = datetime(year, 1, 15, 12, tzinfo=az).utcoffset()
    return summer != winter


def latest_postseason_gameday_by_season(schedules_path: Path) -> pd.Series:
    """Measured diagnostic for docs/dst_transition_battery.md section 2 --
    latest game date, per season, among non-REG games. Population-only, no
    cover-rate outcome."""

    raw = pd.read_parquet(schedules_path)
    post = raw.loc[raw["game_type"] != "REG"].copy()
    post["gameday"] = pd.to_datetime(post["gameday"], errors="raise")
    return post.groupby("season")["gameday"].max()


# ---------------------------------------------------------------------------
# 2. Population + derived DST columns
# ---------------------------------------------------------------------------


def build_population(schedules_path: Path, coords_path: Path) -> pd.DataFrame:
    coords = load_coords(coords_path)
    df = load_population(schedules_path, coords)  # REG SEASON_START-SEASON_END, home_cover, etc.
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)

    seasons = sorted(int(s) for s in df["season"].unique())
    fall_by_season = {s: fall_transition_date(s) for s in seasons}
    spring_by_season = {s: spring_transition_date(s) for s in seasons}

    df["fall_transition_date"] = df["season"].map(fall_by_season)
    df["days_since_fall_transition"] = (
        df["gameday_dt"].dt.normalize() - pd.to_datetime(df["fall_transition_date"])
    ).dt.days

    df["placebo_anchor_date"] = df["fall_transition_date"] - pd.to_timedelta(
        PLACEBO_OFFSET_DAYS, unit="D"
    )
    df["days_since_placebo_anchor"] = (
        df["gameday_dt"].dt.normalize() - pd.to_datetime(df["placebo_anchor_date"])
    ).dt.days

    df.attrs["fall_by_season"] = {str(k): v.isoformat() for k, v in fall_by_season.items()}
    df.attrs["spring_by_season"] = {str(k): v.isoformat() for k, v in spring_by_season.items()}
    return df


# ---------------------------------------------------------------------------
# 3. Cells (docs/dst_transition_battery.md section 4)
# ---------------------------------------------------------------------------


def build_cells(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}

    def add(
        name: str, population: pd.Series, flag: pd.Series, predicted_sign: int, description: str
    ) -> None:
        cells[name] = {
            "population": population.fillna(False).astype(bool),
            "flag": flag.fillna(False).astype(bool),
            "predicted_sign": predicted_sign,
            "description": description,
        }

    everyone = pd.Series(True, index=df.index)

    d1_flag = df["days_since_fall_transition"].between(0, WINDOW_DAYS - 1)
    add(
        "dst_fall_transition_shock",
        everyone,
        d1_flag,
        1,
        "Game falls within the 7 calendar days starting at that season's measured "
        "fall-back Sunday (days_since_fall_transition in [0,6]) vs. the rest of the "
        "REG slate, response home_cover. Predicted sign: POSITIVE (traveling away "
        "team absorbs both ordinary travel fatigue and the national clock shock; "
        "home team does not travel).",
    )

    ari_home = df["home_team"] == "ARI"
    ari_away = df["away_team"] == "ARI"
    ari_in_game = ari_home | ari_away

    add(
        "dst_arizona_home_shield",
        d1_flag & (ari_home | ~ari_in_game),
        d1_flag & ari_home,
        1,
        "Restricted to D1's transition window: home team is ARI (measured "
        "non-DST-observing every year) vs. transition-window games with no ARI "
        "participant at all, response home_cover. Predicted sign: POSITIVE (home "
        "team had zero clock disruption; traveling opponent had both travel fatigue "
        "and the clock shock). Extremely thin by construction -- disclosed in "
        "docs/dst_transition_battery.md section 4.",
    )
    add(
        "dst_arizona_away_shield",
        d1_flag & (ari_away | ~ari_in_game),
        d1_flag & ari_away,
        -1,
        "Restricted to D1's transition window: away team is ARI vs. transition-"
        "window games with no ARI participant at all, response home_cover. "
        "Predicted sign: NEGATIVE (away team ARI carries only ordinary travel "
        "burden, no clock shock; home team still absorbed the national shock in "
        "its own bed). Extremely thin by construction -- disclosed in "
        "docs/dst_transition_battery.md section 4.",
    )

    eastbound = df["tz_delta_eastbound"].notna() & (df["tz_delta_eastbound"] >= EASTBOUND_HOURS)
    add(
        "dst_transition_eastbound_interaction",
        eastbound,
        eastbound & d1_flag,
        1,
        "Restricted to games with tz_delta_eastbound >= 2 hours (the already-"
        "registered travel_rest_eastbound_multizone construct, reused verbatim): "
        "also falling in D1's transition window vs. not, response home_cover. "
        "Predicted sign: POSITIVE (eastbound circadian disruption and the national "
        "clock shock compound).",
    )

    placebo_flag = df["days_since_placebo_anchor"].between(0, WINDOW_DAYS - 1)
    add(
        "dst_placebo_shifted_window",
        everyone,
        placebo_flag,
        0,
        "Negative/specificity control: identically-shaped 7-day window anchored 21 "
        "days before the real fall transition (zero calendar overlap with D1, "
        "measured) vs. the rest of the REG slate, response home_cover. Predicted "
        "sign: NULL -- no DST mechanism operates on this window.",
    )

    expected = 5
    assert len(cells) == expected, f"expected {expected} predeclared cells, got {len(cells)}"
    return cells


# ---------------------------------------------------------------------------
# 4. Bootstrap (algorithm-identical to prior batteries in this family)
# ---------------------------------------------------------------------------


def summarize_cell(
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

    n_blocks = int(work[block_col].nunique())
    n_flag_blocks = int(work.loc[work["_flag"], block_col].nunique())
    n_complement_blocks = int(work.loc[~work["_flag"], block_col].nunique())

    return {
        "n_total": n_total,
        "n_flag": n_flag,
        "n_complement": n_complement,
        "n_blocks": n_blocks,
        "n_flag_blocks": n_flag_blocks,
        "n_complement_blocks": n_complement_blocks,
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

    week_blocked = summarize_cell(
        scored, flag=scored_flag, block_col="week_block", samples=samples, seed=seed
    )
    season_blocked = summarize_cell(
        scored, flag=scored_flag, block_col="season", samples=samples, seed=seed
    )

    return {
        "name": name,
        "description": spec["description"],
        "predicted_sign": spec["predicted_sign"],
        "n_population": int(population.sum()),
        "week_blocked": week_blocked,
        "season_blocked_secondary": season_blocked,
    }


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=DEFAULT_SCHEDULES)
    parser.add_argument("--coords", type=Path, default=DEFAULT_COORDS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "dst_transition_battery" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} + {args.coords} ===")
    df = build_population(args.schedules, args.coords)
    print(f"scored population: {len(df)} REG {SEASON_START}-{SEASON_END} games")

    print("\n=== measured DST transition dates (zoneinfo scan, not hardcoded) ===")
    for season in sorted(int(s) for s in df["season"].unique()):
        print(
            f"  {season}: fall={df.attrs['fall_by_season'][str(season)]} "
            f"spring={df.attrs['spring_by_season'][str(season)]}"
        )

    print("\n=== section 2 diagnostic: spring-transition/postseason overlap ===")
    # A season's postseason plays out in Jan/Feb of the FOLLOWING calendar
    # year, so the relevant spring transition is spring_transition_date of
    # the postseason game's own calendar year -- not the season number's own
    # (pre-season) March date, which would compare against a transition that
    # already passed before that season even started.
    latest_post = latest_postseason_gameday_by_season(args.schedules)
    n_overlap = 0
    for season in sorted(int(s) for s in df["season"].unique()):
        latest = latest_post.get(season)
        if latest is None or pd.isna(latest):
            continue
        spring_date = spring_transition_date(latest.year)
        overlap = latest.date() >= spring_date
        print(
            f"  season {season}: latest postseason {latest.date()} vs. spring transition "
            f"{spring_date} ({'OVERLAP' if overlap else 'no overlap'})"
        )
        if overlap:
            n_overlap += 1
    print(f"  seasons with postseason game on/after spring transition: {n_overlap} (measured)")

    print("\n=== Phoenix DST participation (measured) ===")
    phoenix_ever_dst = any(
        phoenix_observes_dst(s) for s in sorted(int(s) for s in df["season"].unique())
    )
    print(f"  America/Phoenix observes DST in any scored season: {phoenix_ever_dst}")

    cells = build_cells(df)
    results = []
    for name, spec in cells.items():
        print(f"\n=== {name} (predicted_sign={spec['predicted_sign']:+d}) ===")
        cell = score_cell(df, name, spec, samples=args.samples, seed=args.seed)
        results.append(cell)
        wb = cell["week_blocked"]
        sb = cell["season_blocked_secondary"]
        print(f"  n_population={cell['n_population']} n_flag={wb.get('n_flag')}")
        if wb.get("insufficient_data"):
            print("  insufficient data (empty subset or complement)")
            continue
        print(
            f"  subset_cover={wb['subset_cover']:.4f} "
            f"complement_cover={wb['complement_cover']:.4f} "
            f"raw_gap={wb['raw_gap_pts']:+.3f}pts frac_of_slate={wb['fraction_of_slate']:.4f}"
        )
        print(
            f"  [week-blocked]   effect={wb['full_slate_effect_pts']:+.4f}pts "
            f"95% [{wb['ci95_scaled'][0]:+.4f}, {wb['ci95_scaled'][1]:+.4f}] "
            f"P+={wb['probability_positive']:.4f} n_blocks={wb['n_blocks']} "
            f"n_flag_blocks={wb['n_flag_blocks']} (<10 floor: {wb['n_flag_blocks'] < 10})"
        )
        if not sb.get("insufficient_data"):
            print(
                f"  [season-blocked] effect={sb['full_slate_effect_pts']:+.4f}pts "
                f"95% [{sb['ci95_scaled'][0]:+.4f}, {sb['ci95_scaled'][1]:+.4f}] "
                f"P+={sb['probability_positive']:.4f} n_blocks={sb['n_blocks']} "
                f"n_flag_blocks={sb['n_flag_blocks']} (<10 floor: {sb['n_flag_blocks'] < 10})"
            )

    configuration = {
        "command": "dst-transition-battery-screen",
        "schedules": str(args.schedules),
        "coords": str(args.coords),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "window_days": WINDOW_DAYS,
        "placebo_offset_days": PLACEBO_OFFSET_DAYS,
        "eastbound_hours": EASTBOUND_HOURS,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "n_cells": len(cells),
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "n_scored_population": len(df),
        "fall_transition_by_season": df.attrs["fall_by_season"],
        "spring_transition_by_season": df.attrs["spring_by_season"],
        "spring_postseason_overlap_seasons": n_overlap,
        "phoenix_ever_observes_dst": phoenix_ever_dst,
        "predeclaration": "docs/dst_transition_battery.md (frozen before scoring)",
        "results": results,
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="dst-transition-battery-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation battery (5 predeclared cells, DST transition "
            "shock); mined family, every cell predeclared to record via a separate "
            "scripts/record_dst_transition_battery.py call regardless of interval shape "
            "(AGENTS.md)."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
