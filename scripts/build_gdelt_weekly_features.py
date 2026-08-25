"""Derive per-(season, week, team) GDELT news-attention aggregates from the
raw daily archive written by ``scripts/ingest_gdelt_backfill.py``.

Scope: INGESTION/DERIVATION ONLY -- no experiments, no registry writes, no
``src/nfl_ats`` changes. Output is a plain parquet + manifest under
``data/processed/``, for a future feature-integration task to pick up.

**Two as-of cutoffs per (season, week, team)**, both anchored to that
schedule week's own Tuesday (matching
``scripts/attention_battery_screen.py``'s window rule exactly, for direct
comparability with the Wikipedia-pageviews construct):

- ``tuesday_*``: trailing 7-day window ending that week's Tuesday
  (``[T-6, T]``). Point-in-time safe for EVERY game in the week, including
  Thursday -- this is the pool's line-freeze cutoff
  (``docs/late_week_refresh.md``: "Grading is always against the frozen
  Tuesday line").
- ``saturday_*``: trailing 7-day window ending that week's Saturday
  (``[T-2, T+4]``, i.e. Sunday through Saturday). **NOT point-in-time safe
  for that week's Thursday game** (Saturday is 2 days after Thursday's
  kickoff) -- safe only for Saturday/Sunday/Monday games. This is a known,
  documented gap, not a bug: ``docs/late_week_refresh.md`` establishes picks
  stay editable to kickoff via a Saturday refresh pass, so a Saturday-cutoff
  attention read is the complementary "freshest legal information" cutoff
  for the majority of the slate, at the cost of being inapplicable to TNF.
  Any downstream feature use MUST gate on ``gameday`` weekday (TNF rows
  should fall back to the ``tuesday_*`` value) -- this script does not do
  that gating itself since it has no game-outcome/feature-pipeline role in
  this task's scope; it is called out here and in
  ``docs/gdelt_backfill.md`` as a predeclared caveat for whoever builds the
  feature.

Both cutoffs get: raw window-summed article count (``*_raw_count``), the
GDELT-reported total monitored corpus size summed over the same window
(``*_monitored_total``, lets a future user re-derive a normalized
share-of-coverage metric instead of a raw count), a count-weighted average
tone over the window if timelinetone data is present
(``*_avg_tone``), and a trailing z-score of the raw count (mean/std over up
to 8 prior in-season team-week observations of THAT SAME cutoff, min 2 --
identical recipe to the Wikipedia construct's ``attention_z``, so
``tuesday_z`` is the direct GDELT analogue of that construct's ``attention_z``).
"""

from __future__ import annotations

import argparse
import importlib.util
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

from _common import latest_schedules  # noqa: E402

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.features import add_ats_outcomes  # noqa: E402


def _load_base_module():
    spec = importlib.util.spec_from_file_location(
        "attention_battery_screen", REPO / "scripts" / "attention_battery_screen.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()

SEASON_START = 2017
SEASON_END = 2025
TRAILING_WINDOW = 8
TRAILING_MIN = 2


# --------------------------------------------------------------------------
# Parse raw GDELT JSON -> per-team daily frames
# --------------------------------------------------------------------------


def _parse_volraw(payload: dict[str, Any]) -> pd.DataFrame:
    """timelinevolraw payload -> DataFrame[date, raw_count, monitored_total]."""

    timeline = payload.get("timeline") if payload else None
    if not timeline:
        return pd.DataFrame(columns=["date", "raw_count", "monitored_total"]).astype(
            {"date": "datetime64[ns]"}
        )
    data = timeline[0].get("data", [])
    rows = []
    for point in data:
        raw_date = str(point.get("date", ""))[:8]
        if len(raw_date) != 8:
            continue
        rows.append(
            {
                "date": pd.Timestamp(raw_date),
                "raw_count": float(point.get("value", 0.0)),
                "monitored_total": float(point.get("norm", 0.0)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "raw_count", "monitored_total"]).astype(
            {"date": "datetime64[ns]"}
        )
    frame = pd.DataFrame(rows)
    return frame.groupby("date", as_index=False).sum().sort_values("date")


def _parse_tone(payload: dict[str, Any]) -> pd.DataFrame:
    """timelinetone payload -> DataFrame[date, avg_tone]."""

    timeline = payload.get("timeline") if payload else None
    if not timeline:
        return pd.DataFrame(columns=["date", "avg_tone"]).astype({"date": "datetime64[ns]"})
    data = timeline[0].get("data", [])
    rows = []
    for point in data:
        raw_date = str(point.get("date", ""))[:8]
        if len(raw_date) != 8:
            continue
        rows.append({"date": pd.Timestamp(raw_date), "avg_tone": float(point.get("value", 0.0))})
    if not rows:
        return pd.DataFrame(columns=["date", "avg_tone"]).astype({"date": "datetime64[ns]"})
    frame = pd.DataFrame(rows)
    return frame.groupby("date", as_index=False).mean().sort_values("date")


def load_team_daily(raw_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Returns {team: DataFrame[date, raw_count, monitored_total, avg_tone]},
    one row per calendar date the team has ANY signal for, summed/averaged
    across every relocation-era alias. Every team in
    ``base.TEAM_ARTICLES`` gets an entry (empty frame if nothing parsed_ok),
    matching ``ingest_gdelt_attention.py``'s defensive convention so a
    partial ingest never KeyErrors downstream."""

    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    vol_by_team: dict[str, list[pd.DataFrame]] = {}
    tone_by_team: dict[str, list[pd.DataFrame]] = {}
    n_missing = {"timelinevolraw": 0, "timelinetone": 0}
    for req in manifest["requests"]:
        team = req["team"]
        mode = req["mode"]
        if not req["parsed_ok"]:
            n_missing[mode] = n_missing.get(mode, 0) + 1
            continue
        path = raw_dir / req["file"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload is None:
            n_missing[mode] = n_missing.get(mode, 0) + 1
            continue
        if mode == "timelinevolraw":
            vol_by_team.setdefault(team, []).append(_parse_volraw(payload))
        elif mode == "timelinetone":
            tone_by_team.setdefault(team, []).append(_parse_tone(payload))

    out: dict[str, pd.DataFrame] = {}
    for team in base.TEAM_ARTICLES:
        vol_parts = vol_by_team.get(team, [])
        vol = (
            pd.concat(vol_parts, ignore_index=True).groupby("date", as_index=False).sum()
            if vol_parts
            else pd.DataFrame(columns=["date", "raw_count", "monitored_total"])
        )
        tone_parts = tone_by_team.get(team, [])
        tone = (
            pd.concat(tone_parts, ignore_index=True).groupby("date", as_index=False).mean()
            if tone_parts
            else pd.DataFrame(columns=["date", "avg_tone"])
        )
        merged = vol.merge(tone, on="date", how="outer").sort_values("date").reset_index(drop=True)
        out[team] = merged

    n_teams_with_vol = sum(
        1 for t in out.values() if "raw_count" in t and t["raw_count"].notna().any()
    )
    n_teams_with_tone = sum(
        1 for t in out.values() if "avg_tone" in t and t["avg_tone"].notna().any()
    )
    return out, {
        "manifest": manifest,
        "n_missing_by_mode": n_missing,
        "n_teams_with_volume": n_teams_with_vol,
        "n_teams_with_tone": n_teams_with_tone,
        "n_teams_total": len(base.TEAM_ARTICLES),
    }


# --------------------------------------------------------------------------
# Schedule grid + window aggregation
# --------------------------------------------------------------------------


def _canonical(team: pd.Series) -> pd.Series:
    return team.map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def load_schedule_grid(schedules_path: Path) -> pd.DataFrame:
    """One row per (game, side): season, week, team, gameday, is_home,
    home_cover (for reference only -- not used to compute attention)."""

    schedules = pd.read_parquet(schedules_path)
    games = schedules.loc[
        (schedules["game_type"] == "REG")
        & (schedules["season"] >= SEASON_START)
        & (schedules["season"] <= SEASON_END)
    ].copy()
    games["home_team"] = _canonical(games["home_team"])
    games["away_team"] = _canonical(games["away_team"])
    games["gameday"] = pd.to_datetime(games["gameday"], errors="raise")
    games["spread_line"] = pd.to_numeric(games.get("spread_line"), errors="coerce")
    games = add_ats_outcomes(games)
    games["week"] = games["week"].astype(int)
    games["season"] = games["season"].astype(int)

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
                "home_cover": games["home_cover"],
                "spread_line": games["spread_line"],
            }
        )
        sides.append(side)
    long_df = pd.concat(sides, ignore_index=True)

    weekday = long_df["gameday"].dt.weekday  # Monday=0 ... Sunday=6, Tuesday=1
    tuesday_offset = (weekday - 1) % 7
    tuesday_end = long_df["gameday"] - pd.to_timedelta(tuesday_offset, unit="D")
    long_df["tuesday_window_end"] = tuesday_end
    long_df["tuesday_window_start"] = tuesday_end - pd.Timedelta(days=6)
    saturday_end = tuesday_end + pd.Timedelta(days=4)
    long_df["saturday_window_end"] = saturday_end
    long_df["saturday_window_start"] = saturday_end - pd.Timedelta(days=6)
    long_df["gameday_weekday_name"] = long_df["gameday"].dt.day_name()
    long_df["saturday_cutoff_safe"] = long_df["gameday"] >= long_df["saturday_window_end"]

    return long_df.sort_values(["team", "season", "gameday"]).reset_index(drop=True)


def _window_agg(
    daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[float, float, float]:
    if daily.empty:
        return 0.0, 0.0, float("nan")
    sliced = daily.loc[(daily["date"] >= start) & (daily["date"] <= end)]
    raw_count_s = (
        pd.to_numeric(sliced.get("raw_count"), errors="coerce") if "raw_count" in sliced else None
    )
    monitored_s = (
        pd.to_numeric(sliced.get("monitored_total"), errors="coerce")
        if "monitored_total" in sliced
        else None
    )
    raw_count = float(raw_count_s.sum()) if raw_count_s is not None else 0.0
    monitored_total = float(monitored_s.sum()) if monitored_s is not None else 0.0
    if "avg_tone" in sliced and raw_count_s is not None and raw_count > 0:
        weights = raw_count_s.to_numpy(dtype="float64")
        tones = pd.to_numeric(sliced["avg_tone"], errors="coerce").to_numpy(dtype="float64")
        valid = ~np.isnan(tones) & (weights > 0)
        avg_tone = (
            float(np.average(tones[valid], weights=weights[valid])) if valid.any() else float("nan")
        )
    else:
        avg_tone = float("nan")
    return raw_count, monitored_total, avg_tone


def build_weekly_table(grid: pd.DataFrame, team_daily: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for _, r in grid.iterrows():
        daily = team_daily.get(
            r["team"], pd.DataFrame(columns=["date", "raw_count", "monitored_total", "avg_tone"])
        )
        tue_count, tue_monitored, tue_tone = _window_agg(
            daily, r["tuesday_window_start"], r["tuesday_window_end"]
        )
        sat_count, sat_monitored, sat_tone = _window_agg(
            daily, r["saturday_window_start"], r["saturday_window_end"]
        )
        rows.append(
            {
                "season": r["season"],
                "week": r["week"],
                "team": r["team"],
                "game_id": r["game_id"],
                "is_home": r["is_home"],
                "gameday": r["gameday"],
                "gameday_weekday_name": r["gameday_weekday_name"],
                "saturday_cutoff_safe": r["saturday_cutoff_safe"],
                "tuesday_raw_count": tue_count,
                "tuesday_monitored_total": tue_monitored,
                "tuesday_avg_tone": tue_tone,
                "saturday_raw_count": sat_count,
                "saturday_monitored_total": sat_monitored,
                "saturday_avg_tone": sat_tone,
            }
        )
    table = pd.DataFrame(rows).sort_values(["team", "season", "gameday"]).reset_index(drop=True)

    for cutoff in ("tuesday", "saturday"):
        col = f"{cutoff}_raw_count"
        grouped = table.groupby(["team", "season"], sort=False)[col]
        trailing_mean = grouped.transform(
            lambda s: s.shift(1).rolling(window=TRAILING_WINDOW, min_periods=TRAILING_MIN).mean()
        )
        trailing_std = grouped.transform(
            lambda s: s.shift(1).rolling(window=TRAILING_WINDOW, min_periods=TRAILING_MIN).std()
        )
        has_baseline = trailing_mean.notna() & trailing_std.notna() & (trailing_std > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (table[col] - trailing_mean) / trailing_std
        z[~has_baseline] = np.nan
        table[f"{cutoff}_trailing_mean"] = trailing_mean
        table[f"{cutoff}_trailing_std"] = trailing_std
        table[f"{cutoff}_has_baseline"] = has_baseline
        table[f"{cutoff}_z"] = z

    return table


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdelt-raw", type=Path, required=True)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=REPO / "data/processed/gdelt_weekly_attention.parquet"
    )
    args = parser.parse_args()

    schedules_path = args.schedules or latest_schedules()
    print(f"schedules: {schedules_path}")
    print(f"gdelt raw: {args.gdelt_raw}")

    team_daily, meta = load_team_daily(args.gdelt_raw)
    for team, frame in team_daily.items():
        n = len(frame)
        rc = frame["raw_count"].sum() if "raw_count" in frame and n else 0
        print(f"  {team}: {n} dated rows, sum raw_count={rc:.0f}")

    grid = load_schedule_grid(schedules_path)
    print(f"\nschedule grid rows (team-game sides), REG {SEASON_START}-{SEASON_END}: {len(grid)}")

    table = build_weekly_table(grid, team_daily)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)

    coverage_by_season = (
        table.groupby("season")
        .agg(
            n_team_weeks=("team", "size"),
            tuesday_has_baseline=("tuesday_has_baseline", "sum"),
            saturday_has_baseline=("saturday_has_baseline", "sum"),
            tuesday_nonzero=("tuesday_raw_count", lambda s: int((s > 0).sum())),
            saturday_nonzero=("saturday_raw_count", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    coverage_by_team_season = (
        table.groupby(["team", "season"])
        .agg(
            n_weeks=("week", "size"),
            tuesday_raw_count_sum=("tuesday_raw_count", "sum"),
            tuesday_nonzero_weeks=("tuesday_raw_count", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )

    manifest = {
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gdelt_raw_dir": str(args.gdelt_raw),
        "schedules_path": str(schedules_path),
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "trailing_window_games": TRAILING_WINDOW,
        "trailing_min_games": TRAILING_MIN,
        "n_rows": len(table),
        "n_teams_with_volume": meta["n_teams_with_volume"],
        "n_teams_with_tone": meta["n_teams_with_tone"],
        "n_teams_total": meta["n_teams_total"],
        "ingest_manifest_summary": {
            "n_requests": len(meta["manifest"].get("requests", [])),
            "n_parse_failures_by_mode": meta["n_missing_by_mode"],
            "domain_allowlist": meta["manifest"].get("domain_allowlist"),
        },
        "coverage_by_season": coverage_by_season.to_dict(orient="records"),
        "coverage_by_team_season": coverage_by_team_season.to_dict(orient="records"),
        "columns": list(table.columns),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"\nwrote {args.output} ({len(table)} rows)")
    print(f"wrote {manifest_path}")
    print("\ncoverage by season:")
    print(coverage_by_season.to_string(index=False))
    print("\ncoverage by team-season (tuesday_raw_count_sum):")
    pivot = coverage_by_team_season.pivot(
        index="team", columns="season", values="tuesday_raw_count_sum"
    )
    print(pivot.to_string())


if __name__ == "__main__":
    main()
