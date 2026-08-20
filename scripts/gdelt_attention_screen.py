"""Derive a GDELT-based attention z-score construct (same recipe as the
Wikipedia-pageview pilot), cross-validate it against the Wikipedia
construct on overlapping team-weeks, and screen the SAME predeclared
``both_cold`` cell from ``scripts/attention_battery_screen.py`` as an
independent replication attempt on a second attention source.

Reuses ``scripts/attention_battery_screen.py`` functions by import (not
copy/edit): ``build_team_game_long``, ``attach_game_level``,
``summarize_population``, ``block_bootstrap_two_group``, ``load_games``.

Input: either the legacy raw GDELT ``timelinevol`` pilot written by
``scripts/ingest_gdelt_attention.py`` or the processed ``tuesday_z`` table
written by ``scripts/build_gdelt_weekly_features.py`` from the full
``timelinevolraw`` archive. The latter is the canonical input for the frozen
``gdelt_attention_both_cold`` replication in ``docs/gdelt_backfill.md``.

Output: JSON to ``artifacts/gdelt_attention/<UTC timestamp>/results.json``
with (1) the cross-source correlation and (2) the both_cold replication
cell, both on GDELT, both with full provenance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402


def _load_parent_module():
    spec = importlib.util.spec_from_file_location(
        "attention_battery_screen", REPO / "scripts" / "attention_battery_screen.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_parent_module()


# --------------------------------------------------------------------------
# Parse GDELT raw timelinevol JSON into {team: pd.Series(date -> value)}
# --------------------------------------------------------------------------


def _extract_series_from_timelinevol(payload: dict[str, Any]) -> pd.Series:
    """Defensive parse: GDELT timelinevol responses nest under
    ``timeline -> [ {data: [{date, value}, ...]} ]``. ``date`` strings are
    GDELT-style ``YYYYMMDDThhmmssZ`` or ``YYYYMMDD``; ``value`` is the
    normalized share-of-coverage metric (mode=timelinevol)."""

    timeline = payload.get("timeline")
    if not timeline or not isinstance(timeline, list):
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    data = timeline[0].get("data", [])
    if not data:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    dates = []
    values = []
    for point in data:
        raw_date = str(point.get("date", ""))[:8]
        if len(raw_date) != 8:
            continue
        dates.append(pd.Timestamp(raw_date))
        values.append(float(point.get("value", 0.0)))
    if not dates:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    series = pd.Series(values, index=pd.DatetimeIndex(dates), dtype="float64")
    return series.groupby(series.index).mean().sort_index()


def load_gdelt_team_daily_series(raw_dir: Path) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    """Returns {team: pd.Series}. Every team in ``base.TEAM_ARTICLES`` gets an
    entry -- teams with no successfully-fetched chunk get an EMPTY series
    (not omitted), so ``base.build_team_game_long`` never KeyErrors on a
    team missing from a partial/pilot ingestion; an empty series naturally
    yields ``has_baseline=False`` for every game touching that team, which
    correctly excludes it rather than crashing."""

    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    team_alias_series: dict[str, list[pd.Series]] = {}
    n_missing = 0
    for req in manifest["requests"]:
        if not req["parsed_ok"]:
            n_missing += 1
            continue
        path = raw_dir / req["file"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        series = _extract_series_from_timelinevol(payload)
        team_alias_series.setdefault(req["team"], []).append(series)

    team_views: dict[str, pd.Series] = {}
    for team in base.TEAM_ARTICLES:
        alias_series_list = team_alias_series.get(team, [])
        combined: pd.Series | None = None
        for s in alias_series_list:
            combined = s if combined is None else combined.add(s, fill_value=0.0)
        team_views[team] = (
            combined.sort_index()
            if combined is not None
            else pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
        )

    n_teams_covered = sum(1 for s in team_views.values() if len(s))
    return team_views, {
        "manifest": manifest,
        "n_missing_chunks": n_missing,
        "n_teams_covered": n_teams_covered,
        "n_teams_total": len(base.TEAM_ARTICLES),
    }


def load_gdelt_weekly_long(
    weekly_path: Path, games: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the already-derived Tuesday-only construct for the scored games.

    This function deliberately reads only ``tuesday_z`` and
    ``tuesday_has_baseline``. Saturday columns may coexist in the source table,
    but they cannot enter this frozen Tuesday replication.
    """

    required = {
        "game_id",
        "season",
        "week",
        "team",
        "is_home",
        "tuesday_z",
        "tuesday_has_baseline",
    }
    weekly = pd.read_parquet(weekly_path, columns=sorted(required))
    missing = sorted(required - set(weekly.columns))
    if missing:
        raise ValueError(f"GDELT weekly table missing required columns: {missing}")
    if weekly.duplicated(["game_id", "is_home"]).any():
        raise ValueError("GDELT weekly table has duplicate game_id/is_home rows")

    game_ids = set(games["game_id"].astype(str))
    weekly = weekly.loc[weekly["game_id"].astype(str).isin(game_ids)].copy()
    side_counts = weekly.groupby("game_id")["is_home"].agg(["size", "nunique"])
    bad_games = side_counts.loc[(side_counts["size"] != 2) | (side_counts["nunique"] != 2)]
    if not bad_games.empty:
        raise ValueError(
            f"GDELT weekly table lacks one home/away side for {len(bad_games)} scored games"
        )

    long_df = weekly[
        ["game_id", "season", "week", "team", "is_home", "tuesday_z", "tuesday_has_baseline"]
    ].rename(columns={"tuesday_z": "attention_z", "tuesday_has_baseline": "has_baseline"})
    long_df["has_baseline"] = long_df["has_baseline"].fillna(False).astype(bool)
    long_df.loc[~long_df["has_baseline"], "attention_z"] = pd.NA
    long_df["prior_score_margin"] = pd.NA

    manifest_path = weekly_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing processed GDELT manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return long_df.reset_index(drop=True), {
        "manifest": manifest,
        "n_teams_covered": int(manifest.get("n_teams_with_volume", 0)),
        "n_teams_total": int(manifest.get("n_teams_total", 0)),
    }


# --------------------------------------------------------------------------
# Cross-source correlation
# --------------------------------------------------------------------------


def cross_source_correlation(wiki_long: pd.DataFrame, gdelt_long: pd.DataFrame) -> dict[str, Any]:
    wiki = wiki_long.loc[
        wiki_long["has_baseline"], ["team", "season", "week", "attention_z"]
    ].rename(columns={"attention_z": "wiki_z"})
    gdelt = gdelt_long.loc[
        gdelt_long["has_baseline"], ["team", "season", "week", "attention_z"]
    ].rename(columns={"attention_z": "gdelt_z"})
    merged = wiki.merge(gdelt, on=["team", "season", "week"], how="inner").dropna(
        subset=["wiki_z", "gdelt_z"]
    )
    n = len(merged)
    if n < 3:
        return {
            "n_overlapping_team_weeks": n,
            "correlation": None,
            "note": "too few overlapping rows",
        }
    corr = float(merged["wiki_z"].corr(merged["gdelt_z"]))
    return {
        "n_overlapping_team_weeks": n,
        "correlation": corr,
        "n_teams": int(merged["team"].nunique()),
        "season_range": [int(merged["season"].min()), int(merged["season"].max())],
    }


# --------------------------------------------------------------------------
# both_cold replication on GDELT (identical cell definition to the parent
# battery -- same threshold, same eligibility rule, same sign, same value_col)
# --------------------------------------------------------------------------


def both_cold_cell_gdelt(game_df: pd.DataFrame, *, samples: int, seed: int) -> dict[str, Any]:
    both_baseline = (
        game_df["home_has_baseline"].fillna(False) & game_df["away_has_baseline"].fillna(False)
    ).to_numpy(dtype=bool)
    flag = ((game_df["home_z"] <= -0.5) & (game_df["away_z"] <= -0.5)).to_numpy(dtype=bool)

    full_eligible_df = game_df.loc[both_baseline].copy()
    full_eligible_df["_flag"] = flag[both_baseline]
    full_eligible_df = full_eligible_df.loc[full_eligible_df["home_cover"].notna()].reset_index(
        drop=True
    )
    flag_series = full_eligible_df["_flag"]

    primary = base.summarize_population(
        full_eligible_df,
        flag=flag_series,
        value_col="home_cover",
        block_col="week_block",
        sign=-1,
        samples=samples,
        seed=seed,
    )
    secondary = base.summarize_population(
        full_eligible_df,
        flag=flag_series,
        value_col="home_cover",
        block_col="season",
        sign=-1,
        samples=samples,
        seed=seed,
    )
    return {"week_blocked_primary": primary, "season_blocked_secondary": secondary}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--gdelt-raw", type=Path)
    inputs.add_argument("--gdelt-weekly", type=Path)
    parser.add_argument("--wiki-scratch", type=Path, default=base.DEFAULT_SCRATCH)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=base.BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=base.BOOTSTRAP_SEED)
    args = parser.parse_args()

    schedules_path = args.schedules or base._latest_schedules()
    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "gdelt_attention" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== loading Wikipedia raw from {args.wiki_scratch} ===")
    wiki_views = base.load_team_daily_views(args.wiki_scratch)

    games = base.load_games(schedules_path)
    if args.gdelt_weekly is not None:
        manifest_path = args.gdelt_weekly.with_suffix(".manifest.json")
        weekly_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        season_start = int(weekly_manifest["season_start"])
        season_end = int(weekly_manifest["season_end"])
        games = games.loc[games["season"].between(season_start, season_end)].reset_index(drop=True)
        print(f"=== loading processed GDELT Tuesday features from {args.gdelt_weekly} ===")
        gdelt_long, gdelt_meta = load_gdelt_weekly_long(args.gdelt_weekly, games)
        gdelt_game_df = base.attach_game_level(games, gdelt_long)
        gdelt_input_path = args.gdelt_weekly
        gdelt_input_kind = "timelinevolraw_tuesday_z"
    else:
        assert args.gdelt_raw is not None
        season_start = base.SEASON_START
        season_end = base.SEASON_END
        print(f"=== loading legacy GDELT raw from {args.gdelt_raw} ===")
        gdelt_views, gdelt_meta = load_gdelt_team_daily_series(args.gdelt_raw)
        for team, series in gdelt_views.items():
            if len(series):
                print(
                    f"  {team}: {len(series)} distinct dates, "
                    f"{series.index.min()} - {series.index.max()}"
                )
            else:
                print(f"  {team}: EMPTY series")
        gdelt_long = base.build_team_game_long(games, gdelt_views)
        gdelt_game_df = base.attach_game_level(games, gdelt_long)
        gdelt_input_path = args.gdelt_raw
        gdelt_input_kind = "legacy_timelinevol"

    print(f"\nREG {season_start}-{season_end} games with spread_line: {len(games)}")

    wiki_long = base.build_team_game_long(games, wiki_views)

    corr = cross_source_correlation(wiki_long, gdelt_long)
    print(f"\n=== cross-source correlation (Wikipedia z vs GDELT z) ===\n  {corr}")

    both_cold = both_cold_cell_gdelt(gdelt_game_df, samples=args.samples, seed=args.seed)
    wb = both_cold["week_blocked_primary"]
    print("\n=== both_cold replication on GDELT ===")
    if not wb.get("insufficient_data"):
        print(
            f"  n_flag={wb['n_flag']} n_total={wb['n_total']} "
            f"full_slate_effect={wb['full_slate_effect_pts']:+.4f}pts "
            f"P+={wb['probability_positive']:.4f} "
            f"CI=[{wb['week_blocked_ci95_scaled'][0]:+.4f}, "
            f"{wb['week_blocked_ci95_scaled'][1]:+.4f}]"
        )
    else:
        print("  insufficient data")

    n_gdelt_has_baseline = int(gdelt_long["has_baseline"].sum())
    n_gdelt_no_baseline = int((~gdelt_long["has_baseline"]).sum())

    configuration = {
        "command": "gdelt-attention-screen",
        "schedules": str(schedules_path),
        "gdelt_input": str(gdelt_input_path),
        "gdelt_input_kind": gdelt_input_kind,
        "wiki_scratch": str(args.wiki_scratch),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": season_start,
        "season_end": season_end,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": season_start,
        "season_end": season_end,
        "n_reg_games": len(games),
        "gdelt_n_team_game_rows": len(gdelt_long),
        "gdelt_n_has_baseline": n_gdelt_has_baseline,
        "gdelt_n_no_baseline": n_gdelt_no_baseline,
        "gdelt_input": {"kind": gdelt_input_kind, "path": str(gdelt_input_path)},
        "gdelt_ingest_manifest_summary": {
            "n_requests": gdelt_meta["manifest"].get("n_requests")
            or gdelt_meta["manifest"].get("ingest_manifest_summary", {}).get("n_requests"),
            "n_parse_failures": gdelt_meta["manifest"].get("n_parse_failures")
            or gdelt_meta["manifest"]
            .get("ingest_manifest_summary", {})
            .get("n_parse_failures_by_mode"),
            "start_year": gdelt_meta["manifest"].get("start_year")
            or gdelt_meta["manifest"].get("season_start"),
            "end_year": gdelt_meta["manifest"].get("end_year")
            or gdelt_meta["manifest"].get("season_end"),
            "domain_allowlist": gdelt_meta["manifest"].get("domain_allowlist")
            or gdelt_meta["manifest"].get("ingest_manifest_summary", {}).get("domain_allowlist"),
            "n_teams_covered": gdelt_meta.get("n_teams_covered"),
            "n_teams_total": gdelt_meta.get("n_teams_total"),
        },
        "cross_source_correlation": corr,
        "both_cold_gdelt_replication": both_cold,
        "provenance": artifact_provenance(configuration, gdelt_input_path, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="gdelt-attention-screen",
        metrics=payload,
        notes=(
            "GDELT second-source attention construct: cross-source correlation "
            "vs Wikipedia pageview construct, plus a same-definition replication "
            "of attention_battery_both_cold on GDELT. Measure-only; recording "
            "happens via scripts/record_gdelt_attention.py."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
