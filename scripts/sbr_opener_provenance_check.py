"""MEASURE-ONLY: does SBR's retrospective "Open" column reproduce a
genuinely TIMESTAMPED opener on the games where a timestamped source exists?

This is the decisive empirical test for
``docs/sbr_opener_provenance.md``: SBR's ``open_home_spread``
(``data/processed/sbr_odds.parquet``) is a retrospective assertion with no
capture timestamp, book attribution, or revision history of its own (see
that doc, section 1). Two independently-timestamped sources exist that can
be cross-checked against it:

1. **2020-2021**: the project's own purchased point-in-time market archive
   (``data/market/raw``, ``capture_kind="historical_backfill"``), whose
   ``tue_open`` decision label is a real Tuesday-morning consensus snapshot.
   This arm reruns ``scripts/ingest_sbr_odds.py``'s existing ``opener_check``
   function (imported, not reimplemented) so the join/tolerance logic is
   identical to what ``docs/sbr_odds_archive.md`` already published.
2. **2009-2016**: Wayback Machine captures of vegasinsider.com Las Vegas odds
   boards (``artifacts/vegasinsider_backfill/20260822T033952Z``,
   ``docs/vegasinsider_backfill.md``), each with a genuine archive.org capture
   timestamp. Per-game feature construction (Tuesday/Wednesday pre-kickoff
   captures only, earliest qualifying capture per game, cross-book median
   spread) is reused BY IMPORT from ``scripts/vi_dispersion_screen.py``
   (``load_board_instances`` / ``join_schedule`` / ``game_level_features``),
   unmodified, so this is the same "Wayback-derived opener" construction that
   document already uses for its own scoring, not a fresh definition invented
   here.

Known instrument limitation, carried forward from
``docs/vi_dispersion_screen.md`` (measured there, not re-derived here): the
VI board's ``spread_line`` is the displayed FAVORITE-side quote with no
home/away orientation recorded (97.17% negative regardless of which team is
actually favored). It cannot be compared against SBR's signed
``open_home_spread`` directly. The only honest comparison available is
MAGNITUDE: ``abs(open_home_spread)`` vs ``abs(median_spread)``. This script
never attempts to recover side/orientation from the VI board.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/sbr_opener_provenance_check.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _common import bootstrap_pearson_ci, default_schedules  # noqa: E402
from ingest_sbr_odds import opener_check  # noqa: E402
from vi_dispersion_screen import (  # noqa: E402
    DEFAULT_BACKFILL,
    game_level_features,
    join_schedule,
    load_board_instances,
)

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260826
VI_SEASON_START = 2009
VI_SEASON_END = 2016

# Measured this session (not previously documented at this resolution):
# capture 2009-12-16 09:52:59 (VI's Week 15 2009 board) has a spread/total
# token-parsing defect -- named books agree TIGHTLY with each other (cross-
# book range <1.0pt, so it passes vi_dispersion_screen.py's own
# spread_range>10 parse-artifact cap) but on a number in the 37-54pt range,
# which is a TOTAL, not a spread (no real NFL spread is ever that large).
# docs/vi_dispersion_screen.md's predeclaration already flagged "2 instances"
# from this exact capture via its range cap; this check found 7 of that
# capture's games carry the same defect by magnitude, all missed by the
# range cap because the mislabeling is consistent across books, not
# disagreeing. Excluded by capture_ts (a targeted, documented exclusion),
# NOT by a blanket magnitude cap: a genuinely large legitimate spread exists
# in this same window (2013_06_JAX_DEN, VI median 27.5 vs SBR open 24.0/
# close 26.5 -- close agreement, a real historically-lopsided game, not an
# artifact) that a magnitude cap would have wrongly discarded.
KNOWN_DEFECTIVE_VI_CAPTURES = frozenset({"20091216095259"})


def _diff_stats(diff: pd.Series) -> dict[str, float]:
    """Summarize an abs-diff series, dropping NaNs (missing VI median_spread
    when a matched capture had zero named-book spread quotes) rather than
    letting pandas' elementwise comparisons silently count them as
    out-of-tolerance."""

    clean = diff.dropna()
    n = len(clean)
    if n == 0:
        return {
            "n": 0,
            "mean_abs_diff": float("nan"),
            "median_abs_diff": float("nan"),
            "share_within_0.5pt": float("nan"),
            "share_within_1.0pt": float("nan"),
            "share_exact": float("nan"),
        }
    return {
        "n": n,
        "mean_abs_diff": float(clean.mean()),
        "median_abs_diff": float(clean.median()),
        "share_within_0.5pt": float((clean <= 0.5).mean()),
        "share_within_1.0pt": float((clean <= 1.0).mean()),
        "share_exact": float((clean == 0.0).mean()),
    }


def sbr_vs_tue_open_2020_2021(
    sbr: pd.DataFrame, market_root: Path, game_features_path: Path
) -> dict[str, Any]:
    """Arm 1: re-run the existing SBR-vs-timestamped-archive opener check."""

    return opener_check(sbr, market_root, game_features_path)


def build_vi_wayback_openers(schedules_path: Path) -> tuple[pd.DataFrame, int]:
    """Earliest Tuesday/Wednesday pre-kickoff Wayback capture per matched
    game, 2005-2016 boards (reused unmodified from vi_dispersion_screen.py).

    Returns ``(openers, n_defective_capture_dropped)``.
    """

    instances = load_board_instances(DEFAULT_BACKFILL)
    instances = join_schedule(instances, schedules_path)
    games = game_level_features(instances)
    games = games.loc[games["game_key"].notna()].copy()
    n_defective = int(games["capture_ts"].isin(KNOWN_DEFECTIVE_VI_CAPTURES).sum())
    games = games.loc[~games["capture_ts"].isin(KNOWN_DEFECTIVE_VI_CAPTURES)].copy()
    games["game_id"] = games["game_key"].str.split(":", n=1).str[1]
    games["vi_capture_dt"] = pd.to_datetime(games["capture_dt"])
    games["vi_capture_weekday"] = games["vi_capture_dt"].dt.day_name()
    games["days_before_kickoff"] = (
        pd.to_datetime(games["game_day"]) - games["vi_capture_dt"].dt.normalize()
    ).dt.days
    openers = games[
        [
            "game_id",
            "vi_capture_dt",
            "vi_capture_weekday",
            "days_before_kickoff",
            "median_spread",
            "n_books_spread",
        ]
    ]
    return openers, n_defective


def sbr_vs_vi_wayback_2009_2016(sbr: pd.DataFrame, schedules_path: Path) -> dict[str, Any]:
    """Arm 2: SBR Open magnitude vs the VI Wayback-captured board median
    magnitude, on REG games 2009-2016 (the VI backfill's scoreable window,
    per docs/vegasinsider_backfill.md: local schedules start 2009, board
    archive ends 2016)."""

    vi_openers, n_defective = build_vi_wayback_openers(schedules_path)

    sbr_pop = sbr.loc[
        sbr["season"].between(VI_SEASON_START, VI_SEASON_END) & sbr["game_id"].notna()
    ].copy()
    sbr_pop["game_id"] = sbr_pop["game_id"].astype(str)
    vi_openers["game_id"] = vi_openers["game_id"].astype(str)

    joined = sbr_pop.merge(vi_openers, on="game_id", how="inner")
    joined["sbr_open_abs"] = joined["open_home_spread"].abs()
    joined["vi_median_abs"] = joined["median_spread"].abs()
    joined["diff"] = (joined["sbr_open_abs"] - joined["vi_median_abs"]).abs()

    clean_pair = joined.dropna(subset=["sbr_open_abs", "vi_median_abs"])
    overall = _diff_stats(joined["diff"])
    x = clean_pair["sbr_open_abs"].to_numpy(dtype=float)
    y = clean_pair["vi_median_abs"].to_numpy(dtype=float)
    pearson_r = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")
    pearson_ci95 = (
        bootstrap_pearson_ci(x, y, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
        if len(x) >= 2
        else [float("nan"), float("nan")]
    )
    overall["pearson_r"] = pearson_r
    overall["pearson_r_ci95"] = pearson_ci95
    overall["signed_diff_mean"] = float(
        (clean_pair["sbr_open_abs"] - clean_pair["vi_median_abs"]).mean()
    )
    overall["signed_diff_std"] = float(
        (clean_pair["sbr_open_abs"] - clean_pair["vi_median_abs"]).std()
    )

    # Robustness subset: >=3 named books feeding the VI median (the same
    # MIN_BOOKS gate vi_dispersion_screen.py uses for its own scoring frame),
    # since a 1- or 2-book "median" is a much noisier point estimate.
    robust = joined.loc[joined["n_books_spread"] >= 3]
    robust_stats = _diff_stats(robust["diff"]) if len(robust) else {"n": 0}
    robust_clean = robust.dropna(subset=["sbr_open_abs", "vi_median_abs"])
    if len(robust_clean) >= 2:
        rx = robust_clean["sbr_open_abs"].to_numpy(dtype=float)
        ry = robust_clean["vi_median_abs"].to_numpy(dtype=float)
        robust_stats["pearson_r"] = float(np.corrcoef(rx, ry)[0, 1])

    per_season = []
    for season, group in joined.groupby("season"):
        n_sbr_season = int((sbr_pop["season"] == season).sum())
        stats = _diff_stats(group["diff"])
        stats["season"] = int(season)
        stats["sbr_games_in_season"] = n_sbr_season
        stats["match_rate_of_sbr"] = len(group) / n_sbr_season if n_sbr_season else float("nan")
        group_clean = group.dropna(subset=["sbr_open_abs", "vi_median_abs"])
        if len(group_clean) >= 2:
            gx = group_clean["sbr_open_abs"].to_numpy(dtype=float)
            gy = group_clean["vi_median_abs"].to_numpy(dtype=float)
            stats["pearson_r"] = float(np.corrcoef(gx, gy)[0, 1])
        per_season.append(stats)

    capture_weekday_counts = joined["vi_capture_weekday"].value_counts().to_dict()
    days_before_kickoff_summary = {
        "mean": float(joined["days_before_kickoff"].mean()),
        "median": float(joined["days_before_kickoff"].median()),
        "min": float(joined["days_before_kickoff"].min()),
        "max": float(joined["days_before_kickoff"].max()),
    }

    return {
        "window": [VI_SEASON_START, VI_SEASON_END],
        "sbr_population_games": len(sbr_pop),
        "vi_wayback_openers_2005_2016": len(vi_openers),
        "n_defective_capture_dropped": n_defective,
        "matched_games": len(joined),
        "overall": overall,
        "robust_ge3_books": robust_stats,
        "per_season": sorted(per_season, key=lambda r: r["season"]),
        "vi_capture_weekday_counts": capture_weekday_counts,
        "vi_days_before_kickoff": days_before_kickoff_summary,
        "note": (
            "Magnitude-only comparison: the VI Wayback board cannot encode "
            "which team (home/away) its spread favors (measured in "
            "docs/vi_dispersion_screen.md, 97.17% negative regardless of "
            "actual side), so this compares abs(SBR open_home_spread) "
            "against abs(VI board median spread) on the earliest Tuesday/"
            "Wednesday pre-kickoff Wayback capture per matched game. It "
            "does not verify SIDE agreement, only magnitude agreement. "
            f"{n_defective} game(s) excluded from one known-defective "
            "capture (20091216095259, a spread/total token-parsing bug -- "
            "see KNOWN_DEFECTIVE_VI_CAPTURES in this script)."
        ),
    }


def main() -> None:
    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = REPO / "artifacts" / "sbr_opener_provenance" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    sbr_path = REPO / "data/processed/sbr_odds.parquet"
    game_features_path = REPO / "data/processed/game_features.parquet"
    market_root = REPO / "data/market/raw"
    schedules_path = default_schedules()

    sbr = pd.read_parquet(sbr_path)

    print("=== Arm 1: SBR Open vs timestamped purchased-archive tue_open (2020-2021) ===")
    arm1 = sbr_vs_tue_open_2020_2021(sbr, market_root, game_features_path)
    print(arm1)

    print("\n=== Arm 2: SBR Open vs Wayback VI board median, magnitude only (2009-2016) ===")
    arm2 = sbr_vs_vi_wayback_2009_2016(sbr, schedules_path)
    print(arm2)

    configuration = {
        "command": "sbr-opener-provenance-check",
        "sbr_odds": str(sbr_path),
        "game_features": str(game_features_path),
        "market_root": str(market_root),
        "schedules": str(schedules_path),
        "vi_backfill": str(DEFAULT_BACKFILL),
        "vi_season_start": VI_SEASON_START,
        "vi_season_end": VI_SEASON_END,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "arm1_sbr_vs_tue_open_2020_2021": arm1,
        "arm2_sbr_vs_vi_wayback_2009_2016": arm2,
        "provenance": artifact_provenance(configuration, sbr_path, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="sbr-opener-provenance-check",
        metrics=payload,
        notes=(
            "Measure-only provenance audit for docs/sbr_opener_provenance.md. "
            "Arm 1 reruns scripts/ingest_sbr_odds.py's opener_check (import, "
            "unmodified) against the purchased tue_open archive, 2020-2021 "
            "overlap. Arm 2 is new: SBR Open magnitude vs a Wayback-timestamped "
            "VegasInsider board's earliest Tuesday/Wednesday pre-kickoff "
            "capture, 2009-2016, magnitude-only (VI board carries no home/"
            "away orientation, per docs/vi_dispersion_screen.md). No model "
            "was scored, no evaluation window changed, no registry write."
        ),
        source="scripts/sbr_opener_provenance_check.py",
        project_root=REPO,
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
