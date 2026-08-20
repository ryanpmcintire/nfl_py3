"""`docs/pfr_transactions_sourcing.md` section 6's two PREDECLARED experiments,
run 2026-08-20 against the already-ingested PFR transaction-wire archive
(`data/raw/pfr_transactions/20260820T011126Z`) and PFT injury-news archive
(`data/raw/injury_news/20260819T191639Z`), plus this session's own bulk
per-article JSON-LD ``datePublished`` fetch (`scripts/pfr_bulk_date_fetch.py`,
cached into the PFR snapshot's `sample_articles/` directory).

**Owner timing correction, 2026-08-20** (postdates the predeclaration in
`docs/pfr_transactions_sourcing.md`, which framed both questions around a
"Tuesday-noon lock"): only the pool's LINE freezes Tuesday noon ET; PICKS
stay editable up to each game's own deadline. This script therefore runs
each question at TWO cutoffs per game:

* ``tuesday_noon_utc`` -- the original framing (own-week Tuesday noon ET,
  `nfl_ats.clv.live_tuesday_openers` / `injury_tuesday_cutoff_experiment.py`
  convention), kept for comparability with the already-measured PFT-only
  0.43%/8.13%/2.39% figures in `docs/injury_news_sourcing.md` sec 5.1.
* ``saturday_refresh_utc`` -- ``min(kickoff, that game's own-week Saturday
  noon ET)`` -- the operationally playable "late-week refresh" cutoff. The
  ``min`` clamp matters for Thursday-night games, whose own-week Saturday
  falls AFTER kickoff (NFL week runs Tue-Mon: Thu precedes Sat), so the
  refresh cutoff for those games collapses to kickoff itself, same as any
  other genuinely-pregame bound.

Question 1 (PFR-vs-PFT additivity): for each (season, week, team) in the
REG-season ROSTER population (any roster status -- PFR transactions are
about cuts/IR/trades/signings, not just the ACT subset), does a matching PFT
headline exist for every matching PFR headline? Population: rosters table,
seasons 2022-2025 (injuries table caps at 2024 -- see Question 2's narrower
scope note). Reports the PFR-only fraction two ways: of PFR's own matches,
and of the PFR-union-PFT total.

Question 2 (foreshadowing official state): reruns
`scripts/injury_tuesday_cutoff_experiment.py`'s "coverage" construction (the
one that produced the recorded 0.43% official-only / 8.13% PFT-augmented
numbers) with PFR substituted for, and pooled with, PFT. Scope note: the
project's ingested injuries table caps at season 2024 (measured this
session -- neither on-disk player snapshot has 2025 injury rows yet), so
Question 2 covers seasons 2022-2024, one season narrower than Question 1's
2022-2025 (rosters go through 2025).

Matching rule (unchanged from `injury_tuesday_cutoff_experiment.py`'s
``build_pft_match_table`` / `docs/qb_news_channel.md`'s PFR extension of it):
conservative, precision-favoring literal full-normalized-name substring
match against the headline text, 9-day lookback (matching
``injury_tuesday_cutoff_experiment.py``'s own ``--lookback-days`` default),
undercounts last-name-only headlines -- every coverage number below is a
LOWER BOUND on true visibility.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/pfr_pft_additivity_experiment.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.players import load_player_snapshot, player_snapshot_from_root
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact

REPO = Path(__file__).resolve().parents[1]

FEATURES_PATH = REPO / "data/processed/game_features_pbp.parquet"
PFT_INDEX = REPO / "data/raw/injury_news/20260819T191639Z/index.parquet"
PFR_SNAPSHOT = REPO / "data/raw/pfr_transactions/20260820T011126Z"
PFR_INDEX = PFR_SNAPSHOT / "index.parquet"
PFR_SAMPLE_DIR = PFR_SNAPSHOT / "sample_articles"
PLAYER_SNAPSHOT_ID = "20260817T184901Z"

Q1_SEASONS = (2022, 2025)
Q2_SEASONS = (2022, 2024)  # injuries table cap, measured this session
LOOKBACK_DAYS = 9.0  # matches injury_tuesday_cutoff_experiment.py's default

_CURLY_APOSTROPHE = chr(0x2019)
NAME_PUNCTUATION = re.compile("[." + "'" + _CURLY_APOSTROPHE + "]")
NAME_NONALNUM = re.compile(r"[^a-z0-9]+")


def _normalize_name(name: object) -> str:
    text = str(name).lower()
    text = NAME_PUNCTUATION.sub("", text)
    text = NAME_NONALNUM.sub(" ", text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Cutoffs: own-week Tuesday noon ET, own-week Saturday-refresh (owner
# correction) capped at kickoff.
# ---------------------------------------------------------------------------


def team_week_cutoffs(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ("home_team", "away_team"):
        rows.append(
            games[["season", "week", "game_id", "kickoff", side]].rename(columns={side: "team"})
        )
    long = pd.concat(rows, ignore_index=True)
    kickoff_et = long["kickoff"].dt.tz_convert("US/Eastern")
    days_since_tuesday = (kickoff_et.dt.weekday - 1) % 7
    tuesday_date_et = kickoff_et.dt.normalize() - pd.to_timedelta(days_since_tuesday, unit="D")
    tuesday_noon_et = tuesday_date_et + pd.Timedelta(hours=12)
    saturday_noon_et = tuesday_date_et + pd.Timedelta(days=4, hours=12)

    long["tuesday_noon_utc"] = tuesday_noon_et.dt.tz_convert("UTC")
    saturday_noon_utc = saturday_noon_et.dt.tz_convert("UTC")
    long["kickoff_utc"] = long["kickoff"]
    # Owner-corrected 2026-08-20 refresh cutoff: min(kickoff, own-week Saturday
    # noon ET). For Thursday games own-week Saturday is AFTER kickoff, so this
    # collapses to kickoff -- still a genuinely pregame bound, just a tighter one.
    long["saturday_refresh_utc"] = long[["kickoff_utc"]].assign(sat=saturday_noon_utc).min(axis=1)
    return long[
        [
            "season",
            "week",
            "team",
            "game_id",
            "kickoff_utc",
            "tuesday_noon_utc",
            "saturday_refresh_utc",
        ]
    ]


# ---------------------------------------------------------------------------
# Generic headline matcher: earliest match timestamp within
# [cutoff - lookback, cutoff], literal full-name substring, last-name inverted index.
# ---------------------------------------------------------------------------


def build_last_name_index(headlines_norm: np.ndarray) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for pos, text in enumerate(headlines_norm):
        for token in set(str(text).split()):
            if len(token) >= 3:
                index.setdefault(token, []).append(pos)
    return index


def earliest_match(
    names: np.ndarray,
    cutoffs_utc: pd.Series,
    *,
    headlines_norm: np.ndarray,
    timestamps_utc: np.ndarray,
    lookback_days: float,
) -> np.ndarray:
    """Earliest ``timestamps_utc`` entry whose headline contains ``names[i]``
    and falls in ``[cutoffs_utc[i] - lookback_days, cutoffs_utc[i]]``, per row.
    Returns a ``datetime64[ns]`` (naive UTC) array, NaT where no match."""

    last_name_index = build_last_name_index(headlines_norm)
    cutoffs_naive = (
        cutoffs_utc.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    )
    lookback = np.timedelta64(int(lookback_days * 86400), "s")

    out = np.full(len(names), np.datetime64("NaT"), dtype="datetime64[ns]")
    for row_pos in range(len(names)):
        name = names[row_pos]
        if not name:
            continue
        cutoff = cutoffs_naive[row_pos]
        if np.isnat(cutoff):
            continue
        window_start = cutoff - lookback
        last_token = name.split()[-1]
        candidates = last_name_index.get(last_token)
        if not candidates:
            continue
        best = None
        for idx in candidates:
            if name not in headlines_norm[idx]:
                continue
            ts = timestamps_utc[idx]
            if np.isnat(ts) or ts < window_start or ts > cutoff:
                continue
            if best is None or ts < best:
                best = ts
        if best is not None:
            out[row_pos] = best
    return out


def load_pft_relevant() -> tuple[np.ndarray, np.ndarray]:
    pft = pd.read_parquet(PFT_INDEX)
    relevant = pft.loc[pft["injury_relevant"]].copy()
    headlines_norm = relevant["headline_guess"].map(_normalize_name).to_numpy()
    timestamps = (
        relevant["lastmod"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    return headlines_norm, timestamps


def load_pfr_fetched() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Only rows with a successfully fetched JSON-LD datePublished -- month-only
    proxy is NOT sufficient for day/hour cutoff matching, per
    docs/pfr_transactions_sourcing.md sec 1/6."""

    pfr = pd.read_parquet(PFR_INDEX)
    relevant = pfr.loc[pfr["transaction_relevant"]].copy()

    cache_files = sorted(PFR_SAMPLE_DIR.glob("*.json"))
    cache: dict[str, str | None] = {}
    n_cache_fetch_failed = 0
    n_cache_no_date = 0
    for path in cache_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        slug = payload.get("slug")
        if slug is None:
            continue
        dates = payload.get("json_ld_date_published") or []
        if payload.get("fetch_failed"):
            n_cache_fetch_failed += 1
            cache[slug] = None
        elif dates:
            cache[slug] = dates[0]
        else:
            n_cache_no_date += 1
            cache[slug] = None

    relevant["cached_date_published"] = relevant["slug"].map(cache)
    dated = relevant.loc[relevant["cached_date_published"].notna()].copy()
    dated["precise_ts"] = pd.to_datetime(dated["cached_date_published"], errors="coerce", utc=True)
    dated = dated.loc[dated["precise_ts"].notna()]

    headlines_norm = dated["headline_from_slug"].map(_normalize_name).to_numpy()
    timestamps = (
        dated["precise_ts"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    coverage = {
        "transaction_relevant_total": len(relevant),
        "cache_files_on_disk": len(cache_files),
        "cache_fetch_failed": n_cache_fetch_failed,
        "cache_no_date_extracted": n_cache_no_date,
        "transaction_relevant_with_precise_date": len(dated),
    }
    return headlines_norm, timestamps, coverage


def build_name_map(rosters: pd.DataFrame) -> dict[str, str]:
    counts = (
        rosters.dropna(subset=["gsis_id", "full_name"])
        .groupby(["gsis_id", "full_name"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates("gsis_id")
    )
    return {
        str(row.gsis_id): _normalize_name(str(row.full_name))
        for row in counts.itertuples(index=False)
    }


def main() -> None:
    print("Loading games / cutoffs...")
    games = pd.read_parquet(FEATURES_PATH)
    games = games.loc[games["game_type"].eq("REG")]
    cutoffs = team_week_cutoffs(games)

    print("Loading player snapshot (rosters + injuries)...")
    snap = player_snapshot_from_root(REPO / "data/players/raw" / PLAYER_SNAPSHOT_ID)
    injuries, rosters, _snaps = load_player_snapshot(snap)
    rosters = rosters.loc[rosters["game_type"].eq("REG")]
    name_map = build_name_map(rosters)

    print("Loading PFT (injury_relevant subset, local lastmod)...")
    pft_headlines, pft_timestamps = load_pft_relevant()
    print("Loading PFR (transaction_relevant subset, cached precise dates)...")
    pfr_headlines, pfr_timestamps, pfr_coverage = load_pfr_fetched()
    print(f"  PFR precise-date coverage: {json.dumps(pfr_coverage)}")

    results: dict[str, Any] = {
        "pfr_fetch_coverage": pfr_coverage,
        "lookback_days": LOOKBACK_DAYS,
        "q1_seasons": list(Q1_SEASONS),
        "q2_seasons": list(Q2_SEASONS),
    }

    # -----------------------------------------------------------------
    # Question 1: PFR-vs-PFT additivity, roster population, seasons 2022-2025
    # -----------------------------------------------------------------
    print("\n=== Question 1: PFR-vs-PFT additivity (roster population) ===")
    q1_rosters = rosters.loc[rosters["season"].between(*Q1_SEASONS) & rosters["week"].le(18)]
    q1_rosters = q1_rosters.drop_duplicates(["season", "week", "team", "gsis_id"])
    q1_pop = q1_rosters.merge(cutoffs, on=["season", "week", "team"], how="inner")
    q1_pop["player_name"] = q1_pop["gsis_id"].map(name_map)
    q1_pop = q1_pop.loc[q1_pop["player_name"].notna()].reset_index(drop=True)
    print(f"Q1 population rows (season-week-team-player, weeks<=18): {len(q1_pop)}")

    q1_results: dict[str, Any] = {}
    for cutoff_label, cutoff_col in (
        ("tuesday_noon", "tuesday_noon_utc"),
        ("saturday_refresh", "saturday_refresh_utc"),
    ):
        names = q1_pop["player_name"].to_numpy(dtype=object)
        cutoff_series = q1_pop[cutoff_col]
        pft_match = earliest_match(
            names,
            cutoff_series,
            headlines_norm=pft_headlines,
            timestamps_utc=pft_timestamps,
            lookback_days=LOOKBACK_DAYS,
        )
        pfr_match = earliest_match(
            names,
            cutoff_series,
            headlines_norm=pfr_headlines,
            timestamps_utc=pfr_timestamps,
            lookback_days=LOOKBACK_DAYS,
        )
        has_pft = ~pd.isna(pft_match)
        has_pfr = ~pd.isna(pfr_match)
        pfr_only = has_pfr & ~has_pft
        pft_only = has_pft & ~has_pfr
        both = has_pfr & has_pft
        union = has_pfr | has_pft

        n_pfr = int(has_pfr.sum())
        n_pft = int(has_pft.sum())
        n_union = int(union.sum())
        n_pfr_only = int(pfr_only.sum())

        per_season = {}
        for season in range(Q1_SEASONS[0], Q1_SEASONS[1] + 1):
            season_mask = (q1_pop["season"] == season).to_numpy()
            s_pfr = int((has_pfr & season_mask).sum())
            s_pfr_only = int((pfr_only & season_mask).sum())
            per_season[str(season)] = {
                "pfr_matched": s_pfr,
                "pfr_only": s_pfr_only,
                "pfr_only_fraction_of_pfr": (s_pfr_only / s_pfr) if s_pfr else None,
            }

        q1_results[cutoff_label] = {
            "population_rows": len(q1_pop),
            "pfr_matched_rows": n_pfr,
            "pft_matched_rows": n_pft,
            "both_matched_rows": int(both.sum()),
            "union_matched_rows": n_union,
            "pfr_only_rows": n_pfr_only,
            "pft_only_rows": int(pft_only.sum()),
            "pfr_only_fraction_of_pfr_matches": (n_pfr_only / n_pfr) if n_pfr else None,
            "pfr_only_fraction_of_union": (n_pfr_only / n_union) if n_union else None,
            "per_season": per_season,
        }
        print(
            f"[{cutoff_label}] PFR matched={n_pfr} PFT matched={n_pft} union={n_union} "
            f"PFR-only={n_pfr_only} "
            f"(frac of PFR={q1_results[cutoff_label]['pfr_only_fraction_of_pfr_matches']}, "
            f"frac of union={q1_results[cutoff_label]['pfr_only_fraction_of_union']})"
        )

    results["question_1_pfr_vs_pft_additivity"] = q1_results

    # -----------------------------------------------------------------
    # Question 2: foreshadowing official state, injuries population,
    # seasons 2022-2024 (injuries-table cap)
    # -----------------------------------------------------------------
    print("\n=== Question 2: foreshadowing official state (injuries population) ===")
    q2_injuries = injuries.loc[injuries["season"].between(*Q2_SEASONS) & injuries["week"].le(18)]
    q2_pop = q2_injuries.merge(cutoffs, on=["season", "week", "team"], how="inner")
    q2_pop["player_name"] = q2_pop["gsis_id"].map(name_map)
    print(f"Q2 population rows (official injury-report rows): {len(q2_pop)}")

    q2_results: dict[str, Any] = {}
    for cutoff_label, cutoff_col in (
        ("tuesday_noon", "tuesday_noon_utc"),
        ("saturday_refresh", "saturday_refresh_utc"),
    ):
        cutoff_series = q2_pop[cutoff_col]
        official_visible = q2_pop["date_modified"] <= cutoff_series

        names = q2_pop["player_name"].fillna("").to_numpy(dtype=object)
        pft_match = earliest_match(
            names,
            cutoff_series,
            headlines_norm=pft_headlines,
            timestamps_utc=pft_timestamps,
            lookback_days=LOOKBACK_DAYS,
        )
        pfr_match = earliest_match(
            names,
            cutoff_series,
            headlines_norm=pfr_headlines,
            timestamps_utc=pfr_timestamps,
            lookback_days=LOOKBACK_DAYS,
        )
        has_pft = ~pd.isna(pft_match)
        has_pfr = ~pd.isna(pfr_match)

        pft_augmented = official_visible | has_pft
        pfr_augmented = official_visible | has_pfr
        pooled = official_visible | has_pft | has_pfr

        n_total = len(q2_pop)
        q2_results[cutoff_label] = {
            "official_injury_rows_total": n_total,
            "official_only_visible_share": float(official_visible.mean()),
            "pft_augmented_visible_share": float(pft_augmented.mean()),
            "pfr_augmented_visible_share": float(pfr_augmented.mean()),
            "pooled_pft_pfr_visible_share": float(pooled.mean()),
            "additional_from_pft_over_official": int((pft_augmented & ~official_visible).sum()),
            "additional_from_pfr_over_official": int((pfr_augmented & ~official_visible).sum()),
            "additional_from_pooling_over_pft_alone": int((pooled & ~pft_augmented).sum()),
        }
        print(f"[{cutoff_label}] {json.dumps(q2_results[cutoff_label], indent=2)}")

    results["question_2_foreshadowing_official_state"] = q2_results

    out_dir = REPO / "artifacts/pfr_pft_additivity"
    configuration = {
        "command": "pfr-pft-additivity-experiment",
        "features": str(FEATURES_PATH),
        "pft_index": str(PFT_INDEX),
        "pfr_index": str(PFR_INDEX),
        "player_snapshot": PLAYER_SNAPSHOT_ID,
        "lookback_days": LOOKBACK_DAYS,
    }
    results["provenance"] = artifact_provenance(configuration, FEATURES_PATH, project_root=REPO)
    write_experiment_artifact(
        out_dir,
        "result.json",
        results,
        command="pfr-pft-additivity-experiment",
        metrics=results,
        notes=(
            "docs/pfr_transactions_sourcing.md sec 6, both predeclared questions, run at "
            "both Tuesday-noon and Saturday-refresh cutoffs per the 2026-08-20 owner "
            "timing correction (picks editable to kickoff, not locked at Tuesday noon)."
        ),
    )
    print(f"\nwrote {out_dir / 'result.json'}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
