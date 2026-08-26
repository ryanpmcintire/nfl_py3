"""Transaction-wire battery: 7 predeclared cells against the spread on NFL
REG games, week-blocked bootstrap (season-blocked secondary), full-slate
scaled, seeded and deterministic.

**Predeclaration**: ``docs/transaction_wire_battery.md``, written and frozen
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
the separate ``scripts/transaction_wire_battery_record.py``.

**Season completeness rule (``docs/transaction_wire_battery.md`` section 1,
read it, predeclared before any score was seen)**: a season is included in
the scored population if and only if 100% of its ``transaction_relevant``
rows with ``url_year``/``url_month`` in that season's Aug(Y)-Jan(Y+1) target
window have a successfully cached, non-null per-article JSON-LD
``datePublished`` at run time. A partially-fetched season is excluded
WHOLESALE, never scored on whatever subset happened to be fetched first --
see the module docstring of ``scripts/pfr_bulk_date_fetch.py`` for why a
partial season would otherwise silently undercount team-weeks that simply
have not been fetched yet.

Writes JSON to ``artifacts/transaction_wire_battery/<UTC timestamp>/results.json``
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

from _common import block_bootstrap_two_group  # noqa: E402
from pfr_bulk_date_fetch import target_year_months  # noqa: E402

from nfl_ats.cfb_qb_dependence import split_half_reliability  # noqa: E402
from nfl_ats.experiment_runner import scale_subset_effect  # noqa: E402
from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.transaction_wire_features import (  # noqa: E402
    TRANSACTION_CATEGORIES,
    attach_transaction_counts,
    build_team_week_population,
    classify_transaction_slug,
    explode_dated_transactions,
    match_transaction_teams,
)

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260826
SEASON_START = 2014  # PFR true article coverage floor (docs/pfr_transactions_sourcing.md sec 1)
SEASON_END = 2025  # excludes the partial in-progress 2026 season

DEFAULT_PFR_SNAPSHOT = REPO / "data/raw/pfr_transactions/20260820T011126Z"


def _latest(glob_pattern: str, label: str) -> Path:
    candidates = sorted(REPO.glob(glob_pattern))
    if not candidates:
        raise FileNotFoundError(f"no {label} found matching {glob_pattern!r}")
    return candidates[-1]


def default_schedules() -> Path:
    """Resolve lazily so importing this module never requires local data."""
    return _latest("data/raw/*/schedules.parquet", "schedules.parquet snapshot")


# ---------------------------------------------------------------------------
# 1. PFR loading: coverage report (section 2) + dated rows (section 1)
# ---------------------------------------------------------------------------


def load_pfr_index(snapshot_dir: Path) -> pd.DataFrame:
    frame = pd.read_parquet(snapshot_dir / "index.parquet")
    frame["url_year"] = pd.to_numeric(frame["url_year"], errors="coerce")
    frame["url_month"] = pd.to_numeric(frame["url_month"], errors="coerce")
    return frame


def load_dated_cache(snapshot_dir: Path) -> dict[str, str | None]:
    """slug -> JSON-LD ``datePublished`` string, or ``None`` if the cache
    entry exists but has no usable date (fetch failure or no date found)."""

    sample_dir = snapshot_dir / "sample_articles"
    cache: dict[str, str | None] = {}
    for path in sample_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        slug = payload.get("slug")
        if slug is None:
            continue
        if payload.get("fetch_failed"):
            cache[slug] = None
            continue
        dates = payload.get("json_ld_date_published") or []
        cache[slug] = dates[0] if dates else None
    return cache


def season_completeness_report(
    pfr_index: pd.DataFrame, cache: dict[str, str | None], *, seasons: range
) -> dict[int, dict[str, Any]]:
    """Per-season target-scope size, dated count, and completeness flag,
    per ``docs/transaction_wire_battery.md`` section 1's predeclared rule."""

    relevant = pfr_index.loc[pfr_index["transaction_relevant"]].copy()
    relevant["has_date"] = relevant["slug"].map(cache).notna()
    # Precomputed once (vectorized), not inside the per-season loop below --
    # NaN url_year/url_month rows produce a (nan, nan) tuple that matches no
    # season's target-months set, the same "never in scope" outcome the
    # previous per-row notna() check produced.
    year_month = list(zip(relevant["url_year"], relevant["url_month"], strict=True))

    report: dict[int, dict[str, Any]] = {}
    for season in seasons:
        months = target_year_months((season,))
        in_scope = pd.Series(year_month, index=relevant.index).isin(months)
        scoped = relevant.loc[in_scope]
        n_target = len(scoped)
        n_dated = int(scoped["has_date"].sum())
        report[season] = {
            "n_target_rows": n_target,
            "n_dated_rows": n_dated,
            "complete": bool(n_target > 0 and n_dated == n_target),
        }
    return report


def build_dated_transactions(pfr_index: pd.DataFrame, cache: dict[str, str | None]) -> pd.DataFrame:
    """Every ``transaction_relevant`` row with a successfully cached JSON-LD
    date, as ``(slug, precise_ts)`` -- the section-1 timestamp basis this
    whole battery depends on. Season completeness is enforced downstream by
    restricting the SCORED team-week population, not by filtering this
    table (a dated row from an excluded season simply never falls inside any
    included season's team-week window)."""

    relevant = pfr_index.loc[pfr_index["transaction_relevant"]].copy()
    relevant["cached_date"] = relevant["slug"].map(cache)
    dated = relevant.loc[relevant["cached_date"].notna()].copy()
    dated["precise_ts"] = pd.to_datetime(dated["cached_date"], errors="coerce", utc=True)
    dated = dated.loc[dated["precise_ts"].notna()]
    return dated[["slug", "precise_ts"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Game-level frame: team-week counts pivoted to home_*/away_* + home_cover
# ---------------------------------------------------------------------------

COUNT_COLUMNS: tuple[str, ...] = (
    "n_events_since_freeze",
    "n_events_72h",
    *(
        f"n_{category}_{window}"
        for category in TRANSACTION_CATEGORIES
        for window in ("since_freeze", "72h")
    ),
)


def load_schedules(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    df = raw.loc[raw["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)
    n_before_push_drop = len(df)

    df = add_ats_outcomes(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)
    pushes_or_missing = n_before_push_drop - len(df)

    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = pushes_or_missing
    return df


def build_game_level_frame(team_week_scored: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    home_side = team_week_scored.loc[team_week_scored["is_home"]].copy()
    away_side = team_week_scored.loc[~team_week_scored["is_home"]].copy()

    home_cols = {c: f"home_{c.removeprefix('n_')}" for c in COUNT_COLUMNS}
    away_cols = {c: f"away_{c.removeprefix('n_')}" for c in COUNT_COLUMNS}
    home_side = home_side[["game_id", *COUNT_COLUMNS]].rename(columns=home_cols)
    away_side = away_side[["game_id", *COUNT_COLUMNS]].rename(columns=away_cols)

    merged = games[["game_id", "season", "week", "home_cover"]].merge(
        home_side, on="game_id", how="inner"
    )
    merged = merged.merge(away_side, on="game_id", how="inner")
    merged["week_block"] = merged["season"] * 100 + merged["week"]
    return merged


# ---------------------------------------------------------------------------
# 3. Cells (docs/transaction_wire_battery.md section 4)
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
    home_churn = df["home_events_since_freeze"] >= 1
    away_churn = df["away_events_since_freeze"] >= 1

    add(
        "txn_home_churn_elevated",
        everyone,
        home_churn,
        "Home team has >=1 typed transaction (signing/release/trade/IR/PS-elevation/waiver/"
        "suspension) published after that week's Wednesday-noon-ET line freeze and before "
        "kickoff, vs. 0, response home_cover. Predicted sign: NEGATIVE.",
    )
    add(
        "txn_away_churn_elevated",
        everyone,
        away_churn,
        "Away team has >=1 typed transaction since the line freeze vs. 0, response "
        "home_cover. Predicted sign: POSITIVE.",
    )
    diff_population = home_churn != away_churn
    add(
        "txn_differential_home_worse_churn",
        diff_population,
        home_churn & ~away_churn,
        "Restricted to games where exactly one side has post-freeze churn (home XOR away); "
        "subset = home churn & away not, complement = away churn & home not, response "
        "home_cover. Predicted sign: NEGATIVE.",
    )
    add(
        "txn_home_ir_placement_since_freeze",
        everyone,
        df["home_ir_placement_since_freeze"] >= 1,
        "Home team has >=1 IR placement since the line freeze vs. 0, response home_cover. "
        "Predicted sign: NEGATIVE.",
    )
    add(
        "txn_away_ir_placement_since_freeze",
        everyone,
        df["away_ir_placement_since_freeze"] >= 1,
        "Away team has >=1 IR placement since the line freeze vs. 0, response home_cover. "
        "Predicted sign: POSITIVE.",
    )
    add(
        "txn_home_ps_elevation_72h",
        everyone,
        df["home_practice_squad_elevation_72h"] >= 1,
        "Home team has >=1 practice-squad elevation in the 72 hours before kickoff vs. 0, "
        "response home_cover. Predicted sign: NEGATIVE.",
    )
    add(
        "txn_away_ps_elevation_72h",
        everyone,
        df["away_practice_squad_elevation_72h"] >= 1,
        "Away team has >=1 practice-squad elevation in the 72 hours before kickoff vs. 0, "
        "response home_cover. Predicted sign: POSITIVE.",
    )

    expected = 7
    assert len(cells) == expected, f"expected {expected} predeclared cells, got {len(cells)}"
    return cells


# ---------------------------------------------------------------------------
# 4. Bootstrap (algorithm-identical to nfl_weather_battery_screen.py / fluview)
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
        "n_excluded": int((~population).sum()),
        "n_flag": int(scored_flag.sum()),
        "week_blocked": week_blocked,
        "season_blocked_secondary": season_blocked,
    }


# ---------------------------------------------------------------------------
# 5. Reliability check (docs/transaction_wire_battery.md section 5)
# ---------------------------------------------------------------------------


def compute_reliability(team_week_scored: pd.DataFrame) -> dict[str, Any]:
    long = team_week_scored.copy()
    long["team_id"] = long["team"]
    return split_half_reliability(long, "n_events_since_freeze", seed=BOOTSTRAP_SEED)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--pfr-snapshot", type=Path, default=DEFAULT_PFR_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.schedules is None:
        args.schedules = default_schedules()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "transaction_wire_battery" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} ===")
    games = load_schedules(args.schedules)
    print(
        f"REG {SEASON_START}-{SEASON_END} games: {games.attrs['n_before_push_drop']}, "
        f"pushes/missing dropped: {games.attrs['pushes_or_missing']}, "
        f"scored population: {len(games)}"
    )

    print(f"\n=== loading PFR snapshot {args.pfr_snapshot} ===")
    pfr_index = load_pfr_index(args.pfr_snapshot)
    cache = load_dated_cache(args.pfr_snapshot)
    print(f"pfr index rows: {len(pfr_index)}, dated cache entries: {len(cache)}")

    print("\n=== season completeness (docs/transaction_wire_battery.md sec 1) ===")
    completeness = season_completeness_report(
        pfr_index, cache, seasons=range(SEASON_START, SEASON_END + 1)
    )
    for season, info in sorted(completeness.items()):
        print(
            f"  {season}: target={info['n_target_rows']} dated={info['n_dated_rows']} "
            f"complete={info['complete']}"
        )
    included_seasons = sorted(s for s, info in completeness.items() if info["complete"])
    excluded_seasons = sorted(s for s, info in completeness.items() if not info["complete"])
    print(f"included (complete) seasons: {included_seasons}")
    print(f"excluded (incomplete) seasons: {excluded_seasons}")
    if not included_seasons:
        raise SystemExit("no season meets the completeness rule -- nothing to score")

    print("\n=== classification coverage (docs/transaction_wire_battery.md sec 2) ===")
    relevant = pfr_index.loc[pfr_index["transaction_relevant"]].copy()
    relevant["category"] = relevant["slug"].map(classify_transaction_slug)
    relevant["n_teams_matched"] = relevant["slug"].map(lambda s: len(match_transaction_teams(s)))
    category_counts = relevant["category"].value_counts().to_dict()
    per_season_category = (
        relevant.dropna(subset=["url_year"])
        .assign(url_year=lambda d: d["url_year"].astype(int))
        .pivot_table(
            index="url_year", columns="category", values="slug", aggfunc="count", fill_value=0
        )
    )
    print(category_counts)
    team_match_counts = relevant["n_teams_matched"].value_counts().sort_index().to_dict()
    print(f"team-match distribution: {team_match_counts}")

    print("\n=== building team-week population + dated transactions ===")
    team_week = build_team_week_population(games, season_start=SEASON_START, season_end=SEASON_END)
    team_week = team_week.loc[team_week["season"].isin(included_seasons)].reset_index(drop=True)
    print(f"team-week rows (included seasons only): {len(team_week)}")

    dated = build_dated_transactions(pfr_index, cache)
    print(f"dated transaction_relevant rows (all seasons, any cache): {len(dated)}")
    exploded = explode_dated_transactions(dated)
    print(f"team-attributed dated rows (>=1 team match): {len(exploded)}")

    team_week_scored = attach_transaction_counts(team_week, exploded)

    print("\n=== reliability check (section 5) ===")
    reliability = compute_reliability(team_week_scored)
    print(
        f"  n_team_seasons={reliability['n_team_seasons']} "
        f"pearson_r={reliability['pearson_r']:.4f} "
        f"ci95={reliability['pearson_r_ci95']} "
        f"spearman_brown={reliability['spearman_brown_full_length_reliability']:.4f} "
        f"P+={reliability['probability_positive']:.4f}"
    )

    df = build_game_level_frame(team_week_scored, games)
    print(f"\nscored game-level population: {len(df)}")

    cells = build_cells(df)
    results = []
    for name, spec in cells.items():
        print(f"\n=== {name} ===")
        cell = score_cell(df, name, spec, samples=args.samples, seed=args.seed)
        results.append(cell)
        wb = cell["week_blocked"]
        sb = cell["season_blocked_secondary"]
        print(f"  n_population={cell['n_population']} n_flag={cell['n_flag']}")
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
        "command": "transaction-wire-battery-screen",
        "schedules": str(args.schedules),
        "pfr_snapshot": str(args.pfr_snapshot),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "n_cells": len(cells),
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "season_completeness": {str(k): v for k, v in completeness.items()},
        "included_seasons": included_seasons,
        "excluded_seasons": excluded_seasons,
        "n_reg_games_before_push_drop": games.attrs["n_before_push_drop"],
        "n_pushes_or_missing_dropped": games.attrs["pushes_or_missing"],
        "n_scored_game_population": len(df),
        "category_counts_all_inventory": {str(k): int(v) for k, v in category_counts.items()},
        "per_season_category_counts": {
            str(season): {str(cat): int(n) for cat, n in row.items()}
            for season, row in per_season_category.iterrows()
        },
        "team_match_distribution": {str(k): int(v) for k, v in team_match_counts.items()},
        "n_dated_transaction_relevant_rows": len(dated),
        "n_team_attributed_dated_rows": len(exploded),
        "reliability": reliability,
        "predeclaration": "docs/transaction_wire_battery.md (frozen before scoring)",
        "results": results,
        "ranked_by_abs_full_slate_effect": [cell["name"] for cell in ranked],
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="transaction-wire-battery-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation battery (7 predeclared cells, PFR transaction-wire "
            "team-week churn); mined family, every cell predeclared to record via a separate "
            "scripts/transaction_wire_battery_record.py call regardless of interval shape "
            "(AGENTS.md)."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
