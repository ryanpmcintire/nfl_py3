"""Lead-generation bias battery on free CFB data (measure-only, no registry writes).

Screens 19 predeclared, CFB-specific mechanism hypotheses -- motivation
asymmetries, structural spots, pricing lag, and public premiums -- against
the clean-core CFB benchmark population (``CFB_CLEAN_CORE_SEASONS``,
``src/nfl_ats/cfb_benchmark.py``: 2012-2019 + 2021-2025, ~8,933 games). Every
cell is a simple subset-vs-complement cover-rate comparison, never a fitted
model, so a positive cell here is a LEAD, not a confirmed edge.

Full predeclaration, duplicate check against ``registry/weak_signals.json``
CFB entries and ``docs/cfb_role_features.md``, and the exact cell definitions:
``scratchpad/bias_battery_cfb/predeclaration.md`` (this session's scratch
directory; not part of the repository). This script implements that document
verbatim; nothing here was tuned after seeing a result.

Per-cell reporting, identical for all 19 cells:

* subset cover rate minus complement cover rate (accuracy points), both
  drawn from the SAME declared eligible population so the comparison never
  conflates a scoping filter with the effect under test;
* a ``(season, week)``-blocked bootstrap (``nfl_ats.clv.week_blocked_bootstrap``,
  20,000 samples, seed 20260818) for the interval and ``probability_positive``
  -- never "contains zero" (AGENTS.md binding rule: an interval crossing zero
  is not grounds for rejecting a cell);
* a full-slate-scaled effect (``delta_points * n_subset / n_total_clean_core_games``,
  a fixed 8,933-game denominator across every cell so cells are comparable);
* an era split (clean-core seasons at the median, smaller bootstrap sample so
  19 cells x 2 halves stays cheap) to flag cells whose effect concentrates in
  one era, the same shape ``ROADMAP.md`` PER-07 / ``docs/hc_year_one_fade.md``
  found for the year-1-coach fade.

This script never writes ``registry/rotation_registry.json`` or
``registry/weak_signals.json`` (CFB needs no registry entry under rotation
rule 8 -- ``docs/rotation_registry.md`` -- and this is a screen, not a
confirmation run regardless). It prints proposed, NOT executed,
``nfl-ats weak-signals record`` commands for any cell a human chooses to
promote to the ledger.

Usage::

    ./.tools/uv.exe run --no-sync python scripts/cfb_bias_battery_screen.py
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.cfb import latest_cfb_snapshot
from nfl_ats.cfb_benchmark import CFB_CLEAN_CORE_SEASONS
from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact

REPO = Path(__file__).resolve().parents[1]
DATA_PATH = REPO / "data/processed/cfb_game_features.parquet"
CFB_ROOT = REPO / "data/cfb"
DEFAULT_OUTPUT = REPO / "artifacts/cfb_bias_battery"

PREDECLARATION_SEED = 20260818
DEFAULT_SAMPLES = 20_000
DEFAULT_ERA_SAMPLES = 4_000

#: FBS home venues at or above ~4,200 ft elevation. Predeclared, not tuned.
ALTITUDE_TEAMS = frozenset(
    {"Air Force", "BYU", "Colorado", "Colorado State", "New Mexico", "Utah", "Wyoming"}
)

#: National brand programs with a documented public betting following.
#: Predeclared for recognizability, not for any measured trait.
MARQUEE_PROGRAMS = frozenset(
    {
        "Alabama",
        "Ohio State",
        "Michigan",
        "Notre Dame",
        "USC",
        "Texas",
        "Oklahoma",
        "Georgia",
        "Clemson",
        "LSU",
        "Florida",
        "Penn State",
        "Florida State",
        "Texas A&M",
        "Auburn",
        "Tennessee",
        "Nebraska",
        "Miami",
    }
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_clean_core() -> pd.DataFrame:
    """Completed, spread-orientable clean-core CFB games."""

    frame = pd.read_parquet(DATA_PATH)
    frame = frame.loc[frame["season"].isin(CFB_CLEAN_CORE_SEASONS)].copy()
    frame = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["spread_line"], errors="coerce").notna()
    ].copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    return frame.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)


def load_kickoff_weekday() -> pd.DataFrame:
    """``game_id`` -> US-Eastern weekday name, from the raw CFBD schedule snapshot.

    Converts from the UTC ``start_date`` timestamp before taking the weekday so
    a late West-coast kickoff cannot cross a UTC date boundary into the wrong
    calendar day (the failure mode a naive UTC ``.dt.day_name()`` would have).
    """

    snapshot = latest_cfb_snapshot(CFB_ROOT, "schedules")
    parts: list[pd.DataFrame] = []
    for season in CFB_CLEAN_CORE_SEASONS:
        path = snapshot.season_path(season)
        if not path.is_file():
            raise FileNotFoundError(f"Missing CFB schedules partition: {path}")
        part = pd.read_parquet(path, columns=["game_id", "start_date"])
        parts.append(part)
    schedule = pd.concat(parts, ignore_index=True).drop_duplicates("game_id")
    kickoff = pd.to_datetime(schedule["start_date"], utc=True, errors="coerce").dt.tz_convert(
        "America/New_York"
    )
    return pd.DataFrame(
        {"game_id": schedule["game_id"].to_numpy(), "kickoff_weekday": kickoff.dt.day_name()}
    )


# ---------------------------------------------------------------------------
# Team-side long table
# ---------------------------------------------------------------------------


def build_team_side_table(games: pd.DataFrame, weekday: pd.DataFrame) -> pd.DataFrame:
    """Expand one row per game into two team-side rows (home and away perspective)."""

    home_net_epa = games["diff_off_epa_per_play"] - games["diff_def_epa_per_play"]

    common = {
        "game_id": games["game_id"],
        "season": games["season"],
        "week": games["week"],
        "gameday": games["gameday"],
        "neutral_site": pd.to_numeric(games["neutral_site"], errors="coerce").fillna(0).astype(int),
    }

    home = pd.DataFrame(
        {
            **common,
            "side": "home",
            "side_is_home": True,
            "team": games["home_team"],
            "opponent": games["away_team"],
            "side_covers": pd.to_numeric(games["home_cover"], errors="coerce"),
            "side_ats_margin": pd.to_numeric(games["ats_margin"], errors="coerce"),
            "side_result_margin": pd.to_numeric(games["result"], errors="coerce"),
            "side_favored_margin": pd.to_numeric(games["spread_line"], errors="coerce"),
            "side_rest_edge": pd.to_numeric(games["rest_diff"], errors="coerce"),
            "side_net_epa_edge": home_net_epa,
        }
    )
    away = pd.DataFrame(
        {
            **common,
            "side": "away",
            "side_is_home": False,
            "team": games["away_team"],
            "opponent": games["home_team"],
            "side_covers": 1.0 - pd.to_numeric(games["home_cover"], errors="coerce"),
            "side_ats_margin": -pd.to_numeric(games["ats_margin"], errors="coerce"),
            "side_result_margin": -pd.to_numeric(games["result"], errors="coerce"),
            "side_favored_margin": -pd.to_numeric(games["spread_line"], errors="coerce"),
            "side_rest_edge": -pd.to_numeric(games["rest_diff"], errors="coerce"),
            "side_net_epa_edge": -home_net_epa,
        }
    )
    # home_cover is NaN on a push; the sign-flip above turns that NaN into NaN
    # still (1.0 - NaN == NaN), so pushes stay excluded on both sides.
    table = pd.concat([home, away], ignore_index=True)
    table = table.merge(weekday, on="game_id", how="left")

    table = table.sort_values(["team", "season", "week", "gameday", "game_id"]).reset_index(
        drop=True
    )
    grouped = table.groupby(["team", "season"], sort=False)

    # Win/loss for the RECORD is based on the actual final score, not the ATS
    # outcome, so a push (side_covers NaN) still counts toward the record.
    played = table["side_result_margin"].notna()
    table["_win"] = ((table["side_result_margin"] > 0) & played).astype(float)
    table["_loss"] = ((table["side_result_margin"] < 0) & played).astype(float)
    # transform (not apply) keeps the result aligned to the original row
    # order/index directly -- no MultiIndex reshuffling to undo.
    table["side_wins_entering"] = grouped["_win"].transform(
        lambda s: s.shift(1, fill_value=0.0).cumsum()
    )
    table["side_losses_entering"] = grouped["_loss"].transform(
        lambda s: s.shift(1, fill_value=0.0).cumsum()
    )
    table = table.drop(columns=["_win", "_loss"])

    table["side_prior_result_margin"] = grouped["side_result_margin"].shift(1)
    table["side_prior_ats_margin"] = grouped["side_ats_margin"].shift(1)
    table["side_prior_favored_margin"] = grouped["side_favored_margin"].shift(1)

    table["side_is_finale_proxy"] = table["week"] == grouped["week"].transform("max")

    # Opponent record entering the game: self-join the same team-side table on
    # (opponent, season, week) rather than recomputing, since the opponent's
    # own row for this exact game already carries its own entering record.
    opponent_key = table[["team", "season", "week", "side_wins_entering", "side_losses_entering"]]
    opponent_key = opponent_key.rename(
        columns={
            "team": "opponent",
            "side_wins_entering": "opp_wins_entering",
            "side_losses_entering": "opp_losses_entering",
        }
    )
    table = table.merge(opponent_key, on=["opponent", "season", "week"], how="left")

    table["side_altitude_road"] = (~table["side_is_home"]) & table["opponent"].isin(ALTITUDE_TEAMS)
    table["side_marquee"] = table["team"].isin(MARQUEE_PROGRAMS)

    return table.sort_values(["season", "week", "game_id", "side"]).reset_index(drop=True)


def _percentile_rank(values: pd.Series) -> pd.Series:
    return values.rank(pct=True, method="average")


# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------

MaskFn = Callable[[pd.DataFrame], pd.Series]

CELLS: list[dict[str, Any]] = [
    # --- A. Motivation asymmetries ---------------------------------------
    {
        "name": "bowl_eligibility_self",
        "class": "motivation",
        "direction": "unsigned",
        "eligible": lambda t: t["week"] >= 8,
        "flag": lambda t: t["side_wins_entering"] == 5,
    },
    {
        "name": "bowl_eligibility_opponent",
        "class": "motivation",
        "direction": "unsigned",
        "eligible": lambda t: t["week"] >= 8,
        "flag": lambda t: t["opp_wins_entering"] == 5,
    },
    {
        "name": "eliminated_apathy",
        "class": "motivation",
        "direction": "negative",
        "eligible": lambda t: t["week"] >= 10,
        "flag": lambda t: t["side_wins_entering"] <= 2,
    },
    {
        "name": "cfp_style_points_big_favorite",
        "class": "motivation",
        "direction": "positive",
        "eligible": lambda t: (t["season"] >= 2014) & (t["week"] >= 10),
        "flag": lambda t: t["side_favored_margin"] >= 21,
    },
    {
        "name": "letdown_after_upset_win",
        "class": "motivation",
        "direction": "negative",
        "eligible": lambda t: t["side_prior_ats_margin"].notna(),
        "flag": lambda t: (
            (t["side_prior_favored_margin"] <= -14) & (t["side_prior_result_margin"] > 0)
        ),
    },
    {
        "name": "rivalry_finale_proxy",
        "class": "motivation",
        "direction": "unsigned",
        "eligible": lambda t: pd.Series(True, index=t.index),
        "flag": lambda t: t["side_is_finale_proxy"],
    },
    # --- B. Structural spots ----------------------------------------------
    {
        "name": "mactic_short_prep_away",
        "class": "structural",
        "direction": "negative",
        "eligible": lambda t: (~t["side_is_home"]) & t["kickoff_weekday"].notna(),
        "flag": lambda t: t["kickoff_weekday"].isin(["Tuesday", "Wednesday"]),
    },
    {
        "name": "high_altitude_road",
        "class": "structural",
        "direction": "negative",
        "eligible": lambda t: ~t["side_is_home"],
        "flag": lambda t: t["side_altitude_road"],
    },
    {
        "name": "bye_week_rest_edge",
        "class": "structural",
        "direction": "positive",
        "eligible": lambda t: t["side_rest_edge"].notna(),
        "flag": lambda t: t["side_rest_edge"] >= 6,
    },
    {
        "name": "short_week_rest_disadvantage",
        "class": "structural",
        "direction": "negative",
        "eligible": lambda t: t["side_rest_edge"].notna(),
        "flag": lambda t: t["side_rest_edge"] <= -4,
    },
    {
        "name": "post_blowout_win_letdown",
        "class": "structural",
        "direction": "negative",
        "eligible": lambda t: t["side_prior_result_margin"].notna(),
        "flag": lambda t: t["side_prior_result_margin"] >= 28,
    },
    {
        "name": "neutral_site_designated_home",
        "class": "structural",
        "direction": "negative",
        "eligible": lambda t: t["side_is_home"],
        "flag": lambda t: t["neutral_site"] == 1,
    },
    # --- C. Pricing lag ------------------------------------------------------
    {
        "name": "off_big_ats_cover_carryover",
        "class": "pricing_lag",
        "direction": "negative",
        "eligible": lambda t: t["side_prior_ats_margin"].notna(),
        "flag": lambda t: t["side_prior_ats_margin"] >= 14,
    },
    {
        "name": "off_big_ats_miss_carryover",
        "class": "pricing_lag",
        "direction": "positive",
        "eligible": lambda t: t["side_prior_ats_margin"].notna(),
        "flag": lambda t: t["side_prior_ats_margin"] <= -14,
    },
    {
        "name": "week2_overreaction_to_week1_blowout",
        "class": "pricing_lag",
        "direction": "unsigned",
        "eligible": lambda t: t["week"] == 2,
        "flag": lambda t: t["side_prior_result_margin"].abs() >= 21,
    },
    {
        "name": "state_quality_market_gap",
        "class": "pricing_lag",
        "direction": "positive",
        "eligible": lambda t: t["side_net_epa_edge"].notna() & t["side_favored_margin"].notna(),
        # Threshold computed once, inline, on the eligible population itself
        # (see main()); placeholder replaced before use.
        "flag": None,
    },
    # --- D. Public premiums --------------------------------------------------
    {
        "name": "marquee_favorite_premium",
        "class": "public_premium",
        "direction": "negative",
        "eligible": lambda t: t["side_favored_margin"].notna(),
        "flag": lambda t: t["side_marquee"] & (t["side_favored_margin"] > 0),
    },
    {
        "name": "marquee_underdog_value",
        "class": "public_premium",
        "direction": "positive",
        "eligible": lambda t: t["side_favored_margin"].notna(),
        "flag": lambda t: t["side_marquee"] & (t["side_favored_margin"] < 0),
    },
    {
        "name": "heavy_public_favorite",
        "class": "public_premium",
        "direction": "negative",
        "eligible": lambda t: t["side_favored_margin"].notna(),
        "flag": lambda t: t["side_favored_margin"] >= 17,
    },
]

assert len(CELLS) == 19, f"expected 19 predeclared cells, found {len(CELLS)}"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _delta_metric_fn(frame: pd.DataFrame) -> dict[str, float]:
    flagged = frame.loc[frame["_flag"], "side_covers"]
    other = frame.loc[~frame["_flag"], "side_covers"]
    flagged = flagged.dropna()
    other = other.dropna()
    delta = float(flagged.mean() - other.mean()) if len(flagged) and len(other) else float("nan")
    return {"delta": delta}


def score_population(
    population: pd.DataFrame, flag: pd.Series, *, samples: int, seed: int
) -> dict[str, Any]:
    scored = population.copy()
    scored["_flag"] = flag.to_numpy()
    valid = scored.dropna(subset=["side_covers"])
    n_subset = int((valid["_flag"]).sum())
    n_complement = int((~valid["_flag"]).sum())
    if n_subset == 0 or n_complement == 0:
        return {
            "n_eligible": len(valid),
            "n_subset": n_subset,
            "n_complement": n_complement,
            "estimate_points": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "probability_positive": float("nan"),
        }
    point = _delta_metric_fn(valid)["delta"]
    boot = week_blocked_bootstrap(valid, _delta_metric_fn, block="week", samples=samples, seed=seed)
    row = boot.iloc[0]
    return {
        "n_eligible": len(valid),
        "n_subset": n_subset,
        "n_complement": n_complement,
        "estimate_points": 100.0 * point,
        "lower": 100.0 * float(row["lower"]),
        "upper": 100.0 * float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
    }


def score_cell(
    table: pd.DataFrame,
    cell: dict[str, Any],
    *,
    n_total_games: int,
    samples: int,
    era_samples: int,
    seed: int,
) -> dict[str, Any]:
    eligible_mask = cell["eligible"](table)
    population = table.loc[eligible_mask].reset_index(drop=True)
    flag_fn = cell["flag"]
    flag = flag_fn(population)

    overall = score_population(population, flag, samples=samples, seed=seed)
    overall["full_slate_scaled_points"] = (
        overall["estimate_points"] * overall["n_subset"] / n_total_games
        if np.isfinite(overall["estimate_points"])
        else float("nan")
    )

    clean_core_seasons = sorted(CFB_CLEAN_CORE_SEASONS)
    median_season = clean_core_seasons[len(clean_core_seasons) // 2]
    era_result: dict[str, Any] = {}
    for label, era_mask in (
        ("early", population["season"] < median_season),
        ("late", population["season"] >= median_season),
    ):
        era_population = population.loc[era_mask].reset_index(drop=True)
        era_flag = flag.loc[era_mask].reset_index(drop=True)
        era_scored = score_population(era_population, era_flag, samples=era_samples, seed=seed)
        era_result[label] = era_scored

    return {
        "name": cell["name"],
        "class": cell["class"],
        "predicted_direction": cell["direction"],
        "median_split_season": median_season,
        **overall,
        "era_early": era_result["early"],
        "era_late": era_result["late"],
    }


# ---------------------------------------------------------------------------
# Verdict labelling (screening only -- never a closing verdict; see AGENTS.md)
# ---------------------------------------------------------------------------


def _screen_label(result: dict[str, Any]) -> str:
    """A screening lean, not a closing verdict. Every cell stays `unresolved`
    unless the WHOLE interval sits on the wrong side of the predicted
    direction -- the only closing ground this measure-only battery applies.
    """

    direction = result["predicted_direction"]
    lower, upper = result["lower"], result["upper"]
    if not (np.isfinite(lower) and np.isfinite(upper)):
        return "insufficient_data"
    if direction == "positive" and upper < 0.0:
        return "wrong_sign_resolved"
    if direction == "negative" and lower > 0.0:
        return "wrong_sign_resolved"
    return "unresolved"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_battery(
    *, samples: int, era_samples: int, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    t0 = time.time()
    games = load_clean_core()
    weekday = load_kickoff_weekday()
    table = build_team_side_table(games, weekday)
    n_total_games = int(games["game_id"].nunique())
    print(
        f"{len(games)} clean-core games, {len(table)} team-side rows, "
        f"{n_total_games} distinct games",
        flush=True,
    )

    # state_quality_market_gap's flag needs its eligible population's own
    # percentile ranks, computed once inline rather than baked into the
    # static CELLS table.
    cells = [dict(cell) for cell in CELLS]
    for cell in cells:
        if cell["name"] != "state_quality_market_gap":
            continue

        def flag_fn(pop: pd.DataFrame) -> pd.Series:
            quality_pct = _percentile_rank(pop["side_net_epa_edge"])
            market_pct = _percentile_rank(pop["side_favored_margin"])
            return (quality_pct - market_pct) >= 0.40

        cell["flag"] = flag_fn

    rows = []
    for cell in cells:
        result = score_cell(
            table,
            cell,
            n_total_games=n_total_games,
            samples=samples,
            era_samples=era_samples,
            seed=seed,
        )
        result["screen_label"] = _screen_label(result)
        rows.append(result)
        print(
            f"  {result['name']:38s} n_subset={result['n_subset']:5d} "
            f"delta={result['estimate_points']:+.2f} "
            f"[{result['lower']:+.2f},{result['upper']:+.2f}] "
            f"P+={result['probability_positive']:.3f} "
            f"full_slate={result['full_slate_scaled_points']:+.3f} "
            f"[{result['screen_label']}]",
            flush=True,
        )

    table_out = pd.json_normalize(rows, sep=".")
    summary = {
        "seed": seed,
        "samples": samples,
        "era_samples": era_samples,
        "clean_core_games": n_total_games,
        "team_side_rows": len(table),
        "n_cells": len(rows),
        "elapsed_seconds": time.time() - t0,
    }
    return table_out, summary


def propose_record_commands(results: pd.DataFrame) -> list[str]:
    """Proposed, NOT executed, `nfl-ats weak-signals record` commands.

    CFB needs no rotation-registry window (rule 8), so only the weak-signals
    ledger applies, and only as a proposal for a human/orchestrator to run --
    this script never calls it.
    """

    commands = []
    for _, row in results.sort_values("probability_positive", ascending=False).iterrows():
        p_positive = row["probability_positive"]
        if not np.isfinite(p_positive):
            continue
        lean = "positive" if p_positive >= 0.5 else "negative"
        commands.append(
            "./.tools/uv.exe run --no-sync nfl-ats weak-signals record "
            f"--name cfb_bias_battery_{row['name']} --league cfb "
            "--effect-units accuracy_points "
            f"--estimate {row['estimate_points']:.4f} "
            f"--lower {row['lower']:.4f} --upper {row['upper']:.4f} "
            f"--probability-positive {p_positive:.4f} "
            f"--classification unresolved "
            f'--notes "CFB bias battery {row["class"]} cell, {lean} lean, '
            f"n_subset={int(row['n_subset'])}, screen_label={row['screen_label']}; "
            "lead-generation only, not confirmed against the fitted CFB benchmark; "
            'see scratchpad predeclaration.md"'
        )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--era-samples", type=int, default=DEFAULT_ERA_SAMPLES)
    parser.add_argument("--seed", type=int, default=PREDECLARATION_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output = args.output / stamp
    output.mkdir(parents=True, exist_ok=True)
    print(f"writing to {output}", flush=True)

    results, summary = run_battery(
        samples=args.samples, era_samples=args.era_samples, seed=args.seed
    )
    results.to_csv(output / "cells.csv", index=False)
    stamp_sidecar(output / "cells.csv")  # ENG-38

    ranked = results.assign(lean_strength=(results["probability_positive"] - 0.5).abs())
    ranked = ranked.sort_values("lean_strength", ascending=False)
    top = ranked.head(5)[
        [
            "name",
            "class",
            "predicted_direction",
            "estimate_points",
            "lower",
            "upper",
            "probability_positive",
            "full_slate_scaled_points",
            "screen_label",
        ]
    ]
    print("\n=== top 5 leads by |P+ - 0.5| ===")
    print(top.to_string(index=False), flush=True)

    commands = propose_record_commands(results)
    (output / "proposed_record_commands.txt").write_text("\n\n".join(commands), encoding="utf-8")
    stamp_sidecar(output / "proposed_record_commands.txt")  # ENG-38

    summary["output"] = str(output)
    write_stamped_artifact(summary, output / "summary.json")  # ENG-38
    print(f"\nwrote {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
