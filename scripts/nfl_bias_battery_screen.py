"""Lead-generation bias battery: 17 predeclared situational/behavioral NFL
market checks, screened against the spread on 2009-2025 (including the
mined 2018-2025 era) with a week-blocked bootstrap, a full-slate-scaled
effect, and a 2009-2017 vs 2018-2025 era split on every cell.

**Measure-only.** This script never writes to ``registry/`` and never edits
a tracked doc. Every predeclared cell is scored and reported, including the
14+ whose interval crosses zero -- per ``AGENTS.md``, an interval containing
zero is never a rejection, and this battery's entire purpose is to surface
category-3 leads, not to prove any one of them.

The full predeclaration (hypothesis, exact subset definition, mechanism,
direction) is frozen in
``<scratchpad>/bias_battery_nfl/predeclaration.md`` and was written before
this script scored anything. This module implements exactly what that
document specifies; nothing here should diverge from it.

**Method**, reused from two precedents already in this repo:
``scripts/hc_year_one_fade.py`` (team-game long table, one row per side per
game, cluster/block bootstrap on cover rate) and
``scripts/penalty_discipline_interval.py`` (subset-vs-complement gap scaled
by the fraction of the slate the subset represents, week-blocked joint
block bootstrap, `probability_positive` as the fraction of draws favouring
the predeclared direction). This script generalizes the two-quartile block
bootstrap in the second precedent to an arbitrary boolean subset flag vs.
its complement (rather than two extreme quartiles), since most of this
battery's hypotheses are not naturally binnable into quartiles.

Data: ``data/processed/game_features.parquet`` inner-joined on ``game_id``
with the newest ``data/raw/*/schedules.parquet`` snapshot (for
``home_rest``/``away_rest``/``roof``/``surface``/``home_qb_name``/
``away_qb_name``, none of which are in the feature table). REG season only,
2009-2025. Team codes canonicalized with
``nfl_ats.constants.TEAM_ABBREVIATION_ALIASES``.

Writes JSON to ``artifacts/nfl_bias_battery/<UTC timestamp>/results.json``
and a plain-text summary table to the same directory. Proposes (does not
execute) ``nfl-ats weak-signals record`` commands for the top leads at the
end of stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

sys.path.append(str(REPO / "scripts"))

from _common import block_bootstrap_two_group, latest_schedules  # noqa: E402

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402

DEFAULT_FEATURES = REPO / "data/processed/game_features.parquet"


def default_schedules() -> Path:
    """Resolve lazily so importing this module never requires local data."""
    return latest_schedules()


PT_TEAMS = frozenset({"SEA", "SF", "LA", "LAC", "LV"})
MARQUEE_TEAMS = frozenset({"DAL", "GB", "PIT", "NE"})
ERA_SPLITS: tuple[tuple[str, int, int], ...] = (
    ("2009_2017", 2009, 2017),
    ("2018_2025", 2018, 2025),
)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260818


def _canonical(team: pd.Series) -> pd.Series:
    return team.map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def load_merged(features_path: Path, schedules_path: Path) -> pd.DataFrame:
    features = pd.read_parquet(features_path)
    schedules = pd.read_parquet(schedules_path).loc[
        :,
        [
            "game_id",
            "away_rest",
            "home_rest",
            "roof",
            "surface",
            "away_qb_name",
            "home_qb_name",
        ],
    ]
    merged = features.merge(schedules, on="game_id", how="inner", validate="one_to_one")
    return merged.loc[merged["game_type"] == "REG"].copy()


_LONG_BASE_HOME = {
    "team": "home_team",
    "opponent": "away_team",
    "own_rest": "home_rest",
    "opp_rest": "away_rest",
    "own_qb_name": "home_qb_name",
}
_LONG_BASE_AWAY = {
    "team": "away_team",
    "opponent": "home_team",
    "own_rest": "away_rest",
    "opp_rest": "home_rest",
    "own_qb_name": "away_qb_name",
}


def build_long_table(merged: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, side); pushes (``team_covered`` NaN) dropped."""

    merged = merged.copy()
    merged["home_team"] = _canonical(merged["home_team"])
    merged["away_team"] = _canonical(merged["away_team"])
    merged["gameday"] = pd.to_datetime(merged["gameday"], errors="raise")
    merged["result"] = pd.to_numeric(merged["result"], errors="coerce")
    merged["spread_line"] = pd.to_numeric(merged["spread_line"], errors="coerce")
    merged["temp"] = pd.to_numeric(merged["temp"], errors="coerce")
    merged["gametime_hour"] = pd.to_numeric(
        merged["gametime"].astype(str).str.split(":").str[0], errors="coerce"
    )

    sides = []
    for is_home, cols in ((True, _LONG_BASE_HOME), (False, _LONG_BASE_AWAY)):
        side = pd.DataFrame(
            {
                "game_id": merged["game_id"],
                "season": merged["season"].astype(int),
                "week": merged["week"].astype(int),
                "gameday": merged["gameday"],
                "team": merged[cols["team"]],
                "opponent": merged[cols["opponent"]],
                "is_home": is_home,
                "team_covered": merged["home_cover"] if is_home else 1.0 - merged["home_cover"],
                "team_ats_margin": merged["ats_margin"] if is_home else -merged["ats_margin"],
                "team_score_margin": merged["result"] if is_home else -merged["result"],
                "spread_line": merged["spread_line"],
                "team_spread": merged["spread_line"] if is_home else -merged["spread_line"],
                "own_rest": merged[cols["own_rest"]],
                "opp_rest": merged[cols["opp_rest"]],
                "own_qb_name": merged[cols["own_qb_name"]],
                "div_game": merged["div_game"],
                "neutral_site": merged["neutral_site"],
                "weekday": merged["weekday"],
                "gametime_hour": merged["gametime_hour"],
                "temp": merged["temp"],
                "roof": merged["roof"],
            }
        )
        sides.append(side)

    long_df = pd.concat(sides, ignore_index=True)
    long_df = long_df.loc[long_df["team_covered"].notna()].copy()
    long_df["team"] = _canonical(long_df["team"])
    long_df["opponent"] = _canonical(long_df["opponent"])
    long_df["team_is_favorite"] = long_df["team_spread"] > 0
    long_df["week_block"] = long_df["season"] * 100 + long_df["week"]
    long_df = long_df.sort_values(["team", "season", "gameday"]).reset_index(drop=True)
    return long_df


def _qb_backup_flag(group: pd.DataFrame) -> pd.Series:
    """Per (team, season) group, sorted by gameday: True iff this game's QB
    differs from the modal QB over strictly-prior games, requiring >=3 prior
    starts to establish a baseline (else NaN, excluded from both arms).
    """

    counts: Counter[str] = Counter()
    flags: list[float] = []
    for qb_name in group["own_qb_name"]:
        total_prior = sum(counts.values())
        if total_prior >= 3:
            modal_qb = counts.most_common(1)[0][0]
            flags.append(1.0 if qb_name != modal_qb else 0.0)
        else:
            flags.append(np.nan)
        if isinstance(qb_name, str) and qb_name:
            counts[qb_name] += 1
    return pd.Series(flags, index=group.index)


def add_history_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Add every within-season, strictly-prior-game derived column the 17
    hypotheses need. All are pregame-safe: each uses only games with an
    earlier ``gameday`` in the same (team, season), or (for the sandwich
    spot) the team's own known future schedule, never a game outcome that
    has not yet happened relative to the one it is being attached to.
    """

    long_df = long_df.sort_values(["team", "season", "gameday"]).reset_index(drop=True)
    grouped = long_df.groupby(["team", "season"], sort=False)

    # --- cumulative win pct, strictly prior games this season ---
    win = (long_df["team_score_margin"] > 0).astype(float)
    prior_games = grouped.cumcount()
    long_df["_win"] = win
    cum_wins_incl = long_df.groupby(["team", "season"], sort=False)["_win"].cumsum()
    prior_wins = cum_wins_incl - long_df["_win"]
    long_df["prior_games"] = prior_games.to_numpy()
    long_df["prior_win_pct"] = np.where(
        long_df["prior_games"] > 0, prior_wins / long_df["prior_games"], np.nan
    )
    long_df = long_df.drop(columns=["_win"])

    # --- opponent's own prior win pct this season/week (self-join) ---
    opp_stats = long_df[["team", "season", "week", "prior_win_pct", "prior_games"]].rename(
        columns={
            "team": "opponent",
            "prior_win_pct": "opp_prior_win_pct",
            "prior_games": "opp_prior_games",
        }
    )
    long_df = long_df.merge(opp_stats, on=["opponent", "season", "week"], how="left")

    # --- prior game's own score margin (blowout letdown/bounce triggers) ---
    grouped = long_df.groupby(["team", "season"], sort=False)
    long_df["prior_score_margin"] = grouped["team_score_margin"].shift(1)

    # --- backup QB flag (loop per group; ~9,300 rows total, small groups) ---
    long_df["backup_qb_flag"] = np.nan
    for _, group in long_df.groupby(["team", "season"], sort=False):
        long_df.loc[group.index, "backup_qb_flag"] = _qb_backup_flag(group).to_numpy()

    # --- consecutive true road games (this + previous 2, same team/season) ---
    long_df["is_true_road"] = (~long_df["is_home"]) & (long_df["neutral_site"] == 0)
    grouped = long_df.groupby(["team", "season"], sort=False)
    prev1 = grouped["is_true_road"].shift(1).fillna(False)
    prev2 = grouped["is_true_road"].shift(2).fillna(False)
    long_df["three_plus_road_flag"] = long_df["is_true_road"] & prev1 & prev2

    # --- sandwich spot: not div this week, div last week, div next week ---
    grouped = long_df.groupby(["team", "season"], sort=False)
    prior_div = grouped["div_game"].shift(1)
    next_div = grouped["div_game"].shift(-1)
    long_df["sandwich_flag"] = (long_df["div_game"] == 0) & (prior_div == 1) & (next_div == 1)

    # --- division revenge: 2nd+ meeting this season vs same opponent, lost 1st ---
    long_df = long_df.sort_values(["team", "opponent", "season", "gameday"]).reset_index(drop=True)
    grouped = long_df.groupby(["team", "opponent", "season"], sort=False)
    meeting_rank = grouped.cumcount()
    first_margin = grouped["team_score_margin"].transform("first")
    long_df["revenge_flag"] = (meeting_rank >= 1) & (first_margin < 0)

    long_df = long_df.sort_values(["team", "season", "gameday"]).reset_index(drop=True)
    return long_df


def build_hypotheses(long_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return {name: {"flag": bool Series, "sign": +-1, "description": str,
    "eligible": bool Series or None}}. ``eligible`` restricts the population
    for cells (only ``backup_qb_start``) where an undefined baseline must
    exclude rows from BOTH arms rather than default them to the complement.
    """

    n = long_df["own_rest"] - long_df["opp_rest"]
    hyps: dict[str, dict[str, Any]] = {}

    hyps["bad_team_late"] = {
        "flag": long_df["week"].between(11, 18)
        & (long_df["prior_games"] >= 9)
        & (long_df["prior_win_pct"] <= 0.300),
        "sign": -1,
        "mechanism_class": "incentive_misalignment",
        "description": "Team out of realistic contention, weeks 11-18, prior win pct <= 30%.",
    }
    hyps["great_team_late"] = {
        "flag": long_df["week"].between(15, 18)
        & (long_df["prior_games"] >= 13)
        & (long_df["prior_win_pct"] >= 0.800),
        "sign": -1,
        "mechanism_class": "incentive_misalignment",
        "description": "Proxy for a locked seed, weeks 15-18, prior win pct >= 80%.",
    }
    hyps["motivation_mismatch"] = {
        "flag": (long_df["prior_win_pct"] >= 0.400)
        & long_df["week"].between(11, 18)
        & (long_df["opp_prior_games"] >= 9)
        & (long_df["opp_prior_win_pct"] <= 0.300),
        "sign": 1,
        "mechanism_class": "incentive_misalignment",
        "description": "Competitive team (>=40% prior) facing a bad_team_late opponent.",
    }
    hyps["backup_qb_start"] = {
        "flag": long_df["backup_qb_flag"] == 1.0,
        "sign": 1,
        "mechanism_class": "transition_events",
        "description": "Starting QB differs from the team's modal QB this season (>=3 prior "
        "starts).",
        "eligible": long_df["backup_qb_flag"].notna(),
    }
    hyps["post_blowout_win_letdown"] = {
        "flag": long_df["prior_score_margin"] >= 17,
        "sign": -1,
        "mechanism_class": "attention_pricing_lag",
        "description": "Immediately preceding game (this season) was a win by >=17 raw points.",
    }
    hyps["post_blowout_loss_bounce"] = {
        "flag": long_df["prior_score_margin"] <= -17,
        "sign": 1,
        "mechanism_class": "attention_pricing_lag",
        "description": "Immediately preceding game (this season) was a loss by >=17 raw points.",
    }
    hyps["short_week"] = {
        "flag": long_df["own_rest"] <= 5,
        "sign": -1,
        "mechanism_class": "attention_pricing_lag",
        "description": "Team's own rest <= 5 days.",
    }
    hyps["extra_rest_edge"] = {
        "flag": n >= 4,
        "sign": 1,
        "mechanism_class": "attention_pricing_lag",
        "description": "Team's rest minus opponent's rest >= 4 days.",
    }
    hyps["three_plus_road_games"] = {
        "flag": long_df["three_plus_road_flag"],
        "sign": -1,
        "mechanism_class": "attention_pricing_lag",
        "description": "3rd+ consecutive true road game this season.",
    }
    hyps["sandwich_spot"] = {
        "flag": long_df["sandwich_flag"],
        "sign": -1,
        "mechanism_class": "attention_pricing_lag",
        "description": "Non-division game flanked by a division game last week and next week.",
    }
    west_coast_mask = (
        (~long_df["is_home"])
        & long_df["team"].isin(PT_TEAMS)
        & (~long_df["opponent"].isin(PT_TEAMS))
        & (long_df["neutral_site"] == 0)
        & (long_df["gametime_hour"] < 14)
    )
    hyps["west_coast_early_kickoff"] = {
        "flag": west_coast_mask,
        "sign": -1,
        "mechanism_class": "attention_pricing_lag",
        "description": "Traveling Pacific-timezone team, non-PT opponent, kickoff before 14:00 ET.",
    }
    dec_weather_mask = (
        long_df["is_home"]
        & (long_df["week"] >= 14)
        & long_df["roof"].isin(["outdoors", "open"])
        & (long_df["temp"] <= 32)
        & (long_df["spread_line"] < 0)
    )
    hyps["december_weather_dogs"] = {
        "flag": dec_weather_mask,
        "sign": 1,
        "mechanism_class": "attention_pricing_lag",
        "description": "Home underdog, week>=14, outdoor/open roof, temp<=32F.",
    }
    hyps["division_revenge_game"] = {
        "flag": long_df["revenge_flag"],
        "sign": 1,
        "mechanism_class": "attention_pricing_lag",
        "description": "2nd meeting this season vs. same opponent; team lost the 1st meeting.",
    }
    hyps["marquee_favorite"] = {
        "flag": long_df["team"].isin(MARQUEE_TEAMS) & long_df["team_is_favorite"],
        "sign": -1,
        "mechanism_class": "public_pricing_premium",
        "description": "Team in {DAL,GB,PIT,NE} (inferred highest-handle list) and favored.",
    }
    primetime_mask = (
        long_df["weekday"].isin(["Thursday", "Monday"])
        | (long_df["weekday"].eq("Sunday") & (long_df["gametime_hour"] >= 20))
    ) & long_df["team_is_favorite"]
    hyps["primetime_favorite"] = {
        "flag": primetime_mask,
        "sign": -1,
        "mechanism_class": "public_pricing_premium",
        "description": "Thu/Mon or Sunday-night favorite (Saturday excluded, stated limitation).",
    }
    hyps["home_underdog"] = {
        "flag": long_df["is_home"] & (long_df["spread_line"] < 0),
        "sign": 1,
        "mechanism_class": "public_pricing_premium",
        "description": "Home team getting points.",
    }
    hyps["large_favorite"] = {
        "flag": long_df["team_is_favorite"] & (long_df["team_spread"] > 10),
        "sign": -1,
        "mechanism_class": "public_pricing_premium",
        "description": "Favored by more than 10 points.",
    }
    return hyps


def summarize_population(
    df: pd.DataFrame, *, flag: pd.Series, sign: int, samples: int, seed: int
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
    subset_cover = float(work.loc[work["_flag"], "team_covered"].mean())
    complement_cover = float(work.loc[~work["_flag"], "team_covered"].mean())
    raw_gap_pts = sign * (subset_cover - complement_cover) * 100.0
    fraction_of_slate = n_flag / n_total
    full_slate_effect_pts = raw_gap_pts * fraction_of_slate

    draws = block_bootstrap_two_group(
        work,
        flag_col="_flag",
        value_col="team_covered",
        block_col="week_block",
        samples=samples,
        seed=seed,
    )
    signed_draws = sign * draws
    dropped = samples - len(draws)
    scaled_draws = signed_draws * fraction_of_slate
    lower, upper = (
        np.quantile(scaled_draws, [0.025, 0.975]) if len(scaled_draws) else (np.nan, np.nan)
    )

    return {
        "n_total": n_total,
        "n_flag": n_flag,
        "n_complement": n_complement,
        "n_week_blocks": int(work["week_block"].nunique()),
        "subset_cover": subset_cover,
        "complement_cover": complement_cover,
        "raw_gap_pts": raw_gap_pts,
        "fraction_of_slate": fraction_of_slate,
        "full_slate_effect_pts": full_slate_effect_pts,
        "week_blocked_ci95_scaled": [float(lower), float(upper)],
        "probability_positive": float(np.mean(signed_draws > 0)) if len(signed_draws) else np.nan,
        "bootstrap_samples": samples,
        "dropped_draws": int(dropped),
        "insufficient_data": False,
    }


def score_hypothesis(
    long_df: pd.DataFrame, name: str, spec: dict[str, Any], *, samples: int, seed: int
) -> dict[str, Any]:
    population = long_df
    flag = spec["flag"]
    eligible = spec.get("eligible")
    if eligible is not None:
        population = long_df.loc[eligible].reset_index(drop=True)
        flag = flag.loc[eligible].reset_index(drop=True)

    full = summarize_population(
        population, flag=flag, sign=spec["sign"], samples=samples, seed=seed
    )

    era_results = {}
    for era_label, start, end in ERA_SPLITS:
        era_mask = population["season"].between(start, end)
        era_pop = population.loc[era_mask].reset_index(drop=True)
        era_flag = flag.loc[era_mask].reset_index(drop=True)
        era_results[era_label] = summarize_population(
            era_pop, flag=era_flag, sign=spec["sign"], samples=samples, seed=seed
        )

    return {
        "name": name,
        "mechanism_class": spec["mechanism_class"],
        "description": spec["description"],
        "sign_dir": spec["sign"],
        "full_period": full,
        "era_split": era_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.schedules is None:
        args.schedules = default_schedules()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "nfl_bias_battery" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.features} + {args.schedules} ===")
    merged = load_merged(args.features, args.schedules)
    print(f"REG games: {len(merged)}")

    long_df = build_long_table(merged)
    long_df = add_history_features(long_df)
    print(f"team-game rows (pushes dropped): {len(long_df)}")

    hypotheses = build_hypotheses(long_df)
    assert len(hypotheses) == 17, f"expected 17 predeclared cells, got {len(hypotheses)}"

    results = []
    for name, spec in hypotheses.items():
        print(f"\n=== {name} ({spec['mechanism_class']}) ===")
        cell = score_hypothesis(long_df, name, spec, samples=args.samples, seed=args.seed)
        results.append(cell)
        full = cell["full_period"]
        if full.get("insufficient_data"):
            print("  insufficient data (empty subset or complement)")
            continue
        print(
            f"  n_flag={full['n_flag']} n_total={full['n_total']} "
            f"subset_cover={full['subset_cover']:.4f} "
            f"complement_cover={full['complement_cover']:.4f}"
        )
        print(
            f"  raw_gap={full['raw_gap_pts']:+.3f}pts "
            f"frac_of_slate={full['fraction_of_slate']:.4f} "
            f"full_slate_effect={full['full_slate_effect_pts']:+.4f}pts"
        )
        print(
            f"  week-blocked 95% scaled [{full['week_blocked_ci95_scaled'][0]:+.4f}, "
            f"{full['week_blocked_ci95_scaled'][1]:+.4f}] P+={full['probability_positive']:.4f}"
        )
        for era_label, _, _ in ERA_SPLITS:
            era = cell["era_split"][era_label]
            if era.get("insufficient_data"):
                print(f"  [{era_label}] insufficient data")
                continue
            print(
                f"  [{era_label}] n_flag={era['n_flag']} cover={era['subset_cover']:.4f} "
                f"vs {era['complement_cover']:.4f} "
                f"full_slate_effect={era['full_slate_effect_pts']:+.4f}pts "
                f"P+={era['probability_positive']:.4f}"
            )

    ranked = sorted(
        (r for r in results if not r["full_period"].get("insufficient_data")),
        key=lambda r: abs(r["full_period"]["full_slate_effect_pts"]),
        reverse=True,
    )
    print("\n=== ranked by |full-slate effect|, full period 2009-2025 ===")
    for rank, cell in enumerate(ranked, start=1):
        full = cell["full_period"]
        print(
            f"{rank:>2}. {cell['name']:<28} {full['full_slate_effect_pts']:+.4f}pts "
            f"P+={full['probability_positive']:.4f} n_flag={full['n_flag']}"
        )

    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "n_hypotheses": len(hypotheses),
        "n_reg_games": len(merged),
        "n_team_game_rows": len(long_df),
        "predeclaration": "scratchpad/bias_battery_nfl/predeclaration.md (frozen before scoring)",
        "results": results,
        "ranked_by_abs_full_slate_effect": [cell["name"] for cell in ranked],
    }
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
