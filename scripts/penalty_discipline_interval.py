"""Reconstruct and interval the `penalty_discipline` market check (ROADMAP PBP-07).

`registry/weak_signals.json`'s `penalty_discipline` entry records a 0.67-point
quartile-cover-rate gap (most-penalized quartile 49.85% vs least-penalized
50.52%, ~4,431 games, 2009-2025) with no interval and no `probability_positive`
-- its own notes say it "cannot be pooled until re-measured with one." No
script behind that original number survives. This one rebuilds the construct
from scratch on data already local to this repo and adds the missing
uncertainty.

**Team penalty rate, reverse-engineered.** The narrowed 45-column play-by-play
snapshot (`nfl_ats.pbp.PBP_SNAPSHOT_COLUMNS`) carries `penalty` and
`penalty_yards` but no `penalty_team` -- so which side committed a given
penalty is not locally knowable play-by-play. Trying candidate definitions
against the two numbers ROADMAP records (mean 0.0750, sd 0.0101 across
team-seasons, year-over-year reliability +0.261) pins down the one that was
almost certainly used: `mean(penalty)` over EVERY raw play-by-play row
(no play-type filter at all -- kickoffs, no-plays, everything) where
`posteam == team`, regular season only. That reproduces mean 0.07505,
sd 0.01013, and reliability +0.2604 -- all three to 3 significant figures.
It is a mixed signal (offensive penalties by this team plus defensive
penalties by whichever opponent it faced, on this team's own offensive
snaps), not a clean isolate of one side's discipline, but it is what the
recorded figures imply was measured, so this script uses it unchanged.

**The market check.** Team-season penalty rate lagged one season (a team's
CURRENT-season games are checked against its PRIOR-season rate; season 2009
is dropped for every team since it has no prior season locally). Quartiles
are cut globally across all 512 lagged team-seasons (pd.qcut, 4 bins), then
joined onto a long team-game table (`data/processed/game_features.parquet`,
`game_type == "REG"`, pushes dropped since `home_cover` is undefined for
them) to get cover rate in the most- (quartile 4) and least- (quartile 1)
penalized bins. This reproduces 49.90% (Q4) vs 50.56% (Q1) on 4,085 team-game
rows (2,016 + 2,069) -- both cover rates within 0.07 points of the recorded
49.85%/50.52%, and the gap magnitude (0.655 vs the recorded 0.67) matches to
within the ~0.3-point reconstruction tolerance this session was given. The
game count (4,085 team-game rows in the pair; 4,069 unique games) sits about
8% under the recorded "~4,431" -- plausibly a different treatment of pushes,
playoff games, or per-season vs. pooled quartile cuts in whatever produced
the original number, since no script behind it survives to check against.
That gap is reported, not hidden, and does not change the conclusion: the
point estimate reproduces.

**Scale**, following the `hc_year_one_fade` precedent in the same registry:
the quartile-pair comparison only "fires" on the subset of the slate sitting
in the two extreme quartiles (roughly half of it, since 2 of 4 quartiles are
used), so the raw quartile gap is multiplied by that fraction before being
recorded as `effect`, exactly as `hc_year_one_fade` scaled its raw 4.26-point
subset gap by 17.7% before recording 0.7528. The raw (unscaled) quartile gap
and the scaled full-slate effect are BOTH reported below; the old registry
entry's 0.67 was the unscaled number.

**Uncertainty.** Two week-blocked (`season*100 + week` block ids, the
convention `estvar_f_lever_splithalf.py` / `surgical_cfb_recipe_validation.py`
use elsewhere in this repo) and season-blocked (block id = `season`) bootstraps
of the SAME comparison, 20,000 resamples each, jointly resampling blocks so a
single draw's Q1 mean and Q4 mean come from the identical set of resampled
blocks (this project's ICC is hardcoded zero -- AGENTS.md, "within-week
correlation is zero" -- so week-blocking is not compensating for a real
correlation; it is kept because it is this project's established convention
for these intervals). `probability_positive` is the fraction of draws
favouring the candidate hypothesis (least-penalized team covers more), with
the same sign orientation the registry already uses (positive favours the
candidate).

Writes JSON to the scratchpad only; never touches `artifacts/` or any
registry file. `registry/weak_signals.json` is written separately, by hand,
via `nfl-ats weak-signals record --replace`, using this script's printed
numbers -- this script does not call that CLI itself.
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

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot  # noqa: E402
from nfl_ats.provenance import write_stamped_artifact  # noqa: E402

OUT_DIR = Path(
    r"C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3"
    r"\c8c5fbdd-027f-438d-b992-979e83a91c2e\scratchpad\penalty_discipline_interval"
)
DEFAULT_PBP_RAW_ROOT = REPO / "data/pbp/raw"
DEFAULT_FEATURES = REPO / "data/processed/game_features.parquet"

RECORDED_RATE_MEAN = 0.0750
RECORDED_RATE_SD = 0.0101
RECORDED_RELIABILITY = 0.261
RECORDED_Q4_COVER = 0.4985
RECORDED_Q1_COVER = 0.5052
RECORDED_SAMPLE_GAMES = 4431

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260818


def _canonical(team: pd.Series) -> pd.Series:
    return team.map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def team_season_penalty_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    """Reproduce ROADMAP's 0.0750 +/- 0.0101, reliability +0.261.

    ``mean(penalty)`` over every raw regular-season play (no play-type
    filter) where ``posteam == team``. See module docstring for why this
    definition, and not a filtered or defense-side one, is the one that
    reproduces the recorded figures.
    """

    plays = pbp.loc[pbp["posteam"].notna()].copy()
    plays["penalty"] = pd.to_numeric(plays["penalty"], errors="coerce").fillna(0.0)
    plays["team"] = _canonical(plays["posteam"])
    grouped = plays.groupby(["season", "team"]).agg(
        plays=("penalty", "size"), penalties=("penalty", "sum")
    )
    grouped["rate"] = grouped["penalties"] / grouped["plays"]
    return grouped.reset_index()


def year_over_year_reliability(rate: pd.DataFrame) -> tuple[float, int]:
    """Pearson correlation of a team's rate in season t against season t+1."""

    ordered = rate.sort_values(["team", "season"]).copy()
    ordered["next_rate"] = ordered.groupby("team")["rate"].shift(-1)
    ordered["next_season"] = ordered.groupby("team")["season"].shift(-1)
    pairs = ordered.loc[ordered["next_season"] == ordered["season"] + 1]
    return float(pairs["rate"].corr(pairs["next_rate"])), len(pairs)


def lag_and_quartile(rate: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, season) with the PRIOR season's rate and its global quartile.

    Requires an observed, contiguous prior season for that franchise (2009 is
    always dropped, since no team has a locally-observed 2008). Quartiles are
    cut once, globally, across every lagged team-season -- not re-cut within
    each season -- matching ROADMAP's plain "quartile split of teams by
    prior-season penalty rate" with no mention of an era control.
    """

    ordered = rate.sort_values(["team", "season"]).copy()
    ordered["prev_rate"] = ordered.groupby("team")["rate"].shift(1)
    ordered["prev_season"] = ordered.groupby("team")["season"].shift(1)
    lagged = ordered.loc[ordered["season"] - ordered["prev_season"] == 1].copy()
    lagged["quartile"] = pd.qcut(lagged["prev_rate"], 4, labels=[1, 2, 3, 4]).astype(int)
    return lagged[["team", "season", "prev_rate", "quartile"]]


def build_team_game_table(features: pd.DataFrame) -> pd.DataFrame:
    """Long team-game table: REG season only, pushes dropped, week-block id attached."""

    reg = features.loc[features["game_type"] == "REG"].copy()
    reg["home_team"] = _canonical(reg["home_team"])
    reg["away_team"] = _canonical(reg["away_team"])
    columns = ["game_id", "season", "week", "home_team", "away_team", "home_cover", "ats_margin"]

    home_rows = reg[columns].copy()
    home_rows["team"] = home_rows["home_team"]
    home_rows["team_covered"] = home_rows["home_cover"]
    home_rows["team_ats_margin"] = home_rows["ats_margin"]

    away_rows = reg[columns].copy()
    away_rows["team"] = away_rows["away_team"]
    away_rows["team_covered"] = 1.0 - away_rows["home_cover"]
    away_rows["team_ats_margin"] = -away_rows["ats_margin"]

    long = pd.concat([home_rows, away_rows], ignore_index=True)
    long = long.loc[long["team_covered"].notna()].copy()  # pushes: home_cover is NaN
    long["week_block"] = long["season"].astype(int) * 100 + long["week"].astype(int)
    return long


def quartile_market_check(long: pd.DataFrame, lagged: pd.DataFrame) -> dict[str, Any]:
    merged = long.merge(lagged, on=["team", "season"], how="inner")
    by_quartile = merged.groupby("quartile")["team_covered"].agg(["mean", "count"])

    q1_cover = float(by_quartile.loc[1, "mean"])
    q4_cover = float(by_quartile.loc[4, "mean"])
    n_q1 = int(by_quartile.loc[1, "count"])
    n_q4 = int(by_quartile.loc[4, "count"])
    n_total = len(merged)
    n_pair = n_q1 + n_q4

    raw_gap_pct = (q1_cover - q4_cover) * 100.0  # positive favours the candidate hypothesis
    fraction_of_slate = n_pair / n_total
    scaled_effect_pct = raw_gap_pct * fraction_of_slate

    return {
        "merged": merged,
        "q1_cover": q1_cover,
        "q4_cover": q4_cover,
        "n_q1": n_q1,
        "n_q4": n_q4,
        "n_total": n_total,
        "n_pair": n_pair,
        "n_unique_games_pair": int(
            merged.loc[merged["quartile"].isin([1, 4]), "game_id"].nunique()
        ),
        "n_unique_games_total": int(merged["game_id"].nunique()),
        "raw_gap_pct": raw_gap_pct,
        "fraction_of_slate": fraction_of_slate,
        "scaled_effect_pct": scaled_effect_pct,
    }


def block_bootstrap_quartile_gap(
    pair_rows: pd.DataFrame, *, block_col: str, samples: int, seed: int
) -> np.ndarray:
    """Vectorized block bootstrap of ``100 * (Q1 cover - Q4 cover)``.

    Both quartiles' means for a given draw come from the SAME resampled set
    of blocks (one multinomial draw over the shared block ids), not two
    independently resampled draws -- the correct way to jointly bootstrap a
    two-group comparison sharing a blocking structure, and the direct
    two-group generalization of
    ``nfl_ats.estimation_variance.block_bootstrap_means``'s vectorized-sum
    trick (pinned there against an explicit resample loop).
    """

    blocks, block_index = np.unique(pair_rows[block_col].to_numpy(), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    block_count = len(blocks)
    values = pair_rows["team_covered"].to_numpy(dtype=np.float64)

    sums: dict[int, np.ndarray] = {}
    counts: dict[int, np.ndarray] = {}
    for quartile in (1, 4):
        mask = pair_rows["quartile"].to_numpy() == quartile
        sums[quartile] = np.bincount(
            block_index[mask], weights=values[mask], minlength=block_count
        ).astype(np.float64)
        counts[quartile] = np.bincount(block_index[mask], minlength=block_count).astype(np.float64)
        if np.any(counts[quartile] == 0):
            empty = int(np.sum(counts[quartile] == 0))
            print(
                f"  note: {empty}/{block_count} {block_col} blocks have zero "
                f"quartile-{quartile} rows",
                flush=True,
            )

    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)
    mean_q1 = (drawn @ sums[1]) / (drawn @ counts[1])
    mean_q4 = (drawn @ sums[4]) / (drawn @ counts[4])
    return (mean_q1 - mean_q4) * 100.0


def interval_summary(
    draws_pct: np.ndarray, *, fraction_of_slate: float, confidence: float = 0.95
) -> dict[str, float]:
    scaled = draws_pct * fraction_of_slate
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": float(np.mean(scaled)),
        "lower": float(np.quantile(scaled, tail)),
        "upper": float(np.quantile(scaled, 1.0 - tail)),
        "standard_error": float(np.std(scaled, ddof=1)),
        "probability_positive": float(np.mean(scaled > 0.0)),
        "samples": len(scaled),
    }


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    print("=== Step 1: reproduce the recorded team penalty rate (0.0750 +/- 0.0101, r=+0.261) ===")
    snapshot = latest_pbp_snapshot(DEFAULT_PBP_RAW_ROOT)
    pbp = load_pbp_snapshot(snapshot, include_postseason=False)
    print(f"snapshot={snapshot.snapshot_id} rows={len(pbp)}")
    rate = team_season_penalty_rate(pbp)
    mean_rate = float(rate["rate"].mean())
    sd_rate = float(rate["rate"].std())
    reliability, n_pairs = year_over_year_reliability(rate)
    results["rate_reproduction"] = {
        "mean": mean_rate,
        "sd": sd_rate,
        "reliability": reliability,
        "reliability_pairs": n_pairs,
        "team_seasons": len(rate),
        "recorded_mean": RECORDED_RATE_MEAN,
        "recorded_sd": RECORDED_RATE_SD,
        "recorded_reliability": RECORDED_RELIABILITY,
    }
    print(
        f"mean={mean_rate:.5f} (recorded {RECORDED_RATE_MEAN}) "
        f"sd={sd_rate:.5f} (recorded {RECORDED_RATE_SD}) "
        f"reliability={reliability:.4f} (recorded +{RECORDED_RELIABILITY}) "
        f"on {n_pairs} year-over-year pairs"
    )

    print("\n=== Step 2: quartile market check ===")
    lagged = lag_and_quartile(rate)
    features = pd.read_parquet(DEFAULT_FEATURES)
    long = build_team_game_table(features)
    check = quartile_market_check(long, lagged)
    results["market_check"] = {key: value for key, value in check.items() if key != "merged"} | {
        "recorded_q4_cover": RECORDED_Q4_COVER,
        "recorded_q1_cover": RECORDED_Q1_COVER,
        "recorded_sample_games": RECORDED_SAMPLE_GAMES,
    }
    print(
        f"Q1 (least penalized) cover: {check['q1_cover']:.4f} on {check['n_q1']} team-games "
        f"(recorded {RECORDED_Q1_COVER})"
    )
    print(
        f"Q4 (most penalized) cover:  {check['q4_cover']:.4f} on {check['n_q4']} team-games "
        f"(recorded {RECORDED_Q4_COVER})"
    )
    print(
        f"raw gap (Q1-Q4): {check['raw_gap_pct']:+.4f} points "
        f"(recorded +0.67); n_pair={check['n_pair']} team-games / "
        f"{check['n_unique_games_pair']} unique games "
        f"(recorded ~{RECORDED_SAMPLE_GAMES}); n_total universe (4 quartiles)="
        f"{check['n_total']} team-games / {check['n_unique_games_total']} unique games"
    )
    print(
        f"fraction of slate the quartile pair represents: {check['fraction_of_slate']:.4f}\n"
        f"full-slate-scaled effect: {check['scaled_effect_pct']:+.4f} points"
    )

    pair_rows = check["merged"].loc[check["merged"]["quartile"].isin([1, 4])].copy()

    print("\n=== Step 3: week-blocked bootstrap (primary), 20,000 resamples ===")
    week_draws = block_bootstrap_quartile_gap(
        pair_rows, block_col="week_block", samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    n_week_blocks = int(pair_rows["week_block"].nunique())
    week_summary = interval_summary(week_draws, fraction_of_slate=check["fraction_of_slate"])
    week_summary["block_count"] = n_week_blocks
    results["week_blocked"] = week_summary
    print(
        f"blocks={n_week_blocks} estimate={week_summary['estimate']:+.4f} "
        f"95% [{week_summary['lower']:+.4f}, {week_summary['upper']:+.4f}] "
        f"se={week_summary['standard_error']:.4f} "
        f"P+={week_summary['probability_positive']:.4f}"
    )

    print("\n=== Step 4: season-blocked bootstrap (secondary), 20,000 resamples ===")
    season_draws = block_bootstrap_quartile_gap(
        pair_rows, block_col="season", samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    n_season_blocks = int(pair_rows["season"].nunique())
    season_summary = interval_summary(season_draws, fraction_of_slate=check["fraction_of_slate"])
    season_summary["block_count"] = n_season_blocks
    results["season_blocked"] = season_summary
    print(
        f"blocks={n_season_blocks} estimate={season_summary['estimate']:+.4f} "
        f"95% [{season_summary['lower']:+.4f}, {season_summary['upper']:+.4f}] "
        f"se={season_summary['standard_error']:.4f} "
        f"P+={season_summary['probability_positive']:.4f}"
    )

    results["elapsed_seconds"] = time.time() - started
    output_path = OUT_DIR / "results.json"
    write_stamped_artifact(results, output_path)  # ENG-38
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
