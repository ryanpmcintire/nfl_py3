"""LEAD-02: first-half/full-game script disagreement (gated on LEAD-60).

Mechanism (predeclared, ``docs/lead02_half_line_script.md``): a short
first-half spread relative to a big full-game spread implies a slow starter
who wins late and covers the full game less often. Predeclared direction:
**FADE the full-game favorite** on script disagreement (equivalently, back
the underdog). A cheap 2H sibling uses the same mechanism/direction on the
second-half leg instead of the first.

Source: the LEAD-60 half-line archive
(``artifacts/vegasinsider_backfill/20260822T033952Z/half_lines_<year>.parquet``,
docs/vi_half_lines.md), joined to the same run's full-game
``season_<year>.parquet`` on (capture, matchup, book) -- CLOSE-graded lines,
2005-2016 REG. Outcomes come from the newest local
``data/raw/<stamp>/schedules.parquet`` snapshot, matched on season/team/date
(the schedules snapshot itself only starts in 2009 -- seasons 2005-2008 are
therefore market-only in this screen, never scored; documented, not hidden).

**Encoding, frozen before any outcome was read:** disagreement = the half
spread divided by the full-game spread (both stored in this archive's
favorite-side convention, so both <= 0 in the plausible range and the ratio
is normally positive) is below the empirical 20th percentile of that ratio's
own distribution, measured over games where the full-game favorite is laying
3+ points. The cut is MEASURED from market data only (never touches
``home_cover``/``result``/schedule join output) and frozen before scoring --
see ``freeze_cut`` and its caller order in ``run_half_cell``.

**ENG-40 (fixed 2026-09-05, at the source):** 155 of 14,727 non-null
``spread_line`` values across the underlying ``season_<year>.parquet`` tidy
table were POSITIVE and in the 40-54 range -- e.g. away=LAC home=TEN, capture
20091226095259, ``spread_line=53.5`` at Caesars/Mirage -- a total (O/U) value
misfiled into the spread column by a board-layout ambiguity in
``scripts/backfill_vegasinsider.py::classify_line_tokens`` (VegasInsider
renders a cell's spread/total lines in either vertical order, and an
explicit "+"-signed total with no o/u marker, e.g. "+54", used to satisfy the
spread-token regex before the real spread token was ever seen). That parser
is now fixed (a sign-convention rule: a "+"-prefixed token can never be this
archive's favorite-only spread, so it is routed to the total instead -- see
that function's docstring) and all 12 seasons were rebuilt from the same
cached HTML; MEASURED 2026-09-05: zero rows in the rebuilt archive have a
positive or >30-magnitude full-game ``spread_line`` (checked over every
``season_<year>.parquet`` file). ``filter_plausible`` no longer needs a
full-game-leg guard.

**Residual, UNRELATED, half-leg-only guard (measured 2026-09-05, kept):** one
row (1H) still has an implausible HALF spread even after the ENG-40 fix --
away=IND home=TEN, capture 20111026112359, book HARRAH'S,
``full_spread=-9.0`` but ``half1_spread=-31.5`` (a half spread numerically
LARGER than the full-game spread, impossible for a real quote). Read
directly from the cached line-movement HTML
(``data/raw/vegasinsider/20260822T033952Z/line_movement/20111026112359_bbcbd8fd.html``):
VegasInsider's own page shows "TEN-31.5" / "IND+31.5" verbatim in the raw
1H-Fav/1H-Dog cells of its last three movement rows for that book -- this is
NOT a parser defect (``extract_book_half_lines`` reads exactly what the
source page shows), it is an anomalous/erroneous quote baked into the
archive's own source HTML, on the LEAD-60 half-line code path
(``build_half_lines``/``extract_book_half_lines``), which is a different
function from the ENG-40 bug this module's parser fix addressed and is out
of that fix's scope. ``filter_plausible`` keeps a narrow half-leg-only
magnitude guard for this one confirmed case rather than silently passing an
impossible value into the bootstrap.

This is a **lead-generation screen on a CLOSE-graded archive**
(docs/rotation_registry.md rule 8): no rotation window is opened or spent.
Method mirrors ``scripts/nfl_bias_battery_screen.py`` (subset-vs-complement
week-blocked bootstrap, full-slate scaling, era split) plus a positive
control (a flag planted directly from each game's own realized ATS margin,
matched to the real cell's flagged fraction, proving what this population
size CAN show for a known real effect) and a 200-draw within-week
permutation null.

Closing-grounds taxonomy (binding, AGENTS.md, restated verbatim because this
script's own verdicts must never be phrased as a rejection): an interval or
CI that contains zero is NEVER grounds to reject, fail, or close an
experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line
of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is ``unresolved_below_power``: record it with ``nfl-ats weak-signals
record``, report ``probability_positive``, never the binary "contains zero".
This script never writes ``registry/weak_signals.json`` itself -- it writes
its own run artifact plus an experiment-registry row via
``write_experiment_artifact``; the weak-signal ledger entries are recorded
separately by explicit ``nfl-ats weak-signals record`` invocations after
this artifact is inspected.
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

from _common import block_bootstrap_two_group, default_schedules  # noqa: E402
from vi_dispersion_screen import vi_to_sched  # noqa: E402

from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.io import atomic_parquet  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

RUN_ID = "20260822T033952Z"
DEFAULT_BACKFILL = REPO / "artifacts" / "vegasinsider_backfill" / RUN_ID
YEARS: tuple[int, ...] = tuple(range(2005, 2017))
# Board book-identity fallback rate 0.643 for 2006 (docs/vegasinsider_backfill.md
# "Reduced-confidence flag"), the only season over the >20% threshold. Both the
# full-game and half legs used here derive from the same board/movement pages,
# so this exclusion applies to both, consistent with scripts/vi_dispersion_screen.py.
EXCLUDED_REDUCED_CONFIDENCE_SEASONS = frozenset({2006})

FAVORITE_MIN = 3.0
RATIO_PERCENTILE = 0.20
PLAUSIBLE_MAX_ABS_SPREAD = 30.0

ERA_SPLITS: tuple[tuple[str, int, int], ...] = (
    ("2005_2010", 2005, 2010),
    ("2011_2016", 2011, 2016),
)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260905
NULL_DRAWS = 200
NULL_SEED = 20260905

JOIN_KEYS: list[str] = ["capture_ts", "game_date", "away", "home", "book"]
SCHEDULE_COLUMNS: list[str] = [
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
    "result",
    "spread_line",
    "home_score",
    "away_score",
]


def load_full_game(backfill_dir: Path) -> pd.DataFrame:
    """Full-game tidy rows, all seasons found, tagged with the file's own
    season and the reduced-confidence exclusion applied."""

    frames = []
    for year in YEARS:
        path = backfill_dir / f"season_{year}.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        frame["season"] = year
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True)
    full["spread_line"] = pd.to_numeric(full["spread_line"], errors="coerce")
    full = full.loc[~full["season"].isin(EXCLUDED_REDUCED_CONFIDENCE_SEASONS)].copy()
    full = full.drop_duplicates(JOIN_KEYS).rename(columns={"spread_line": "full_spread"})
    return full.reset_index(drop=True)


def load_half_lines(backfill_dir: Path) -> pd.DataFrame:
    """Half-line rows (``half`` in {1, 2}), all seasons found."""

    frames = []
    for year in YEARS:
        path = backfill_dir / f"half_lines_{year}.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        frame["season"] = year
        frames.append(frame)
    half = pd.concat(frames, ignore_index=True)
    if "in_play" in half:
        half = half.loc[half["in_play"].eq(False)].copy()
    half["spread_line"] = pd.to_numeric(half["spread_line"], errors="coerce")
    half = half.loc[~half["season"].isin(EXCLUDED_REDUCED_CONFIDENCE_SEASONS)].copy()
    return half.reset_index(drop=True)


def join_half_leg(full: pd.DataFrame, half: pd.DataFrame, half_num: int) -> pd.DataFrame:
    """Inner-join on (capture, matchup, book): the population is games with
    BOTH a usable full-game closing spread and a usable half-``half_num``
    spread from the same capture and book."""

    leg = (
        half.loc[half["half"] == half_num, [*JOIN_KEYS, "spread_line"]]
        .drop_duplicates(JOIN_KEYS)
        .rename(columns={"spread_line": f"half{half_num}_spread"})
    )
    merged = full.merge(leg, on=JOIN_KEYS, how="inner", validate="one_to_one")
    merged = merged.dropna(subset=["full_spread", f"half{half_num}_spread"])
    return merged.reset_index(drop=True)


def filter_plausible(merged: pd.DataFrame, half_num: int) -> tuple[pd.DataFrame, int]:
    """Drop rows with an implausible HALF spread (see module docstring: one
    measured, source-HTML-confirmed anomalous quote, unrelated to ENG-40).
    ENG-40's full-game-leg guard was removed 2026-09-05 once the underlying
    parser fix made it a structural no-op (measured: zero rows in the
    rebuilt archive violate the full-game favorite-side/magnitude
    convention)."""

    half_col = f"half{half_num}_spread"
    plausible = merged.loc[merged[half_col].abs() <= PLAUSIBLE_MAX_ABS_SPREAD].copy()
    dropped = len(merged) - len(plausible)
    return plausible.reset_index(drop=True), dropped


def eligible_favorites(plausible: pd.DataFrame) -> pd.DataFrame:
    """Restrict to full-game favorites of 3+ points -- the predeclared scope
    the ratio and its frozen cut are defined over."""

    return plausible.loc[plausible["full_spread"] <= -FAVORITE_MIN].copy().reset_index(drop=True)


def dedup_to_one_row_per_game(fav: pd.DataFrame) -> pd.DataFrame:
    """One row per real game (independence unit for the week-blocked
    bootstrap): keep the LATEST capture_ts among usable (capture, book)
    instances -- the best available proxy for a close read this archive
    offers (no true closing timestamp is recorded, only Wayback capture
    time)."""

    ordered = fav.sort_values("capture_ts")
    deduped = ordered.drop_duplicates(["season", "game_date", "away", "home"], keep="last")
    return deduped.reset_index(drop=True)


def compute_ratio(df: pd.DataFrame, half_num: int) -> pd.Series:
    return df[f"half{half_num}_spread"] / df["full_spread"]


def freeze_cut(ratio: pd.Series, percentile: float = RATIO_PERCENTILE) -> float:
    """The frozen disagreement cut. Computed from the ratio series ONLY --
    this function never sees an outcome column, by construction, which is
    the leakage guarantee: nothing this function returns can depend on
    ``home_cover``/``result``."""

    return float(ratio.quantile(percentile))


def apply_flag(df: pd.DataFrame, half_num: int, cut: float) -> pd.DataFrame:
    """Attach ``ratio``/``flag`` columns. ``flag`` is a pure function of
    ``full_spread``/``half<n>_spread`` (market data) and ``cut`` (frozen
    before any outcome was read) -- it cannot change if a game's outcome
    changes, which ``tests/test_lead02_half_line_script.py`` checks
    directly."""

    out = df.copy()
    out["ratio"] = compute_ratio(out, half_num)
    out["flag"] = out["ratio"] < cut
    return out


def build_schedule_index(schedules_path: Path) -> dict[tuple[int, str, str], list[Any]]:
    sched = pd.read_parquet(schedules_path, columns=SCHEDULE_COLUMNS)
    sched = sched.loc[sched["game_type"] == "REG"].copy()
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    sched["season"] = pd.to_numeric(sched["season"], errors="raise").astype(int)
    index: dict[tuple[int, str, str], list[Any]] = {}
    for row in sched.itertuples(index=False):
        index.setdefault((int(row.season), row.away_team, row.home_team), []).append(row)
    return index


def match_game(
    index: dict[tuple[int, str, str], list[Any]], season: int, away: str, home: str, game_date: str
) -> Any | None:
    """Season/team/date match (+/-1 day) against the schedule index. Team
    codes are normalized through the same ``vi_to_sched`` mapping
    ``scripts/vi_dispersion_screen.py`` already established for this
    archive (LAR/LAC/LV relocation-era aliases). Returns ``None`` on zero or
    multiple candidates -- ambiguous matches are dropped, never guessed."""

    away_s = vi_to_sched(away, season)
    home_s = vi_to_sched(home, season)
    candidates = index.get((season, away_s, home_s), [])
    game_day = pd.Timestamp(game_date)
    hits = [row for row in candidates if abs((pd.Timestamp(row.gameday) - game_day).days) <= 1]
    if len(hits) == 1:
        return hits[0]
    return None


def attach_schedule_outcomes(
    flagged: pd.DataFrame, index: dict[tuple[int, str, str], list[Any]]
) -> tuple[pd.DataFrame, int]:
    """Join each flagged (deduped) game to its schedule outcome. Returns the
    matched rows plus a count of unmatched games."""

    rows: list[dict[str, Any]] = []
    n_unmatched = 0
    for row in flagged.itertuples(index=False):
        match = match_game(index, int(row.season), str(row.away), str(row.home), str(row.game_date))
        if match is None:
            n_unmatched += 1
            continue
        rows.append(
            {
                "season": int(match.season),
                "week": int(match.week),
                "game_id": match.game_id,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "spread_line": float(match.spread_line) if pd.notna(match.spread_line) else np.nan,
                "result": float(match.result) if pd.notna(match.result) else np.nan,
                "flag": bool(row.flag),
                "ratio": float(row.ratio),
                "full_spread": float(row.full_spread),
                "capture_ts": row.capture_ts,
                "game_date": row.game_date,
                "book": row.book,
            }
        )
    return pd.DataFrame(rows), n_unmatched


def add_dog_outcome(outcome_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``home_is_favorite``/``dog_covered``/``dog_margin`` from the
    SCHEDULE's own (standard, home-signed) ``spread_line`` -- the VI archive
    cannot encode favorite side on its own (docs/vegasinsider_pilot.md), so
    side attribution always comes from the schedule join, never from VI."""

    out = add_ats_outcomes(outcome_df)
    out["home_is_favorite"] = out["spread_line"] < 0
    out["dog_covered"] = np.where(
        out["home_is_favorite"], 1.0 - out["home_cover"], out["home_cover"]
    )
    out["dog_margin"] = np.where(out["home_is_favorite"], -out["ats_margin"], out["ats_margin"])
    return out


def summarize(
    df: pd.DataFrame, *, flag_col: str, value_col: str, block_col: str, samples: int, seed: int
) -> dict[str, Any]:
    """Subset-vs-complement week-blocked bootstrap with full-slate scaling,
    mirroring ``scripts/nfl_bias_battery_screen.py::summarize_population``
    (direction is fixed here: ``flag=True`` always predicts a HIGHER
    ``value_col`` mean, i.e. sign is always +1 -- ``value_col`` is already
    oriented to the predeclared FADE-the-favorite direction)."""

    n_total = len(df)
    flag = df[flag_col].astype(bool)
    n_flag = int(flag.sum())
    n_complement = n_total - n_flag
    if n_total == 0 or n_flag == 0 or n_complement == 0:
        return {
            "n_total": n_total,
            "n_flag": n_flag,
            "n_complement": n_complement,
            "insufficient_data": True,
        }

    work = df.copy()
    work["_flag"] = flag.to_numpy()
    subset_mean = float(work.loc[work["_flag"], value_col].mean())
    complement_mean = float(work.loc[~work["_flag"], value_col].mean())
    raw_gap_pts = (subset_mean - complement_mean) * 100.0
    fraction_of_slate = n_flag / n_total
    full_slate_effect_pts = raw_gap_pts * fraction_of_slate

    draws = block_bootstrap_two_group(
        work, flag_col="_flag", value_col=value_col, block_col=block_col, samples=samples, seed=seed
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
        "n_week_blocks": int(work[block_col].nunique()),
        "subset_mean": subset_mean,
        "complement_mean": complement_mean,
        "raw_gap_pts": raw_gap_pts,
        "fraction_of_slate": fraction_of_slate,
        "full_slate_effect_pts": full_slate_effect_pts,
        "week_blocked_ci95_scaled": [float(lower), float(upper)],
        "probability_positive": float(np.mean(draws > 0)) if len(draws) else float("nan"),
        "bootstrap_samples": samples,
        "dropped_draws": int(dropped),
        "insufficient_data": False,
    }


def positive_control_flag(df: pd.DataFrame, margin_col: str, n_flag: int) -> pd.Series:
    """Plant a control flag directly from each game's own REALIZED ats
    margin: the ``n_flag`` games (matched to the real cell's flagged count)
    where the underdog beat the closing spread by the MOST. This is an
    oracle by construction (it reads the outcome to build the flag) -- its
    purpose is not to predict anything, it is to measure what THIS
    population size and week-block structure CAN show for a real effect at
    least this large, i.e. a positive-control power check, never a claim
    about the real (market-only) flag."""

    ranked = df[margin_col].rank(method="first", ascending=False)
    return (ranked <= n_flag).astype(bool)


def within_block_permutation_null(
    df: pd.DataFrame, *, flag_col: str, value_col: str, block_col: str, draws: int, seed: int
) -> dict[str, Any]:
    """Permute which rows carry ``flag_col`` WITHIN each ``block_col`` group
    (preserving each block's flagged count and slate composition), recompute
    the raw subset-minus-complement gap each draw. Not zero-centred by
    construction (uneven block composition), matching the caveat already
    established for this project's other within-block permutation nulls.

    Reports ``n_blocks_multi_row``/``n_rows_in_multi_row_blocks`` alongside
    the null: a block of size 1 can never be permuted (there is nothing to
    swap it with), so a population this sparse relative to its own block
    count can leave the null near-degenerate (every draw reproduces the
    observed gap) even though nothing is wrong with the procedure -- that
    is a measured property of the population size, reported rather than
    hidden, per this run's own diagnostic fields."""

    df = df.reset_index(drop=True)
    values = df[value_col].to_numpy(dtype=float)
    flags = df[flag_col].to_numpy(dtype=bool)
    n_flag = int(flags.sum())
    if n_flag == 0 or n_flag == len(flags):
        observed = float("nan")
    else:
        observed = float(values[flags].mean() - values[~flags].mean()) * 100.0

    block_positions = df.groupby(block_col, sort=False).indices
    n_blocks_multi_row = sum(1 for positions in block_positions.values() if len(positions) >= 2)
    n_rows_in_multi_row_blocks = sum(
        len(positions) for positions in block_positions.values() if len(positions) >= 2
    )
    rng = np.random.default_rng(seed)
    null_gaps: list[float] = []
    for _ in range(draws):
        permuted = flags.copy()
        for positions in block_positions.values():
            positions = np.asarray(positions)
            permuted[positions] = rng.permutation(flags[positions])
        if 0 < permuted.sum() < len(permuted):
            sub = float(values[permuted].mean())
            comp = float(values[~permuted].mean())
            null_gaps.append((sub - comp) * 100.0)
    arr = np.asarray(null_gaps, dtype=float)
    return {
        "block_col": block_col,
        "n_blocks_total": len(block_positions),
        "n_blocks_multi_row": n_blocks_multi_row,
        "n_rows_in_multi_row_blocks": n_rows_in_multi_row_blocks,
        "observed_raw_gap_pts": observed,
        "null_mean": float(arr.mean()) if len(arr) else float("nan"),
        "null_sd": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
        "null_lower95": float(np.quantile(arr, 0.025)) if len(arr) else float("nan"),
        "null_upper95": float(np.quantile(arr, 0.975)) if len(arr) else float("nan"),
        "share_at_or_beyond_observed_abs": (
            float(np.mean(np.abs(arr) >= abs(observed)))
            if len(arr) and np.isfinite(observed)
            else float("nan")
        ),
        "draws_requested": draws,
        "draws_used": len(arr),
        "seed": seed,
        "note": (
            "within-block flag-label permutation; not zero-centred by construction "
            "(small/uneven block composition); a block with exactly 1 row cannot be "
            "permuted at all, so a near-degenerate null (all draws equal the observed "
            "gap) is expected, not a bug, when n_blocks_multi_row is small relative to "
            "n_blocks_total -- see the sibling season-blocked supplementary null for a "
            "block structure with real swap room in this population"
        ),
    }


def run_half_cell(
    half_num: int,
    full: pd.DataFrame,
    half: pd.DataFrame,
    sched_index: dict[tuple[int, str, str], list[Any]],
    *,
    samples: int,
    seed: int,
    null_draws: int,
    null_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    merged = join_half_leg(full, half, half_num)
    plausible, n_implausible_dropped = filter_plausible(merged, half_num)
    fav = eligible_favorites(plausible)
    dedup = dedup_to_one_row_per_game(fav)

    # --- predeclared: freeze the ratio cut from MARKET DATA ONLY ---
    ratio = compute_ratio(dedup, half_num)
    cut = freeze_cut(ratio)
    flagged = apply_flag(dedup, half_num, cut)

    # --- only now does an outcome column enter the picture ---
    outcome_df, n_unmatched = attach_schedule_outcomes(flagged, sched_index)
    n_matched = len(outcome_df)
    outcome_df = add_dog_outcome(outcome_df)
    n_pushes = int(outcome_df["dog_covered"].isna().sum())
    scored = outcome_df.dropna(subset=["dog_covered"]).reset_index(drop=True)
    scored["week_block"] = scored["season"] * 100 + scored["week"]

    full_period = summarize(
        scored,
        flag_col="flag",
        value_col="dog_covered",
        block_col="week_block",
        samples=samples,
        seed=seed,
    )

    era_results: dict[str, Any] = {}
    for era_label, start, end in ERA_SPLITS:
        era_df = scored.loc[scored["season"].between(start, end)].reset_index(drop=True)
        if era_df.empty:
            era_results[era_label] = {"n_total": 0, "n_flag": 0, "insufficient_data": True}
            continue
        era_results[era_label] = summarize(
            era_df,
            flag_col="flag",
            value_col="dog_covered",
            block_col="week_block",
            samples=samples,
            seed=seed,
        )

    n_flag_scored = int(scored["flag"].sum())
    control_df = scored.copy()
    if n_flag_scored:
        control_df["control_flag"] = positive_control_flag(control_df, "dog_margin", n_flag_scored)
        positive_control = summarize(
            control_df,
            flag_col="control_flag",
            value_col="dog_covered",
            block_col="week_block",
            samples=samples,
            seed=seed,
        )
    else:
        positive_control = {
            "insufficient_data": True,
            "reason": "no flagged games survived the schedule join",
        }

    permutation = within_block_permutation_null(
        scored,
        flag_col="flag",
        value_col="dog_covered",
        block_col="week_block",
        draws=null_draws,
        seed=null_seed,
    )
    # Supplementary: this population's week blocks are mostly size 1 (a game
    # every ~1.1-1.9 weeks on average per flagged/unflagged pairing), which
    # leaves the required week-blocked null above near-degenerate -- season
    # blocks give real swap room and are reported alongside it, never in
    # place of it.
    permutation_season_supplementary = within_block_permutation_null(
        scored,
        flag_col="flag",
        value_col="dog_covered",
        block_col="season",
        draws=null_draws,
        seed=null_seed,
    )

    cell = {
        "half_num": half_num,
        "direction": "fade_full_game_favorite" if half_num == 1 else "back_favorable_2h_underdog",
        "frozen_ratio_cut": cut,
        "ratio_distribution_summary": {
            "n": len(ratio),
            "mean": float(ratio.mean()) if len(ratio) else float("nan"),
            "median": float(ratio.median()) if len(ratio) else float("nan"),
            "p20_frozen_cut": cut,
            "min": float(ratio.min()) if len(ratio) else float("nan"),
            "max": float(ratio.max()) if len(ratio) else float("nan"),
            "n_sign_flip_negative_ratio": int((ratio < 0).sum()),
        },
        "population": {
            "raw_joined_rows_both_spreads": len(merged),
            "implausible_dropped": n_implausible_dropped,
            "favorites_3plus_rows": len(fav),
            "deduped_games_market_only": len(dedup),
            "matched_to_schedule": n_matched,
            "unmatched": n_unmatched,
            "schedule_match_rate": (n_matched / len(dedup)) if len(dedup) else float("nan"),
            "pushes_dropped": n_pushes,
            "scored_games": len(scored),
            "flagged_scored_games": n_flag_scored,
            "archive_half_vs_full_join_rate_documented": 0.436,
        },
        "full_period": full_period,
        "era_split": era_results,
        "positive_control_planted_from_realized_margin": positive_control,
        "within_week_permutation_null": permutation,
        "within_season_permutation_null_supplementary": permutation_season_supplementary,
    }
    return cell, scored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill", type=Path, default=DEFAULT_BACKFILL)
    parser.add_argument("--half-backfill", type=Path, default=None)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--null-draws", type=int, default=NULL_DRAWS)
    parser.add_argument("--null-seed", type=int, default=NULL_SEED)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    schedules_path: Path = args.schedules or default_schedules()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "lead02_half_line_script" / timestamp)

    full = load_full_game(args.backfill)
    half = load_half_lines(args.half_backfill or args.backfill)
    sched_index = build_schedule_index(schedules_path)

    cells: dict[str, Any] = {}
    for half_num in (1, 2):
        print(f"\n=== half{half_num} ===")
        cell, scored = run_half_cell(
            half_num,
            full,
            half,
            sched_index,
            samples=args.samples,
            seed=args.seed,
            null_draws=args.null_draws,
            null_seed=args.null_seed,
        )
        cells[f"half{half_num}"] = cell
        print(json.dumps(cell, indent=2, default=str))
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_parquet(scored, output_dir / f"scored_half{half_num}.parquet")

    configuration = {
        "command": "lead02-half-line-script-screen",
        "backfill_dir": str(args.backfill),
        "half_backfill_dir": str(args.half_backfill or args.backfill),
        "schedules": str(schedules_path),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "null_draws": args.null_draws,
        "null_seed": args.null_seed,
        "favorite_min": FAVORITE_MIN,
        "ratio_percentile": RATIO_PERCENTILE,
        "plausible_max_abs_spread": PLAUSIBLE_MAX_ABS_SPREAD,
        "era_splits": [list(era) for era in ERA_SPLITS],
        "excluded_reduced_confidence_seasons": sorted(EXCLUDED_REDUCED_CONFIDENCE_SEASONS),
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "predeclaration": "docs/lead02_half_line_script.md (frozen before scoring)",
        "results": cells,
        "instrument_note": (
            "VI spread_line carries no home/away orientation on its own "
            "(docs/vegasinsider_pilot.md); favorite side always comes from "
            "the schedule's own (home-signed) spread_line via the season/"
            "team/date join, never from the VI archive. Seasons 2005-2008 "
            "are market-only (local schedules snapshot starts 2009); season "
            "2006 excluded entirely (reduced-confidence book-identity "
            "fallback 0.64, both legs derive from the same board pages)."
        ),
        "provenance": artifact_provenance(configuration, schedules_path, project_root=REPO),
    }
    metrics = {
        "half1_scored_games": cells["half1"]["population"]["scored_games"],
        "half1_flagged": cells["half1"]["population"]["flagged_scored_games"],
        "half1_full_slate_effect_pts": cells["half1"]["full_period"].get("full_slate_effect_pts"),
        "half1_probability_positive": cells["half1"]["full_period"].get("probability_positive"),
        "half2_scored_games": cells["half2"]["population"]["scored_games"],
        "half2_flagged": cells["half2"]["population"]["flagged_scored_games"],
        "half2_full_slate_effect_pts": cells["half2"]["full_period"].get("full_slate_effect_pts"),
        "half2_probability_positive": cells["half2"]["full_period"].get("probability_positive"),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="lead02-half-line-script-screen",
        metrics=metrics,
        notes=(
            "Measure-only lead-generation screen gated on LEAD-60's half-line "
            "archive (CLOSE-graded, 2005-2016). Predeclared direction: FADE "
            "the full-game favorite / BACK the underdog on script "
            "disagreement, both 1H (LEAD-02) and 2H sibling cells. Ratio cut "
            "frozen at the empirical 20th percentile, market data only, "
            "before any outcome was read. Every scoreable cell records via "
            "nfl-ats weak-signals record as unresolved_below_power "
            "regardless of interval shape (AGENTS.md taxonomy) -- an "
            "interval containing zero is never grounds to close a line of "
            "work."
        ),
        source="scripts/lead02_half_line_script_screen.py",
        registry_root=output_dir / "experiment_registry" if args.half_backfill else None,
        project_root=REPO,
    )
    print(f"\nwrote {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
