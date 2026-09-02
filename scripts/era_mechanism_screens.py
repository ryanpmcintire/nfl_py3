"""Three predeclared era-mechanism cells (WP34, 2026-09-01).

``docs/era_magnitude_report.md`` (WP4, same session) proposed exactly three
unrun mechanism cells, one per sign-flipping era-split construct it found.
This script runs those three, and only those three, exactly as
``docs/era_mechanism_screens_20260901.md`` froze them BEFORE any cover rate,
effect, or sign was computed:

1. ``bye_overval_install_need_moderator`` -- does a bye week's (now
   practice-capped) extra preparation time still carry more marginal value
   for a home team with an installation need? Extends
   ``scripts/bye_overvaluation_screen.py``; the base flag is imported from
   it verbatim, never re-derived.
2. ``pt_post_mnf_sunday_changepoint`` -- the primetime battery's 2017/2018
   split is its own fixed convention with no rule change behind it, so:
   does the data itself locate a break, and where? Reuses
   ``scripts/era_magnitude_profile.py``'s Stage-2a changepoint machinery and
   ``scripts/primetime_cells_screen.py``'s flag builder.
3. ``sagarin_battery_large_divergence_coverage_matched_era`` (+ its
   predeclared late-era companion) -- the Sagarin divergence era split on a
   coverage-matched population, NOW on WP19's corrected 2012/2013 coverage.
   A NEW family (``sagarin_divergence_coverage_matched``) on a DIFFERENT
   (coverage-matched) population, never a re-score: no ``sagarin_battery_*``
   entry is read back, overwritten, or corrected by this script.

**Binding taxonomy this script and its output must respect (verbatim,
because a script has no access to AGENTS.md/CLAUDE.md's session context
injection):**

    An interval or CI that contains zero is NEVER grounds to reject, fail,
    or close an experiment. At this evaluator's ~2-point resolution,
    "contains zero" is the EXPECTED outcome for a real small signal. Only
    two grounds ever close a line of work: (1) refuted mechanism -- a
    RESOLVED wrong sign (whole interval on the wrong side of zero) or zero
    split-half reliability; (2) bounded by a positive control proven able
    to detect an effect that size. Everything else is
    ``unresolved_below_power``: record it with ``nfl-ats weak-signals
    record``, report ``probability_positive``, never the binary "contains
    zero". If a record command errors, the verdict is wrong, not the
    validator.

Also binding: "era magnitude, not presence" -- per-era magnitudes are
reported separately and a sign flip is never averaged away; within-week
correlation is ZERO, so no ICC is estimated or padded anywhere here.

**Mined-window discount**: all three cells run on the full 2009-2025 local
history, which includes the mined 2018-2025 seasons, and cells 1-2 reuse the
exact rows their parent batteries already scored. Every number here is
reinforcing evidence about an already-seen window, not an independent
out-of-sample vote, and none may be pooled with its parent battery's entries
as an independent input. These are screens, not play decisions.

**Measure-only.** This script never writes ``registry/weak_signals.json`` or
``registry/rotation_registry.json``; recording happens through separate
explicit ``nfl-ats weak-signals record`` calls. It DOES write its results
artifact and a low-stakes experiment-provenance stamp through
``nfl_ats.provenance.write_experiment_artifact`` -- a run log, not a verdict.

Usage::

    ./.tools/uv.exe run --no-sync python scripts/era_mechanism_screens.py \
        --cell <name|all> --mode <screen|null|positive-control>
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.append(str(REPO / "scripts"))

import bye_overvaluation_screen as bye_battery  # noqa: E402
import era_magnitude_profile as era_profile  # noqa: E402
import hc_year_one_fade as hc_module  # noqa: E402
import nfl_travel_rest_battery_screen as travel_battery  # noqa: E402
import primetime_cells_screen as pt_battery  # noqa: E402
import sagarin_divergence_battery as sagarin_battery  # noqa: E402

from nfl_ats.clv import week_blocked_bootstrap  # noqa: E402
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.pbp import latest_pbp_snapshot  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

CELL_BYE = "bye_overval_install_need_moderator"
CELL_PT = "pt_post_mnf_sunday_changepoint"
CELL_SAGARIN = "sagarin_battery_large_divergence_coverage_matched_era"
CELL_SAGARIN_LATE = "sagarin_battery_large_divergence_coverage_matched_era_late"
ALL_CELLS = (CELL_BYE, CELL_PT, CELL_SAGARIN)

# Frozen in docs/era_mechanism_screens_20260901.md section 1.
PRIMARY_SAMPLES = 20_000
NULL_DRAWS = 200
CONTROL_REPLICATES = 25
CONTROL_SAMPLES = 2_000
SEASON_BLOCK_FLOOR = 10
COVERAGE_MATCH_THRESHOLD_PCT = 80.0
BYE_INJECTION_TARGETS = (0.25, 0.50, 1.00, 2.00)
SAGARIN_INJECTION_TARGETS = (0.5, 1.0, 2.0, 4.0)

TAXONOMY = (
    "An interval or CI that contains zero is NEVER grounds to reject, fail, or close an "
    "experiment; only a RESOLVED wrong sign / zero split-half reliability (refuted mechanism) "
    "or a positive control proven able to detect an effect that size (bounded by control) ever "
    "closes a line of work. Everything else is unresolved_below_power, reported with "
    "probability_positive, never the binary 'contains zero'."
)
MINED_WINDOW_DISCOUNT = (
    "Runs on the full 2009-2025 local history, which INCLUDES the mined 2018-2025 seasons, and "
    "cells 1-2 reuse the exact rows their parent batteries already scored: reinforcing evidence "
    "about an already-seen window, not an independent out-of-sample vote. Never pool with the "
    "parent battery's entries as an independent input. Screens, not play decisions."
)


def _log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Shared uncertainty plumbing (frozen: clv.week_blocked_bootstrap, 20k draws)
# ---------------------------------------------------------------------------


def _n_blocks(frame: pd.DataFrame, block: str) -> int:
    columns = ["season", "week"] if block == "week" else ["season"]
    return int(frame.loc[:, columns].drop_duplicates().shape[0])


def _bootstrap(
    frame: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], dict[str, float]],
    *,
    block: str,
    samples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """``clv.week_blocked_bootstrap`` reshaped into ``{metric: {...}}``."""

    table = week_blocked_bootstrap(frame, metric_fn, block=block, samples=samples, seed=seed)
    blocks = _n_blocks(frame, block)
    out: dict[str, dict[str, Any]] = {}
    for row in table.to_dict(orient="records"):
        out[str(row["metric"])] = {
            "estimate": float(row["estimate"]),
            "lower": float(row["lower"]),
            "upper": float(row["upper"]),
            "probability_positive": float(row["probability_positive"]),
            "block": block,
            "samples": int(row["samples"]),
            "n_blocks": blocks,
            "degenerate_below_block_floor": bool(block == "season" and blocks < SEASON_BLOCK_FLOOR),
        }
    return out


def _excludes_zero(entry: dict[str, Any]) -> bool:
    return bool(entry["lower"] > 0.0 or entry["upper"] < 0.0)


def _within_week_permutation(
    frame: pd.DataFrame,
    *,
    label_col: str,
    statistic: Callable[[pd.DataFrame, np.ndarray], dict[str, float]],
    draws: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Permute ``label_col`` WITHIN each (season, week) block, ``draws`` times.

    The null is not assumed to be centred on zero: a within-week permutation
    preserves each week's own home tilt and slate composition, and this
    project has already measured such a null to sit off zero by construction
    (project MEMORY ``home-tilt-null-artifact``). One permutation run is a
    spread, never a test.
    """

    rng = np.random.default_rng(seed)
    block_indices = list(frame.groupby(["season", "week"], sort=False).indices.values())
    labels = frame[label_col].to_numpy()
    observed = statistic(frame, labels)
    collected: dict[str, list[float]] = {name: [] for name in observed}
    for _ in range(draws):
        permuted = labels.copy()
        for positions in block_indices:
            permuted[positions] = rng.permutation(labels[positions])
        drawn = statistic(frame, permuted)
        for name, value in drawn.items():
            collected[name].append(value)

    summary: dict[str, dict[str, Any]] = {}
    for name, values_list in collected.items():
        values = np.asarray(values_list, dtype=float)
        finite = values[np.isfinite(values)]
        obs = observed[name]
        summary[name] = {
            "observed": float(obs),
            "null_mean": float(np.mean(finite)) if finite.size else float("nan"),
            "null_lower_p2_5": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
            "null_upper_p97_5": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
            "share_null_at_or_above_observed": float(np.mean(finite >= obs))
            if finite.size
            else float("nan"),
            "share_null_abs_at_or_above_abs_observed": float(np.mean(np.abs(finite) >= abs(obs)))
            if finite.size
            else float("nan"),
            "requested_draws": draws,
            "draws": int(finite.size),
            "note": (
                "within-week permutation null; NOT zero-centred by design (home-tilt artifact). "
                "One permutation run is a spread, never a test. 'draws' counts the FINITE "
                "permutations; a draw whose shuffle empties an arm inside some block is dropped."
            ),
        }
    return summary


def _injection_control(
    frame: pd.DataFrame,
    *,
    response_col: str,
    flip_mask: np.ndarray,
    rows_per_point: float,
    targets: tuple[float, ...],
    metric_fn: Callable[[pd.DataFrame], dict[str, float]],
    metric_key: str,
    replicates: int,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Inject a known effect of a declared size, then run the identical estimator.

    ``rows_per_point`` is how many 0-response rows inside ``flip_mask`` must be
    flipped to 1 to move ``metric_key`` by one accuracy point (exact for both
    cells that use this: the metric is linear in the flipped count). A target
    needing more 0-rows than exist is reported ``not_achievable`` together
    with the achievable ceiling -- that ceiling is itself the answer to "how
    large an effect could this instrument even represent".
    """

    response = frame[response_col].to_numpy(dtype=float)
    zero_positions = np.flatnonzero(flip_mask & (response == 0.0))
    ceiling = len(zero_positions) / rows_per_point if rows_per_point > 0 else float("inf")
    results: list[dict[str, Any]] = []
    for target in targets:
        needed = round(target * rows_per_point)
        if needed > len(zero_positions):
            results.append(
                {
                    "target_accuracy_points": target,
                    "not_achievable": True,
                    "rows_needed": needed,
                    "zero_rows_available": len(zero_positions),
                    "achievable_ceiling_accuracy_points": float(ceiling),
                }
            )
            continue
        recovered: list[float] = []
        detected = 0
        for replicate in range(replicates):
            rng = np.random.default_rng(seed + 1_000 * replicate + int(target * 100))
            chosen = rng.choice(zero_positions, size=needed, replace=False)
            injected = frame.copy()
            values = response.copy()
            values[chosen] = 1.0
            injected[response_col] = values
            table = _bootstrap(
                injected, metric_fn, block="week", samples=samples, seed=seed + replicate
            )[metric_key]
            recovered.append(table["estimate"])
            detected += int(_excludes_zero(table))
        results.append(
            {
                "target_accuracy_points": target,
                "not_achievable": False,
                "rows_flipped": needed,
                "zero_rows_available": len(zero_positions),
                "replicates": replicates,
                "bootstrap_samples": samples,
                "mean_recovered_effect": float(np.mean(recovered)),
                "min_recovered_effect": float(np.min(recovered)),
                "max_recovered_effect": float(np.max(recovered)),
                "detection_rate_interval_excludes_zero": detected / replicates,
                "achievable_ceiling_accuracy_points": float(ceiling),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Cell 1: bye_overval_install_need_moderator
# ---------------------------------------------------------------------------


def _canonical_team(series: pd.Series) -> pd.Series:
    return series.replace(TEAM_ABBREVIATION_ALIASES)


def opener_starters_from_plays(plays: pd.DataFrame) -> pd.DataFrame:
    """(season, team, starter_id, dropbacks) from one season's REG plays.

    "Starting QB" = the passer with the most ``qb_dropback`` plays for that team
    in the team's FIRST REG game of the season (ties broken by passer id, so the
    result is deterministic). "First REG game" rather than literally week 1 so
    the two 2017 hurricane-postponed team-seasons (TB, MIA) are covered rather
    than dropped. **Leakage**: only the opener's plays are read, so no game's
    own result can feed its own moderator value, and the opener is settled
    before any strict >=12-day bye can occur (week 3 at the earliest).
    """

    reg = plays.loc[plays["season_type"].eq("REG") & plays["posteam"].notna()].copy()
    first_week = reg.groupby("posteam")["week"].min().rename("first_week")
    reg = reg.merge(first_week, left_on="posteam", right_index=True, how="left")
    opener = reg.loc[
        reg["week"].eq(reg["first_week"])
        & reg["qb_dropback"].eq(1)
        & reg["passer_player_id"].notna()
    ]
    counts = opener.groupby(["season", "posteam", "passer_player_id"]).size().rename("dropbacks")
    counts = counts.reset_index().sort_values(
        ["season", "posteam", "dropbacks", "passer_player_id"],
        ascending=[True, True, False, True],
    )
    return counts.groupby(["season", "posteam"], as_index=False).first()


def build_first_game_starters(pbp_root: Path | None = None) -> pd.DataFrame:
    """``opener_starters_from_plays`` over every season of the newest PBP snapshot."""

    snapshot = latest_pbp_snapshot(pbp_root or (REPO / "data" / "pbp" / "raw"))
    frames = [
        opener_starters_from_plays(
            pd.read_parquet(
                snapshot.season_path(season),
                columns=[
                    "season",
                    "season_type",
                    "week",
                    "posteam",
                    "passer_player_id",
                    "qb_dropback",
                ],
            )
        )
        for season in snapshot.seasons
    ]
    starters = pd.concat(frames, ignore_index=True)
    starters["season"] = starters["season"].astype(int)
    starters["team"] = _canonical_team(starters["posteam"])
    return starters.loc[:, ["season", "team", "passer_player_id", "dropbacks"]].rename(
        columns={"passer_player_id": "starter_id"}
    )


def install_need_from_starters(starters: pd.DataFrame) -> pd.DataFrame:
    """(season, team, install_need) -- season S opener QB differs from S-1's.

    The predeclared moderator is "new primary starting QB OR new offensive
    coordinator". **Measured**: this repository holds no offensive-coordinator
    source (``data/raw/interim_coaches`` is a HEAD-coach capture and
    ``schedules.parquet`` carries ``home_coach``/``away_coach`` only), so the
    OC leg cannot be built and the moderator runs on the QB leg alone. That
    makes the install-need subset strictly NARROWER than the predeclared
    definition -- some genuine install-need teams (new OC, same QB) land in
    the complement, DILUTING the contrast toward zero. A conservative power
    reduction, disclosed; not a change of direction, population or comparator.

    A team-season with no observed immediately-prior season gets ``NaN`` and is
    excluded from BOTH moderator arms by the caller, with the count disclosed.
    """

    ordered = starters.sort_values(["team", "season"]).copy()
    ordered["prev_starter_id"] = ordered.groupby("team")["starter_id"].shift(1)
    ordered["prev_season"] = ordered.groupby("team")["season"].shift(1)
    known = ordered["prev_starter_id"].notna() & ordered["season"].sub(ordered["prev_season"]).eq(1)
    ordered["install_need"] = np.where(
        known, ordered["starter_id"].ne(ordered["prev_starter_id"]), np.nan
    )
    return ordered.loc[:, ["season", "team", "starter_id", "prev_starter_id", "install_need"]]


def build_install_need_map(pbp_root: Path | None = None) -> pd.DataFrame:
    """``install_need_from_starters`` over the newest local PBP snapshot."""

    return install_need_from_starters(build_first_game_starters(pbp_root))


def bye_base_flag(population: pd.DataFrame) -> pd.Series:
    """``bye_overval_home_edge_post2011``'s own flag, imported and never re-derived.

    HOME team off strict bye (>= ``bye_overvaluation_screen.POST_BYE_GAP_DAYS``
    calendar days since its own immediately preceding game that season) AND the
    opponent NOT off bye, using ``bye_overvaluation_screen.build_bye_maps``
    verbatim -- including its 2026-08-22 within-season fix.
    """

    home_pb, away_pb = bye_battery.build_bye_maps(population)
    return pd.Series(
        home_pb.to_numpy(dtype=bool) & ~away_pb.to_numpy(dtype=bool), index=population.index
    )


def _attach_install_need(frame: pd.DataFrame, moderator: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["home_team_canonical"] = _canonical_team(work["home_team"])
    merged = work.merge(
        moderator.loc[:, ["season", "team", "install_need"]],
        left_on=["season", "home_team_canonical"],
        right_on=["season", "team"],
        how="left",
    )
    return merged


def build_bye_moderator_frame(
    schedules_path: Path, *, pbp_root: Path | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Post-2012 bye population with the imported base flag + the moderator."""

    population = bye_battery.load_population(schedules_path)
    population["base_flag"] = bye_base_flag(population)

    moderator = build_install_need_map(pbp_root)
    attached = _attach_install_need(population, moderator)

    post = attached.loc[attached["season"] >= bye_battery.ERA_POST_MIN_SEASON].copy()
    n_post = len(post)
    unknown = int(post["install_need"].isna().sum())
    scored = post.loc[post["install_need"].notna()].reset_index(drop=True)
    scored["install_need"] = scored["install_need"].astype(bool)

    diagnostics = {
        "n_population_2009_2025": len(attached),
        "n_post2011_population": n_post,
        "n_post2011_moderator_unknown_excluded": unknown,
        "n_post2011_scored": len(scored),
        "n_base_flag_scored": int(scored["base_flag"].sum()),
        "n_install_need_rows": int(scored["install_need"].sum()),
        "n_no_need_rows": int((~scored["install_need"]).sum()),
        "n_base_flag_install_need": int((scored["base_flag"] & scored["install_need"]).sum()),
        "n_base_flag_no_need": int((scored["base_flag"] & ~scored["install_need"]).sum()),
        "post_bye_gap_days": bye_battery.POST_BYE_GAP_DAYS,
        "era_post_min_season": bye_battery.ERA_POST_MIN_SEASON,
        "moderator_legs_built": ["qb_opener_change"],
        "moderator_legs_unavailable": ["offensive_coordinator_change"],
    }
    columns = ["season", "week", "home_cover", "base_flag", "install_need"]
    return scored.loc[:, columns].reset_index(drop=True), diagnostics


def _arm_effect(cover: np.ndarray, base: np.ndarray, arm: np.ndarray) -> float:
    n_arm = int(arm.sum())
    flagged = arm & base
    complement = arm & ~base
    n_flag = int(flagged.sum())
    if n_arm == 0 or n_flag == 0 or complement.sum() == 0:
        return float("nan")
    gap = float(cover[flagged].mean() - cover[complement].mean()) * 100.0
    return gap * (n_flag / n_arm)


def bye_contrast_statistic(frame: pd.DataFrame, moderator_labels: np.ndarray) -> dict[str, float]:
    cover = frame["home_cover"].to_numpy(dtype=float)
    base = frame["base_flag"].to_numpy(dtype=bool)
    arm = np.asarray(moderator_labels, dtype=bool)
    install = _arm_effect(cover, base, arm)
    no_need = _arm_effect(cover, base, ~arm)
    return {
        "effect_install_need": install,
        "effect_no_need": no_need,
        "contrast_install_minus_no_need": install - no_need,
    }


def make_bye_metric(fallback: dict[str, float], nan_counter: list[int]):
    def metric(frame: pd.DataFrame) -> dict[str, float]:
        values = bye_contrast_statistic(frame, frame["install_need"].to_numpy(dtype=bool))
        for name, value in values.items():
            if not np.isfinite(value):
                nan_counter[0] += 1
                values[name] = fallback[name]
        return values

    return metric


def run_bye_screen(schedules_path: Path, *, samples: int, seed: int) -> dict[str, Any]:
    frame, diagnostics = build_bye_moderator_frame(schedules_path)
    observed = bye_contrast_statistic(frame, frame["install_need"].to_numpy(dtype=bool))
    nan_counter = [0]
    metric = make_bye_metric(observed, nan_counter)
    week = _bootstrap(frame, metric, block="week", samples=samples, seed=seed)
    season = _bootstrap(frame, metric, block="season", samples=samples, seed=seed)
    return {
        "cell": CELL_BYE,
        "mode": "screen",
        "diagnostics": diagnostics,
        "point_estimates": observed,
        "week_blocked_primary": week,
        "season_blocked_secondary": season,
        "bootstrap_nan_fallbacks": nan_counter[0],
        "per_era_magnitudes": {},
        "predeclared_direction": "contrast_install_minus_no_need > 0 (one-sided)",
    }


def bye_per_era_magnitudes(schedules_path: Path) -> dict[str, Any]:
    """Four magnitudes: {pre-CBA, post-CBA} x {install-need, no-need}, never averaged."""

    population = bye_battery.load_population(schedules_path)
    population["base_flag"] = bye_base_flag(population)
    attached = _attach_install_need(population, build_install_need_map())
    attached = attached.loc[attached["install_need"].notna()].copy()
    attached["install_need"] = attached["install_need"].astype(bool)

    out: dict[str, Any] = {}
    for era_label, mask in (
        ("pre2011_2009_2011", attached["season"] < bye_battery.ERA_POST_MIN_SEASON),
        ("post2011_2012_2025", attached["season"] >= bye_battery.ERA_POST_MIN_SEASON),
    ):
        era = attached.loc[mask]
        cover = era["home_cover"].to_numpy(dtype=float)
        base = era["base_flag"].to_numpy(dtype=bool)
        arm = era["install_need"].to_numpy(dtype=bool)
        out[era_label] = {
            "n_rows": len(era),
            "n_base_flag": int(base.sum()),
            "effect_install_need": _arm_effect(cover, base, arm),
            "effect_no_need": _arm_effect(cover, base, ~arm),
            "n_base_flag_install_need": int((base & arm).sum()),
            "n_base_flag_no_need": int((base & ~arm).sum()),
        }
    out["note"] = (
        "Reported as four separate magnitudes per the 'era magnitude, not presence' rule; "
        "never averaged across the parent construct's known era sign flip."
    )
    return out


def run_bye_null(schedules_path: Path, *, draws: int, seed: int) -> dict[str, Any]:
    frame, diagnostics = build_bye_moderator_frame(schedules_path)
    summary = _within_week_permutation(
        frame,
        label_col="install_need",
        statistic=bye_contrast_statistic,
        draws=draws,
        seed=seed,
    )
    return {"cell": CELL_BYE, "mode": "null", "diagnostics": diagnostics, "permutation": summary}


def build_travel_rest_control_frame(schedules_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The predeclared external control: the SAME moderator on ``travel_rest_home_off_bye``."""

    population = bye_battery.load_population(schedules_path)
    rest = pd.read_parquet(schedules_path, columns=["game_id", "home_rest"])
    merged = population.merge(rest, on="game_id", how="left")
    merged = merged.loc[merged["home_rest"].notna()].copy()
    merged["base_flag"] = merged["home_rest"] >= travel_battery.OFF_BYE_REST_DAYS
    attached = _attach_install_need(merged, build_install_need_map())
    scored = attached.loc[attached["install_need"].notna()].reset_index(drop=True)
    scored["install_need"] = scored["install_need"].astype(bool)
    diagnostics = {
        "control_construct": "travel_rest_home_off_bye",
        "control_threshold_home_rest_days": travel_battery.OFF_BYE_REST_DAYS,
        "n_rows_scored": len(scored),
        "n_base_flag": int(scored["base_flag"].sum()),
        "n_moderator_unknown_excluded": int(attached["install_need"].isna().sum()),
        "season_min": int(scored["season"].min()),
        "season_max": int(scored["season"].max()),
        "window_note": (
            "2009 rows carry no prior-season opener QB and are excluded from both arms by the "
            "frozen unknown-moderator rule, so the control's effective window starts at 2010."
        ),
    }
    return scored.loc[:, ["season", "week", "home_cover", "base_flag", "install_need"]], diagnostics


def run_bye_positive_control(schedules_path: Path, *, samples: int, seed: int) -> dict[str, Any]:
    control_frame, control_diagnostics = build_travel_rest_control_frame(schedules_path)
    observed = bye_contrast_statistic(
        control_frame, control_frame["install_need"].to_numpy(dtype=bool)
    )
    nan_counter = [0]
    metric = make_bye_metric(observed, nan_counter)
    external = {
        "diagnostics": control_diagnostics,
        "point_estimates": observed,
        "week_blocked": _bootstrap(control_frame, metric, block="week", samples=samples, seed=seed),
        "interpretation_rule": (
            "A positive-control BOUND requires this control to have DETECTED a moderator effect "
            "of the size claimed for the main cell while the main cell is flat. If this control "
            "does not detect one either, it bounds nothing."
        ),
    }

    frame, diagnostics = build_bye_moderator_frame(schedules_path)
    real = bye_contrast_statistic(frame, frame["install_need"].to_numpy(dtype=bool))
    counter = [0]
    injection = _injection_control(
        frame,
        response_col="home_cover",
        flip_mask=(frame["base_flag"] & frame["install_need"]).to_numpy(dtype=bool),
        rows_per_point=len(frame.loc[frame["install_need"]]) / 100.0,
        targets=BYE_INJECTION_TARGETS,
        metric_fn=make_bye_metric(real, counter),
        metric_key="contrast_install_minus_no_need",
        replicates=CONTROL_REPLICATES,
        samples=CONTROL_SAMPLES,
        seed=seed,
    )
    return {
        "cell": CELL_BYE,
        "mode": "positive-control",
        "external_control_travel_rest_home_off_bye": external,
        "injection_control": injection,
        "injection_diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Cell 2: pt_post_mnf_sunday_changepoint
# ---------------------------------------------------------------------------

PT_SIGN = -1


def build_post_mnf_frame(schedules_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The primetime battery's own ``pt_post_mnf_sunday`` population and flag."""

    population = pt_battery.load_population(schedules_path)
    long_df = pt_battery.build_long_table(population)
    cells = pt_battery.build_cells(long_df)
    spec = cells["pt_post_mnf_sunday"]
    eligible = spec["eligible"]
    frame = long_df.loc[eligible].reset_index(drop=True)
    flag = spec["flag"].loc[eligible].reset_index(drop=True)
    frame = frame.loc[:, ["season", "week", "team_covered"]].copy()
    frame["flag"] = flag.to_numpy(dtype=bool)
    seasons = sorted(int(s) for s in frame["season"].unique())
    frame["season_idx"] = frame["season"].map({s: i for i, s in enumerate(seasons)}).astype(int)
    diagnostics = {
        "n_eligible_rows": len(frame),
        "n_flag": int(frame["flag"].sum()),
        "seasons": seasons,
        "sign_convention": PT_SIGN,
        "description": spec["description"],
    }
    return frame, diagnostics


def _per_season_series(
    season_idx: np.ndarray,
    cover: np.ndarray,
    flag: np.ndarray,
    n_seasons: int,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Full-slate-scaled per-season effect, the primetime battery's own convention."""

    sum_flag = np.bincount(season_idx[flag], weights=cover[flag], minlength=n_seasons)
    count_flag = np.bincount(season_idx[flag], minlength=n_seasons).astype(float)
    sum_comp = np.bincount(season_idx[~flag], weights=cover[~flag], minlength=n_seasons)
    count_comp = np.bincount(season_idx[~flag], minlength=n_seasons).astype(float)
    total = count_flag + count_comp
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = (sum_flag / count_flag - sum_comp / count_comp) * 100.0
        series = PT_SIGN * gap * (count_flag / total)
    if fallback is not None:
        series = np.where(np.isfinite(series), series, fallback)
    return series


def _changepoint_summary(series: np.ndarray, seasons: list[int]) -> dict[str, Any]:
    best_k, best_sse = era_profile._changepoint_grid(series, era_profile.MIN_SEGMENT_SEASONS)
    pre = float(series[:best_k].mean())
    post = float(series[best_k:].mean())
    return {
        "break_season": int(seasons[best_k]),
        "break_index": int(best_k),
        "sse": float(best_sse),
        "pre_break_mean": pre,
        "post_break_mean": post,
        "break_magnitude_post_minus_pre": post - pre,
    }


def make_pt_metric(seasons: list[int], real_series: np.ndarray, fixed_index: int):
    n_seasons = len(seasons)
    real_break, _ = era_profile._changepoint_grid(real_series, era_profile.MIN_SEGMENT_SEASONS)

    def metric(frame: pd.DataFrame) -> dict[str, float]:
        series = _per_season_series(
            frame["season_idx"].to_numpy(dtype=int),
            frame["team_covered"].to_numpy(dtype=float),
            frame["flag"].to_numpy(dtype=bool),
            n_seasons,
            fallback=real_series,
        )
        free_k, _ = era_profile._changepoint_grid(series, era_profile.MIN_SEGMENT_SEASONS)
        return {
            "break_magnitude_at_fixed_break": float(
                series[real_break:].mean() - series[:real_break].mean()
            ),
            "pre_break_mean_at_fixed_break": float(series[:real_break].mean()),
            "post_break_mean_at_fixed_break": float(series[real_break:].mean()),
            "break_season_free": float(seasons[free_k]),
            "break_magnitude_at_free_break": float(series[free_k:].mean() - series[:free_k].mean()),
            "fixed_convention_gap_2018_minus_2009_2017": float(
                series[fixed_index:].mean() - series[:fixed_index].mean()
            ),
        }

    return metric


def run_pt_screen(schedules_path: Path, *, samples: int, seed: int) -> dict[str, Any]:
    frame, diagnostics = build_post_mnf_frame(schedules_path)
    seasons = diagnostics["seasons"]
    series = _per_season_series(
        frame["season_idx"].to_numpy(dtype=int),
        frame["team_covered"].to_numpy(dtype=float),
        frame["flag"].to_numpy(dtype=bool),
        len(seasons),
    )
    fixed_index = seasons.index(2018)
    located = _changepoint_summary(series, seasons)
    fixed = {
        "boundary_season": 2018,
        "pre_mean_2009_2017": float(series[:fixed_index].mean()),
        "post_mean_2018_2025": float(series[fixed_index:].mean()),
        "gap_post_minus_pre": float(series[fixed_index:].mean() - series[:fixed_index].mean()),
        "note": (
            "The primetime battery's FIXED convention, applied uniformly to all seven of its "
            "cells; docs/era_magnitude_report.md found no rule change at this boundary."
        ),
    }
    metric = make_pt_metric(seasons, series, fixed_index)
    week = _bootstrap(frame, metric, block="week", samples=samples, seed=seed)
    season = _bootstrap(frame, metric, block="season", samples=samples, seed=seed)
    return {
        "cell": CELL_PT,
        "mode": "screen",
        "diagnostics": diagnostics,
        "real_per_season_effect": {str(s): float(v) for s, v in zip(seasons, series, strict=True)},
        "located_changepoint": located,
        "fixed_convention_boundary": fixed,
        "week_blocked_primary": week,
        "season_blocked_secondary": season,
        "predeclared_direction": "none (structural/diagnostic, two-sided)",
        "registry_fixed_arms_for_comparison": {
            "pt_post_mnf_sunday_era_2009_2017": 0.2546,
            "pt_post_mnf_sunday_era_2018_2025": -0.1367,
            "source": "docs/era_magnitude_report.md:109-110 (read); reported, not re-measured",
        },
    }


def run_pt_null(schedules_path: Path, *, draws: int, seed: int) -> dict[str, Any]:
    frame, diagnostics = build_post_mnf_frame(schedules_path)
    seasons = diagnostics["seasons"]
    n_seasons = len(seasons)
    fixed_index = seasons.index(2018)

    def statistic(inner: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
        series = _per_season_series(
            inner["season_idx"].to_numpy(dtype=int),
            inner["team_covered"].to_numpy(dtype=float),
            np.asarray(labels, dtype=bool),
            n_seasons,
        )
        if not np.all(np.isfinite(series)):
            return {
                "break_season_free": float("nan"),
                "break_magnitude_at_free_break": float("nan"),
                "fixed_convention_gap": float("nan"),
            }
        free_k, _ = era_profile._changepoint_grid(series, era_profile.MIN_SEGMENT_SEASONS)
        return {
            "break_season_free": float(seasons[free_k]),
            "break_magnitude_at_free_break": float(series[free_k:].mean() - series[:free_k].mean()),
            "fixed_convention_gap": float(
                series[fixed_index:].mean() - series[:fixed_index].mean()
            ),
        }

    summary = _within_week_permutation(
        frame, label_col="flag", statistic=statistic, draws=draws, seed=seed
    )
    summary["_reading_note"] = {
        "note": (
            "An optimal single-changepoint estimator ALWAYS finds some break, even on pure "
            "noise, so the null's own |break magnitude| is the size of break this machinery "
            "manufactures from a shuffled label -- the number the observed break must be read "
            "against."
        )
    }
    return {"cell": CELL_PT, "mode": "null", "diagnostics": diagnostics, "permutation": summary}


def build_hc_control_construct() -> tuple[pd.DataFrame, list[int]]:
    """``hc_year_one_fade`` on the same shape as cell 2, for the known-answer check.

    Built from ``scripts/hc_year_one_fade.py``'s own imported builders
    (``build_team_game_table``/``team_season_primary_coach``/``flag_year_one``),
    mirroring ``era_magnitude_profile.build_hc_year_one_fade`` -- which cannot
    be called directly because it invokes ``hc_year_one_fade.default_schedules``,
    a function that module does not define (**measured** this session).
    """

    schedules = pd.read_parquet(hc_module.DEFAULT_SCHEDULES)
    features = pd.read_parquet(hc_module.DEFAULT_FEATURES)
    long = hc_module.build_team_game_table(schedules, features)
    tenure = hc_module.flag_year_one(hc_module.team_season_primary_coach(long))
    weeks = long.loc[long["game_type"].eq("REG") & long["week"].le(8)].copy()
    weeks = weeks.merge(tenure, on=["team", "season"], how="inner")
    weeks = weeks.loc[weeks["team_covered"].notna()].copy()
    weeks = weeks.loc[
        weeks["season"].between(era_profile.POPULATION_START, era_profile.POPULATION_END)
    ]
    flag = weeks["year_one"].fillna(False).astype(bool)
    eligible = flag | weeks["kept_coach"].fillna(False).astype(bool)
    frame = weeks.loc[eligible, ["season", "week", "team_covered"]].reset_index(drop=True)
    frame["flag"] = flag.loc[eligible].to_numpy(dtype=bool)
    seasons = sorted(int(s) for s in frame["season"].unique())
    frame["season_idx"] = frame["season"].map({s: i for i, s in enumerate(seasons)}).astype(int)
    return frame, seasons


def run_pt_positive_control(schedules_path: Path, *, samples: int, seed: int) -> dict[str, Any]:
    del schedules_path
    frame, seasons = build_hc_control_construct()
    series = _per_season_series(
        frame["season_idx"].to_numpy(dtype=int),
        frame["team_covered"].to_numpy(dtype=float),
        frame["flag"].to_numpy(dtype=bool),
        len(seasons),
    )
    located = _changepoint_summary(series, seasons)
    metric = make_pt_metric(seasons, series, seasons.index(2018) if 2018 in seasons else 0)
    week = _bootstrap(frame, metric, block="week", samples=samples, seed=seed)
    return {
        "cell": CELL_PT,
        "mode": "positive-control",
        "control_construct": "hc_year_one_fade",
        "n_rows": len(frame),
        "seasons": seasons,
        "real_per_season_effect": {str(s): float(v) for s, v in zip(seasons, series, strict=True)},
        "located_changepoint": located,
        "week_blocked": week,
        "known_answer_reported_not_measured": {
            "source": "docs/era_magnitude_profile.md:84 (read)",
            "claim": "+0.09 pts 2009-2017 vs -8.08 pts 2018-2025",
            "status": "reported, unverified here; this control checks WHERE the estimator "
            "puts the break, not the exact magnitudes",
        },
    }


# ---------------------------------------------------------------------------
# Cell 3: sagarin coverage-matched eras
# ---------------------------------------------------------------------------


def sagarin_coverage_table(schedules_path: Path, sagarin_root: Path) -> tuple[pd.DataFrame, Any]:
    """Per-season screen-population coverage, the convention docs/sagarin_backfill.md 9.3 uses."""

    close_pop, _ = sagarin_battery.build_close_population(schedules_path, sagarin_root)
    schedule = sagarin_battery.load_raw_schedule(schedules_path)
    reg = schedule.groupby("season").size().rename("reg_games")
    usable = close_pop.groupby("season").size().rename("usable_games")
    coverage = pd.concat([reg, usable], axis=1).fillna(0.0).reset_index()
    coverage["coverage_pct"] = 100.0 * coverage["usable_games"] / coverage["reg_games"]
    return coverage, close_pop


def large_divergence_rows(close_pop: pd.DataFrame) -> pd.DataFrame:
    """The battery's own ``|divergence_close| >= LARGE_DIVERGENCE_THRESHOLD`` subset."""

    rows = close_pop.loc[
        close_pop["divergence_close"].abs() >= sagarin_battery.LARGE_DIVERGENCE_THRESHOLD
    ].copy()
    rows["sagarin_side_home"] = rows["divergence_close"] > 0.0
    return rows


def coverage_matched_seasons(
    coverage: pd.DataFrame, *, season_lo: int, season_hi: int, threshold_pct: float
) -> tuple[list[int], list[int], list[int]]:
    """(era seasons, kept at/above threshold, dropped below it). Rule frozen, list derived."""

    present = sorted(int(s) for s in coverage["season"].unique())
    era_seasons = [s for s in present if season_lo <= s <= season_hi]
    kept_all = {
        int(s) for s in coverage.loc[coverage["coverage_pct"] >= threshold_pct, "season"].tolist()
    }
    kept = [s for s in era_seasons if s in kept_all]
    dropped = [s for s in era_seasons if s not in kept_all]
    return era_seasons, kept, dropped


def build_sagarin_matched_populations(
    schedules_path: Path, sagarin_root: Path
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    coverage, close_pop = sagarin_coverage_table(schedules_path, sagarin_root)
    large = large_divergence_rows(close_pop)

    frames: dict[str, pd.DataFrame] = {}
    era_meta: dict[str, Any] = {}
    for label, lo, hi in sagarin_battery.ERA_SPLITS:
        era_seasons, matched_seasons, dropped_seasons = coverage_matched_seasons(
            coverage, season_lo=lo, season_hi=hi, threshold_pct=COVERAGE_MATCH_THRESHOLD_PCT
        )
        era_rows = large.loc[large["season"].between(lo, hi)]
        matched = era_rows.loc[era_rows["season"].isin(matched_seasons)].reset_index(drop=True)
        frames[label] = matched
        era_meta[label] = {
            "era_seasons": era_seasons,
            "coverage_matched_seasons_kept": matched_seasons,
            "seasons_dropped_below_threshold": dropped_seasons,
            "n_matched": len(matched),
            "n_unmatched_full_era": len(era_rows),
        }
        frames[f"{label}__unmatched"] = era_rows.reset_index(drop=True)
    diagnostics = {
        "coverage_threshold_pct": COVERAGE_MATCH_THRESHOLD_PCT,
        "large_divergence_threshold": sagarin_battery.LARGE_DIVERGENCE_THRESHOLD,
        "coverage_by_season": coverage.to_dict(orient="records"),
        "close_population_n": len(close_pop),
        "large_divergence_n": len(large),
        "eras": era_meta,
        "family_note": (
            "New family sagarin_divergence_coverage_matched on a DIFFERENT (coverage-matched) "
            "population. This is NOT a re-score: no sagarin_battery_* registry entry is "
            "overwritten, corrected or reclassified. The unmatched corrected-coverage reads "
            "below are run diagnostics only and are deliberately NOT recorded -- those would be "
            "the frozen entries' own identities re-measured, which docs/sagarin_backfill.md "
            "section 9.4 says needs its own predeclaration and rotation window."
        ),
    }
    return frames, diagnostics


def sagarin_statistic(frame: pd.DataFrame, side_home: np.ndarray) -> dict[str, float]:
    home_cover = frame["home_cover"].to_numpy(dtype=float)
    side = np.asarray(side_home, dtype=bool)
    cover = np.where(side, home_cover, 1.0 - home_cover)
    return {"effect_pts": float(cover.mean() - 0.5) * 100.0}


def sagarin_metric(frame: pd.DataFrame) -> dict[str, float]:
    return {"effect_pts": (float(frame["sagarin_side_cover"].mean()) - 0.5) * 100.0}


def run_sagarin_screen(
    schedules_path: Path, sagarin_root: Path, *, samples: int, seed: int
) -> dict[str, Any]:
    frames, diagnostics = build_sagarin_matched_populations(schedules_path, sagarin_root)
    arms: dict[str, Any] = {}
    naming = {
        "2010_2016": CELL_SAGARIN,
        "2017_2025": CELL_SAGARIN_LATE,
    }
    for label, _lo, _hi in sagarin_battery.ERA_SPLITS:
        for suffix, key in (("", label), ("__unmatched", f"{label}__unmatched")):
            frame = frames[key]
            entry: dict[str, Any] = {
                "era": label,
                "coverage_matched": suffix == "",
                "recorded": suffix == "",
                "registry_name": naming[label] if suffix == "" else None,
                "n_games": len(frame),
                "seasons": sorted(int(s) for s in frame["season"].unique()),
            }
            if len(frame):
                entry["point_estimate"] = sagarin_metric(frame)
                entry["week_blocked_primary"] = _bootstrap(
                    frame, sagarin_metric, block="week", samples=samples, seed=seed
                )
                entry["season_blocked_secondary"] = _bootstrap(
                    frame, sagarin_metric, block="season", samples=samples, seed=seed
                )
            else:
                entry["insufficient_data"] = True
            arms[key] = entry
    return {
        "cell": CELL_SAGARIN,
        "mode": "screen",
        "diagnostics": diagnostics,
        "arms": arms,
        "predeclared_direction": "none (instrument-composition diagnostic, two-sided)",
        "registry_frozen_prefix_values_for_comparison": {
            "sagarin_battery_large_divergence_era_2010_2016": 1.8072,
            "sagarin_battery_large_divergence_era_2017_2025": -2.2989,
            "source": "docs/era_magnitude_report.md:122-123 (read); pre-fix coverage; "
            "reported for comparison, NOT re-measured and NOT replaced",
        },
    }


def run_sagarin_null(
    schedules_path: Path, sagarin_root: Path, *, draws: int, seed: int
) -> dict[str, Any]:
    frames, diagnostics = build_sagarin_matched_populations(schedules_path, sagarin_root)
    out: dict[str, Any] = {}
    for label, _lo, _hi in sagarin_battery.ERA_SPLITS:
        frame = frames[label]
        if not len(frame):
            out[label] = {"insufficient_data": True}
            continue
        out[label] = _within_week_permutation(
            frame,
            label_col="sagarin_side_home",
            statistic=sagarin_statistic,
            draws=draws,
            seed=seed,
        )
    return {
        "cell": CELL_SAGARIN,
        "mode": "null",
        "diagnostics": diagnostics,
        "permutation": out,
        "permutation_note": (
            "sagarin_side_home (WHICH side Sagarin favours) is permuted within each (season, "
            "week) block; permuting the outcome alone would leave the cover rate unchanged and "
            "would not be a null for this cell at all."
        ),
    }


def run_sagarin_positive_control(
    schedules_path: Path, sagarin_root: Path, *, seed: int
) -> dict[str, Any]:
    frames, diagnostics = build_sagarin_matched_populations(schedules_path, sagarin_root)
    frame = frames["2010_2016"].copy()
    injection = _injection_control(
        frame,
        response_col="sagarin_side_cover",
        flip_mask=np.ones(len(frame), dtype=bool),
        rows_per_point=len(frame) / 100.0,
        targets=SAGARIN_INJECTION_TARGETS,
        metric_fn=sagarin_metric,
        metric_key="effect_pts",
        replicates=CONTROL_REPLICATES,
        samples=CONTROL_SAMPLES,
        seed=seed,
    )
    return {
        "cell": CELL_SAGARIN,
        "mode": "positive-control",
        "diagnostics": diagnostics,
        "injection_control": injection,
        "already_in_hand_precedent": {
            "source": "docs/sagarin_backfill.md section 9 (read)",
            "claim": "the Era-B coverage-completeness fix already moved this exact cell's point "
            "estimate +2.926 -> +1.807 as 2010/2011 coverage rose",
            "status": "reported, not re-run here; this is instrument SENSITIVITY to coverage "
            "composition, which is not the same thing as detectability -- the injection control "
            "above is the detectability statement",
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_one(cell: str, mode: str, args: argparse.Namespace) -> dict[str, Any]:
    if cell == CELL_BYE:
        if mode == "screen":
            payload = run_bye_screen(args.schedules, samples=args.samples, seed=20260821)
            payload["per_era_magnitudes"] = bye_per_era_magnitudes(args.schedules)
            return payload
        if mode == "null":
            return run_bye_null(args.schedules, draws=args.null_draws, seed=20260821)
        return run_bye_positive_control(args.schedules, samples=args.samples, seed=20260821)
    if cell == CELL_PT:
        if mode == "screen":
            return run_pt_screen(args.schedules, samples=args.samples, seed=20260821)
        if mode == "null":
            return run_pt_null(args.schedules, draws=args.null_draws, seed=20260821)
        return run_pt_positive_control(args.schedules, samples=args.samples, seed=20260821)
    if mode == "screen":
        return run_sagarin_screen(
            args.schedules, args.sagarin_root, samples=args.samples, seed=20260820
        )
    if mode == "null":
        return run_sagarin_null(
            args.schedules, args.sagarin_root, draws=args.null_draws, seed=20260820
        )
    return run_sagarin_positive_control(args.schedules, args.sagarin_root, seed=20260820)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="all", choices=(*ALL_CELLS, "all"))
    parser.add_argument("--mode", default="screen", choices=("screen", "null", "positive-control"))
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument(
        "--sagarin-root",
        type=Path,
        default=REPO / "data" / "raw" / "sagarin" / sagarin_battery.DEFAULT_SAGARIN_SNAPSHOT,
    )
    parser.add_argument("--samples", type=int, default=PRIMARY_SAMPLES)
    parser.add_argument("--null-draws", type=int, default=NULL_DRAWS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.schedules is None:
        args.schedules = bye_battery.default_schedules()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "era_mechanism_screens" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = ALL_CELLS if args.cell == "all" else (args.cell,)
    results: list[dict[str, Any]] = []
    for cell in cells:
        _log(f"\n=== {cell} [{args.mode}] ===")
        result = _run_one(cell, args.mode, args)
        results.append(result)
        _log(_summarise(result))

    configuration = {
        "command": "era-mechanism-screens",
        "cell": args.cell,
        "mode": args.mode,
        "schedules": str(args.schedules),
        "sagarin_root": str(args.sagarin_root),
        "samples": args.samples,
        "null_draws": args.null_draws,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "predeclaration": "docs/era_mechanism_screens_20260901.md (frozen before scoring)",
        "binding_taxonomy": TAXONOMY,
        "mined_window_discount": MINED_WINDOW_DISCOUNT,
        "within_week_correlation": "ZERO by project mandate; no ICC estimated or padded",
        "mode": args.mode,
        "cells_run": list(cells),
        "primary_uncertainty": "nfl_ats.clv.week_blocked_bootstrap(block='week')",
        "secondary_uncertainty": "nfl_ats.clv.week_blocked_bootstrap(block='season'), "
        "informational only, DEGENERATE below 10 blocks",
        "results": results,
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="era-mechanism-screens",
        metrics=payload,
        notes=(
            "Three predeclared era-mechanism cells from docs/era_magnitude_report.md, frozen in "
            "docs/era_mechanism_screens_20260901.md before scoring. Measure-only: writes no "
            "registry JSON. Mined-window discount applies; every cell is predeclared to record "
            "unresolved_below_power regardless of interval shape unless a terminal ground in "
            "AGENTS.md is literally met."
        ),
    )
    _log(f"\nwrote {output_dir / 'results.json'}")
    return 0


def _summarise(result: dict[str, Any]) -> str:
    lines: list[str] = []
    mode = result.get("mode")
    if mode == "screen" and result["cell"] == CELL_BYE:
        point = result["point_estimates"]
        week = result["week_blocked_primary"]["contrast_install_minus_no_need"]
        lines.append(
            f"  install-need arm {point['effect_install_need']:+.4f}pts | "
            f"no-need arm {point['effect_no_need']:+.4f}pts"
        )
        lines.append(
            f"  contrast {week['estimate']:+.4f}pts week-blocked 95% "
            f"[{week['lower']:+.4f}, {week['upper']:+.4f}] P+={week['probability_positive']:.4f} "
            f"n_blocks={week['n_blocks']}"
        )
    elif mode == "screen" and result["cell"] == CELL_PT:
        located = result["located_changepoint"]
        week = result["week_blocked_primary"]["break_magnitude_at_fixed_break"]
        free = result["week_blocked_primary"]["break_season_free"]
        lines.append(
            f"  located break {located['break_season']} | pre "
            f"{located['pre_break_mean']:+.4f}pts | post {located['post_break_mean']:+.4f}pts"
        )
        lines.append(
            f"  break magnitude {week['estimate']:+.4f}pts week-blocked 95% "
            f"[{week['lower']:+.4f}, {week['upper']:+.4f}] P+={week['probability_positive']:.4f}"
        )
        lines.append(
            f"  break-season bootstrap median {free['estimate']:.1f} "
            f"[{free['lower']:.0f}, {free['upper']:.0f}]"
        )
        fixed = result["fixed_convention_boundary"]
        lines.append(
            f"  fixed 2017/2018 convention: pre {fixed['pre_mean_2009_2017']:+.4f} | "
            f"post {fixed['post_mean_2018_2025']:+.4f}"
        )
    elif mode == "screen":
        for key, entry in result["arms"].items():
            if entry.get("insufficient_data"):
                lines.append(f"  {key}: insufficient data")
                continue
            week = entry["week_blocked_primary"]["effect_pts"]
            tag = "RECORDED" if entry["recorded"] else "diagnostic-only"
            lines.append(
                f"  {key} [{tag}] n={entry['n_games']} seasons={entry['seasons']} "
                f"{week['estimate']:+.4f}pts 95% [{week['lower']:+.4f}, {week['upper']:+.4f}] "
                f"P+={week['probability_positive']:.4f}"
            )
    else:
        lines.append(f"  {mode} complete (see artifact)")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
