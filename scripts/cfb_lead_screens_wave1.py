"""CFB free-screen wave 1 (LEAD-48, LEAD-50, LEAD-46; LEAD-44 recorded as a
data-gap skip -- see ``docs/cfb_lead_screens_wave1.md``).

Predeclared before any outcome is scored: three signed/flag columns, each its
own weak-signal family, on top of the frozen XLG-03 benchmark arm. Spends no
NFL evaluation window and no rotation window -- CFB is this project's
sanctioned free replication ground. All cells are recorded regardless of
sign; an interval crossing zero is never a rejection.

Closing-grounds taxonomy (binding, restated verbatim per AGENTS.md): an
interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is
the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control -- the instrument was PROVEN able to detect an effect that
size and it was absent. Everything else is ``unresolved_below_power``:
record it with ``nfl-ats weak-signals record --league cfb``, report
``probability_positive``, never the binary "contains zero". If a record
command errors, the verdict is wrong, not the validator.

Leads:

* ``post_bye`` (LEAD-48): signed ``cfb_lead48_post_bye_signed`` -- +1 when the
  HOME team is off a 13+ day bye and the AWAY team is not, -1 for the mirror,
  0 when both sides are known and neither/both qualify. NOT a re-score of
  ``docs/cfb_rest_bye_replication.md``'s ``bye_edge_home`` cell (that cell
  uses a 12-day gap, is unsigned/home-only, and predicts the OPPOSITE
  direction); see the predeclaration doc section 1 for the full comparison.
* ``rivalry_home_dog`` (LEAD-50): unsigned ``cfb_lead50_rivalry_home_dog`` --
  1 when the game's team pair met in >=8 consecutive seasons in the local
  schedule (the deterministic proxy; no local rivalry field exists on the
  CFBD schedule rows) AND the home team is the market underdog
  (``spread_line < 0``), else 0.
* ``altitude_cold`` (LEAD-46): unsigned ``cfb_lead46_altitude_cold_home`` -- 1
  when the HOME team is one of the frozen named altitude programs (Colorado
  State, Wyoming, Air Force, Utah -- no local elevation/venue table exists,
  so the frozen list stands in) and the game is in October or later, else 0.
* ``sandwich`` (LEAD-44): no local AP Top-25 poll table exists under
  ``data/cfb`` or ``data/raw`` (measured this session). Per the no-fetch rule
  this lead is a recorded skip, not a screen: ``--lead sandwich`` prints the
  gap and exits without touching the feature table.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_CLEAN_CORE_SEASONS,
    fit_cfb_residual_model,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS
from nfl_ats.cfb_rest_bye_feature import OFF_BYE_REST_DAYS, default_cfb_schedules, derive_side_rest
from nfl_ats.clv import pick_correct, week_blocked_bootstrap
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPO_ROOT / "data" / "processed" / "cfb_game_features.parquet"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "cfb_lead_screens_wave1"
BOOTSTRAP_SAMPLES = 1_000
SEED = 20260905
PERMUTATIONS = 200
ERAS: tuple[tuple[str, int, int], ...] = (("2012_2019", 2012, 2019), ("2021_2025", 2021, 2025))

RIVALRY_MIN_CONSECUTIVE_SEASONS = 8

#: Frozen named list (LEAD-46): no local elevation/venue table exists, so the
#: ROADMAP's own named Mountain-West altitude outs stand in unchanged.
ALTITUDE_HOME_TEAMS = frozenset({"Colorado State", "Wyoming", "Air Force", "Utah"})
#: "Cold" is calendar-proxied (October on) because no local weather/temperature
#: column exists on the benchmark table; disclosed, not silently assumed.
ALTITUDE_MONTH_FLOOR = 10

CANDIDATE_COLUMNS: dict[str, str] = {
    "post_bye": "cfb_lead48_post_bye_signed",
    "rivalry_home_dog": "cfb_lead50_rivalry_home_dog",
    "altitude_cold": "cfb_lead46_altitude_cold_home",
}

PREDICTED_DIRECTION: dict[str, str] = {
    "post_bye": (
        "back the HOME team when it is off a 13+ day bye and the away team is "
        "not (signed column; mirror image backs the away team)"
    ),
    "rivalry_home_dog": "back the HOME underdog in a rivalry game",
    "altitude_cold": "back the altitude HOME team from October onward",
}

SCORABLE_LEADS = ("post_bye", "rivalry_home_dog", "altitude_cold")

SANDWICH_GAP_MESSAGE = (
    "LEAD-44 (CFB sandwich/lookahead fade) needs a local AP Top-25 poll table "
    "to identify ranked favorites and their next-week rivalry/top-10 "
    "opponent. Measured this session: no rankings/poll directory exists under "
    "data/cfb (draft_picks, espn_betting, lines, participants, pbp, portal, "
    "recruiting_players, recruiting_teams, returning_production, rosters, "
    "schedules, team_info, usage -- no polls) or data/raw, and "
    "docs/cfb_data.md's provider table lists only odds/line sources; a "
    "case-insensitive grep for 'rival'/'poll'/'ranking' across src/, scripts/ "
    "and docs/ turns up nothing but this document and the ROADMAP row "
    "itself. Per the fleet brief's no-fetch rule this lead is SKIPPED, not "
    "screened -- see docs/cfb_lead_screens_wave1.md section 4 and the "
    "ROADMAP.md LEAD-44 row for the recorded gap."
)


# ---------------------------------------------------------------------------
# LEAD-48: post-bye prep asymmetry
# ---------------------------------------------------------------------------


def attach_post_bye_flag(
    features: pd.DataFrame, *, schedules: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Signed edge column: +1 home off 13+ days & away not, -1 mirror, 0 else.

    Reuses ``nfl_ats.cfb_rest_bye_feature.derive_side_rest`` -- the exact
    helper the frozen ``rest_diff`` column and the ``cfb_rest_bye_replication``
    cells are built from -- rather than re-deriving per-side rest. Both sides
    must be KNOWN to resolve to 0 or a sign; either side missing (a season
    opener) leaves the row NaN, never 0, matching the project's rest-missingness
    convention.
    """

    frame = features.copy()
    rested = derive_side_rest(frame, schedules)
    lookup = rested.set_index("game_id")[["cfb_home_rest_days", "cfb_away_rest_days"]]
    frame = frame.join(lookup, on="game_id")
    home = frame["cfb_home_rest_days"].to_numpy(dtype=float)
    away = frame["cfb_away_rest_days"].to_numpy(dtype=float)
    home_known = np.isfinite(home)
    away_known = np.isfinite(away)
    both_known = home_known & away_known
    with np.errstate(invalid="ignore"):
        home_off = home >= OFF_BYE_REST_DAYS
        away_off = away >= OFF_BYE_REST_DAYS
    signed = np.where(both_known, 0.0, np.nan)
    signed = np.where(both_known & home_off & ~away_off, 1.0, signed)
    signed = np.where(both_known & away_off & ~home_off, -1.0, signed)
    frame[CANDIDATE_COLUMNS["post_bye"]] = signed
    return frame.drop(columns=["cfb_home_rest_days", "cfb_away_rest_days"])


# ---------------------------------------------------------------------------
# LEAD-50: rivalry home dog
# ---------------------------------------------------------------------------


def compute_rivalry_pairs(
    schedules: pd.DataFrame, *, min_consecutive_seasons: int = RIVALRY_MIN_CONSECUTIVE_SEASONS
) -> frozenset[frozenset[str]]:
    """Deterministic rivalry proxy, declared before any outcome is read.

    A team pair is a "rivalry" if they met (regular season, completed) in at
    least ``min_consecutive_seasons`` back-to-back seasons ANYWHERE in the
    local schedule history -- captures long-running annual pairings, which
    includes true historic rivalries but also conference-mandated annual
    opponents (disclosed limitation, not a defect: no local rivalry field
    exists to disambiguate the two). Uses only team identity and season, never
    a score or a line, so it cannot encode any game's own outcome.
    """

    regular = schedules.loc[
        schedules["season_type"].eq("regular") & schedules["completed"].eq(True)
    ].dropna(subset=["home_team", "away_team", "season"])
    seasons_by_pair: dict[frozenset[str], set[int]] = {}
    for home, away, season in zip(
        regular["home_team"], regular["away_team"], regular["season"], strict=True
    ):
        key = frozenset({str(home), str(away)})
        if len(key) != 2:
            continue
        seasons_by_pair.setdefault(key, set()).add(int(season))
    rivalry_pairs: set[frozenset[str]] = set()
    for key, seasons in seasons_by_pair.items():
        ordered = sorted(seasons)
        run = best = 1
        for index in range(1, len(ordered)):
            if ordered[index] == ordered[index - 1] + 1:
                run += 1
                best = max(best, run)
            else:
                run = 1
        if best >= min_consecutive_seasons:
            rivalry_pairs.add(key)
    return frozenset(rivalry_pairs)


def attach_rivalry_home_dog_flag(
    features: pd.DataFrame, *, schedules: pd.DataFrame | None = None
) -> pd.DataFrame:
    """1 iff (rivalry pair) AND (home is the market underdog, spread_line<0)."""

    frame = features.copy()
    source = schedules if schedules is not None else default_cfb_schedules()
    rivalry_pairs = compute_rivalry_pairs(source)
    pair_keys = [
        frozenset({str(home), str(away)})
        for home, away in zip(frame["home_team"], frame["away_team"], strict=True)
    ]
    is_rivalry = np.array([key in rivalry_pairs for key in pair_keys], dtype=bool)
    spread = pd.to_numeric(frame["spread_line"], errors="coerce").to_numpy(dtype=float)
    home_underdog = spread < 0.0
    frame[CANDIDATE_COLUMNS["rivalry_home_dog"]] = (is_rivalry & home_underdog).astype(float)
    frame["_lead50_is_rivalry_pair"] = is_rivalry.astype(float)
    frame["_lead50_home_underdog"] = home_underdog.astype(float)
    return frame


# ---------------------------------------------------------------------------
# LEAD-46: altitude-plus-cold home
# ---------------------------------------------------------------------------


def attach_altitude_cold_home_flag(features: pd.DataFrame) -> pd.DataFrame:
    """1 iff the HOME team is a frozen altitude program and month >= October."""

    frame = features.copy()
    gameday = pd.to_datetime(frame["gameday"], errors="raise")
    is_altitude_home = frame["home_team"].isin(ALTITUDE_HOME_TEAMS).to_numpy()
    is_late_season = gameday.dt.month.ge(ALTITUDE_MONTH_FLOOR).to_numpy()
    frame[CANDIDATE_COLUMNS["altitude_cold"]] = (is_altitude_home & is_late_season).astype(float)
    return frame


def attach_candidate(
    lead: str, features: pd.DataFrame, *, schedules: pd.DataFrame | None = None
) -> pd.DataFrame:
    if lead == "post_bye":
        return attach_post_bye_flag(features, schedules=schedules)
    if lead == "rivalry_home_dog":
        return attach_rivalry_home_dog_flag(features, schedules=schedules)
    if lead == "altitude_cold":
        return attach_altitude_cold_home_flag(features)
    raise ValueError(f"no scoring attacher for lead {lead!r}")


# ---------------------------------------------------------------------------
# Shared walk-forward harness (mirrors scripts/cfb_option_prep_screen.py)
# ---------------------------------------------------------------------------


def run_walk_forward(
    attached: pd.DataFrame,
    scored_seasons: tuple[int, ...],
    candidate_column: str,
    *,
    leak_treatment: bool,
) -> pd.DataFrame:
    completed = attached.loc[
        pd.to_numeric(attached["result"], errors="coerce").notna()
        & pd.to_numeric(attached["ats_margin"], errors="coerce").notna()
    ].copy()
    candidate_source = completed
    if leak_treatment:
        candidate_source = completed.copy()
        candidate_source[candidate_column] = pd.to_numeric(
            candidate_source["ats_margin"], errors="coerce"
        )
    candidate_columns = (*CFB_MODEL_FEATURE_COLUMNS, candidate_column)
    scored = completed.loc[completed["season"].astype(int).isin(scored_seasons)]
    rows: list[dict[str, Any]] = []
    for (season_value, week_value), group in scored.groupby(["season", "week"], sort=True):
        cutoff = group["gameday"].min()
        baseline_training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(baseline_training) < CFB_BENCHMARK_MIN_TRAIN_GAMES:
            continue
        candidate_training = candidate_source.loc[candidate_source["gameday"].lt(cutoff)]
        baseline_model = fit_cfb_residual_model(
            baseline_training,
            ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA,
            feature_columns=CFB_MODEL_FEATURE_COLUMNS,
        )
        candidate_model = fit_cfb_residual_model(
            candidate_training,
            ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA,
            feature_columns=candidate_columns,
        )
        candidate_scoring = (
            group
            if not leak_treatment
            else group.assign(
                **{candidate_column: pd.to_numeric(group["ats_margin"], errors="coerce")}
            )
        )
        settle_margin = pd.to_numeric(group["result"], errors="coerce") - pd.to_numeric(
            group["spread_line"], errors="coerce"
        )
        baseline_probability = baseline_model.predict(group)["home_cover_probability"]
        candidate_probability = candidate_model.predict(candidate_scoring)["home_cover_probability"]
        for game_id, margin, base, cand, feature_value in zip(
            group["game_id"],
            settle_margin,
            baseline_probability,
            candidate_probability,
            group[candidate_column],
            strict=True,
        ):
            rows.append(
                {
                    "game_id": game_id,
                    "season": int(str(season_value)),
                    "week": int(str(week_value)),
                    "settle_margin": margin,
                    "baseline_probability": base,
                    "candidate_probability": cand,
                    "feature_value": feature_value,
                }
            )
    return pd.DataFrame(rows)


def grade(frame: pd.DataFrame, margins: pd.Series | None = None) -> pd.DataFrame:
    settle = frame["settle_margin"] if margins is None else margins
    graded = frame.copy()
    for arm, column in (
        ("baseline", "baseline_probability"),
        ("candidate", "candidate_probability"),
    ):
        graded[f"{arm}_correct"] = pick_correct(graded[column].ge(0.5), settle)
    return graded


def _paired_metric(reference: str, candidate: str) -> Any:
    def metric(df: pd.DataFrame) -> dict[str, float]:
        valid = df.dropna(subset=[reference, candidate])
        if valid.empty:
            return {
                "delta_accuracy": float("nan"),
                "candidate_accuracy": float("nan"),
                "reference_accuracy": float("nan"),
            }
        return {
            "delta_accuracy": float((valid[candidate] - valid[reference]).mean()),
            "candidate_accuracy": float(valid[candidate].mean()),
            "reference_accuracy": float(valid[reference].mean()),
        }

    return metric


def week_positions(frame: pd.DataFrame) -> list[np.ndarray]:
    return [
        np.asarray(positions, dtype=np.intp)
        for positions in frame.groupby(["season", "week"], sort=False).indices.values()
    ]


def null_distribution(frame: pd.DataFrame, *, permutations: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    metric = _paired_metric("baseline_correct", "candidate_correct")
    groups = week_positions(frame)
    deltas = []
    for _ in range(permutations):
        values = frame["settle_margin"].to_numpy(dtype=float, copy=True)
        for positions in groups:
            values[positions] = rng.permutation(values[positions])
        permuted = pd.Series(values, index=frame.index)
        deltas.append(metric(grade(frame, permuted))["delta_accuracy"])
    values_arr = np.asarray(deltas, dtype=float)
    finite = values_arr[np.isfinite(values_arr)]
    observed = metric(grade(frame))["delta_accuracy"]
    return {
        "permutations": len(finite),
        "null_mean_delta": float(finite.mean()),
        "null_sd_delta": float(finite.std(ddof=1)),
        "null_q025": float(np.quantile(finite, 0.025)),
        "null_q975": float(np.quantile(finite, 0.975)),
        "observed_delta": float(observed),
        "fraction_of_null_below_observed": float((finite < observed).mean()),
    }


def summarize_pair(paired: pd.DataFrame, samples: int, seed: int) -> dict[str, Any] | None:
    if paired.empty or paired.dropna(subset=["baseline_correct", "candidate_correct"]).empty:
        return None
    metric = _paired_metric("baseline_correct", "candidate_correct")
    point = metric(paired)
    week = week_blocked_bootstrap(paired, metric, block="week", samples=samples, seed=seed)
    week_row = week.loc[week["metric"].eq("delta_accuracy")].iloc[0]
    feature_value = pd.to_numeric(paired["feature_value"], errors="coerce")
    summary: dict[str, Any] = {
        "delta_accuracy": point["delta_accuracy"],
        "candidate_accuracy": point["candidate_accuracy"],
        "baseline_accuracy": point["reference_accuracy"],
        "week_blocked_ci95": [float(week_row["lower"]), float(week_row["upper"])],
        "week_blocked_probability_positive": float(week_row["probability_positive"]),
        "n_games": len(paired.dropna(subset=["baseline_correct", "candidate_correct"])),
        "n_weeks": int(paired[["season", "week"]].drop_duplicates().shape[0]),
        "n_seasons": int(paired["season"].nunique()),
        # NaN (missing feature value, e.g. a season-opener rest gap) must NOT
        # count as "flagged nonzero" -- pandas' != treats NaN as unequal to 0,
        # so notna() is required alongside ne(0.0).
        "n_flagged_nonzero": int((feature_value.notna() & feature_value.ne(0.0)).sum()),
    }
    if paired["season"].nunique() >= 2:
        season = week_blocked_bootstrap(paired, metric, block="season", samples=samples, seed=seed)
        season_row = season.loc[season["metric"].eq("delta_accuracy")].iloc[0]
        summary["season_blocked_ci95"] = [float(season_row["lower"]), float(season_row["upper"])]
        summary["season_blocked_probability_positive"] = float(season_row["probability_positive"])
    else:
        summary["season_blocked_ci95"] = None
        summary["season_blocked_probability_positive"] = None
    return summary


def _print_screen_summary(result: dict[str, Any]) -> None:
    pooled = result["candidate_vs_baseline_pooled"]
    low, high = pooled["week_blocked_ci95"]
    print(
        f"pooled: delta {pooled['delta_accuracy'] * 100:+.3f} pts  P+ "
        f"{pooled['week_blocked_probability_positive']:.3f}  week 95% "
        f"[{low * 100:+.3f}, {high * 100:+.3f}]  n={pooled['n_games']} games, "
        f"{pooled['n_weeks']} weeks, flagged={pooled['n_flagged_nonzero']}"
    )
    null = result["permutation_null"]
    print(
        f"null: mean {null['null_mean_delta'] * 100:+.3f}, observed at the "
        f"{null['fraction_of_null_below_observed'] * 100:.1f}th percentile"
    )
    for label, _start, _end in ERAS:
        era = result["era_results"][label]
        if era is None:
            print(f"era {label}: no scored games")
        else:
            elow, ehigh = era["week_blocked_ci95"]
            print(
                f"era {label}: delta {era['delta_accuracy'] * 100:+.3f} pts  P+ "
                f"{era['week_blocked_probability_positive']:.3f}  "
                f"[{elow * 100:+.3f}, {ehigh * 100:+.3f}]"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lead",
        choices=("post_bye", "rivalry_home_dog", "altitude_cold", "sandwich"),
        required=True,
    )
    parser.add_argument(
        "--mode", choices=("coverage", "null", "positive-control", "screen"), required=True
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.lead == "sandwich":
        print(SANDWICH_GAP_MESSAGE, flush=True)
        return 0

    print(f"=== loading features (lead={args.lead}) ===", flush=True)
    features = pd.read_parquet(args.features)
    features["gameday"] = pd.to_datetime(features["gameday"], errors="raise")
    candidate_column = CANDIDATE_COLUMNS[args.lead]
    attached = attach_candidate(args.lead, features)
    attached = attached.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    if args.mode == "coverage":
        clean = attached.loc[attached["season"].astype(int).isin(CFB_CLEAN_CORE_SEASONS)]
        values = pd.to_numeric(clean[candidate_column], errors="coerce")
        nonzero = values.notna() & values.ne(0.0)
        print(f"clean-core games: {len(clean)}")
        print(f"flagged nonzero: {int(nonzero.sum())}")
        print(f"missing: {int(values.isna().sum())}")
        if args.lead == "post_bye":
            print(f"flagged +1 (home off bye, away not): {int((values == 1).sum())}")
            print(f"flagged -1 (away off bye, home not): {int((values == -1).sum())}")
        if args.lead == "rivalry_home_dog":
            print(f"rivalry-pair games (any spread): {int(clean['_lead50_is_rivalry_pair'].sum())}")
            print(f"home-underdog games (any pair): {int(clean['_lead50_home_underdog'].sum())}")
        print("flagged-nonzero by season:")
        print(nonzero.groupby(clean["season"]).sum().to_string())
        return 0

    fitted = run_walk_forward(
        attached,
        tuple(CFB_CLEAN_CORE_SEASONS),
        candidate_column,
        leak_treatment=args.mode == "positive-control",
    )
    if fitted.empty:
        print("no scored games")
        return 1
    if args.mode == "null":
        null = null_distribution(fitted, permutations=args.permutations, seed=args.seed)
        print(json.dumps(null, indent=2))
        return 0

    graded = grade(fitted)
    result: dict[str, Any] = {
        "status": "scored",
        "lead": args.lead,
        "candidate_column": candidate_column,
        "grade": "close_proxy_median_book_spread_line",
        "league": "cfb",
        "seed": args.seed,
        "candidate_vs_baseline_pooled": summarize_pair(
            graded, samples=args.bootstrap_samples, seed=args.seed
        ),
        "permutation_null": null_distribution(
            fitted, permutations=args.permutations, seed=args.seed
        ),
        "era_results": {
            label: (
                summarize_pair(
                    graded.loc[graded["season"].between(start, end)],
                    samples=args.bootstrap_samples,
                    seed=args.seed,
                )
            )
            for label, start, end in ERAS
        },
        "home_pick_rate": {
            "baseline": float(graded["baseline_probability"].ge(0.5).mean()),
            "candidate": float(graded["candidate_probability"].ge(0.5).mean()),
        },
    }
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = ARTIFACT_ROOT / args.lead / stamp
    configuration = {
        "cell": args.lead,
        "predicted_direction": PREDICTED_DIRECTION[args.lead],
        "mode": args.mode,
        "scored_seasons": list(CFB_CLEAN_CORE_SEASONS),
        "grade": "close_proxy_median_book_spread_line",
        "league": "cfb",
        "baseline_feature_columns": list(CFB_MODEL_FEATURE_COLUMNS),
        "candidate_column": candidate_column,
        "regressor": "ridge",
        "ridge_alpha": CFB_BENCHMARK_RIDGE_ALPHA,
        "min_train_games": CFB_BENCHMARK_MIN_TRAIN_GAMES,
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "seed": args.seed,
        "predeclaration": "docs/cfb_lead_screens_wave1.md",
        "features_path": str(args.features),
    }
    payload = {
        **configuration,
        "result": result,
        "provenance": artifact_provenance(configuration, args.features, project_root=REPO_ROOT),
    }
    write_experiment_artifact(
        out_dir,
        "results.json",
        payload,
        command=f"cfb-lead-screens-wave1-{args.lead}",
        metrics={"cell": args.lead, "mode": args.mode, "status": result.get("status")},
        notes=(
            f"Free CFB screen wave 1, lead={args.lead}, on the frozen XLG-03 "
            "benchmark arm; no NFL window spent. See "
            "docs/cfb_lead_screens_wave1.md."
        ),
    )
    _print_screen_summary(result)
    print(f"wrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
