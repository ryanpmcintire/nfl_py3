"""Section 5, experiment 1 of ``docs/injury_news_sourcing.md``: does the pool's
Tuesday-noon lock already price the injury information a Saturday-cutoff
feature (``decision_hours_before_kickoff=24``) sees?

Free re-read precedent, no new rotation window: this rebuilds
``injury_value_lost_narrowed`` (``docs/injury_value_lost.md`` sec 4, D-A
contrast) on the ALREADY-SPENT ``mod07_weak_signal_stack`` ``[2020, 2021]``
opener window, exactly as ``scripts/availability_ablation.py`` already does.
Nothing here touches ``rotation_registry.json`` (no ``assign``, no
``record_look``) and the ``[2022, 2023]`` window this family would draw next
is never referenced. Three arms, all built from the identical
player/pbp/player-value snapshots the frozen ``game_features_player_value.parquet``
manifest recorded (``player_snapshot=20260812T200527Z``,
``pbp_snapshot=20260812T142851Z``, ``player_value_snapshot=20260813T121050Z``):

* ``saturday`` -- fresh rebuild at ``decision_hours_before_kickoff=24``
  (Saturday), a reproduction of the on-disk table used for the +1.316 pt
  finding.
* ``tuesday_official`` -- every injury-report row whose OWN ``date_modified``
  is at or before that game's own-week Tuesday noon ET (no PFT text used).
* ``tuesday_pft`` -- ``tuesday_official`` plus any row whose player has a
  ProFootballTalk injury-relevant headline (``data/raw/injury_news``, 299,739
  URLs ingested this session) mentioning them, dated at or before that same
  Tuesday-noon cutoff. A designation foreshadowed by an earlier PFT post
  counts as known; one filed cold on Wednesday with no PFT mention does not.

Mechanism (why one function call per arm is enough): ``decision_hours_before_
kickoff`` only gates which rows of the injuries table are "visible" for the
CURRENT game being scored (``players.py:_injury_rows_asof``); every other
piece of accumulated state (snap shares, production, QB ratings, roster
continuity) is built from strictly prior COMPLETED games and is untouched by
this parameter. So instead of the (day-of-week-dependent) hours offset,
each Tuesday arm pre-filters the INJURIES DATAFRAME to rows that pass its own
per-game Tuesday-noon test, then calls the unmodified
``nfl_ats.players.enrich_with_player_features`` with
``decision_hours_before_kickoff=0`` (decision_at = kickoff): because every
surviving row's own timestamp is already <= that game's Tuesday noon, which
is always well before kickoff, the function's own internal filter never
removes anything further. ``players.py`` itself is never edited.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/injury_tuesday_cutoff_experiment.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import opener_pick_evaluation, week_blocked_bootstrap
from nfl_ats.modeling import regular_season_rows
from nfl_ats.pbp import load_pbp_snapshot
from nfl_ats.pbp import snapshot_from_root as pbp_snapshot_from_root
from nfl_ats.players import (
    enrich_with_player_features,
    load_player_snapshot,
    load_player_value_snapshot,
    player_snapshot_from_root,
    player_value_snapshot_from_root,
)
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact
from nfl_ats.rotation import load_registry

REPO = Path(__file__).resolve().parents[1]

REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0
MIN_TRAIN_GAMES = 500
SAMPLES = 20000
SEED = 20260819

# Exact snapshots the on-disk game_features_player_value.parquet manifest
# recorded (data/processed/game_features_player_value.manifest.json, read).
PLAYER_SNAPSHOT_ID = "20260812T200527Z"
PBP_SNAPSHOT_ID = "20260812T142851Z"
PLAYER_VALUE_SNAPSHOT_ID = "20260813T121050Z"
PFT_SNAPSHOT_ID = "20260819T191639Z"

# The recorded D-A "cleaner isolation" number from docs/injury_value_lost.md
# sec 4, quoted here as the reproduction target for the fresh Saturday rebuild.
RECORDED_SATURDAY_D_MINUS_A = {
    "delta_points": 1.316,
    "probability_positive": 0.8875,
}

_CURLY_APOSTROPHE = chr(0x2019)
NAME_PUNCTUATION = re.compile("[." + "'" + _CURLY_APOSTROPHE + "]")
NAME_NONALNUM = re.compile(r"[^a-z0-9]+")


def _normalize_name(name: str) -> str:
    text = str(name).lower()
    text = NAME_PUNCTUATION.sub("", text)
    text = NAME_NONALNUM.sub(" ", text)
    return " ".join(text.split())


def _config(profile: str) -> dict[str, Any]:
    return {
        "feature_profile": profile,
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "target": "market_residual",
    }


# ---------------------------------------------------------------------------
# Tuesday-noon-ET cutoff per (season, week, team)
# ---------------------------------------------------------------------------


def team_week_tuesday_noon(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) with that game's own-week Tuesday noon ET.

    Mirrors ``nfl_ats.clv.live_tuesday_openers``'s own-week-Tuesday convention
    (``(weekday - 1) % 7`` days back from kickoff), but anchored at noon ET
    (the pool's stated lock time, ``docs/pool_edge_plan.md`` line 80) rather
    than UTC midnight, and returned in UTC for direct comparison against
    ``date_modified``/PFT ``lastmod``.
    """

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
    long["tuesday_noon_utc"] = tuesday_noon_et.dt.tz_convert("UTC")
    long["kickoff_utc"] = long["kickoff"]
    return long[["season", "week", "team", "game_id", "kickoff_utc", "tuesday_noon_utc"]]


# ---------------------------------------------------------------------------
# PFT article <-> player matching
# ---------------------------------------------------------------------------


def build_player_name_map(rosters: pd.DataFrame) -> dict[str, str]:
    """gsis_id -> normalized full name, majority vote across weekly rows."""

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


def build_pft_match_table(
    injuries: pd.DataFrame,
    name_map: dict[str, str],
    pft: pd.DataFrame,
    cutoffs: pd.DataFrame,
    *,
    lookback_days: float,
    window_end_col: str = "tuesday_noon_utc",
) -> pd.DataFrame:
    """For each (season, week, team, gsis_id) row, earliest matching PFT lastmod
    within ``[cutoff - lookback_days, cutoff]``, else NaT.

    Matching is deliberately conservative (precision over recall): a hit
    requires the player's full normalized name (first + last) as a literal
    substring of the article's normalized headline text. This will UNDERCOUNT
    last-name-only headlines ("Fuller doubtful for Sunday"); any measured
    overlap fraction below is therefore a lower bound on true PFT foreshadowing.
    """

    relevant = pft.loc[pft["injury_relevant"]].copy()
    relevant["headline_norm"] = relevant["headline_guess"].map(_normalize_name)
    # Inverted index on the last name token (>=3 chars) to avoid an O(rows x articles) scan.
    last_name_index: dict[str, list[int]] = {}
    headlines = relevant["headline_norm"].to_numpy()
    # Naive UTC datetime64[ns] throughout below: avoids tz-aware/object-array
    # dtype pitfalls in numpy while staying an exact UTC instant comparison
    # (every timestamp involved -- lastmod, kickoff, tuesday_noon -- already
    # went through tz_convert("UTC") upstream).
    lastmods = (
        relevant["lastmod"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )

    tokens_per_article = relevant["headline_norm"].str.split()
    for idx, tokens in enumerate(tokens_per_article):
        for token in set(tokens):
            if len(token) >= 3:
                last_name_index.setdefault(token, []).append(idx)

    merged = injuries.merge(cutoffs, on=["season", "week", "team"], how="left")
    merged["player_name"] = merged["gsis_id"].map(name_map)

    window_end_naive = merged[window_end_col].dt.tz_convert("UTC").dt.tz_localize(None)
    window_start_naive = window_end_naive - pd.Timedelta(days=lookback_days)
    window_end = window_end_naive.to_numpy(dtype="datetime64[ns]")
    window_start = window_start_naive.to_numpy(dtype="datetime64[ns]")

    nat = np.datetime64("NaT")
    earliest_match = np.full(len(merged), nat, dtype="datetime64[ns]")
    player_names = merged["player_name"].to_numpy(dtype=object)

    for row_pos in range(len(merged)):
        name = player_names[row_pos]
        w_end = window_end[row_pos]
        if name is None or (isinstance(name, float) and np.isnan(name)) or np.isnat(w_end):
            continue
        name_tokens = name.split()
        if not name_tokens:
            continue
        last_token = name_tokens[-1]
        candidates = last_name_index.get(last_token)
        if not candidates:
            continue
        w_start = window_start[row_pos]
        best = None
        for cand_idx in candidates:
            if name not in headlines[cand_idx]:
                continue
            lm = lastmods[cand_idx]
            if lm < w_start or lm > w_end:
                continue
            if best is None or lm < best:
                best = lm
        if best is not None:
            earliest_match[row_pos] = best

    merged["pft_match_lastmod"] = pd.Series(earliest_match, index=merged.index).dt.tz_localize(
        "UTC"
    )
    return merged


# ---------------------------------------------------------------------------
# Arm construction (re-derived from scripts/availability_ablation.py)
# ---------------------------------------------------------------------------


def spent_window_split(
    features: pd.DataFrame, seasons: tuple[int, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    span = list(range(int(seasons[0]), int(seasons[1]) + 1))
    missing = [s for s in span if s not in set(frame["season"].astype(int))]
    if missing:
        raise ValueError(f"Feature table is missing window seasons: {missing}")
    window = frame.loc[frame["season"].astype(int).isin(span)].copy()
    cutoff = window["gameday"].min()
    training = frame.loc[frame["gameday"].lt(cutoff) & frame["result"].notna()].copy()
    if training.empty:
        raise ValueError("No completed games before the window; cannot forward-chain")
    return training, window


def arm(
    features: pd.DataFrame,
    *,
    seasons: tuple[int, int],
    profile: str,
    market_root: Path,
    min_train_games: int,
) -> pd.DataFrame:
    training, window = spent_window_split(features, seasons)
    scoped = pd.concat([training, window], ignore_index=True)
    scored = opener_pick_evaluation(
        market_root,
        scoped,
        active_model_config=_config(profile),
        min_train_games=min_train_games,
    )
    span = sorted(window["season"].astype(int).unique())
    scored = scored.loc[scored["season"].isin(span)]
    return scored.loc[scored["correct_at_open"].notna()].copy()


def paired_frame(left_arm: pd.DataFrame, right_arm: pd.DataFrame) -> pd.DataFrame:
    left = left_arm[["game_id", "season", "week", "correct_at_open", "pick_home_at_open"]].rename(
        columns={"correct_at_open": "left_correct", "pick_home_at_open": "left_pick_home"}
    )
    right = right_arm[["game_id", "correct_at_open", "pick_home_at_open"]].rename(
        columns={"correct_at_open": "right_correct", "pick_home_at_open": "right_pick_home"}
    )
    paired = left.merge(right, on="game_id", how="inner")
    paired["left_correct"] = paired["left_correct"].astype(float)
    paired["right_correct"] = paired["right_correct"].astype(float)
    return paired.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def _metric_fn(frame: pd.DataFrame) -> dict[str, float]:
    return {"right_minus_left": float(frame["right_correct"].mean() - frame["left_correct"].mean())}


def contrast(left_arm: pd.DataFrame, right_arm: pd.DataFrame, *, name: str) -> dict[str, Any]:
    paired = paired_frame(left_arm, right_arm)
    if paired.empty:
        raise ValueError(f"{name}: no game scored by both arms")
    bootstrap = week_blocked_bootstrap(paired, _metric_fn, block="week", samples=SAMPLES, seed=SEED)
    row = bootstrap.iloc[0]
    disagree = paired.loc[paired["left_pick_home"].ne(paired["right_pick_home"])]
    return {
        "contrast": name,
        "paired_games": len(paired),
        "weeks": int(paired.groupby(["season", "week"]).ngroups),
        "left_accuracy": float(paired["left_correct"].mean()),
        "right_accuracy": float(paired["right_correct"].mean()),
        "delta_points": float(row["estimate"]) * 100.0,
        "bootstrap_lower_points": float(row["lower"]) * 100.0,
        "bootstrap_upper_points": float(row["upper"]) * 100.0,
        "probability_positive": float(row["probability_positive"]),
        "picks_disagreeing": len(disagree),
    }, paired


def build_enriched(
    *,
    games: pd.DataFrame,
    injuries: pd.DataFrame,
    rosters: pd.DataFrame,
    snaps: pd.DataFrame,
    pbp: pd.DataFrame,
    player_stats: pd.DataFrame,
    decision_hours_before_kickoff: float,
) -> pd.DataFrame:
    return enrich_with_player_features(
        games,
        injuries,
        rosters,
        snaps,
        pbp,
        player_stats,
        decision_hours_before_kickoff=decision_hours_before_kickoff,
        role_span=8,
        qb_span=12,
        qb_min_dropbacks=20,
        offseason_retention=0.75,
        value_span=16,
        value_prior_snaps=200.0,
    )


def cutoff_delta(
    arm_a_sat: pd.DataFrame,
    arm_d_sat: pd.DataFrame,
    arm_a_tue: pd.DataFrame,
    arm_d_tue: pd.DataFrame,
    *,
    name: str,
) -> dict[str, Any]:
    """Paired Saturday-D-minus-A vs Tuesday-D-minus-A: the Tuesday->Saturday channel."""

    sat = paired_frame(arm_a_sat, arm_d_sat).rename(
        columns={"left_correct": "sat_a", "right_correct": "sat_d"}
    )[["game_id", "season", "week", "sat_a", "sat_d"]]
    tue = paired_frame(arm_a_tue, arm_d_tue).rename(
        columns={"left_correct": "tue_a", "right_correct": "tue_d"}
    )[["game_id", "tue_a", "tue_d"]]
    merged = sat.merge(tue, on="game_id", how="inner")
    merged["sat_diff"] = merged["sat_d"] - merged["sat_a"]
    merged["tue_diff"] = merged["tue_d"] - merged["tue_a"]

    def metric(frame: pd.DataFrame) -> dict[str, float]:
        return {"channel_delta": float(frame["sat_diff"].mean() - frame["tue_diff"].mean())}

    bootstrap = week_blocked_bootstrap(merged, metric, block="week", samples=SAMPLES, seed=SEED)
    row = bootstrap.iloc[0]
    return {
        "contrast": name,
        "paired_games": len(merged),
        "weeks": int(merged.groupby(["season", "week"]).ngroups),
        "saturday_D_minus_A_points": float(merged["sat_diff"].mean()) * 100.0,
        "tuesday_D_minus_A_points": float(merged["tue_diff"].mean()) * 100.0,
        "channel_delta_points": float(row["estimate"]) * 100.0,
        "bootstrap_lower_points": float(row["lower"]) * 100.0,
        "bootstrap_upper_points": float(row["upper"]) * 100.0,
        "probability_positive": float(row["probability_positive"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="mod07_weak_signal_stack")
    parser.add_argument(
        "--features", type=Path, default=REPO / "data/processed/game_features_pbp.parquet"
    )
    parser.add_argument("--market-root", type=Path, default=REPO / "data/market/raw")
    parser.add_argument("--lookback-days", type=float, default=9.0)
    parser.add_argument("--exp2-lookback-days", type=float, default=2.25)
    parser.add_argument(
        "--out", type=Path, default=REPO / "artifacts/injury_tuesday_cutoff/result.json"
    )
    args = parser.parse_args()

    registry = load_registry(None)
    declared = registry.families[args.family]
    spent = [w for w in declared.windows if w.state == "spent"]
    if not spent:
        raise SystemExit(f"{args.family} has no spent window.")
    seasons = (int(spent[-1].seasons[0]), int(spent[-1].seasons[1]))

    games = pd.read_parquet(args.features)
    player_root = REPO / "data/players/raw"
    pbp_root = REPO / "data/pbp/raw"
    player_value_root = REPO / "data/players/values/raw"

    player_snapshot = player_snapshot_from_root(player_root / PLAYER_SNAPSHOT_ID)
    pbp_snapshot = pbp_snapshot_from_root(pbp_root / PBP_SNAPSHOT_ID)
    player_value_snapshot = player_value_snapshot_from_root(
        player_value_root / PLAYER_VALUE_SNAPSHOT_ID
    )

    injuries, rosters, snaps = load_player_snapshot(player_snapshot)
    pbp = load_pbp_snapshot(pbp_snapshot)
    player_stats = load_player_value_snapshot(player_value_snapshot)

    pft = pd.read_parquet(REPO / f"data/raw/injury_news/{PFT_SNAPSHOT_ID}/index.parquet")

    cutoffs = team_week_tuesday_noon(games)
    name_map = build_player_name_map(rosters)

    matched_exp1 = build_pft_match_table(
        injuries, name_map, pft, cutoffs, lookback_days=args.lookback_days
    )

    # -----------------------------------------------------------------
    # Experiment 1: three injuries variants
    # -----------------------------------------------------------------
    official_visible = matched_exp1["date_modified"] <= matched_exp1["tuesday_noon_utc"]
    pft_augmented_visible = official_visible | matched_exp1["pft_match_lastmod"].notna()

    injury_cols = list(injuries.columns)
    injuries_tuesday_official = matched_exp1.loc[official_visible, injury_cols].reset_index(
        drop=True
    )
    injuries_tuesday_pft = matched_exp1.loc[pft_augmented_visible, injury_cols].reset_index(
        drop=True
    )

    coverage = {
        "official_injury_rows_total": len(injuries),
        "official_rows_visible_by_tuesday_noon": int(official_visible.sum()),
        "official_rows_visible_share": float(official_visible.mean()),
        "additional_rows_from_pft_foreshadowing": int(
            (pft_augmented_visible & ~official_visible).sum()
        ),
        "pft_augmented_rows_visible_share": float(pft_augmented_visible.mean()),
    }

    print("Building Saturday-cutoff arm (fresh rebuild, decision_hours=24)...")
    enriched_saturday = build_enriched(
        games=games,
        injuries=injuries,
        rosters=rosters,
        snaps=snaps,
        pbp=pbp,
        player_stats=player_stats,
        decision_hours_before_kickoff=24,
    )
    print("Building Tuesday-official-only arm (decision_hours=0, pre-filtered injuries)...")
    enriched_tue_official = build_enriched(
        games=games,
        injuries=injuries_tuesday_official,
        rosters=rosters,
        snaps=snaps,
        pbp=pbp,
        player_stats=player_stats,
        decision_hours_before_kickoff=0,
    )
    print("Building Tuesday+PFT arm (decision_hours=0, pre-filtered injuries)...")
    enriched_tue_pft = build_enriched(
        games=games,
        injuries=injuries_tuesday_pft,
        rosters=rosters,
        snaps=snaps,
        pbp=pbp,
        player_stats=player_stats,
        decision_hours_before_kickoff=0,
    )

    results: dict[str, Any] = {"window": list(seasons), "grade": "opener", "coverage": coverage}

    for label, enriched in (
        ("saturday", enriched_saturday),
        ("tuesday_official", enriched_tue_official),
        ("tuesday_pft", enriched_tue_pft),
    ):
        arm_a = arm(
            enriched,
            seasons=seasons,
            profile="player",
            market_root=args.market_root,
            min_train_games=MIN_TRAIN_GAMES,
        )
        arm_d = arm(
            enriched,
            seasons=seasons,
            profile="player_value",
            market_root=args.market_root,
            min_train_games=MIN_TRAIN_GAMES,
        )
        c, _ = contrast(arm_a, arm_d, name=f"D_minus_A_{label}")
        results[label] = {"contrast": c}
        results[label]["_arm_a"] = arm_a
        results[label]["_arm_d"] = arm_d

    saturday_reproduction_check = {
        "recorded_delta_points": RECORDED_SATURDAY_D_MINUS_A["delta_points"],
        "reproduced_delta_points": results["saturday"]["contrast"]["delta_points"],
        "recorded_probability_positive": RECORDED_SATURDAY_D_MINUS_A["probability_positive"],
        "reproduced_probability_positive": results["saturday"]["contrast"]["probability_positive"],
    }

    delta_official = cutoff_delta(
        results["saturday"]["_arm_a"],
        results["saturday"]["_arm_d"],
        results["tuesday_official"]["_arm_a"],
        results["tuesday_official"]["_arm_d"],
        name="tuesday_to_saturday_channel_official_only",
    )
    delta_pft = cutoff_delta(
        results["saturday"]["_arm_a"],
        results["saturday"]["_arm_d"],
        results["tuesday_pft"]["_arm_a"],
        results["tuesday_pft"]["_arm_d"],
        name="tuesday_to_saturday_channel_pft_augmented",
    )

    for label in ("saturday", "tuesday_official", "tuesday_pft"):
        del results[label]["_arm_a"]
        del results[label]["_arm_d"]

    results["saturday_reproduction_check"] = saturday_reproduction_check
    results["tuesday_to_saturday_delta"] = {
        "vs_official_only_tuesday": delta_official,
        "vs_pft_augmented_tuesday": delta_pft,
    }

    # -----------------------------------------------------------------
    # Experiment 2: fraction of Friday-final designations already
    # headline-visible by Tuesday noon (Sunday-night -> Tuesday-morning window)
    # -----------------------------------------------------------------
    final_report = (
        injuries.sort_values("date_modified")
        .drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")
        .reset_index(drop=True)
    )
    final_matched = build_pft_match_table(
        final_report, name_map, pft, cutoffs, lookback_days=args.exp2_lookback_days
    )
    has_match = final_matched["pft_match_lastmod"].notna()
    exp2 = {
        "window_description": (
            "Sunday ~18:00 ET through Tuesday 12:00 ET of the same market week "
            f"({args.exp2_lookback_days} days back from each row's own Tuesday-noon cutoff)"
        ),
        "friday_final_designations_total": len(final_matched),
        "already_headline_visible_by_tuesday_noon": int(has_match.sum()),
        "fraction_already_visible": float(has_match.mean()),
        "by_report_status": final_matched.assign(has_match=has_match)
        .groupby("report_status")["has_match"]
        .agg(["size", "mean"])
        .rename(columns={"size": "n", "mean": "fraction_visible"})
        .to_dict(orient="index"),
        "matching_note": (
            "Conservative full-name substring match against PFT injury-relevant "
            "headlines; undercounts last-name-only headlines, so this fraction is "
            "a LOWER BOUND on true pre-Tuesday visibility."
        ),
    }
    results["experiment_2_thin_slice_check"] = exp2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    configuration = {
        "command": "injury-tuesday-cutoff-experiment",
        "family": args.family,
        "features": str(args.features),
        "market_root": str(args.market_root),
        "lookback_days": args.lookback_days,
        "exp2_lookback_days": args.exp2_lookback_days,
    }
    results["provenance"] = artifact_provenance(configuration, args.features, project_root=REPO)
    write_experiment_artifact(
        args.out.parent,
        args.out.name,
        results,
        command="injury-tuesday-cutoff-experiment",
        metrics=results,
        notes=(
            "Tuesday-noon-visible injury arms (official-only and PFT-augmented) vs. the "
            "Saturday-cutoff reproduction, on the already-spent mod07_weak_signal_stack "
            "opener window -- no new rotation window drawn."
        ),
    )
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
