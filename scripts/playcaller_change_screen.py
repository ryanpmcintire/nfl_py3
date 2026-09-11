from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.bye_edge_fade_overlay import bye_edge_flag_by_game  # noqa: E402
from nfl_ats.clv import (  # noqa: E402
    opener_pick_evaluation,
    pick_correct,
    resolve_active_model_config,
    week_blocked_bootstrap,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.evidence_conventions import probability_positive_from_draws  # noqa: E402
from nfl_ats.provenance import write_stamped_artifact  # noqa: E402
from nfl_ats.rotation import confirmation_split, load_registry  # noqa: E402

GAME_FEATURES = REPO / "data" / "processed" / "game_features.parquet"
PRODUCTION_FEATURES = REPO / "data" / "processed" / "game_features_weak_stack.parquet"
OPENER_ARCHIVE = REPO / "artifacts" / "opener_evaluation" / "20260907T152026Z" / "per_game.parquet"
MARKET_ROOT = REPO / "data" / "market" / "raw"
ARTIFACTS_ROOT = REPO / "artifacts"
OUT_DIR = REPO / "artifacts" / "playcaller_change_leads"

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260817
HISTORY_LAST_SEASON = 2025

FAMILY_LEAD29_PROD = "lead29_playcaller_first_game_on_production"
FAMILY_LEAD28_PROD = "lead28_post_bye_new_playcaller_on_production"


def _latest_schedules() -> Path:
    candidates = sorted((REPO / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise SystemExit("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


def _latest_coordinator_dir() -> Path:
    candidates = sorted(
        (REPO / "data" / "raw" / "coordinators").glob("*/coordinator_history.parquet")
    )
    if not candidates:
        raise SystemExit("no data/raw/coordinators/*/coordinator_history.parquet snapshot found")
    return candidates[-1].parent


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def _normalize_person(name: Any) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(name)).strip().casefold()


def _naive_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_localize(None)


def load_counted_events() -> list[dict[str, Any]]:
    coord_dir = _latest_coordinator_dir()
    validation = json.loads((coord_dir / "change_validation.json").read_text(encoding="utf-8"))
    counted = []
    for row in validation["changes"]:
        if row["validation_status"] not in ("staff_change", "playcaller_role"):
            continue
        counted.append(
            {
                "season": int(row["season"]),
                "team": row["team"],
                "role": row["role"],
                "previous_person": row["previous_person"],
                "person": row["person"],
                "revision_at": pd.Timestamp(row["revision_at"]),
                "validation_status": row["validation_status"],
            }
        )
    counted.sort(key=lambda r: (r["team"], r["revision_at"]))
    return counted


def load_oc_tenure_table() -> pd.DataFrame:
    coord_dir = _latest_coordinator_dir()
    history = pd.read_parquet(coord_dir / "coordinator_history.parquet")
    preseason_oc = history.loc[
        history["sample_mode"].eq("preseason") & history["role"].eq("OC")
    ].copy()
    preseason_oc["team"] = _canonical_team(preseason_oc["team"])
    preseason_oc["person_norm"] = preseason_oc["person"].map(_normalize_person)
    preseason_oc["season"] = preseason_oc["season"].astype(int)
    lookup: dict[tuple[int, str], str] = {
        (int(row.season), row.team): row.person_norm for row in preseason_oc.itertuples(index=False)
    }
    seasons = sorted(preseason_oc["season"].unique())
    teams = sorted(preseason_oc["team"].unique())
    rows = []
    for season in seasons:
        for team in teams:
            oc_s = lookup.get((season, team))
            if oc_s is None:
                continue
            oc_s1 = lookup.get((season - 1, team))
            if oc_s1 is None:
                tenure = "unknown"
            elif oc_s != oc_s1:
                tenure = "1"
            else:
                oc_s2 = lookup.get((season - 2, team))
                if oc_s2 is None:
                    tenure = "unknown"
                elif oc_s1 == oc_s2:
                    tenure = "3+"
                else:
                    tenure = "2"
            rows.append({"season": season, "team": team, "oc_tenure": tenure})
    return pd.DataFrame(rows)


def load_bye_table() -> pd.DataFrame:
    schedules = pd.read_parquet(_latest_schedules())
    frame = bye_edge_flag_by_game(schedules)
    frame["game_id"] = frame["game_id"].astype(str)
    return frame


def build_close_team_long(features: pd.DataFrame) -> pd.DataFrame:
    reg = features.loc[features["game_type"].eq("REG")].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = reg["season"].astype(int)
    reg["week"] = reg["week"].astype(int)
    reg["game_id"] = reg["game_id"].astype(str)
    reg["kickoff"] = _naive_utc(reg["kickoff"])

    sides = []
    for is_home, team_col, opp_col in (
        (True, "home_team", "away_team"),
        (False, "away_team", "home_team"),
    ):
        side = pd.DataFrame(
            {
                "game_id": reg["game_id"],
                "season": reg["season"],
                "week": reg["week"],
                "team": reg[team_col],
                "opponent": reg[opp_col],
                "is_home": is_home,
                "team_covered": reg["home_cover"] if is_home else 1.0 - reg["home_cover"],
                "kickoff": reg["kickoff"],
            }
        )
        sides.append(side)
    long_df = pd.concat(sides, ignore_index=True)
    long_df = long_df.loc[long_df["team_covered"].notna()].reset_index(drop=True)
    long_df["week_block"] = long_df["season"] * 100 + long_df["week"]
    return long_df


def build_opener_team_long(archive: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    frame = archive.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    meta = meta.copy()
    meta["game_id"] = meta["game_id"].astype(str)
    meta["home_team"] = _canonical_team(meta["home_team"])
    meta["away_team"] = _canonical_team(meta["away_team"])
    meta["kickoff"] = _naive_utc(meta["kickoff"])
    frame = frame.merge(
        meta[["game_id", "home_team", "away_team", "kickoff"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    frame["home_cover"] = np.where(
        frame["margin_vs_open"].gt(0.0), 1.0, np.where(frame["margin_vs_open"].lt(0.0), 0.0, np.nan)
    )
    sides = []
    for is_home, team_col, opp_col in (
        (True, "home_team", "away_team"),
        (False, "away_team", "home_team"),
    ):
        side = pd.DataFrame(
            {
                "game_id": frame["game_id"],
                "season": frame["season"].astype(int),
                "week": frame["week"].astype(int),
                "team": frame[team_col],
                "opponent": frame[opp_col],
                "is_home": is_home,
                "team_covered": frame["home_cover"] if is_home else 1.0 - frame["home_cover"],
                "kickoff": frame["kickoff"],
            }
        )
        sides.append(side)
    long_df = pd.concat(sides, ignore_index=True)
    long_df = long_df.loc[long_df["team_covered"].notna()].reset_index(drop=True)
    long_df["week_block"] = long_df["season"] * 100 + long_df["week"]
    return long_df


def attach_games_after_change(long_df: pd.DataFrame, events: list[dict[str, Any]]) -> pd.DataFrame:
    events_by_team: dict[str, list[pd.Timestamp]] = {}
    for event in events:
        events_by_team.setdefault(event["team"], []).append(
            _naive_utc(pd.Series([event["revision_at"]])).iloc[0]
        )
    for team in events_by_team:
        events_by_team[team] = sorted(events_by_team[team])

    result = long_df.copy()
    result["game_number_after_change"] = np.nan
    for team, event_times in events_by_team.items():
        mask = result["team"].eq(team)
        if not mask.any():
            continue
        sub = result.loc[mask].sort_values("kickoff")
        event_array = np.array(event_times, dtype="datetime64[ns]")
        kickoffs = sub["kickoff"].to_numpy(dtype="datetime64[ns]")
        regime = np.searchsorted(event_array, kickoffs, side="left")
        sub = sub.assign(_regime=regime)
        flagged = sub.loc[sub["_regime"] > 0].copy()
        if flagged.empty:
            continue
        numbered = flagged.groupby("_regime").cumcount() + 1
        result.loc[flagged.index, "game_number_after_change"] = numbered.to_numpy()
    return result


def attach_oc_tenure(long_df: pd.DataFrame, tenure_table: pd.DataFrame) -> pd.DataFrame:
    return long_df.merge(tenure_table, on=["season", "team"], how="left")


def attach_bye_flag(long_df: pd.DataFrame, bye_table: pd.DataFrame) -> pd.DataFrame:
    merged = long_df.merge(
        bye_table[["game_id", "home_off_bye", "away_off_bye"]], on="game_id", how="left"
    )
    merged["own_off_bye"] = np.where(
        merged["is_home"], merged["home_off_bye"], merged["away_off_bye"]
    )
    merged["own_off_bye"] = merged["own_off_bye"].fillna(False).astype(bool)
    return merged.drop(columns=["home_off_bye", "away_off_bye"])


def game_level_signal(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    home = long_df.loc[long_df["is_home"], ["game_id", value_col]].rename(
        columns={value_col: "home_signal"}
    )
    away = long_df.loc[~long_df["is_home"], ["game_id", value_col]].rename(
        columns={value_col: "away_signal"}
    )
    return home.merge(away, on="game_id", how="outer")


def apply_tilt(
    baseline_pick_home: pd.Series, home_signal: pd.Series, away_signal: pd.Series
) -> pd.Series:
    home_only = home_signal.ne(0) & away_signal.eq(0)
    away_only = away_signal.ne(0) & home_signal.eq(0)
    result = baseline_pick_home.copy()
    result = result.where(~home_only, home_signal.gt(0))
    result = result.where(~away_only, away_signal.lt(0))
    return result


def _block_bootstrap_gap(
    frame: pd.DataFrame,
    outcome_column: str,
    group_a_mask: pd.Series,
    *,
    block_column: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    valid = frame[outcome_column].notna()
    working = frame.loc[valid].reset_index(drop=True)
    a_mask = group_a_mask.loc[valid].reset_index(drop=True).to_numpy(dtype=bool)
    outcome = working[outcome_column].to_numpy(dtype=float)
    blocks = list(working.groupby(block_column, sort=False).indices.values())

    def _metric(positions: np.ndarray) -> float:
        sub_outcome = outcome[positions]
        sub_a = a_mask[positions]
        if sub_a.sum() == 0 or (~sub_a).sum() == 0:
            return float("nan")
        return float(sub_outcome[sub_a].mean() - sub_outcome[~sub_a].mean())

    point = _metric(np.arange(len(working)))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for i in range(samples):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        positions = (
            np.concatenate([blocks[j] for j in selected]) if blocks else np.array([], dtype=int)
        )
        draws[i] = _metric(positions)
    finite = draws[~np.isnan(draws)]
    if finite.size == 0:
        lower = upper = prob_pos = float("nan")
    else:
        lower, upper = (float(x) for x in np.quantile(finite, [0.025, 0.975]))
        prob_pos = float(probability_positive_from_draws(finite))
    return {
        "point_estimate": point,
        "n_group_a": int(a_mask.sum()),
        "n_group_b": int((~a_mask).sum()),
        "ci95": [lower, upper],
        "probability_positive": prob_pos,
        "samples": samples,
        "nan_draws": int(np.isnan(draws).sum()),
        "block_column": block_column,
    }


def _rate(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    sub = frame.loc[mask, "team_covered"].dropna()
    return {
        "n": int(mask.sum()),
        "n_graded": len(sub),
        "cover_rate": float(sub.mean()) if len(sub) else float("nan"),
    }


def _signed_subset_gap(
    frame: pd.DataFrame,
    outcome_column: str,
    flag: pd.Series,
    *,
    direction: int,
    block_column: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if direction > 0:
        signed = _block_bootstrap_gap(
            frame, outcome_column, flag, block_column=block_column, samples=samples, seed=seed
        )
        signed["sign_convention"] = (
            "flag-minus-complement; positive favours the predeclared BACK direction"
        )
        return signed
    raw = _block_bootstrap_gap(
        frame, outcome_column, ~flag, block_column=block_column, samples=samples, seed=seed
    )
    signed = dict(raw)
    signed["n_group_a"], signed["n_group_b"] = raw["n_group_b"], raw["n_group_a"]
    signed["sign_convention"] = (
        "complement-minus-flag; positive favours the predeclared FADE direction"
    )
    return signed


def cmd_screen(_: argparse.Namespace) -> None:
    events = load_counted_events()
    tenure_table = load_oc_tenure_table()
    bye_table = load_bye_table()

    features = pd.read_parquet(GAME_FEATURES)
    close_long_full = build_close_team_long(features)
    close_long = close_long_full.loc[close_long_full["season"].le(HISTORY_LAST_SEASON)].reset_index(
        drop=True
    )

    archive = pd.read_parquet(OPENER_ARCHIVE)
    meta = pd.read_parquet(GAME_FEATURES, columns=["game_id", "home_team", "away_team", "kickoff"])
    opener_long = build_opener_team_long(archive, meta)

    r1_pop = attach_games_after_change(opener_long, events)
    r1_flag_first = r1_pop["game_number_after_change"].eq(1)
    r1_flag_2_4 = r1_pop["game_number_after_change"].between(2, 4)

    gap_first_week = _block_bootstrap_gap(
        r1_pop,
        "team_covered",
        r1_flag_first,
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    gap_first_season = _block_bootstrap_gap(
        r1_pop,
        "team_covered",
        r1_flag_first,
        block_column="season",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    gap_2_4_week = _block_bootstrap_gap(
        r1_pop,
        "team_covered",
        r1_flag_2_4,
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    gap_2_4_season = _block_bootstrap_gap(
        r1_pop,
        "team_covered",
        r1_flag_2_4,
        block_column="season",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    events_table = [
        {
            "season": e["season"],
            "team": e["team"],
            "role": e["role"],
            "previous_person": e["previous_person"],
            "person": e["person"],
            "revision_at": e["revision_at"].isoformat(),
        }
        for e in events
    ]

    game_number_cover = (
        r1_pop.loc[r1_pop["game_number_after_change"].notna()]
        .groupby(r1_pop["game_number_after_change"].astype("Int64"))["team_covered"]
        .agg(["mean", "size"])
        .reset_index()
        .to_dict(orient="records")
    )

    odd_r1 = r1_pop.loc[r1_pop["season"] % 2 == 1].reset_index(drop=True)
    even_r1 = r1_pop.loc[r1_pop["season"] % 2 == 0].reset_index(drop=True)

    def _r1_half(frame: pd.DataFrame) -> dict[str, Any]:
        flag = frame["game_number_after_change"].eq(1)
        return _block_bootstrap_gap(
            frame,
            "team_covered",
            flag,
            block_column="week_block",
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        )

    r1_reliability = {
        "method": "not_applicable",
        "note": (
            "playcaller_first_game is a rare in-season event flag (12 events across 4 "
            "seasons, 2022-2025), not a repeated-measures per-team trait with enough "
            "seasons to correlate; a formal split-half coefficient does not apply the way "
            "it would to a persistent team trait (matching this project's convention for "
            "other thin per-event flags, e.g. docs/snow_game_home_prep.md section 4). "
            "Reported instead: the first-game-vs-population cover gap measured "
            "independently on odd-numbered and even-numbered seasons of the same R1 "
            "population."
        ),
        "odd_seasons": _r1_half(odd_r1),
        "even_seasons": _r1_half(even_r1),
    }

    close_r2 = attach_oc_tenure(close_long, tenure_table)
    close_r2 = attach_bye_flag(close_r2, bye_table)
    r2_pop = close_r2.loc[close_r2["oc_tenure"].isin(["1", "2", "3+"])].reset_index(drop=True)

    back_flag = r2_pop["own_off_bye"] & r2_pop["oc_tenure"].isin(["1", "2"])
    fade_flag = r2_pop["own_off_bye"] & r2_pop["oc_tenure"].eq("3+")
    post_bye_pop = r2_pop.loc[r2_pop["own_off_bye"]].reset_index(drop=True)
    interaction_flag = post_bye_pop["oc_tenure"].isin(["1", "2"])

    r2_back_week = _signed_subset_gap(
        r2_pop,
        "team_covered",
        back_flag,
        direction=1,
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    r2_back_season = _signed_subset_gap(
        r2_pop,
        "team_covered",
        back_flag,
        direction=1,
        block_column="season",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    r2_fade_week = _signed_subset_gap(
        r2_pop,
        "team_covered",
        fade_flag,
        direction=-1,
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    r2_fade_season = _signed_subset_gap(
        r2_pop,
        "team_covered",
        fade_flag,
        direction=-1,
        block_column="season",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    r2_interaction_week = _signed_subset_gap(
        post_bye_pop,
        "team_covered",
        interaction_flag,
        direction=1,
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    r2_interaction_season = _signed_subset_gap(
        post_bye_pop,
        "team_covered",
        interaction_flag,
        direction=1,
        block_column="season",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    opener_r2 = attach_oc_tenure(opener_long, tenure_table)
    opener_r2 = attach_bye_flag(opener_r2, bye_table)
    opener_r2_pop = opener_r2.loc[opener_r2["oc_tenure"].isin(["1", "2", "3+"])].reset_index(
        drop=True
    )
    opener_back_flag = opener_r2_pop["own_off_bye"] & opener_r2_pop["oc_tenure"].isin(["1", "2"])
    opener_fade_flag = opener_r2_pop["own_off_bye"] & opener_r2_pop["oc_tenure"].eq("3+")
    opener_post_bye_pop = opener_r2_pop.loc[opener_r2_pop["own_off_bye"]].reset_index(drop=True)
    opener_interaction_flag = opener_post_bye_pop["oc_tenure"].isin(["1", "2"])

    opener_back_week = _signed_subset_gap(
        opener_r2_pop,
        "team_covered",
        opener_back_flag,
        direction=1,
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    opener_fade_week = _signed_subset_gap(
        opener_r2_pop,
        "team_covered",
        opener_fade_flag,
        direction=-1,
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    opener_interaction_week = _signed_subset_gap(
        opener_post_bye_pop,
        "team_covered",
        opener_interaction_flag,
        direction=1,
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    back_season_rate = r2_pop.loc[back_flag].groupby("season")["team_covered"].agg(["mean", "size"])
    fade_season_rate = r2_pop.loc[fade_flag].groupby("season")["team_covered"].agg(["mean", "size"])
    seasons_present = sorted(set(r2_pop["season"].unique()))
    pair_rows = []
    it = iter(seasons_present)
    for odd_season in it:
        if odd_season % 2 == 0:
            continue
        try:
            even_season = next(s for s in seasons_present if s == odd_season + 1)
        except StopIteration:
            continue
        if odd_season not in back_season_rate.index or odd_season not in fade_season_rate.index:
            continue
        if even_season not in back_season_rate.index or even_season not in fade_season_rate.index:
            continue
        odd_gap = 100.0 * (
            back_season_rate.loc[odd_season, "mean"] - fade_season_rate.loc[odd_season, "mean"]
        )
        even_gap = 100.0 * (
            back_season_rate.loc[even_season, "mean"] - fade_season_rate.loc[even_season, "mean"]
        )
        pair_rows.append(
            {
                "odd_season": int(odd_season),
                "even_season": int(even_season),
                "odd_gap": float(odd_gap),
                "even_gap": float(even_gap),
                "odd_n_back": int(back_season_rate.loc[odd_season, "size"]),
                "odd_n_fade": int(fade_season_rate.loc[odd_season, "size"]),
                "even_n_back": int(back_season_rate.loc[even_season, "size"]),
                "even_n_fade": int(fade_season_rate.loc[even_season, "size"]),
            }
        )

    n_pairs = len(pair_rows)
    if n_pairs >= 3:
        odd_vals = np.array([p["odd_gap"] for p in pair_rows], dtype=float)
        even_vals = np.array([p["even_gap"] for p in pair_rows], dtype=float)
        r_point = float(np.corrcoef(odd_vals, even_vals)[0, 1])
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        boots = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
        for i in range(BOOTSTRAP_SAMPLES):
            selected = rng.integers(0, n_pairs, size=n_pairs)
            boots[i] = np.corrcoef(odd_vals[selected], even_vals[selected])[0, 1]
        finite_boots = boots[np.isfinite(boots)]
        spearman_brown = (2.0 * r_point) / (1.0 + r_point) if r_point > -1.0 else float("nan")
        r2_reliability = {
            "method": "season_pair_correlation",
            "trait": (
                "post_bye_oc_interaction (back cover rate minus fade cover rate, "
                "per season, accuracy points)"
            ),
            "n_pairs": n_pairs,
            "pairs": pair_rows,
            "pearson_r": r_point,
            "pearson_r_ci95": [
                float(np.nanquantile(finite_boots, 0.025)) if finite_boots.size else float("nan"),
                float(np.nanquantile(finite_boots, 0.975)) if finite_boots.size else float("nan"),
            ],
            "spearman_brown_full_length_reliability": spearman_brown,
            "probability_positive": float(probability_positive_from_draws(finite_boots))
            if finite_boots.size
            else float("nan"),
        }
    else:
        r2_reliability = {
            "method": "not_applicable",
            "n_pairs": n_pairs,
            "pairs": pair_rows,
            "note": (
                f"only {n_pairs} odd/even season pairs had both a back-flagged and a "
                "fade-flagged observation; too few to correlate"
            ),
        }

    odd_r2 = r2_pop.loc[r2_pop["season"] % 2 == 1].reset_index(drop=True)
    even_r2 = r2_pop.loc[r2_pop["season"] % 2 == 0].reset_index(drop=True)
    r2_reliability["odd_seasons_back_gap"] = _block_bootstrap_gap(
        odd_r2,
        "team_covered",
        odd_r2["own_off_bye"] & odd_r2["oc_tenure"].isin(["1", "2"]),
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    r2_reliability["even_seasons_back_gap"] = _block_bootstrap_gap(
        even_r2,
        "team_covered",
        even_r2["own_off_bye"] & even_r2["oc_tenure"].isin(["1", "2"]),
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    result = {
        "sources": {
            "coordinator_history_dir": str(_latest_coordinator_dir()),
            "schedules_snapshot": str(_latest_schedules()),
            "opener_archive": str(OPENER_ARCHIVE),
            "game_features": str(GAME_FEATURES),
        },
        "events": events_table,
        "R1_lead29_first_game": {
            "population_n_team_games": len(r1_pop),
            "cells": {
                "first_game": {
                    "n": gap_first_week["n_group_a"],
                    "week_blocked": gap_first_week,
                    "season_blocked": gap_first_season,
                },
                "games_2_to_4": {
                    "n": gap_2_4_week["n_group_a"],
                    "week_blocked": gap_2_4_week,
                    "season_blocked": gap_2_4_season,
                },
            },
            "cover_by_game_number": game_number_cover,
            "reliability_odd_even_seasons": r1_reliability,
        },
        "R2_lead28_post_bye_tenure": {
            "primary_close_grade_2009_2025": {
                "population_n_team_games": len(r2_pop),
                "post_bye_new_oc_back": {
                    "week_blocked": r2_back_week,
                    "season_blocked": r2_back_season,
                },
                "post_bye_veteran_oc_fade": {
                    "week_blocked": r2_fade_week,
                    "season_blocked": r2_fade_season,
                },
                "post_bye_oc_interaction": {
                    "week_blocked": r2_interaction_week,
                    "season_blocked": r2_interaction_season,
                },
            },
            "secondary_opener_grade_2020_2025": {
                "population_n_team_games": len(opener_r2_pop),
                "post_bye_new_oc_back": opener_back_week,
                "post_bye_veteran_oc_fade": opener_fade_week,
                "post_bye_oc_interaction": opener_interaction_week,
            },
            "reliability_odd_even_seasons": r2_reliability,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "screen_results.json"
    write_stamped_artifact(result, out_path, project_root=REPO)
    r1_pop.to_parquet(OUT_DIR / "r1_population.parquet", index=False)
    r2_pop.to_parquet(OUT_DIR / "r2_population.parquet", index=False)
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {out_path}")


def _accuracy_metric(frame: pd.DataFrame) -> dict[str, float]:
    valid = frame.dropna(subset=["baseline_correct", "candidate_correct"])
    return {
        "accuracy_points": 100.0
        * float((valid["candidate_correct"] - valid["baseline_correct"]).mean()),
        "candidate_accuracy": 100.0 * float(valid["candidate_correct"].mean()),
        "baseline_accuracy": 100.0 * float(valid["baseline_correct"].mean()),
    }


def _summarize(frame: pd.DataFrame, samples: int, seed: int) -> dict[str, Any]:
    point = _accuracy_metric(frame)
    week = week_blocked_bootstrap(frame, _accuracy_metric, block="week", samples=samples, seed=seed)
    season = week_blocked_bootstrap(
        frame, _accuracy_metric, block="season", samples=samples, seed=seed
    )
    w = week.loc[week["metric"].eq("accuracy_points")].iloc[0]
    s = season.loc[season["metric"].eq("accuracy_points")].iloc[0]
    return {
        **point,
        "week_blocked_ci95": [float(w["lower"]), float(w["upper"])],
        "week_blocked_probability_positive": float(w["probability_positive"]),
        "season_blocked_ci95": [float(s["lower"]), float(s["upper"])],
        "season_blocked_probability_positive": float(s["probability_positive"]),
        "n_games": len(frame),
        "n_weeks": int(frame[["season", "week"]].drop_duplicates().shape[0]),
        "n_seasons": int(frame["season"].nunique()),
    }


def _signal_long_table(lead: int) -> tuple[pd.DataFrame, int | None]:
    full_features = pd.read_parquet(GAME_FEATURES)
    long_df = build_close_team_long(full_features)
    if lead == 29:
        events = load_counted_events()
        long_df = attach_games_after_change(long_df, events)
        long_df["signal"] = np.where(long_df["game_number_after_change"].eq(1), 1, 0)
        return long_df, len(events)
    tenure_table = load_oc_tenure_table()
    bye_table = load_bye_table()
    long_df = attach_oc_tenure(long_df, tenure_table)
    long_df = attach_bye_flag(long_df, bye_table)
    long_df["signal"] = 0
    long_df.loc[long_df["own_off_bye"] & long_df["oc_tenure"].isin(["1", "2"]), "signal"] = 1
    long_df.loc[long_df["own_off_bye"] & long_df["oc_tenure"].eq("3+"), "signal"] = -1
    return long_df, None


def _score_and_tilt(
    scope: pd.DataFrame, signals: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    scored = opener_pick_evaluation(
        MARKET_ROOT, scope, active_model_config=config, min_train_games=DEFAULT_MIN_TRAIN_GAMES
    )
    scored["game_id"] = scored["game_id"].astype(str)
    scored = scored.merge(signals, on="game_id", how="left")
    scored["home_signal"] = scored["home_signal"].fillna(0)
    scored["away_signal"] = scored["away_signal"].fillna(0)
    scored["baseline_pick_home"] = scored["home_cover_probability_at_open"].ge(0.5)
    scored["candidate_pick_home"] = apply_tilt(
        scored["baseline_pick_home"], scored["home_signal"], scored["away_signal"]
    )
    scored["baseline_correct"] = pick_correct(
        scored["baseline_pick_home"], scored["margin_vs_open"]
    )
    scored["candidate_correct"] = pick_correct(
        scored["candidate_pick_home"], scored["margin_vs_open"]
    )
    return scored


def _production_read(scored: pd.DataFrame) -> dict[str, Any]:
    graded = scored.dropna(subset=["baseline_correct", "candidate_correct"])
    picks_changed = int((scored["baseline_pick_home"] != scored["candidate_pick_home"]).sum())
    picks_changed_graded = int(
        (graded["baseline_pick_home"] != graded["candidate_pick_home"]).sum()
    )
    n_flagged = int(((scored["home_signal"] != 0) | (scored["away_signal"] != 0)).sum())
    return {
        "n_games": len(scored),
        "n_flagged": n_flagged,
        "picks_changed": picks_changed,
        "picks_changed_graded": picks_changed_graded,
        "summary": _summarize(graded, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED),
    }


def cmd_production(lead: int) -> None:
    family = FAMILY_LEAD29_PROD if lead == 29 else FAMILY_LEAD28_PROD
    registry = load_registry()
    features = pd.read_parquet(PRODUCTION_FEATURES)
    training, window = confirmation_split(features, registry, family)
    if pd.to_datetime(training["gameday"]).max() >= pd.to_datetime(window["gameday"]).min():
        raise SystemExit("confirmation split leaked a training row into the assigned window")
    seasons = tuple(sorted(int(s) for s in window["season"].unique()))
    scoped = pd.concat([training, window], ignore_index=True)

    config = resolve_active_model_config(ARTIFACTS_ROOT)
    long_df, n_events = _signal_long_table(lead)
    signals = game_level_signal(long_df, "signal")
    signals["game_id"] = signals["game_id"].astype(str)

    windowed_scored = _score_and_tilt(scoped, signals, config)
    windowed_scored = windowed_scored.loc[
        windowed_scored["season"].astype(int).isin(seasons)
    ].reset_index(drop=True)
    windowed_read = _production_read(windowed_scored)

    full_scored = _score_and_tilt(features, signals, config)
    full_read = _production_read(full_scored)

    result = {
        "family": family,
        "lead": lead,
        "grade": "opener",
        "active_model_id": config.get("model_id"),
        "n_events_source": n_events,
        "assigned_window": {"window_seasons": list(seasons), **windowed_read},
        "full_archive_descriptive": {
            "seasons": sorted(int(s) for s in full_scored["season"].unique()),
            **full_read,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"production_results_lead{lead}.json"
    write_stamped_artifact(result, out_path, project_root=REPO)
    windowed_scored.to_parquet(
        OUT_DIR / f"production_paired_lead{lead}_window.parquet", index=False
    )
    full_scored.to_parquet(OUT_DIR / f"production_paired_lead{lead}_full.parquet", index=False)
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LEAD-28/LEAD-29 playcaller-change screen")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("screen", help="subset-bias cells and split-half reliability for R1/R2")
    sub.add_parser("production-lead28", help="post-bye new-playcaller tilt stacked on production")
    sub.add_parser("production-lead29", help="playcaller-first-game tilt stacked on production")
    args = parser.parse_args()
    if args.command == "screen":
        cmd_screen(args)
    elif args.command == "production-lead28":
        cmd_production(28)
    elif args.command == "production-lead29":
        cmd_production(29)


if __name__ == "__main__":
    main()
