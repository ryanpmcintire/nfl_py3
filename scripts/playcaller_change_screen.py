"""Playcaller-change screen: LEAD-29 (first game after an in-season OC/DC
change) and LEAD-28 (post-bye teams split by OC tenure at the September 1
observation).

Predeclared in ``docs/playcaller_change_leads.md`` before any outcome was
read; this module implements exactly those cells and nothing else. It is a
SCREEN in the shape of ``docs/interim_coach_screen.md``: it spends no
rotation window, wires no overlay or challenger and changes no played
card. It never writes ``registry/`` itself -- it emits the exact
``nfl-ats weak-signals record`` argv for every cell so the recording step
is explicit and auditable.

Method (reused, not re-derived): ``nfl_ats.experiment_runner``'s team-game
long table, joint week-blocked block bootstrap of the flag-minus-complement
cover-rate gap, full-slate scaling and the mechanical classifier. Seed
20260817, 20,000 draws, within-week correlation ZERO (no design-effect
padding). Grades: the paired Tuesday-opener archive
(``artifacts/opener_evaluation/<run>/per_game.parquet``, line =
``tue_open_home_spread``) and the nflverse close (``game_features.parquet``
``spread_line``).

Pregame safety: R1 flags a game only when the coordinator change's revision
instant is strictly before the game's kickoff
(``nfl_ats.coordinator_changes.games_after_coordinator_change``); R2 reads
season-start (preseason) observations only
(``nfl_ats.coordinator_changes.oc_tenure_at_season_start``) and asserts that
every observation used sits strictly before the flagged kickoff.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.bye_edge_fade_overlay import bye_edge_flag_by_game  # noqa: E402
from nfl_ats.coordinator_changes import (  # noqa: E402
    games_after_coordinator_change,
    oc_tenure_at_season_start,
)
from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.estimation_variance import MIN_BLOCKS_FOR_INTERVAL  # noqa: E402
from nfl_ats.experiment_runner import (  # noqa: E402
    _base_team_game_table,
    _block_bootstrap_subset_gap,
    _interval_summary,
    _latest_schedules_snapshot,
    classify_subset_bias_result,
    scale_subset_effect,
)
from nfl_ats.provenance import write_stamped_artifact  # noqa: E402

FAMILY = "lead29_playcaller_change_v1"
SEED = 20260817
SAMPLES = 20_000
CONFIDENCE = 0.95
COUNTED_STATUSES = ("staff_change", "playcaller_role")
NEW_OC_MAX_TENURE = 2
OPENER_SEASONS = (2020, 2025)
FULL_SEASONS = (2009, 2025)
DOC = "docs/playcaller_change_leads.md"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def latest_coordinator_snapshot(repo_root: Path) -> Path:
    candidates = sorted(
        (repo_root / "data" / "raw" / "coordinators").glob("*/coordinator_history.parquet")
    )
    if not candidates:
        raise FileNotFoundError("no data/raw/coordinators/*/coordinator_history.parquet snapshot")
    return candidates[-1]


def latest_opener_per_game(repo_root: Path) -> Path:
    candidates = sorted((repo_root / "artifacts" / "opener_evaluation").glob("*/per_game.parquet"))
    if not candidates:
        raise FileNotFoundError("no artifacts/opener_evaluation/*/per_game.parquet archive")
    return candidates[-1]


def load_counted_events(validation_path: Path) -> pd.DataFrame:
    """The in-season events the predeclaration counts: ``staff_change`` and
    ``playcaller_role`` adjudications, one row per event."""

    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    rows = []
    for change in payload["changes"]:
        if change["validation_status"] not in COUNTED_STATUSES:
            continue
        revision_at = pd.Timestamp(change["revision_at"])
        if revision_at.tzinfo is None:
            revision_at = revision_at.tz_localize("UTC")
        rows.append(
            {
                "event_id": (
                    f"{change['season']}_{change['team']}_{change['role']}_"
                    f"{revision_at.strftime('%Y%m%dT%H%M%S')}"
                ),
                "season": int(change["season"]),
                "team": str(change["team"]),
                "role": str(change["role"]),
                "side": "offence" if str(change["role"]) == "OC" else "defence",
                "previous_person": change.get("previous_person"),
                "person": change.get("person"),
                "revision_at": revision_at,
                "validation_status": change["validation_status"],
                "validation_note": change.get("validation_note"),
                "independent_source": change.get("independent_source"),
            }
        )
    events = pd.DataFrame(rows)
    if events.empty:
        raise DataContractError(f"{validation_path} carries no counted events")
    return events.sort_values(["season", "team", "revision_at"]).reset_index(drop=True)


def opener_graded_features(features: pd.DataFrame, per_game: pd.DataFrame) -> pd.DataFrame:
    """Re-grade the REG feature rows in the opener archive against the
    Tuesday opener (``tue_open_home_spread``)."""

    required = {"game_id", "tue_open_home_spread", "result"}
    missing = sorted(required.difference(per_game.columns))
    if missing:
        raise DataContractError(f"opener per_game archive is missing columns: {missing}")
    opener = per_game[["game_id", "tue_open_home_spread"]].rename(
        columns={"tue_open_home_spread": "_opener_home_spread"}
    )
    merged = features.loc[features["game_type"].eq("REG")].merge(opener, on="game_id", how="inner")
    merged["result"] = pd.to_numeric(merged["result"], errors="coerce")
    merged = merged.loc[merged["result"].notna()].copy()
    merged["spread_line"] = pd.to_numeric(merged["_opener_home_spread"], errors="coerce")
    ats_margin = merged["result"] - merged["spread_line"]
    merged["home_cover"] = np.select([ats_margin > 0, ats_margin < 0], [1.0, 0.0], default=np.nan)
    return merged.drop(columns=["_opener_home_spread"]).reset_index(drop=True)


def team_game_table(features: pd.DataFrame, seasons: tuple[int, int]) -> pd.DataFrame:
    table = _base_team_game_table(features)
    kickoff = features[["game_id", "kickoff"]].drop_duplicates("game_id")
    table = table.merge(kickoff, on="game_id", how="left")
    if table["kickoff"].isna().any():
        raise DataContractError("every team-game row needs a kickoff timestamp")
    table["kickoff"] = pd.to_datetime(table["kickoff"], utc=True)
    table = table.loc[table["season"].between(seasons[0], seasons[1])]
    return table.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def r1_flags(table: pd.DataFrame, features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    games = features.loc[features["game_type"].eq("REG")][
        ["game_id", "season", "kickoff", "home_team", "away_team"]
    ].drop_duplicates("game_id")
    after = games_after_coordinator_change(games, events)
    merged = table.merge(
        after[["game_id", "team", "event_id", "role", "revision_at", "game_number_after_change"]],
        on=["game_id", "team"],
        how="left",
    )
    merged["game_number_after_change"] = merged["game_number_after_change"].fillna(0).astype(int)
    flagged = merged["game_number_after_change"].gt(0)
    if not merged.loc[flagged, "revision_at"].lt(merged.loc[flagged, "kickoff"]).all():
        raise DataContractError("a flagged game's revision instant is not before its kickoff")
    merged["first_game_after_change"] = merged["game_number_after_change"].eq(1)
    merged["games_2_to_4_after_change"] = merged["game_number_after_change"].between(2, 4)
    return merged


def r2_flags(table: pd.DataFrame, history: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    tenure = oc_tenure_at_season_start(history)
    bye = bye_edge_flag_by_game(schedules)
    merged = table.merge(
        tenure[["season", "team", "oc_name", "oc_tenure_years", "observed_at"]],
        on=["season", "team"],
        how="left",
    )
    merged = merged.merge(
        bye[["game_id", "home_off_bye", "away_off_bye"]], on="game_id", how="left"
    )
    if merged["home_off_bye"].isna().any():
        raise DataContractError("every team-game row needs a schedules-derived bye flag")
    merged["post_bye"] = np.where(
        merged["is_home"], merged["home_off_bye"].astype(bool), merged["away_off_bye"].astype(bool)
    )
    merged["tenure_known"] = merged["oc_tenure_years"].notna()
    known = merged["tenure_known"]
    if not merged.loc[known, "observed_at"].lt(merged.loc[known, "kickoff"]).all():
        raise DataContractError("a season-start observation is not before the game's kickoff")
    merged["new_oc"] = known & merged["oc_tenure_years"].le(NEW_OC_MAX_TENURE)
    merged["veteran_oc"] = known & merged["oc_tenure_years"].gt(NEW_OC_MAX_TENURE)
    return merged


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_cell(
    *,
    name: str,
    table: pd.DataFrame,
    flag: pd.Series,
    eligible: pd.Series | None,
    sign: int,
    grade: str,
    seasons: tuple[int, int],
) -> dict[str, Any]:
    comparison = table if eligible is None else table.loc[eligible]
    comparison_flag = flag if eligible is None else flag.loc[eligible]
    n_total = len(table)
    n_flag = int(comparison_flag.sum())
    n_complement = int(len(comparison) - n_flag)
    if n_flag == 0 or n_complement == 0:
        raise DataContractError(f"{name}: flag or complement is empty")
    flag_rows = comparison.loc[comparison_flag]
    complement_rows = comparison.loc[~comparison_flag]
    flag_cover = float(flag_rows["team_covered"].mean())
    complement_cover = float(complement_rows["team_covered"].mean())
    raw_gap_fraction = flag_cover - complement_cover
    slate_numerator = n_flag if eligible is None else len(comparison)
    fraction_of_slate = slate_numerator / n_total
    effect = scale_subset_effect(raw_gap_fraction, sign=sign, fraction_of_slate=fraction_of_slate)

    def block(kind: str, column: str) -> dict[str, Any]:
        block_count = int(comparison[column].nunique())
        draws = _block_bootstrap_subset_gap(
            comparison,
            flag=comparison_flag,
            value_col="team_covered",
            block_col=column,
            samples=SAMPLES,
            seed=SEED,
        )
        raw = _interval_summary(
            draws,
            block_kind=kind,
            block_count=block_count,
            degenerate=block_count < MIN_BLOCKS_FOR_INTERVAL,
            sign=sign,
            fraction_of_slate=1.0,
            confidence=CONFIDENCE,
        )
        scaled = _interval_summary(
            draws,
            block_kind=kind,
            block_count=block_count,
            degenerate=block_count < MIN_BLOCKS_FOR_INTERVAL,
            sign=sign,
            fraction_of_slate=fraction_of_slate,
            confidence=CONFIDENCE,
        )
        return {
            "block_kind": kind,
            "block_count": block_count,
            "degenerate": block_count < MIN_BLOCKS_FOR_INTERVAL,
            "samples": int(raw.samples),
            "raw_gap": {
                "estimate": raw.estimate,
                "lower": raw.lower,
                "upper": raw.upper,
                "standard_error": raw.standard_error,
            },
            "scaled": {
                "estimate": scaled.estimate,
                "lower": scaled.lower,
                "upper": scaled.upper,
                "standard_error": scaled.standard_error,
            },
            "probability_positive": raw.probability_positive,
        }

    week = block("week", "week_block")
    season = block("season", "season")
    classification = classify_subset_bias_result(
        estimate=effect, lower=week["scaled"]["lower"], upper=week["scaled"]["upper"]
    )
    return {
        "name": name,
        "grade": grade,
        "seasons": list(seasons),
        "sign": sign,
        "design": "one_sided_vs_everyone_else" if eligible is None else "within_eligible",
        "n_total": n_total,
        "n_flag": n_flag,
        "n_complement": n_complement,
        "flag_record": {
            "covers": int(flag_rows["team_covered"].sum()),
            "games": n_flag,
            "cover_rate": flag_cover,
        },
        "complement_cover_rate": complement_cover,
        "raw_gap_points": sign * raw_gap_fraction * 100.0,
        "fraction_of_slate": fraction_of_slate,
        "effect_full_slate_points": effect,
        "week_blocked": week,
        "season_blocked": season,
        "classification": classification.classification,
        "closing_ground": classification.closing_ground,
        "classification_note": classification.note,
    }


# ---------------------------------------------------------------------------
# Record-command emission (the registry write stays an explicit CLI step)
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value:+.4f}"


def record_argv(
    cell: dict[str, Any],
    *,
    description: str,
    plain_summary: str,
    source: str,
    extra_notes: str,
) -> list[str]:
    week = cell["week_blocked"]
    season = cell["season_blocked"]
    record = cell["flag_record"]
    evidence = (
        f"Predeclared cell ({DOC}), family {FAMILY}, screened before any sign was read. "
        f"Grade={cell['grade']}, seasons {cell['seasons'][0]}-{cell['seasons'][1]}, design "
        f"{cell['design']}, sign={cell['sign']:+d}. Flag {record['covers']}-"
        f"{record['games'] - record['covers']} ({100 * record['cover_rate']:.2f}%) vs complement "
        f"{100 * cell['complement_cover_rate']:.2f}% on {cell['n_complement']} team-games "
        f"({cell['n_total']} in population, fraction_of_slate={cell['fraction_of_slate']:.4f}). "
        f"Raw covered-minus-baseline gap {_fmt(cell['raw_gap_points'])} pts, week-blocked 95% "
        f"[{_fmt(week['raw_gap']['lower'])}, {_fmt(week['raw_gap']['upper'])}] "
        f"({week['block_count']} blocks, {SAMPLES} draws, seed {SEED}, within-week correlation "
        f"zero), probability_positive={week['probability_positive']:.4f}; season-blocked "
        f"[{_fmt(season['raw_gap']['lower'])}, {_fmt(season['raw_gap']['upper'])}] "
        f"({season['block_count']} blocks{', DEGENERATE' if season['degenerate'] else ''}), "
        f"probability_positive={season['probability_positive']:.4f}. Full-slate-scaled effect "
        f"{_fmt(cell['effect_full_slate_points'])} pts, 95% [{_fmt(week['scaled']['lower'])}, "
        f"{_fmt(week['scaled']['upper'])}]. {cell['classification_note']} No positive control "
        "was run, so bounded_by_control cannot be claimed."
    )
    argv = [
        "weak-signals",
        "record",
        "--name",
        cell["name"],
        "--description",
        description,
        "--source",
        source,
        "--effect",
        f"{cell['effect_full_slate_points']:.6f}",
        "--effect-units",
        "accuracy_points",
        "--classification",
        cell["classification"],
        "--league",
        "nfl",
        "--season-start",
        str(cell["seasons"][0]),
        "--season-end",
        str(cell["seasons"][1]),
        "--standard-error",
        f"{week['scaled']['standard_error']:.6f}",
        "--interval-low",
        f"{week['scaled']['lower']:.6f}",
        "--interval-high",
        f"{week['scaled']['upper']:.6f}",
        "--probability-positive",
        f"{week['probability_positive']:.4f}",
        "--sample-games",
        str(cell["n_flag"] + cell["n_complement"]),
        "--sample-blocks",
        str(week["block_count"]),
        "--family",
        FAMILY,
        "--classification-evidence",
        evidence,
        "--plain-summary",
        plain_summary,
        "--category",
        "offfield",
        "--notes",
        (
            f"seed={SEED}; samples={SAMPLES}; screen only (no rotation window, no overlay, no "
            f"challenger). {extra_notes}"
        ),
    ]
    if cell["closing_ground"]:
        argv += ["--closing-ground", cell["closing_ground"]]
    return argv


def _summary_line(cell: dict[str, Any]) -> str:
    week = cell["week_blocked"]
    record = cell["flag_record"]
    return (
        f"{cell['name']:<58} {cell['grade']:<6} n={record['games']:>5} "
        f"{record['covers']}-{record['games'] - record['covers']} "
        f"({100 * record['cover_rate']:.2f}% vs {100 * cell['complement_cover_rate']:.2f}%) "
        f"gap={_fmt(cell['raw_gap_points'])} [{_fmt(week['raw_gap']['lower'])}, "
        f"{_fmt(week['raw_gap']['upper'])}] P+={week['probability_positive']:.4f} "
        f"scaled={_fmt(cell['effect_full_slate_points'])} {cell['classification']}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _interim_overlap(repo_root: Path, first_games: pd.DataFrame) -> dict[str, Any]:
    """Descriptive: how many R1 first games are ALSO an interim-HC first game
    (the live dual-tracked challenger's own flag)."""

    try:
        from nfl_ats.experiment_runner import _build_interim_coach_trait_data

        trait = _build_interim_coach_trait_data(repo_root).game_trait
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}
    interim_first = trait.loc[trait["first_game_under_interim"], ["game_id", "team"]]
    overlap = first_games.merge(interim_first, on=["game_id", "team"], how="inner")
    return {
        "available": True,
        "interim_first_games": len(interim_first),
        "overlap_rows": overlap[["game_id", "team"]].to_dict("records"),
        "overlap_count": len(overlap),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    history_path = Path(args.history) if args.history else latest_coordinator_snapshot(repo_root)
    validation_path = history_path.parent / "change_validation.json"
    per_game_path = (
        Path(args.opener_per_game) if args.opener_per_game else latest_opener_per_game(repo_root)
    )
    features_path = (
        Path(args.features)
        if args.features
        else (repo_root / "data" / "processed" / "game_features.parquet")
    )
    schedules_path = (
        Path(args.schedules) if args.schedules else _latest_schedules_snapshot(repo_root)
    )

    history = pd.read_parquet(history_path)
    events = load_counted_events(validation_path)
    features = pd.read_parquet(features_path)
    per_game = pd.read_parquet(per_game_path)
    schedules = pd.read_parquet(schedules_path)

    close_features = features.loc[features["game_type"].eq("REG")].copy()
    opener_features = opener_graded_features(features, per_game)

    close_full = team_game_table(close_features, FULL_SEASONS)
    close_2020 = team_game_table(close_features, OPENER_SEASONS)
    opener = team_game_table(opener_features, OPENER_SEASONS)

    # R1 ------------------------------------------------------------------
    r1_open = r1_flags(opener, opener_features, events)
    r1_close = r1_flags(close_2020, close_features, events)
    first_open = r1_open.loc[r1_open["first_game_after_change"]]
    lead_hours = (first_open["kickoff"] - first_open["revision_at"]).dt.total_seconds() / 3600.0
    r1_cells = [
        score_cell(
            name=f"{FAMILY}_first_game",
            table=r1_open,
            flag=r1_open["first_game_after_change"],
            eligible=None,
            sign=1,
            grade="opener",
            seasons=OPENER_SEASONS,
        ),
        score_cell(
            name=f"{FAMILY}_games_2_to_4",
            table=r1_open,
            flag=r1_open["games_2_to_4_after_change"],
            eligible=None,
            sign=1,
            grade="opener",
            seasons=OPENER_SEASONS,
        ),
    ]
    r1_secondary = [
        score_cell(
            name=f"{FAMILY}_first_game",
            table=r1_close,
            flag=r1_close["first_game_after_change"],
            eligible=None,
            sign=1,
            grade="close",
            seasons=OPENER_SEASONS,
        ),
        score_cell(
            name=f"{FAMILY}_games_2_to_4",
            table=r1_close,
            flag=r1_close["games_2_to_4_after_change"],
            eligible=None,
            sign=1,
            grade="close",
            seasons=OPENER_SEASONS,
        ),
    ]
    first_game_rows = first_open.assign(
        opener_spread=first_open["team_spread"],
        covered=first_open["team_covered"].astype(int),
    )[
        [
            "event_id",
            "game_id",
            "team",
            "opponent",
            "is_home",
            "role",
            "revision_at",
            "kickoff",
            "opener_spread",
            "covered",
        ]
    ]
    first_game_rows["revision_at"] = first_game_rows["revision_at"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    first_game_rows["kickoff"] = first_game_rows["kickoff"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    events_without_first_game = sorted(
        set(events["event_id"]).difference(first_open["event_id"].astype(str))
    )
    by_game_number = (
        r1_open.loc[r1_open["game_number_after_change"].gt(0)]
        .groupby("game_number_after_change")["team_covered"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"count": "games", "sum": "covers", "mean": "cover_rate"})
    )

    # R2 ------------------------------------------------------------------
    r2_full = r2_flags(close_full, history, schedules)
    r2_open = r2_flags(opener, history, schedules)

    def r2_cells(table: pd.DataFrame, grade: str, seasons: tuple[int, int]) -> list[dict[str, Any]]:
        known = table.loc[table["tenure_known"]].reset_index(drop=True)
        post_bye_known = known["post_bye"]
        return [
            score_cell(
                name=f"{FAMILY}_post_bye_new_oc_back",
                table=known,
                flag=known["post_bye"] & known["new_oc"],
                eligible=None,
                sign=1,
                grade=grade,
                seasons=seasons,
            ),
            score_cell(
                name=f"{FAMILY}_post_bye_veteran_oc_fade",
                table=known,
                flag=known["post_bye"] & known["veteran_oc"],
                eligible=None,
                sign=-1,
                grade=grade,
                seasons=seasons,
            ),
            score_cell(
                name=f"{FAMILY}_post_bye_oc_interaction",
                table=known,
                flag=known["new_oc"],
                eligible=post_bye_known,
                sign=1,
                grade=grade,
                seasons=seasons,
            ),
        ]

    r2_primary = r2_cells(r2_full, "close", FULL_SEASONS)
    r2_secondary = r2_cells(r2_open, "opener", OPENER_SEASONS)
    tenure_coverage = (
        r2_full.groupby("season")
        .agg(rows=("game_id", "size"), known=("tenure_known", "sum"))
        .reset_index()
    )
    tenure_coverage["known"] = tenure_coverage["known"].astype(int)

    # Report --------------------------------------------------------------
    print("R1 first-game events (opener grade):")
    print(first_game_rows.to_string(index=False))
    print()
    print("R1 by game number after the change (opener grade):")
    print(by_game_number.to_string(index=False))
    print()
    for cell in [*r1_cells, *r1_secondary, *r2_primary, *r2_secondary]:
        print(_summary_line(cell))

    source_note = f"{DOC}; results at {Path(args.output_dir) / 'results.json'}"
    event_table_note = "Events counted (12): " + "; ".join(
        f"{row.season} {row.team} {row.role} {row.previous_person}->{row.person} "
        f"@{row.revision_at.strftime('%Y-%m-%dT%H:%M:%SZ')} ({row.validation_status})"
        for row in events.itertuples(index=False)
    )
    secondary_note = {
        cell["name"]: (
            f"Secondary {cell['grade']} grade {cell['seasons'][0]}-{cell['seasons'][1]}: "
            f"{cell['flag_record']['covers']}-"
            f"{cell['flag_record']['games'] - cell['flag_record']['covers']} "
            f"({100 * cell['flag_record']['cover_rate']:.2f}% vs "
            f"{100 * cell['complement_cover_rate']:.2f}%), raw gap "
            f"{_fmt(cell['raw_gap_points'])} pts, week-blocked 95% "
            f"[{_fmt(cell['week_blocked']['raw_gap']['lower'])}, "
            f"{_fmt(cell['week_blocked']['raw_gap']['upper'])}], probability_positive "
            f"{cell['week_blocked']['probability_positive']:.4f}."
        )
        for cell in [*r1_secondary, *r2_secondary]
    }
    descriptions = {
        f"{FAMILY}_first_game": (
            "LEAD-29: a team's FIRST regular-season game after an in-season offensive or "
            "defensive coordinator change (dated Wikipedia staff-template revision instant "
            "strictly before kickoff; staff_change and playcaller_role adjudications), vs "
            "everyone else in the paired Tuesday-opener archive. Direction: back that team."
        ),
        f"{FAMILY}_games_2_to_4": (
            "LEAD-29 descriptive companion: games 2-4 after an in-season OC/DC change vs "
            "everyone else, opener archive. Same sign convention as the first-game cell; "
            "the interim-HC screen found its bump in game 1 only, so this is context."
        ),
        f"{FAMILY}_post_bye_new_oc_back": (
            "LEAD-28: post-bye team-games (strict >=12-day gap) where the September 1 "
            "offensive coordinator (playcaller proxy) is in season 1 or 2 with the team, vs "
            "everyone else with known OC tenure. Direction: back the post-bye team."
        ),
        f"{FAMILY}_post_bye_veteran_oc_fade": (
            "LEAD-28: post-bye team-games where the September 1 offensive coordinator "
            "(playcaller proxy) is in season 3+ with the team, vs everyone else with known OC "
            "tenure. Direction: fade the post-bye team."
        ),
        f"{FAMILY}_post_bye_oc_interaction": (
            "LEAD-28 interaction: within post-bye team-games with known OC tenure, season-1/2 "
            "offensive coordinator vs season-3+. Direction: the newer staff covers more."
        ),
    }
    plain = {
        f"{FAMILY}_first_game": (
            "A team's first game right after it changes its offensive or defensive "
            "coordinator in the middle of the season. Back that team."
        ),
        f"{FAMILY}_games_2_to_4": (
            "Games two through four after a team changes its coordinator in the middle of "
            "the season. Reported for context only; any bump is expected in game one."
        ),
        f"{FAMILY}_post_bye_new_oc_back": (
            "A team coming off its bye week whose offensive coordinator is in his first or "
            "second season with the team. Back that team."
        ),
        f"{FAMILY}_post_bye_veteran_oc_fade": (
            "A team coming off its bye week whose offensive coordinator has been there "
            "three seasons or more. Fade that team."
        ),
        f"{FAMILY}_post_bye_oc_interaction": (
            "Among teams coming off a bye, the ones with a newer offensive coordinator "
            "against the ones with a long-tenured one. Back the newer staff."
        ),
    }
    commands = []
    for cell in [*r1_cells, *r2_primary]:
        extra = secondary_note[cell["name"]]
        if cell["name"].endswith("_first_game") or cell["name"].endswith("_games_2_to_4"):
            extra += " " + event_table_note
            extra += (
                " Line: tue_open_home_spread from the opener archive; decision timestamp = "
                "kickoff; minimum revision-to-kickoff lead "
                f"{float(lead_hours.min()):.1f} h."
            )
        else:
            extra += (
                " OC tenure from preseason (September 1 cutoff) observations only; names "
                "compared after stripping the parenthetical disambiguator; unknown tenure "
                "excluded (2009 entirely, 2010 year-1 only)."
            )
        commands.append(
            record_argv(
                cell,
                description=descriptions[cell["name"]],
                plain_summary=plain[cell["name"]],
                source=source_note,
                extra_notes=extra,
            )
        )

    results = {
        "family": FAMILY,
        "predeclaration": DOC,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "coordinator_history": str(history_path),
            "change_validation": str(validation_path),
            "opener_per_game": str(per_game_path),
            "features": str(features_path),
            "schedules": str(schedules_path),
        },
        "seed": SEED,
        "samples": SAMPLES,
        "events": [
            {
                **{
                    k: (v if not isinstance(v, pd.Timestamp) else v.isoformat())
                    for k, v in row.items()
                }
            }
            for row in events.to_dict("records")
        ],
        "events_without_first_game": events_without_first_game,
        "r1_first_game_rows": first_game_rows.to_dict("records"),
        "r1_min_lead_hours": float(lead_hours.min()),
        "r1_by_game_number": by_game_number.to_dict("records"),
        "r1_interim_hc_overlap": _interim_overlap(repo_root, first_open[["game_id", "team"]]),
        "r1_cells_opener": r1_cells,
        "r1_cells_close_secondary": r1_secondary,
        "r2_tenure_coverage_by_season": tenure_coverage.to_dict("records"),
        "r2_post_bye_rows_known_tenure": int((r2_full["tenure_known"] & r2_full["post_bye"]).sum()),
        "r2_cells_close_primary": r2_primary,
        "r2_cells_opener_secondary": r2_secondary,
        "record_commands": commands,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_stamped_artifact(results, output_dir / "results.json", project_root=repo_root)
    (output_dir / "record_commands.json").write_text(
        json.dumps(commands, indent=1), encoding="utf-8"
    )
    print()
    print(f"results: {output_dir / 'results.json'}")
    print(f"record commands ({len(commands)}): {output_dir / 'record_commands.json'}")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--history", default=None, help="coordinator_history.parquet path")
    parser.add_argument("--opener-per-game", default=None, help="opener archive per_game.parquet")
    parser.add_argument("--features", default=None, help="game_features.parquet path")
    parser.add_argument("--schedules", default=None, help="schedules.parquet snapshot path")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    warnings.simplefilter("ignore", category=RuntimeWarning)
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
