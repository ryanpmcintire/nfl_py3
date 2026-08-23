"""Immutable, pre-kickoff forecast records for prospective evaluation."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_json, atomic_parquet, run_id
from nfl_ats.pick_refresh import (
    MOVEMENT_POLICY_THRESHOLD,
    current_captured_home_spread,
    original_card,
    sunday_pick_lock,
)
from nfl_ats.prediction_safety import validate_prediction_card
from nfl_ats.prospective_scoring import (
    ACTIVE_CHALLENGER_STATUS,
    CHALLENGER_DECISION_COLUMNS,
    artifact_model_config,
    challenger_ledger_path,
    config_fingerprint,
    find_challenger,
    load_challenger_decisions,
)
from nfl_ats.provenance import sha256_file

FROZEN_PREDICTION_COLUMNS = (
    "game_id",
    "season",
    "week",
    "gameday",
    "kickoff",
    "away_team",
    "home_team",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "home_cover_probability",
    "pick",
    "bet_side",
    "edge",
    "bet_odds",
    "break_even_probability",
    "market_home_no_vig_probability",
    "market_hold",
    "train_rows",
    "train_max_gameday",
)


@dataclass(frozen=True)
class FrozenForecast:
    forecast_id: str
    directory: Path
    games: int


def verify_frozen_forecast(directory: Path) -> dict[str, Any]:
    """Verify file integrity and re-run the prediction safety contract."""

    manifest_path = directory / "manifest.json"
    prediction_path = directory / "predictions.parquet"
    if not manifest_path.is_file() or not prediction_path.is_file():
        raise ValueError(f"Incomplete frozen forecast: {directory}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid frozen forecast manifest: {manifest_path}")
    if sha256_file(prediction_path) != payload.get("predictions_sha256"):
        raise ValueError(f"Frozen forecast digest mismatch: {directory}")
    predictions = pd.read_parquet(prediction_path)
    if len(predictions) != payload.get("games"):
        raise ValueError(f"Frozen forecast row-count mismatch: {directory}")
    forecast_id = payload.get("forecast_id")
    if forecast_id != directory.name or not predictions["forecast_id"].eq(forecast_id).all():
        raise ValueError(f"Frozen forecast identity mismatch: {directory}")
    model = payload.get("model", {})
    min_edge = float(model.get("min_edge", 0.02)) if isinstance(model, dict) else 0.02
    audit = validate_prediction_card(
        predictions,
        min_edge=min_edge,
        expected_season=int(payload["season"]),
        expected_week=int(payload["week"]),
        prospective=True,
        created_at=datetime.fromisoformat(str(payload["frozen_at_utc"])),
    )
    if int(payload.get("schema_version", 1)) >= 2:
        recorded = payload.get("prediction_safety")
        if not isinstance(recorded, dict):
            raise ValueError(f"Frozen forecast is missing its safety audit: {directory}")
        if recorded != audit.to_dict():
            raise ValueError(f"Frozen forecast safety audit mismatch: {directory}")
    return payload


def _utc(instant: datetime | None) -> datetime:
    value = instant or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def freeze_forecast(
    predictions: pd.DataFrame,
    metadata: dict[str, Any],
    root: Path,
    *,
    created_at: datetime | None = None,
) -> FrozenForecast:
    """Write a new immutable forecast directory after enforcing pre-kickoff timing."""

    missing = sorted(set(FROZEN_PREDICTION_COLUMNS).difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions cannot be frozen; missing columns: {', '.join(missing)}")
    if predictions.empty:
        raise ValueError("Predictions cannot be frozen because there are no games")
    if predictions["game_id"].duplicated().any():
        raise ValueError("Predictions cannot be frozen with duplicate game IDs")
    outcome_columns = [
        column for column in ("home_cover", "result", "ats_margin") if column in predictions
    ]
    if outcome_columns and predictions[outcome_columns].notna().any(axis=None):
        raise ValueError("Prospective forecasts can only be frozen before outcomes are known")
    required_market_values = ("spread_line", "home_spread_odds", "away_spread_odds")
    missing_market = predictions.loc[
        predictions.loc[:, list(required_market_values)]
        .apply(pd.to_numeric, errors="coerce")
        .isna()
        .any(axis=1),
        "game_id",
    ].astype(str)
    if not missing_market.empty:
        examples = ", ".join(missing_market.tolist()[:5])
        raise ValueError(f"Cannot freeze games with missing lines or prices: {examples}")

    frozen_at = _utc(created_at)
    kickoffs = pd.to_datetime(predictions["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        missing_games = predictions.loc[kickoffs.isna(), "game_id"].astype(str).tolist()
        examples = ", ".join(missing_games[:5])
        raise ValueError(f"Cannot prove pre-kickoff timing; kickoff is missing for: {examples}")
    if kickoffs.le(frozen_at).any():
        started = predictions.loc[kickoffs.le(frozen_at), "game_id"].astype(str).tolist()
        examples = ", ".join(started[:5])
        raise ValueError(f"Cannot freeze games at or after kickoff: {examples}")

    season = int(predictions["season"].iloc[0])
    week = int(predictions["week"].iloc[0])
    if predictions["season"].nunique() != 1 or predictions["week"].nunique() != 1:
        raise ValueError("A frozen forecast must contain exactly one season and week")
    min_edge = float(metadata.get("min_edge", 0.02))
    audit = validate_prediction_card(
        predictions,
        min_edge=min_edge,
        expected_season=season,
        expected_week=week,
        prospective=True,
        created_at=frozen_at,
    )
    forecast_id = f"{season}-week-{week:02d}-{run_id(frozen_at)}"
    destination = root / forecast_id
    if destination.exists():
        raise ValueError(f"Frozen forecast already exists: {destination}")

    frozen = predictions.loc[:, list(FROZEN_PREDICTION_COLUMNS)].copy()
    frozen["kickoff"] = kickoffs
    frozen.insert(0, "frozen_at_utc", frozen_at)
    frozen.insert(0, "forecast_id", forecast_id)
    prediction_path = destination / "predictions.parquet"
    atomic_parquet(frozen, prediction_path)
    manifest = {
        "schema_version": 2,
        "forecast_id": forecast_id,
        "frozen_at_utc": frozen_at.isoformat(),
        "season": season,
        "week": week,
        "games": len(frozen),
        "earliest_kickoff_utc": kickoffs.min().isoformat(),
        "latest_kickoff_utc": kickoffs.max().isoformat(),
        "predictions_sha256": sha256_file(prediction_path),
        "prediction_safety": audit.to_dict(),
        "model": metadata,
    }
    # Written last: its presence is the commit marker for a complete record.
    atomic_json(manifest, destination / "manifest.json")
    return FrozenForecast(forecast_id=forecast_id, directory=destination, games=len(frozen))


# ---------------------------------------------------------------------------
# Publish-time challenger recorders (POL-10): both challengers below are
# post-prediction transforms of the PLAYED production-chain pick, so their
# base card IS the paper-decision ledger row record_paper_decisions wrote
# earlier in the same publish-predictions --record-decisions call. Neither is
# wired into publishing.py; both only ever append to the SEPARATE prospective
# challenger ledger. Every absent-input path fails open (a skipped week or a
# kept pick), never an exception out of publish.
# ---------------------------------------------------------------------------

MOVEMENT_RULE_COMPOSED_CHALLENGER_ID = "movement_rule_composed_v1"
NFLCOM_REFRESH_OUT2_STARTERS_CHALLENGER_ID = "nflcom_friday_refresh_out2_starters_v1"

#: Frozen by docs/nflcom_friday_refresh.md ("Overlay rule (frozen before
#: scoring)") and registry/weak_signals.json:nflcom_refresh_out2_starters_on_chain.
NFLCOM_STARTER_OUT_THRESHOLD = 2


def _require_active_status(entry: Mapping[str, Any], challenger_id: str) -> None:
    status = str(entry.get("status"))
    if status != ACTIVE_CHALLENGER_STATUS:
        raise ValueError(
            f"Challenger {challenger_id!r} is registered as {status!r}; only "
            f"{ACTIVE_CHALLENGER_STATUS} challengers have picks recorded"
        )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def _active_forecast_context(
    artifacts_root: Path, entry: Mapping[str, Any], challenger_id: str
) -> tuple[dict[str, Any], str, str, str]:
    """The active model's linked weekly forecast, fingerprint-gated.

    Mirrors nfl_ats.surface_switch_tilt_overlay's recorder exactly: refuses to
    record when the active model's live configuration fingerprint no longer
    matches the snapshot this challenger was registered against. Returns
    (metadata, observed_fingerprint, source_artifact_name, source_card_sha256).
    """

    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError("No synchronized active ATS model is available to record decisions from")
    forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast is None:
        raise ValueError("Active ATS model has no linked weekly forecast")
    metadata_path = forecast / "metadata.json"
    card_path = forecast / "recommendations.csv"
    if not metadata_path.is_file() or not card_path.is_file():
        raise ValueError(f"Linked weekly forecast is incomplete: {forecast}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("active_model_id") != active.get("model_id"):
        raise ValueError("Weekly forecast model ID does not match the active model")
    if metadata.get("synchronization_status") != "SYNCHRONIZED":
        raise ValueError("Weekly forecast is not synchronized with an evaluation")
    declared_fingerprint = config_fingerprint(entry.get("model", {}))
    observed_fingerprint = config_fingerprint(artifact_model_config(metadata))
    if declared_fingerprint != observed_fingerprint:
        raise DataContractError(
            f"Challenger {challenger_id!r} is registered pinned to configuration "
            f"fingerprint {declared_fingerprint}, but the current active forecast "
            f"{forecast} was produced with {observed_fingerprint}; the active model "
            "changed underneath this challenger -- re-register before recording"
        )
    return metadata, observed_fingerprint, forecast.name, sha256_file(card_path)


def _chain_card(artifacts_root: Path, metadata: Mapping[str, Any]) -> pd.DataFrame:
    """The week's played-chain picks from the paper-decision ledger.

    Both challengers compose ON TOP of the published chain (raw model -> coach
    fade -> player-arrests policy), whose frozen side and grading line live in
    artifacts/clv_ledger/decisions.parquet -- the same read
    nfl_ats.pick_refresh.original_card uses. Empty means the week was never
    recorded by --record-decisions, which for these recorders is a hard error:
    they run inside that very call, after the paper ledger is written.
    """

    season = int(metadata["season"])
    week = int(metadata["week"])
    ledger = original_card(artifacts_root, season=season, week=week)
    if ledger.empty:
        raise ValueError(
            f"No recorded original card for {season} week {week}: these challengers "
            "compose on top of the paper ledger's played chain picks, which only "
            "`publish-predictions --record-decisions` writes."
        )
    required = {"game_id", "kickoff", "away_team", "home_team", "decision_home_spread", "pick_side"}
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise DataContractError(f"Paper ledger is missing columns: {', '.join(missing)}")
    if ledger["game_id"].duplicated().any():
        raise DataContractError("Paper ledger contains duplicate games for this week")
    spreads = pd.to_numeric(ledger["decision_home_spread"], errors="coerce")
    if not np.isfinite(spreads.to_numpy(dtype=float)).all():
        raise DataContractError("Paper ledger has games without a decision spread")
    kickoffs = pd.to_datetime(ledger["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Paper ledger has games without a kickoff timestamp")
    sides = set(ledger["pick_side"].astype(str))
    if not sides <= {"HOME", "AWAY"}:
        raise DataContractError(f"Paper ledger has invalid pick sides: {sorted(sides)}")
    return ledger.reset_index(drop=True)


def _append_challenger_decisions(artifacts_root: Path, rows: pd.DataFrame) -> int:
    existing = load_challenger_decisions(artifacts_root)
    combined = rows if existing.empty else pd.concat([existing, rows], ignore_index=True)
    atomic_parquet(
        combined[list(CHALLENGER_DECISION_COLUMNS)], challenger_ledger_path(artifacts_root)
    )
    return len(combined)


def movement_rule_pick(chain_pick_side: str, movement_delta: float | None) -> str:
    """The composed movement rule, as a pure function.

    ``movement_delta`` is current-captured minus frozen-Tuesday home spread,
    home-oriented (the exact quantity nfl_ats.pick_refresh.plan_refresh already
    computes as ``RefreshedGame.movement_delta``). At least 1.0 point of move
    follows the market side (HOME if the home-oriented number rose, else AWAY);
    below threshold -- or no usable captured line, ``None`` -- keeps the chain
    pick. The 1.0 threshold is MOVEMENT_POLICY_THRESHOLD, frozen by the
    predeclared 0.5/1.0 grid in docs/observed_movement_channel.md; the side
    logic mirrors pick_refresh._movement_side verbatim.
    """

    if movement_delta is None or abs(float(movement_delta)) < MOVEMENT_POLICY_THRESHOLD:
        return chain_pick_side
    return "HOME" if float(movement_delta) > 0.0 else "AWAY"


def record_movement_rule_composed_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the movement-rule-on-chain arm to the prospective challenger ledger.

    Rule (registry/weak_signals.json:movement_rule_composed_chain,
    docs/movement_composition_eval.md): if the latest locally captured market
    home spread moved at least 1.0 point from the frozen Tuesday decision line,
    follow the market side; otherwise keep the chain pick. Reuses
    nfl_ats.pick_refresh.current_captured_home_spread -- the identical
    read-only captured-line read the played movement policy consumes -- so this
    challenger never fetches and never rebuilds market plumbing.

    FAIL-OPEN: with no fresh capture at all the whole week is skipped
    (``{"recorded": 0, "skipped": True}``), because recording kept picks with
    no market look would be indistinguishable from "no move". A game merely
    missing from an otherwise-fresh capture keeps the chain pick and is counted
    in ``games_without_captured_line``.
    """

    entry = find_challenger(artifacts_root, MOVEMENT_RULE_COMPOSED_CHALLENGER_ID)
    _require_active_status(entry, MOVEMENT_RULE_COMPOSED_CHALLENGER_ID)
    metadata, fingerprint, source_artifact, source_sha = _active_forecast_context(
        artifacts_root, entry, MOVEMENT_RULE_COMPOSED_CHALLENGER_ID
    )
    ledger = _chain_card(artifacts_root, metadata)

    recorded_at = _record_instant(now)
    kickoffs = pd.to_datetime(ledger["kickoff"], errors="coerce", utc=True)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")

    current_lines, line_metadata = current_captured_home_spread(data_root, now=recorded_at)
    if not line_metadata.get("fresh", False):
        return {
            "challenger_id": MOVEMENT_RULE_COMPOSED_CHALLENGER_ID,
            "season": int(metadata["season"]),
            "week": int(metadata["week"]),
            "recorded": 0,
            "skipped": True,
            "reason": (
                "no fresh captured market line this pass "
                f"({line_metadata.get('reason', 'unknown')}); recording kept picks "
                "with no market look would be indistinguishable from no move"
            ),
        }

    existing = load_challenger_decisions(artifacts_root)
    mine = existing.loc[
        existing["challenger_id"].astype(str).eq(MOVEMENT_RULE_COMPOSED_CHALLENGER_ID)
    ]
    already = set(mine["game_id"].astype(str))
    pre_kickoff = kickoffs.gt(recorded_at)

    picks: list[str] = []
    without_line = 0
    flips = 0
    keep_positions: list[int] = []
    game_ids = ledger["game_id"].astype(str).tolist()
    chain_sides = ledger["pick_side"].astype(str).tolist()
    for position, game_id in enumerate(game_ids):
        if not bool(pre_kickoff.iloc[position]) or game_id in already:
            continue
        chain_side = chain_sides[position]
        delta = current_lines.get(game_id)
        delta = float(delta) if delta is not None and pd.notna(delta) else None
        if delta is None:
            without_line += 1
        side = movement_rule_pick(chain_side, delta)
        flips += int(side != chain_side)
        picks.append(side)
        keep_positions.append(position)

    created_raw = metadata.get("created_at_utc")
    kept = ledger.iloc[keep_positions] if keep_positions else ledger.iloc[0:0]
    decision_rows = (
        pd.DataFrame(
            {
                "recorded_at_utc": recorded_at,
                "challenger_id": MOVEMENT_RULE_COMPOSED_CHALLENGER_ID,
                "config_fingerprint": fingerprint,
                "source_artifact": source_artifact,
                "source_sha256": source_sha,
                "forecast_created_at_utc": (
                    pd.to_datetime(created_raw, utc=True, errors="coerce")
                    if created_raw is not None
                    else pd.NaT
                ),
                "feature_profile": str(metadata.get("feature_profile")),
                "feature_table_sha256": str(
                    artifact_model_config(metadata).get("feature_table_sha256")
                ),
                "game_id": kept["game_id"].astype(str).to_numpy(),
                "season": kept["season"].astype(int).to_numpy(),
                "week": kept["week"].astype(int).to_numpy(),
                "kickoff": kickoffs.iloc[np.asarray(keep_positions, dtype=np.int64)].to_numpy()
                if keep_positions
                else [],
                "away_team": _canonical_team(kept["away_team"]).to_numpy(),
                "home_team": _canonical_team(kept["home_team"]).to_numpy(),
                "pick_side": np.asarray(picks, dtype=str),
                "bet_side": "PASS",
                "decision_home_spread": (
                    pd.to_numeric(kept["decision_home_spread"], errors="coerce")
                    .astype(float)
                    .to_numpy()
                ),
                "edge": np.nan,
            }
        )
        if keep_positions
        else pd.DataFrame({column: [] for column in CHALLENGER_DECISION_COLUMNS})
    )

    ledger_rows = (
        _append_challenger_decisions(artifacts_root, decision_rows)
        if not decision_rows.empty
        else len(existing)
    )
    return {
        "challenger_id": MOVEMENT_RULE_COMPOSED_CHALLENGER_ID,
        "season": int(metadata["season"]),
        "week": int(metadata["week"]),
        "source_artifact": source_artifact,
        "config_fingerprint": fingerprint,
        "recorded": len(decision_rows),
        "already_recorded": len(already & set(ledger["game_id"].astype(str))),
        "post_kickoff_skipped": int((~pre_kickoff).sum()),
        "ledger_rows": int(ledger_rows),
        "movement_flips": flips,
        "games_without_captured_line": without_line,
        "captured_line_games": int(line_metadata.get("games_with_current_line", 0)),
    }


def nflcom_out2_starters_flip(
    chain_pick_home: bool, picked_starter_out: float, opp_starter_out: float
) -> bool:
    """The frozen NFL.com Friday-refresh rule, as a pure function.

    Flip to the opponent iff the picked team carries at least
    NFLCOM_STARTER_OUT_THRESHOLD Out designations on starter-caliber players
    AND the opponent carries fewer; both flagged keeps (docs/nflcom_friday_refresh.md,
    "Overlay rule (frozen before scoring)"). Returns the flipped arm's HOME side.
    """

    picked_flag = picked_starter_out >= NFLCOM_STARTER_OUT_THRESHOLD
    opp_flag = opp_starter_out >= NFLCOM_STARTER_OUT_THRESHOLD
    if picked_flag and not opp_flag:
        return not chain_pick_home
    return chain_pick_home


# The four helpers below are PORTED VERBATIM from
# scripts/nflcom_friday_designation_screen.py (QA_OR_WORSE / SUFFIXES /
# STARTER_SNAP_SHARE / normalize_name / initial_last_key / load_report_flags /
# build_starter_keys), the machinery registry/weak_signals.json:
# nflcom_refresh_out2_starters_on_chain was measured with -- same constants,
# same regex-free normalization steps, same week+1 starter-proxy keying.
# Duplicated rather than imported for the reason nfl_ats.pick_refresh gives for
# its own duplicated helper: this src/ module must not depend on another
# file's script-level names, and neither copy should change without the other.

_NFLCOM_QA_OR_WORSE = frozenset({"questionable", "doubtful", "out"})
_NFLCOM_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_NFLCOM_STARTER_SNAP_SHARE = 0.50


def _normalize_nflcom_name(name: object) -> str:
    if not isinstance(name, str):
        return ""
    lowered = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    lowered = lowered.casefold().replace("'", "").replace(".", " ")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return " ".join(tok for tok in lowered.split() if tok not in _NFLCOM_SUFFIXES)


def _initial_last_key(name: str) -> tuple[str, str]:
    tokens = _normalize_nflcom_name(name).split()
    if not tokens:
        return ("", "")
    return (tokens[0][0], tokens[-1] if len(tokens) > 1 else "")


def _load_nflcom_report(injuries_parquet: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    parsed = pd.read_parquet(injuries_parquet).copy()
    counts = {"rows_total": len(parsed)}
    parsed["norm_name"] = parsed["player"].map(_normalize_nflcom_name)
    parsed["status_norm"] = parsed["game_status"].astype("string").str.strip().str.casefold()
    qa = parsed.loc[parsed["status_norm"].isin(_NFLCOM_QA_OR_WORSE)].copy()
    counts["qa_or_worse_rows"] = len(qa)
    counts["out_rows"] = int((qa["status_norm"] == "out").sum())
    return qa, counts


def _nflcom_starter_key_sets(
    snaps_path: Path,
) -> tuple[set[tuple[int, int, str, str]], set[tuple[int, int, str, str, str]]]:
    snaps = pd.read_parquet(
        snaps_path,
        columns=["season", "game_type", "week", "team", "player", "offense_pct", "defense_pct"],
    )
    snaps = snaps.loc[snaps["game_type"] == "REG"].copy()
    snaps["norm_name"] = snaps["player"].map(_normalize_nflcom_name)
    snaps["share"] = snaps[["offense_pct", "defense_pct"]].max(axis=1)
    starters = snaps.loc[snaps["share"] >= _NFLCOM_STARTER_SNAP_SHARE].copy()
    exact: set[tuple[int, int, str, str]] = set()
    fuzzy: set[tuple[int, int, str, str, str]] = set()
    for season_value, week_value, team_value, name_value in zip(
        starters["season"],
        starters["week"],
        starters["team"],
        starters["norm_name"],
        strict=True,
    ):
        season = int(str(season_value))
        week = int(str(week_value))
        team = str(team_value)
        name = str(name_value)
        key_next = (season, week + 1, team)
        exact.add((*key_next, name))
        init_last = _initial_last_key(name)
        if init_last != ("", ""):
            fuzzy.add((*key_next, *init_last))
    return exact, fuzzy


def _latest_nflcom_injuries_snapshot(
    data_root: Path,
) -> tuple[Path, dict[tuple[int, int], str]] | None:
    root = data_root / "raw" / "nflcom_injuries"
    if not root.is_dir():
        return None
    manifests = sorted(root.glob("*/manifest.json"))
    if not manifests:
        return None
    snapshot_dir = manifests[-1].parent
    manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    pages: dict[tuple[int, int], str] = {}
    for page in manifest.get("pages", []) or []:
        try:
            pages[(int(page["season"]), int(page["week"]))] = str(page["fetched_at_utc"])
        except (KeyError, TypeError, ValueError):
            continue
    return snapshot_dir, pages


def record_nflcom_refresh_out2_starters_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the NFL.com Friday out>=2-starters fade arm to the challenger ledger.

    Rule text frozen in docs/nflcom_friday_refresh.md ("2026 prospective
    challenger registration"), pasted verbatim into this challenger's
    challengers.json registration: flip the chain pick to the opponent iff the
    picked team carries >=2 Out designations on starter-caliber players
    (>=50% of offense or defense snaps in the team's most recent prior REG
    game; Week 1 proxy unavailable = no flag) per the week's FINAL NFL.com
    league injury page, and the opponent carries <2; both flagged keeps.

    Signal inputs are PORTED VERBATIM from
    scripts/nflcom_friday_designation_screen.py's normalization/starter-proxy
    machinery (see the helper block below), the same machinery
    registry/weak_signals.json:nflcom_refresh_out2_starters_on_chain measured
    (+2.1795 pts, P+ 0.9954, three seasons, selection-inflated upper bound).

    FAIL-OPEN per the frozen rule text: no snapshot, no snap-counts table, or a
    week's page failing the freshness gate (fetched >= Friday 16:00 ET of that
    game week AND < the week's earliest kickoff) skips the week with
    ``{"recorded": 0, "skipped": True}`` rather than ever raising into publish.
    """

    entry = find_challenger(artifacts_root, NFLCOM_REFRESH_OUT2_STARTERS_CHALLENGER_ID)
    _require_active_status(entry, NFLCOM_REFRESH_OUT2_STARTERS_CHALLENGER_ID)
    metadata, fingerprint, source_artifact, source_sha = _active_forecast_context(
        artifacts_root, entry, NFLCOM_REFRESH_OUT2_STARTERS_CHALLENGER_ID
    )
    ledger = _chain_card(artifacts_root, metadata)
    season = int(metadata["season"])
    week = int(metadata["week"])

    recorded_at = _record_instant(now)
    kickoffs = pd.to_datetime(ledger["kickoff"], errors="coerce", utc=True)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")

    snapshot = _latest_nflcom_injuries_snapshot(data_root)
    snaps_candidates = sorted((data_root / "players" / "raw").glob("*/snap_counts.parquet"))
    if snapshot is None:
        return {
            "challenger_id": NFLCOM_REFRESH_OUT2_STARTERS_CHALLENGER_ID,
            "season": season,
            "week": week,
            "recorded": 0,
            "skipped": True,
            "reason": "no_nflcom_injuries_snapshot",
            "detail": "no data/raw/nflcom_injuries/*/manifest.json snapshot exists yet",
        }
    if not snaps_candidates:
        return {
            "challenger_id": NFLCOM_REFRESH_OUT2_STARTERS_CHALLENGER_ID,
            "season": season,
            "week": week,
            "recorded": 0,
            "skipped": True,
            "reason": "no_snap_counts_snapshot",
            "detail": "no data/players/raw/*/snap_counts.parquet snapshot exists yet",
        }
    snapshot_dir, fetched_by_week = snapshot

    fetched_raw = fetched_by_week.get((season, week))
    friday_gate = sunday_pick_lock(kickoffs) - pd.Timedelta(days=2)
    earliest_kickoff = kickoffs.min()
    gate_reason = ""
    if fetched_raw is None:
        gate_reason = f"page ({season}, week {week}) absent from snapshot manifest"
    else:
        fetched = pd.Timestamp(fetched_raw)
        fetched = (
            fetched.tz_localize("UTC") if fetched.tzinfo is None else fetched.tz_convert("UTC")
        )
        if fetched < friday_gate:
            gate_reason = (
                f"page fetched {fetched.isoformat()} is before Friday 16:00 ET of the "
                f"game week ({friday_gate.isoformat()})"
            )
        elif fetched >= earliest_kickoff:
            gate_reason = (
                f"page fetched {fetched.isoformat()} is at or after the week's earliest "
                f"kickoff ({earliest_kickoff.isoformat()})"
            )
    if gate_reason:
        return {
            "challenger_id": NFLCOM_REFRESH_OUT2_STARTERS_CHALLENGER_ID,
            "season": season,
            "week": week,
            "recorded": 0,
            "skipped": True,
            "reason": f"freshness gate failed: {gate_reason}",
        }

    qa, _counts = _load_nflcom_report(snapshot_dir / "injuries.parquet")
    out_rows = qa.loc[qa["status_norm"].eq("out")].copy()
    starter_exact, starter_fuzzy = _nflcom_starter_key_sets(snaps_candidates[-1])
    is_starter: list[bool] = []
    for row_season, row_week, row_team, name in zip(
        out_rows["season"], out_rows["week"], out_rows["team"], out_rows["norm_name"], strict=True
    ):
        key3 = (int(row_season), int(row_week), str(row_team))
        init_last = _initial_last_key(str(name))
        is_starter.append(
            (*key3, str(name)) in starter_exact
            or (init_last != ("", "") and (*key3, *init_last) in starter_fuzzy)
        )
    out_rows["is_starter_caliber"] = is_starter
    grouped = (
        out_rows.groupby(["season", "week", "team"], as_index=False)["is_starter_caliber"]
        .sum()
        .rename("starter_out")
        .reset_index()
    )
    starter_out: dict[tuple[int, int, str], int] = {}
    for group_season, group_week, group_team, count in zip(
        grouped["season"],
        grouped["week"],
        grouped["team"],
        grouped["starter_out"],
        strict=True,
    ):
        team_code = str(group_team)
        key_team = TEAM_ABBREVIATION_ALIASES.get(team_code, team_code)
        starter_out[(int(group_season), int(group_week), key_team)] = int(count)

    existing = load_challenger_decisions(artifacts_root)
    mine = existing.loc[
        existing["challenger_id"].astype(str).eq(NFLCOM_REFRESH_OUT2_STARTERS_CHALLENGER_ID)
    ]
    already = set(mine["game_id"].astype(str))
    pre_kickoff = kickoffs.gt(recorded_at)

    picks = []
    flips = 0
    keeps_from_tie_or_missing = 0
    keep_positions: list[int] = []
    game_ids = ledger["game_id"].astype(str).tolist()
    chain_sides = ledger["pick_side"].astype(str).tolist()
    home_teams = ledger["home_team"].astype(str).tolist()
    away_teams = ledger["away_team"].astype(str).tolist()
    for position, game_id in enumerate(game_ids):
        if not bool(pre_kickoff.iloc[position]) or game_id in already:
            continue
        chain_home = chain_sides[position] == "HOME"
        home_team = TEAM_ABBREVIATION_ALIASES.get(home_teams[position], home_teams[position])
        away_team = TEAM_ABBREVIATION_ALIASES.get(away_teams[position], away_teams[position])
        picked_team, opp_team = (home_team, away_team) if chain_home else (away_team, home_team)
        picked_count = starter_out.get((season, week, picked_team), 0)
        opp_count = starter_out.get((season, week, opp_team), 0)
        new_home = nflcom_out2_starters_flip(chain_home, picked_count, opp_count)
        if new_home != chain_home:
            flips += 1
        else:
            keeps_from_tie_or_missing += 1
        picks.append("HOME" if new_home else "AWAY")
        keep_positions.append(position)

    created_raw = metadata.get("created_at_utc")
    kept = ledger.iloc[keep_positions] if keep_positions else ledger.iloc[0:0]
    decision_rows = (
        pd.DataFrame(
            {
                "recorded_at_utc": recorded_at,
                "challenger_id": NFLCOM_REFRESH_OUT2_STARTERS_CHALLENGER_ID,
                "config_fingerprint": fingerprint,
                "source_artifact": source_artifact,
                "source_sha256": source_sha,
                "forecast_created_at_utc": (
                    pd.to_datetime(created_raw, utc=True, errors="coerce")
                    if created_raw is not None
                    else pd.NaT
                ),
                "feature_profile": str(metadata.get("feature_profile")),
                "feature_table_sha256": str(
                    artifact_model_config(metadata).get("feature_table_sha256")
                ),
                "game_id": kept["game_id"].astype(str).to_numpy(),
                "season": kept["season"].astype(int).to_numpy(),
                "week": kept["week"].astype(int).to_numpy(),
                "kickoff": kickoffs.iloc[np.asarray(keep_positions, dtype=np.int64)].to_numpy()
                if keep_positions
                else [],
                "away_team": _canonical_team(kept["away_team"]).to_numpy(),
                "home_team": _canonical_team(kept["home_team"]).to_numpy(),
                "pick_side": np.asarray(picks, dtype=str),
                "bet_side": "PASS",
                "decision_home_spread": (
                    pd.to_numeric(kept["decision_home_spread"], errors="coerce")
                    .astype(float)
                    .to_numpy()
                ),
                "edge": np.nan,
            }
        )
        if keep_positions
        else pd.DataFrame({column: [] for column in CHALLENGER_DECISION_COLUMNS})
    )

    ledger_rows = (
        _append_challenger_decisions(artifacts_root, decision_rows)
        if not decision_rows.empty
        else len(existing)
    )
    return {
        "challenger_id": NFLCOM_REFRESH_OUT2_STARTERS_CHALLENGER_ID,
        "season": season,
        "week": week,
        "source_artifact": source_artifact,
        "config_fingerprint": fingerprint,
        "snapshot_dir": snapshot_dir.name,
        "recorded": len(decision_rows),
        "already_recorded": len(already & set(ledger["game_id"].astype(str))),
        "post_kickoff_skipped": int((~pre_kickoff).sum()),
        "ledger_rows": int(ledger_rows),
        "overlay_flips": flips,
        "games_kept_no_flip": keeps_from_tie_or_missing,
    }
