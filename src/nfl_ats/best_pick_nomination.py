from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.margin import MARGIN_FEATURE_PROFILES, MarginFeatureProfile
from nfl_ats.market_data import load_quote_history, tuesday_opener_quotes
from nfl_ats.outcomes import fit_margin_models_for_week
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
from nfl_ats.recorder_override import replace_week_rows, resolve_recording_forecast

NOMINATION_RIDGE_ALPHA = 2_000.0

NOMINATION_V2_ENABLED = True

SERVED_SPREAD_THRESHOLD = 7.0

CHALLENGER_ID = "best_pick_nomination_v2"

CHALLENGER_ID_V3 = "best_pick_nomination_v3"


@dataclass(frozen=True)
class DispersionPool:
    frame: pd.DataFrame
    fallback: bool
    fallback_reason: str | None
    n_games: int
    n_missing: int
    n_pool_pass: int


def week_dispersion_pool(
    market_root: Path, game_ids: Sequence[str], *, since: pd.Timestamp | None = None
) -> DispersionPool:

    ids = [str(game_id) for game_id in game_ids]
    if not ids:
        raise ValueError("week_dispersion_pool needs at least one game_id")

    quotes = (
        load_quote_history(market_root, since=since)
        if since is not None
        else load_quote_history(market_root)
    )
    opener = tuesday_opener_quotes(quotes)
    dispersion = (
        opener.rename(columns={"nflverse_game_id": "game_id", "opener_std": "spread_std"})[
            ["game_id", "spread_std"]
        ]
        .assign(game_id=lambda frame: frame["game_id"].astype(str))
        .drop_duplicates("game_id")
    )

    frame = pd.DataFrame({"game_id": ids}).merge(dispersion, on="game_id", how="left")
    return dispersion_pool_from_frame(frame)


def dispersion_pool_from_frame(frame: pd.DataFrame) -> DispersionPool:

    required = {"game_id", "spread_std"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"dispersion_pool_from_frame is missing columns: {', '.join(missing)}")
    frame = frame[["game_id", "spread_std"]].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    if frame.empty:
        raise ValueError("dispersion_pool_from_frame needs at least one game")
    n_games = len(frame)
    n_missing = int(frame["spread_std"].isna().sum())

    if n_missing > 0:
        frame["pool_pass"] = True
        return DispersionPool(frame, True, "missing_data", n_games, n_missing, n_games)

    median = frame["spread_std"].median()
    below = frame["spread_std"].lt(median)
    if not bool(below.any()):
        frame["pool_pass"] = True
        return DispersionPool(frame, True, "empty_filter", n_games, n_missing, n_games)

    frame["pool_pass"] = below
    return DispersionPool(frame, False, None, n_games, n_missing, int(below.sum()))


def fit_candidate_probabilities(
    features: pd.DataFrame,
    *,
    season: int,
    week: int,
    regressor: str,
    feature_profile: MarginFeatureProfile,
    min_train_games: int,
    ridge_alpha: float = NOMINATION_RIDGE_ALPHA,
) -> pd.DataFrame:

    if feature_profile not in MARGIN_FEATURE_PROFILES:
        raise ValueError(f"Unknown feature profile: {feature_profile!r}")
    target, margin_models = fit_margin_models_for_week(
        features,
        season=season,
        week=week,
        regressor=regressor,
        min_train_games=min_train_games,
        feature_profile=feature_profile,
        ridge_alpha=ridge_alpha,
        methods=("market_residual",),
    )
    model = margin_models["market_residual"]
    forecast = model.predict(target)
    result = target[["game_id"]].copy()
    result["game_id"] = result["game_id"].astype(str)
    result["candidate_home_cover_probability"] = forecast["home_cover_probability"].to_numpy()
    result["candidate_dist"] = (result["candidate_home_cover_probability"] - 0.5).abs()
    return result.reset_index(drop=True)


@dataclass(frozen=True)
class NominationV2Result:
    game_id: str
    n_tied_at_max: int
    tie_break: str
    probability_table: pd.DataFrame
    dispersion: DispersionPool
    spread_threshold: float | None = None
    spread_fallback: bool = False
    base_game_id: str | None = None


def _select_nominee(
    candidates: pd.DataFrame, *, rule_name: str, dispersion_tiebreak: bool
) -> tuple[str, int, str]:

    if candidates.empty:
        raise ValueError(f"{rule_name} needs at least one candidate")
    top_value = candidates["candidate_dist"].max()
    tied = candidates.loc[candidates["candidate_dist"].eq(top_value)]
    n_tied = len(tied)
    if n_tied <= 1:
        tie_break = "none"
    elif dispersion_tiebreak and tied["spread_std"].dropna().nunique() >= 2:
        tie_break = "dispersion"
    else:
        tie_break = "game_id"
    if dispersion_tiebreak:
        ordered = tied.sort_values(
            ["spread_std", "game_id"], ascending=[True, True], na_position="last"
        )
    else:
        ordered = tied.sort_values("game_id", ascending=True)
    return str(ordered.iloc[0]["game_id"]), n_tied, tie_break


def select_nominee(candidates: pd.DataFrame) -> tuple[str, int, str]:

    return _select_nominee(candidates, rule_name="select_nominee", dispersion_tiebreak=True)


def select_nominee_v3(candidates: pd.DataFrame) -> tuple[str, int, str]:

    return _select_nominee(candidates, rule_name="select_nominee_v3", dispersion_tiebreak=False)


def _nominate(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    market_root: Path,
    season: int,
    week: int,
    regressor: str,
    feature_profile: MarginFeatureProfile,
    min_train_games: int,
    ridge_alpha: float,
    select_fn: Callable[[pd.DataFrame], tuple[str, int, str]],
) -> tuple[str, int, str, pd.DataFrame, DispersionPool] | None:

    if predictions.empty or "game_id" not in predictions.columns:
        return None
    if (
        "game_type" in predictions.columns
        and not predictions["game_type"].astype(str).eq("REG").all()
    ):
        return None

    game_ids = predictions["game_id"].astype(str).tolist()
    probabilities = fit_candidate_probabilities(
        features,
        season=season,
        week=week,
        regressor=regressor,
        feature_profile=feature_profile,
        min_train_games=min_train_games,
        ridge_alpha=ridge_alpha,
    )
    available = set(probabilities["game_id"])
    missing_games = sorted(set(game_ids) - available)
    if missing_games:
        raise DataContractError(
            "Candidate probabilities are missing games on the published card: "
            f"{', '.join(missing_games[:5])}"
        )
    probabilities = probabilities.loc[probabilities["game_id"].isin(game_ids)].reset_index(
        drop=True
    )

    dispersion = week_dispersion_pool(market_root, game_ids)
    table = probabilities.merge(dispersion.frame, on="game_id", how="inner", validate="one_to_one")
    if len(table) != len(game_ids):
        raise DataContractError("Dispersion pool join dropped or duplicated games")

    candidates = table.loc[table["pool_pass"]]
    if candidates.empty:
        raise DataContractError("Dispersion-pool fallback left zero eligible candidates")

    nominee, n_tied, tie_break = select_fn(candidates)
    sorted_table = table.sort_values("game_id").reset_index(drop=True)
    return nominee, n_tied, tie_break, sorted_table, dispersion


def nominate_v2(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    market_root: Path,
    season: int,
    week: int,
    regressor: str,
    feature_profile: MarginFeatureProfile,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    ridge_alpha: float = NOMINATION_RIDGE_ALPHA,
) -> NominationV2Result | None:

    nominated = _nominate(
        predictions,
        features,
        market_root=market_root,
        season=season,
        week=week,
        regressor=regressor,
        feature_profile=feature_profile,
        min_train_games=min_train_games,
        ridge_alpha=ridge_alpha,
        select_fn=select_nominee,
    )
    if nominated is None:
        return None
    nominee, n_tied, tie_break, probability_table, dispersion = nominated
    return NominationV2Result(
        game_id=nominee,
        n_tied_at_max=n_tied,
        tie_break=tie_break,
        probability_table=probability_table,
        dispersion=dispersion,
    )


@dataclass(frozen=True)
class SpreadEligibilityResult:
    game_id: str
    n_tied_at_max: int
    tie_break: str
    probability_table: pd.DataFrame
    excluded_game_ids: tuple[str, ...]
    fallback_to_v2: bool
    base_v2_game_id: str


def apply_spread_eligibility(
    predictions: pd.DataFrame,
    base: NominationV2Result,
    *,
    threshold: float,
) -> SpreadEligibilityResult:

    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Best-Pick spread eligibility threshold must be finite and positive")
    required_predictions = {"game_id", "spread_line"}
    missing_predictions = sorted(required_predictions.difference(predictions.columns))
    if missing_predictions:
        raise DataContractError(
            "Best-Pick spread eligibility is missing card columns: "
            f"{', '.join(missing_predictions)}"
        )
    if predictions["game_id"].astype(str).duplicated().any():
        raise DataContractError("Best-Pick spread eligibility card contains duplicate games")

    required_table = {"game_id", "candidate_dist", "spread_std", "pool_pass"}
    missing_table = sorted(required_table.difference(base.probability_table.columns))
    if missing_table:
        raise DataContractError(
            f"Best-Pick v2 probability table is missing columns: {', '.join(missing_table)}"
        )

    spreads = predictions[["game_id", "spread_line"]].copy()
    spreads["game_id"] = spreads["game_id"].astype(str)
    spreads["spread_line"] = pd.to_numeric(spreads["spread_line"], errors="coerce")
    if not np.isfinite(spreads["spread_line"].to_numpy(dtype=float)).all():
        raise DataContractError("Best-Pick spread eligibility found a non-finite decision spread")

    table = base.probability_table.copy()
    table["game_id"] = table["game_id"].astype(str)
    table = table.merge(spreads, on="game_id", how="left", validate="one_to_one")
    if len(table) != len(spreads) or table["spread_line"].isna().any():
        raise DataContractError("Best-Pick spread eligibility join dropped or duplicated games")

    v2_pool = table["pool_pass"].astype(bool)
    if not bool(v2_pool.any()):
        raise DataContractError("Best-Pick v2 eligibility pool contains no candidates")
    table["spread_eligible"] = table["spread_line"].abs().lt(threshold)
    screened_pool = v2_pool & table["spread_eligible"]
    fallback_to_v2 = not bool(screened_pool.any())
    final_pool = v2_pool if fallback_to_v2 else screened_pool

    nominee, n_tied, tie_break = select_nominee(table.loc[final_pool])
    excluded = tuple(
        sorted(table.loc[v2_pool & ~table["spread_eligible"], "game_id"].astype(str).tolist())
    )
    return SpreadEligibilityResult(
        game_id=nominee,
        n_tied_at_max=n_tied,
        tie_break=tie_break,
        probability_table=table.sort_values("game_id").reset_index(drop=True),
        excluded_game_ids=excluded,
        fallback_to_v2=fallback_to_v2,
        base_v2_game_id=base.game_id,
    )


def _screened_dispersion(base: DispersionPool, eligible_game_ids: set[str]) -> DispersionPool:

    frame = base.frame.copy()
    frame["pool_pass"] = frame["pool_pass"].astype(bool) & frame["game_id"].astype(str).isin(
        eligible_game_ids
    )
    return DispersionPool(
        frame=frame,
        fallback=base.fallback,
        fallback_reason=base.fallback_reason,
        n_games=base.n_games,
        n_missing=base.n_missing,
        n_pool_pass=int(frame["pool_pass"].sum()),
    )


def nominate_v2_small_spread(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    market_root: Path,
    season: int,
    week: int,
    regressor: str,
    feature_profile: MarginFeatureProfile,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    ridge_alpha: float = NOMINATION_RIDGE_ALPHA,
    threshold: float = SERVED_SPREAD_THRESHOLD,
) -> NominationV2Result | None:

    base = nominate_v2(
        predictions,
        features,
        market_root=market_root,
        season=season,
        week=week,
        regressor=regressor,
        feature_profile=feature_profile,
        min_train_games=min_train_games,
        ridge_alpha=ridge_alpha,
    )
    if base is None:
        return None
    screened = apply_spread_eligibility(predictions, base, threshold=threshold)
    eligible = set(
        screened.probability_table.loc[
            screened.probability_table["spread_eligible"].astype(bool), "game_id"
        ]
        .astype(str)
        .tolist()
    )
    dispersion = (
        base.dispersion
        if screened.fallback_to_v2
        else _screened_dispersion(base.dispersion, eligible)
    )
    return NominationV2Result(
        game_id=screened.game_id,
        n_tied_at_max=screened.n_tied_at_max,
        tie_break=screened.tie_break,
        probability_table=screened.probability_table,
        dispersion=dispersion,
        spread_threshold=float(threshold),
        spread_fallback=screened.fallback_to_v2,
        base_game_id=screened.base_v2_game_id,
    )


@dataclass(frozen=True)
class NominationV3Result:
    game_id: str
    n_tied_at_max: int
    tie_break: str
    probability_table: pd.DataFrame
    dispersion: DispersionPool


def nominate_v3(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    market_root: Path,
    season: int,
    week: int,
    regressor: str,
    feature_profile: MarginFeatureProfile,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    ridge_alpha: float = NOMINATION_RIDGE_ALPHA,
) -> NominationV3Result | None:

    nominated = _nominate(
        predictions,
        features,
        market_root=market_root,
        season=season,
        week=week,
        regressor=regressor,
        feature_profile=feature_profile,
        min_train_games=min_train_games,
        ridge_alpha=ridge_alpha,
        select_fn=select_nominee_v3,
    )
    if nominated is None:
        return None
    nominee, n_tied, tie_break, probability_table, dispersion = nominated
    return NominationV3Result(
        game_id=nominee,
        n_tied_at_max=n_tied,
        tie_break=tie_break,
        probability_table=probability_table,
        dispersion=dispersion,
    )


def nomination_v2_tie_note(result: NominationV2Result) -> str:

    if result.n_tied_at_max <= 1:
        return ""
    if result.tie_break == "dispersion":
        return (
            f"{result.n_tied_at_max} games tied on calibrated-probability distance this week; "
            "the tie was broken by lower cross-book dispersion, not chosen arbitrarily."
        )
    return (
        f"This week {result.n_tied_at_max} games tie at the top of that signal, and the "
        "dispersion tie-break did not discriminate between them either, so choosing between "
        "them is arbitrary -- reproducible, but not a lean."
    )


def nomination_v3_tie_note(result: NominationV3Result) -> str:

    if result.n_tied_at_max <= 1:
        return ""
    return (
        f"{result.n_tied_at_max} games tie at the top of that signal in the "
        "dispersion-filtered pool; v3 breaks ties on ascending game_id alone "
        "(no dispersion tie-break), so choosing between them is arbitrary -- "
        "reproducible, but not a lean."
    )


NOMINATION_V2_METHOD_SENTENCE = "nominated by calibrated probability among low-disagreement games"

NOMINATION_SMALL_SPREAD_CLAUSE = " with a spread of six and a half or less"

NOMINATION_SMALL_SPREAD_FALLBACK_CLAUSE = (
    " (no game in this week's pool had a spread that small, so the star came from the full pool)"
)


def nomination_v2_disclosure_note(result: NominationV2Result) -> str:

    sentence = NOMINATION_V2_METHOD_SENTENCE
    if result.spread_threshold is not None:
        sentence += (
            NOMINATION_SMALL_SPREAD_FALLBACK_CLAUSE
            if result.spread_fallback
            else NOMINATION_SMALL_SPREAD_CLAUSE
        )
    dispersion = result.dispersion
    if dispersion.fallback:
        reason = (
            "at least one game was missing cross-book opener data"
            if dispersion.fallback_reason == "missing_data"
            else "no game sat strictly below the week's own median dispersion"
        )
        sentence += f" (fell back to the full, unfiltered week this week: {reason})"
    sentence += "."
    tie_note = nomination_v2_tie_note(result)
    return f"{sentence} {tie_note}".rstrip() if tie_note else sentence


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def _record_nomination_for_challenger(
    artifacts_root: Path,
    data_root: Path,
    *,
    challenger_id: str,
    nominate_fn: Callable[..., NominationV2Result | NominationV3Result | None],
    tie_note_fn: Callable[[Any], str],
    now: datetime | None,
    forecast_artifact: str | None,
    replace_week: bool,
) -> dict[str, Any]:

    entry = find_challenger(artifacts_root, challenger_id)
    status = str(entry.get("status"))
    if status != ACTIVE_CHALLENGER_STATUS:
        raise ValueError(
            f"Challenger {challenger_id!r} is registered as {status!r}; only "
            f"{ACTIVE_CHALLENGER_STATUS} challengers have picks recorded"
        )

    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError(
            "No synchronized active ATS model is available to record nomination decisions from"
        )
    forecast, metadata = resolve_recording_forecast(
        artifacts_root, active, forecast_artifact=forecast_artifact
    )
    card_path = forecast / "recommendations.csv"

    observed_config = artifact_model_config(metadata)
    declared_fingerprint = config_fingerprint(entry.get("model", {}))
    observed_fingerprint = config_fingerprint(observed_config)
    if declared_fingerprint != observed_fingerprint:
        raise DataContractError(
            f"Challenger {challenger_id!r} is registered pinned to configuration "
            f"fingerprint {declared_fingerprint}, but the current active forecast "
            f"{forecast} was produced with {observed_fingerprint}; the active model "
            "changed underneath this nomination rule -- re-register before recording"
        )

    card = pd.read_csv(card_path)
    required = {
        "game_id",
        "season",
        "week",
        "kickoff",
        "away_team",
        "home_team",
        "spread_line",
        "home_cover_probability",
    }
    missing = sorted(required.difference(card.columns))
    if missing:
        raise DataContractError(f"Active forecast card is missing columns: {', '.join(missing)}")
    if card["game_id"].duplicated().any():
        raise DataContractError("Active forecast card contains duplicate games")
    spreads = pd.to_numeric(card["spread_line"], errors="coerce")
    if not np.isfinite(spreads.to_numpy(dtype=float)).all():
        raise DataContractError("Active forecast card has games without a decision spread")
    kickoffs = pd.to_datetime(card["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Active forecast card has games without a kickoff timestamp")

    feature_table_path = observed_config.get("feature_table")
    if not feature_table_path:
        raise DataContractError(
            "Active forecast metadata carries no feature_table path to refit the "
            "alpha=2000 nomination candidate from"
        )
    features = pd.read_parquet(feature_table_path)
    season = int(card["season"].iloc[0])
    week = int(card["week"].iloc[0])
    min_train_games = metadata.get("min_train_games")
    result = nominate_fn(
        card,
        features,
        market_root=data_root / "market" / "raw",
        season=season,
        week=week,
        regressor=str(metadata.get("regressor", "ridge")),
        feature_profile=metadata.get("feature_profile"),
        min_train_games=int(min_train_games) if min_train_games else DEFAULT_MIN_TRAIN_GAMES,
    )
    nominee_id = result.game_id if result is not None else None

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    whole_week_pre_kickoff = bool(kickoffs.gt(recorded_at).all())
    existing = load_challenger_decisions(artifacts_root)
    replaced_rows = 0
    left_post_kickoff = 0
    if replace_week and nominee_id is not None and whole_week_pre_kickoff:
        existing, replaced_rows, left_post_kickoff = replace_week_rows(
            existing,
            challenger_ledger_path(artifacts_root),
            season=season,
            week=week,
            recorded_at=recorded_at,
            columns=CHALLENGER_DECISION_COLUMNS,
            challenger_id=challenger_id,
        )
    mine = existing.loc[existing["challenger_id"].astype(str).eq(challenger_id)]
    already_ids = set(mine["game_id"].astype(str))

    already = nominee_id is not None and nominee_id in already_ids
    post_kickoff_skipped = nominee_id is not None and not already and not whole_week_pre_kickoff
    if nominee_id is None or already or not whole_week_pre_kickoff:
        fresh = card.iloc[0:0]
    else:
        fresh = card.loc[card["game_id"].astype(str).eq(nominee_id)]

    decisions = pd.DataFrame(
        {
            "recorded_at_utc": recorded_at,
            "challenger_id": challenger_id,
            "config_fingerprint": observed_fingerprint,
            "source_artifact": forecast.name,
            "source_sha256": sha256_file(card_path),
            "forecast_created_at_utc": pd.to_datetime(
                metadata.get("created_at_utc"), utc=True, errors="coerce"
            ),
            "feature_profile": str(metadata.get("feature_profile")),
            "feature_table_sha256": str(observed_config.get("feature_table_sha256")),
            "game_id": fresh["game_id"].astype(str),
            "season": fresh["season"].astype(int),
            "week": fresh["week"].astype(int),
            "kickoff": kickoffs.loc[fresh.index],
            "away_team": fresh["away_team"].astype(str),
            "home_team": fresh["home_team"].astype(str),
            "pick_side": np.where(
                pd.to_numeric(fresh["home_cover_probability"], errors="coerce").ge(0.5),
                "HOME",
                "AWAY",
            ).astype(str),
            "bet_side": "PASS",
            "decision_home_spread": spreads.loc[fresh.index].astype(float),
            "edge": np.nan,
        }
    )
    if not decisions.empty:
        combined = (
            decisions if existing.empty else pd.concat([existing, decisions], ignore_index=True)
        )
        atomic_parquet(
            combined[list(CHALLENGER_DECISION_COLUMNS)], challenger_ledger_path(artifacts_root)
        )
        ledger_rows = len(combined)
    else:
        ledger_rows = len(existing)

    return {
        "challenger_id": challenger_id,
        "season": season,
        "week": week,
        "source_artifact": forecast.name,
        "config_fingerprint": observed_fingerprint,
        "nominated_game_id": nominee_id,
        "nomination_tie_note": tie_note_fn(result) if result is not None else "",
        "recorded": len(decisions),
        "replaced_rows": replaced_rows,
        "left_post_kickoff": left_post_kickoff,
        "already_recorded": int(already),
        "post_kickoff_skipped": int(post_kickoff_skipped),
        "ledger_rows": int(ledger_rows),
    }


def record_nomination_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
    forecast_artifact: str | None = None,
    replace_week: bool = False,
) -> dict[str, Any]:

    return _record_nomination_for_challenger(
        artifacts_root,
        data_root,
        challenger_id=CHALLENGER_ID,
        nominate_fn=nominate_v2,
        tie_note_fn=nomination_v2_tie_note,
        now=now,
        forecast_artifact=forecast_artifact,
        replace_week=replace_week,
    )


def record_nomination_v3_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
    forecast_artifact: str | None = None,
    replace_week: bool = False,
) -> dict[str, Any]:

    return _record_nomination_for_challenger(
        artifacts_root,
        data_root,
        challenger_id=CHALLENGER_ID_V3,
        nominate_fn=nominate_v3,
        tie_note_fn=nomination_v3_tie_note,
        now=now,
        forecast_artifact=forecast_artifact,
        replace_week=replace_week,
    )
