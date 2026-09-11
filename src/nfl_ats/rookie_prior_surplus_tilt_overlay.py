from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
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

CHALLENGER_ID = "rookie_prior_surplus_tilt_overlay"


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def _load_screen_module(repo_root: Path) -> Any:
    name = "xlg06_rookie_priors_screen"
    path = repo_root / "scripts" / f"{name}.py"
    scripts_dir = str(repo_root / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DataContractError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rookie_history_panel(screen: Any, priors_table: pd.DataFrame) -> pd.DataFrame:
    priors = priors_table.loc[priors_table["identity_ok"]].copy()
    priors["gsis_id"] = priors["gsis_id"].astype(str)
    panel: pd.DataFrame = screen.load_snap_panel()
    panel["team"] = _canonical_team(panel["team"])
    panel = panel.merge(
        priors[
            [
                "gsis_id",
                "draft_year",
                "draft_round",
                "overall",
                "nfl_position",
                "nfl_position_detail",
                "is_pass_rush",
                "surplus_z",
                "prior_channel",
                "draft_team",
            ]
        ],
        on="gsis_id",
        how="inner",
    )
    rookie = panel.loc[panel["season"].eq(panel["draft_year"])].copy()
    rookie = rookie.sort_values(["gsis_id", "week"]).reset_index(drop=True)
    rookie["share"] = rookie["offense_pct"] + rookie["defense_pct"]
    rookie["snaps"] = rookie["offense_snaps"] + rookie["defense_snaps"]
    grouped = rookie.groupby("gsis_id", sort=False)
    rookie["prior_snaps"] = grouped["snaps"].cumsum() - rookie["snaps"]
    rookie["prior_share_sum"] = grouped["share"].cumsum() - rookie["share"]
    rookie["prior_games"] = grouped.cumcount()
    return rookie


def _week1_fallback_for_season(rookie_hist: pd.DataFrame, season: int) -> pd.DataFrame:
    week1 = rookie_hist.loc[rookie_hist["week"].eq(1) & rookie_hist["season"].lt(season)]
    if week1.empty:
        return pd.DataFrame(columns=["draft_round", "nfl_position", "expected_week1_share"])
    return week1.groupby(["draft_round", "nfl_position"], as_index=False).agg(
        expected_week1_share=("share", "mean")
    )


def _with_contribution(rookie: pd.DataFrame, screen: Any) -> pd.DataFrame:
    working = screen.carry_forward_presence(rookie) if not rookie.empty else rookie.copy()
    expected = screen.week1_expected_share(working)
    merged = working.merge(expected, on=["season", "draft_round", "nfl_position"], how="left")
    merged["expected_week1_share"] = merged["expected_week1_share"].fillna(0.0)
    trailing = np.where(
        merged["prior_games"] > 0,
        merged["prior_share_sum"] / merged["prior_games"].replace(0, np.nan),
        np.nan,
    )
    merged["expected_share"] = np.where(
        merged["prior_games"] > 0,
        trailing,
        np.where(merged["week"].eq(1), merged["expected_week1_share"], 0.0),
    )
    merged["expected_share"] = pd.to_numeric(merged["expected_share"], errors="coerce").fillna(0.0)
    merged["decay_weight"] = screen.N0_SNAPS / (
        merged["prior_snaps"].astype(float) + screen.N0_SNAPS
    )
    merged["surplus_z"] = pd.to_numeric(merged["surplus_z"], errors="coerce").fillna(0.0)
    merged["contribution"] = merged["expected_share"] * merged["decay_weight"] * merged["surplus_z"]
    return merged


def _historical_threshold_by_season(screen: Any, rookie_hist: pd.DataFrame) -> dict[int, float]:
    contributed = _with_contribution(rookie_hist, screen)
    aggregated = contributed.groupby(["season", "week", "team"], as_index=False).agg(
        team_surplus=("contribution", "sum")
    )
    features = pd.read_parquet(
        screen.GAME_FEATURES,
        columns=["game_id", "season", "week", "game_type", "home_team", "away_team"],
    )
    games = features.loc[features["game_type"].eq("REG")].copy()
    games["season"] = games["season"].astype(int)
    games["week"] = games["week"].astype(int)
    games["home_team"] = _canonical_team(games["home_team"])
    games["away_team"] = _canonical_team(games["away_team"])
    home = aggregated.rename(columns={"team": "home_team", "team_surplus": "home_surplus"})
    away = aggregated.rename(columns={"team": "away_team", "team_surplus": "away_surplus"})
    games = games.merge(home, on=["season", "week", "home_team"], how="left")
    games = games.merge(away, on=["season", "week", "away_team"], how="left")
    games["home_surplus"] = games["home_surplus"].fillna(0.0)
    games["away_surplus"] = games["away_surplus"].fillna(0.0)
    games["surplus_diff"] = games["home_surplus"] - games["away_surplus"]

    thresholds: dict[int, float] = {}
    for season in sorted(int(s) for s in games["season"].unique()):
        history = games.loc[games["season"].lt(season)]
        if history["season"].nunique() < screen.MIN_PRIOR_SEASONS:
            continue
        thresholds[season] = float(
            np.percentile(history["surplus_diff"].abs(), screen.FLAG_PERCENTILE)
        )
    return thresholds


def _live_team_surplus(
    screen: Any,
    priors_table: pd.DataFrame,
    rookie_hist: pd.DataFrame,
    players_master: pd.DataFrame,
    *,
    season: int,
    week: int,
) -> pd.DataFrame:
    rookies = priors_table.loc[
        priors_table["identity_ok"]
        & priors_table["surplus_z"].notna()
        & priors_table["draft_year"].eq(season)
    ].copy()
    rookies["gsis_id"] = rookies["gsis_id"].astype(str)

    this_season = rookie_hist.loc[rookie_hist["season"].eq(season)].copy()
    if not this_season.empty:
        this_season = screen.carry_forward_presence(this_season)
    this_week = (
        this_season.loc[this_season["week"].eq(week)] if not this_season.empty else this_season
    )

    fallback = _week1_fallback_for_season(rookie_hist, season)

    rows: list[dict[str, Any]] = []
    for _, rookie in rookies.iterrows():
        gsis_id = str(rookie["gsis_id"])
        observed = (
            this_week.loc[this_week["gsis_id"].eq(gsis_id)] if not this_week.empty else this_week
        )
        if not observed.empty:
            record = observed.iloc[0]
            prior_games = float(record["prior_games"])
            prior_snaps = float(record["prior_snaps"])
            prior_share_sum = float(record["prior_share_sum"])
            team = str(record["team"])
            expected_share = prior_share_sum / prior_games if prior_games > 0 else 0.0
        else:
            master_row = players_master.loc[players_master["gsis_id"].eq(gsis_id)]
            team = str(master_row["latest_team"].iloc[0]) if not master_row.empty else ""
            prior_snaps = 0.0
            if week == 1 and team:
                match = fallback.loc[
                    fallback["draft_round"].eq(rookie["draft_round"])
                    & fallback["nfl_position"].eq(rookie["nfl_position"])
                ]
                expected_share = (
                    float(match["expected_week1_share"].iloc[0]) if not match.empty else 0.0
                )
            else:
                expected_share = 0.0
        if not team:
            continue
        decay_weight = screen.N0_SNAPS / (prior_snaps + screen.N0_SNAPS)
        surplus_z = float(rookie["surplus_z"]) if pd.notna(rookie["surplus_z"]) else 0.0
        contribution = expected_share * decay_weight * surplus_z
        rows.append({"team": team, "contribution": contribution})

    if not rows:
        return pd.DataFrame(columns=["team", "team_surplus"])
    frame = pd.DataFrame(rows)
    return frame.groupby("team", as_index=False).agg(team_surplus=("contribution", "sum"))


def rookie_prior_surplus_flags(repo_root: Path, games: pd.DataFrame) -> pd.DataFrame:
    screen = _load_screen_module(repo_root)
    seasons_needed = sorted(int(s) for s in games["season"].unique())
    if seasons_needed:
        screen.LAST_DRAFT_CLASS = max(int(screen.LAST_DRAFT_CLASS), max(seasons_needed))

    table = screen.fit_priors(screen.build_prior_inputs())
    rookie_hist = _rookie_history_panel(screen, table)
    thresholds = _historical_threshold_by_season(screen, rookie_hist)

    players_master = pd.read_parquet(screen.PLAYERS_PARQUET, columns=["gsis_id", "latest_team"])
    players_master["gsis_id"] = players_master["gsis_id"].astype(str)
    players_master["latest_team"] = _canonical_team(players_master["latest_team"].fillna(""))

    frame = games.reset_index(drop=True).copy()
    frame["home_team"] = _canonical_team(frame["home_team"])
    frame["away_team"] = _canonical_team(frame["away_team"])
    frame["home_surplus"] = 0.0
    frame["away_surplus"] = 0.0
    for season in seasons_needed:
        weeks = sorted(int(w) for w in frame.loc[frame["season"].eq(season), "week"].unique())
        for week in weeks:
            mask = frame["season"].eq(season) & frame["week"].eq(week)
            team_surplus = _live_team_surplus(
                screen, table, rookie_hist, players_master, season=season, week=week
            )
            lookup = team_surplus.set_index("team")["team_surplus"].to_dict()
            frame.loc[mask, "home_surplus"] = frame.loc[mask, "home_team"].map(lookup).fillna(0.0)
            frame.loc[mask, "away_surplus"] = frame.loc[mask, "away_team"].map(lookup).fillna(0.0)

    frame["surplus_diff"] = frame["home_surplus"] - frame["away_surplus"]
    frame["threshold"] = frame["season"].map(thresholds).astype(float)
    frame["flag"] = 0
    fires = frame["threshold"].notna()
    frame.loc[fires & frame["surplus_diff"].gt(frame["threshold"]), "flag"] = 1
    frame.loc[fires & frame["surplus_diff"].lt(-frame["threshold"]), "flag"] = -1
    return frame[
        [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_surplus",
            "away_surplus",
            "surplus_diff",
            "threshold",
            "flag",
        ]
    ]


@dataclass(frozen=True)
class TiltFlip:
    game_id: str
    matchup: str
    backed_team: str
    opponent_team: str


@dataclass(frozen=True)
class TiltResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    both_flagged_games: tuple[str, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_rookie_prior_surplus_tilt_overlay(
    predictions: pd.DataFrame,
    repo_root: Path,
    *,
    enabled: bool = True,
) -> TiltResult:

    required = {
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_cover_probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return TiltResult(base, (), (), enabled)

    flags = rookie_prior_surplus_flags(
        repo_root, base[["game_id", "season", "week", "home_team", "away_team"]]
    )
    flags["game_id"] = flags["game_id"].astype(str)
    merged = base.merge(
        flags[["game_id", "flag"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    merged["flag"] = merged["flag"].fillna(0).astype(int)
    merged["home_flagged"] = merged["flag"].eq(1)
    merged["away_flagged"] = merged["flag"].eq(-1)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_pick = merged["home_cover_probability"].ge(0.5)
    unique_flag = merged["home_flagged"].ne(merged["away_flagged"])
    already_on_flagged_side = merged["home_flagged"].where(home_pick, merged["away_flagged"])

    flip_mask = eligible & unique_flag & ~already_on_flagged_side

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for idx in merged.index[flip_mask]:
        row = merged.loc[idx]
        new_home_pick = bool(overlaid.loc[idx, "home_cover_probability"] >= 0.5)
        backed_team = str(row["home_team"] if new_home_pick else row["away_team"])
        opponent_team = str(row["away_team"] if new_home_pick else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                backed_team=backed_team,
                opponent_team=opponent_team,
            )
        )

    both_ids = tuple(
        merged.loc[eligible & merged["home_flagged"] & merged["away_flagged"], "game_id"].astype(
            str
        )
    )
    return TiltResult(overlaid, tuple(flips), both_ids, enabled)


def overlay_disclosure_note(result: TiltResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.opponent_team} -> {flip.backed_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the rookie-starter "
        "value tilt (the team's rookie starters' pre-NFL track record outscores the "
        f"opponent's by more than the frozen threshold). {detail}. See "
        "docs/rookie_priors_on_production.md. Prospective evidence only -- not applied "
        "to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_rookie_prior_surplus_tilt_overlay_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    repo_root: Path,
    now: datetime | None = None,
    forecast_artifact: str | None = None,
    replace_week: bool = False,
) -> dict[str, Any]:

    del data_root
    entry = find_challenger(artifacts_root, CHALLENGER_ID)
    status = str(entry.get("status"))
    if status != ACTIVE_CHALLENGER_STATUS:
        raise ValueError(
            f"Challenger {CHALLENGER_ID!r} is registered as {status!r}; only "
            f"{ACTIVE_CHALLENGER_STATUS} challengers have picks recorded"
        )

    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError(
            "No synchronized active ATS model is available to record overlay decisions from"
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
            f"Challenger {CHALLENGER_ID!r} is registered pinned to configuration "
            f"fingerprint {declared_fingerprint}, but the current active forecast "
            f"{forecast} was produced with {observed_fingerprint}; the active model "
            "changed underneath this overlay -- re-register before recording"
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

    tilt = apply_rookie_prior_surplus_tilt_overlay(card, repo_root)
    tilted_card = tilt.overlaid_predictions

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    pre_kickoff = kickoffs.gt(recorded_at)
    existing = load_challenger_decisions(artifacts_root)
    replaced_rows = 0
    left_post_kickoff = 0
    if replace_week and bool(pre_kickoff.any()):
        existing, replaced_rows, left_post_kickoff = replace_week_rows(
            existing,
            challenger_ledger_path(artifacts_root),
            season=int(card["season"].iloc[0]),
            week=int(card["week"].iloc[0]),
            recorded_at=recorded_at,
            columns=CHALLENGER_DECISION_COLUMNS,
            challenger_id=CHALLENGER_ID,
        )
    mine = existing.loc[existing["challenger_id"].astype(str).eq(CHALLENGER_ID)]
    already = card["game_id"].astype(str).isin(set(mine["game_id"].astype(str)))
    keep = pre_kickoff & ~already
    fresh = tilted_card.loc[keep]

    decisions = pd.DataFrame(
        {
            "recorded_at_utc": recorded_at,
            "challenger_id": CHALLENGER_ID,
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
        "challenger_id": CHALLENGER_ID,
        "season": int(card["season"].iloc[0]),
        "week": int(card["week"].iloc[0]),
        "source_artifact": forecast.name,
        "config_fingerprint": observed_fingerprint,
        "recorded": len(decisions),
        "already_recorded": int(already.sum()),
        "post_kickoff_skipped": int((~pre_kickoff & ~already).sum()),
        "replaced_rows": replaced_rows,
        "left_post_kickoff": left_post_kickoff,
        "ledger_rows": int(ledger_rows),
        "flip_count": tilt.flip_count,
        "flipped_game_ids": [flip.game_id for flip in tilt.flips],
        "both_flagged_games": list(tilt.both_flagged_games),
    }
