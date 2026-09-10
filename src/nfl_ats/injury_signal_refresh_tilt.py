from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.pick_refresh import (
    MOVEMENT_GOVERNED_POLICIES,
    RefreshResult,
    original_card,
)

CHALLENGER_ID = "injury_signal_refresh_tilt"

SEVERITY: dict[str, float] = {"Out": 4.0, "Doubtful": 3.0, "Questionable": 2.0, "Probable": 1.0}
SKILL_POSITIONS: frozenset[str] = frozenset({"QB", "RB", "WR", "TE"})
INJURY_NET_THRESHOLD = 2.0
PFT_NET_THRESHOLD = 1.0

SOURCE_OFFICIAL = "official"
SOURCE_PFT_FALLBACK = "pft_fallback"
SOURCE_NONE = "none"

DISAGREEMENT_INJURY_ONLY = "injury_only"
DISAGREEMENT_MOVEMENT_ONLY = "movement_only"
DISAGREEMENT_BOTH_AGREE = "both_agree"
DISAGREEMENT_BOTH_DISAGREE = "both_disagree"
DISAGREEMENT_NEITHER = "neither"

TEAM_NICKNAMES: dict[str, tuple[str, ...]] = {
    "ARI": ("cardinals",),
    "ATL": ("falcons",),
    "BAL": ("ravens",),
    "BUF": ("bills",),
    "CAR": ("panthers",),
    "CHI": ("bears",),
    "CIN": ("bengals",),
    "CLE": ("browns",),
    "DAL": ("cowboys",),
    "DEN": ("broncos",),
    "DET": ("lions",),
    "GB": ("packers",),
    "HOU": ("texans",),
    "IND": ("colts",),
    "JAX": ("jaguars",),
    "KC": ("chiefs",),
    "LA": ("rams",),
    "LAC": ("chargers",),
    "LV": ("raiders",),
    "MIA": ("dolphins",),
    "MIN": ("vikings",),
    "NE": ("patriots",),
    "NO": ("saints",),
    "NYG": ("giants",),
    "NYJ": ("jets",),
    "PHI": ("eagles",),
    "PIT": ("steelers",),
    "SEA": ("seahawks",),
    "SF": ("49ers", "niners"),
    "TB": ("buccaneers", "bucs"),
    "TEN": ("titans",),
    "WAS": ("commanders", "washington", "football team", "redskins"),
}


def own_week_tuesday_noon_utc(kickoff_utc: pd.Series) -> pd.Series:

    kickoff_et = kickoff_utc.dt.tz_convert("US/Eastern")
    days_since_tuesday = (kickoff_et.dt.weekday - 1) % 7
    tuesday_date_et = kickoff_et.dt.normalize() - pd.to_timedelta(days_since_tuesday, unit="D")
    tuesday_noon_et = tuesday_date_et + pd.Timedelta(hours=12)
    return cast(pd.Series, tuesday_noon_et.dt.tz_convert("UTC"))


def _canonical_team(code: str) -> str:
    return TEAM_ABBREVIATION_ALIASES.get(str(code), str(code))


def _latest_official_injuries_fail_open(data_root: Path) -> pd.DataFrame | None:

    from nfl_ats.players import latest_player_snapshot, load_player_snapshot

    try:
        snapshot = latest_player_snapshot(data_root / "players" / "raw")
        injuries, _rosters, _snaps = load_player_snapshot(snapshot)
    except Exception as exc:
        warnings.warn(
            f"{CHALLENGER_ID}: no official injury-report snapshot available, falling back "
            f"to PFT news / no signal ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return injuries


def _latest_pft_index_fail_open(data_root: Path) -> pd.DataFrame | None:

    root = data_root / "raw" / "injury_news"
    try:
        candidates = sorted(path for path in root.glob("*") if (path / "index.parquet").is_file())
        if not candidates:
            raise FileNotFoundError(f"No injury-news snapshot under {root}")
        pft = pd.read_parquet(candidates[-1] / "index.parquet")
    except Exception as exc:
        warnings.warn(
            f"{CHALLENGER_ID}: no PFT injury-news snapshot available, proceeding with no "
            f"fallback signal ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    required = {"lastmod", "injury_relevant", "headline_guess"}
    missing = required.difference(pft.columns)
    if missing:
        warnings.warn(
            f"{CHALLENGER_ID}: PFT injury-news snapshot is missing columns "
            f"{sorted(missing)}, proceeding with no fallback signal",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    pft = pft.loc[pft["injury_relevant"]].copy()
    pft["lastmod"] = pd.to_datetime(pft["lastmod"], utc=True, errors="coerce")
    pft = pft.dropna(subset=["lastmod"])
    pft["headline_norm"] = pft["headline_guess"].fillna("")
    return pft


def _season_has_readable_official_rows(injuries: pd.DataFrame, season: int) -> bool:

    scoped = injuries.loc[injuries["season"].eq(season)]
    if scoped.empty:
        return False
    readable = pd.to_datetime(scoped["date_modified"], utc=True, errors="coerce").notna()
    if "observed_at_is_proxy" in scoped.columns:
        readable &= ~scoped["observed_at_is_proxy"].fillna(True).astype(bool)
    return bool(readable.any())


def _severity_asof(rows: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    eligible = rows.loc[rows["date_modified"] <= cutoff]
    if eligible.empty:
        return pd.DataFrame(columns=["gsis_id", "severity"])
    eligible = eligible.assign(severity=eligible["report_status"].map(SEVERITY).fillna(0.0))
    eligible = eligible.sort_values("date_modified")
    return eligible.drop_duplicates("gsis_id", keep="last")[["gsis_id", "severity"]]


def _official_team_delta(
    injuries: pd.DataFrame,
    *,
    season: int,
    week: int,
    team: str,
    tuesday_noon_utc: pd.Timestamp,
    now: pd.Timestamp,
) -> float:

    scoped = injuries.loc[
        injuries["season"].eq(season)
        & injuries["week"].eq(week)
        & injuries["team"].eq(team)
        & injuries["position"].isin(SKILL_POSITIONS)
    ]
    if scoped.empty:
        return 0.0
    tue = _severity_asof(scoped, tuesday_noon_utc).rename(columns={"severity": "severity_tue"})
    cur = _severity_asof(scoped, now).rename(columns={"severity": "severity_cur"})
    both = tue.merge(cur, on="gsis_id", how="outer")
    both[["severity_tue", "severity_cur"]] = both[["severity_tue", "severity_cur"]].fillna(0.0)
    return float((both["severity_cur"] - both["severity_tue"]).sum())


def _pft_team_hits(pft: pd.DataFrame, team: str, start: pd.Timestamp, end: pd.Timestamp) -> int:
    if end <= start:
        return 0
    window = pft.loc[pft["lastmod"].gt(start) & pft["lastmod"].le(end)]
    if window.empty:
        return 0
    nicknames = TEAM_NICKNAMES.get(team, ())
    if not nicknames:
        return 0
    mask = window["headline_norm"].str.contains("|".join(nicknames), case=False, regex=True)
    return int(mask.sum())


@dataclass(frozen=True)
class InjurySignalReading:
    game_id: str
    picked_team: str
    opponent_team: str
    source: str
    net_score: float
    threshold: float
    fires: bool


def injury_signal_for_game(
    *,
    game_id: str,
    season: int,
    week: int,
    kickoff: pd.Timestamp,
    picked_team: str,
    opponent_team: str,
    now: pd.Timestamp,
    injuries: pd.DataFrame | None,
    pft: pd.DataFrame | None,
) -> InjurySignalReading:

    picked = _canonical_team(picked_team)
    opponent = _canonical_team(opponent_team)
    tuesday_noon = own_week_tuesday_noon_utc(pd.Series([kickoff])).iloc[0]

    if injuries is not None and _season_has_readable_official_rows(injuries, season):
        delta_picked = _official_team_delta(
            injuries,
            season=season,
            week=week,
            team=picked,
            tuesday_noon_utc=tuesday_noon,
            now=now,
        )
        delta_opponent = _official_team_delta(
            injuries,
            season=season,
            week=week,
            team=opponent,
            tuesday_noon_utc=tuesday_noon,
            now=now,
        )
        net = delta_picked - delta_opponent
        return InjurySignalReading(
            game_id=game_id,
            picked_team=picked,
            opponent_team=opponent,
            source=SOURCE_OFFICIAL,
            net_score=net,
            threshold=INJURY_NET_THRESHOLD,
            fires=net >= INJURY_NET_THRESHOLD,
        )

    if pft is not None:
        hits_picked = _pft_team_hits(pft, picked, tuesday_noon, now)
        hits_opponent = _pft_team_hits(pft, opponent, tuesday_noon, now)
        net = float(hits_picked - hits_opponent)
        return InjurySignalReading(
            game_id=game_id,
            picked_team=picked,
            opponent_team=opponent,
            source=SOURCE_PFT_FALLBACK,
            net_score=net,
            threshold=PFT_NET_THRESHOLD,
            fires=net >= PFT_NET_THRESHOLD,
        )

    return InjurySignalReading(
        game_id=game_id,
        picked_team=picked,
        opponent_team=opponent,
        source=SOURCE_NONE,
        net_score=0.0,
        threshold=float("nan"),
        fires=False,
    )


FOLLOW_NEWS_CONFIRMS = "confirms"
FOLLOW_NEWS_CONTRADICTS = "contradicts"
FOLLOW_NEWS_NEITHER = "neither"


@dataclass(frozen=True)
class FollowNewsReading:
    game_id: str
    source: str
    net_toward_market: float
    threshold: float
    moved_toward_team: str
    moved_against_team: str
    verdict: str

    @property
    def confirms(self) -> bool:
        return self.verdict == FOLLOW_NEWS_CONFIRMS

    @property
    def contradicts(self) -> bool:
        return self.verdict == FOLLOW_NEWS_CONTRADICTS


def load_news_sources(data_root: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:

    return _latest_official_injuries_fail_open(data_root), _latest_pft_index_fail_open(data_root)


def _news_team_delta(
    injuries: pd.DataFrame,
    *,
    season: int,
    week: int,
    team: str,
    tuesday_noon_utc: pd.Timestamp,
    end: pd.Timestamp,
) -> float:

    season_rows = injuries.loc[
        injuries["season"].eq(season)
        & injuries["team"].eq(team)
        & injuries["position"].isin(SKILL_POSITIONS)
    ]
    if season_rows.empty:
        return 0.0
    week_rows = season_rows.loc[season_rows["week"].eq(week)]
    if week_rows.empty:
        return 0.0
    modified = pd.to_datetime(week_rows["date_modified"], utc=True, errors="coerce")
    filed = set(week_rows.loc[modified.gt(tuesday_noon_utc) & modified.le(end), "gsis_id"])
    if not filed:
        return 0.0
    prior = _severity_asof(season_rows, tuesday_noon_utc).rename(columns={"severity": "prior"})
    final = _severity_asof(week_rows, end).rename(columns={"severity": "final"})
    both = prior.merge(final, on="gsis_id", how="outer")
    both[["prior", "final"]] = both[["prior", "final"]].fillna(0.0)
    both = both.loc[both["gsis_id"].isin(filed)]
    return float((both["final"] - both["prior"]).sum())


def follow_news_for_game(
    *,
    game_id: str,
    season: int,
    week: int,
    kickoff: pd.Timestamp,
    home_team: str,
    away_team: str,
    leader_median_net_move: float,
    now: pd.Timestamp,
    injuries: pd.DataFrame | None,
    pft: pd.DataFrame | None,
) -> FollowNewsReading:

    home = _canonical_team(home_team)
    away = _canonical_team(away_team)
    move = float(leader_median_net_move)
    if move == 0.0 or not pd.notna(move):
        return FollowNewsReading(
            game_id=game_id,
            source=SOURCE_NONE,
            net_toward_market=0.0,
            threshold=float("nan"),
            moved_toward_team="",
            moved_against_team="",
            verdict=FOLLOW_NEWS_NEITHER,
        )
    toward, against = (home, away) if move > 0.0 else (away, home)
    tuesday_noon = own_week_tuesday_noon_utc(pd.Series([kickoff])).iloc[0]

    if injuries is not None and _season_has_readable_official_rows(injuries, season):
        source = SOURCE_OFFICIAL
        threshold = INJURY_NET_THRESHOLD
        net = _news_team_delta(
            injuries,
            season=season,
            week=week,
            team=against,
            tuesday_noon_utc=tuesday_noon,
            end=now,
        ) - _news_team_delta(
            injuries,
            season=season,
            week=week,
            team=toward,
            tuesday_noon_utc=tuesday_noon,
            end=now,
        )
    elif pft is not None:
        source = SOURCE_PFT_FALLBACK
        threshold = PFT_NET_THRESHOLD
        net = float(
            _pft_team_hits(pft, against, tuesday_noon, now)
            - _pft_team_hits(pft, toward, tuesday_noon, now)
        )
    else:
        return FollowNewsReading(
            game_id=game_id,
            source=SOURCE_NONE,
            net_toward_market=0.0,
            threshold=float("nan"),
            moved_toward_team=toward,
            moved_against_team=against,
            verdict=FOLLOW_NEWS_NEITHER,
        )

    if net >= threshold:
        verdict = FOLLOW_NEWS_CONFIRMS
    elif net <= -threshold:
        verdict = FOLLOW_NEWS_CONTRADICTS
    else:
        verdict = FOLLOW_NEWS_NEITHER
    return FollowNewsReading(
        game_id=game_id,
        source=source,
        net_toward_market=net,
        threshold=threshold,
        moved_toward_team=toward,
        moved_against_team=against,
        verdict=verdict,
    )


def classify_disagreement(
    *,
    injury_fires: bool,
    injury_tilt_pick_side: str,
    movement_policy: str,
    movement_pick_side: str,
) -> str:

    movement_fires = movement_policy in MOVEMENT_GOVERNED_POLICIES
    if injury_fires and not movement_fires:
        return DISAGREEMENT_INJURY_ONLY
    if injury_fires and movement_fires:
        return (
            DISAGREEMENT_BOTH_AGREE
            if injury_tilt_pick_side == movement_pick_side
            else DISAGREEMENT_BOTH_DISAGREE
        )
    if not injury_fires and movement_fires:
        return DISAGREEMENT_MOVEMENT_ONLY
    return DISAGREEMENT_NEITHER


INJURY_SIGNAL_LEDGER_COLUMNS: tuple[str, ...] = (
    "revision_recorded_at_utc",
    "refresh_run_id",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "decision_home_spread",
    "hold_pick_side",
    "injury_tilt_pick_side",
    "injury_signal_fires",
    "injury_signal_source",
    "injury_signal_net_score",
    "injury_signal_threshold",
    "movement_policy",
    "movement_delta",
    "movement_pick_side",
    "played_pick_side",
    "disagreement_type",
    "model_id",
    "feature_table_sha256",
)


def injury_signal_ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "injury_signal_refresh_decisions.parquet"


def load_injury_signal_decisions(artifacts_root: Path) -> pd.DataFrame:

    path = injury_signal_ledger_path(artifacts_root)
    if not path.is_file():
        return pd.DataFrame(columns=list(INJURY_SIGNAL_LEDGER_COLUMNS))
    ledger = pd.read_parquet(path)
    missing = sorted(set(INJURY_SIGNAL_LEDGER_COLUMNS).difference(ledger.columns))
    if missing:
        raise DataContractError(
            f"Injury-signal refresh ledger is missing columns: {', '.join(missing)}"
        )
    return ledger[list(INJURY_SIGNAL_LEDGER_COLUMNS)]


def build_injury_signal_rows(plan: RefreshResult, *, data_root: Path) -> pd.DataFrame:

    eligible_games = [game for game in plan.games if game.eligible]
    if not eligible_games:
        return pd.DataFrame(columns=list(INJURY_SIGNAL_LEDGER_COLUMNS))

    injuries = _latest_official_injuries_fail_open(data_root)
    pft = _latest_pft_index_fail_open(data_root)

    rows: list[dict[str, Any]] = []
    for game in eligible_games:
        hold_side = game.model_only_pick_side
        picked_team = game.home_team if hold_side == "HOME" else game.away_team
        opponent_team = game.away_team if hold_side == "HOME" else game.home_team

        reading = injury_signal_for_game(
            game_id=game.game_id,
            season=plan.season,
            week=plan.week,
            kickoff=game.kickoff,
            picked_team=picked_team,
            opponent_team=opponent_team,
            now=plan.computed_at_utc,
            injuries=injuries,
            pft=pft,
        )
        flip_side = "AWAY" if hold_side == "HOME" else "HOME"
        injury_tilt_side = flip_side if reading.fires else hold_side

        disagreement = classify_disagreement(
            injury_fires=reading.fires,
            injury_tilt_pick_side=injury_tilt_side,
            movement_policy=game.movement_policy,
            movement_pick_side=game.movement_pick_side,
        )

        rows.append(
            {
                "revision_recorded_at_utc": plan.computed_at_utc,
                "refresh_run_id": plan.refresh_run_id,
                "season": plan.season,
                "week": plan.week,
                "game_id": game.game_id,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "kickoff": game.kickoff,
                "decision_home_spread": game.decision_home_spread,
                "hold_pick_side": hold_side,
                "injury_tilt_pick_side": injury_tilt_side,
                "injury_signal_fires": bool(reading.fires),
                "injury_signal_source": reading.source,
                "injury_signal_net_score": reading.net_score,
                "injury_signal_threshold": reading.threshold,
                "movement_policy": game.movement_policy,
                "movement_delta": game.movement_delta,
                "movement_pick_side": game.movement_pick_side,
                "played_pick_side": game.new_pick_side,
                "disagreement_type": disagreement,
                "model_id": plan.model_id,
                "feature_table_sha256": plan.feature_table_sha256,
            }
        )
    return pd.DataFrame(rows, columns=list(INJURY_SIGNAL_LEDGER_COLUMNS))


def record_injury_signal_refresh_tilt(
    artifacts_root: Path,
    data_root: Path,
    plan: RefreshResult,
    *,
    record_decisions: bool = False,
) -> dict[str, Any]:

    if not record_decisions:
        return {
            "recorded": 0,
            "skipped": True,
            "reason": (
                "pass --record-decisions to append this pass's injury-signal reading to "
                "the injury-signal refresh ledger"
            ),
        }

    original = original_card(artifacts_root, season=plan.season, week=plan.week)
    refuse_if_outside_recording_lock_window(
        original["kickoff"], plan.computed_at_utc, ledger="injury-signal-refresh"
    )

    rows = build_injury_signal_rows(plan, data_root=data_root)
    existing = load_injury_signal_decisions(artifacts_root)
    if rows.empty:
        return {
            "recorded": 0,
            "ledger_rows": len(existing),
            "reason": "no eligible games in this refresh pass",
        }

    combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows
    atomic_parquet(
        combined[list(INJURY_SIGNAL_LEDGER_COLUMNS)], injury_signal_ledger_path(artifacts_root)
    )

    fired = rows.loc[rows["injury_signal_fires"], "game_id"].tolist()
    disagreements = rows.loc[
        rows["disagreement_type"].isin((DISAGREEMENT_INJURY_ONLY, DISAGREEMENT_BOTH_DISAGREE)),
        "game_id",
    ].tolist()
    return {
        "recorded": len(rows),
        "ledger_rows": len(combined),
        "games_considered": len(rows),
        "injury_signal_fired_game_ids": fired,
        "disagreement_game_ids": disagreements,
        "source_counts": rows["injury_signal_source"].value_counts().to_dict(),
    }
