from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.best_pick_nomination import dispersion_pool_from_frame, week_dispersion_pool
from nfl_ats.io import atomic_text
from nfl_ats.market_data import load_quote_history, spread_consensus
from nfl_ats.pick_refresh import (
    PICK_LOCK_TIMEZONE,
    RefreshResult,
    original_card,
    pick_deadline,
    served_best_pick,
    sunday_pick_lock,
)
from nfl_ats.publishing import BEST_PICK_MARK

RENOMINATION_ARM = "best_pick_sunday_refresh_s3"

TIE_DECIMALS = 12

BEST_PICK_NOTE_PREFIX = "**Best Pick of the week"


@dataclass(frozen=True)
class SundayRenomination:
    season: int
    week: int
    computed_at_utc: pd.Timestamp
    tuesday_game_id: str
    previous_game_id: str
    game_id: str
    candidate_game_id: str
    served: bool
    reason: str
    statistic: float | None
    n_tied: int
    tie_break: str
    pool_source: str
    pool_reason: str
    pool_day: str
    n_candidates: int
    matchup: str
    previous_matchup: str
    candidate_matchup: str
    played_pick_side: str
    frozen_pick_side: str
    frozen_side_statistic: float | None
    ranking: tuple[dict[str, Any], ...] = ()

    @property
    def moved(self) -> bool:
        return self.served and bool(self.game_id) and self.game_id != self.previous_game_id


def _matchup(plan: RefreshResult, game_id: str) -> str:
    for game in plan.games:
        if game.game_id == game_id:
            return f"{game.away_team} at {game.home_team}"
    return game_id


def _held(
    plan: RefreshResult,
    *,
    reason: str,
    tuesday_game_id: str = "",
    previous_game_id: str = "",
) -> SundayRenomination:
    held = previous_game_id or tuesday_game_id
    return SundayRenomination(
        season=plan.season,
        week=plan.week,
        computed_at_utc=plan.computed_at_utc,
        tuesday_game_id=tuesday_game_id,
        previous_game_id=held,
        game_id=held,
        candidate_game_id="",
        served=False,
        reason=reason,
        statistic=None,
        n_tied=0,
        tie_break="",
        pool_source="",
        pool_reason="",
        pool_day="",
        n_candidates=0,
        matchup=_matchup(plan, held) if held else "",
        previous_matchup=_matchup(plan, held) if held else "",
        candidate_matchup="",
        played_pick_side="",
        frozen_pick_side="",
        frozen_side_statistic=None,
    )


def refresh_day_dispersion(
    data_root: Path, game_ids: list[str], *, instant: pd.Timestamp
) -> pd.DataFrame:

    columns = ["game_id", "spread_std"]
    quotes = load_quote_history(
        data_root / "market" / "raw", since=pd.Timestamp(instant) - pd.Timedelta(days=2)
    )
    if quotes.empty:
        return pd.DataFrame(columns=columns)
    observed = pd.to_datetime(quotes["observed_at_utc"], utc=True)
    commence = pd.to_datetime(quotes["commence_time_utc"], utc=True)
    local_day = observed.dt.tz_convert(PICK_LOCK_TIMEZONE).dt.date
    window = quotes.loc[
        quotes["nflverse_game_id"].astype(str).isin(set(game_ids))
        & observed.le(instant)
        & observed.lt(commence)
        & local_day.eq(instant.tz_convert(PICK_LOCK_TIMEZONE).date())
    ]
    if window.empty:
        return pd.DataFrame(columns=columns)
    consensus = spread_consensus(window)
    if consensus.empty:
        return pd.DataFrame(columns=columns)
    frame = consensus.rename(columns={"nflverse_game_id": "game_id"})[columns].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    return frame.drop_duplicates("game_id")


def renomination_pool(
    data_root: Path, game_ids: list[str], *, instant: pd.Timestamp
) -> tuple[pd.DataFrame, str, str]:

    anchor = pd.DataFrame({"game_id": game_ids})
    try:
        same_day = refresh_day_dispersion(data_root, game_ids, instant=instant)
    except (OSError, ValueError, KeyError):
        same_day = pd.DataFrame(columns=["game_id", "spread_std"])
    if not same_day.empty and bool(same_day["spread_std"].notna().any()):
        pool = dispersion_pool_from_frame(anchor.merge(same_day, on="game_id", how="left"))
        return pool.frame, "refresh_day_cross_book", pool.fallback_reason or ""
    try:
        opener = week_dispersion_pool(
            data_root / "market" / "raw",
            game_ids,
            since=pd.Timestamp(instant) - pd.Timedelta(days=21),
        )
    except (OSError, ValueError, KeyError):
        fallback = anchor.copy()
        fallback["spread_std"] = float("nan")
        fallback["pool_pass"] = True
        return fallback, "all_playable_games", "no_market_data"
    return opener.frame, "tuesday_opener_fallback", opener.fallback_reason or ""


def select_renominee(candidates: pd.DataFrame) -> tuple[str, int, str]:

    table = candidates.copy()
    statistic = pd.to_numeric(table["statistic"], errors="coerce").round(TIE_DECIMALS)
    table = table.loc[statistic.eq(statistic.max())]
    n_tied = len(table)
    if n_tied == 1:
        return str(table.iloc[0]["game_id"]), 1, "none"
    dispersion = pd.to_numeric(table["spread_std"], errors="coerce").round(6)
    if bool(dispersion.notna().any()) and dispersion.nunique(dropna=True) > 1:
        table = table.loc[dispersion.eq(dispersion.min())]
        if len(table) == 1:
            return str(table.iloc[0]["game_id"]), n_tied, "dispersion"
    magnitude = pd.to_numeric(table["decision_home_spread"], errors="coerce").abs().round(6)
    if magnitude.nunique(dropna=True) > 1:
        table = table.loc[magnitude.eq(magnitude.min())]
        if len(table) == 1:
            return str(table.iloc[0]["game_id"]), n_tied, "spread_size"
    kickoff = pd.to_datetime(table["kickoff"], utc=True)
    if kickoff.nunique() > 1:
        table = table.loc[kickoff.eq(kickoff.min())]
        if len(table) == 1:
            return str(table.iloc[0]["game_id"]), n_tied, "kickoff"
    return sorted(str(value) for value in table["game_id"])[0], n_tied, "arbitrary"


def plan_best_pick_renomination(
    artifacts_root: Path, data_root: Path, plan: RefreshResult
) -> SundayRenomination:

    try:
        original = original_card(artifacts_root, season=plan.season, week=plan.week)
    except (OSError, ValueError, KeyError):
        return _held(plan, reason="the Tuesday card could not be read")
    if original.empty or "is_best_pick" not in original.columns:
        return _held(plan, reason="no recorded Tuesday card")
    flagged = original.loc[original["is_best_pick"].fillna(False).astype(bool)]
    if len(flagged) != 1:
        return _held(plan, reason="the Tuesday card does not carry exactly one Best Pick")
    tuesday_game_id = str(flagged.iloc[0]["game_id"])
    previous_game_id = (
        served_best_pick(artifacts_root, season=plan.season, week=plan.week) or tuesday_game_id
    )

    instant = plan.computed_at_utc
    local = instant.tz_convert(PICK_LOCK_TIMEZONE)
    kickoffs = pd.to_datetime(original["kickoff"], utc=True)
    lock = sunday_pick_lock(kickoffs)

    frame = original[["game_id", "kickoff", "decision_home_spread", "pick_side"]].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True)
    frame["deadline"] = frame["kickoff"].map(lambda value: pick_deadline(value, lock))

    held = frame.loc[frame["game_id"].eq(previous_game_id)]
    if held.empty:
        return _held(
            plan,
            reason="the Best Pick is not on this week's recorded card",
            tuesday_game_id=tuesday_game_id,
            previous_game_id=previous_game_id,
        )

    serve_reason = ""
    if local.date() != lock.tz_convert(PICK_LOCK_TIMEZONE).date():
        serve_reason = "the Best Pick is re-nominated on the Sunday pass only"
    elif instant >= lock:
        serve_reason = "this week's picks are already locked"
    elif pd.Timestamp(held.iloc[0]["deadline"]) <= instant:
        serve_reason = "the Best Pick's own game is past its deadline"

    playable = frame.loc[frame["deadline"].gt(instant)].copy()
    if playable.empty:
        return _held(
            plan,
            reason=serve_reason or "no game is still playable",
            tuesday_game_id=tuesday_game_id,
            previous_game_id=previous_game_id,
        )
    refreshed = {
        game.game_id: game
        for game in plan.games
        if game.eligible and instant < min(game.kickoff, game.deadline, lock)
    }
    if not set(playable["game_id"]).issubset(refreshed):
        return _held(
            plan,
            reason=serve_reason or "refreshed probabilities are missing for a playable game",
            tuesday_game_id=tuesday_game_id,
            previous_game_id=previous_game_id,
        )
    probabilities = {
        game_id: float(refreshed[game_id].new_home_cover_probability)
        for game_id in playable["game_id"]
    }
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities.values()):
        return _held(
            plan,
            reason=serve_reason or "a refreshed probability is not usable",
            tuesday_game_id=tuesday_game_id,
            previous_game_id=previous_game_id,
        )
    playable["home_probability"] = playable["game_id"].map(probabilities)
    playable["played_pick_side"] = playable["game_id"].map(
        {game_id: str(refreshed[game_id].new_pick_side) for game_id in playable["game_id"]}
    )
    playable["statistic"] = playable["home_probability"].where(
        playable["played_pick_side"].eq("HOME"), 1.0 - playable["home_probability"]
    )
    playable["tuesday_statistic"] = playable["home_probability"].where(
        playable["pick_side"].astype(str).eq("HOME"), 1.0 - playable["home_probability"]
    )

    pool_frame, pool_source, pool_reason = renomination_pool(
        data_root, playable["game_id"].astype(str).tolist(), instant=instant
    )
    ranked = playable.merge(
        pool_frame[["game_id", "spread_std", "pool_pass"]], on="game_id", how="left"
    )
    candidates = ranked.loc[ranked["pool_pass"].fillna(False).astype(bool)]
    if candidates.empty:
        candidates = ranked
        pool_reason = "empty_filter"
    candidates = candidates.loc[candidates["statistic"].ge(BEST_PICK_MINIMUM_COVER_CHANCE)]
    if candidates.empty:
        return _held(
            plan,
            reason="no eligible game has a played side at or above an even chance to cover",
            tuesday_game_id=tuesday_game_id,
            previous_game_id=previous_game_id,
        )

    candidate_game_id, n_tied, tie_break = select_renominee(candidates)
    chosen = candidates.loc[candidates["game_id"].eq(candidate_game_id)].iloc[0]
    served = not serve_reason
    game_id = candidate_game_id if served else previous_game_id
    ranking = _ranking_rows(plan, ranked, candidate_game_id)
    return SundayRenomination(
        season=plan.season,
        week=plan.week,
        computed_at_utc=instant,
        tuesday_game_id=tuesday_game_id,
        previous_game_id=previous_game_id,
        game_id=game_id,
        candidate_game_id=candidate_game_id,
        served=served,
        reason=serve_reason,
        statistic=float(chosen["statistic"]),
        n_tied=n_tied,
        tie_break=tie_break,
        pool_source=pool_source,
        pool_reason=pool_reason,
        pool_day=str(local.date()),
        n_candidates=len(candidates),
        matchup=_matchup(plan, game_id),
        previous_matchup=_matchup(plan, previous_game_id),
        candidate_matchup=_matchup(plan, candidate_game_id),
        played_pick_side=str(chosen["played_pick_side"]),
        frozen_pick_side=str(chosen["pick_side"]),
        frozen_side_statistic=float(chosen["tuesday_statistic"]),
        ranking=ranking,
    )


BEST_PICK_MINIMUM_COVER_CHANCE = 0.5


def _ranking_rows(
    plan: RefreshResult, ranked: pd.DataFrame, candidate_game_id: str
) -> tuple[dict[str, Any], ...]:
    teams = {game.game_id: (game.away_team, game.home_team) for game in plan.games}
    frame = ranked.copy()
    frame["pool_pass"] = frame["pool_pass"].fillna(False).astype(bool)
    frame["spread_std"] = pd.to_numeric(frame["spread_std"], errors="coerce")
    frame["spread_size"] = pd.to_numeric(frame["decision_home_spread"], errors="coerce").abs()
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True)
    ordered = frame.sort_values(
        ["pool_pass", "statistic", "spread_std", "spread_size", "kickoff", "game_id"],
        ascending=[False, False, True, True, True, True],
        na_position="last",
    )
    rows: list[dict[str, Any]] = []
    rank = 0
    for _, row in ordered.iterrows():
        game_id = str(row["game_id"])
        away, home = teams.get(game_id, ("", ""))
        side = str(row["played_pick_side"])
        eligible = (
            bool(row["pool_pass"]) and float(row["statistic"]) >= BEST_PICK_MINIMUM_COVER_CHANCE
        )
        if eligible:
            rank += 1
        dispersion = row["spread_std"]
        rows.append(
            {
                "rank": rank if eligible else None,
                "game_id": game_id,
                "matchup": f"{away} at {home}" if away else game_id,
                "pick_team": home if side == "HOME" else away,
                "pick_side": side,
                "decision_home_spread": float(row["decision_home_spread"]),
                "cover_probability": float(row["statistic"]),
                "book_dispersion": None if pd.isna(dispersion) else float(dispersion),
                "eligible": eligible,
                "in_book_pool": bool(row["pool_pass"]),
                "is_candidate": game_id == candidate_game_id,
            }
        )
    return tuple(rows)


def renomination_summary(renomination: SundayRenomination) -> dict[str, Any]:

    return {
        "arm": RENOMINATION_ARM,
        "served": renomination.served,
        "moved": renomination.moved,
        "reason": renomination.reason,
        "tuesday_game_id": renomination.tuesday_game_id,
        "previous_game_id": renomination.previous_game_id,
        "game_id": renomination.game_id,
        "candidate_game_id": renomination.candidate_game_id,
        "matchup": renomination.matchup,
        "previous_matchup": renomination.previous_matchup,
        "candidate_matchup": renomination.candidate_matchup,
        "played_pick_side": renomination.played_pick_side,
        "played_side_cover_probability": renomination.statistic,
        "frozen_tuesday_pick_side": renomination.frozen_pick_side,
        "frozen_tuesday_side_cover_probability": renomination.frozen_side_statistic,
        "candidates": renomination.n_candidates,
        "tied_at_top": renomination.n_tied,
        "tie_break": renomination.tie_break,
        "dispersion_pool": renomination.pool_source,
        "dispersion_pool_fallback": renomination.pool_reason,
        "dispersion_pool_day": renomination.pool_day,
    }


def _restar(cell: str, *, star: bool) -> str:
    width = len(cell)
    text = cell.strip()
    if text.startswith(BEST_PICK_MARK.strip()):
        text = text.removeprefix(BEST_PICK_MARK.strip()).strip()
    value = f"{BEST_PICK_MARK}{text}" if star else text
    return f" {value}".ljust(width)


def best_pick_note_line(renomination: SundayRenomination, pick: str, previous_pick: str) -> str:

    tuesday = (
        f" Tuesday's Best Pick was {previous_pick} in {renomination.previous_matchup}."
        if previous_pick
        else ""
    )
    return (
        f"**Best Pick of the week ({BEST_PICK_MARK.strip()}):** {pick} in {renomination.matchup}. "
        "The pool scores one Best Pick per regular-season week. The star moved here on Sunday "
        "morning: of the games that have not kicked off yet, this is the one the model is now "
        f"most sure about at the spread the pool locked on Tuesday.{tuesday}"
    )


def apply_star_to_card(destination: Path, renomination: SundayRenomination) -> dict[str, Any]:

    if not renomination.moved:
        return {"moved": False, "reason": renomination.reason or "the Best Pick did not move"}
    if not destination.is_file():
        raise ValueError(
            f"Cannot move the Best Pick star: no published card at {destination}; "
            "run `nfl-ats publish-predictions` first."
        )
    lines = destination.read_text(encoding="utf-8").split("\n")
    pick = ""
    previous_pick = ""
    starred = False
    for index, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        if len(cells) < 5:
            continue
        matchup = cells[2].strip()
        wants_star = matchup == renomination.matchup
        if cells[3].strip().startswith(BEST_PICK_MARK.strip()) and not wants_star:
            previous_pick = _restar(cells[3], star=False).strip()
        elif not wants_star:
            continue
        cells[3] = _restar(cells[3], star=wants_star)
        if wants_star:
            pick = cells[3].strip().removeprefix(BEST_PICK_MARK.strip()).strip()
            starred = True
        lines[index] = "|".join(cells)
    if not starred:
        raise ValueError(
            f"The re-nominated Best Pick {renomination.matchup} is not on the published card"
        )
    note_index = next(
        (index for index, line in enumerate(lines) if line.startswith(BEST_PICK_NOTE_PREFIX)),
        None,
    )
    if note_index is not None:
        lines[note_index] = best_pick_note_line(renomination, pick, previous_pick)
    atomic_text("\n".join(lines), destination)
    return {
        "moved": True,
        "from": renomination.previous_matchup,
        "to": renomination.matchup,
        "pick": pick,
        "note_rewritten": note_index is not None,
    }
