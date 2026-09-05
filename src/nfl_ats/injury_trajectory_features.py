"""Real-timestamp injury trajectories for deadline-bounded refresh screens."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nfl_ats.nfl_week import pool_decision_cutoff, week_cycle_sunday
from nfl_ats.players import canonicalize_injuries

LEADS = ("LEAD-08", "LEAD-09", "LEAD-10", "LEAD-11")
KEY = ["season", "week", "team", "gsis_id", "date_modified"]
REASONS = (
    "report_primary_injury",
    "report_secondary_injury",
    "practice_primary_injury",
    "practice_secondary_injury",
)


def prepare_revisions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exclude proxies before canonicalization; preserve only exact-key text."""
    rows = raw.loc[raw["game_type"].eq("REG") & raw["season"].between(2022, 2025)].copy()
    rows["date_modified"] = pd.to_datetime(rows["date_modified"], utc=True, errors="coerce")
    proxy = rows.get("observed_at_is_proxy", pd.Series(False, index=rows.index)).fillna(True)
    proxy = proxy.astype(bool) | rows.get(
        "observed_at_basis", pd.Series("date_modified", index=rows.index)
    ).eq("week_proxy")
    coverage: dict[str, Any] = {}
    for season in range(2022, 2026):
        mask = rows["season"].eq(season)
        coverage[str(season)] = {
            "raw_rows": int(mask.sum()),
            "proxy_excluded": int((mask & proxy).sum()),
            "missing_timestamp_excluded": int((mask & ~proxy & rows.date_modified.isna()).sum()),
            "real_timestamp_rows": int((mask & ~proxy & rows.date_modified.notna()).sum()),
        }
    rows = rows.loc[~proxy & rows.date_modified.notna()].copy()
    present = [c for c in REASONS if c in rows]
    rows["reason"] = rows[present].fillna("").astype(str).agg(" ".join, axis=1) if present else ""
    rows["reason_schema_present"] = bool(present)
    extras = rows[[*KEY, "reason", "reason_schema_present"]].drop_duplicates()
    if extras.duplicated(KEY).any():
        raise ValueError("Conflicting injury text at an identical revision key")
    result = canonicalize_injuries(rows).merge(extras, on=KEY, validate="one_to_one")
    result["observed"] = result["date_modified"]
    result["weekday"] = result["observed"].dt.tz_convert("America/New_York").dt.weekday
    result["practice_rank"] = (
        result["practice_status"]
        .str.lower()
        .map(
            {
                "did not participate in practice": 0,
                "limited participation in practice": 1,
                "full participation in practice": 2,
                "dnp": 0,
                "lp": 1,
                "fp": 2,
            }
        )
    )
    return result, coverage


def build_flags(
    games: pd.DataFrame,
    revisions: pd.DataFrame,
    *,
    birth_dates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per game/team; unavailable observations never manufacture a flag.

    ``revisions`` must be the output of prepare_revisions. All fields, including
    report designation, are filtered before any latest-row operation. Static
    birth dates are optional, uniquely keyed by gsis_id (never career age).
    """
    birthdays: dict[str, Any] = {}
    if birth_dates is not None:
        if birth_dates.gsis_id.duplicated().any():
            raise ValueError("Birth dates must be uniquely keyed by gsis_id")
        birthdays = dict(
            zip(birth_dates.gsis_id, pd.to_datetime(birth_dates.birth_date), strict=True)
        )
    groups = dict(iter(revisions.groupby(["season", "week", "team"])))
    output: list[dict[str, Any]] = []
    for game in games.sort_values(["kickoff", "game_id"]).itertuples(index=False):
        kickoff = pd.Timestamp(str(game.kickoff))
        cutoff = pd.Timestamp(pool_decision_cutoff(kickoff.to_pydatetime()))
        sunday = pd.Timestamp(week_cycle_sunday(kickoff.tz_convert("America/New_York").date()))
        start = (sunday - pd.Timedelta(days=5)).tz_localize("America/New_York").tz_convert("UTC")
        for team in (game.home_team, game.away_team):
            all_rows = groups.get((game.season, game.week, team), revisions.iloc[:0])
            visible = all_rows.loc[all_rows.observed.ge(start) & all_rows.observed.lt(cutoff)]
            visible = visible.sort_values("observed")
            friday = visible.loc[visible.weekday.eq(4)].drop_duplicates("gsis_id", keep="last")
            early = visible.loc[visible.weekday.isin([2, 3])].drop_duplicates(
                "gsis_id", keep="last"
            )
            paired = (
                early[["gsis_id", "practice_rank"]]
                .merge(
                    friday[["gsis_id", "practice_rank"]],
                    on="gsis_id",
                    suffixes=("_early", "_friday"),
                )
                .dropna()
            )
            change = paired.practice_rank_early - paired.practice_rank_friday
            negative = int((change.gt(0).astype(int) - change.lt(0).astype(int)).sum())
            latest = visible.drop_duplicates("gsis_id", keep="last").set_index("gsis_id")
            q_status = (
                friday.gsis_id.map(latest.report_status)
                .astype("string")
                .fillna("")
                .str.lower()
                .eq("questionable")
            )
            q_dnp = friday.position.eq("QB") & friday.practice_rank.eq(0) & q_status
            illness = friday.reason.str.contains(r"illness|\bflu\b|sick", case=False, regex=True)
            wed = visible.loc[visible.weekday.eq(2)].drop_duplicates("gsis_id", keep="last")
            rest = wed.loc[wed.practice_rank.eq(0) & wed.reason.str.contains("rest", case=False)]
            eligible = []
            for player in rest.gsis_id:
                birthday = birthdays.get(player)
                if birthday is not None and pd.notna(birthday):
                    local_date = kickoff.tz_convert("America/New_York").date()
                    age = (
                        local_date.year
                        - birthday.year
                        - ((local_date.month, local_date.day) < (birthday.month, birthday.day))
                    )
                    eligible.append(age >= 30)
            row: dict[str, Any] = {
                "game_id": game.game_id,
                "season": game.season,
                "week": game.week,
                "team": team,
                "cutoff": cutoff,
                "visible_rows": len(visible),
                "after_cutoff_excluded": int(all_rows.observed.ge(cutoff).sum()),
                "friday_players": len(friday),
                "paired_trajectory_players": len(paired),
                "net_deterioration": negative,
                "illness_players": int(illness.sum()),
                "rest_players_missing_age": len(rest) - len(eligible),
                "LEAD-08": negative > 0,
                "LEAD-08_covered": len(paired) > 0,
                "LEAD-09": bool(q_dnp.any()),
                "LEAD-09_covered": len(friday) > 0,
                "LEAD-10": int(illness.sum()) >= 3,
                "LEAD-10_covered": bool(friday.reason_schema_present.any()),
                "LEAD-11": any(eligible),
                "LEAD-11_covered": bool(len(wed) and len(birthdays) and len(rest) == len(eligible)),
            }
            output.append(row)
    return pd.DataFrame(output)


def split_half_reliability(flags: pd.DataFrame, lead: str) -> dict[str, Any]:
    covered = flags.loc[flags[f"{lead}_covered"]].copy()
    covered["half"] = covered.week % 2
    rates = covered.groupby(["season", "team", "half"])[lead].mean().unstack("half")
    if not {0, 1}.issubset(rates.columns):
        return {"correlation": None, "paired_team_seasons": 0, "status": "missing_halves"}
    rates = rates.dropna()
    if len(rates) < 2 or rates[0].nunique() < 2 or rates[1].nunique() < 2:
        return {"correlation": None, "paired_team_seasons": len(rates), "status": "undefined"}
    return {
        "correlation": float(rates[0].corr(rates[1])),
        "paired_team_seasons": len(rates),
        "status": "measured",
    }
