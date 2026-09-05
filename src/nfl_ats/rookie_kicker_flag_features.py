"""Two pregame flags stacked on PRODUCTION (``docs/schedule_flag_battery.md``
"Wave 8"): LEAD-24 stage 2 (rookie-wall dependence fade, weeks 12-17) and
LEAD-16 (midweek kicker change -> take the underdog).

**LEAD-24 stage 2** reuses, never rebuilds, lane P's own Stage-1 builders in
:mod:`nfl_ats.rookie_wall` (``docs/rookie_wall.md``): the trailing 4-
completed-game top-50-pick-rookie snap-share metric
(:func:`nfl_ats.rookie_wall.team_week_dependence_shares` /
:func:`nfl_ats.rookie_wall.trailing_dependence_feature`), its cross-
sectional top-quintile-and-week->=12 gate
(:func:`nfl_ats.rookie_wall.late_season_high_dependence_flag`, already built
with ``percentile=0.80`` -- the top quintile -- and ``late_week_min=12``,
unchanged here), and the position-group panel it is built on
(:func:`nfl_ats.rookie_wall.build_rookie_wall_panel`). This module only adds
the per-GAME signed encoding: which side (home/away) carries the dependent
team, at the season-week's own cross-sectional threshold.

**LEAD-16** reuses, never rebuilds, lane Q/Z's own clause-anchored,
confirmed-vs-speculative headline discipline in
:mod:`nfl_ats.transaction_flag_features` (PFR transaction-wire loader,
token-anchored player-name resolution, month-precision leakage guard) and
:mod:`nfl_ats.schedule_flag_features` (the Tuesday-opener consensus spread
store and the additive-merge helper every sibling on-production candidate
already uses).

Closing-grounds taxonomy (binding, restated here because this module reports
numbers a later session may be tempted to adjudicate): an interval or CI
that contains zero is NEVER grounds to reject, fail, or close a line of
work. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close
anything: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on
the wrong side of zero) or a measured ZERO split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is ``unresolved_below_power``. Nothing in this module
classifies itself.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.constants import (
    KICKER_CHANGE_UNDERDOG_ON_PRODUCTION_FEATURE_COLUMNS,
    ROOKIE_WALL_DEPENDENCE_ON_PRODUCTION_FEATURE_COLUMNS,
)
from nfl_ats.data import DataContractError
from nfl_ats.rookie_wall import (
    DEFAULT_COMBINE_RAW_ROOT,
    DEFAULT_PBP_RAW_ROOT,
    DEFAULT_PLAYERS_RAW_ROOT,
    DEFAULT_PLAYERS_VALUES_RAW_ROOT,
    build_rookie_wall_panel,
    late_season_high_dependence_flag,
    load_rookie_wall_inputs,
    team_week_dependence_shares,
    trailing_dependence_feature,
)
from nfl_ats.schedule_flag_features import _attach, default_opener_lines, default_schedule
from nfl_ats.transaction_flag_features import (
    _month_end_timestamp,
    default_snap_counts,
    default_transactions_index,
    distinct_player_slugs,
    find_player_in_segment,
)
from nfl_ats.transaction_wire_features import match_transaction_teams

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The one new column each candidate profile adds. Frozen names.
ROOKIE_WALL_DEPENDENCE_COLUMN = ROOKIE_WALL_DEPENDENCE_ON_PRODUCTION_FEATURE_COLUMNS[0]
KICKER_CHANGE_COLUMN = KICKER_CHANGE_UNDERDOG_ON_PRODUCTION_FEATURE_COLUMNS[0]

# ---------------------------------------------------------------------------
# LEAD-24 stage 2: rookie-wall dependence fade, weeks 12-17
# ---------------------------------------------------------------------------

_ROOKIE_WALL_DEP_REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
}


def rookie_wall_dependence_table(
    repo_root: Path | None = None,
    *,
    as_of_season: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per (team, season, week): the pregame-safe ``late_season_high_dependence``
    boolean, reused verbatim from LEAD-24 Stage 1
    (:mod:`nfl_ats.rookie_wall`) -- trailing 4-completed-game top-50-pick-
    rookie offense+defense snap share, cross-sectional top-quintile
    (``percentile=0.80``) threshold within the same season/week, AND
    ``week >= 12``. Returns ``(table, diagnostics)``; ``table`` carries
    exactly ``team``, ``season``, ``week``, ``late_season_high_dependence``.
    """

    root = repo_root or REPO_ROOT
    players_raw_root = root / "data" / "players" / "raw"
    players_values_raw_root = root / "data" / "players" / "values" / "raw"
    pbp_raw_root = root / "data" / "pbp" / "raw"
    combine_raw_root = root / "data" / "raw" / "combine"
    if repo_root is None:
        players_raw_root = DEFAULT_PLAYERS_RAW_ROOT
        players_values_raw_root = DEFAULT_PLAYERS_VALUES_RAW_ROOT
        pbp_raw_root = DEFAULT_PBP_RAW_ROOT
        combine_raw_root = DEFAULT_COMBINE_RAW_ROOT

    raw_snaps, raw_rosters, raw_stats, pbp_frames, raw_combine, snapshot_ids = (
        load_rookie_wall_inputs(
            players_raw_root,
            players_values_raw_root,
            pbp_raw_root,
            combine_raw_root,
            as_of_season=as_of_season,
        )
    )
    panel, panel_diagnostics = build_rookie_wall_panel(
        raw_snaps, raw_rosters, raw_stats, pbp_frames, raw_combine, as_of_season=as_of_season
    )
    shares = team_week_dependence_shares(panel)
    trailing = trailing_dependence_feature(shares)
    flagged = late_season_high_dependence_flag(trailing)

    diagnostics = {
        "snapshot_ids": snapshot_ids,
        "panel": panel_diagnostics,
        "n_team_weeks": len(flagged),
    }
    table = flagged.loc[:, ["team", "season", "week", "late_season_high_dependence"]].copy()
    return table, diagnostics


def derive_rookie_wall_dependence_fade_features(
    schedule: pd.DataFrame, dependence_table: pd.DataFrame
) -> pd.DataFrame:
    """Return ``(game_id, rookie_wall_dependence_fade_flag)`` for every game.

    ``+1`` when the AWAY team's ``late_season_high_dependence`` is ``True``
    (top-quintile trailing rookie-snap dependence AND week >= 12) and the
    HOME team's is not -- favouring HOME, the predeclared FADE direction
    applied to whichever side carries the dependent team; ``-1`` when the
    reverse; ``0`` if both, neither, or a side has no resolved dependence
    row for that season/week (never silently treated as dependent).
    """

    missing = sorted(
        {"team", "season", "week", "late_season_high_dependence"}.difference(
            dependence_table.columns
        )
    )
    if missing:
        raise DataContractError(f"dependence table is missing columns: {', '.join(missing)}")
    missing_sched = sorted(_ROOKIE_WALL_DEP_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing_sched:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing_sched)}")

    base = schedule.loc[:, ["game_id", "season", "week", "home_team", "away_team"]].copy()
    base["game_id"] = base["game_id"].astype(str)
    base["season"] = pd.to_numeric(base["season"], errors="raise").astype(int)
    base["week"] = pd.to_numeric(base["week"], errors="raise").astype(int)

    dep = dependence_table.loc[:, ["team", "season", "week", "late_season_high_dependence"]].copy()
    dep["season"] = pd.to_numeric(dep["season"], errors="raise").astype(int)
    dep["week"] = pd.to_numeric(dep["week"], errors="raise").astype(int)
    dep["late_season_high_dependence"] = (
        dep["late_season_high_dependence"].fillna(False).astype(bool)
    )
    dep = dep.drop_duplicates(["team", "season", "week"])

    home_dep = (
        base.merge(
            dep.rename(
                columns={"team": "home_team", "late_season_high_dependence": "home_dependent"}
            ),
            on=["season", "week", "home_team"],
            how="left",
            validate="many_to_one",
        )["home_dependent"]
        .fillna(False)
        .astype(bool)
    )
    away_dep = (
        base.merge(
            dep.rename(
                columns={"team": "away_team", "late_season_high_dependence": "away_dependent"}
            ),
            on=["season", "week", "away_team"],
            how="left",
            validate="many_to_one",
        )["away_dependent"]
        .fillna(False)
        .astype(bool)
    )

    flag = np.where(
        away_dep.to_numpy() & ~home_dep.to_numpy(),
        1.0,
        np.where(home_dep.to_numpy() & ~away_dep.to_numpy(), -1.0, 0.0),
    )
    return pd.DataFrame({"game_id": base["game_id"], ROOKIE_WALL_DEPENDENCE_COLUMN: flag})


def attach_rookie_wall_dependence_fade_features(
    features: pd.DataFrame,
    *,
    repo_root: Path | None = None,
    schedule: pd.DataFrame | None = None,
    dependence_table: pd.DataFrame | None = None,
    as_of_season: int | None = None,
) -> pd.DataFrame:
    """Additively join ``rookie_wall_dependence_fade_flag`` onto ``features``.

    ``dependence_table``, when given (tests, or a precomputed cache for a
    real on-production run -- the underlying panel build is expensive and
    identical across ``--mode`` invocations), is used directly instead of
    rebuilding the LEAD-24 Stage 1 panel from raw snapshots.
    """

    resolved_table = dependence_table
    if resolved_table is None:
        resolved_table, _ = rookie_wall_dependence_table(repo_root, as_of_season=as_of_season)

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        return derive_rookie_wall_dependence_fade_features(sched, resolved_table)

    return _attach(features, schedule, _derive, (ROOKIE_WALL_DEPENDENCE_COLUMN,))


# ---------------------------------------------------------------------------
# LEAD-16: midweek kicker change -> take the underdog
# ---------------------------------------------------------------------------

KICKER_POSITION = "K"
#: A signing/waiver-claim/practice-squad-elevation/IR-activation event only
#: -- release/waive/trade/suspension categories never represent a team
#: ACQUIRING a kicker, so they are excluded from this population regardless
#: of headline wording (a genuinely two-sided "swap" slug, e.g.
#: "saints-to-swap-kickers-by-signing-cade-york-waiving-k-blake-grupe",
#: still classifies "signing" under
#: ``nfl_ats.transaction_wire_features.classify_transaction_slug``'s own
#: priority order -- trade is checked after signing there but the SIGN verb
#: is what this construct keys on, matching the roadmap's own "a placekicker
#: signing/activation" framing literally).
KICKER_ACQUIRE_CATEGORIES = frozenset(
    {"signing", "waiver_claim", "practice_squad_elevation", "ir_activation"}
)
#: Only the SINGLE game immediately following a confirmed kicker-acquisition
#: event qualifies -- the roadmap's own mechanism ("week-of kicker swaps
#: inject PAT/FG variance the Tuesday line cannot contain") is a one-week
#: disruption, unlike LEAD-14's multi-game suspension-return rust window.
KICKER_CHANGE_GAMES = 1

#: Confirmed (present/past-tense) acquisition-direction verb, hyphen-token-
#: anchored on both sides (the same anchoring discipline as HOLDOUT_END_RE/
#: ACQUISITION_RE/REINSTATED_RE in ``nfl_ats.transaction_flag_features``),
#: covering: sign/signs/signed/signing, re-sign/re-signs/re-signed
#: (resign/resigns/resigned without the hyphen), claim/claims/claimed,
#: elevate*, activate*.
KICKER_ACQUIRE_RE = re.compile(
    r"(?:^|-)(?:signs?|signed|signing|re-signs?|re-signed|resigns?|resigned|"
    r"claims?|claimed|elevat\w*|activat\w*)(?:-|$)"
)
#: Excludes NEGATED or SPECULATIVE signing language -- measured against the
#: real corpus (31 slugs mentioning "kicker" literally, 2026-09-05 snapshot):
#: "cowboys-wont-sign-kicker-this-week" and "cowboys-not-signing-kicker" are
#: explicit negations; "lions-expected-to-sign-ufl-kicker-jake-bates" is a
#: prediction, not a confirmation (same class HOLDOUT_END_RE already
#: excludes for "expected-to-report-to-camp"); "giants-ben-mcadoo-on-
#: signing-another-kicker-never-say-never" is a QUOTE about a hypothetical,
#: matched via the general "-on-signing-" pattern (a person speaking ABOUT
#: signing, not a team actually doing it) plus the slug's own unique
#: "never-say-never" marker. The remaining alternatives (could/would/might/
#: looking-to/in-talks-to/interested-in/hopes-to/hoping-to/wants-to) are NOT
#: observed in this specific 31-slug sample -- they are a disclosed,
#: precautionary generalization from ``ACQUISITION_RE``'s own
#: ``SPECULATIVE_ACQUISITION_RE`` sibling pattern for the "acquire" verb,
#: applied here to "sign" for the same reason. Claim/elevate/activate have
#: no analogous speculative form excluded here -- none was observed or is a
#: precedented risk for those smaller, near-always-retrospective categories
#: (a disclosed simplification, not an oversight).
KICKER_ACQUIRE_SPECULATIVE_RE = re.compile(
    r"wont-sign|won-t-sign|not-sign|not-signing|expected-to-sign|expected-to-target|"
    r"hopes-to-sign|hoping-to-sign|wants-to-sign|could-sign|would-sign|might-sign|"
    r"interested-in-sign|looking-to-sign|in-talks-to-sign|-on-signing-|never-say-never"
)

_KICKER_REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
}


def kicker_player_slugs(snap_counts: pd.DataFrame) -> pd.DataFrame:
    """:func:`nfl_ats.transaction_flag_features.distinct_player_slugs`
    restricted to players who have ever appeared in ``snap_counts`` at
    position :data:`KICKER_POSITION` -- the name universe
    :func:`_kicker_change_events` resolves against, so a non-kicker with a
    similar name can never match."""

    kicker_rows = snap_counts.loc[snap_counts["position"] == KICKER_POSITION]
    return distinct_player_slugs(kicker_rows)


def confirmed_kicker_change_transactions(transactions_index: pd.DataFrame) -> pd.DataFrame:
    """Every transaction-wire row using confirmed (not negated or
    speculative) sign/claim/elevate/activate language, restricted to
    :data:`KICKER_ACQUIRE_CATEGORIES`."""

    required = {"slug", "category"}
    missing = sorted(required.difference(transactions_index.columns))
    if missing:
        raise DataContractError(f"transactions index is missing columns: {', '.join(missing)}")

    slug = transactions_index["slug"].astype(str)
    has_verb = slug.str.contains(KICKER_ACQUIRE_RE)
    speculative = slug.str.contains(KICKER_ACQUIRE_SPECULATIVE_RE)
    in_category = transactions_index["category"].isin(KICKER_ACQUIRE_CATEGORIES)
    return transactions_index.loc[has_verb & ~speculative & in_category].copy()


def _kicker_change_events(
    transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> pd.DataFrame:
    """One row per resolvable kicker-change event: the acquiring team, and
    the calendar (year, month) it was reported. A resolution that fails at
    any step (zero or more-than-one team, no kicker-name match, or the
    matched (player, team) pair never actually appearing in ``snap_counts``
    at position K) drops the row -- never guessed."""

    rows = confirmed_kicker_change_transactions(transactions_index)
    player_slugs = kicker_player_slugs(snap_counts)

    records: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        if pd.isna(row["url_year"]) or pd.isna(row["url_month"]):
            continue
        teams = match_transaction_teams(str(row["slug"]))
        if len(teams) != 1:
            continue
        team = next(iter(teams))
        player = find_player_in_segment(str(row["slug"]), player_slugs)
        if player is None:
            continue
        confirmed = bool(
            (
                (snap_counts["player"] == player)
                & (snap_counts["team"] == team)
                & (snap_counts["position"] == KICKER_POSITION)
            ).any()
        )
        if not confirmed:
            continue
        records.append(
            {
                "player": player,
                "team": team,
                "report_year": int(row["url_year"]),
                "report_month": int(row["url_month"]),
                "slug": row["slug"],
            }
        )
    return pd.DataFrame.from_records(
        records, columns=["player", "team", "report_year", "report_month", "slug"]
    )


def describe_kicker_change_population(
    transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> dict[str, object]:
    """Diagnostic counts for the kicker-change population (never used to
    build the flag itself)."""

    rows = confirmed_kicker_change_transactions(transactions_index)
    resolved_team = rows["slug"].astype(str).map(lambda s: len(match_transaction_teams(s)))
    events = _kicker_change_events(transactions_index, snap_counts)
    return {
        "n_candidate_slugs": len(rows),
        "n_resolved_exactly_one_team": int((resolved_team == 1).sum()),
        "n_resolved_kicker_and_team": len(events),
        "resolved_slugs": events["slug"].tolist(),
    }


def derive_kicker_change_underdog_features(
    schedule: pd.DataFrame,
    transactions_index: pd.DataFrame,
    snap_counts: pd.DataFrame,
    opener_lines: pd.DataFrame,
) -> pd.DataFrame:
    """Return ``(game_id, kicker_change_underdog_flag)`` for every game.

    ``+1`` when the HOME team is the underdog at the Tuesday opener AND
    either team changed its kicker (a confirmed signing/claim/elevation/
    activation event whose calendar report falls strictly before this
    game -- the FIRST such qualifying REG game after the event, matching
    LEAD-14/LEAD-23's own "next game(s) after the news" convention);
    ``-1`` when the AWAY team is the underdog under the same eligibility;
    ``0`` otherwise -- including neither team changing kickers, an exact
    opener pick'em, or a missing opener spread (never silently treated as
    satisfying either threshold).
    """

    missing = sorted(_KICKER_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")
    if "game_id" not in opener_lines.columns:
        raise DataContractError("opener_lines is missing the game_id join key")

    reg = schedule.loc[schedule["game_type"].eq("REG")].copy()
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["week"] = pd.to_numeric(reg["week"], errors="raise").astype(int)
    reg["gameday_dt"] = pd.to_datetime(reg["gameday"], errors="raise")

    events = _kicker_change_events(transactions_index, snap_counts)
    qualifying_records: list[dict[str, object]] = []
    for _, event in events.iterrows():
        team = event["team"]
        report_end = _month_end_timestamp(int(event["report_year"]), int(event["report_month"]))
        team_games = reg.loc[
            ((reg["home_team"] == team) | (reg["away_team"] == team))
            & (reg["gameday_dt"] > report_end)
        ].sort_values("gameday_dt")
        for _, game in team_games.head(KICKER_CHANGE_GAMES).iterrows():
            qualifying_records.append(
                {"season": int(game["season"]), "week": int(game["week"]), "team": str(team)}
            )
    qualifying = pd.DataFrame.from_records(
        qualifying_records, columns=["season", "week", "team"]
    ).drop_duplicates()

    base = schedule.loc[:, ["game_id", "season", "week", "home_team", "away_team"]].copy()
    base["game_id"] = base["game_id"].astype(str)
    base["season"] = pd.to_numeric(base["season"], errors="raise").astype(int)
    base["week"] = pd.to_numeric(base["week"], errors="raise").astype(int)

    if qualifying.empty:
        home_changed = pd.Series(False, index=base.index)
        away_changed = pd.Series(False, index=base.index)
    else:
        qual = qualifying.assign(qualifies=True)
        home_changed = (
            base.merge(
                qual.rename(columns={"team": "home_team"}),
                on=["season", "week", "home_team"],
                how="left",
                validate="many_to_one",
            )["qualifies"]
            .fillna(False)
            .astype(bool)
        )
        away_changed = (
            base.merge(
                qual.rename(columns={"team": "away_team"}),
                on=["season", "week", "away_team"],
                how="left",
                validate="many_to_one",
            )["qualifies"]
            .fillna(False)
            .astype(bool)
        )

    eligible = (home_changed | away_changed).to_numpy()

    lines = opener_lines.loc[:, ["game_id", "tue_open_home_spread"]].copy()
    lines["game_id"] = lines["game_id"].astype(str)
    merged = base[["game_id"]].merge(lines, on="game_id", how="left", validate="one_to_one")
    spread = merged["tue_open_home_spread"]
    home_dog = eligible & spread.notna().to_numpy() & spread.lt(0.0).to_numpy()
    away_dog = eligible & spread.notna().to_numpy() & spread.gt(0.0).to_numpy()
    flag = np.where(home_dog, 1.0, np.where(away_dog, -1.0, 0.0))
    return pd.DataFrame({"game_id": merged["game_id"], KICKER_CHANGE_COLUMN: flag})


def attach_kicker_change_underdog_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    transactions_index: pd.DataFrame | None = None,
    snap_counts: pd.DataFrame | None = None,
    opener_lines: pd.DataFrame | None = None,
    market_root: Path | None = None,
) -> pd.DataFrame:
    """Additively join ``kicker_change_underdog_flag`` onto ``features``."""

    resolved_transactions = (
        transactions_index if transactions_index is not None else default_transactions_index()
    )
    resolved_snaps = snap_counts if snap_counts is not None else default_snap_counts()

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        lines = (
            opener_lines
            if opener_lines is not None
            else default_opener_lines(sched, market_root=market_root)
        )
        return derive_kicker_change_underdog_features(
            sched, resolved_transactions, resolved_snaps, lines
        )

    return _attach(features, schedule, _derive, (KICKER_CHANGE_COLUMN,))


__all__ = [
    "KICKER_ACQUIRE_CATEGORIES",
    "KICKER_ACQUIRE_RE",
    "KICKER_ACQUIRE_SPECULATIVE_RE",
    "KICKER_CHANGE_COLUMN",
    "KICKER_CHANGE_GAMES",
    "KICKER_POSITION",
    "ROOKIE_WALL_DEPENDENCE_COLUMN",
    "attach_kicker_change_underdog_features",
    "attach_rookie_wall_dependence_fade_features",
    "confirmed_kicker_change_transactions",
    "default_schedule",
    "derive_kicker_change_underdog_features",
    "derive_rookie_wall_dependence_fade_features",
    "describe_kicker_change_population",
    "kicker_player_slugs",
    "rookie_wall_dependence_table",
]
