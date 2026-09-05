"""Two Phase 12 roster-availability flags from the PFR transaction wire and
the nflverse weekly injury report, each stacked on PRODUCTION
(``docs/schedule_flag_battery.md`` "Wave 7"): LEAD-13 IR-return
reinforcement bump, LEAD-17 specialist (long-snapper/punter) absence fade.

**Data sources, all already-captured local snapshots, no network fetch**:
the newest ``data/raw/pfr_transactions/<snapshot>/index.parquet`` (reused,
never duplicated, via ``nfl_ats.transaction_flag_features``'s own loader and
team-nickname matching), the PINNED ``data/players/raw/20260817T184901Z/
snap_counts.parquet`` (per-game ``player``/``team``/``offense_pct``/
``defense_pct``, LEAD-13's own "trailing snap share before going on IR"
input -- pinned, not "newest", per the fleet task's explicit instruction,
since a newer player snapshot landed mid-session from a concurrent lane and
this lead must not silently pick it up), and the PINNED
``data/raw/nflverse_injuries/20260826T122850Z/injuries.parquet`` (LEAD-17's
own weekly ``report_status``/``position`` input).

**Population construction is text-based and inherently approximate for the
wire component**, exactly as ``nfl_ats.transaction_flag_features``'s module
docstring already establishes for LEAD-12/23/14: every wire-derived event
starts from a free-text PFR headline slug, resolved to a single team and a
single player via the SAME token-anchored substring match against a known
player-name universe (:func:`nfl_ats.transaction_flag_features.
distinct_player_slugs` / :func:`find_player_in_segment`), never a free-text
name parser, and a resolution that fails at any step drops the row -- never
guessed.

**Frozen headline-phrase discipline (LEAD-13), predeclared before scoring**:
the task's own two phrase families, "activated from injured reserve" and
"designated to return", are treated as two INDEPENDENT event types, each
extracted by its own team/player-clause-anchored regex (mirroring
``transaction_flag_features._parse_acquisition``'s own clause-splitting, not
a whole-slug scan) so a compound headline naming two different players'
two different events (measured in the real corpus, e.g. one team activating
one player from IR while separately designating a second player to return)
attributes each event to the correct player, never either to both or to
neither:

- ``IR_ACTIVATE_RE``: ``<team-prefix>-activat(e|es|ed)-<player>-(from|off)-
  (injured-reserve|ir)`` -- both prepositions appear in the real corpus for
  the identical event ("activate ... off IR" and "activate ... from IR").
  Reused unchanged to CLOSE a LEAD-17 specialist-out window (the same
  "player is back" fact closes either lead's window).
- ``DESIGNATE_RETURN_RE``: ``<team-prefix>-designat(e|es|ed)-<player>-(for|
  to)-return``, with ``-for-ir-return`` first normalized to ``-for-return-
  from-ir`` (:func:`_normalize_designate_phrasing`, a disclosed, deterministic
  text rewrite -- the real corpus uses both word orders for the identical
  event) so a single suffix check applies uniformly. A trailing ``-from-
  pup...``/``-from-nfi...``/``-from-covid...`` qualifier EXCLUDES the row
  (measured in the real corpus: PUP-list, NFI-list, and COVID-19-reserve
  "designated to return" headlines use this identical verb phrase for a
  return that is NOT an IR return); a bare or ``-from-ir``/``-from-injured-
  reserve`` suffix is treated as an IR return, the task's own default
  reading of the phrase.

**Frozen headline-phrase discipline (LEAD-17 wire component)**: ``IR_PLACE_RE``
matches ``<team-prefix>-(to-)?place(s|d)?-<player>-(back-)?on-ir``, the
same clause-anchored shape, so a compound "place PLAYER-A on IR, promote
PLAYER-B" headline never misattributes PLAYER-B's promotion as an IR
placement. The matched player is confirmed against a player-name universe
restricted to LEAD-17's own LS/P positions (:func:`specialist_player_slugs`,
built from ``injuries.parquet`` itself, never guessed from headline
position abbreviations), which doubles as the "is this player a specialist"
gate -- a name that resolves at all from that restricted universe is, by
construction, a long snapper or punter.

**Leakage.** Every wire event's own report month/year is converted to the
LATEST possible instant consistent with that month-only precision
(:func:`_month_end_timestamp`, duplicated from
``nfl_ats.transaction_flag_features`` -- a tiny, stable, private helper,
duplicated rather than imported per this repo's convention for small
cross-module helpers), and every qualifying game requires the team's own
kickoff to fall STRICTLY AFTER that instant. LEAD-13's own starter gate
additionally restricts "trailing snap share" to weeks whose OWN kickoff is
strictly before the event's report-month-end, so it can never read a week
that has not happened yet relative to the report.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.constants import (
    IR_RETURN_REINFORCEMENT_ON_PRODUCTION_FEATURE_COLUMNS,
    SPECIALIST_ABSENCE_FADE_ON_PRODUCTION_FEATURE_COLUMNS,
)
from nfl_ats.data import DataContractError
from nfl_ats.transaction_flag_features import (
    default_schedule,
    default_transactions_index,
    distinct_player_slugs,
    find_player_in_segment,
    latest_pfr_transactions_snapshot,
)
from nfl_ats.transaction_wire_features import canonical_team, match_transaction_teams

REPO_ROOT = Path(__file__).resolve().parents[2]

IR_RETURN_REINFORCEMENT_COLUMN = IR_RETURN_REINFORCEMENT_ON_PRODUCTION_FEATURE_COLUMNS[0]
SPECIALIST_ABSENCE_FADE_COLUMN = SPECIALIST_ABSENCE_FADE_ON_PRODUCTION_FEATURE_COLUMNS[0]

#: Pinned, not "newest" -- see module docstring. Frozen 2026-09-05.
DEFAULT_SNAP_COUNTS_PATH = REPO_ROOT / "data/players/raw/20260817T184901Z/snap_counts.parquet"
#: Pinned, not "newest" -- see module docstring. Frozen 2026-09-05.
DEFAULT_INJURIES_PATH = REPO_ROOT / "data/raw/nflverse_injuries/20260826T122850Z/injuries.parquet"

HIGH_SNAP_SHARE_THRESHOLD = 0.5
IR_RETURN_WEEK_START = 5
IR_RETURN_WEEK_END = 8
SPECIALIST_POSITIONS: tuple[str, ...] = ("LS", "P")
SPECIALIST_INJURY_SEASON_END = 2024  # "full 2009-2024 depth" per the task/ROADMAP row.

_REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
}


# ---------------------------------------------------------------------------
# Frozen phrase regexes (predeclared 2026-09-05, before any scoring)
# ---------------------------------------------------------------------------

IR_ACTIVATE_RE = re.compile(
    r"^(?P<prefix>.*?)-activat(?:e|es|ed)-(?P<player>.+?)-(?:from|off)-(?:injured-reserve|ir)(?:-|$)"
)
IR_PLACE_RE = re.compile(
    r"^(?P<prefix>.*?)-(?:to-)?place[sd]?-(?P<player>.+?)-(?:back-)?on-ir(?:-|$)"
)
DESIGNATE_RETURN_RE = re.compile(
    r"^(?P<prefix>.*?)-designat(?:e|es|ed)-(?P<player>.+?)-(?:for|to)-return"
    r"(?P<suffix>-from-[a-z0-9]+)?(?:-|$)"
)
_NON_IR_RETURN_SUFFIX_RE = re.compile(r"^-from-(?:pup|nfi|covid)")


def _normalize_designate_phrasing(slug: str) -> str:
    """Rewrite the ``-for-ir-return``/``-to-ir-return`` word order (measured
    in the real corpus, e.g. ``"...-for-ir-return"``) to the ``-from-ir``
    suffix shape :data:`DESIGNATE_RETURN_RE` otherwise expects -- a
    disclosed, deterministic text rewrite, not a guess."""

    return slug.replace("-for-ir-return", "-for-return-from-ir").replace(
        "-to-ir-return", "-to-return-from-ir"
    )


def _month_end_timestamp(year: int, month: int) -> pd.Timestamp:
    """Duplicated (not imported) from
    ``nfl_ats.transaction_flag_features``'s identical private helper -- the
    latest calendar instant consistent with a month-only-precision date."""

    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def _prefix_team(prefix: str) -> str | None:
    teams = match_transaction_teams(prefix)
    if len(teams) != 1:
        return None
    return next(iter(teams))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def pinned_snap_counts(path: Path | None = None) -> pd.DataFrame:
    """The PINNED ``snap_counts.parquet`` (see module docstring), team codes
    canonicalized, plus ``snap_share = max(offense_pct, defense_pct)`` --
    same shape as ``nfl_ats.transaction_flag_features.default_snap_counts``,
    built independently against the pinned path rather than "newest"."""

    frame = pd.read_parquet(path or DEFAULT_SNAP_COUNTS_PATH)
    required = {"player", "team", "season", "week", "offense_pct", "defense_pct"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"snap_counts is missing columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["team"] = frame["team"].astype(str).map(canonical_team)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)
    frame["snap_share"] = frame[["offense_pct", "defense_pct"]].max(axis=1)
    return frame


def pinned_injuries(path: Path | None = None) -> pd.DataFrame:
    """The PINNED weekly injury report (see module docstring), team codes
    canonicalized, ``week`` coerced to a nullable integer (raw ``week`` is
    ``float64`` with the occasional postseason NaN season-total row)."""

    frame = pd.read_parquet(path or DEFAULT_INJURIES_PATH)
    required = {"season", "week", "team", "position", "report_status", "game_type"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"injuries is missing columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["team"] = frame["team"].astype(str).map(canonical_team)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce").astype("Int64")
    return frame


def _require_schedule_columns(schedule: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")


def _reg_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    _require_schedule_columns(schedule)
    reg = schedule.loc[schedule["game_type"].eq("REG")].copy()
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["week"] = pd.to_numeric(reg["week"], errors="raise").astype(int)
    reg["gameday_dt"] = pd.to_datetime(reg["gameday"], errors="raise")
    return reg


def _team_week_kickoffs(reg_schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(season, team, week, gameday_dt)`` -- every REG team-week
    kickoff, home and away sides both melted into one ``team`` column."""

    home = reg_schedule[["season", "week", "home_team", "gameday_dt"]].rename(
        columns={"home_team": "team"}
    )
    away = reg_schedule[["season", "week", "away_team", "gameday_dt"]].rename(
        columns={"away_team": "team"}
    )
    return pd.concat([home, away], ignore_index=True)


# ---------------------------------------------------------------------------
# Shared additive-merge / sign-convention helper
# ---------------------------------------------------------------------------


def _signed_flag_from_qualifying(
    schedule: pd.DataFrame, qualifying: pd.DataFrame, column: str
) -> pd.DataFrame:
    """Duplicated (not imported) from
    ``transaction_flag_features._attach_qualifying_sides`` -- ``+1`` when the
    AWAY team qualifies and the HOME team does not, ``-1`` when the reverse,
    ``0`` otherwise (both, neither, or no schedule row for that
    season/week/team)."""

    reg = schedule.loc[
        :, ["game_id", "season", "week", "game_type", "home_team", "away_team"]
    ].copy()
    reg["game_id"] = reg["game_id"].astype(str)
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["week"] = pd.to_numeric(reg["week"], errors="raise").astype(int)

    if qualifying.empty:
        flag = np.zeros(len(reg), dtype=float)
        return pd.DataFrame({"game_id": reg["game_id"], column: flag})

    qual = qualifying.drop_duplicates(["season", "week", "team"]).assign(qualifies=True)
    merged = reg.merge(
        qual.rename(columns={"team": "home_team"}),
        on=["season", "week", "home_team"],
        how="left",
    ).rename(columns={"qualifies": "home_qualifies"})
    merged = merged.merge(
        qual.rename(columns={"team": "away_team"}),
        on=["season", "week", "away_team"],
        how="left",
    ).rename(columns={"qualifies": "away_qualifies"})
    home_q = merged["home_qualifies"].fillna(False).astype(bool)
    away_q = merged["away_qualifies"].fillna(False).astype(bool)
    flag = np.where(away_q & ~home_q, 1.0, np.where(home_q & ~away_q, -1.0, 0.0))
    return pd.DataFrame({"game_id": merged["game_id"], column: flag})


def _attach(features: pd.DataFrame, derived: pd.DataFrame, column: str) -> pd.DataFrame:
    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    if column in features.columns:
        raise DataContractError(f"features already carries {column}")
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_roster_flag"),
        validate="one_to_one",
    )
    merged = merged.drop(
        columns=[c for c in ("key_0", "game_id_roster_flag") if c in merged.columns]
    )
    merged.index = features.index
    return merged


# ---------------------------------------------------------------------------
# Shared wire-event extraction (activation is used by BOTH leads)
# ---------------------------------------------------------------------------


def ir_activation_events(
    transactions_index: pd.DataFrame, player_slugs: pd.DataFrame
) -> pd.DataFrame:
    """Every resolvable ``<team-prefix>-activat(e|es|ed)-<player>-from-
    (injured-reserve|ir)`` event: ``player``, ``team``, ``report_year``,
    ``report_month``, ``slug``. ``player_slugs`` fixes which player universe
    resolves the match (LEAD-13: the full snap-count universe; LEAD-17: the
    LS/P-restricted universe used to CLOSE a specialist-out window)."""

    rows = transactions_index.loc[transactions_index["category"] == "ir_activation"]
    records: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        if pd.isna(row["url_year"]) or pd.isna(row["url_month"]):
            continue
        match = IR_ACTIVATE_RE.search(str(row["slug"]))
        if match is None:
            continue
        team = _prefix_team(match.group("prefix"))
        if team is None:
            continue
        player = find_player_in_segment(match.group("player"), player_slugs)
        if player is None:
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


# ---------------------------------------------------------------------------
# LEAD-13: IR-return reinforcement bump
# ---------------------------------------------------------------------------


def designate_return_events(
    transactions_index: pd.DataFrame, player_slugs: pd.DataFrame
) -> pd.DataFrame:
    """Every resolvable ``<team-prefix>-designat(e|es|ed)-<player>-(for|to)-
    return`` event whose suffix is NOT a PUP/NFI/COVID-list return (see
    module docstring). Scanned over the FULL ``transaction_relevant`` index,
    never pre-filtered by ``category`` -- measured against the real corpus:
    a real designate-for-return headline can classify under ``signing`` or
    ``other`` (e.g. a compound "sign PLAYER-A, designate PLAYER-B for
    return" headline), so an ``ir_placement``/``ir_activation`` category
    pre-filter would silently drop real events."""

    records: list[dict[str, object]] = []
    for _, row in transactions_index.iterrows():
        if pd.isna(row["url_year"]) or pd.isna(row["url_month"]):
            continue
        normalized = _normalize_designate_phrasing(str(row["slug"]))
        match = DESIGNATE_RETURN_RE.search(normalized)
        if match is None:
            continue
        suffix = match.group("suffix")
        if suffix is not None and _NON_IR_RETURN_SUFFIX_RE.search(suffix):
            continue  # PUP/NFI/COVID-19-list return, not an IR return.
        team = _prefix_team(match.group("prefix"))
        if team is None:
            continue
        player = find_player_in_segment(match.group("player"), player_slugs)
        if player is None:
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


def _ir_return_events(transactions_index: pd.DataFrame, snap_counts: pd.DataFrame) -> pd.DataFrame:
    """The two phrase families combined, deduplicated to the EARLIEST
    qualifying report per ``(player, team, season)`` (a player can be both
    "designated to return" and later "activated" for the same IR stint; the
    market reaction the task's mechanism describes begins at the earlier
    public report)."""

    player_slugs = distinct_player_slugs(snap_counts)
    activated = ir_activation_events(transactions_index, player_slugs)
    activated["event_type"] = "activated"
    designated = designate_return_events(transactions_index, player_slugs)
    designated["event_type"] = "designated_return"
    combined = pd.concat([activated, designated], ignore_index=True)
    if combined.empty:
        return combined
    combined["season"] = combined["report_year"].astype(int)
    combined["_month_idx"] = combined["report_year"].astype(int) * 12 + combined[
        "report_month"
    ].astype(int)
    combined = combined.sort_values("_month_idx")
    return combined.drop_duplicates(["player", "team", "season"], keep="first").reset_index(
        drop=True
    )


def describe_ir_return_population(
    transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> dict[str, object]:
    """Diagnostic counts for the IR-return population (never used to build
    the flag itself)."""

    player_slugs = distinct_player_slugs(snap_counts)
    activated = ir_activation_events(transactions_index, player_slugs)
    designated = designate_return_events(transactions_index, player_slugs)
    events = _ir_return_events(transactions_index, snap_counts)
    return {
        "n_resolved_activation_events": len(activated),
        "n_resolved_designation_events": len(designated),
        "n_resolved_deduplicated_events": len(events),
        "resolved_slugs": events["slug"].tolist() if not events.empty else [],
    }


def derive_ir_return_reinforcement_features(
    schedule: pd.DataFrame, transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> pd.DataFrame:
    """Return ``(game_id, ir_return_reinforcement_flag)`` for every game in
    ``schedule``.

    ``+1`` when the HOME team fields a confirmed, snap-share-confirmed
    STARTER (trailing mean ``snap_share`` >= :data:`HIGH_SNAP_SHARE_THRESHOLD`
    over that player's own recorded weeks with that team in the SAME season,
    restricted to weeks strictly before the report's month-end -- i.e.
    "before going on IR", never a later week) returning from IR in this
    game's own week, if that week is one of :data:`IR_RETURN_WEEK_START`-
    :data:`IR_RETURN_WEEK_END`; ``-1`` when the AWAY team does; ``0``
    otherwise -- including a return outside weeks 5-8, a return whose report
    is not strictly pregame for that week (leakage guard), or a starter
    determination that could not be resolved (never guessed).

    Predeclared direction: BACK the team with the returning starter (task's
    own "BACK teams with returning IR starters"), so the sign here is
    HOME-positive for a home return -- the mirror of every sibling FADE
    construct's sign, disclosed explicitly since this is the one BACK-signed
    lead in the battery.
    """

    reg = _reg_schedule(schedule)
    kickoffs = _team_week_kickoffs(reg)

    events = _ir_return_events(transactions_index, snap_counts)
    qualifying_records: list[dict[str, object]] = []
    for _, event in events.iterrows():
        player, team, season = event["player"], event["team"], event["season"]
        report_end = _month_end_timestamp(int(event["report_year"]), int(event["report_month"]))

        prior_weeks = kickoffs.loc[
            (kickoffs["season"] == season)
            & (kickoffs["team"] == team)
            & (kickoffs["gameday_dt"] < report_end),
            "week",
        ]
        prior_rows = snap_counts.loc[
            (snap_counts["player"] == player)
            & (snap_counts["team"] == team)
            & (snap_counts["season"] == season)
            & (snap_counts["week"].isin(prior_weeks))
        ]
        if prior_rows.empty:
            continue  # cannot resolve "starter before going on IR" -- never guessed.
        trailing_share = float(prior_rows["snap_share"].mean())
        if trailing_share < HIGH_SNAP_SHARE_THRESHOLD:
            continue

        team_games = reg.loc[
            (reg["season"] == season)
            & ((reg["home_team"] == team) | (reg["away_team"] == team))
            & (reg["week"] >= IR_RETURN_WEEK_START)
            & (reg["week"] <= IR_RETURN_WEEK_END)
            & (reg["gameday_dt"] > report_end)
        ]
        for _, game in team_games.iterrows():
            qualifying_records.append(
                {"season": int(game["season"]), "week": int(game["week"]), "team": team}
            )

    qualifying = pd.DataFrame.from_records(qualifying_records, columns=["season", "week", "team"])
    # Sign convention is HOME-positive (see docstring): reuse the shared
    # AWAY-positive helper, then flip.
    derived = _signed_flag_from_qualifying(schedule, qualifying, IR_RETURN_REINFORCEMENT_COLUMN)
    derived[IR_RETURN_REINFORCEMENT_COLUMN] = -derived[IR_RETURN_REINFORCEMENT_COLUMN]
    return derived


def attach_ir_return_reinforcement_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    transactions_index: pd.DataFrame | None = None,
    snap_counts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Additively join ``ir_return_reinforcement_flag`` onto ``features`` by
    ``game_id``."""

    resolved_schedule = schedule if schedule is not None else default_schedule()
    resolved_transactions = (
        transactions_index if transactions_index is not None else default_transactions_index()
    )
    resolved_snaps = snap_counts if snap_counts is not None else pinned_snap_counts()
    derived = derive_ir_return_reinforcement_features(
        resolved_schedule, resolved_transactions, resolved_snaps
    )
    return _attach(features, derived, IR_RETURN_REINFORCEMENT_COLUMN)


# ---------------------------------------------------------------------------
# LEAD-17: specialist (LS/P) absence fade
# ---------------------------------------------------------------------------


def specialist_player_slugs(injuries: pd.DataFrame) -> pd.DataFrame:
    """The LS/P-restricted player-name universe (see module docstring): any
    name resolving from this universe is, by construction, a long snapper or
    punter -- doubles as the position gate for the wire component."""

    lsp = injuries.loc[injuries["position"].isin(SPECIALIST_POSITIONS)]
    renamed = lsp.rename(columns={"full_name": "player"})
    return distinct_player_slugs(renamed)


def weekly_specialist_out_qualifying(injuries: pd.DataFrame) -> pd.DataFrame:
    """``(season, week, team)`` rows where a LS/P is on the weekly injury
    report as ``report_status == "Out"``, REG season only, seasons through
    :data:`SPECIALIST_INJURY_SEASON_END` (the task's own "full 2009-2024
    depth", excluding the disclosed 2025 ``date_modified`` schema break --
    ``docs/injury_timestamp_fallback.md``). A Wednesday-Friday report; this
    family is declared a late-week REFRESH-channel candidate, graded here at
    the frozen Tuesday line, per the officiating-crew leads' own precedent
    (``docs/officials_crew_leads.md``)."""

    mask = (
        injuries["position"].isin(SPECIALIST_POSITIONS)
        & injuries["report_status"].eq("Out")
        & injuries["game_type"].eq("REG")
        & injuries["week"].notna()
        & (injuries["season"] <= SPECIALIST_INJURY_SEASON_END)
    )
    rows = injuries.loc[mask, ["season", "week", "team"]].copy()
    rows["week"] = rows["week"].astype(int)
    return rows.drop_duplicates().reset_index(drop=True)


def specialist_ir_placement_events(
    transactions_index: pd.DataFrame, lsp_slugs: pd.DataFrame
) -> pd.DataFrame:
    """Every resolvable ``<team-prefix>-(to-)?place(s|d)?-<player>-(back-)?
    on-ir`` event whose player resolves from the LS/P-restricted universe.
    Pre-filtered to ``category == "ir_placement"`` (:data:`IR_PLACE_RE`'s own
    "place ... on ir" shape, with no "activat" token, is exactly what that
    category requires -- measured against the real corpus, every match here
    already classifies that way)."""

    rows = transactions_index.loc[transactions_index["category"] == "ir_placement"]
    records: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        if pd.isna(row["url_year"]) or pd.isna(row["url_month"]):
            continue
        match = IR_PLACE_RE.search(str(row["slug"]))
        if match is None:
            continue
        team = _prefix_team(match.group("prefix"))
        if team is None:
            continue
        player = find_player_in_segment(match.group("player"), lsp_slugs)
        if player is None:
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


def describe_specialist_population(
    injuries: pd.DataFrame, transactions_index: pd.DataFrame
) -> dict[str, object]:
    """Diagnostic counts for the specialist-absence population (never used
    to build the flag itself)."""

    weekly = weekly_specialist_out_qualifying(injuries)
    lsp_slugs = specialist_player_slugs(injuries)
    placements = specialist_ir_placement_events(transactions_index, lsp_slugs)
    activations = ir_activation_events(transactions_index, lsp_slugs)
    return {
        "n_weekly_out_team_weeks": len(weekly),
        "weekly_out_by_season": {
            str(season): len(group) for season, group in weekly.groupby("season")
        },
        "n_resolved_ir_placement_events": len(placements),
        "n_resolved_ir_activation_events_lsp": len(activations),
        "resolved_placement_slugs": placements["slug"].tolist() if not placements.empty else [],
    }


def _specialist_wire_window_qualifying(
    reg: pd.DataFrame, transactions_index: pd.DataFrame, injuries: pd.DataFrame
) -> pd.DataFrame:
    """``(season, week, team)`` rows for every REG game strictly after a
    confirmed LS/P IR placement and at/before a confirmed same-player,
    same-team, same-season activation report's own month-end (or through
    the rest of that season if no activation is confirmed) -- extends
    coverage into weeks a placed specialist has already dropped off the
    weekly injury report entirely."""

    lsp_slugs = specialist_player_slugs(injuries)
    placements = specialist_ir_placement_events(transactions_index, lsp_slugs)
    if placements.empty:
        return pd.DataFrame(columns=["season", "week", "team"])
    activations = ir_activation_events(transactions_index, lsp_slugs)

    placements = placements.copy()
    placements["season"] = placements["report_year"].astype(int)
    placements["_month_idx"] = placements["report_year"].astype(int) * 12 + placements[
        "report_month"
    ].astype(int)
    placements = placements.sort_values("_month_idx").drop_duplicates(
        ["player", "team", "season"], keep="first"
    )

    records: list[dict[str, object]] = []
    for _, placement in placements.iterrows():
        player, team, season = placement["player"], placement["team"], placement["season"]
        placement_end = _month_end_timestamp(
            int(placement["report_year"]), int(placement["report_month"])
        )
        closing_end: pd.Timestamp | None = None
        if not activations.empty:
            same = activations.loc[
                (activations["player"] == player)
                & (activations["team"] == team)
                & (activations["report_year"].astype(int) == season)
                & (
                    activations["report_year"].astype(int) * 12
                    + activations["report_month"].astype(int)
                    > int(placement["_month_idx"])
                )
            ]
            if not same.empty:
                earliest = (
                    same.assign(
                        _idx=same["report_year"].astype(int) * 12 + same["report_month"].astype(int)
                    )
                    .sort_values("_idx")
                    .iloc[0]
                )
                closing_end = _month_end_timestamp(
                    int(earliest["report_year"]), int(earliest["report_month"])
                )

        team_games = reg.loc[
            (reg["season"] == season)
            & ((reg["home_team"] == team) | (reg["away_team"] == team))
            & (reg["gameday_dt"] > placement_end)
        ]
        if closing_end is not None:
            team_games = team_games.loc[team_games["gameday_dt"] <= closing_end]
        for _, game in team_games.iterrows():
            records.append({"season": int(game["season"]), "week": int(game["week"]), "team": team})
    return pd.DataFrame.from_records(records, columns=["season", "week", "team"])


def derive_specialist_absence_features(
    schedule: pd.DataFrame, transactions_index: pd.DataFrame, injuries: pd.DataFrame
) -> pd.DataFrame:
    """Return ``(game_id, specialist_absence_fade_flag)`` for every game in
    ``schedule``.

    ``+1`` when the AWAY team is missing its long snapper or punter (weekly
    "Out" report OR a confirmed wire IR-placement window, see module
    docstring); ``-1`` when the HOME team is; ``0`` otherwise (including
    both or neither). Rare by construction, per the task's own framing --
    recorded regardless of width.
    """

    reg = _reg_schedule(schedule)
    weekly = weekly_specialist_out_qualifying(injuries)
    wire = _specialist_wire_window_qualifying(reg, transactions_index, injuries)
    qualifying = pd.concat([weekly, wire], ignore_index=True).drop_duplicates()
    qualifying = qualifying.loc[qualifying["season"] <= SPECIALIST_INJURY_SEASON_END]
    return _signed_flag_from_qualifying(schedule, qualifying, SPECIALIST_ABSENCE_FADE_COLUMN)


def attach_specialist_absence_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    transactions_index: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Additively join ``specialist_absence_fade_flag`` onto ``features`` by
    ``game_id``."""

    resolved_schedule = schedule if schedule is not None else default_schedule()
    resolved_transactions = (
        transactions_index if transactions_index is not None else default_transactions_index()
    )
    resolved_injuries = injuries if injuries is not None else pinned_injuries()
    derived = derive_specialist_absence_features(
        resolved_schedule, resolved_transactions, resolved_injuries
    )
    return _attach(features, derived, SPECIALIST_ABSENCE_FADE_COLUMN)


__all__ = [
    "DEFAULT_INJURIES_PATH",
    "DEFAULT_SNAP_COUNTS_PATH",
    "DESIGNATE_RETURN_RE",
    "HIGH_SNAP_SHARE_THRESHOLD",
    "IR_ACTIVATE_RE",
    "IR_PLACE_RE",
    "IR_RETURN_WEEK_END",
    "IR_RETURN_WEEK_START",
    "SPECIALIST_INJURY_SEASON_END",
    "SPECIALIST_POSITIONS",
    "attach_ir_return_reinforcement_features",
    "attach_specialist_absence_features",
    "derive_ir_return_reinforcement_features",
    "derive_specialist_absence_features",
    "describe_ir_return_population",
    "describe_specialist_population",
    "designate_return_events",
    "ir_activation_events",
    "latest_pfr_transactions_snapshot",
    "pinned_injuries",
    "pinned_snap_counts",
    "specialist_ir_placement_events",
    "specialist_player_slugs",
]
