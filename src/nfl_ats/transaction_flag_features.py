"""Three pregame flags from the Pro Football Rumors (PFR) transaction wire,
each stacked on PRODUCTION (``docs/schedule_flag_battery.md`` "Wave 6"):
LEAD-12 holdout slow-start fade, LEAD-23 trade-deadline integration drag,
LEAD-14 suspension-return rust.

**Data sources, all already-captured local snapshots, no network fetch**:
the newest ``data/raw/pfr_transactions/<snapshot>/index.parquet`` (URL/slug
inventory, ``nfl_ats.transaction_wire_features``'s own team-nickname
matching and 8-category classifier are reused, never duplicated), the
newest ``data/players/raw/<snapshot>/snap_counts.parquet`` (per-game
``player``/``team``/``offense_pct``/``defense_pct``), and the newest
``data/raw/*/schedules.parquet``.

**Population construction is text-based and inherently approximate** --
unlike ``nfl_ats.schedule_flag_features`` (pure calendar facts) or
``nfl_ats.qb_identity_features`` (structured roster/combine joins), every
population here starts from free-text PFR headline slugs. Three shared
disciplines apply across all three leads, each measured against the real
2026-09-05 snapshot (``data/raw/pfr_transactions/20260904T215655Z``):

1. **Retrospective posts are excluded wholesale.** PFR runs a recurring "on
   this date in transactions history" column. Measured: the slug
   ``this-date-in-transactions-history-chargers-melvin-gordon-ends-holdout``
   is published 2021-09 but describes Gordon's real 2019 preseason holdout
   ending -- using this post's own publish date as the event date would
   place a 2019 fact two years late and under the wrong season entirely.
   :func:`default_transactions_index` drops every slug containing
   :data:`RETROSPECTIVE_SLUG_MARKERS` before any population is built.
2. **Player identity is resolved by token-anchored substring match**
   against the FULL universe of distinct player names appearing in
   ``snap_counts.parquet`` (:func:`distinct_player_slugs` /
   :func:`find_player_in_segment`), never by a free-text name parser. A
   name is anchored on both sides by a hyphen (``f"-{name}-" in
   f"-{segment}-"``) so a short name can never match a mere substring
   across a token boundary (e.g. ``ryan`` cannot match inside ``bryant``).
   Matches are also cross-checked against ``snap_counts`` for that
   player/team pair before being trusted (never guessed).
3. **A resolution that fails at any step drops the row, never guesses.**
   Zero teams, more than one team, no player match, or no snap-count
   history to confirm usage/duration all exclude the row from the
   population -- consistent with every other on-production candidate in
   this repo (``nfl_ats.qb_identity_features``'s "never guessed" rule for
   an unjoined starter).

Every population here is measured to be TINY (single digits to low
dozens) -- this is a property of how rarely PFR headlines use unambiguous,
non-speculative, confirmatory language for these specific mechanisms, not
an engineering shortfall. ``docs/schedule_flag_battery.md`` "Wave 6"
predeclares this and the fleet task's own instruction is to run the harness
and report the count honestly even when it is zero.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.constants import (
    DEADLINE_INTEGRATION_DRAG_ON_PRODUCTION_FEATURE_COLUMNS,
    HOLDOUT_SLOW_START_ON_PRODUCTION_FEATURE_COLUMNS,
    SUSPENSION_RETURN_RUST_ON_PRODUCTION_FEATURE_COLUMNS,
)
from nfl_ats.data import DataContractError
from nfl_ats.players import latest_player_snapshot
from nfl_ats.transaction_wire_features import (
    canonical_team,
    classify_transaction_slug,
    match_transaction_teams,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The one new column each candidate profile adds. Frozen names.
HOLDOUT_SLOW_START_COLUMN = HOLDOUT_SLOW_START_ON_PRODUCTION_FEATURE_COLUMNS[0]
DEADLINE_INTEGRATION_DRAG_COLUMN = DEADLINE_INTEGRATION_DRAG_ON_PRODUCTION_FEATURE_COLUMNS[0]
SUSPENSION_RETURN_RUST_COLUMN = SUSPENSION_RETURN_RUST_ON_PRODUCTION_FEATURE_COLUMNS[0]

DEFAULT_PLAYERS_RAW_ROOT = REPO_ROOT / "data/players/raw"

_REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
}

HIGH_SNAP_SHARE_THRESHOLD = 0.5
SUSPENSION_MIN_GAMES = 6

RETROSPECTIVE_SLUG_MARKERS: tuple[str, ...] = (
    "this-date-in-transactions-history",
    "this-date-in-nfl-transactions-history",
)


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------


def default_schedule(repo_root: Path | None = None) -> pd.DataFrame:
    """Load the newest ``data/raw/*/schedules.parquet`` snapshot.

    Duplicated (not imported) from ``nfl_ats.schedule_flag_features``/
    ``nfl_ats.qb_identity_features``'s own identical helper, per this
    repo's convention of not cross-importing between concurrently-edited
    on-production feature modules.
    """

    root = repo_root or REPO_ROOT
    candidates = sorted((root / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no data/raw/*/schedules.parquet snapshot found under {root}")
    return pd.read_parquet(candidates[-1])


def latest_pfr_transactions_snapshot(repo_root: Path | None = None) -> Path:
    """Newest ``data/raw/pfr_transactions/<snapshot>/index.parquet``."""

    root = repo_root or REPO_ROOT
    candidates = sorted((root / "data" / "raw" / "pfr_transactions").glob("*/index.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"no data/raw/pfr_transactions/*/index.parquet snapshot found under {root}"
        )
    return candidates[-1]


def default_transactions_index(
    repo_root: Path | None = None, *, snapshot: Path | None = None
) -> pd.DataFrame:
    """Newest PFR transaction-wire index, ``transaction_relevant`` rows
    only, retrospective posts excluded (see module docstring), with
    ``category`` attached via
    ``nfl_ats.transaction_wire_features.classify_transaction_slug``."""

    path = snapshot if snapshot is not None else latest_pfr_transactions_snapshot(repo_root)
    frame = pd.read_parquet(path)
    required = {"slug", "transaction_relevant", "url_year", "url_month"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"transactions index is missing columns: {', '.join(missing)}")

    frame = frame.loc[frame["transaction_relevant"]].copy()
    retrospective_pattern = "|".join(re.escape(marker) for marker in RETROSPECTIVE_SLUG_MARKERS)
    is_retrospective = frame["slug"].astype(str).str.contains(retrospective_pattern, regex=True)
    frame = frame.loc[~is_retrospective].copy()
    frame["url_year"] = pd.to_numeric(frame["url_year"], errors="coerce")
    frame["url_month"] = pd.to_numeric(frame["url_month"], errors="coerce")
    frame["category"] = frame["slug"].map(classify_transaction_slug)
    return frame.reset_index(drop=True)


def default_snap_counts(repo_root: Path | None = None) -> pd.DataFrame:
    """Load and lightly canonicalize the newest
    ``data/players/raw/<snapshot>/snap_counts.parquet``: team codes through
    :func:`nfl_ats.transaction_wire_features.canonical_team` (so an old
    OAK/SD/STL row joins correctly against nickname-derived, already
    canonical team codes), plus a ``snap_share`` column
    (``max(offense_pct, defense_pct)``, matching how this module reasons
    about "started"/"high-snap" for both offensive and defensive players).
    """

    root = repo_root or REPO_ROOT
    snapshot = latest_player_snapshot(root / "data" / "players" / "raw")
    frame = pd.read_parquet(snapshot.snaps_path)
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


def _require_schedule_columns(schedule: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# Player-name substring matching against the snap-count universe
# ---------------------------------------------------------------------------


def _normalize_name_to_slug(name: str) -> str:
    lowered = name.strip().lower().replace("'", "").replace(".", "")
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")


def distinct_player_slugs(snap_counts: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct ``player`` name in ``snap_counts``
    (``player``, ``name_slug``), longest ``name_slug`` first -- so
    :func:`find_player_in_segment`'s linear scan prefers the more
    specific/longer name when more than one known name could otherwise
    match."""

    names = snap_counts["player"].dropna().astype(str).unique()
    frame = pd.DataFrame({"player": names})
    frame["name_slug"] = frame["player"].map(_normalize_name_to_slug)
    frame = frame.loc[frame["name_slug"].str.len() > 0].copy()
    frame["_len"] = frame["name_slug"].str.len()
    frame = frame.sort_values("_len", ascending=False).drop(columns="_len")
    return frame.drop_duplicates("name_slug").reset_index(drop=True)


def find_player_in_segment(segment: str, player_slugs: pd.DataFrame) -> str | None:
    """The longest known player full name whose hyphen-token sequence
    appears, token-anchored, inside ``segment`` -- or ``None``.

    Anchored with a leading/trailing hyphen on both sides so a name can
    never match a mere substring across an unrelated token boundary.
    ``player_slugs`` must already be sorted longest-``name_slug``-first
    (:func:`distinct_player_slugs`'s own contract) so the first match found
    is the most specific one.
    """

    padded = f"-{segment}-"
    for name_slug, player in zip(player_slugs["name_slug"], player_slugs["player"], strict=True):
        if f"-{name_slug}-" in padded:
            return str(player)
    return None


def _confirm_player_team(player: str, team: str, snap_counts: pd.DataFrame) -> bool:
    """A resolved (player, team) pair is trusted only if ``snap_counts``
    shows that player actually appearing for that team at least once, in
    any season -- a cheap sanity check against a coincidental name/nickname
    collision. Never used to build the flag's population beyond this
    binary gate."""

    return bool(((snap_counts["player"] == player) & (snap_counts["team"] == team)).any())


def _month_end_timestamp(year: int, month: int) -> pd.Timestamp:
    """The latest calendar instant consistent with a month-only-precision
    date -- the conservative (latest-possible) anchor used for every
    leakage check in this module, since PFR's own free-text dating here is
    never more precise than year/month."""

    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def _implied_season(year: int, month: int) -> int:
    """NFL season label for a calendar (year, month): a January/February
    date belongs to the season that STARTED the previous September."""

    return year - 1 if month <= 2 else year


# ---------------------------------------------------------------------------
# Shared additive-merge helper (identical shape to
# nfl_ats.schedule_flag_features._attach / nfl_ats.qb_identity_features'
# inlined equivalent)
# ---------------------------------------------------------------------------


def _attach_qualifying_sides(
    schedule: pd.DataFrame,
    qualifying: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Build ``(game_id, column)`` for every game in ``schedule``, from a
    ``qualifying`` table of ``(season, week, team)`` rows that each qualify
    the named team for the fade. ``+1`` when the AWAY team qualifies and
    the HOME team does not; ``-1`` when the reverse; ``0`` otherwise
    (including both, neither, or a game with no schedule row for that
    season/week/team at all)."""

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


def _attach(
    features: pd.DataFrame,
    derived: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    if column in features.columns:
        raise DataContractError(f"features already carries {column}")

    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_txn_flag"),
        validate="one_to_one",
    )
    merged = merged.drop(columns=[c for c in ("key_0", "game_id_txn_flag") if c in merged.columns])
    merged.index = features.index
    return merged


# ---------------------------------------------------------------------------
# LEAD-12: holdout slow-start fade
# ---------------------------------------------------------------------------

#: Confirmatory ("this already happened") holdout-ending language only.
#: Measured against the real corpus (2026-09-05): every other holdout
#: mention in the archive is speculative/negated ("threatens holdout",
#: "won't hold out", "expected to hold out", "hints at holdout") and
#: correctly does NOT match this pattern. Both alternation branches are
#: hyphen-anchored on both sides -- naive substring matching without this
#: anchor produces two measured false positives: (a) "ended-holdout" is a
#: substring of "...hints-at-EXTENDED-HOLDOUT" (the word "extended" itself
#: contains "ended"); (b) bare "report-to-camp" (no "s"/"ed") is a
#: substring of "...adams-EXPECTED-TO-REPORT-TO-CAMP" -- an infinitive
#: PREDICTION, not the confirmed present/past tense this pattern requires.
HOLDOUT_END_RE = re.compile(
    r"(?:^|-)(?:ends-holdout|ended-holdout|reports-to-camp|reported-to-camp)(?:-|$)"
)


def holdout_ending_transactions(transactions_index: pd.DataFrame) -> pd.DataFrame:
    """Every transaction-wire row using confirmatory holdout-ending
    language (:data:`HOLDOUT_END_RE`); retrospective posts are already
    excluded upstream by :func:`default_transactions_index`."""

    mask = transactions_index["slug"].astype(str).str.contains(HOLDOUT_END_RE)
    return transactions_index.loc[mask].copy()


def _holdout_events(transactions_index: pd.DataFrame, snap_counts: pd.DataFrame) -> pd.DataFrame:
    """One row per resolvable holdout-ending event: player, team, the
    calendar (year, month) it was reported, and the season it precedes
    (== ``url_year``: camp holdouts always end within the same calendar
    year as the season they precede, before that season's Week 1)."""

    rows = holdout_ending_transactions(transactions_index)
    player_slugs = distinct_player_slugs(snap_counts)

    records: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        if pd.isna(row["url_year"]) or pd.isna(row["url_month"]):
            continue
        teams = match_transaction_teams(str(row["slug"]))
        if len(teams) != 1:
            continue
        team = next(iter(teams))
        player = find_player_in_segment(str(row["slug"]), player_slugs)
        if player is None or not _confirm_player_team(player, team, snap_counts):
            continue
        records.append(
            {
                "player": player,
                "team": team,
                "season": int(row["url_year"]),
                "report_year": int(row["url_year"]),
                "report_month": int(row["url_month"]),
                "slug": row["slug"],
            }
        )
    return pd.DataFrame.from_records(
        records, columns=["player", "team", "season", "report_year", "report_month", "slug"]
    )


def describe_holdout_population(
    transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> dict[str, object]:
    """Diagnostic counts for the holdout-ending population (never used to
    build the flag itself): how many confirmatory slugs exist, how many
    resolve to exactly one team, and how many further resolve to a
    confirmed player."""

    rows = holdout_ending_transactions(transactions_index)
    resolved_team = rows["slug"].astype(str).map(lambda s: len(match_transaction_teams(s)))
    events = _holdout_events(transactions_index, snap_counts)
    return {
        "n_holdout_ending_slugs": len(rows),
        "n_resolved_exactly_one_team": int((resolved_team == 1).sum()),
        "n_resolved_player_and_team": len(events),
        "resolved_slugs": events["slug"].tolist(),
    }


def _player_started_prior_week(
    player: str, team: str, season: int, week: int, snap_counts: pd.DataFrame
) -> bool | None:
    """``True``/``False`` if resolvable, ``None`` if unresolved (never
    guessed). Week 1 falls back to the player's own last recorded game with
    ``team`` in the PRIOR season (the frozen "roster starter status" proxy
    the task allows when no in-season prior week exists yet)."""

    if week <= 1:
        prior = snap_counts.loc[
            (snap_counts["player"] == player)
            & (snap_counts["team"] == team)
            & (snap_counts["season"] == season - 1)
        ]
        if prior.empty:
            return None
        last_week = prior["week"].max()
        share = prior.loc[prior["week"] == last_week, "snap_share"].max()
        return bool(share >= HIGH_SNAP_SHARE_THRESHOLD)

    prior = snap_counts.loc[
        (snap_counts["player"] == player)
        & (snap_counts["team"] == team)
        & (snap_counts["season"] == season)
        & (snap_counts["week"] == week - 1)
    ]
    if prior.empty:
        return None
    return bool(prior["snap_share"].max() >= HIGH_SNAP_SHARE_THRESHOLD)


def derive_holdout_slow_start_features(
    schedule: pd.DataFrame, transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> pd.DataFrame:
    """Return ``(game_id, holdout_slow_start_flag)`` for every game in
    ``schedule``.

    ``+1`` when the AWAY team fields a confirmed post-holdout regular
    (snap-share-confirmed "started") in one of that team's REG weeks 1-4 of
    the season the holdout precedes; ``-1`` when the HOME team does; ``0``
    otherwise -- including a game outside weeks 1-4, a holdout report whose
    latest-possible date (month-end, since only month precision exists)
    is NOT strictly before that week's own kickoff (leakage guard; should
    never trigger by construction since camp always precedes Week 1, but
    checked rather than assumed), or a "started" determination that could
    not be resolved (never guessed).
    """

    _require_schedule_columns(schedule)
    reg = schedule.loc[schedule["game_type"].eq("REG")].copy()
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["week"] = pd.to_numeric(reg["week"], errors="raise").astype(int)
    reg["gameday_dt"] = pd.to_datetime(reg["gameday"], errors="raise")

    events = _holdout_events(transactions_index, snap_counts)
    qualifying_records: list[dict[str, object]] = []
    for _, event in events.iterrows():
        season = event["season"]
        team = event["team"]
        report_end = _month_end_timestamp(event["report_year"], event["report_month"])
        team_games = reg.loc[
            (reg["season"] == season) & ((reg["home_team"] == team) | (reg["away_team"] == team))
        ]
        for week in range(1, 5):
            week_games = team_games.loc[team_games["week"] == week]
            if week_games.empty:
                continue
            kickoff = week_games["gameday_dt"].min()
            if not (report_end < kickoff):
                continue  # leakage guard: report not confirmed pregame for this week
            started = _player_started_prior_week(event["player"], team, season, week, snap_counts)
            if started:
                qualifying_records.append({"season": season, "week": week, "team": team})

    qualifying = pd.DataFrame.from_records(qualifying_records, columns=["season", "week", "team"])
    return _attach_qualifying_sides(schedule, qualifying, HOLDOUT_SLOW_START_COLUMN)


def attach_holdout_slow_start_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    transactions_index: pd.DataFrame | None = None,
    snap_counts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Additively join ``holdout_slow_start_flag`` onto ``features`` by
    ``game_id``."""

    resolved_schedule = schedule if schedule is not None else default_schedule()
    resolved_transactions = (
        transactions_index if transactions_index is not None else default_transactions_index()
    )
    resolved_snaps = snap_counts if snap_counts is not None else default_snap_counts()
    derived = derive_holdout_slow_start_features(
        resolved_schedule, resolved_transactions, resolved_snaps
    )
    return _attach(features, derived, HOLDOUT_SLOW_START_COLUMN)


# ---------------------------------------------------------------------------
# LEAD-23: trade-deadline integration drag
# ---------------------------------------------------------------------------

#: A confirmed player acquisition, not a draft-pick trade and not
#: speculation ("looking to trade for", "eyeing", "tried to trade for" all
#: fail this pattern -- see module docstring's measured examples). Matches
#: "<team>-(to-)?(re)?acquire(s|d)-<rest>".
ACQUISITION_RE = re.compile(r"(?:^|-)(?:to-)?(?:re)?acquir(?:e|es|ed)-")
_ACQUIRE_SPLIT_RE = re.compile(r"^(?P<prefix>.*?)-(?:to-)?(?:re)?acquir(?:e|es|ed)-(?P<rest>.+)$")
#: Excludes a draft-PICK acquisition (a real player-for-picks trade, e.g.
#: "patriots-acquire-brandin-cooks", is kept; "bills-acquire-no-23-select-
#: cb-kaiir-elam" and "broncos-acquire-no-42-from-bengals" are excluded).
DRAFT_PICK_RE = re.compile(r"-no-\d+|-\d+(?:st|nd|rd|th)-pick|-pick-\d+|select-")
#: Excludes a FAILED or merely rumored acquisition. Measured against the
#: real corpus: "saints-tried-to-acquire-giants-wr-darius-slayton",
#: "packers-attempted-to-acquire-raiders-te-darren-waller-at-deadline", and
#: "browns-attempted-to-acquire-calvin-ridley-in-2022" all match
#: :data:`ACQUISITION_RE` (they contain "...to-acquire-...") but describe an
#: attempt that did not happen, not a completed trade.
SPECULATIVE_ACQUISITION_RE = re.compile(
    r"tried-to-acquir|attempted-to-acquir|wants-to-acquire|hopes-to-acquire|"
    r"hoping-to-acquire|could-acquire|would-acquire|looking-to-acquire|"
    r"interested-in-acquir|eyeing-.*acquir|exploring-.*acquir|in-talks-to-acquire"
)
#: In-season trading window only (the trade-DEADLINE mechanism this lead
#: tests does not apply to an offseason draft-capital trade with a full
#: training camp to integrate).
DEADLINE_WINDOW_MONTHS: tuple[int, ...] = (9, 10, 11, 12)
DEADLINE_INTEGRATION_GAMES = 3


def confirmed_acquisition_transactions(transactions_index: pd.DataFrame) -> pd.DataFrame:
    """Every ``trade``-category row using confirmed (not speculative or
    failed), non-draft-pick acquisition language, in the in-season trading
    window (:data:`DEADLINE_WINDOW_MONTHS`)."""

    trades = transactions_index.loc[transactions_index["category"] == "trade"].copy()
    slug = trades["slug"].astype(str)
    has_acquire = slug.str.contains(ACQUISITION_RE)
    is_pick = slug.str.contains(DRAFT_PICK_RE)
    is_speculative = slug.str.contains(SPECULATIVE_ACQUISITION_RE)
    in_window = trades["url_month"].isin(DEADLINE_WINDOW_MONTHS)
    return trades.loc[has_acquire & ~is_pick & ~is_speculative & in_window].copy()


def _parse_acquisition(slug: str) -> tuple[str, str] | None:
    """``(acquiring_team, player_text_segment)`` or ``None`` if the
    acquiring team cannot be resolved to exactly one code from the text
    preceding the acquire verb."""

    match = _ACQUIRE_SPLIT_RE.search(slug)
    if match is None:
        return None
    prefix, rest = match.group("prefix"), match.group("rest")
    acquiring_teams = match_transaction_teams(prefix)
    if len(acquiring_teams) != 1:
        return None
    player_part = rest.split("-from-", 1)[0]
    return next(iter(acquiring_teams)), player_part


def _acquisition_events(
    transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> pd.DataFrame:
    """One row per resolvable, high-snap, in-season acquisition: player,
    acquiring team, the giving (previous) team inferred from
    ``snap_counts`` itself (the last team, within the same season, other
    than the acquiring team, that the player's own snap-count history shows
    him playing for), the trailing snap share with that giving team, and
    the last week he is recorded with it (the anchor for "first three games
    after the trade")."""

    rows = confirmed_acquisition_transactions(transactions_index)
    player_slugs = distinct_player_slugs(snap_counts)

    records: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        if pd.isna(row["url_year"]) or pd.isna(row["url_month"]):
            continue
        parsed = _parse_acquisition(str(row["slug"]))
        if parsed is None:
            continue
        acquiring_team, player_part = parsed
        player = find_player_in_segment(player_part, player_slugs)
        if player is None:
            continue

        season = int(row["url_year"])
        season_rows = snap_counts.loc[
            (snap_counts["player"] == player) & (snap_counts["season"] == season)
        ]
        prior_rows = season_rows.loc[season_rows["team"] != acquiring_team]
        if prior_rows.empty:
            continue  # cannot resolve a "previous team" this season -- never guessed

        last_prior_week = int(prior_rows["week"].max())
        giving_team = str(prior_rows.loc[prior_rows["week"] == last_prior_week, "team"].iloc[0])
        trailing_rows = prior_rows.loc[prior_rows["team"] == giving_team]
        trailing_share = float(trailing_rows["snap_share"].mean())
        if trailing_share < HIGH_SNAP_SHARE_THRESHOLD:
            continue  # not "high-snap" -- population excludes low-usage acquisitions

        records.append(
            {
                "player": player,
                "acquiring_team": acquiring_team,
                "giving_team": giving_team,
                "season": season,
                "last_prior_week": last_prior_week,
                "trailing_snap_share": trailing_share,
                "report_year": season,
                "report_month": int(row["url_month"]),
                "slug": row["slug"],
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "player",
            "acquiring_team",
            "giving_team",
            "season",
            "last_prior_week",
            "trailing_snap_share",
            "report_year",
            "report_month",
            "slug",
        ],
    )


def describe_deadline_acquisition_population(
    transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> dict[str, object]:
    """Diagnostic counts for the deadline-acquisition population (never
    used to build the flag itself)."""

    rows = confirmed_acquisition_transactions(transactions_index)
    parsed = rows["slug"].astype(str).map(_parse_acquisition)
    n_team_resolved = int(parsed.notna().sum())
    events = _acquisition_events(transactions_index, snap_counts)
    return {
        "n_confirmed_acquisition_slugs": len(rows),
        "n_resolved_acquiring_team": n_team_resolved,
        "n_resolved_player_and_high_snap": len(events),
        "resolved_slugs": events["slug"].tolist(),
    }


def derive_deadline_integration_drag_features(
    schedule: pd.DataFrame, transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> pd.DataFrame:
    """Return ``(game_id, deadline_integration_drag_flag)`` for every game
    in ``schedule``.

    ``+1`` when the AWAY team acquired a confirmed high-snap player at the
    in-season deadline and this game is one of that team's first
    :data:`DEADLINE_INTEGRATION_GAMES` (3) REG games strictly after the
    player's last recorded week with his PREVIOUS team; ``-1`` when the
    HOME team did; ``0`` otherwise. The acquiring team's next games are
    read directly from its own schedule -- no month-precision date math is
    needed for game selection, since the anchor (``last_prior_week``) comes
    from the player's own week-indexed snap-count history, which is more
    precise than the wire's month-only report date. A month-end leakage
    guard is still applied against the wire's own report date as a
    belt-and-suspenders check (should never bind, since a player cannot
    appear on his new team's snap counts before the trade is public).
    """

    _require_schedule_columns(schedule)
    reg = schedule.loc[schedule["game_type"].eq("REG")].copy()
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["week"] = pd.to_numeric(reg["week"], errors="raise").astype(int)
    reg["gameday_dt"] = pd.to_datetime(reg["gameday"], errors="raise")

    events = _acquisition_events(transactions_index, snap_counts)
    qualifying_records: list[dict[str, object]] = []
    for _, event in events.iterrows():
        team = event["acquiring_team"]
        season = event["season"]
        report_end = _month_end_timestamp(event["report_year"], event["report_month"])
        team_games = reg.loc[
            (reg["season"] == season)
            & ((reg["home_team"] == team) | (reg["away_team"] == team))
            & (reg["week"] > event["last_prior_week"])
        ].sort_values("week")
        team_games = team_games.loc[team_games["gameday_dt"] > report_end]
        for _, game in team_games.head(DEADLINE_INTEGRATION_GAMES).iterrows():
            qualifying_records.append({"season": season, "week": int(game["week"]), "team": team})

    qualifying = pd.DataFrame.from_records(qualifying_records, columns=["season", "week", "team"])
    return _attach_qualifying_sides(schedule, qualifying, DEADLINE_INTEGRATION_DRAG_COLUMN)


def attach_deadline_integration_drag_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    transactions_index: pd.DataFrame | None = None,
    snap_counts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Additively join ``deadline_integration_drag_flag`` onto ``features``
    by ``game_id``."""

    resolved_schedule = schedule if schedule is not None else default_schedule()
    resolved_transactions = (
        transactions_index if transactions_index is not None else default_transactions_index()
    )
    resolved_snaps = snap_counts if snap_counts is not None else default_snap_counts()
    derived = derive_deadline_integration_drag_features(
        resolved_schedule, resolved_transactions, resolved_snaps
    )
    return _attach(features, derived, DEADLINE_INTEGRATION_DRAG_COLUMN)


# ---------------------------------------------------------------------------
# LEAD-14: suspension-return rust
# ---------------------------------------------------------------------------

#: A player CONFIRMED reinstated (returning), never a mere reinstatement
#: FILING/request, and never a suspension itself being reinstated
#: (reimposed). Measured against the real corpus: "josh-gordon-files-
#: reinstatement-suspension" describes a PETITION, not a grant --
#: "reinstatement" (the noun) never matches this pattern because it lacks
#: the "-d" of "reinstated" (the two words diverge right after
#: "reinstate"). "tom-bradys-suspension-reinstated-by-appeals-court" means
#: the SUSPENSION was reinstated (reimposed) by a court, the opposite of a
#: player returning -- excluded by the negative lookbehind on
#: "suspension-" immediately preceding "reinstated".
REINSTATED_RE = re.compile(r"(?<!suspension-)reinstated")
SUSPENSION_RETURN_GAMES = 2


def suspension_category_transactions(transactions_index: pd.DataFrame) -> pd.DataFrame:
    return transactions_index.loc[transactions_index["category"] == "suspension"].copy()


def _team_for_player_before(player: str, snap_counts: pd.DataFrame, season: int) -> str | None:
    """The team ``player`` most recently played for at or before
    ``season`` -- his last recorded team that season if he has any games
    that season yet, else his last recorded team in any earlier season.
    ``None`` if unresolved (never guessed)."""

    same_season = snap_counts.loc[
        (snap_counts["player"] == player) & (snap_counts["season"] == season)
    ]
    if not same_season.empty:
        last_week = same_season["week"].max()
        return str(same_season.loc[same_season["week"] == last_week, "team"].iloc[0])

    prior = snap_counts.loc[
        (snap_counts["player"] == player) & (snap_counts["season"] < season)
    ].sort_values(["season", "week"])
    if prior.empty:
        return None
    return str(prior.iloc[-1]["team"])


def _team_games_between(
    schedule_reg: pd.DataFrame, team: str, start_month_idx: int, end_month_idx: int
) -> int:
    """Count of ``team``'s REG games whose own calendar (year*12+month)
    index falls in the half-open interval ``[start_month_idx,
    end_month_idx)`` -- spans a season boundary correctly since this
    compares raw calendar months, never a ``season`` label."""

    team_games = schedule_reg.loc[
        (schedule_reg["home_team"] == team) | (schedule_reg["away_team"] == team)
    ]
    month_idx = team_games["gameday_dt"].dt.year * 12 + team_games["gameday_dt"].dt.month
    return int(((month_idx >= start_month_idx) & (month_idx < end_month_idx)).sum())


def _suspension_events(
    transactions_index: pd.DataFrame, snap_counts: pd.DataFrame, schedule: pd.DataFrame
) -> pd.DataFrame:
    """One row per confirmed 6+-game suspension return: player, team, and
    the calendar (year, month) of the reinstatement report.

    Duration is MEASURED, not read from a headline's own (sometimes
    word-form, e.g. "suspended-nine-games") number: the number of the
    player's own team's REG games falling between the earliest "imposed"
    report and the "reinstated" report is counted directly from the
    schedule. This generalizes to reinstatement slugs that never state an
    explicit game count at all (measured against the real corpus: PFR's
    own headlines rarely name a team in the reinstatement slug itself, so
    the team is independently resolved from the player's own snap-count
    history around the imposed date, never guessed from slug text).
    """

    _require_schedule_columns(schedule)
    reg = schedule.loc[schedule["game_type"].eq("REG")].copy()
    reg["gameday_dt"] = pd.to_datetime(reg["gameday"], errors="raise")

    susp = suspension_category_transactions(transactions_index)
    player_slugs = distinct_player_slugs(snap_counts)
    slug_text = susp["slug"].astype(str)
    is_reinstated = slug_text.str.contains(REINSTATED_RE)

    imposed = susp.loc[~is_reinstated].copy()
    reinstated = susp.loc[is_reinstated].copy()
    imposed["player"] = (
        imposed["slug"].astype(str).map(lambda s: find_player_in_segment(s, player_slugs))
    )
    reinstated["player"] = (
        reinstated["slug"].astype(str).map(lambda s: find_player_in_segment(s, player_slugs))
    )
    imposed = imposed.loc[imposed["player"].notna() & imposed["url_year"].notna()]
    reinstated = reinstated.loc[reinstated["player"].notna() & reinstated["url_year"].notna()]

    records: list[dict[str, object]] = []
    for _, r_row in reinstated.iterrows():
        player = r_row["player"]
        r_idx = int(r_row["url_year"]) * 12 + int(r_row["url_month"])

        candidates = imposed.loc[imposed["player"] == player].copy()
        if candidates.empty:
            continue
        candidates["_idx"] = candidates["url_year"].astype(int) * 12 + candidates[
            "url_month"
        ].astype(int)
        earlier = candidates.loc[candidates["_idx"] < r_idx]
        if earlier.empty:
            continue  # no confirmed "imposed" bracket -- can't measure duration
        imposed_row = earlier.sort_values("_idx").iloc[0]
        i_idx = int(imposed_row["_idx"])

        season = _implied_season(int(imposed_row["url_year"]), int(imposed_row["url_month"]))
        team = _team_for_player_before(player, snap_counts, season)
        if team is None:
            continue

        n_games = _team_games_between(reg, team, i_idx, r_idx)
        if n_games < SUSPENSION_MIN_GAMES:
            continue

        records.append(
            {
                "player": player,
                "team": team,
                "n_games_measured": n_games,
                "report_year": int(r_row["url_year"]),
                "report_month": int(r_row["url_month"]),
                "slug": r_row["slug"],
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=["player", "team", "n_games_measured", "report_year", "report_month", "slug"],
    )


def describe_suspension_return_population(
    transactions_index: pd.DataFrame, snap_counts: pd.DataFrame, schedule: pd.DataFrame
) -> dict[str, object]:
    """Diagnostic counts for the suspension-return population (never used
    to build the flag itself)."""

    susp = suspension_category_transactions(transactions_index)
    is_reinstated = susp["slug"].astype(str).str.contains(REINSTATED_RE)
    events = _suspension_events(transactions_index, snap_counts, schedule)
    return {
        "n_suspension_category_slugs": len(susp),
        "n_reinstatement_slugs": int(is_reinstated.sum()),
        "n_resolved_6plus_game_returns": len(events),
        "resolved_slugs": events["slug"].tolist(),
        "measured_game_counts": events["n_games_measured"].tolist(),
    }


def derive_suspension_return_rust_features(
    schedule: pd.DataFrame, transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> pd.DataFrame:
    """Return ``(game_id, suspension_return_rust_flag)`` for every game in
    ``schedule``.

    ``+1`` when the AWAY team is playing one of its first
    :data:`SUSPENSION_RETURN_GAMES` (2) REG games -- the return game plus
    one -- on or after a confirmed 6+-game suspension return; ``-1`` when
    the HOME team is; ``0`` otherwise. Population is measured to be
    extremely small by construction (small-n, per the task's own framing);
    recorded regardless of width.
    """

    _require_schedule_columns(schedule)
    reg = schedule.loc[schedule["game_type"].eq("REG")].copy()
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["week"] = pd.to_numeric(reg["week"], errors="raise").astype(int)
    reg["gameday_dt"] = pd.to_datetime(reg["gameday"], errors="raise")

    events = _suspension_events(transactions_index, snap_counts, schedule)
    qualifying_records: list[dict[str, object]] = []
    for _, event in events.iterrows():
        team = event["team"]
        report_end = _month_end_timestamp(event["report_year"], event["report_month"])
        team_games = reg.loc[
            ((reg["home_team"] == team) | (reg["away_team"] == team))
            & (reg["gameday_dt"] > report_end)
        ].sort_values("gameday_dt")
        for _, game in team_games.head(SUSPENSION_RETURN_GAMES).iterrows():
            qualifying_records.append(
                {"season": int(game["season"]), "week": int(game["week"]), "team": team}
            )

    qualifying = pd.DataFrame.from_records(qualifying_records, columns=["season", "week", "team"])
    return _attach_qualifying_sides(schedule, qualifying, SUSPENSION_RETURN_RUST_COLUMN)


def attach_suspension_return_rust_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    transactions_index: pd.DataFrame | None = None,
    snap_counts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Additively join ``suspension_return_rust_flag`` onto ``features`` by
    ``game_id``."""

    resolved_schedule = schedule if schedule is not None else default_schedule()
    resolved_transactions = (
        transactions_index if transactions_index is not None else default_transactions_index()
    )
    resolved_snaps = snap_counts if snap_counts is not None else default_snap_counts()
    derived = derive_suspension_return_rust_features(
        resolved_schedule, resolved_transactions, resolved_snaps
    )
    return _attach(features, derived, SUSPENSION_RETURN_RUST_COLUMN)


__all__ = [
    "ACQUISITION_RE",
    "DEADLINE_INTEGRATION_DRAG_COLUMN",
    "DEADLINE_INTEGRATION_GAMES",
    "DEADLINE_WINDOW_MONTHS",
    "DEFAULT_PLAYERS_RAW_ROOT",
    "DRAFT_PICK_RE",
    "HIGH_SNAP_SHARE_THRESHOLD",
    "HOLDOUT_END_RE",
    "HOLDOUT_SLOW_START_COLUMN",
    "REINSTATED_RE",
    "RETROSPECTIVE_SLUG_MARKERS",
    "SUSPENSION_MIN_GAMES",
    "SUSPENSION_RETURN_GAMES",
    "SUSPENSION_RETURN_RUST_COLUMN",
    "attach_deadline_integration_drag_features",
    "attach_holdout_slow_start_features",
    "attach_suspension_return_rust_features",
    "confirmed_acquisition_transactions",
    "default_schedule",
    "default_snap_counts",
    "default_transactions_index",
    "derive_deadline_integration_drag_features",
    "derive_holdout_slow_start_features",
    "derive_suspension_return_rust_features",
    "describe_deadline_acquisition_population",
    "describe_holdout_population",
    "describe_suspension_return_population",
    "distinct_player_slugs",
    "find_player_in_segment",
    "holdout_ending_transactions",
    "latest_pfr_transactions_snapshot",
    "suspension_category_transactions",
]
