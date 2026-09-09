"""The spreads the pool actually grades: Splash Sports board capture and contract.

The owner's pool is graded on
the spreads printed on the Splash Sports contest board, which lock Tuesday at
noon Eastern. Those numbers are not the same as either market source this
repository already stores:

- ``schedules.spread_line`` (nflverse) is a *closing* proxy, published after
  the fact, and drifts away from the pool's frozen Tuesday number all week.
- The Odds API consensus (``nfl_ats.market_data``) is a book consensus, useful
  as research signal, but no book is the grader here.

Measured 2026-09-08 on the Week 1 board: the published card sat on a different
number than Splash for 4 of 16 games, and the card the noon lock regenerated
was on a different number for 8 of 16. So this module exists to make the
graded line a first-class, validated input rather than a proxy.

**Every Splash line is a half point.** Observed directly on all 16 Week 1
games. A half-point line cannot push, so this pool has no push branch at all --
which is why a whole-number line is treated here as a contract violation
rather than as data: it would silently re-enable push handling that has never
applied to this pool. See :func:`is_half_point` and
:func:`validate_splash_capture`.

Sign convention, repository-wide (``docs/bye_overvaluation_screen.md``):
``home_spread`` positive means the HOME team is favored. Splash displays the
same fact as two sides ("NE +3.5" / "SEA -3.5"); ``home_spread`` is the home
side's printed number negated.

This module is the ingestion layer only. It reads, parses and validates
captures; it does not decide anything and is not wired into the card, the
forecast or the published board.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.market_data import NFL_TEAM_NAMES, POOL_TIMEZONE

SPLASH_SUBDIRECTORY = "splash"

SPLASH_SOURCE = "splashsports.com"

SPLASH_CONVENTION = "home_spread positive = HOME favored (nflverse convention)"

SPLASH_HALF_POINT_NOTE = "Every quoted number is a half point. No push is possible in this pool."

SPLASH_CONTEST_NAME = "FTPL - Your favorite pick 'em league!"
SPLASH_CONTEST_CHANNEL: str | None = None
SPLASH_CONTEST_FORMAT = "NFL Pick'Em, Winner (ATS)"

SPLASH_BOARD_CYCLE = timedelta(days=7)

MAX_ABS_SPREAD = 40.0

NFLVERSE_TEAM_ABBREVIATIONS = frozenset(NFL_TEAM_NAMES.values())

SPLASH_TEAM_ALIASES: dict[str, str] = {
    **TEAM_ABBREVIATION_ALIASES,
    "JAC": "JAX",
    "LAR": "LA",
    "LVR": "LV",
    "WSH": "WAS",
    "GNB": "GB",
    "KAN": "KC",
    "NWE": "NE",
    "NOR": "NO",
    "SFO": "SF",
    "TAM": "TB",
    "TBB": "TB",
}

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

_WHEN_PATTERN = re.compile(
    r"^(?:(?P<weekday>Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\.?,?\s+)?"
    r"(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+"
    r"(?P<day>\d{1,2}),?\s+"
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<meridiem>[AaPp]\.?[Mm]\.?)"
    r"(?:\s*(?:ET|EST|EDT))?$"
)

_MATCHUP_PATTERN = re.compile(r"^(?P<away>[A-Z]{2,4})\s+(?P<when>.+?)\s+(?P<home>[A-Z]{2,4})$")

_OPTION_PATTERN = re.compile(
    r"^(?P<nickname>.*?)\s*(?P<team>[A-Z]{2,4})\s+(?P<sign>[+-])(?P<value>\d+(?:\.\d+)?)$"
)


def is_half_point(value: float) -> bool:
    """True when ``value`` is a genuine half point (``x.5``), not a whole number.

    Two conditions, both required: the value doubles onto an integer, and it is
    not itself within 0.4 of an integer. The second is what rejects a whole
    number that arrived as ``3.0``.
    """

    return abs(value * 2 - round(value * 2)) < 1e-9 and abs(value - round(value)) > 0.4


def normalize_team(abbreviation: str, *, context: str) -> str:
    """Fold a board abbreviation onto its nflverse identity, or refuse it."""

    token = abbreviation.strip().upper()
    resolved = SPLASH_TEAM_ALIASES.get(token, token)
    if resolved not in NFLVERSE_TEAM_ABBREVIATIONS:
        raise DataContractError(
            f"{context}: '{abbreviation}' is not an NFL team abbreviation this repository "
            "recognises. Add it to SPLASH_TEAM_ALIASES only after confirming the mapping on "
            "the board; a guessed abbreviation puts a line on the wrong game."
        )
    return resolved


def build_game_id(season: int, week: int, away: str, home: str) -> str:
    """nflverse game id: ``{season}_{week:02d}_{AWAY}_{HOME}``."""

    return f"{season}_{week:02d}_{away}_{home}"


@dataclass(frozen=True)
class SplashGame:
    """One board matchup and the half-point line the pool grades it on."""

    game_id: str
    away: str
    home: str
    away_line: float
    home_spread: float
    kickoff_et: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "away": self.away,
            "home": self.home,
            "away_line": self.away_line,
            "home_spread": self.home_spread,
            "kickoff_et": self.kickoff_et.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, context: str) -> SplashGame:
        missing = sorted(
            {"game_id", "away", "home", "away_line", "home_spread", "kickoff_et"}.difference(
                payload
            )
        )
        if missing:
            raise DataContractError(
                f"{context}: game entry is missing required keys: {', '.join(missing)}"
            )
        return cls(
            game_id=_require_text(payload["game_id"], field="game_id", context=context),
            away=_require_text(payload["away"], field="away", context=context),
            home=_require_text(payload["home"], field="home", context=context),
            away_line=_require_number(payload["away_line"], field="away_line", context=context),
            home_spread=_require_number(
                payload["home_spread"], field="home_spread", context=context
            ),
            kickoff_et=_require_datetime(
                payload["kickoff_et"], field="kickoff_et", context=context
            ),
        )


@dataclass(frozen=True)
class SplashCapture:
    """One point-in-time read of the pool's contest board."""

    season: int
    week: int
    captured_at_et: datetime
    games: tuple[SplashGame, ...]
    source: str = SPLASH_SOURCE
    capture_method: str = "browser_read_by_agent"
    picks_lock_et: datetime | None = None
    convention: str = SPLASH_CONVENTION
    note: str = SPLASH_HALF_POINT_NOTE
    contest: Mapping[str, Any] | None = None
    tiebreaker: Mapping[str, Any] | None = None
    submitted_entry: Mapping[str, Any] | None = None
    path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """The on-disk schema, byte-compatible in content with the hand capture."""

        payload: dict[str, Any] = {
            "source": self.source,
            "capture_method": self.capture_method,
            "captured_at_et": self.captured_at_et.isoformat(),
            "season": self.season,
            "week": self.week,
            "convention": self.convention,
            "note": self.note,
            "games": [game.to_dict() for game in self.games],
        }
        if self.contest is not None:
            payload["contest"] = dict(self.contest)
        if self.picks_lock_et is not None:
            payload["picks_lock_et"] = self.picks_lock_et.isoformat()
        if self.tiebreaker is not None:
            payload["tiebreaker"] = dict(self.tiebreaker)
        if self.submitted_entry is not None:
            payload["submitted_entry"] = dict(self.submitted_entry)
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, context: str, path: Path | None = None
    ) -> SplashCapture:
        for key in ("season", "week", "captured_at_et", "games"):
            if key not in payload:
                raise DataContractError(f"{context}: capture is missing required key '{key}'")
        raw_games = payload["games"]
        if not isinstance(raw_games, Sequence) or isinstance(raw_games, str | bytes):
            raise DataContractError(f"{context}: 'games' must be a list of game objects")
        games = []
        for entry in raw_games:
            if not isinstance(entry, Mapping):
                raise DataContractError(f"{context}: 'games' must be a list of game objects")
            games.append(SplashGame.from_dict(entry, context=context))
        return cls(
            season=_require_int(payload["season"], field="season", context=context),
            week=_require_int(payload["week"], field="week", context=context),
            captured_at_et=_require_datetime(
                payload["captured_at_et"], field="captured_at_et", context=context
            ),
            games=tuple(games),
            source=str(payload.get("source", SPLASH_SOURCE)),
            capture_method=str(payload.get("capture_method", "browser_read_by_agent")),
            picks_lock_et=(
                _require_datetime(payload["picks_lock_et"], field="picks_lock_et", context=context)
                if payload.get("picks_lock_et") is not None
                else None
            ),
            convention=str(payload.get("convention", SPLASH_CONVENTION)),
            note=str(payload.get("note", SPLASH_HALF_POINT_NOTE)),
            contest=_optional_mapping(payload.get("contest"), field="contest", context=context),
            tiebreaker=_optional_mapping(
                payload.get("tiebreaker"), field="tiebreaker", context=context
            ),
            submitted_entry=_optional_mapping(
                payload.get("submitted_entry"), field="submitted_entry", context=context
            ),
            path=path,
        )


def _require_text(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataContractError(f"{context}: '{field}' must be a non-empty string, got {value!r}")
    return value.strip()


def _require_int(value: Any, *, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataContractError(f"{context}: '{field}' must be an integer, got {value!r}")
    return value


def _require_number(value: Any, *, field: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DataContractError(f"{context}: '{field}' must be a number, got {value!r}")
    return float(value)


def _require_datetime(value: Any, *, field: str, context: str) -> datetime:
    """Parse an ISO-8601 instant. A naive value takes the pool's Eastern clock.

    The field names carry the zone (``*_et``), so attaching Eastern to a naive
    string is reading the declared schema, not guessing at it.
    """

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise DataContractError(
                f"{context}: '{field}' is not an ISO-8601 timestamp: {value!r}"
            ) from error
    else:
        raise DataContractError(
            f"{context}: '{field}' must be an ISO-8601 timestamp string, got {value!r}"
        )
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=POOL_TIMEZONE)
    return parsed


def _optional_mapping(value: Any, *, field: str, context: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DataContractError(f"{context}: '{field}' must be an object, got {value!r}")
    return dict(value)


def validate_splash_capture(
    capture: SplashCapture,
    *,
    expected_season: int | None = None,
    expected_week: int | None = None,
    context: str | None = None,
) -> None:
    """Fail closed on anything that would put a wrong number on the card.

    Raises :class:`~nfl_ats.data.DataContractError` on the first violation of:

    1. the capture holds at least one game;
    2. the capture's season/week match what the caller asked for;
    3. every ``game_id`` is nflverse-shaped ``{season}_{week:02d}_{AWAY}_{HOME}``
       and agrees with that game's own ``away``/``home`` fields;
    4. ``game_id`` values are unique and no game has a team playing itself;
    5. both teams are recognised NFL abbreviations;
    6. every ``home_spread`` and ``away_line`` is a half point;
    7. ``away_line`` and ``home_spread`` agree (they are the same fact: the
       away side's printed number is exactly the home team's handicap);
    8. no magnitude beyond :data:`MAX_ABS_SPREAD`, which is a mis-read.
    """

    where = context or (str(capture.path) if capture.path is not None else "splash capture")

    if not capture.games:
        raise DataContractError(f"{where}: capture holds no games")

    if expected_season is not None and capture.season != expected_season:
        raise DataContractError(
            f"{where}: capture is for season {capture.season}, not the requested {expected_season}"
        )
    if expected_week is not None and capture.week != expected_week:
        raise DataContractError(
            f"{where}: capture is for week {capture.week}, not the requested {expected_week}"
        )
    if not 1 <= capture.week <= 22:
        raise DataContractError(f"{where}: week {capture.week} is outside the NFL season")

    seen: set[str] = set()
    for game in capture.games:
        away = normalize_team(game.away, context=where)
        home = normalize_team(game.home, context=where)
        if away == home:
            raise DataContractError(f"{where}: {game.game_id} lists the same team on both sides")

        expected_id = build_game_id(capture.season, capture.week, away, home)
        if game.game_id != expected_id:
            raise DataContractError(
                f"{where}: game_id '{game.game_id}' does not match the nflverse form for this "
                f"capture's season/week and teams (expected '{expected_id}')"
            )
        if game.game_id in seen:
            raise DataContractError(f"{where}: duplicate game_id '{game.game_id}'")
        seen.add(game.game_id)

        for field_name, value in (
            ("home_spread", game.home_spread),
            ("away_line", game.away_line),
        ):
            if not is_half_point(value):
                raise DataContractError(
                    f"{where}: {game.game_id} {field_name} is {value}, which is not a half "
                    "point. Every line this pool has ever posted is a half point, so no push "
                    "is possible; a whole-number line would silently re-enable push machinery "
                    "that cannot apply to this pool. Re-read the board and fix the capture."
                )
            if abs(value) > MAX_ABS_SPREAD:
                raise DataContractError(
                    f"{where}: {game.game_id} {field_name} is {value}, beyond the "
                    f"{MAX_ABS_SPREAD} sanity bound -- that is a mis-read of the board, "
                    "not a line."
                )

        if abs(game.away_line - game.home_spread) > 1e-9:
            raise DataContractError(
                f"{where}: {game.game_id} away_line {game.away_line} disagrees with home_spread "
                f"{game.home_spread}. The away side's printed number IS the home team's "
                "handicap under this repository's sign convention (home_spread positive = home "
                "favored), so the two must be equal."
            )


def splash_capture_paths(data_root: Path, season: int, week: int) -> list[Path]:
    """Every capture file on disk for one season/week, oldest filename first."""

    directory = Path(data_root) / SPLASH_SUBDIRECTORY
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"{season}_week{week:02d}_*.json"))


def read_splash_capture(
    path: Path,
    *,
    expected_season: int | None = None,
    expected_week: int | None = None,
) -> SplashCapture:
    """Read and validate one capture file. Raises on anything malformed."""

    context = str(path)
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DataContractError(f"{context}: file is not valid JSON ({error})") from error
    if not isinstance(raw, Mapping):
        raise DataContractError(f"{context}: capture must be a JSON object")
    capture = SplashCapture.from_dict(raw, context=context, path=Path(path))
    validate_splash_capture(
        capture,
        expected_season=expected_season,
        expected_week=expected_week,
        context=context,
    )
    return capture


def load_splash_capture(data_root: Path, season: int, week: int) -> SplashCapture | None:
    """Newest validated capture for ``season``/``week``, or ``None`` if there is none.

    ``data_root`` is the repository ``data/`` directory; captures live in
    ``data/splash/``. Absence is not an error -- most weeks in history have no
    capture, because the pool board was never recorded before 2026-09-08.

    A malformed capture IS an error, including a malformed older sibling: every
    candidate for the week is read and validated, then the one with the latest
    ``captured_at_et`` is returned (filename breaks a tie). A capture file that
    cannot be trusted is a defect to fix, not a file to skip past.
    """

    candidates = splash_capture_paths(data_root, season, week)
    if not candidates:
        return None
    captures = [
        read_splash_capture(path, expected_season=season, expected_week=week) for path in candidates
    ]
    captures.sort(key=lambda item: (item.captured_at_et, str(item.path)))
    return captures[-1]


def splash_decision_lines(capture: SplashCapture) -> dict[str, float]:
    """``game_id`` -> ``home_spread``: the number the pool grades on."""

    return {game.game_id: game.home_spread for game in capture.games}


def capture_age(capture: SplashCapture, as_of: datetime) -> timedelta:
    """How long before ``as_of`` the board was read. Negative if ``as_of`` precedes it."""

    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware; the pool's clock is Eastern")
    return as_of - capture.captured_at_et


def is_stale(
    capture: SplashCapture,
    as_of: datetime,
    *,
    max_age: timedelta = SPLASH_BOARD_CYCLE,
) -> bool:
    """True when the capture is older than one board cycle and so belongs to a past week.

    The pool replaces the board every Tuesday at noon ET, so a capture more
    than :data:`SPLASH_BOARD_CYCLE` old is a previous week's numbers no matter
    what week its filename claims. Callers that must not grade against a stale
    board refuse on this.
    """

    return capture_age(capture, as_of) > max_age


def picks_locked(capture: SplashCapture, as_of: datetime) -> bool | None:
    """Whether the pool's pick deadline has passed, or ``None`` if unrecorded."""

    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware; the pool's clock is Eastern")
    if capture.picks_lock_et is None:
        return None
    return as_of >= capture.picks_lock_et


def _parse_when(text: str, *, season: int, context: str) -> datetime | None:
    match = _WHEN_PATTERN.match(text.strip())
    if match is None:
        return None
    month = _MONTHS[match.group("month").lower()[:3]]
    day = int(match.group("day"))
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    meridiem = match.group("meridiem").replace(".", "").lower()
    if not 1 <= hour <= 12:
        raise DataContractError(f"{context}: '{text.strip()}' has an impossible clock hour")
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    year = season + 1 if month <= 2 else season
    try:
        kickoff = datetime(year, month, day, hour, minute, tzinfo=POOL_TIMEZONE)
    except ValueError as error:
        raise DataContractError(f"{context}: '{text.strip()}' is not a real date") from error
    weekday = match.group("weekday")
    if weekday is not None and kickoff.weekday() != _WEEKDAYS[weekday.lower()[:3]]:
        raise DataContractError(
            f"{context}: the board says {weekday} but {kickoff:%Y-%m-%d} is a "
            f"{kickoff:%A}. Either the season year is wrong for this board or the line was "
            "mis-read; refusing to guess."
        )
    return kickoff


def parse_splash_board(text: str, season: int, week: int) -> tuple[SplashGame, ...]:
    """Turn a browser read of the contest board into validated games.

    The board renders as repeated blocks::

        CHI   Sun, Sep 13 1:00 PM   CAR
        Winner (ATS)
        Bears      CHI -2.5
        Panthers   CAR +2.5

    A matchup header (away abbreviation, kickoff, home abbreviation) opens a
    block; the two ``TEAM -/+ N.5`` option rows inside it carry the line. Team
    nicknames vary and are ignored -- the abbreviations and the signed numbers
    drive everything. Any line that is neither a header nor an option is
    ignored as page chrome, but a block that does not yield exactly two
    consistent options raises: a silently mis-parsed line is worse than no
    capture at all.
    """

    context = f"splash board {season} week {week}"
    lines = [line.strip() for line in text.splitlines()]

    blocks: list[tuple[str, str, datetime, list[tuple[str, float]]]] = []
    current: tuple[str, str, datetime, list[tuple[str, float]]] | None = None

    for raw_line in lines:
        if not raw_line:
            continue

        matchup = _MATCHUP_PATTERN.match(raw_line)
        if matchup is not None:
            kickoff = _parse_when(matchup.group("when"), season=season, context=context)
            if kickoff is not None:
                away = normalize_team(matchup.group("away"), context=context)
                home = normalize_team(matchup.group("home"), context=context)
                current = (away, home, kickoff, [])
                blocks.append(current)
                continue

        option = _OPTION_PATTERN.match(raw_line)
        if option is not None:
            value = float(option.group("value"))
            if option.group("sign") == "-":
                value = -value
            team = normalize_team(option.group("team"), context=context)
            if current is None:
                raise DataContractError(
                    f"{context}: found the option row '{raw_line}' before any matchup header. "
                    "The board text is not in the expected block shape; refusing to guess "
                    "which game this line belongs to."
                )
            current[3].append((team, value))

    if not blocks:
        raise DataContractError(
            f"{context}: no matchup headers found. A header looks like "
            "'CHI   Sun, Sep 13 1:00 PM   CAR' (away abbreviation, kickoff, home "
            "abbreviation); nothing in this text matched."
        )

    games: list[SplashGame] = []
    for away, home, kickoff, options in blocks:
        label = f"{away}@{home}"
        if len(options) != 2:
            raise DataContractError(
                f"{context}: {label} has {len(options)} spread option(s), expected exactly 2 "
                "('TEAM -N.5' and 'TEAM +N.5'). A pick'em or otherwise unreadable block is a "
                "capture defect, not a line."
            )
        by_team = dict(options)
        if len(by_team) != 2 or set(by_team) != {away, home}:
            raise DataContractError(
                f"{context}: {label} option rows name {sorted(by_team)}, which does not match "
                "the matchup header."
            )
        away_option = by_team[away]
        home_option = by_team[home]
        if abs(away_option + home_option) > 1e-9:
            raise DataContractError(
                f"{context}: {label} option rows are {away_option:+} and {home_option:+}; the "
                "two sides of a spread must be exact opposites."
            )
        games.append(
            SplashGame(
                game_id=build_game_id(season, week, away, home),
                away=away,
                home=home,
                away_line=away_option,
                home_spread=-home_option,
                kickoff_et=kickoff,
            )
        )

    validate_splash_capture(
        SplashCapture(
            season=season,
            week=week,
            captured_at_et=datetime.now(tz=POOL_TIMEZONE),
            games=tuple(games),
        ),
        expected_season=season,
        expected_week=week,
        context=context,
    )
    return tuple(games)


def default_picks_lock_et(games: Iterable[SplashGame]) -> datetime | None:
    """The pool's pick deadline for a slate: Sunday 16:00 ET of that game week.

    Owner rule (2026-08-20, re-confirmed 2026-09-01): the per-game deadline is
    ``min(own kickoff, Sunday 16:00 ET)``, so the slate-level deadline printed
    on the board is that Sunday 4 PM. Derived from the earliest kickoff so a
    capture never has to restate it by hand.
    """

    kickoffs = sorted(game.kickoff_et for game in games)
    if not kickoffs:
        return None
    first = kickoffs[0].astimezone(POOL_TIMEZONE)
    days_ahead = (6 - first.weekday()) % 7
    sunday = (first + timedelta(days=days_ahead)).replace(
        hour=16, minute=0, second=0, microsecond=0
    )
    return sunday
