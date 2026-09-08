"""Turn a read of the Splash Sports contest board into a validated capture file.

The pool grades on the spreads printed on the Splash board, which lock Tuesday
at noon ET. Until 2026-09-08 those numbers were never recorded and the card
used nflverse's closing ``spread_line`` as a stand-in; see
``docs/splash_lines.md`` for what that cost.

Taking a capture is a two-step job: a human (or a browser agent) copies the
board text off the page, and this script does the rest -- parse, validate,
write. Nothing here scrapes the site.

Usage::

    # from a file
    uv run --no-sync python scripts/capture_splash_lines.py \
        --season 2026 --week 2 --text-file board.txt --dry

    # from stdin
    Get-Content board.txt | uv run --no-sync python scripts/capture_splash_lines.py \
        --season 2026 --week 2

The board text is the repeated block shape the page renders::

    CHI   Sun, Sep 13 1:00 PM   CAR
    Winner (ATS)
    Bears      CHI -2.5
    Panthers   CAR +2.5

Every line is validated before anything is written (half points only, nflverse
game ids, both sides of each spread exact opposites). A capture that fails
validation is not written at all -- a silently wrong number on the card is the
failure this whole path exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.io import atomic_json  # noqa: E402
from nfl_ats.market_data import POOL_TIMEZONE  # noqa: E402
from nfl_ats.splash_lines import (  # noqa: E402
    SPLASH_CONTEST_CHANNEL,
    SPLASH_CONTEST_FORMAT,
    SPLASH_CONTEST_NAME,
    SPLASH_SUBDIRECTORY,
    SplashCapture,
    default_picks_lock_et,
    parse_splash_board,
    splash_capture_paths,
    validate_splash_capture,
)


def capture_label(captured_at: datetime) -> str:
    """``noon`` for a capture taken in the pool's noon lock hour, else ``HHMM``."""

    if captured_at.hour == 12:
        return "noon"
    return f"{captured_at:%H%M}"


def capture_filename(season: int, week: int, captured_at: datetime, label: str) -> str:
    return f"{season}_week{week:02d}_{captured_at:%Y%m%d}_{label}.json"


def build_capture(
    text: str,
    *,
    season: int,
    week: int,
    captured_at: datetime,
    capture_method: str,
    contest: dict[str, Any],
    picks_lock: datetime | None,
    extra: dict[str, Any],
) -> SplashCapture:
    games = parse_splash_board(text, season, week)
    capture = SplashCapture(
        season=season,
        week=week,
        captured_at_et=captured_at,
        games=games,
        capture_method=capture_method,
        picks_lock_et=picks_lock if picks_lock is not None else default_picks_lock_et(games),
        contest=contest or None,
        tiebreaker=extra.get("tiebreaker"),
        submitted_entry=extra.get("submitted_entry"),
    )
    validate_splash_capture(
        capture,
        expected_season=season,
        expected_week=week,
        context=f"splash board {season} week {week}",
    )
    return capture


def _parse_et(value: str, *, flag: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise SystemExit(f"{flag} is not an ISO-8601 timestamp: {value!r}") from error
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=POOL_TIMEZONE)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--text-file",
        type=Path,
        help="File holding the copied board text. Defaults to reading stdin.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO / "data",
        help=f"Repository data root; captures land in <data-root>/{SPLASH_SUBDIRECTORY}/.",
    )
    parser.add_argument(
        "--captured-at",
        help="ISO-8601 capture instant, Eastern if no offset is given. Defaults to now.",
    )
    parser.add_argument(
        "--label",
        help="Filename label. Defaults to 'noon' inside the pool's noon lock hour, else HHMM.",
    )
    parser.add_argument(
        "--capture-method",
        default="browser_read_by_agent",
        help="How the board text was obtained; recorded as provenance.",
    )
    parser.add_argument("--contest-name", default=SPLASH_CONTEST_NAME)
    parser.add_argument("--channel", default=SPLASH_CONTEST_CHANNEL)
    parser.add_argument("--contest-format", default=SPLASH_CONTEST_FORMAT)
    parser.add_argument("--contest-id", help="Per-week contest identifier from the board URL.")
    parser.add_argument("--slate-id", help="Per-week slate identifier from the board URL.")
    parser.add_argument("--entries", type=int, help="Entry count shown on the contest page.")
    parser.add_argument(
        "--picks-lock",
        help=(
            "ISO-8601 pick deadline, Eastern if no offset is given. Defaults to the Sunday "
            "16:00 ET of the slate's game week."
        ),
    )
    parser.add_argument(
        "--extra-json",
        type=Path,
        help=(
            "JSON file whose top-level 'tiebreaker' and 'submitted_entry' objects are merged "
            "into the capture. Those are hand-entered, never parsed from the board."
        ),
    )
    parser.add_argument("--dry", action="store_true", help="Print the capture; write nothing.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Allow writing when a capture for this season/week already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.text_file is not None:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    if not text.strip():
        print(
            "no board text supplied (use --text-file or pipe the board on stdin)", file=sys.stderr
        )
        return 2

    captured_at = (
        _parse_et(args.captured_at, flag="--captured-at")
        if args.captured_at
        else datetime.now(tz=POOL_TIMEZONE)
    )
    picks_lock = _parse_et(args.picks_lock, flag="--picks-lock") if args.picks_lock else None

    extra: dict[str, Any] = {}
    if args.extra_json is not None:
        loaded = json.loads(Path(args.extra_json).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            print(
                f"--extra-json must hold a JSON object, got {type(loaded).__name__}",
                file=sys.stderr,
            )
            return 2
        extra = loaded

    contest: dict[str, Any] = {
        "name": args.contest_name,
        "channel": args.channel,
        "format": args.contest_format,
    }
    for key, value in (
        ("contest_id", args.contest_id),
        ("slate_id", args.slate_id),
        ("entries", args.entries),
    ):
        if value is not None:
            contest[key] = value

    try:
        capture = build_capture(
            text,
            season=args.season,
            week=args.week,
            captured_at=captured_at,
            capture_method=args.capture_method,
            contest=contest,
            picks_lock=picks_lock,
            extra=extra,
        )
    except DataContractError as error:
        print(f"refusing to write a capture: {error}", file=sys.stderr)
        return 2

    payload = capture.to_dict()

    # Read back what we are about to write, through the same loader the card
    # side will use. A capture that does not survive its own round trip is a
    # defect, and finding it here costs nothing.
    reread = SplashCapture.from_dict(payload, context="round-trip check")
    validate_splash_capture(
        reread,
        expected_season=args.season,
        expected_week=args.week,
        context="round-trip check",
    )

    label = args.label or capture_label(captured_at)
    directory = Path(args.data_root) / SPLASH_SUBDIRECTORY
    destination = directory / capture_filename(args.season, args.week, captured_at, label)

    existing = splash_capture_paths(Path(args.data_root), args.season, args.week)
    if existing and not args.replace:
        print(
            f"a capture for {args.season} week {args.week} already exists: "
            f"{', '.join(path.name for path in existing)}. Pass --replace to write anyway.",
            file=sys.stderr,
        )
        return 2

    if args.dry:
        print(json.dumps(payload, indent=2, sort_keys=True))
        print(f"[dry] would write {destination}", file=sys.stderr)
        return 0

    atomic_json(payload, destination)
    print(f"wrote {destination} ({len(capture.games)} games)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
