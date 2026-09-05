"""Command-line entry point for the NFL ATS research pipeline.

This module owns three things and nothing else: the parser skeleton, the
dispatch in :func:`main`, and a small set of backwards-compatible re-exports
for callers that already reach into ``nfl_ats.cli`` by name.

Everything a command actually *does* lives in :mod:`nfl_ats.cli_commands` (one
module per domain, each exposing registrars with the uniform signature
``(subparsers, current_year) -> None``); the helpers shared by more than one
domain live in :mod:`nfl_ats.cli_common`. See ``docs/cli_architecture.md``.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from nfl_ats.cli_commands import REGISTRARS
from nfl_ats.cli_commands.cfb import _load_cfb_role_inputs
from nfl_ats.cli_commands.market import _cmd_market_open_close_backfill
from nfl_ats.cli_commands.prediction import _latest_margin_prediction_dir
from nfl_ats.cli_commands.prospective import _prospective_primary_entrants
from nfl_ats.cli_commands.publishing import (
    PUBLISH_CHALLENGER_RESULT_KEYS,
    _cmd_publish_predictions,
    _cmd_refresh_picks,
    _write_public_site,
)
from nfl_ats.cli_common import (
    _artifacts_root,
    _data_root,
    _registry_root,
    _repo_root_on_path,
)

#: Names that existed on ``nfl_ats.cli`` before the ENG-10 split and are read by
#: tests or ``scripts/``. Listed so the linter keeps the re-exports. Note that
#: monkeypatching a re-export here does NOT affect the handler that uses it --
#: patch the owning module instead.
__all__ = [
    "PUBLISH_CHALLENGER_RESULT_KEYS",
    "_artifacts_root",
    "_cmd_market_open_close_backfill",
    "_cmd_publish_predictions",
    "_cmd_refresh_picks",
    "_data_root",
    "_latest_margin_prediction_dir",
    "_load_cfb_role_inputs",
    "_prospective_primary_entrants",
    "_registry_root",
    "_repo_root_on_path",
    "_write_public_site",
    "build_parser",
    "main",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the full ``nfl-ats`` parser by running every registrar in order.

    ``current_year`` is read once here and passed down so a single parser can
    never mix two calendar years across its clock-derived defaults.
    """

    current_year = datetime.now().year
    parser = argparse.ArgumentParser(
        prog="nfl-ats",
        description="Leak-safe NFL against-the-spread research pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for register in REGISTRARS:
        register(subparsers, current_year)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
