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
