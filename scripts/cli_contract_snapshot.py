"""Dump a normalised JSON snapshot of the ``nfl-ats`` argparse contract.

The snapshot is the oracle for CLI refactors (ENG-10): it records every command
path, its help text, every argument's option strings/dest/default/required/
choices/type/nargs/help, and each handler's module and qualname. Two snapshots
that compare equal describe argparse trees that behave identically from the
outside.

Usage::

    uv run python scripts/cli_contract_snapshot.py OUTPUT.json [--normalize-years]

``--normalize-years`` replaces integer defaults derived from ``datetime.now()``
(the current year and the previous year) with stable tokens so a tracked
fixture does not rot at the turn of a calendar year.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

CURRENT_YEAR_TOKEN = "<CURRENT_YEAR>"
PREVIOUS_YEAR_TOKEN = "<CURRENT_YEAR-1>"


def _describe_callable(value: Any) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return repr(value)


def _jsonify(value: Any) -> Any:
    """Render an argparse attribute as a stable, JSON-serialisable value."""

    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return {"__type__": "Path", "value": value.as_posix()}
    if isinstance(value, list | tuple):
        return {
            "__type__": type(value).__name__,
            "items": [_jsonify(item) for item in value],
        }
    if isinstance(value, set | frozenset):
        return {
            "__type__": type(value).__name__,
            "items": sorted(repr(item) for item in value),
        }
    if isinstance(value, dict):
        return {
            "__type__": "dict",
            "items": [[str(key), _jsonify(item)] for key, item in value.items()],
        }
    if callable(value):
        return {"__type__": "callable", "value": _describe_callable(value)}
    return {"__type__": type(value).__name__, "repr": repr(value)}


def _action_payload(action: argparse.Action) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action_class": type(action).__name__,
        "option_strings": list(action.option_strings),
        "dest": action.dest,
        "nargs": _jsonify(action.nargs),
        "const": _jsonify(action.const),
        "default": _jsonify(action.default),
        "type": _jsonify(action.type),
        "choices": _jsonify(action.choices) if not isinstance(action.choices, dict) else None,
        "required": bool(action.required),
        "help": action.help,
        "metavar": _jsonify(action.metavar),
    }
    version = getattr(action, "version", None)
    if version is not None:
        payload["version"] = version
    return payload


def _subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _choice_help(action: argparse._SubParsersAction) -> list[dict[str, Any]]:
    return [
        {
            "name": choice.metavar or choice.dest,
            "help": choice.help,
        }
        for choice in action._choices_actions
    ]


def _parser_payload(parser: argparse.ArgumentParser, path: list[str]) -> dict[str, Any]:
    handler = parser._defaults.get("handler")
    defaults = {
        key: _jsonify(value) for key, value in sorted(parser._defaults.items()) if key != "handler"
    }
    payload: dict[str, Any] = {
        "path": list(path),
        "prog": parser.prog,
        "usage": parser.usage,
        "description": parser.description,
        "epilog": parser.epilog,
        "formatter_class": parser.formatter_class.__name__,
        "prefix_chars": parser.prefix_chars,
        "add_help": parser.add_help,
        "allow_abbrev": parser.allow_abbrev,
        "argument_default": _jsonify(parser.argument_default),
        "conflict_handler": parser.conflict_handler,
        "handler": _describe_callable(handler) if handler is not None else None,
        "other_defaults": defaults,
        "actions": [
            _action_payload(action)
            for action in parser._actions
            if not isinstance(action, argparse._SubParsersAction)
        ],
    }

    sub_action = _subparsers_action(parser)
    if sub_action is None:
        payload["subcommands"] = None
        return payload

    payload["subcommands"] = {
        "dest": sub_action.dest,
        "required": bool(sub_action.required),
        "title": getattr(sub_action, "title", None),
        "metavar": _jsonify(sub_action.metavar),
        "help": sub_action.help,
        "order": list(sub_action.choices),
        "choice_help": _choice_help(sub_action),
        "parsers": [
            _parser_payload(child, [*path, name]) for name, child in sub_action.choices.items()
        ],
    }
    return payload


def snapshot(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Return the normalised contract payload for ``parser`` and its subparsers."""

    return {"schema": 1, "root": _parser_payload(parser, [])}


def _normalize_years_value(value: Any, current_year: int) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == current_year:
            return CURRENT_YEAR_TOKEN
        if value == current_year - 1:
            return PREVIOUS_YEAR_TOKEN
        return value
    if isinstance(value, dict):
        return {key: _normalize_years_value(item, current_year) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_years_value(item, current_year) for item in value]
    return value


def normalize_years(payload: Any, current_year: int | None = None) -> Any:
    """Replace clock-derived year defaults with stable tokens."""

    year = datetime.now().year if current_year is None else current_year
    return _normalize_years_value(payload, year)


def dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="path to write the JSON snapshot to")
    parser.add_argument(
        "--normalize-years",
        action="store_true",
        help="replace current-year-derived integer defaults with stable tokens",
    )
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    from nfl_ats.cli import build_parser

    payload = snapshot(build_parser())
    if args.normalize_years:
        payload = normalize_years(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dumps(payload), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
