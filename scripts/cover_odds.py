"""Cover-odds query: pick a hypothetical spread for a game on THIS WEEK'S
published card and print the model's read of the odds of covering.

Owner request, 2026-08-20 -- the command-line half of the same feature the
picks-page "spread explorer" widget serves. Both surfaces call the SAME
library (:mod:`nfl_ats.spread_explorer`) so a number printed here can never
disagree with the picks page for the same game/line: no separate model-
fitting logic lives in this script -- it reuses
``nfl_ats.spread_explorer.compute_spread_explorer_distribution`` (which
itself reuses ``nfl_ats.outcomes.fit_margin_models_for_week``, the same
public entry point the picks-page widget and the ``smooth_cdf_mapping``
prospective challenger already use), and it never touches ``margin.py``,
``pool.py``, or ``src/nfl_ats/cli.py`` (owned by another agent at the time
of writing).

Reads the SAME synchronized forecast/artifacts the published card uses --
``artifacts/active_ats_model.json`` -> its linked weekly forecast -- and
fails closed, with a clear message on stderr and a non-zero exit code,
rather than silently answering from a different forecast, whenever:

  * no synchronized active ATS model is available;
  * the active model's probability method is not "gaussian" (the MOD-08
    mapping this whole feature reads, docs/smooth_cdf_mapping.md -- an
    older/rolled-back configuration has no closed-form residual fit this
    tool's formula can honestly read);
  * ``--season``/``--week`` do not match the active model's own linked
    weekly forecast (this tool only ever answers from THE current
    published card, never an arbitrary past forecast someone might have
    lying around in ``artifacts/margin_predictions/``);
  * ``--game`` does not resolve to exactly one game on that card;
  * a refit of that game's model does not reproduce the published card's
    own ``home_cover_probability`` (the feature table drifted since the
    card was built).

``--spread`` is home-oriented, matching this project's ``spread_line``
convention throughout (a MORE POSITIVE value means the home team is a
BIGGER favorite -- see ``nfl_ats.public_board.spread_words``, reused here
for the same "TEAM -X.5" formatting the picks page shows). ``--side``
merely selects whose cover/push/no-cover numbers to headline; the queried
line itself is always given in home-oriented terms regardless.

Usage::

    ./.tools/uv.exe run --no-sync python scripts/cover_odds.py \\
        --season 2026 --week 1 --game "NE@SEA" --spread -3.5

    ./.tools/uv.exe run --no-sync python scripts/cover_odds.py \\
        --season 2026 --week 1 --game 2026_01_NE_SEA --spread 3.5 --side away --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.calibration import smoothed_home_cover_probability
from nfl_ats.data import DataContractError
from nfl_ats.public_board import spread_words
from nfl_ats.reporting import read_json
from nfl_ats.spread_explorer import (
    compute_spread_explorer_distribution,
    load_feature_table_for_forecast,
    spread_explorer_three_way,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class CoverOddsError(RuntimeError):
    """A clear, fail-closed error for this tool's own preconditions -- never
    a stack trace for something a caller can fix by re-checking their
    arguments or the state of the active model."""


def _default_artifacts_root() -> Path:
    # Same env var, same default as `nfl_ats.cli._artifacts_root` --
    # duplicated rather than imported because this script deliberately does
    # not import `nfl_ats.cli` (owned by another agent at the time of
    # writing; see the module docstring).
    return Path(os.environ.get("NFL_ATS_ARTIFACTS_DIR", "artifacts"))


def _default_data_root() -> Path:
    return Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))


# ---------------------------------------------------------------------------
# Fail-closed loading of the SAME synchronized forecast the published card
# uses. Mirrors `nfl_ats.public_board.load_public_board_artifacts`'s own
# validation chain closely (not imported, since that function's return type
# also loads the line-sweep/explanations this script does not need) so this
# tool can never answer from a forecast the public site itself would refuse
# to publish.
# ---------------------------------------------------------------------------


def load_active_forecast(
    artifacts_root: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], pd.DataFrame]:
    """Return (active manifest, forecast directory, forecast metadata,
    recommendations) for the CURRENT synchronized weekly forecast."""

    try:
        active = load_active_ats_model(artifacts_root)
    except ValueError as error:
        raise CoverOddsError(str(error)) from error
    if active is None:
        raise CoverOddsError(
            "No synchronized active ATS model is available "
            f"(checked {artifacts_root / 'active_ats_model.json'})."
        )
    forecast_directory = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast_directory is None:
        raise CoverOddsError("Active ATS model has no linked weekly forecast.")
    metadata_path = forecast_directory / "metadata.json"
    recommendations_path = forecast_directory / "recommendations.csv"
    if not metadata_path.is_file() or not recommendations_path.is_file():
        raise CoverOddsError(f"Linked weekly forecast is incomplete: {forecast_directory}")
    metadata = read_json(metadata_path)
    if metadata.get("active_model_id") != active.get("model_id"):
        raise CoverOddsError("Weekly forecast model ID does not match the active model.")
    if metadata.get("synchronization_status") != "SYNCHRONIZED":
        raise CoverOddsError("Weekly forecast is not synchronized with an evaluation.")

    predictions = pd.read_csv(recommendations_path)
    method = str(active.get("method"))
    if "method" in predictions.columns and not predictions["method"].eq(method).all():
        raise CoverOddsError(
            "Weekly recommendations contain a method other than the active method."
        )

    probability_method = str(active.get("probability_method", "ecdf"))
    if probability_method != "gaussian":
        raise CoverOddsError(
            f"The active model's probability method is {probability_method!r}, not "
            "'gaussian'. This tool reads the MOD-08 Gaussian mapping "
            "(docs/smooth_cdf_mapping.md) and refuses to invent a formula for a "
            "different active configuration -- see nfl_ats.spread_explorer's module "
            "docstring."
        )
    return active, forecast_directory, metadata, predictions


# ---------------------------------------------------------------------------
# --game resolution: an exact game_id, or a "matchup" -- team codes joined
# by any of @ / - _ or whitespace, in either order, or a single team code
# that plays exactly one game the requested week.
# ---------------------------------------------------------------------------


def _tokenize_game_query(query: str) -> list[str]:
    normalized = query.strip().upper()
    for literal in ("@", "/", "_", "-"):
        normalized = normalized.replace(literal, " ")
    normalized = normalized.replace(" VS ", " ").replace(" AT ", " ").replace(" V ", " ")
    return [token for token in normalized.split() if token]


def _available_games_text(predictions: pd.DataFrame) -> str:
    return "; ".join(
        f"{away} at {home} ({game_id})"
        for game_id, home, away in zip(
            predictions["game_id"], predictions["home_team"], predictions["away_team"], strict=True
        )
    )


def resolve_game(predictions: pd.DataFrame, query: str) -> pd.Series:
    """Resolve ``--game`` against this week's card, or raise ``CoverOddsError``
    naming every available game so the caller can fix their query without
    re-reading a CSV by hand."""

    exact = predictions.loc[predictions["game_id"].astype(str).str.upper() == query.strip().upper()]
    if len(exact) == 1:
        return exact.iloc[0]

    tokens = _tokenize_game_query(query)
    if not tokens:
        raise CoverOddsError(f"Could not parse --game {query!r}.")
    home = predictions["home_team"].astype(str).str.upper()
    away = predictions["away_team"].astype(str).str.upper()
    if len(tokens) >= 2:
        wanted = {tokens[0], tokens[1]}
        mask = [{h, a} == wanted for h, a in zip(home, away, strict=True)]
    else:
        token = tokens[0]
        mask = [(h == token) or (a == token) for h, a in zip(home, away, strict=True)]
    matches = predictions.loc[mask]
    if len(matches) == 1:
        return matches.iloc[0]

    available = _available_games_text(predictions)
    if matches.empty:
        raise CoverOddsError(
            f"No game on this week's card matches --game {query!r}. Available: {available}"
        )
    raise CoverOddsError(
        f"--game {query!r} matches more than one game this week. Available: {available}"
    )


# ---------------------------------------------------------------------------
# The query itself
# ---------------------------------------------------------------------------


def query_cover_odds(
    *,
    season: int,
    week: int,
    game: str,
    spread: float,
    side: str | None,
    artifacts_root: Path,
    data_root: Path,
) -> dict[str, Any]:
    """Everything printed by this tool, as one plain dict -- shared by the
    human-readable and ``--json`` output paths so they can never disagree."""

    active, forecast_directory, metadata, predictions = load_active_forecast(artifacts_root)

    active_season, active_week = metadata.get("season"), metadata.get("week")
    if (
        active_season is None
        or active_week is None
        or int(active_season) != season
        or int(active_week) != week
    ):
        raise CoverOddsError(
            f"--season {season} --week {week} does not match the active model's published "
            f"card ({active_season} week {active_week}, {forecast_directory.name}). This tool "
            "only ever reads the CURRENT published card, never an arbitrary past forecast."
        )

    row = resolve_game(predictions, game)
    game_id = str(row["game_id"])

    features = load_feature_table_for_forecast(metadata, data_root)
    distribution = compute_spread_explorer_distribution(
        predictions,
        features,
        game_id=game_id,
        regressor=str(metadata.get("regressor")),
        ridge_alpha=float(metadata.get("ridge_alpha", 10.0)),
        feature_profile=str(metadata.get("feature_profile")),
        min_train_games=int(metadata.get("min_train_games", 500)),
        probability_method="gaussian",
    )

    home_cover = float(
        smoothed_home_cover_probability(
            distribution.residuals,
            [distribution.center],
            [spread],
            method="gaussian",
        )[0]
    )
    home_excl_push, push, home_no_cover = spread_explorer_three_way(distribution, spread)

    headline_side = side if side is not None else ("home" if home_cover >= 0.5 else "away")
    if headline_side == "home":
        headline_team = distribution.home_team
        headline_cover_two_way = home_cover
        headline_cover_excl_push = home_excl_push
        headline_no_cover = home_no_cover
    else:
        headline_team = distribution.away_team
        headline_cover_two_way = 1.0 - home_cover
        headline_cover_excl_push = home_no_cover
        headline_no_cover = home_excl_push

    caveat = (
        "These odds reflect the model's market read and features as of the forecast created "
        f"at {metadata.get('created_at_utc')} UTC -- only the hypothetical spread queried above "
        "changes; this is not a live re-forecast and never uses information from after that "
        "build."
    )

    return {
        "game_id": game_id,
        "matchup": f"{distribution.away_team} at {distribution.home_team}",
        "season": season,
        "week": week,
        "queried_spread_home_oriented": spread,
        "queried_spread_words": spread_words(
            distribution.home_team, distribution.away_team, spread
        ),
        "side": headline_side,
        "side_team": headline_team,
        "cover_probability": round(headline_cover_excl_push, 6),
        "push_probability": round(push, 6),
        "no_cover_probability": round(headline_no_cover, 6),
        "two_way_forced_pick_probability": round(headline_cover_two_way, 6),
        "home_cover_probability": round(home_cover, 6),
        "away_cover_probability": round(1.0 - home_cover, 6),
        "provenance": {
            "model_id": active.get("model_id"),
            "probability_method": active.get("probability_method"),
            "market_line_the_model_was_anchored_to": distribution.card_line,
            "market_anchored_line_words": spread_words(
                distribution.home_team, distribution.away_team, distribution.card_line
            ),
            "published_home_cover_probability": round(distribution.card_home_cover_probability, 6),
            "forecast_created_at_utc": metadata.get("created_at_utc"),
            "forecast_artifact": forecast_directory.name,
            "regressor": metadata.get("regressor"),
            "ridge_alpha": metadata.get("ridge_alpha"),
            "feature_profile": metadata.get("feature_profile"),
        },
        "information_as_of_caveat": caveat,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def format_text(payload: dict[str, Any]) -> str:
    """Plain ASCII only (no smart quotes, middle dots, or arrows): this
    prints straight to a console, and Windows' legacy codepages mojibake
    anything outside ASCII rather than raising -- garbled but "successful"
    output is worse than none, so this avoids the character class entirely
    rather than hoping every terminal is UTF-8."""

    provenance = payload["provenance"]
    lines = [
        f"{payload['matchup']} -- {payload['season']} Week {payload['week']}",
        f"Queried spread: {payload['queried_spread_words']} "
        f"(home-oriented: {payload['queried_spread_home_oriented']:+g})",
        "",
        f"  {payload['side_team']} ({payload['side']}) covers:       "
        f"{payload['cover_probability']:.1%}",
        f"  Push:                              {payload['push_probability']:.1%}",
        f"  {payload['side_team']} does not cover:      {payload['no_cover_probability']:.1%}",
        "  (a discrete read of the raw residual sample -- the same three-way math "
        "production always uses, regardless of probability method)",
        "",
        "  Two-way forced-pick read (production rule; the Gaussian mapping this pool "
        "actually plays; does not split out push):",
        f"    home: {payload['home_cover_probability']:.1%}   "
        f"away: {payload['away_cover_probability']:.1%}",
        "",
        "Provenance:",
        f"  model_id: {provenance['model_id']}",
        f"  probability_method: {provenance['probability_method']}",
        f"  market line the model was anchored to: {provenance['market_anchored_line_words']} "
        f"(published home_cover_probability: {provenance['published_home_cover_probability']:.2%})",
        f"  forecast created_at (UTC): {provenance['forecast_created_at_utc']}",
        f"  forecast artifact: {provenance['forecast_artifact']}",
        f"  regressor / ridge_alpha / feature_profile: {provenance['regressor']} / "
        f"{provenance['ridge_alpha']} / {provenance['feature_profile']}",
        "",
        payload["information_as_of_caveat"],
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pick a hypothetical spread for a game on this week's published card "
        "and see the model's read of the odds of covering, as of the last synchronized "
        "weekly forecast.",
    )
    parser.add_argument("--season", type=int, required=True, help="e.g. 2026")
    parser.add_argument("--week", type=int, required=True, help="e.g. 1")
    parser.add_argument(
        "--game",
        type=str,
        required=True,
        help="a game_id (e.g. 2026_01_NE_SEA) or a matchup (e.g. NE@SEA, NE-SEA, or just SEA)",
    )
    parser.add_argument(
        "--spread",
        type=float,
        required=True,
        help="hypothetical home-oriented spread to query, e.g. -3.5 (positive = home favored)",
    )
    parser.add_argument(
        "--side",
        choices=("home", "away"),
        default=None,
        help="which side's cover/push/no-cover numbers to headline "
        "(default: whichever side the queried spread currently favors)",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=None,
        help="defaults to $NFL_ATS_ARTIFACTS_DIR or ./artifacts",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="defaults to $NFL_ATS_DATA_DIR or ./data",
    )
    parser.add_argument(
        "--json", action="store_true", help="print machine-readable JSON instead of text"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    artifacts_root = (
        args.artifacts_root if args.artifacts_root is not None else _default_artifacts_root()
    )
    data_root = args.data_root if args.data_root is not None else _default_data_root()

    try:
        payload = query_cover_odds(
            season=args.season,
            week=args.week,
            game=args.game,
            spread=args.spread,
            side=args.side,
            artifacts_root=artifacts_root,
            data_root=data_root,
        )
    except (CoverOddsError, DataContractError, ValueError) as error:
        print(f"cover_odds: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=False))
    else:
        print(format_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
