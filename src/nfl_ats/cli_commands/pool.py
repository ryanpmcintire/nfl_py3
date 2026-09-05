"""Pool-facing commands: tiebreaker, pool observables and totals backtest."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from nfl_ats.cli_common import (
    _add_bootstrap_args,
    _add_features_arg,
    _artifacts_root,
    _data_root,
    _print_json,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.tiebreaker import format_report as format_tiebreaker_report
from nfl_ats.tiebreaker import tiebreaker_report
from nfl_ats.totals import format_results as format_totals_results
from nfl_ats.totals import run_backtest as run_totals_backtest


def _cmd_tiebreaker(args: argparse.Namespace) -> None:
    report = tiebreaker_report(
        _data_root(),
        artifacts_root=_artifacts_root(),
        season=args.season,
        week=args.week,
        game_id=args.game_id,
    )
    print(format_tiebreaker_report(report))


def _cmd_pool_observables(args: argparse.Namespace) -> None:
    from nfl_ats.pool_observables import (
        DistributionObservation,
        FieldObservation,
        record_distribution,
        record_field_observation,
    )

    observed_at = args.observed_at or datetime.now(UTC).isoformat()
    if args.distribution:
        game_id, _, shares = args.distribution.partition("=")
        home_raw, _, away_raw = shares.partition(",")
        result = record_distribution(
            _data_root(),
            DistributionObservation(
                season=args.season,
                week=args.week,
                game_id=game_id,
                home_share=float(home_raw),
                away_share=float(away_raw),
                unlocked_at_utc=args.unlocked_at or "",
                observed_at_utc=observed_at,
                observer=args.observer or "",
            ),
        )
    else:
        result = record_field_observation(
            _data_root(),
            FieldObservation(
                season=args.season,
                week=args.week,
                entries=args.entries or 0,
                paid_places=args.paid_places or 0,
                prize_notes=args.prize_notes or "",
                observed_at_utc=observed_at,
                observer=args.observer or "",
            ),
        )
    _print_json(result)


def _cmd_totals_backtest(args: argparse.Namespace) -> None:
    results = run_totals_backtest(
        _data_root(),
        args.features,
        _artifacts_root(),
        min_train_games=args.min_train_games,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(format_totals_results(results))


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:
    """Register the pool/totals commands."""

    tiebreaker = subparsers.add_parser(
        "tiebreaker",
        help="final-score guess for the pool's tiebreaker game (the week's last kickoff)",
    )
    tiebreaker.add_argument("--season", type=int, help="default: the next upcoming week's season")
    tiebreaker.add_argument("--week", type=int, help="default: the next upcoming week")
    tiebreaker.add_argument(
        "--game-id", dest="game_id", help="explicit nflverse game id, overrides season/week"
    )
    tiebreaker.set_defaults(handler=_cmd_tiebreaker)

    pool_observables = subparsers.add_parser(
        "pool-observables",
        help=(
            "record the pool's field size, prize structure, or an unlocked "
            "pick distribution (manual entry, LEAD-52)"
        ),
    )
    pool_observables.add_argument("--season", type=int, required=True)
    pool_observables.add_argument("--week", type=int, required=True)
    pool_observables.add_argument("--entries", type=int, help="field size (field record)")
    pool_observables.add_argument("--paid-places", type=int, help="paid places (field record)")
    pool_observables.add_argument("--prize-notes", help="what the payout is (field record)")
    pool_observables.add_argument(
        "--distribution",
        help="GAME_ID=home_share,away_share, e.g. 2026_01_MIA_LV=0.62,0.38",
    )
    pool_observables.add_argument("--unlocked-at", help="ISO-8601 kickoff/unlock instant")
    pool_observables.add_argument("--observed-at", help="ISO-8601 read instant (default: now)")
    pool_observables.add_argument("--observer", help="who read the pool page")
    pool_observables.set_defaults(handler=_cmd_pool_observables)

    totals_backtest = subparsers.add_parser(
        "totals-backtest",
        help="walk-forward over/under regime: ridge on the market total's residual",
    )
    _add_features_arg(totals_backtest)
    totals_backtest.add_argument(
        "--min-train-games",
        dest="min_train_games",
        type=int,
        default=DEFAULT_MIN_TRAIN_GAMES,
        help="warm-up floor before a week is scored (default: production's constant)",
    )
    _add_bootstrap_args(totals_backtest, seed=20260901)
    totals_backtest.set_defaults(handler=_cmd_totals_backtest)
