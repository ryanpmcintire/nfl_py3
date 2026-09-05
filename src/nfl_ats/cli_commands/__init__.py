"""Per-domain ``nfl-ats`` subcommand modules.

Each module owns the ``add_parser`` blocks and the ``_cmd_*`` handlers for one
domain and exposes one or more registrar callables with the uniform signature
``(subparsers, current_year) -> None``.

``nfl_ats.cli.build_parser`` walks :data:`REGISTRARS` in order and calls each
entry once. **That order is the ``nfl-ats --help`` listing order**, so it is
part of the CLI contract: ``tests/test_cli_contract.py`` pins it against
``tests/fixtures/cli_contract.json``. A domain whose commands are not
contiguous in that historical order exposes one registrar per contiguous run
(``market.register_odds`` and ``market.register_backfill``, for example) rather
than being reordered.

``current_year`` is threaded through instead of each registrar reading the
clock so that one ``build_parser()`` call can never mix two calendar years in
its defaults.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from nfl_ats.cli_commands import (
    cfb,
    clv,
    data,
    evaluation,
    features,
    market,
    operations,
    pool,
    prediction,
    prospective,
    publishing,
    registry,
)

Registrar = Callable[["argparse._SubParsersAction[argparse.ArgumentParser]", int], None]

#: Registration order == ``nfl-ats --help`` order. Append-only in spirit:
#: reordering entries renames nothing but does reorder the help listing, which
#: the contract fixture treats as a behaviour change.
REGISTRARS: tuple[Registrar, ...] = (
    operations.register_health,
    data.register_player_arrests,
    publishing.register,
    operations.register_handoff,
    data.register,
    cfb.register,
    market.register_odds,
    pool.register,
    market.register_backfill,
    clv.register_scoring,
    prospective.register,
    clv.register_diagnostics,
    features.register,
    evaluation.register,
    prediction.register,
    registry.register,
    evaluation.register_anytime,
    operations.register_weekly,
)

__all__ = [
    "REGISTRARS",
    "Registrar",
    "cfb",
    "clv",
    "data",
    "evaluation",
    "features",
    "market",
    "operations",
    "pool",
    "prediction",
    "prospective",
    "publishing",
    "registry",
]
