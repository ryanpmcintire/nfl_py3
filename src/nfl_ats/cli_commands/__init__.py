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
    operations.register_waterfall_feed,
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
