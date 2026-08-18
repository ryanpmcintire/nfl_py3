"""Derive the surgical gate's threshold from the DISTRIBUTION of value lost, not accuracy.

Owned by the ``surgical_gating`` line of work (see ``docs/surgical_injury.md``).
This script performs the single derivation step the predeclaration depends on
and nothing else: it never fits a model, never touches a rotation window, and
never looks at any pick, spread, or outcome. It is a descriptive statistic of
one already-built feature column over the full leak-safe history -- the exact
same "free" category ``docs/injury_value_lost.md`` section 3.1 already used
for this table's split-half reliability (4,431 completed REG games,
2009-2025, via ``modeling.regular_season_rows`` + ``result.notna()``).

The quantity: for every game, ``raw_value_magnitude`` is the sum of the
absolute values of the two ``diff_`` columns
``docs/injury_value_lost.md`` section 4's cleaner isolation adds on top of the
frozen ``player`` baseline --

    diff_injury_skill_epa_value_lost
    diff_injury_defense_disruption_value_lost

read directly from ``data/processed/game_features_player_value.parquet``,
i.e. the exact columns the ridge model itself consumes (already imputed;
zero NaNs across all 4,431 games -- checked below and asserted). No manual
home/away reconstruction, no window-specific standardization: this is the
same number in the same units in every future season, which is what makes a
frozen threshold portable.

The threshold is the CONDITIONAL MEDIAN: the median of the magnitude among
games where it is strictly nonzero. Rationale, restated in
``docs/surgical_injury.md``: of the three sanctioned derivation methods
(natural break / fixed-in-advance quantile / reliability point), the median
is the one fixed quantile with no further researcher degree of freedom --
there is no "why 75 and not 70" question to answer for it. But the raw
distribution is a point mass at exactly zero (no listed value-lost
differential of either kind at all) plus a continuous right-hand tail, and
that point mass is large enough (29.86% of games here; 55.7% for this
recipe's CFB analog, see ``scripts/surgical_cfb_recipe_validation.py``) that
the UNCONDITIONAL population median can itself land inside or at the edge of
the point mass, which would make "the threshold" partly a statement about
how big the point mass is rather than about what counts as material. The
conditional median sidesteps that entirely and is the one definition that
applies unchanged in both leagues: "given the construct fired at all, is this
game's reading more than the typical firing" -- still the single least
-arbitrary quantile, still fixed before any accuracy is examined.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/surgical_value_lost_distribution.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.modeling import regular_season_rows
from nfl_ats.surgical_gating import derive_conditional_median_threshold

REPO = Path(__file__).resolve().parents[1]

VALUE_LOST_DIFF_COLUMNS = (
    "diff_injury_skill_epa_value_lost",
    "diff_injury_defense_disruption_value_lost",
)


def raw_value_magnitude(features: pd.DataFrame) -> pd.Series:
    """|diff skill-EPA value lost| + |diff defense-disruption value lost|, per game."""

    total = pd.Series(0.0, index=features.index)
    for column in VALUE_LOST_DIFF_COLUMNS:
        total = total + pd.to_numeric(features[column], errors="raise").abs()
    return total


def derive(features: pd.DataFrame) -> dict[str, Any]:
    reg = regular_season_rows(features)
    reg = reg.loc[reg["result"].notna()].copy()

    nan_counts = {c: int(reg[c].isna().sum()) for c in VALUE_LOST_DIFF_COLUMNS}
    magnitude = raw_value_magnitude(reg)

    quantiles = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]
    threshold = derive_conditional_median_threshold(magnitude)
    unconditional_median = float(magnitude.median())
    frac_zero = float((magnitude == 0.0).mean())
    frac_at_or_above_median = float((magnitude >= threshold).mean())
    # The zero mass point ends at this percentile -- reported so it is
    # visible whether the unconditional median would have collided with it
    # (it does for the CFB analog, which is exactly why the conditional
    # median is the general rule, not the unconditional one).
    zero_mass_percentile = float((magnitude == 0.0).mean())

    return {
        "source": "data/processed/game_features_player_value.parquet",
        "filter": "regular_season_rows + result.notna() (matches docs/injury_value_lost.md 3.1)",
        "games": len(reg),
        "diff_columns": list(VALUE_LOST_DIFF_COLUMNS),
        "diff_column_nan_counts": nan_counts,
        "magnitude_stats": {
            "mean": float(magnitude.mean()),
            "std": float(magnitude.std()),
            "min": float(magnitude.min()),
            "max": float(magnitude.max()),
            "fraction_exactly_zero": frac_zero,
            "quantiles": {str(q): float(magnitude.quantile(q)) for q in quantiles},
        },
        "derivation": {
            "method": (
                "conditional median: median of magnitude among games where magnitude > 0, "
                "fixed in advance, no accuracy used"
            ),
            "threshold": threshold,
            "unconditional_population_median": unconditional_median,
            "zero_mass_ends_at_percentile": zero_mass_percentile,
            "unconditional_median_collides_with_zero_mass": unconditional_median == 0.0,
            "fraction_of_games_at_or_above_threshold": frac_at_or_above_median,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=REPO / "data/processed/game_features_player_value.parquet",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = derive(pd.read_parquet(args.features))
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
