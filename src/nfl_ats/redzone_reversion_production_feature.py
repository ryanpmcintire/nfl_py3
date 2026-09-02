"""Third-down mean-reversion fade indicator, stacked on PRODUCTION.

``docs/redzone_reversion_screen.md`` predeclared and froze six cells of a
red-zone / third-down mean-reversion battery against a BARE market baseline.
Its leading cell, ``redzone_reversion_c3_third_down_over_fade`` (+0.3665
accuracy points, week-blocked 95% [-0.2587, +0.9990], ``probability_positive``
0.8719), carries both the highest lean in the battery and its highest
split-half reliability (0.407, the year-over-year Pearson of the centred
trait). The project's own recorded lesson ("composition is not the signal") is
that a component positive alone can go negative once stacked on the chain that
is actually PLAYED. This module builds that same fade indicator as a single
game-level feature column, additively joined onto a feature table by
``game_id``, so ``docs/redzone_reversion_on_production.md`` can measure it on
top of PRODUCTION ``weak_stack`` instead of a bare baseline.

**Reuses the frozen panel construction verbatim, does not rebuild it**:
``build_efficiency_panels``, ``OFF_TRAITS``, ``_alias_team`` and ``_prior`` are
imported directly from ``scripts/redzone_reversion_screen.py`` (read-only,
never edited), which in turn reads the local play-by-play snapshot through
``nfl_ats.pbp.latest_pbp_snapshot`` / ``load_pbp_snapshot`` /
``analysis_plays`` -- the house v1 efficiency filter, REG plays only, franchise
aliases applied via ``nfl_ats.constants.TEAM_ABBREVIATION_ALIASES``.
``scripts`` is not part of the installed package, so this module puts the
repository root on ``sys.path`` the same guarded way
``nfl_ats.fluview_production_feature`` and ``nfl_ats.illness_production_feature``
already do for the same reason.

**Two deliberate deviations from the registered cell, both declared in
``docs/redzone_reversion_on_production.md`` section 2 BEFORE scoring:**

1. *Team-level cell to game-level column.* The registered cell is a team-game
   flag on a long table of 8,634 team-games; a feature column on the
   game-level production table must be game-level, so the two team flags
   combine into one signed difference,
   ``int(home flagged) - int(away flagged)`` in {-1, 0, +1}. C3's predicted
   sign is -1 (fade the over-performer), so a home over-performer to fade
   pushes the column positive and an away over-performer negative; the
   ordering of the three states is the construct's own. Both-flagged and
   neither-flagged collapse to 0, which is correct for a fade-the-extreme
   construct and is disclosed as a real loss of information that can only
   attenuate toward the null.
2. *Expanding, strictly-prior threshold.* The frozen screen computes its
   ``third_down_q75`` cut over the WHOLE 2009-2025 panel (a mild look-ahead: a
   2012 game flagged with a cut estimated partly from 2020 data). A pregame
   feature column may not carry that, so the threshold here is recomputed
   expanding over STRICTLY PRIOR seasons only -- for a game in season S, the
   75th percentile of ``third_down_conv_rate_centered`` across every
   team-season strictly before S -- and is NaN where no prior season exists.
   This makes the column a slightly different quantity from the registered
   cell; a noisier threshold can only attenuate toward the null, and
   ``tests/test_redzone_reversion_production_feature.py`` pins both halves of
   the pregame claim.

Mirrors ``nfl_ats.forecast_weather_features.attach_forecast_weather_features``'s
additive-merge discipline: every pre-existing column comes back bit-identical,
only the one new column is added. Games with no prior-season panel row for
either side, or with no threshold, come back NaN and are LEFT NaN on purpose:
imputation belongs to the model's own training-fold median
(``fit_margin_model``), not to a feature builder that can see every season at
once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError
from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.redzone_reversion_screen import (  # noqa: E402
    _alias_team,
    _prior,
    build_efficiency_panels,
)

#: The one new column this module adds. Frozen name, matching
#: ``nfl_ats.constants.REDZONE_THIRD_DOWN_OVER_FADE_ON_PRODUCTION_FEATURE_COLUMNS``.
REDZONE_THIRD_DOWN_OVER_FADE_COLUMN = "redzone_third_down_over_fade_diff"
REDZONE_REVERSION_ON_PRODUCTION_FEATURE_COLUMNS = (REDZONE_THIRD_DOWN_OVER_FADE_COLUMN,)

#: The centred trait the frozen screen's C3 cell flags on.
TRAIT_COLUMN = "third_down_conv_rate_centered"

#: Top-quartile cut, inherited unchanged from the frozen screen's C3 cell
#: (``prior_third_down_conv_rate_centered >= third_down_q75``). Only the
#: ESTIMATION population changes (deviation 2), never the quantile.
TOP_QUARTILE = 0.75

_REQUIRED_COLUMNS = {"game_id", "season", "home_team", "away_team"}


def load_offense_panel(pbp_root: Path | None = None) -> pd.DataFrame:
    """Build the frozen offensive efficiency panel from the local PBP snapshot.

    Delegates entirely to ``scripts.redzone_reversion_screen`` -- the panel is
    the screen's own, not a re-derivation. Production callers use this; tests
    pass a synthetic panel to ``derive_*``/``attach_*`` instead so they run in
    a fresh clone with no local data.
    """

    root = pbp_root if pbp_root is not None else REPO_ROOT / "data/pbp/raw"
    snapshot = latest_pbp_snapshot(root)
    offense, _defense = build_efficiency_panels(load_pbp_snapshot(snapshot))
    return offense


def expanding_top_quartile_thresholds(offense: pd.DataFrame) -> pd.Series:
    """Per-season top-quartile cut estimated from STRICTLY PRIOR seasons only.

    For season ``S`` the value is the 75th percentile of ``TRAIT_COLUMN`` over
    every team-season with ``season < S``. The panel's first season has no
    prior seasons and therefore no threshold (absent from the returned index),
    which is what makes every value of the resulting column pregame-legal.
    This is deviation 2 of ``docs/redzone_reversion_on_production.md``: the
    frozen screen instead takes one quantile over the whole 2009-2025 panel.
    """

    if TRAIT_COLUMN not in offense.columns or "season" not in offense.columns:
        raise DataContractError(f"offense panel is missing season/{TRAIT_COLUMN}")

    panel = offense.loc[:, ["season", TRAIT_COLUMN]].copy()
    panel["season"] = pd.to_numeric(panel["season"], errors="raise").astype(int)
    panel = panel.loc[panel[TRAIT_COLUMN].notna()]

    thresholds: dict[int, float] = {}
    if panel.empty:
        return pd.Series(thresholds, dtype=float, name="third_down_top_quartile")

    for season in range(int(panel["season"].min()) + 1, int(panel["season"].max()) + 2):
        prior = panel.loc[panel["season"].lt(season), TRAIT_COLUMN]
        if prior.empty:
            continue
        thresholds[season] = float(prior.quantile(TOP_QUARTILE))
    return pd.Series(thresholds, dtype=float, name="third_down_top_quartile")


def _side_flags(
    frame: pd.DataFrame,
    offense: pd.DataFrame,
    thresholds: pd.Series,
    *,
    team_column: str,
) -> np.ndarray:
    """Flag array for one side, ``NaN`` where the prior value or the threshold
    is unavailable."""

    prior = _prior(offense, [TRAIT_COLUMN], team_column)
    merged = frame.merge(prior, on=[team_column, "season"], how="left", validate="many_to_one")
    values = pd.to_numeric(merged[f"prior_{TRAIT_COLUMN}"], errors="coerce").to_numpy(dtype=float)
    cuts = merged["season"].map(thresholds).to_numpy(dtype=float)
    usable = np.isfinite(values) & np.isfinite(cuts)
    return np.where(usable, (values >= cuts).astype(float), np.nan)


def derive_redzone_third_down_features(
    features: pd.DataFrame,
    *,
    offense: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a ``(game_id, redzone_third_down_over_fade_diff)`` frame.

    ``offense`` defaults to the frozen panel built from the latest local PBP
    snapshot (:func:`load_offense_panel`); it is accepted as a parameter purely
    for testability -- production callers leave it unset.

    A game whose home or away side has no prior-season panel row, or whose
    season has no strictly-prior threshold, comes back **NaN**, never 0. "No
    prior-season information" and "prior-season information showing an
    unflagged team" are different states, and only the model's own
    training-fold median may decide what to do with the first.
    """

    missing = sorted(_REQUIRED_COLUMNS.difference(features.columns))
    if missing:
        raise DataContractError(f"features is missing columns: {', '.join(missing)}")

    frame = features.loc[:, sorted(_REQUIRED_COLUMNS)].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["home_team"] = _alias_team(frame["home_team"])
    frame["away_team"] = _alias_team(frame["away_team"])

    if offense is None:
        offense = load_offense_panel()
    panel = offense.loc[:, ["team", "season", TRAIT_COLUMN]].copy()
    panel["team"] = _alias_team(panel["team"])
    panel["season"] = pd.to_numeric(panel["season"], errors="raise").astype(int)
    if panel.duplicated(subset=["team", "season"]).any():
        raise DataContractError("offense panel carries duplicate (team, season) rows")

    thresholds = expanding_top_quartile_thresholds(panel)
    home = _side_flags(frame, panel, thresholds, team_column="home_team")
    away = _side_flags(frame, panel, thresholds, team_column="away_team")

    both_known = np.isfinite(home) & np.isfinite(away)
    difference = np.where(both_known, home - away, np.nan)
    return pd.DataFrame(
        {
            "game_id": frame["game_id"],
            REDZONE_THIRD_DOWN_OVER_FADE_COLUMN: difference,
        }
    )


def attach_redzone_third_down_features(
    features: pd.DataFrame,
    *,
    offense: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Additively join the one new column onto ``features``.

    Every pre-existing column is returned bit-identical; only
    ``redzone_third_down_over_fade_diff`` is added.
    """

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    collisions = sorted(
        set(REDZONE_REVERSION_ON_PRODUCTION_FEATURE_COLUMNS).intersection(features.columns)
    )
    if collisions:
        raise DataContractError(f"features already carries {', '.join(collisions)}")

    derived = derive_redzone_third_down_features(features, offense=offense)
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_redzone"),
        validate="one_to_one",
    )
    merged = merged.drop(columns=[c for c in ("key_0", "game_id_redzone") if c in merged.columns])
    merged.index = features.index
    return merged


__all__ = [
    "REDZONE_REVERSION_ON_PRODUCTION_FEATURE_COLUMNS",
    "REDZONE_THIRD_DOWN_OVER_FADE_COLUMN",
    "TOP_QUARTILE",
    "TRAIT_COLUMN",
    "attach_redzone_third_down_features",
    "derive_redzone_third_down_features",
    "expanding_top_quartile_thresholds",
    "load_offense_panel",
]
