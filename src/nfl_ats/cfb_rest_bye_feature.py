"""Rest / bye candidate columns for COLLEGE FOOTBALL (cross-league replication
of the NFL rest-and-bye constructs frozen in ``docs/travel_rest_battery.md``
and ``docs/bye_overvaluation_screen.md``).

Predeclared in ``docs/cfb_rest_bye_replication.md``. Read that first: it freezes
the population, the four cells, the per-side rest derivation, the comparator,
the null, the positive control and the recording rules before any outcome sign
was computed.

**The one thing that must be said before anything else**: the frozen XLG-03
benchmark contract ``nfl_ats.cfb_features.CFB_MODEL_FEATURE_COLUMNS`` ALREADY
carries ``rest_diff`` (``src/nfl_ats/cfb_features.py:82``). Every column built
here is therefore a MARGINAL on top of a model that already prices the linear
rest difference -- the project's own "composition is not the signal" discipline
applied to rest: evaluate on top of what is played, not on top of a bare
baseline.

**Reuses the derivation, does not reinvent it.** Per-side rest days come from
``nfl_ats.cfb_features._rest_base_schedule`` / ``_add_rest_features`` -- the
exact private helpers that built the frozen ``rest_diff`` column in
``data/processed/cfb_game_features.parquet`` -- applied to the FULL CFB
schedules snapshot (every completed regular-season appearance, any division),
not to the filtered benchmark table. That distinction matters: the benchmark
table is FBS-vs-FBS with an orientable spread and play-by-play, so a team's
actual previous game is often absent from it, and a rest value computed from
the subset alone would be wrong.

Missing rest is NaN, never 0 and never "not off bye". A team's FIRST game of a
season has no defined rest, so every cell that needs that side's rest is NaN
for that row -- the identical rows ``rest_diff`` is already NaN for in the
frozen contract. The model's own training-fold ``SimpleImputer(strategy=
"median", add_indicator=True)`` (``nfl_ats.margin.make_margin_estimator``)
handles it, exactly as it already handles ``rest_diff``'s own missingness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.cfb import latest_cfb_snapshot, load_cfb_snapshot
from nfl_ats.cfb_features import _add_rest_features, _rest_base_schedule
from nfl_ats.data import DataContractError

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Frozen thresholds (docs/cfb_rest_bye_replication.md section 3)
# ---------------------------------------------------------------------------

#: NFL ``travel_rest_home_off_bye`` / ``travel_rest_away_off_bye``: rest >= 13
#: days, a SIDE-SPECIFIC ABSOLUTE threshold. Transcribed verbatim; see the
#: predeclaration section 3 for why 13 survives the CFB calendar unchanged.
OFF_BYE_REST_DAYS = 13

#: NFL ``bye_overval_home_edge_post2011``: "off strict bye" is a >= 12-day gap
#: (``scripts/venue_milestone_screen.py``'s ``POST_BYE_GAP_DAYS=12``,
#: ``docs/bye_overvaluation_screen.md`` "Strict bye definition").
STRICT_BYE_GAP_DAYS = 12

#: NFL ``travel_rest_short_week_road``: away rest <= 5 days.
SHORT_WEEK_REST_DAYS = 5

#: Declared sensitivity arms, frozen with the primaries and never substituted
#: for them (predeclaration section 3). 12 is the CFB-calendar-widened off-bye
#: gap (a Saturday -> open date -> Thursday turnaround is 12 days, not 13); 6
#: is the CFB-calendar-widened short week (Saturday -> Friday).
OFF_BYE_SENSITIVITY_REST_DAYS = 12
SHORT_WEEK_SENSITIVITY_REST_DAYS = 6

# ---------------------------------------------------------------------------
# Column names. ``cfb_`` prefixed so a CFB column can never be confused with
# the NFL flags of the same construct in registry/weak_signals.json.
# ---------------------------------------------------------------------------

CFB_HOME_OFF_BYE_COLUMN = "cfb_rest_home_off_bye"
CFB_AWAY_OFF_BYE_COLUMN = "cfb_rest_away_off_bye"
CFB_BYE_EDGE_HOME_COLUMN = "cfb_rest_bye_edge_home"
CFB_SHORT_WEEK_ROAD_COLUMN = "cfb_rest_short_week_road"

CFB_HOME_OFF_BYE_GAP12_COLUMN = "cfb_rest_home_off_bye_gap12"
CFB_AWAY_OFF_BYE_GAP12_COLUMN = "cfb_rest_away_off_bye_gap12"
CFB_SHORT_WEEK_ROAD_LE6_COLUMN = "cfb_rest_short_week_road_le6"

#: The four predeclared cells, in the order section 3 declares them.
CFB_REST_BYE_PRIMARY_COLUMNS = (
    CFB_HOME_OFF_BYE_COLUMN,
    CFB_AWAY_OFF_BYE_COLUMN,
    CFB_BYE_EDGE_HOME_COLUMN,
    CFB_SHORT_WEEK_ROAD_COLUMN,
)
CFB_REST_BYE_SENSITIVITY_COLUMNS = (
    CFB_HOME_OFF_BYE_GAP12_COLUMN,
    CFB_AWAY_OFF_BYE_GAP12_COLUMN,
    CFB_SHORT_WEEK_ROAD_LE6_COLUMN,
)
CFB_REST_BYE_FEATURE_COLUMNS = (
    *CFB_REST_BYE_PRIMARY_COLUMNS,
    *CFB_REST_BYE_SENSITIVITY_COLUMNS,
)

#: Per-side rest columns the derivation adds alongside the cells.
CFB_SIDE_REST_COLUMNS = ("cfb_home_rest_days", "cfb_away_rest_days")

_REQUIRED_COLUMNS = {"game_id", "season", "week", "gameday", "home_id", "away_id"}

#: The team-season panel metrics section 7 computes split-half reliability on.
CFB_REST_PANEL_METRICS = (
    "own_rest_days",
    "own_off_bye_13",
    "own_strict_bye_edge",
    "own_short_week_5",
    "own_off_bye_12",
    "own_short_week_6",
)

#: Which panel metric carries each candidate column's own propensity. Home and
#: away cells share one entry on purpose: it is the SAME team-season trait,
#: read off a different side of the game (section 7).
CFB_REST_CELL_PANEL_METRIC: dict[str, str] = {
    CFB_HOME_OFF_BYE_COLUMN: "own_off_bye_13",
    CFB_AWAY_OFF_BYE_COLUMN: "own_off_bye_13",
    CFB_BYE_EDGE_HOME_COLUMN: "own_strict_bye_edge",
    CFB_SHORT_WEEK_ROAD_COLUMN: "own_short_week_5",
    CFB_HOME_OFF_BYE_GAP12_COLUMN: "own_off_bye_12",
    CFB_AWAY_OFF_BYE_GAP12_COLUMN: "own_off_bye_12",
    CFB_SHORT_WEEK_ROAD_LE6_COLUMN: "own_short_week_6",
}


# ---------------------------------------------------------------------------
# Per-side rest derivation
# ---------------------------------------------------------------------------


def default_cfb_schedules(cfb_root: Path | None = None) -> pd.DataFrame:
    """The FULL CFB schedules snapshot -- every season it carries, any division.

    Resolved through ``nfl_ats.cfb.latest_cfb_snapshot`` so this module never
    hardcodes a snapshot id, and never reads the filtered benchmark table for a
    fact about a team's previous game.
    """

    root = cfb_root if cfb_root is not None else REPO_ROOT / "data" / "cfb"
    return load_cfb_snapshot(latest_cfb_snapshot(root, "schedules"))


def derive_side_rest(games: pd.DataFrame, schedules: pd.DataFrame | None = None) -> pd.DataFrame:
    """``(game_id, home_rest, away_rest, rest_diff)`` for every row of ``games``.

    Reuses ``nfl_ats.cfb_features._rest_base_schedule`` and ``_add_rest_features``
    -- the exact helpers that produced the frozen ``rest_diff`` column -- so the
    ``home_rest - away_rest`` reconstruction is bit-identical to the benchmark's
    own column wherever both are defined (pinned by
    ``tests/test_cfb_rest_bye_feature.py``).

    Pregame by construction: only ``season``, ``start_date``/``gameday``,
    ``completed``, ``season_type`` and team ids are read. No score, no line, no
    outcome column is touched anywhere in this function.
    """

    missing = sorted(_REQUIRED_COLUMNS.difference(games.columns))
    if missing:
        raise DataContractError(f"CFB games frame is missing columns: {', '.join(missing)}")

    frame = games.loc[:, ["game_id", "season", "week", "gameday", "home_id", "away_id"]].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    for column in ("home_id", "away_id"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")

    source = schedules if schedules is not None else default_cfb_schedules()
    base = _rest_base_schedule(source, int(frame["season"].min()), int(frame["season"].max()))
    rested = _add_rest_features(frame, base).rename(
        columns={"home_rest": "cfb_home_rest_days", "away_rest": "cfb_away_rest_days"}
    )
    return rested.loc[
        :, ["game_id", "season", "week", "home_id", "away_id", *CFB_SIDE_REST_COLUMNS]
    ]


def _cell_columns(home_rest: pd.Series, away_rest: pd.Series) -> dict[str, np.ndarray]:
    """The four predeclared cells plus the three declared sensitivity arms.

    Every cell is NaN -- missing, never "not off bye" -- when a rest value it
    needs is undefined. That is the season-opener rule frozen in section 4, and
    it lands on exactly the rows ``rest_diff`` is already NaN for.
    """

    home = home_rest.to_numpy(dtype=float)
    away = away_rest.to_numpy(dtype=float)
    home_known = np.isfinite(home)
    away_known = np.isfinite(away)
    both_known = home_known & away_known

    def masked(values: np.ndarray, known: np.ndarray) -> np.ndarray:
        return np.where(known, values.astype(float), np.nan)

    with np.errstate(invalid="ignore"):
        return {
            CFB_HOME_OFF_BYE_COLUMN: masked(home >= OFF_BYE_REST_DAYS, home_known),
            CFB_AWAY_OFF_BYE_COLUMN: masked(away >= OFF_BYE_REST_DAYS, away_known),
            CFB_BYE_EDGE_HOME_COLUMN: masked(
                (home >= STRICT_BYE_GAP_DAYS) & (away < STRICT_BYE_GAP_DAYS), both_known
            ),
            CFB_SHORT_WEEK_ROAD_COLUMN: masked(away <= SHORT_WEEK_REST_DAYS, away_known),
            CFB_HOME_OFF_BYE_GAP12_COLUMN: masked(
                home >= OFF_BYE_SENSITIVITY_REST_DAYS, home_known
            ),
            CFB_AWAY_OFF_BYE_GAP12_COLUMN: masked(
                away >= OFF_BYE_SENSITIVITY_REST_DAYS, away_known
            ),
            CFB_SHORT_WEEK_ROAD_LE6_COLUMN: masked(
                away <= SHORT_WEEK_SENSITIVITY_REST_DAYS, away_known
            ),
        }


def build_cfb_rest_team_panel(rested: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: the team-season panel reliability is read on.

    ``split_half_reliability`` (``nfl_ats.cfb_qb_dependence``) wants
    ``team_id``/``season``/``week`` plus a metric, and splits each team-season
    by odd/even week. Stacking both sides gives every team its own rest history
    -- the propensity "how rested does this team tend to arrive", which is the
    trait each cell reads off one side of. Home and away cells therefore share
    one reliability figure by construction; the cell differs only in which side
    of the game the same trait is evaluated on.
    """

    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        sides.append(
            pd.DataFrame(
                {
                    "team_id": rested[f"{side}_id"].to_numpy(),
                    "season": rested["season"].to_numpy(),
                    "week": rested["week"].to_numpy(),
                    "is_home": side == "home",
                    "own_rest_days": rested[f"cfb_{side}_rest_days"].to_numpy(dtype=float),
                    "opponent_rest_days": rested[f"cfb_{other}_rest_days"].to_numpy(dtype=float),
                }
            )
        )
    panel = pd.concat(sides, ignore_index=True)
    own = panel["own_rest_days"].to_numpy(dtype=float)
    opponent = panel["opponent_rest_days"].to_numpy(dtype=float)
    known = np.isfinite(own)
    both = known & np.isfinite(opponent)
    panel["own_off_bye_13"] = np.where(known, (own >= OFF_BYE_REST_DAYS).astype(float), np.nan)
    panel["own_strict_bye_edge"] = np.where(
        both,
        ((own >= STRICT_BYE_GAP_DAYS) & (opponent < STRICT_BYE_GAP_DAYS)).astype(float),
        np.nan,
    )
    panel["own_short_week_5"] = np.where(known, (own <= SHORT_WEEK_REST_DAYS).astype(float), np.nan)
    panel["own_off_bye_12"] = np.where(
        known, (own >= OFF_BYE_SENSITIVITY_REST_DAYS).astype(float), np.nan
    )
    panel["own_short_week_6"] = np.where(
        known, (own <= SHORT_WEEK_SENSITIVITY_REST_DAYS).astype(float), np.nan
    )
    return panel


def derive_cfb_rest_bye_features(
    features: pd.DataFrame, *, schedules: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return ``(derived, diagnostics)``.

    ``derived`` carries ``game_id``, the two per-side rest columns and the seven
    candidate columns. ``diagnostics`` carries per-season coverage, per-cell
    flagged/missing counts, the reconstructed-vs-frozen ``rest_diff`` agreement
    and the team-season panel, so the harness can report them without
    re-deriving anything. Predictor-only: nothing here reads an outcome.
    """

    rested = derive_side_rest(features, schedules)
    home_rest = rested["cfb_home_rest_days"]
    away_rest = rested["cfb_away_rest_days"]
    cells = _cell_columns(home_rest, away_rest)

    derived = pd.DataFrame({"game_id": rested["game_id"].to_numpy()})
    derived["cfb_home_rest_days"] = home_rest.to_numpy(dtype=float)
    derived["cfb_away_rest_days"] = away_rest.to_numpy(dtype=float)
    for column, values in cells.items():
        derived[column] = values

    # Season is already an int64 column (derive_side_rest casts it), so the
    # groupby keys stringify as "2012", never "2012.0".
    season = pd.Series(rested["season"].to_numpy(dtype="int64"))
    coverage: dict[str, dict[str, float]] = {}
    for column in CFB_REST_BYE_FEATURE_COLUMNS:
        covered = pd.Series(np.isfinite(derived[column].to_numpy(dtype=float)).astype(float))
        coverage[column] = {
            str(key): float(value) for key, value in covered.groupby(season).mean().items()
        }

    reconstruction: dict[str, Any] = {"frozen_rest_diff_column_present": "rest_diff" in features}
    if "rest_diff" in features.columns:
        frozen = pd.to_numeric(features["rest_diff"], errors="coerce").to_numpy(dtype=float)
        mine = (home_rest - away_rest).to_numpy(dtype=float)
        both_defined = np.isfinite(frozen) & np.isfinite(mine)
        reconstruction.update(
            {
                "n_both_defined": int(both_defined.sum()),
                "n_exact_match": int((mine[both_defined] == frozen[both_defined]).sum()),
                "n_missingness_pattern_mismatch": int(
                    (np.isfinite(frozen) != np.isfinite(mine)).sum()
                ),
                "max_abs_difference": (
                    float(np.abs(mine[both_defined] - frozen[both_defined]).max())
                    if both_defined.any()
                    else float("nan")
                ),
            }
        )

    diagnostics: dict[str, Any] = {
        "n_games": len(derived),
        "n_home_rest_missing": int((~np.isfinite(home_rest.to_numpy(dtype=float))).sum()),
        "n_away_rest_missing": int((~np.isfinite(away_rest.to_numpy(dtype=float))).sum()),
        "n_either_rest_missing": int(
            (
                ~np.isfinite(home_rest.to_numpy(dtype=float))
                | ~np.isfinite(away_rest.to_numpy(dtype=float))
            ).sum()
        ),
        "flagged_by_column": {
            column: int(np.nansum(derived[column].to_numpy(dtype=float)))
            for column in CFB_REST_BYE_FEATURE_COLUMNS
        },
        "missing_by_column": {
            column: int((~np.isfinite(derived[column].to_numpy(dtype=float))).sum())
            for column in CFB_REST_BYE_FEATURE_COLUMNS
        },
        "coverage_by_season": coverage,
        "rest_days_histogram": {
            side: {
                str(key): int(value)
                for key, value in rested[f"cfb_{side}_rest_days"]
                .dropna()
                .astype(int)
                .value_counts()
                .sort_index()
                .items()
            }
            for side in ("home", "away")
        },
        "rest_diff_reconstruction": reconstruction,
        "team_panel": build_cfb_rest_team_panel(rested),
    }
    return derived, diagnostics


def attach_cfb_rest_bye_features(
    features: pd.DataFrame, *, schedules: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Additively join the rest columns and the candidate cells onto ``features``.

    Every pre-existing column comes back untouched -- including the frozen
    ``rest_diff`` the baseline arm already uses -- and only the new columns are
    added, mirroring ``nfl_ats.fluview_cfb_feature.attach_cfb_fluview_features``'s
    additive-merge discipline.
    """

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    added = (*CFB_SIDE_REST_COLUMNS, *CFB_REST_BYE_FEATURE_COLUMNS)
    collisions = sorted(set(added).intersection(features.columns))
    if collisions:
        raise DataContractError(f"features already carries {', '.join(collisions)}")

    derived, diagnostics = derive_cfb_rest_bye_features(features, schedules=schedules)
    merged = features.copy()
    for column in added:
        merged[column] = derived[column].to_numpy()
    return merged, diagnostics


__all__ = [
    "CFB_AWAY_OFF_BYE_COLUMN",
    "CFB_AWAY_OFF_BYE_GAP12_COLUMN",
    "CFB_BYE_EDGE_HOME_COLUMN",
    "CFB_HOME_OFF_BYE_COLUMN",
    "CFB_HOME_OFF_BYE_GAP12_COLUMN",
    "CFB_REST_BYE_FEATURE_COLUMNS",
    "CFB_REST_BYE_PRIMARY_COLUMNS",
    "CFB_REST_BYE_SENSITIVITY_COLUMNS",
    "CFB_REST_CELL_PANEL_METRIC",
    "CFB_REST_PANEL_METRICS",
    "CFB_SHORT_WEEK_ROAD_COLUMN",
    "CFB_SHORT_WEEK_ROAD_LE6_COLUMN",
    "CFB_SIDE_REST_COLUMNS",
    "OFF_BYE_REST_DAYS",
    "OFF_BYE_SENSITIVITY_REST_DAYS",
    "SHORT_WEEK_REST_DAYS",
    "SHORT_WEEK_SENSITIVITY_REST_DAYS",
    "STRICT_BYE_GAP_DAYS",
    "attach_cfb_rest_bye_features",
    "build_cfb_rest_team_panel",
    "default_cfb_schedules",
    "derive_cfb_rest_bye_features",
    "derive_side_rest",
]
