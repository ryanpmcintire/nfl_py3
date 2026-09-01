"""FluView home/away-market illness indicators, stacked on PRODUCTION.

``docs/fluview_battery.md`` predeclared and froze five cells against a BARE
market baseline (``fluview_home_market_elevated`` +0.309 accuracy points, P+
0.818; ``fluview_away_market_elevated`` +0.368, P+ 0.883 -- both week-blocked,
both ``unresolved_below_power``, recorded 2026-08-20). The project's own
recorded lesson ("composition is not the signal") is that a component
positive alone can go negative once stacked on the chain that is actually
PLAYED. This module builds the same two elevated-illness indicators, at the
SAME frozen construction, additively joined onto a feature table by
``game_id``, so ``docs/fluview_on_production.md`` can measure them on top of
PRODUCTION ``weak_stack`` instead of a bare baseline.

**Reuses the frozen as-of/threshold construction verbatim, does not rebuild
it**: ``build_checkpoint_tables`` and ``attach_asof_ili`` are imported
directly from ``scripts/fluview_battery_screen.py`` (the same point-in-time
-safe checkpoint-table / ``merge_asof`` mechanism the frozen battery uses),
and the per-state top-decile thresholds are read from that battery's own
already-recorded results artifact (computed ONCE on the battery's own
population, per ``docs/fluview_battery.md`` section 3 -- never re-derived
here on a different population). ``scripts`` is not part of the installed
package, so this module puts the repository root on ``sys.path`` the same
guarded way ``nfl_ats.cli._ensure_repo_root_on_path`` already does for the
same reason (a script-level construction reused from a src module).

Mirrors ``nfl_ats.forecast_weather_features.attach_forecast_weather_features``'s
additive-merge discipline: every pre-existing column comes back bit-identical,
only the two new columns are added.

**Location restriction, inherited unchanged from the frozen battery**
(``docs/fluview_battery.md`` section 2): the home-market illness mechanism
does not apply at a neutral or displaced site, so both columns are set to
NaN (missing, not "not elevated") for any game whose ``location`` is not
``"Home"`` -- exactly the same population restriction the frozen battery
enforced by dropping those rows outright; here the row is kept (this is a
feature builder, not an evaluator) but the feature is disclosed as
inapplicable via NaN, left for the model's own training-fold median
imputation, never defaulted to a value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fluview_battery_ingest import STATE_BY_TEAM  # noqa: E402
from scripts.fluview_battery_screen import attach_asof_ili, build_checkpoint_tables  # noqa: E402

#: The two new columns this module adds. Frozen names, matching the already
#: -recorded weak-signal registry cell names 1:1 (docs/fluview_battery.md
#: section 5, F1/F2) so the lineage between the bare-baseline screen and this
#: on-production stacking is legible from the column name alone.
FLUVIEW_HOME_ELEVATED_COLUMN = "fluview_home_market_elevated"
FLUVIEW_AWAY_ELEVATED_COLUMN = "fluview_away_market_elevated"
FLUVIEW_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS = (
    FLUVIEW_HOME_ELEVATED_COLUMN,
    FLUVIEW_AWAY_ELEVATED_COLUMN,
)

_REQUIRED_COLUMNS = {"game_id", "season", "week", "gameday", "home_team", "away_team", "location"}


def _latest(glob_pattern: str, label: str) -> Path:
    candidates = sorted(REPO_ROOT.glob(glob_pattern))
    if not candidates:
        raise FileNotFoundError(f"no {label} found matching {glob_pattern!r}")
    return candidates[-1]


def default_fluview_raw_path() -> Path:
    """The already-ingested FluView snapshot (docs/fluview_battery.md section
    1) -- resolved lazily so importing this module never requires local data."""

    return _latest("data/raw/fluview/*/fluview_raw.parquet", "fluview_raw.parquet snapshot")


def default_fluview_results_path() -> Path:
    """The frozen battery's own latest scored results artifact, source of the
    frozen per-state thresholds (docs/fluview_battery.md section 3)."""

    return _latest("artifacts/fluview_battery/*/results.json", "fluview_battery results.json")


def load_frozen_state_thresholds(results_path: Path | None = None) -> tuple[dict[str, float], Path]:
    """The per-state top-decile thresholds already computed and recorded by
    ``scripts/fluview_battery_screen.py`` (docs/fluview_battery.md section 3:
    "computed ONCE from the full panel ... not re-derived per season").

    Read from the battery's own results artifact rather than recomputed here,
    so this module can never silently drift from the frozen, already-recorded
    battery even if it is later applied to a different game population.
    """

    path = results_path or default_fluview_results_path()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    thresholds = {str(state): float(value) for state, value in payload["state_thresholds"].items()}
    return thresholds, path


def _cutoff_dates(gameday: pd.Series) -> pd.Series:
    """The Tuesday of the game's own week -- identical formula to
    ``scripts/fluview_battery_screen.py::load_schedules`` and
    ``scripts/attention_battery_screen.py``'s own ``tuesday_offset``
    convention (Monday=0)."""

    weekday = gameday.dt.weekday
    tuesday_offset = (weekday - 1) % 7
    result: pd.Series = gameday - pd.to_timedelta(tuesday_offset, unit="D")
    return result


def derive_fluview_elevated_features(
    features: pd.DataFrame,
    *,
    fluview_raw: pd.DataFrame | None = None,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return a ``(game_id, fluview_home_market_elevated,
    fluview_away_market_elevated)`` frame, point-in-time-safe.

    ``fluview_raw`` defaults to the already-ingested snapshot
    (``default_fluview_raw_path``); ``thresholds`` defaults to the frozen
    battery's own recorded per-state deciles (``load_frozen_state_thresholds``).
    Both are accepted as parameters purely for testability -- production
    callers should leave them unset.
    """

    missing = sorted(_REQUIRED_COLUMNS.difference(features.columns))
    if missing:
        raise DataContractError(f"features is missing columns: {', '.join(missing)}")

    frame = features.loc[:, sorted(_REQUIRED_COLUMNS)].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame["cutoff_date"] = _cutoff_dates(frame["gameday"])
    frame["home_state"] = frame["home_team"].map(STATE_BY_TEAM)
    frame["away_state"] = frame["away_team"].map(STATE_BY_TEAM)
    unmapped = frame.loc[frame["home_state"].isna() | frame["away_state"].isna()]
    if len(unmapped):
        raise DataContractError(
            f"{len(unmapped)} games have a home/away team not in STATE_BY_TEAM: "
            f"{sorted(set(unmapped['home_team']) | set(unmapped['away_team']))}"
        )

    if fluview_raw is None:
        fluview_raw = pd.read_parquet(default_fluview_raw_path())
    checkpoints = build_checkpoint_tables(fluview_raw.loc[fluview_raw["region"] != "nat"])
    frame = attach_asof_ili(frame, checkpoints)  # adds home_ili, away_ili

    resolved_thresholds = (
        thresholds if thresholds is not None else load_frozen_state_thresholds()[0]
    )
    frame["home_threshold"] = frame["home_state"].map(resolved_thresholds)
    frame["away_threshold"] = frame["away_state"].map(resolved_thresholds)

    home_missing = frame["home_ili"].isna() | frame["home_threshold"].isna()
    away_missing = frame["away_ili"].isna() | frame["away_threshold"].isna()
    is_home_location = frame["location"].astype(str).eq("Home")

    home_elevated_raw = np.where(
        frame["home_ili"].to_numpy() >= frame["home_threshold"].to_numpy(), 1.0, 0.0
    )
    away_elevated_raw = np.where(
        frame["away_ili"].to_numpy() >= frame["away_threshold"].to_numpy(), 1.0, 0.0
    )

    derived = pd.DataFrame(
        {
            "game_id": frame["game_id"],
            FLUVIEW_HOME_ELEVATED_COLUMN: np.where(
                (home_missing | ~is_home_location).to_numpy(), np.nan, home_elevated_raw
            ),
            FLUVIEW_AWAY_ELEVATED_COLUMN: np.where(
                (away_missing | ~is_home_location).to_numpy(), np.nan, away_elevated_raw
            ),
        }
    )
    return derived


def attach_fluview_elevated_features(
    features: pd.DataFrame,
    *,
    fluview_raw: pd.DataFrame | None = None,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Additively join the two new columns onto ``features``.

    Every pre-existing column is returned bit-identical; only
    ``fluview_home_market_elevated``/``fluview_away_market_elevated`` are
    added. Missing as-of coverage (nearly all of 2010-2017, measured in
    docs/fluview_battery.md section 1) or a non-``"Home"``-location game
    comes back NaN, left NaN on purpose: imputation belongs to the model's
    own training-fold median (``fit_margin_model``), not to a feature
    builder that can see every season at once.
    """

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    collisions = sorted(
        set(FLUVIEW_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS).intersection(features.columns)
    )
    if collisions:
        raise DataContractError(f"features already carries {', '.join(collisions)}")

    derived = derive_fluview_elevated_features(
        features, fluview_raw=fluview_raw, thresholds=thresholds
    )
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_fluview"),
        validate="one_to_one",
    )
    merged = merged.drop(columns=[c for c in ("key_0", "game_id_fluview") if c in merged.columns])
    merged.index = features.index
    return merged


__all__ = [
    "FLUVIEW_AWAY_ELEVATED_COLUMN",
    "FLUVIEW_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS",
    "FLUVIEW_HOME_ELEVATED_COLUMN",
    "attach_fluview_elevated_features",
    "default_fluview_raw_path",
    "default_fluview_results_path",
    "derive_fluview_elevated_features",
    "load_frozen_state_thresholds",
]
