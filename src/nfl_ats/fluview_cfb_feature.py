"""FluView elevated-illness indicators for COLLEGE FOOTBALL (cross-league
replication of the NFL construct frozen in ``docs/fluview_battery.md``).

Predeclared in ``docs/fluview_cfb_replication.md``. Read that first: it freezes
the population, the comparator, the cells, the null, the positive control and
the recording rules before any outcome sign was computed.

**Replicates the construct, does not reinvent it.** The point-in-time-safe
as-of machinery is IMPORTED verbatim from ``scripts/fluview_battery_screen.py``
-- the same ``build_checkpoint_tables`` / ``attach_asof_ili`` checkpoint-table
and ``merge_asof``-against-a-Tuesday-cutoff mechanism the frozen NFL battery
itself calls, and the same ``compute_state_thresholds`` top-decile rule with
its >=10-observation floor. ``scripts`` is not part of the installed package,
so this module puts the repository root on ``sys.path`` the same guarded way
``nfl_ats.fluview_production_feature`` and ``nfl_ats.cli._ensure_repo_root_on_path``
already do for the same reason.

Exactly two things differ from the NFL module, both because the league differs:

1. **Team -> state.** The NFL uses a static 34-code dict. CFB uses a
   per-season school -> venue-state map read from cfbfastR-data's
   ``team_info`` snapshot and joined on the CFBD/ESPN ``team_id`` (never on a
   school name). No local CFB snapshot carries venue state --
   ``data/processed/cfb_game_features.parquet`` has no venue/city/state
   column, the schedules snapshot's venue NAME resolves a state for only 4.2%
   of games, and ``registry/stadium_coordinates.json`` is NFL-only.
2. **The state panel the top-decile threshold is computed on** is CFB's own
   (one row per state/season/week over the benchmark table's non-neutral
   games), because the frozen rule is "the top decile of THAT STATE'S OWN
   history" -- copying the NFL's numeric thresholds would replicate a number
   measured on a different panel, not the construct.

The home-market restriction is inherited unchanged: the NFL battery uses
``location == "Home"``; the CFB mirror is ``neutral_site`` false. A
neutral-site game's columns come back NaN (missing, not "not elevated"),
left NaN on purpose so the model's own training-fold median imputation
handles it, exactly as ``nfl_ats.fluview_production_feature`` does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fluview_battery_screen import (  # noqa: E402
    attach_asof_ili,
    build_checkpoint_tables,
    compute_state_thresholds,
)

#: The two candidate columns. Named with a ``cfb_`` prefix so a CFB column can
#: never be confused with the NFL ``fluview_home_market_elevated`` /
#: ``fluview_away_market_elevated`` columns that already exist in
#: ``nfl_ats.fluview_production_feature`` and in the weak-signal registry.
CFB_FLUVIEW_HOME_ELEVATED_COLUMN = "cfb_fluview_home_market_elevated"
CFB_FLUVIEW_AWAY_ELEVATED_COLUMN = "cfb_fluview_away_market_elevated"
CFB_FLUVIEW_FEATURE_COLUMNS = (
    CFB_FLUVIEW_HOME_ELEVATED_COLUMN,
    CFB_FLUVIEW_AWAY_ELEVATED_COLUMN,
)

#: cfbfastR-data ``team_info`` columns this module needs. ``state`` sits beside
#: ``venue_id``/``venue_name``/``city``/``zip``/``latitude``/``longitude``, so
#: it is the school's own listed VENUE state -- the home-market state the
#: mechanism is about -- and it is per-season, so a venue change is carried.
TEAM_INFO_COLUMNS = ("team_id", "school", "venue_id", "venue_name", "city", "state")

_REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "week",
    "gameday",
    "home_id",
    "away_id",
    "neutral_site",
}


# ---------------------------------------------------------------------------
# Input resolution (lazy, so importing this module never requires local data)
# ---------------------------------------------------------------------------


def _latest(glob_pattern: str, label: str) -> Path:
    candidates = sorted(REPO_ROOT.glob(glob_pattern))
    if not candidates:
        raise FileNotFoundError(f"no {label} found matching {glob_pattern!r}")
    return candidates[-1]


def default_team_info_dir() -> Path:
    """Latest cfbfastR-data ``team_info`` snapshot directory."""

    return _latest("data/cfb/team_info/raw/*/manifest.json", "cfb team_info snapshot").parent


def default_fluview_paths() -> tuple[Path, ...]:
    """The two FluView snapshots this replication concatenates.

    ``data/raw/fluview/*`` is the ALREADY-FROZEN NFL snapshot (23 NFL states +
    ``nat``, ingested 2026-08-20) -- reused as-is, never re-fetched, so every
    shared state's checkpoint table is bit-identical to the one the NFL battery
    froze. ``data/raw/fluview_cfb/*`` holds only the 19 states that host an FBS
    venue but no NFL franchise, fetched separately into a directory nothing
    else globs so the NFL "latest snapshot" resolution is untouched.
    """

    return (
        _latest("data/raw/fluview/*/fluview_raw.parquet", "NFL fluview_raw.parquet snapshot"),
        _latest("data/raw/fluview_cfb/*/fluview_raw.parquet", "CFB fluview_raw.parquet snapshot"),
    )


def load_team_state_map(team_info_dir: Path | None = None) -> pd.DataFrame:
    """``(season, team_id) -> state`` from the cfbfastR-data team_info snapshot.

    Returns lower-cased two-letter state codes, matching Delphi FluView's own
    region codes (``pa``, ``ca``, ...), so no case normalisation is needed at
    the join.
    """

    directory = team_info_dir or default_team_info_dir()
    paths = sorted(Path(directory).glob("season=*/team_info.parquet"))
    if not paths:
        raise FileNotFoundError(f"no season=*/team_info.parquet files under {directory}")
    frames = []
    for path in paths:
        season = int(path.parent.name.split("=", 1)[1])
        frame = pd.read_parquet(path, columns=list(TEAM_INFO_COLUMNS))
        frame["season"] = season
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table["team_id"] = pd.to_numeric(table["team_id"], errors="coerce").astype("Int64")
    table["state"] = table["state"].astype("string").str.strip().str.lower()
    table = table.loc[table["team_id"].notna()].drop_duplicates(subset=["season", "team_id"])
    return table.loc[:, ["season", "team_id", "state", "venue_id", "venue_name", "city"]]


def load_fluview_panel(paths: tuple[Path, ...] | None = None) -> pd.DataFrame:
    """Concatenate the NFL-state and CFB-only-state FluView snapshots.

    ``nat`` is dropped here (it is a national aggregate, not a state region),
    the same filter ``scripts/fluview_battery_screen.py`` applies before
    building checkpoint tables.
    """

    resolved = paths or default_fluview_paths()
    frames = [pd.read_parquet(path) for path in resolved]
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.loc[panel["region"] != "nat"].reset_index(drop=True)
    duplicated = panel.duplicated(subset=["region", "epiweek", "issue"]).sum()
    if duplicated:
        raise DataContractError(
            f"{duplicated} duplicate (region, epiweek, issue) rows across the FluView "
            "snapshots -- the two snapshots are meant to cover disjoint regions"
        )
    return panel


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def cutoff_dates(gameday: pd.Series) -> pd.Series:
    """The Tuesday on or before ``gameday`` -- the identical formula
    ``scripts/fluview_battery_screen.py::load_schedules`` and
    ``nfl_ats.fluview_production_feature._cutoff_dates`` use (Monday=0, so
    ``tuesday_offset = (weekday - 1) % 7``).

    In CFB this is never after kickoff even for the November Tuesday and
    Wednesday games: a Tuesday kickoff yields an offset of 0 (the cutoff is the
    gameday itself, still pregame), and every other weekday yields a strictly
    earlier date. ``tests/test_fluview_cfb_feature.py`` asserts this on the
    real population, not only on a fixture.
    """

    weekday = pd.to_datetime(gameday).dt.weekday
    offset = (weekday - 1) % 7
    result: pd.Series = pd.to_datetime(gameday) - pd.to_timedelta(offset, unit="D")
    return result


def attach_cfb_market_states(
    features: pd.DataFrame, team_states: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return a working frame carrying ``home_state``/``away_state``/``cutoff_date``.

    The join is on ``(season, team_id)`` -- CFBD/ESPN ids on both sides, never
    a school name (**measured**, ``docs/fluview_cfb_replication.md`` section 4:
    the id join resolves every clean-core row, the name join leaves 39 home and
    47 away rows unresolved).
    """

    missing = sorted(_REQUIRED_COLUMNS.difference(features.columns))
    if missing:
        raise DataContractError(f"CFB features are missing columns: {', '.join(missing)}")

    frame = features.loc[:, sorted(_REQUIRED_COLUMNS)].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame["cutoff_date"] = cutoff_dates(frame["gameday"])
    frame["home_id"] = pd.to_numeric(frame["home_id"], errors="coerce").astype("Int64")
    frame["away_id"] = pd.to_numeric(frame["away_id"], errors="coerce").astype("Int64")
    frame["is_neutral"] = (
        pd.to_numeric(frame["neutral_site"], errors="coerce").fillna(0).astype(int).eq(1)
    )

    states = team_states if team_states is not None else load_team_state_map()
    lookup = states.loc[:, ["season", "team_id", "state"]]
    frame = frame.merge(
        lookup.rename(columns={"team_id": "home_id", "state": "home_state"}),
        on=["season", "home_id"],
        how="left",
    )
    frame = frame.merge(
        lookup.rename(columns={"team_id": "away_id", "state": "away_state"}),
        on=["season", "away_id"],
        how="left",
    )
    frame["home_state"] = frame["home_state"].astype(object)
    frame["away_state"] = frame["away_state"].astype(object)
    return frame


def build_cfb_state_week_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(state, season, week)`` -- the panel the frozen top-decile
    threshold and the split-half reliability are both computed on.

    Mirrors ``scripts/fluview_battery_screen.py::build_state_week_panel``
    exactly (both sides stacked, then de-duplicated on state/season/week, so a
    two-school state contributes ONE panel row per week rather than one per
    game). Neutral-site games are excluded from the panel because the
    home-market mechanism -- and therefore the population the threshold is
    meant to describe -- does not apply there.
    """

    playing = frame.loc[~frame["is_neutral"]]
    home_side = playing.loc[:, ["home_state", "season", "week", "cutoff_date", "home_ili"]].rename(
        columns={"home_state": "state", "home_ili": "ili"}
    )
    away_side = playing.loc[:, ["away_state", "season", "week", "cutoff_date", "away_ili"]].rename(
        columns={"away_state": "state", "away_ili": "ili"}
    )
    panel = pd.concat([home_side, away_side], ignore_index=True).drop_duplicates(
        subset=["state", "season", "week"]
    )
    return panel.loc[panel["state"].notna()].reset_index(drop=True)


def derive_cfb_fluview_features(
    features: pd.DataFrame,
    *,
    fluview_raw: pd.DataFrame | None = None,
    team_states: pd.DataFrame | None = None,
    thresholds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return ``(derived, diagnostics)``.

    ``derived`` is a ``(game_id, cfb_fluview_home_market_elevated,
    cfb_fluview_away_market_elevated)`` frame, point-in-time-safe by
    construction. ``diagnostics`` carries the frozen per-state thresholds, the
    state-week panel, per-season coverage and missingness counts, so the
    harness can report them without re-deriving anything.

    ``thresholds`` defaults to the top decile of each state's own as-of panel
    computed from ``features`` itself. Pass the FULL benchmark table so the
    thresholds are frozen once on the whole population, per
    ``docs/fluview_cfb_replication.md`` section 3; the parameter exists so a
    caller can freeze them explicitly and so tests can inject them.
    """

    frame = attach_cfb_market_states(features, team_states)
    unmapped = int((frame["home_state"].isna() | frame["away_state"].isna()).sum())

    panel_source = fluview_raw if fluview_raw is not None else load_fluview_panel()
    checkpoints = build_checkpoint_tables(panel_source.loc[panel_source["region"] != "nat"])
    frame = attach_asof_ili(frame, checkpoints)

    state_week_panel = build_cfb_state_week_panel(frame)
    resolved_thresholds = (
        thresholds if thresholds is not None else compute_state_thresholds(state_week_panel)
    )

    frame["home_threshold"] = frame["home_state"].map(resolved_thresholds)
    frame["away_threshold"] = frame["away_state"].map(resolved_thresholds)
    home_missing = frame["home_ili"].isna() | frame["home_threshold"].isna()
    away_missing = frame["away_ili"].isna() | frame["away_threshold"].isna()
    inapplicable = frame["is_neutral"]

    home_elevated = np.where(
        frame["home_ili"].to_numpy(dtype=float) >= frame["home_threshold"].to_numpy(dtype=float),
        1.0,
        0.0,
    )
    away_elevated = np.where(
        frame["away_ili"].to_numpy(dtype=float) >= frame["away_threshold"].to_numpy(dtype=float),
        1.0,
        0.0,
    )

    derived = pd.DataFrame(
        {
            "game_id": features["game_id"].to_numpy(),
            CFB_FLUVIEW_HOME_ELEVATED_COLUMN: np.where(
                (home_missing | inapplicable).to_numpy(), np.nan, home_elevated
            ),
            CFB_FLUVIEW_AWAY_ELEVATED_COLUMN: np.where(
                (away_missing | inapplicable).to_numpy(), np.nan, away_elevated
            ),
        }
    )

    coverage = (
        frame.assign(_covered=(~(home_missing | inapplicable)).astype(float))
        .groupby("season")["_covered"]
        .mean()
    )
    diagnostics: dict[str, object] = {
        "state_thresholds": {str(k): float(v) for k, v in sorted(resolved_thresholds.items())},
        "state_week_panel": state_week_panel,
        "n_games": len(frame),
        "n_unmapped_state": unmapped,
        "n_neutral_site": int(inapplicable.sum()),
        "n_home_missing": int(home_missing.sum()),
        "n_away_missing": int(away_missing.sum()),
        "home_coverage_by_season": {str(k): float(v) for k, v in coverage.items()},
        "states_present": sorted(
            {str(s) for s in frame["home_state"].dropna().unique()}
            | {str(s) for s in frame["away_state"].dropna().unique()}
        ),
    }
    return derived, diagnostics


def attach_cfb_fluview_features(
    features: pd.DataFrame,
    *,
    fluview_raw: pd.DataFrame | None = None,
    team_states: pd.DataFrame | None = None,
    thresholds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Additively join the two candidate columns onto ``features``.

    Every pre-existing column comes back bit-identical; only the two new
    columns are added, mirroring
    ``nfl_ats.fluview_production_feature.attach_fluview_elevated_features``'s
    additive-merge discipline. Missing as-of coverage and neutral-site games
    come back NaN on purpose -- imputation belongs to the model's own
    training-fold median, never to a feature builder that can see every season
    at once.
    """

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    collisions = sorted(set(CFB_FLUVIEW_FEATURE_COLUMNS).intersection(features.columns))
    if collisions:
        raise DataContractError(f"features already carries {', '.join(collisions)}")

    derived, diagnostics = derive_cfb_fluview_features(
        features, fluview_raw=fluview_raw, team_states=team_states, thresholds=thresholds
    )
    merged = features.copy()
    for column in CFB_FLUVIEW_FEATURE_COLUMNS:
        merged[column] = derived[column].to_numpy()
    return merged, diagnostics


__all__ = [
    "CFB_FLUVIEW_AWAY_ELEVATED_COLUMN",
    "CFB_FLUVIEW_FEATURE_COLUMNS",
    "CFB_FLUVIEW_HOME_ELEVATED_COLUMN",
    "TEAM_INFO_COLUMNS",
    "attach_cfb_fluview_features",
    "attach_cfb_market_states",
    "build_cfb_state_week_panel",
    "cutoff_dates",
    "default_fluview_paths",
    "default_team_info_dir",
    "derive_cfb_fluview_features",
    "load_fluview_panel",
    "load_team_state_map",
]
