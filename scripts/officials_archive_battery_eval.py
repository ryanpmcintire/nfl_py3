"""Lane AD (LEAD-59 / LEAD-33): what is the officials archive worth to the crew battery?

Predeclared in ``docs/officials_archive_battery.md`` BEFORE any outcome was
read. Read-only against the repository's own data; writes only under
``artifacts/research/laneAD/`` through the sanctioned provenance helpers, and
never runs a registry command -- ``--stage record`` writes the
``nfl-ats weak-signals record`` lines to a PowerShell file for the coordinator.

Stages, in order:

``traits``       Part A. What changes when ``include_archive=True``, trait by
                 trait, game by game. Reads no game outcome.
``opener``       Part B. Opener-graded paired arms against the active model,
                 2020-2025, plus the replay gate against the served opener
                 evaluation artifact.
``era``          Part C. Walk-forward close-proxy grade for the eras the
                 opener store cannot reach (2011-2019), reported as per-era
                 magnitudes beside the 2020-2025 opener era.
``composition``  Part D. LEAD-33's crew-composition statistic: distribution by
                 season and week, plus the one predeclared graded cell.
``record``       Emit ``record_commands.ps1``.

``INCLUDE_ARCHIVE_DEFAULT`` is never touched. Every archive-on arm forces the
flag through :func:`archive_enabled`, a scoped patch of the loader the
consumers import, so the shipped default and the played card are unaffected.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nfl_ats.experiment_runner as experiment_runner
import nfl_ats.officials_archive as officials_archive
import nfl_ats.officials_flag_features as flag_features
from nfl_ats.clv import opener_pick_evaluation
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.margin import fit_margin_model
from nfl_ats.modeling import regular_season_rows
from nfl_ats.officials_archive import CORE_CREW_POSITIONS, load_officials, normalize_position
from nfl_ats.officials_flag_features import (
    ROOKIE_CREW_UNDERDOG_COLUMN,
    ROOKIE_ELIGIBLE_SEASON_FLOOR,
    ROOKIE_PRIOR_EXPERIENCE_MAX,
    SECOND_MEETING_FAVORITE_COLUMN,
)
from nfl_ats.provenance import sha256_file, stamp_sidecar, write_stamped_artifact
from nfl_ats.schedule_flag_features import default_opener_lines
from nfl_ats.weak_stack_v3_features import latest_schedules_snapshot

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts/research/laneAD"
FEATURES = REPO / "data/processed/game_features_weak_stack.parquet"
MARKET_ROOT = REPO / "data/market/raw"
PREDECLARATION = REPO / "docs/officials_archive_battery.md"

SEED = 20260817
SAMPLES = 20_000
#: MOD-18's own minimum training set for a walk-forward week (its ``build_stream``).
MIN_ERA_TRAIN_GAMES = 500
#: docs/officials_archive_battery.md, Part D.
SCRAMBLE_THRESHOLD = 4
SCRAMBLE_FALLBACK_THRESHOLD = 3
SCRAMBLE_MIN_FLAGGED = 30

ERAS: tuple[tuple[str, int, int, str], ...] = (
    ("2011-2014", 2011, 2014, "nflverse spread (close proxy)"),
    ("2015-2019", 2015, 2019, "nflverse spread (close proxy)"),
    ("2020-2025", 2020, 2025, "Tuesday opener"),
)

#: First archived season plus one, mirroring the shipped
#: ``ROOKIE_ELIGIBLE_SEASON_FLOOR`` (2015 feed floor + 1). Used ONLY by the
#: era-floor arms, which exist because the shipped constant is 2016 and makes
#: the rookie flag identically zero on every pre-2016 game.
ARCHIVE_ROOKIE_SEASON_FLOOR = 2010

ARM_PROFILE = {
    "rookie_feed": "weak_stack_rookie_crew_underdog",
    "rookie_archive_loader": "weak_stack_rookie_crew_underdog",
    "rookie_archive_tenure": "weak_stack_rookie_crew_underdog",
    "rookie_archive_era_floor": "weak_stack_rookie_crew_underdog",
    "second_meeting_feed": "weak_stack_crew_second_meeting_favorite",
    "second_meeting_archive": "weak_stack_crew_second_meeting_favorite",
    "second_meeting_archive_era": "weak_stack_crew_second_meeting_favorite",
}

ARM_COLUMN = {
    "rookie_feed": ROOKIE_CREW_UNDERDOG_COLUMN,
    "rookie_archive_loader": ROOKIE_CREW_UNDERDOG_COLUMN,
    "rookie_archive_tenure": ROOKIE_CREW_UNDERDOG_COLUMN,
    "rookie_archive_era_floor": ROOKIE_CREW_UNDERDOG_COLUMN,
    "second_meeting_feed": SECOND_MEETING_FAVORITE_COLUMN,
    "second_meeting_archive": SECOND_MEETING_FAVORITE_COLUMN,
    "second_meeting_archive_era": SECOND_MEETING_FAVORITE_COLUMN,
}

#: The two Part-B arms are the shipped construction; the two ``*_era`` arms are
#: the only ones that can carry a value on a pre-2015 game at all (see
#: ``docs/officials_archive_battery.md``, "Part C amendment").
ERA_ARMS: tuple[str, ...] = (
    "rookie_feed",
    "rookie_archive_tenure",
    "rookie_archive_era_floor",
    "second_meeting_feed",
    "second_meeting_archive_era",
)


@contextmanager
def archive_enabled() -> Iterator[None]:
    """Force ``include_archive=True`` at every consumer's loader, scoped.

    The shipped ``INCLUDE_ARCHIVE_DEFAULT`` is a default argument value bound
    at definition time, so rebinding the module constant would do nothing;
    the loader itself has to be wrapped.
    """

    original = officials_archive.load_officials

    def wrapped(repo_root: Path | None = None, **kwargs: Any) -> pd.DataFrame:
        kwargs["include_archive"] = True
        return original(repo_root, **kwargs)

    experiment_runner.load_officials = wrapped  # type: ignore[assignment]
    flag_features.load_officials = wrapped  # type: ignore[assignment]
    try:
        yield
    finally:
        experiment_runner.load_officials = original  # type: ignore[assignment]
        flag_features.load_officials = original  # type: ignore[assignment]


def jsonable(value: Any) -> Any:
    """numpy/pandas scalars into plain JSON types, recursively."""

    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# ---------------------------------------------------------------------------
# Shared loading
# ---------------------------------------------------------------------------


def active_model(profile: str = "weak_stack") -> dict[str, Any]:
    """The active model's own configuration, with the feature profile swapped."""

    active = json.loads((REPO / "artifacts/active_ats_model.json").read_text(encoding="utf-8"))
    return {
        "feature_profile": profile,
        "regressor": active["regressor"],
        "ridge_alpha": active["ridge_alpha"],
        "target": active["target"],
        "calibration_method": active["calibration_method"],
        "probability_method": active["probability_method"],
    }


def schedules() -> pd.DataFrame:
    frame = pd.read_parquet(latest_schedules_snapshot(REPO))
    return frame.loc[:, ["game_id", "old_game_id", "season", "week", "home_team", "away_team"]]


def referee_game_table(*, include_archive: bool, schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per REG game with a matched head referee, standard ``game_id``.

    The same officials -> ``schedules.old_game_id`` crosswalk
    ``_build_referee_trait_data`` performs, WITHOUT its inner join to the
    per-game penalty aggregate (which exists for 2015-2025 only).
    """

    officials = load_officials(REPO, include_archive=include_archive)
    refs = officials.loc[
        (officials["position"] == experiment_runner._REFEREE_POSITION)
        & (officials["season_type"] == experiment_runner._REFEREE_SEASON_TYPE)
    ]
    joined = refs.merge(
        schedule.rename(columns={"season": "schedule_season", "week": "schedule_week"}),
        left_on="game_id",
        right_on="old_game_id",
        how="inner",
        suffixes=("_legacy", ""),
    )
    out = joined.loc[
        :,
        [
            "game_id",
            "official_name",
            "schedule_season",
            "schedule_week",
            "home_team",
            "away_team",
        ],
    ]
    return out.rename(columns={"schedule_season": "season", "schedule_week": "week"}).astype(
        {"season": int, "week": int}
    )


def tenure_table(games: pd.DataFrame) -> pd.DataFrame:
    """``prior_seasons_experience`` as the docs DEFINE it: the count of distinct
    PRIOR seasons the official appears as ``Referee`` in the officials table."""

    pairs = (
        games.loc[:, ["official_name", "season"]]
        .drop_duplicates()
        .sort_values(["official_name", "season"])
        .reset_index(drop=True)
    )
    pairs["prior_seasons_experience"] = pairs.groupby("official_name").cumcount()
    return games.merge(pairs, on=["official_name", "season"], how="left")


def crew_long_table(*, include_archive: bool, schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, on-field position) with the game's own referee attached."""

    officials = load_officials(REPO, include_archive=include_archive)
    frame = officials.loc[officials["season_type"] == experiment_runner._REFEREE_SEASON_TYPE].copy()
    frame["position"] = frame["position"].map(normalize_position)
    frame = frame.loc[frame["position"].isin(CORE_CREW_POSITIONS)]
    joined = frame.merge(
        schedule.rename(columns={"season": "schedule_season", "week": "schedule_week"}),
        left_on="game_id",
        right_on="old_game_id",
        how="inner",
        suffixes=("_legacy", ""),
    )
    joined = joined.loc[
        :, ["game_id", "schedule_season", "schedule_week", "position", "official_name"]
    ].rename(columns={"schedule_season": "season", "schedule_week": "week"})
    joined = joined.drop_duplicates(["game_id", "position"], keep="first")
    referees = joined.loc[
        joined["position"].eq(experiment_runner._REFEREE_POSITION), ["game_id", "official_name"]
    ].rename(columns={"official_name": "referee"})
    joined = joined.merge(referees, on="game_id", how="inner")
    return joined.astype({"season": int, "week": int}).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The two era-floor arms: the shipped rules with their one 2015-shaped
# constraint lifted, so that a pre-2015 game can carry a value at all
# ---------------------------------------------------------------------------


def _signed_by_line(
    frame: pd.DataFrame, lines: pd.DataFrame, flag: pd.Series, column: str, *, favourite: bool
) -> pd.DataFrame:
    """The shipped sign convention: +1 home side, -1 away side, 0 otherwise."""

    merged = frame.assign(_flag=flag.to_numpy()).merge(
        lines.loc[:, ["game_id", "tue_open_home_spread"]], on="game_id", how="left"
    )
    spread = merged["tue_open_home_spread"]
    home = merged["_flag"] & spread.notna() & (spread.gt(0.0) if favourite else spread.lt(0.0))
    away = merged["_flag"] & spread.notna() & (spread.lt(0.0) if favourite else spread.gt(0.0))
    values = np.where(home, 1.0, np.where(away, -1.0, 0.0))
    out = pd.DataFrame({"game_id": merged["game_id"].astype(str), column: values})
    return out.drop_duplicates("game_id").reset_index(drop=True)


def archive_rookie_era_flag(games: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """``rookie_crew_underdog_flag`` with ONLY the season floor changed.

    The shipped floor is ``ROOKIE_ELIGIBLE_SEASON_FLOOR`` = 2016 (the feed's
    first season plus one), which makes the flag identically zero on every
    pre-2016 game no matter what the archive knows.
    :data:`ARCHIVE_ROOKIE_SEASON_FLOOR` is the same rule applied to the
    archive's own first season (2009 + 1). Nothing else differs.
    """

    rookie = games["prior_seasons_experience"].le(ROOKIE_PRIOR_EXPERIENCE_MAX) & games["season"].ge(
        ARCHIVE_ROOKIE_SEASON_FLOOR
    )
    return _signed_by_line(games, lines, rookie, ROOKIE_CREW_UNDERDOG_COLUMN, favourite=False)


def archive_second_meeting_flag(games: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """``crew_second_meeting_favorite_flag`` without the penalty-aggregate gate.

    ``crew_familiarity_table`` builds this on top of
    ``home_away_penalty_game_table``, whose inner join to
    ``game_penalties.parquet`` confines it to 2015-2025. The flag itself needs
    nothing but referee identity, team identity and week order, so this rebuilds
    it directly on the archive-extended referee table. Reproduction of the
    shipped values on 2015-2025 is asserted by the era stage before use.
    """

    ordered = games.sort_values(["official_name", "season", "week", "game_id"]).reset_index(
        drop=True
    )
    seen: dict[tuple[str, int], set[str]] = {}
    flags: list[bool] = []
    for row in ordered.itertuples(index=False):
        key = (str(row.official_name), int(row.season))
        teams = seen.setdefault(key, set())
        flags.append(bool(row.home_team in teams or row.away_team in teams))
        teams.add(str(row.home_team))
        teams.add(str(row.away_team))
    return _signed_by_line(
        ordered, lines, pd.Series(flags), SECOND_MEETING_FAVORITE_COLUMN, favourite=True
    )


# ---------------------------------------------------------------------------
# Part A: what moves when the archive is switched on
# ---------------------------------------------------------------------------


def change_report(
    before: pd.DataFrame, after: pd.DataFrame, key: str, columns: tuple[str, ...]
) -> dict[str, Any]:
    """Per-column count and magnitude of the rows whose value moved.

    Rows exclusive to either side are counted, never silently dropped.
    """

    left = before.drop_duplicates(key).set_index(key).sort_index()
    right = after.drop_duplicates(key).set_index(key).sort_index()
    shared = left.index.intersection(right.index)
    report: dict[str, Any] = {
        "rows_before": len(left),
        "rows_after": len(right),
        "rows_shared": len(shared),
        "rows_only_before": len(left.index.difference(right.index)),
        "rows_only_after": len(right.index.difference(left.index)),
        "columns": {},
    }
    for column in columns:
        a = left.loc[shared, column]
        b = right.loc[shared, column]
        changed = ~(a.eq(b) | (a.isna() & b.isna()))
        entry: dict[str, Any] = {"n_changed": int(changed.sum())}
        numeric = (pd.to_numeric(b, errors="coerce") - pd.to_numeric(a, errors="coerce")).dropna()
        if len(numeric):
            entry["mean_delta"] = float(numeric.mean())
            entry["max_abs_delta"] = float(numeric.abs().max())
        report["columns"][column] = entry
    return report


def stage_traits() -> dict[str, Any]:
    schedule = schedules()
    result: dict[str, Any] = {"predeclaration": str(PREDECLARATION)}

    shipped_trait = experiment_runner._build_referee_trait_data(REPO).game_trait
    shipped_penalty = flag_features.home_away_penalty_game_table(REPO)
    shipped_censoring = flag_features.describe_referee_left_censoring(REPO)

    with archive_enabled():
        archive_trait = experiment_runner._build_referee_trait_data(REPO).game_trait
        archive_penalty = flag_features.home_away_penalty_game_table(REPO)
        archive_censoring = flag_features.describe_referee_left_censoring(REPO)
        type_traits: dict[str, Any] = {}
        for label, penalty_type in (
            ("dpi", experiment_runner._DPI_PENALTY_TYPE),
            ("holding", experiment_runner._HOLDING_PENALTY_TYPE),
        ):
            try:
                built = experiment_runner._build_referee_type_trait_data(REPO, penalty_type)
                type_traits[label] = {"raised": None, "rows": len(built.game_trait)}
            except Exception as error:
                type_traits[label] = {"raised": f"{type(error).__name__}: {error}", "rows": None}

    result["referee_trait"] = change_report(
        shipped_trait,
        archive_trait,
        "game_id",
        ("prior_seasons_experience", "lag_penalty_rate_quartile", "lag_home_away_diff_quartile"),
    )
    result["home_away_penalty_table_identical"] = bool(
        shipped_penalty.reset_index(drop=True).equals(archive_penalty.reset_index(drop=True))
    )
    result["left_censoring"] = {"archive_off": shipped_censoring, "archive_on": archive_censoring}
    result["referee_type_trait_archive_on"] = type_traits

    feed_games = tenure_table(referee_game_table(include_archive=False, schedule=schedule))
    archive_games = tenure_table(referee_game_table(include_archive=True, schedule=schedule))
    pair_columns = ["official_name", "season"]
    result["referee_game_rows"] = {
        "archive_off": len(feed_games),
        "archive_on": len(archive_games),
        "official_season_pairs_off": len(feed_games.loc[:, pair_columns].drop_duplicates()),
        "official_season_pairs_on": len(archive_games.loc[:, pair_columns].drop_duplicates()),
    }
    shipped_tenure = flag_features.rookie_crew_table(REPO)
    result["shipped_tenure_reproduced"] = change_report(
        shipped_tenure, feed_games, "game_id", ("prior_seasons_experience",)
    )
    result["archive_aware_tenure"] = change_report(
        shipped_tenure, archive_games, "game_id", ("prior_seasons_experience",)
    )

    indexed_shipped = shipped_tenure.drop_duplicates("game_id").set_index("game_id")
    indexed_archive = archive_games.drop_duplicates("game_id").set_index("game_id")
    shared = indexed_shipped.index.intersection(indexed_archive.index)
    rookie_counts = []
    for season, group in indexed_shipped.loc[shared].groupby("season"):
        eligible = int(season) >= ROOKIE_ELIGIBLE_SEASON_FLOOR
        off = group["prior_seasons_experience"].le(ROOKIE_PRIOR_EXPERIENCE_MAX) & eligible
        on = (
            indexed_archive.loc[group.index, "prior_seasons_experience"].le(
                ROOKIE_PRIOR_EXPERIENCE_MAX
            )
            & eligible
        )
        rookie_counts.append(
            {
                "season": int(season),
                "games": len(group),
                "rookie_games_archive_off": int(off.sum()),
                "rookie_games_archive_on": int(on.sum()),
            }
        )
    result["rookie_crew_games_by_season"] = rookie_counts

    flags = flag_columns(schedule)
    result["flag_columns"] = {
        arm: {
            "column": ARM_COLUMN[arm],
            "n_games": len(frame),
            "n_nonzero": int((frame[ARM_COLUMN[arm]] != 0).sum()),
        }
        for arm, frame in flags.items()
    }
    result["flag_column_changes"] = {
        f"{arm}_vs_{reference}": change_report(
            flags[reference], flags[arm], "game_id", (ARM_COLUMN[arm],)
        )
        for arm, reference in (
            ("rookie_archive_loader", "rookie_feed"),
            ("rookie_archive_tenure", "rookie_feed"),
            ("second_meeting_archive", "second_meeting_feed"),
        )
    }
    write_stamped_artifact(jsonable(result), OUT / "traits.json")
    return result


# ---------------------------------------------------------------------------
# The candidate flag columns, one per declared arm
# ---------------------------------------------------------------------------


def opener_line_frame(schedule: pd.DataFrame, *, close_proxy: bool) -> pd.DataFrame:
    """Tuesday-opener consensus lines; with ``close_proxy`` the archived nflverse
    spread fills the pre-2020 seasons the opener store does not reach (MOD-18's
    own era rule, ``docs/spread_regime_program.md`` lines 36-41)."""

    lines = default_opener_lines(schedule.loc[:, ["game_id", "season", "week"]])
    if not close_proxy:
        return lines
    features = pd.read_parquet(FEATURES, columns=["game_id", "spread_line"])
    merged = features.merge(lines, on="game_id", how="left")
    merged["tue_open_home_spread"] = merged["tue_open_home_spread"].fillna(merged["spread_line"])
    return merged.loc[:, ["game_id", "tue_open_home_spread"]]


def flag_columns(schedule: pd.DataFrame, *, close_proxy: bool = False) -> dict[str, pd.DataFrame]:
    """Every declared arm's candidate column, keyed by arm name."""

    lines = opener_line_frame(schedule, close_proxy=close_proxy)
    shipped_trait = flag_features.rookie_crew_table(REPO)
    archive_games = tenure_table(referee_game_table(include_archive=True, schedule=schedule))
    archive_trait = archive_games.loc[
        :, ["game_id", "official_name", "season", "prior_seasons_experience"]
    ]

    columns = {
        "rookie_feed": flag_features.derive_rookie_crew_underdog_features(
            REPO, lines, trait=shipped_trait
        ),
        "rookie_archive_tenure": flag_features.derive_rookie_crew_underdog_features(
            REPO, lines, trait=archive_trait
        ),
        "rookie_archive_era_floor": archive_rookie_era_flag(archive_games, lines),
        "second_meeting_feed": flag_features.derive_second_meeting_favorite_features(REPO, lines),
        "second_meeting_archive_era": archive_second_meeting_flag(archive_games, lines),
    }
    with archive_enabled():
        columns["rookie_archive_loader"] = flag_features.derive_rookie_crew_underdog_features(
            REPO, lines, trait=flag_features.rookie_crew_table(REPO)
        )
        columns["second_meeting_archive"] = flag_features.derive_second_meeting_favorite_features(
            REPO, lines
        )
    return {arm: columns[arm] for arm in ARM_COLUMN}


def second_meeting_reproduction(flags: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Pin the rebuilt second-meeting flag against the shipped one, 2015-2025.

    A second PATH, never a second definition: on the games the shipped builder
    can reach, the two must agree exactly, or the era arm is not the same rule.
    """

    shipped = (
        flags["second_meeting_feed"]
        .drop_duplicates("game_id")
        .set_index("game_id")[SECOND_MEETING_FAVORITE_COLUMN]
    )
    rebuilt = (
        flags["second_meeting_archive_era"]
        .drop_duplicates("game_id")
        .set_index("game_id")[SECOND_MEETING_FAVORITE_COLUMN]
    )
    shared = shipped.index.intersection(rebuilt.index)
    disagreements = int((shipped.loc[shared] != rebuilt.loc[shared]).sum())
    return {
        "shared_games": len(shared),
        "disagreements": disagreements,
        "rebuilt_games": len(rebuilt),
    }


def attach(features: pd.DataFrame, column: str, values: pd.DataFrame) -> pd.DataFrame:
    """Add one candidate column by ``game_id``, preserving the frame's own index."""

    mapping = values.drop_duplicates("game_id").set_index("game_id")[column]
    out = features.copy()
    out[column] = out["game_id"].map(mapping).astype(float).fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Paired scoring
# ---------------------------------------------------------------------------


def paired_accuracy(
    frame: pd.DataFrame,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    margin_column: str = "margin_vs_open",
    samples: int = SAMPLES,
    seed: int = SEED,
) -> dict[str, Any]:
    """Week-blocked paired accuracy delta, in accuracy POINTS.

    Whole season-week blocks are resampled; within-week correlation is ZERO by
    owner mandate and is never estimated or padded. Pushes (final margin
    exactly on the line) are excluded, as every graded read here excludes them.
    """

    margin = frame[margin_column]
    valid = (margin.notna() & margin.ne(0)).to_numpy()
    scored = frame.loc[valid].reset_index(drop=True)
    truth = scored[margin_column].gt(0).to_numpy()
    cp = np.asarray(candidate, dtype=bool)[valid]
    bp = np.asarray(baseline, dtype=bool)[valid]
    diff = (cp == truth).astype(float) - (bp == truth).astype(float)
    groups = list(scored.groupby(["season", "week"], sort=True).indices.values())
    if not groups:
        return {"delta": float("nan"), "n": 0, "weeks": 0, "flips": 0}
    sums = np.array([diff[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(groups), (samples, len(groups)))
    draws = 100 * sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return {
        "delta": float(100 * diff.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "probability_positive": float((draws > 0).mean()),
        "standard_error": float(draws.std(ddof=1)),
        "n": len(scored),
        "weeks": len(groups),
        "candidate_accuracy": float((cp == truth).mean()),
        "baseline_accuracy": float((bp == truth).mean()),
        "flips": int((cp != bp).sum()),
    }


# ---------------------------------------------------------------------------
# Part B: opener-graded arms against the active model
# ---------------------------------------------------------------------------


def served_artifact_id() -> str:
    """The newest opener evaluation artifact built from the ACTIVE model."""

    active = json.loads((REPO / "artifacts/active_ats_model.json").read_text(encoding="utf-8"))
    matches = [
        path.parent.name
        for path in sorted((REPO / "artifacts/opener_evaluation").glob("*/metadata.json"))
        if json.loads(path.read_text(encoding="utf-8")).get("active_model_id") == active["model_id"]
    ]
    if not matches:
        raise ValueError("No opener evaluation artifact matches the active model")
    return matches[-1]


def replay_gate(baseline: pd.DataFrame, served: pd.DataFrame, artifact: str) -> dict[str, Any]:
    """Refuse to score anything if the harness does not reproduce the served read."""

    merged = baseline.merge(served, on="game_id", suffixes=("_new", "_served"))
    gap = float(
        (
            merged["home_cover_probability_at_open_new"]
            - merged["home_cover_probability_at_open_served"]
        )
        .abs()
        .max()
    )
    disagreements = int(
        (
            merged["pick_home_at_open_probability_rule_new"]
            != merged["pick_home_at_open_probability_rule_served"]
        ).sum()
    )
    if gap > 1e-9 or disagreements:
        raise ValueError(
            f"STOP: opener replay mismatch (max probability gap {gap}, {disagreements} picks)"
        )
    return {
        "games": len(merged),
        "max_probability_gap": gap,
        "pick_disagreements": disagreements,
        "served_artifact": artifact,
        "feature_table_sha256": sha256_file(FEATURES),
    }


def score_opener_arm(features: pd.DataFrame, profile: str) -> pd.DataFrame:
    scored = opener_pick_evaluation(
        MARKET_ROOT,
        features,
        active_model_config=active_model(profile),
        min_train_games=DEFAULT_MIN_TRAIN_GAMES,
    )
    return scored.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def stage_opener() -> dict[str, Any]:
    schedule = schedules()
    base_features = pd.read_parquet(FEATURES)
    flags = flag_columns(schedule)
    artifact = served_artifact_id()

    with threadpool_limits(limits=1):
        baseline = score_opener_arm(base_features, "weak_stack")
        served = pd.read_parquet(
            REPO / "artifacts/opener_evaluation" / artifact / "per_game.parquet"
        )
        gate = replay_gate(baseline, served, artifact)
        arms: dict[str, pd.DataFrame] = {}
        identical: dict[str, str] = {}
        for arm, values in flags.items():
            column = ARM_COLUMN[arm]
            twin = next(
                (
                    other
                    for other in arms
                    if ARM_PROFILE[other] == ARM_PROFILE[arm]
                    and flags[other]
                    .set_index("game_id")[column]
                    .sort_index()
                    .equals(values.set_index("game_id")[column].sort_index())
                ),
                None,
            )
            if twin is not None:
                identical[arm] = twin
                arms[arm] = arms[twin]
                continue
            arms[arm] = score_opener_arm(attach(base_features, column, values), ARM_PROFILE[arm])

    for arm, scored in arms.items():
        if not scored["game_id"].equals(baseline["game_id"]):
            raise ValueError(f"arm {arm} scored a different game set than production")

    result: dict[str, Any] = {
        "replay_gate": gate,
        "arms_identical_to": identical,
        "windows": {},
    }
    truth = baseline.loc[:, ["game_id", "season", "week", "margin_vs_open"]]
    for label, seasons in (("2020_2025", tuple(range(2020, 2026))), ("2020_2021", (2020, 2021))):
        mask = truth["season"].isin(list(seasons)).to_numpy()
        window = truth.loc[mask].reset_index(drop=True)
        base_pick = baseline.loc[mask, "pick_home_at_open_probability_rule"].to_numpy()
        cells: dict[str, Any] = {}
        for arm, scored in arms.items():
            cells[f"{arm}_vs_production"] = paired_accuracy(
                window,
                scored.loc[mask, "pick_home_at_open_probability_rule"].to_numpy(),
                base_pick,
            )
        for arm, reference in (
            ("rookie_archive_loader", "rookie_feed"),
            ("rookie_archive_tenure", "rookie_feed"),
            ("second_meeting_archive", "second_meeting_feed"),
        ):
            cells[f"{arm}_vs_{reference}"] = paired_accuracy(
                window,
                arms[arm].loc[mask, "pick_home_at_open_probability_rule"].to_numpy(),
                arms[reference].loc[mask, "pick_home_at_open_probability_rule"].to_numpy(),
            )
        result["windows"][label] = cells

    picks = truth.copy()
    picks["production"] = baseline["pick_home_at_open_probability_rule"].to_numpy()
    for arm, scored in arms.items():
        picks[arm] = scored["pick_home_at_open_probability_rule"].to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    picks.to_parquet(OUT / "opener_picks.parquet", index=False)
    stamp_sidecar(OUT / "opener_picks.parquet")
    write_stamped_artifact(jsonable(result), OUT / "opener.json")
    return result


PROXY_LINE_ARMS: tuple[str, ...] = ("rookie_archive_era_floor", "second_meeting_archive_era")


def stage_served_proxy() -> dict[str, Any]:
    """The served opener harness, with the flags built on close-proxy lines.

    Part B measured that the archive cannot reach the served read while the
    flag is built from the Tuesday-opener store alone: that store starts in
    2020, so every pre-2020 training game carries a zero flag whatever the
    archive knows. This stage relaxes exactly one thing -- the flag's own line
    source, filled with the archived nflverse spread before 2020 (the same
    close-proxy substitution MOD-18's era rule already sanctions) -- and scores
    the result through the SAME served harness Part B's replay gate validated.
    It is the decision-grade version of the era stream's 2020-2025 row.
    """

    schedule = schedules()
    base_features = pd.read_parquet(FEATURES)
    proxy = flag_columns(schedule, close_proxy=True)
    shipped = flag_columns(schedule, close_proxy=False)
    artifact = served_artifact_id()

    with threadpool_limits(limits=1):
        baseline = score_opener_arm(base_features, "weak_stack")
        served = pd.read_parquet(
            REPO / "artifacts/opener_evaluation" / artifact / "per_game.parquet"
        )
        gate = replay_gate(baseline, served, artifact)
        arms = {
            arm: score_opener_arm(
                attach(base_features, ARM_COLUMN[arm], proxy[arm]), ARM_PROFILE[arm]
            )
            for arm in PROXY_LINE_ARMS
        }
        reference = {
            arm: score_opener_arm(
                attach(base_features, ARM_COLUMN[arm], shipped[arm]), ARM_PROFILE[arm]
            )
            for arm in PROXY_LINE_ARMS
        }

    truth = baseline.loc[:, ["game_id", "season", "week", "margin_vs_open"]]
    base_pick = baseline["pick_home_at_open_probability_rule"].to_numpy()
    cells: dict[str, Any] = {}
    for arm in PROXY_LINE_ARMS:
        pick = arms[arm]["pick_home_at_open_probability_rule"].to_numpy()
        cells[f"{arm}_proxyline_vs_production"] = paired_accuracy(truth, pick, base_pick)
        cells[f"{arm}_proxyline_vs_openerline"] = paired_accuracy(
            truth, pick, reference[arm]["pick_home_at_open_probability_rule"].to_numpy()
        )
    result = {"replay_gate": gate, "cells": cells}
    write_stamped_artifact(jsonable(result), OUT / "opener_proxy.json")
    return result


# ---------------------------------------------------------------------------
# Part C: the era-extended walk-forward read (close proxy before 2020)
# ---------------------------------------------------------------------------


def stage_era() -> dict[str, Any]:
    schedule = schedules()
    base = regular_season_rows(pd.read_parquet(FEATURES)).reset_index(drop=True)
    base["gameday"] = pd.to_datetime(base["gameday"])
    flags = flag_columns(schedule, close_proxy=True)
    lines = opener_line_frame(schedule, close_proxy=True)
    opener_ids = set(
        default_opener_lines(schedule.loc[:, ["game_id", "season", "week"]])["game_id"]
    )

    reproduction = second_meeting_reproduction(flags)
    if reproduction["disagreements"]:
        raise ValueError(
            "STOP: the rebuilt second-meeting flag disagrees with the shipped builder on "
            f"{reproduction['disagreements']} games"
        )
    nonzero = {
        arm: {
            "pre_2015": int(
                (
                    flags[arm]
                    .merge(schedule.loc[:, ["game_id", "season"]], on="game_id", how="left")
                    .query("season < 2015")[ARM_COLUMN[arm]]
                    != 0
                ).sum()
            ),
            "total": int((flags[arm][ARM_COLUMN[arm]] != 0).sum()),
        }
        for arm in ERA_ARMS
    }

    frames = {"base": base}
    for arm in ERA_ARMS:
        frames[arm] = attach(base, ARM_COLUMN[arm], flags[arm])

    completed = base.loc[base["result"].notna()].sort_values(["gameday", "game_id"])
    targets = base.loc[base["season"].between(2011, 2025) & base["result"].notna()]
    batches = []
    with threadpool_limits(limits=1):
        for (season, week), group in targets.groupby(["season", "week"], sort=True):
            training = completed.loc[completed["gameday"].lt(group["gameday"].min())]
            if len(training) < MIN_ERA_TRAIN_GAMES:
                continue
            quoted = (
                group[["game_id"]]
                .merge(lines, on="game_id", how="left")["tue_open_home_spread"]
                .to_numpy()
            )
            row = group[["game_id", "season", "week"]].copy()
            row["line"] = np.where(pd.isna(quoted), group["spread_line"].to_numpy(), quoted)
            row["grade"] = np.where(
                group["game_id"].isin(opener_ids).to_numpy(), "opener", "close_proxy"
            )
            row["margin_vs_line"] = group["result"].to_numpy() - row["line"].to_numpy()
            for arm, frame in frames.items():
                profile = "weak_stack" if arm == "base" else ARM_PROFILE[arm]
                model = fit_margin_model(
                    frame.loc[training.index],
                    target="market_residual",
                    model_name="ridge",
                    feature_profile=profile,
                    ridge_alpha=10.0,
                )
                scoring = frame.loc[group.index].copy()
                scoring["spread_line"] = row["line"].to_numpy()
                predicted = model.predict(scoring, probability_method="gaussian_median")
                row[f"p_{arm}"] = predicted["home_cover_probability"].to_numpy()
            batches.append(row)
            if int(week) == 1:
                print(f"era stream {season} (training rows {len(training)})", flush=True)

    stream = pd.concat(batches, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stream.to_parquet(OUT / "era_stream.parquet", index=False)
    stamp_sidecar(OUT / "era_stream.parquet")

    cells: dict[str, Any] = {}
    for label, start, end, grade in ERAS:
        era = stream.loc[stream["season"].between(start, end)].reset_index(drop=True)
        if era.empty:
            cells[label] = {"grade": grade, "games": 0}
            continue
        base_pick = era["p_base"].ge(0.5).to_numpy()
        entry: dict[str, Any] = {
            "grade": grade,
            "grade_labels": {str(k): int(v) for k, v in era["grade"].value_counts().items()},
            "games": len(era),
            "base_accuracy": float(
                (
                    base_pick[era["margin_vs_line"].ne(0).to_numpy()]
                    == era.loc[era["margin_vs_line"].ne(0), "margin_vs_line"].gt(0).to_numpy()
                ).mean()
            ),
        }
        for arm in ERA_ARMS:
            entry[arm] = paired_accuracy(
                era, era[f"p_{arm}"].ge(0.5).to_numpy(), base_pick, margin_column="margin_vs_line"
            )
        cells[label] = entry
    result = {
        "eras": cells,
        "arms": list(ERA_ARMS),
        "stream_rows": len(stream),
        "second_meeting_reproduction": reproduction,
        "nonzero_flag_games": nonzero,
    }
    write_stamped_artifact(jsonable(result), OUT / "era.json")
    return result


# ---------------------------------------------------------------------------
# Part D: LEAD-33's crew-composition statistic
# ---------------------------------------------------------------------------


def _modal(counts: Counter[str], most_recent: str | None) -> str | None:
    if not counts:
        return None
    best = max(counts.values())
    tied = sorted(name for name, value in counts.items() if value == best)
    if most_recent is not None and most_recent in tied:
        return most_recent
    return tied[0]


def crew_composition(long: pd.DataFrame) -> pd.DataFrame:
    """Per game: how many of the seven positions sit off the referee's modal crew.

    ``positions_off_modal_crew_prior`` uses only that official's EARLIER games
    in the same season, so it is pregame-safe and is the only variant graded.
    ``positions_off_modal_crew_season`` uses the whole season and is
    DESCRIPTIVE ONLY -- it can see games that had not been played. Ties in a
    modal crew are broken by the most recent qualifying game's referee, else
    alphabetically.
    """

    rows = long.sort_values(["season", "official_name", "week", "game_id"]).reset_index(drop=True)
    season_counts: dict[tuple[int, str], Counter[str]] = {}
    season_last: dict[tuple[int, str], str] = {}
    for row in rows.itertuples(index=False):
        key = (int(row.season), str(row.official_name))
        season_counts.setdefault(key, Counter())[str(row.referee)] += 1
        season_last[key] = str(row.referee)

    prior_counts: dict[tuple[int, str], Counter[str]] = {}
    prior_last: dict[tuple[int, str], str] = {}
    off_prior: list[float] = []
    resolved_prior: list[float] = []
    off_season: list[float] = []
    for row in rows.itertuples(index=False):
        key = (int(row.season), str(row.official_name))
        modal_prior = _modal(prior_counts.get(key, Counter()), prior_last.get(key))
        resolved_prior.append(1.0 if modal_prior is not None else 0.0)
        off_prior.append(
            1.0 if modal_prior is not None and modal_prior != str(row.referee) else 0.0
        )
        off_season.append(
            1.0 if _modal(season_counts[key], season_last[key]) != str(row.referee) else 0.0
        )
        prior_counts.setdefault(key, Counter())[str(row.referee)] += 1
        prior_last[key] = str(row.referee)

    rows = rows.assign(off_prior=off_prior, resolved_prior=resolved_prior, off_season=off_season)
    grouped = rows.groupby(["game_id", "season", "week"], as_index=False).agg(
        positions=("position", "size"),
        positions_off_modal_crew_prior=("off_prior", "sum"),
        positions_resolved=("resolved_prior", "sum"),
        positions_off_modal_crew_season=("off_season", "sum"),
    )
    for column in (
        "positions",
        "positions_off_modal_crew_prior",
        "positions_resolved",
        "positions_off_modal_crew_season",
    ):
        grouped[column] = grouped[column].astype(int)
    return grouped.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def stage_composition() -> dict[str, Any]:
    schedule = schedules()
    table = crew_composition(crew_long_table(include_archive=True, schedule=schedule))
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT / "crew_composition.parquet", index=False)
    stamp_sidecar(OUT / "crew_composition.parquet")

    aggregate = {
        "games": ("game_id", "size"),
        "mean_prior": ("positions_off_modal_crew_prior", "mean"),
        "mean_season": ("positions_off_modal_crew_season", "mean"),
        "mean_resolved": ("positions_resolved", "mean"),
        "mean_positions": ("positions", "mean"),
    }
    by_season = table.groupby("season").agg(**aggregate).reset_index()
    by_week = table.groupby("week").agg(**aggregate).reset_index()

    picks = pd.read_parquet(OUT / "opener_picks.parquet")
    artifact = served_artifact_id()
    opener = pd.read_parquet(
        REPO / "artifacts/opener_evaluation" / artifact / "per_game.parquet",
        columns=["game_id", "tue_open_home_spread"],
    )
    graded = picks.merge(
        table.loc[:, ["game_id", "positions_off_modal_crew_prior", "positions_resolved"]],
        on="game_id",
        how="left",
    ).merge(opener, on="game_id", how="left")
    graded["positions_off_modal_crew_prior"] = graded["positions_off_modal_crew_prior"].fillna(0)

    threshold = SCRAMBLE_THRESHOLD
    flagged = graded["positions_off_modal_crew_prior"].ge(threshold)
    fallback_used = False
    if int(flagged.sum()) < SCRAMBLE_MIN_FLAGGED:
        threshold = SCRAMBLE_FALLBACK_THRESHOLD
        flagged = graded["positions_off_modal_crew_prior"].ge(threshold)
        fallback_used = True
    quoted = graded["tue_open_home_spread"]
    rule = np.where(
        (flagged & quoted.notna() & quoted.ne(0)).to_numpy(),
        quoted.gt(0).to_numpy(),
        graded["production"].to_numpy(),
    )
    cell = paired_accuracy(graded, rule, graded["production"].to_numpy())
    graded = graded.assign(scramble_flagged=flagged.to_numpy(), rule_pick=rule)
    graded.to_parquet(OUT / "crew_scramble_picks.parquet", index=False)
    stamp_sidecar(OUT / "crew_scramble_picks.parquet")

    result = {
        "threshold": threshold,
        "fallback_used": fallback_used,
        "flagged_games_in_graded_window": int(flagged.sum()),
        "distribution_prior": {
            str(key): int(value)
            for key, value in table["positions_off_modal_crew_prior"]
            .value_counts()
            .sort_index()
            .items()
        },
        "distribution_season": {
            str(key): int(value)
            for key, value in table["positions_off_modal_crew_season"]
            .value_counts()
            .sort_index()
            .items()
        },
        "by_season": json.loads(by_season.to_json(orient="records")),
        "by_week": json.loads(by_week.to_json(orient="records")),
        "crew_scramble_backs_favorite": cell,
    }
    write_stamped_artifact(jsonable(result), OUT / "composition.json")
    return result


# ---------------------------------------------------------------------------
# Through the played card: the only rule in this lane that moves a pick
# ---------------------------------------------------------------------------


def _load_by_path(name: str) -> Any:
    """Import a sibling script by file location (``scripts/`` is not a package)."""

    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - environment guard
        raise ValueError(f"cannot load scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stage_card() -> dict[str, Any]:
    """Run the crew-scramble rule through the composed card, not just standalone.

    Reuses ``scripts/spread_regime_opener_eval.composed_picks`` verbatim: it
    returns the four-member union and the three-member union, and the
    three-member one is the PLAYED card (the spread-gap zone flip was retired
    on 2026-09-07).
    """

    mod18 = _load_by_path("spread_regime_opener_eval")
    artifact = served_artifact_id()
    served_dir = REPO / "artifacts/opener_evaluation" / artifact
    served = pd.read_parquet(served_dir / "per_game.parquet")
    graded = pd.read_parquet(OUT / "crew_scramble_picks.parquet").set_index("game_id")

    candidate = served.copy()
    picks = graded.loc[candidate["game_id"], "rule_pick"].to_numpy(dtype=bool)
    candidate["pick_home_at_open_probability_rule"] = picks
    candidate["correct_at_open_probability_rule"] = (
        pd.Series(picks, index=candidate.index)
        .eq(candidate["margin_vs_open"].gt(0))
        .astype(float)
        .where(candidate["margin_vs_open"].ne(0))
    )
    directory = OUT / "opener_evaluation" / "crew_scramble"
    directory.mkdir(parents=True, exist_ok=True)
    candidate.to_parquet(directory / "per_game.parquet", index=False)
    stamp_sidecar(directory / "per_game.parquet")
    metadata = json.loads((served_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata.update(
        research_arm="crew_scramble_backs_favorite",
        incumbent_model_id=metadata.get("active_model_id"),
        active_model_id="research_laneAD_crew_scramble",
        research_scope="Opener pick columns replaced by the rule; close columns untouched.",
    )
    metadata["active_model_config"] = {
        **metadata["active_model_config"],
        "model_id": "research_laneAD_crew_scramble",
    }
    write_stamped_artifact(metadata, directory / "metadata.json")

    incumbent_four, incumbent_three = mod18.composed_picks(served, served_dir / "per_game.parquet")
    candidate_four, candidate_three = mod18.composed_picks(
        candidate, directory / "per_game.parquet"
    )
    frame = served.loc[:, ["game_id", "season", "week", "margin_vs_open"]]
    result = {
        "served_artifact": artifact,
        "standalone_vs_production": paired_accuracy(
            frame,
            candidate["pick_home_at_open_probability_rule"].to_numpy(),
            served["pick_home_at_open_probability_rule"].to_numpy(),
        ),
        "through_played_card": paired_accuracy(frame, candidate_three, incumbent_three),
        "through_four_member_union": paired_accuracy(frame, candidate_four, incumbent_four),
    }
    write_stamped_artifact(jsonable(result), OUT / "card.json")
    return result


# ---------------------------------------------------------------------------
# Record commands (written, never run, by this lane)
# ---------------------------------------------------------------------------


PLAIN_SUMMARY = {
    "rookie_feed": (
        "When the referee crew is in its first or second year on record, this rule backs the "
        "underdog. This version only knows officiating crews from 2015 onward."
    ),
    "rookie_archive_loader": (
        "The same first-or-second-year referee rule, with the 2009-2014 crew history switched "
        "on where the data is loaded."
    ),
    "rookie_archive_tenure": (
        "The same first-or-second-year referee rule, except a referee who already worked games "
        "before 2015 now counts as experienced instead of looking brand new."
    ),
    "second_meeting_feed": (
        "When the officiating crew has already worked a game involving either of these two "
        "teams this season, this rule backs the favourite."
    ),
    "second_meeting_archive": (
        "The same repeat-crew rule with the older seasons of crew history switched on."
    ),
    "rookie_archive_era_floor": (
        "The first-or-second-year referee rule extended back to the older seasons, so a crew "
        "working in 2011 can be judged new or experienced instead of being skipped."
    ),
    "second_meeting_archive_era": (
        "The repeat-crew rule extended back to the older seasons, so a 2011 game can be marked "
        "as a crew's second look at one of these teams."
    ),
}


def _record_line(
    name: str,
    description: str,
    plain: str,
    metrics: dict[str, Any],
    start: int,
    end: int,
    source: str,
    notes: str,
) -> str:
    fields = [
        ".\\.tools\\uv.exe run --no-sync nfl-ats weak-signals record",
        f'--name "{name}"',
        '--family "officials_archive_battery"',
        f'--description "{description}"',
        f'--plain-summary "{plain}"',
        f'--source "{source}"',
        '--classification "unresolved_below_power"',
        '--classification-evidence "No refuted mechanism and no positive-control bound; '
        'retained unresolved per AGENTS.md."',
        '--league "nfl"',
        f"--season-start {start}",
        f"--season-end {end}",
        '--category "onfield"',
        '--effect-units "accuracy_points"',
        f"--effect {metrics['delta']}",
        f"--interval-low {metrics['lower']}",
        f"--interval-high {metrics['upper']}",
        f"--probability-positive {metrics['probability_positive']}",
        f"--sample-games {metrics['n']}",
        f"--sample-blocks {metrics['weeks']}",
        f'--notes "{notes}"',
    ]
    return " ".join(fields)


def stage_record() -> dict[str, Any]:
    opener = json.loads((OUT / "opener.json").read_text(encoding="utf-8"))
    era = json.loads((OUT / "era.json").read_text(encoding="utf-8"))
    composition = json.loads((OUT / "composition.json").read_text(encoding="utf-8"))
    lines: list[str] = [
        "# Lane AD (LEAD-59 / LEAD-33) record commands.",
        "# Written by scripts/officials_archive_battery_eval.py --stage record.",
        "# This lane never runs them; the coordinator runs them serially.",
        "$ErrorActionPreference = 'Stop'",
    ]

    window = opener["windows"]["2020_2025"]
    aliases = opener.get("arms_identical_to", {})
    for arm, plain in PLAIN_SUMMARY.items():
        metrics = window.get(f"{arm}_vs_production")
        if not metrics or not metrics.get("weeks"):
            continue
        if arm in aliases:
            lines.append(
                f"# SKIPPED {arm} vs production: this arm's candidate column is identical to "
                f"{aliases[arm]}'s on every game, so its picks and its number are the same "
                "measurement under a second name. Recorded once, under the twin."
            )
            continue
        lines.append(
            _record_line(
                f"officials_archive_{arm}_opener_2020_2025",
                f"Opener-graded paired accuracy of the {arm} arm stacked on production weak_stack",
                plain,
                metrics,
                2020,
                2025,
                "artifacts/research/laneAD/opener.json",
                "Lane AD; 2020-2025 opener window, correlated with the recorded 2020-2021 "
                "screen for the same rule; descriptive reuse, not independent confirmation.",
            )
        )
    for arm, reference in (
        ("rookie_archive_loader", "rookie_feed"),
        ("rookie_archive_tenure", "rookie_feed"),
        ("second_meeting_archive", "second_meeting_feed"),
    ):
        metrics = window.get(f"{arm}_vs_{reference}")
        if not metrics or not metrics.get("weeks"):
            continue
        if metrics["flips"] == 0:
            # Not a measurement: the two arms' candidate columns are identical
            # on every game, so the fitted model, the picks and the delta are
            # the same object twice. Recording an exact algebraic zero with a
            # degenerate `probability_positive` would dress a proof up as an
            # estimate. The identity is reported in
            # docs/officials_archive_battery.md instead.
            lines.append(
                f"# SKIPPED {arm} vs {reference}: identical candidate columns on all "
                f"{metrics['n']} games, zero picks changed, delta exactly 0. An identity, "
                "not an estimate -- see docs/officials_archive_battery.md."
            )
            continue
        lines.append(
            _record_line(
                f"officials_archive_delta_{arm}_opener_2020_2025",
                f"Opener-graded paired accuracy of {arm} against its archive-off twin",
                "What switching the older seasons of officiating history on is worth to this "
                "rule, on the seasons the pool actually grades.",
                metrics,
                2020,
                2025,
                "artifacts/research/laneAD/opener.json",
                "Lane AD; archive-on minus archive-off on the same games and the same model.",
            )
        )
    for label, entry in era["eras"].items():
        start, end = (int(part) for part in label.split("-"))
        for arm in ERA_ARMS:
            metrics = entry.get(arm)
            if not metrics or not metrics.get("weeks"):
                continue
            if metrics["flips"] == 0:
                lines.append(
                    f"# SKIPPED {arm} in {label}: the candidate column is identically zero on "
                    "every game in this era, so the arm IS production. A structural fact about "
                    "what the data can reach, not an estimate."
                )
                continue
            lines.append(
                _record_line(
                    f"officials_archive_{arm}_{start}_{end}",
                    f"Paired accuracy of the {arm} arm on production, {label}, "
                    f"graded at the {entry['grade']}",
                    PLAIN_SUMMARY[arm] + f" Measured on the {label} seasons.",
                    metrics,
                    start,
                    end,
                    "artifacts/research/laneAD/era.json",
                    f"Lane AD era read; grade {entry['grade']}; a close-proxy number never "
                    "stands in for an opener number.",
                )
            )
    proxy_path = OUT / "opener_proxy.json"
    if proxy_path.is_file():
        proxy = json.loads(proxy_path.read_text(encoding="utf-8"))["cells"]
        for arm in PROXY_LINE_ARMS:
            for suffix, description, notes in (
                (
                    "vs_production",
                    "stacked on production weak_stack, served opener harness",
                    "Lane AD; the archive-extended flag built on a close-proxy line before "
                    "2020, scored through the same served harness the replay gate validated.",
                ),
                (
                    "vs_openerline",
                    "against the same rule built from the Tuesday-opener store alone",
                    "Lane AD; this IS the archive-delta cell for this rule -- what the older "
                    "seasons of crew history are worth once the flag is allowed to use them. "
                    "Added after the predeclared arms measured as structurally empty, so it "
                    "carries a second-look discount: descriptive, not confirmation.",
                ),
            ):
                metrics = proxy.get(f"{arm}_proxyline_{suffix}")
                if not metrics or not metrics.get("weeks"):
                    continue
                lines.append(
                    _record_line(
                        f"officials_archive_{arm}_proxyline_{suffix}_opener_2020_2025",
                        f"Opener-graded paired accuracy of {arm} {description}",
                        PLAIN_SUMMARY[arm],
                        metrics,
                        2020,
                        2025,
                        "artifacts/research/laneAD/opener_proxy.json",
                        notes,
                    )
                )
    cell = composition["crew_scramble_backs_favorite"]
    if cell.get("weeks"):
        lines.append(
            _record_line(
                "officials_archive_crew_scramble_backs_favorite_opener_2020_2025",
                "Opener-graded paired accuracy of backing the favourite when a majority of the "
                "seven officials normally work under other referees",
                "When most of the seven officials working a game normally work under different "
                "referees, this rule backs the favourite instead of the model's own pick.",
                cell,
                2020,
                2025,
                "artifacts/research/laneAD/composition.json",
                "Lane AD, LEAD-33; one predeclared descriptive cell on a mined population; a "
                "pick rule on top of the served picks, not a feature inside the model.",
            )
        )
    card_path = OUT / "card.json"
    if card_path.is_file():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        for key, description in (
            ("through_played_card", "through the played three-member card"),
            ("through_four_member_union", "through the four-member union (zone included)"),
        ):
            metrics = card.get(key)
            if not metrics or not metrics.get("weeks"):
                continue
            lines.append(
                _record_line(
                    f"officials_archive_crew_scramble_{key}_2020_2025",
                    f"Paired accuracy of the crew-scramble favourite rule {description}",
                    "What the scrambled-crew rule is worth once the card's other rules have "
                    "had their say, rather than on its own.",
                    metrics,
                    2020,
                    2025,
                    "artifacts/research/laneAD/card.json",
                    "Lane AD, LEAD-33; composed read on the served opener archive; the "
                    "three-member union is the card actually played.",
                )
            )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "record_commands.ps1").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"commands": len(lines) - 4, "path": str(OUT / "record_commands.ps1")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("traits", "opener", "served-proxy", "era", "composition", "card", "record"),
        required=True,
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stages = {
        "traits": stage_traits,
        "opener": stage_opener,
        "served-proxy": stage_served_proxy,
        "era": stage_era,
        "composition": stage_composition,
        "card": stage_card,
        "record": stage_record,
    }
    print(json.dumps(jsonable(stages[args.stage]()), indent=2, default=str)[:14000], flush=True)


if __name__ == "__main__":
    main()
