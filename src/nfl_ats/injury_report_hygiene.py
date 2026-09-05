"""Injury-report text hygiene measurements for the availability model.

Pure, side-effect-free functions backing two Phase 12 QUALITY leads
(ROADMAP.md ``LEAD-18``, ``LEAD-19``): neither carries an ATS direction or a
rotation window, so nothing here touches ``nfl_ats.rotation`` or
``nfl_ats.weak_signals``. Both leads measure how well
``nfl_ats.availability``'s ``report_category`` / ``practice_category`` /
``position_group`` feature construction (see ``src/nfl_ats/availability.py``)
matches what actually happens on Sunday, using the raw nflverse text fields
(``report_primary_injury``, ``practice_primary_injury``) that
``nfl_ats.players.canonicalize_injuries`` drops before the availability model
ever sees them.

All population definitions (the frozen designation string sets, the
skill/line position split, the DNP status string, the era boundary) are
predeclared in ``docs/injury_report_hygiene.md`` BEFORE the numbers in that
document's Results section were computed. ``scripts/injury_report_hygiene_screen.py``
is the only caller that touches a filesystem path outside ``artifacts/``;
everything in this module operates on in-memory frames the caller supplies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError, require_columns
from nfl_ats.pbp import season_scope_mask
from nfl_ats.players import attach_snap_player_ids, canonicalize_rosters, canonicalize_snaps

#: nflverse's raw per-report free-text columns that
#: ``nfl_ats.players.canonicalize_injuries`` selects away (see
#: ``INJURY_REQUIRED_COLUMNS`` there). This module needs them, so it
#: canonicalizes the raw frame itself instead of reusing that function.
INJURY_TEXT_REQUIRED_COLUMNS = (
    "season",
    "game_type",
    "team",
    "week",
    "gsis_id",
    "position",
    "report_status",
    "report_primary_injury",
    "practice_status",
    "practice_primary_injury",
    "date_modified",
)

#: Exact practice-status string nflverse uses for a Wednesday/Thursday/Friday
#: "did not practice" designation. Measured
#: (``data/raw/nflverse_injuries/20260826T122850Z/injuries.parquet``,
#: REG rows, ``practice_status`` value counts): 24,691 of 90,752 raw rows.
#: Deliberately distinct from ``"Out (Definitely Will Not Play)"`` (974 raw
#: rows), which is a separate, stricter designation this module does not
#: fold in.
DID_NOT_PARTICIPATE_STATUS = "Did Not Participate In Practice"

#: LEAD-18's predeclared "skill" and literal "OL/DL" line position sets.
#: Deliberately narrower than ``nfl_ats.availability.position_group``'s
#: ``_FRONT`` (which folds LB/OLB/ILB in with DE/DT/NT) because LEAD-18's
#: predeclared hypothesis names "OL/DL" specifically, not "the defensive
#: front seven" -- widening it to ``_FRONT`` would silently change the
#: comparison the ROADMAP row promised. ``FB``/``HB`` are included on the
#: skill side for consistency with ``nfl_ats.players._SKILL`` /
#: ``nfl_ats.availability._SKILL``, though neither appears in the measured
#: 2013-2025 concussion+DNP population.
CONCUSSION_SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE", "FB", "HB"})
CONCUSSION_LINE_POSITIONS = frozenset(
    {"C", "G", "OG", "OL", "OT", "T", "DE", "DL", "DT", "NT", "EDGE"}
)

#: Frozen, exact-match (lowercased, stripped) non-injury designation
#: vocabularies. Built by grepping the distinct ``report_primary_injury`` /
#: ``practice_primary_injury`` values in the 2013-2025 REG rows of
#: ``data/raw/nflverse_injuries/20260826T122850Z/injuries.parquet`` for
#: "personal", "not injury", "rest", "illness", "coach", "suspend", "travel",
#: "discipline", "team decision" and hand-sorting the distinct hits into the
#: five buckets below (see docs/injury_report_hygiene.md section 2 for the
#: full grep output). Matching is EXACT against these frozen strings, never
#: substring: compound entries that mix a body part with a non-injury tag
#: (e.g. ``"Ankle [Not Injury Related - Personal, Thursday Only]"``,
#: ``"Knee/Rested"``) and one-off narrative sentences (e.g. "Did not travel
#: to Brazil due to a personal matter...") are deliberately left OUT of every
#: bucket -- they fall through to ``"injury"`` in :func:`classify_designation`
#: rather than being guessed at, which is the conservative direction for an
#: exclusion audit (it can only shrink the personal-matter population, never
#: inflate it with an ambiguous compound string).
PERSONAL_MATTER_DESIGNATIONS = frozenset(
    {"personal matter", "not injury related - personal matter", "personal"}
)
REST_DAY_DESIGNATIONS = frozenset(
    {
        "rest",
        "rested",
        "resting veteran",
        "not injury related - resting player",
        "not injury related -- resting veteran",
    }
)
ILLNESS_DESIGNATIONS = frozenset(
    {"illness", "illness (non-covid)", "medical illness", "non-football illness"}
)
COACH_TEAM_DECISION_DESIGNATIONS = frozenset(
    {
        "coach's decision",
        "coaching",
        "coaching decision",
        "not injury related - coach's decision",
        "not injury related - coaching decision",
        "not injury related - team decision",
    }
)
OTHER_NON_INJURY_DESIGNATIONS = frozenset(
    {
        "not injury related",
        "not injury related - other",
        "not injury related - discipline",
        "not injury related - returning from suspension",
        "not injury related - did not travel",
        "not injury related - travel",
        "travel after trade",
    }
)
#: Union of every frozen non-injury bucket, exposed for callers that just
#: need "is this a non-injury designation at all" without the per-bucket
#: split (e.g. building the LEAD-19 "injury" baseline population).
NON_INJURY_DESIGNATIONS = (
    PERSONAL_MATTER_DESIGNATIONS
    | REST_DAY_DESIGNATIONS
    | ILLNESS_DESIGNATIONS
    | COACH_TEAM_DECISION_DESIGNATIONS
    | OTHER_NON_INJURY_DESIGNATIONS
)

DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 20260905


def canonical_text(value: object) -> str | None:
    """Lowercase/strip a free-text designation field; ``None`` for missing."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip().lower()
    return text or None


def canonicalize_injury_text_rows(
    frame: pd.DataFrame, *, include_postseason: bool = False
) -> pd.DataFrame:
    """Normalize raw nflverse injury rows while KEEPING the free-text designation columns.

    Mirrors ``nfl_ats.players.canonicalize_injuries`` (same season scoping,
    team-alias normalization, and null-filtering) but selects
    :data:`INJURY_TEXT_REQUIRED_COLUMNS` instead of that function's narrower
    ``INJURY_REQUIRED_COLUMNS``, since ``report_primary_injury`` /
    ``practice_primary_injury`` are exactly the columns this module exists to
    read. Does not deduplicate to one row per player-week -- see
    :func:`select_earliest_revision_per_player_week` for that step, kept
    separate so callers can measure the multi-revision rate before collapsing
    it away.
    """

    require_columns(frame, INJURY_TEXT_REQUIRED_COLUMNS, "injuries")
    result = frame.loc[:, list(INJURY_TEXT_REQUIRED_COLUMNS)].copy()
    result = result.loc[
        season_scope_mask(
            result["game_type"],
            include_postseason=include_postseason,
            dataset="injuries",
            column="game_type",
        )
    ].copy()
    result["season"] = pd.to_numeric(result["season"], errors="coerce")
    result["week"] = pd.to_numeric(result["week"], errors="coerce")
    result["date_modified"] = pd.to_datetime(result["date_modified"], errors="coerce", utc=True)
    result["team"] = result["team"].replace(TEAM_ABBREVIATION_ALIASES).astype("string")
    result["gsis_id"] = result["gsis_id"].astype("string")
    result["position"] = result["position"].astype("string").str.upper()
    for column in (
        "report_status",
        "report_primary_injury",
        "practice_status",
        "practice_primary_injury",
    ):
        result[column] = result[column].astype("string")
    result = result.loc[
        result["season"].notna()
        & result["week"].notna()
        & result["team"].notna()
        & result["gsis_id"].notna()
        & result["date_modified"].notna()
    ].copy()
    result["season"] = result["season"].astype(int)
    result["week"] = result["week"].astype(int)
    result = result.drop_duplicates().sort_values(
        ["season", "week", "team", "gsis_id", "date_modified"]
    )
    return result.reset_index(drop=True)


def select_earliest_revision_per_player_week(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse to one row per (season, week, team, gsis_id): the earliest ``date_modified``.

    nflverse's injury feed is documented (and, separately, measured here) to
    carry essentially one report per player-week rather than a Wed/Thu/Fri
    sequence of revisions, so "earliest report of the week" is an
    approximation for the rare player-week that DOES carry more than one row.
    Returns the deduplicated frame and the count of player-weeks that had
    more than one revision, so a caller can disclose exactly how often the
    approximation bites instead of asserting it away.
    """

    require_columns(frame, ("season", "week", "team", "gsis_id", "date_modified"), "injury rows")
    ordered = frame.sort_values(["season", "week", "team", "gsis_id", "date_modified"])
    revision_counts = ordered.groupby(["season", "week", "team", "gsis_id"], sort=False).size()
    multi_revision_player_weeks = int((revision_counts > 1).sum())
    earliest = ordered.drop_duplicates(
        ["season", "week", "team", "gsis_id"], keep="first"
    ).reset_index(drop=True)
    return earliest, multi_revision_player_weeks


def attach_played_outcome(
    frame: pd.DataFrame, snaps: pd.DataFrame, rosters: pd.DataFrame
) -> pd.DataFrame:
    """Attach a boolean ``played`` column: any recorded offense/defense/special-teams snap.

    Joins through ``nfl_ats.players.canonicalize_snaps`` /
    ``canonicalize_rosters`` / ``attach_snap_player_ids`` exactly as
    production's ``nfl_ats.availability.build_availability_outcomes`` does,
    reusing the same crosswalk rather than re-implementing it. Unlike that
    function this join needs no ``games`` table and no decision-time cutoff:
    the outcome is simply "did this gsis_id record any snaps for this
    season/week/team", which the snap table alone answers. A player-week with
    no matching snap row is treated as unavailable (``played=False``), which
    is correct for nflverse's snap-count feed: a player who did not play
    generates no row at all, rather than a zero-snap row.
    """

    require_columns(frame, ("season", "week", "team", "gsis_id"), "player-week rows")
    canonical_rosters = canonicalize_rosters(rosters)
    canonical_snaps = canonicalize_snaps(snaps)
    linked = attach_snap_player_ids(canonical_snaps, canonical_rosters)
    linked = linked.loc[linked["gsis_id"].notna()].copy()
    linked["gsis_id"] = linked["gsis_id"].astype("string")
    linked["total_snaps"] = (
        linked["offense_snaps"].fillna(0.0)
        + linked["defense_snaps"].fillna(0.0)
        + linked["st_snaps"].fillna(0.0)
    )
    played = (
        linked.groupby(["season", "week", "team", "gsis_id"], observed=True)["total_snaps"]
        .max()
        .gt(0)
        .rename("played")
        .reset_index()
    )
    result = frame.merge(
        played, on=["season", "week", "team", "gsis_id"], how="left", validate="many_to_one"
    )
    result["played"] = result["played"].fillna(False).astype(bool)
    return result


def concussion_position_group(position: object) -> str:
    """LEAD-18's predeclared "skill" vs literal "OL/DL" line split; else "other"."""

    normalized = str(position).strip().upper()
    if normalized in CONCUSSION_SKILL_POSITIONS:
        return "skill"
    if normalized in CONCUSSION_LINE_POSITIONS:
        return "line"
    return "other"


def classify_designation(report_canon: str | None, practice_canon: str | None) -> str:
    """Classify a player-week's designation from its canonical report/practice text.

    Checks the frozen buckets in a fixed order (they are disjoint by
    construction, so order only matters as a documented tie-break) and falls
    through to ``"injury"`` -- covering every genuine body-part/medical
    designation AND any compound or narrative string this module's frozen
    exact-match sets deliberately did not claim (see the module docstring on
    :data:`PERSONAL_MATTER_DESIGNATIONS`).
    """

    texts = {text for text in (report_canon, practice_canon) if text}
    if texts & PERSONAL_MATTER_DESIGNATIONS:
        return "personal_matter"
    if texts & REST_DAY_DESIGNATIONS:
        return "rest_day"
    if texts & ILLNESS_DESIGNATIONS:
        return "illness"
    if texts & COACH_TEAM_DECISION_DESIGNATIONS:
        return "coach_team_decision"
    if texts & OTHER_NON_INJURY_DESIGNATIONS:
        return "other_non_injury"
    return "injury"


def build_player_week_frame(
    injuries: pd.DataFrame,
    snaps: pd.DataFrame,
    rosters: pd.DataFrame,
    *,
    season_start: int,
    season_end: int,
    include_postseason: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Shared LEAD-18/LEAD-19 base frame: one player-week row per designation/outcome/group label.

    Returns ``(frame, multi_revision_player_weeks)``. ``frame`` columns
    include ``season``, ``week``, ``team``, ``gsis_id``, ``position``,
    ``report_status``, ``practice_status``, ``played``, ``sat_out`` (``1.0 -
    played``), ``report_canon``, ``practice_canon``, ``designation``
    (:func:`classify_designation`), ``is_concussion_report`` (``report_canon
    == "concussion"``), and ``concussion_group``
    (:func:`concussion_position_group`).
    """

    if season_start > season_end:
        raise ValueError("season_start must be <= season_end")
    canonical = canonicalize_injury_text_rows(injuries, include_postseason=include_postseason)
    canonical = canonical.loc[canonical["season"].between(season_start, season_end)].copy()
    earliest, multi_revision_count = select_earliest_revision_per_player_week(canonical)
    enriched = attach_played_outcome(earliest, snaps, rosters)
    enriched["sat_out"] = (~enriched["played"]).astype(float)
    enriched["played_f"] = enriched["played"].astype(float)
    enriched["report_canon"] = enriched["report_primary_injury"].map(canonical_text)
    enriched["practice_canon"] = enriched["practice_primary_injury"].map(canonical_text)
    enriched["designation"] = [
        classify_designation(report, practice)
        for report, practice in zip(
            enriched["report_canon"], enriched["practice_canon"], strict=True
        )
    ]
    enriched["is_concussion_report"] = enriched["report_canon"].eq("concussion")
    enriched["concussion_group"] = enriched["position"].map(concussion_position_group)
    return enriched, multi_revision_count


@dataclass(frozen=True)
class RateGapInterval:
    """A season-blocked bootstrap interval for ``mean(outcome|group_a) - mean(outcome|group_b)``."""

    estimate: float
    lower: float
    upper: float
    probability_positive: float
    n_a: int
    n_b: int
    rate_a: float
    rate_b: float
    block_count: int
    samples: int
    seed: int
    dropped_draws: int


def season_block_bootstrap_gap(
    frame: pd.DataFrame,
    *,
    group_column: str,
    outcome_column: str,
    group_a: str,
    group_b: str,
    season_column: str = "season",
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> RateGapInterval:
    """Season-blocked bootstrap of a two-group rate gap.

    Resamples whole seasons with replacement (never individual player-weeks)
    so that rows sharing a season move together in every draw -- the same
    block unit every other season-blocked interval in this project uses.
    Weeks are NOT the block unit here: AGENTS.md's "within-week correlation
    is zero" mandate governs game-level correlation within a single week's
    slate, a different population from this player-week-across-a-season
    frame, and seasons are the coarsest unit this population plausibly
    clusters on (a shared season can share reporting-culture drift, rule
    changes, etc.; individual weeks within a season do not share anything
    group_a/group_b membership would care about).

    Both groups reuse the SAME per-sample multinomial draw (one draw over
    the season index, applied to both groups' per-season sums/counts), so
    the two groups' resampled rates are correlated exactly as they would be
    if the underlying player-weeks were literally resampled by season and
    then split by group -- not two independently-seeded bootstraps.

    A season with zero rows for one group contributes (0, 0) for that
    group and does not break the estimator; a draw whose resampled seasons
    happen to carry zero total weight for a group (only possible when that
    group's raw per-season counts are concentrated in very few seasons)
    produces a non-finite rate and is dropped, counted in
    ``dropped_draws``.
    """

    if samples < 10:
        raise ValueError("samples must be at least 10")
    require_columns(frame, (season_column, group_column, outcome_column), "rate-gap frame")
    group_a_frame = frame.loc[frame[group_column].eq(group_a)]
    group_b_frame = frame.loc[frame[group_column].eq(group_b)]
    if group_a_frame.empty or group_b_frame.empty:
        raise DataContractError(
            f"season_block_bootstrap_gap requires both groups populated: "
            f"{group_a!r}={len(group_a_frame)}, {group_b!r}={len(group_b_frame)}"
        )
    seasons = sorted(frame[season_column].unique())
    season_index = {season: position for position, season in enumerate(seasons)}
    block_count = len(seasons)
    season_positions = frame[season_column].map(season_index).to_numpy()
    is_a = frame[group_column].eq(group_a).to_numpy(dtype=np.float64)
    is_b = frame[group_column].eq(group_b).to_numpy(dtype=np.float64)
    outcome = frame[outcome_column].to_numpy(dtype=np.float64)
    a_sum = np.bincount(season_positions, weights=outcome * is_a, minlength=block_count)
    a_cnt = np.bincount(season_positions, weights=is_a, minlength=block_count)
    b_sum = np.bincount(season_positions, weights=outcome * is_b, minlength=block_count)
    b_cnt = np.bincount(season_positions, weights=is_b, minlength=block_count)
    rng = np.random.default_rng(seed)
    drawn: npt.NDArray[np.int64] = rng.multinomial(
        block_count, np.full(block_count, 1.0 / block_count), size=samples
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_a_draws = (drawn @ a_sum) / (drawn @ a_cnt)
        rate_b_draws = (drawn @ b_sum) / (drawn @ b_cnt)
    diff = rate_a_draws - rate_b_draws
    finite = np.isfinite(diff)
    dropped = int((~finite).sum())
    diff = diff[finite]
    point_a = float(a_sum.sum() / a_cnt.sum())
    point_b = float(b_sum.sum() / b_cnt.sum())
    tail = 0.025
    return RateGapInterval(
        estimate=point_a - point_b,
        lower=float(np.quantile(diff, tail)) if len(diff) else float("nan"),
        upper=float(np.quantile(diff, 1.0 - tail)) if len(diff) else float("nan"),
        probability_positive=float(np.mean(diff > 0.0)) if len(diff) else float("nan"),
        n_a=int(a_cnt.sum()),
        n_b=int(b_cnt.sum()),
        rate_a=point_a,
        rate_b=point_b,
        block_count=block_count,
        samples=samples,
        seed=seed,
        dropped_draws=dropped,
    )


def rate_gap_to_dict(result: RateGapInterval) -> dict[str, Any]:
    """JSON-serializable view of a :class:`RateGapInterval`, for artifact payloads."""

    return {
        "estimate": result.estimate,
        "ci95": [result.lower, result.upper],
        "probability_positive": result.probability_positive,
        "n_a": result.n_a,
        "n_b": result.n_b,
        "rate_a": result.rate_a,
        "rate_b": result.rate_b,
        "block_count": result.block_count,
        "samples": result.samples,
        "seed": result.seed,
        "dropped_draws": result.dropped_draws,
    }
