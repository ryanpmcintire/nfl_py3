"""Split-half reliability for the 46 ``health_roster`` registry cells (ORCH-D).

**What these cells are.** Injury value-lost channels, the player-family
feature-block ablation ladder, the NFL.com Friday injury-report screens, the
interim-head-coach family, fantasy-ADP divergence, USA Today player arrests,
the regional FluView illness index, first-year head coaches, participation
RAPM and the backup-QB news-visibility screen -- 46 of the 365
``reliability: null`` NFL ``accuracy_points`` entries in
``registry/weak_signals.json``.

**Method.** ``scripts/reliability_lib.py``, imported, never reimplemented:
unit = team-season, halves = odd/even weeks, Spearman-Brown corrected, block
bootstrap over team-seasons, seed 20260901, 4000 draws, restricted to each
cell's OWN registry seasons. Two of the three tags are used here:

``METHOD_TRAIT``
    for a continuous per-team-week parent quantity (injury value lost,
    lineup continuity, an NFL.com Out-count, the FluView illness index).
    This is the only tag whose low value is a candidate
    ``no_split_half_reliability`` ground.

``METHOD_EXPOSURE``
    for a per-game/per-team-game FLAG with no continuous parent (an interim
    head coach is active, an arrest occurred, backup-QB news was visible). A
    low EXPOSURE reliability is **NOT** a closing ground.

**Binding taxonomy (verbatim, AGENTS.md / CLAUDE.md).** An interval or CI that
contains zero is NEVER grounds to reject, fail, or close an experiment. Only
two grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED
wrong sign (whole interval on the wrong side of zero) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an effect
that size. Everything else is ``unresolved_below_power``; report
``probability_positive``, never "contains zero". This script CLOSES NOTHING and
RECLASSIFIES NOTHING: it measures, and a low number is a candidate for the
reliability ground, never the closure itself. Within-week correlation is ZERO.

A construct with too few usable team-seasons is reported as UNMEASURED, never
as reliability 0 -- writing a NaN through as a number would manufacture the
appearance of a closing ground out of nothing. A construct whose parent column
is NEAR-CONSTANT, or is CONSTANT WITHIN a team-season by construction, is
reported as ``not_informative_near_constant``: the first can return a large |r|
of either sign that flips with the season window, and the second returns
exactly +1.0 no matter what the world does. Both are artifacts of the
degeneracy, not traits, and neither is recorded.

**Rolling-aggregate caveat, stated once.** Most of the continuous parents here
(lineup continuity, injury unavailability, QB ratings, injury value lost) are
season-to-date or span-smoothed rolling aggregates, so their odd/even-week
split-half is high partly by construction of the smoothing. That does not
invalidate the number -- it is the feature exactly as the model consumes it,
and it matches the registry's existing 0.98-0.99 entries -- but a reader must
not read 0.95 here as "this construct is 95% repeatable news".

Writes ``artifacts/reliability_sweep/health_roster/<stamp>/results.json`` and
prints the ``set-reliability`` commands to run; recording goes through the
locked CLI, never from inside this script.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.append(str(REPO / "scripts"))

import reliability_lib as rlib  # noqa: E402
import reliability_map as relmap  # noqa: E402

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.weak_signals import default_registry_path, load_registry  # noqa: E402

# --------------------------------------------------------------------------
# Feature tables each group's cells were actually scored on (read from the
# cells' own artifact metadata, not guessed from the name).
# --------------------------------------------------------------------------

#: artifacts/player_experiments/20260813T122348Z/metadata.json,
#: provenance.feature_table.path -- the exact table the player-family ablation
#: ladder was fitted on (manifest decision_hours_before_kickoff=24, Saturday).
PLAYER_VALUE_TABLE = REPO / "data" / "processed" / "game_features_player_value.parquet"

#: artifacts/participation_experiments/20260813T132030Z/metadata.json,
#: provenance.feature_table.path.
PARTICIPATION_TABLE = REPO / "data" / "processed" / "game_features_player_participation.parquet"

#: The injury-channel experiments' own game-level input table.
PBP_TABLE = REPO / "data" / "processed" / "game_features_pbp.parquet"

#: MEASURED 2026-09-01: of the three snapshots under data/raw/nflcom_injuries,
#: only 20260821T222602Z holds rows (17,483, seasons 2022-2024); both
#: 20260825T* snapshots hold ZERO rows. The screens select their snapshot with
#: ``latest()`` = lexicographically newest, so running them unpinned today
#: would silently produce all-zero Out-counts. This is the snapshot the
#: registry cells were actually measured on (their source artifact is
#: artifacts/nflcom_friday_designation_screen/20260821T224931Z, 2026-08-21).
NFLCOM_SNAPSHOT = REPO / "data" / "raw" / "nflcom_injuries" / "20260821T222602Z"

#: The frozen QB-news artifact. The screen's PFR phase needs a live, budget-
#: truncated network fetch, so its flag is NOT deterministically rebuildable;
#: this artifact is the run the registry cell was recorded from.
QB_NEWS_ARTIFACT = REPO / "artifacts" / "qb_news_channel" / "20260820T093852Z"

STATUS_NEAR_CONSTANT = "not_informative_near_constant"
STATUS_NO_TRAIT = "no_underlying_trait"
STATUS_DATA_MISSING = "data_not_present_locally"

#: A parent column with fewer than this many distinct values, or with fewer
#: than this share of non-zero rows, is reported as
#: ``not_informative_near_constant`` rather than recorded.
NEAR_CONSTANT_MIN_DISTINCT = 3
NEAR_CONSTANT_NONZERO_SHARE = 0.02


# --------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------


def _clean(value: Any) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _summarise(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary rule for a feature BLOCK: the MINIMUM member reliability.

    Why the minimum and not the mean: the registry cell is the whole block
    added at once, and the block is only as repeatable out of sample as its
    least repeatable member -- an ablation arm that gains from one stable
    column plus one noise column will not replicate on the noise column's
    contribution. The minimum never manufactures reliability the weakest input
    does not have, and it is one named rule rather than a per-cell judgement
    call. Every member's own number stays in the artifact so a reader can
    apply a different rule.
    """

    usable = [m for m in members if m["status"] == rlib.STATUS_MEASURED]
    if not usable:
        return {"summary_rule": "min_member_reliability", "chosen": None, "usable_members": 0}
    chosen = min(usable, key=lambda m: float(m["reliability"]))
    return {
        "summary_rule": "min_member_reliability",
        "chosen": chosen["metric"],
        "usable_members": len(usable),
        "member_reliabilities": {m["metric"]: m["reliability"] for m in usable},
    }


def _column_stats(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "n_nonnull": len(values),
        "n_distinct": int(values.nunique()),
        "nonzero_share": float((values != 0).mean()) if len(values) else 0.0,
        "std": float(values.std()) if len(values) > 1 else 0.0,
    }


def _near_constant(series: pd.Series, *, binary_flag: bool = False) -> tuple[bool, dict[str, Any]]:
    """Is this column too degenerate for its split-half correlation to mean anything?

    Two objective tests, and deliberately no third:

    * ``n_distinct < 3`` -- a CONTINUOUS parent that takes at most two values
      is not a continuous parent. This test is skipped for ``binary_flag``
      columns: a 0/1 exposure indicator always has exactly two values, and
      that is what a flag IS. What the estimator correlates for a flag is the
      per-unit-season MEAN, which is continuous, so applying the distinct-value
      test to the raw indicator would condemn every EXPOSURE measurement.
    * ``nonzero_share < 0.02`` -- almost every row is exactly zero, so any
      correlation is carried by a handful of rows and can be a large value of
      either sign that flips with the season window.

    Within-unit constancy (a season-constant quantity, whose split-half is
    +1.0 by construction) is a separate test handled by
    :func:`within_unit_variation`, because it is a property of the design
    rather than of the column's marginal distribution.
    """

    stats = _column_stats(series)
    stats["binary_flag"] = binary_flag
    if stats["n_nonnull"] == 0:
        return True, stats
    stats["sparse_event_flag"] = bool(stats["nonzero_share"] < NEAR_CONSTANT_NONZERO_SHARE)
    if binary_flag:
        # A sparse flag is a CAVEAT, not a veto. Its split-half is carried by
        # "does this unit have any exposure at all", which is inflated when one
        # event spans both halves -- reported loudly, and EXPOSURE is not an
        # admissible closing ground either way. It is only treated as an
        # artifact when the value actually behaves like one, i.e. flips with
        # the season window; every sparse flag here was measured on two
        # disjoint windows and none flipped (see the artifact's era slices).
        return False, stats
    degenerate = (
        stats["n_distinct"] < NEAR_CONSTANT_MIN_DISTINCT
        or stats["nonzero_share"] < NEAR_CONSTANT_NONZERO_SHARE
    )
    return bool(degenerate), stats


def within_unit_variation(
    long: pd.DataFrame, metric: str, *, unit_col: str, seasons: tuple[int, int]
) -> dict[str, Any]:
    """Does ``metric`` vary WITHIN a (unit, season) at all?

    A quantity that is constant inside every unit-season (a preseason ADP
    aggregate, a season-long head-coach tenure flag) has an odd/even-week
    split-half of exactly +1.0 no matter what the world does -- the two halves
    are the same number. Detecting that BEFORE reading the correlation is what
    stops a structural +1.0 being recorded as a trait.
    """

    frame = long.loc[:, [unit_col, "season", "week", metric]].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame = frame.dropna(subset=["season", metric])
    frame = frame.loc[frame["season"].between(int(seasons[0]), int(seasons[1]))]
    if frame.empty:
        return {"unit_seasons": 0, "share_zero_within_variance": None, "mean_within_std": None}
    within = frame.groupby([unit_col, "season"])[metric].std(ddof=0)
    return {
        "unit_seasons": len(within),
        "share_zero_within_variance": float((within.fillna(0.0) == 0.0).mean()),
        "mean_within_std": float(within.fillna(0.0).mean()),
    }


def _row(
    *,
    entry: str,
    parent: str,
    unit: str,
    seasons: tuple[int, int],
    measured: dict[str, Any] | None,
    method_tag: str,
    provenance: str,
    status: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "entry": entry,
        "parent_quantity": parent,
        "unit": unit,
        "seasons": [int(seasons[0]), int(seasons[1])],
        "method_tag": method_tag,
        "builder_provenance": provenance,
    }
    if measured is None:
        base.update(
            {
                "n_units": 0,
                "pearson_r": None,
                "pearson_r_ci95": None,
                "spearman_rho": None,
                "spearman_brown_full_length_reliability": None,
                "probability_positive": None,
                "reliability": None,
                "reliability_low": None,
                "reliability_high": None,
                "status": status or STATUS_NO_TRAIT,
                "method": None,
            }
        )
    else:
        base.update(
            {
                "n_units": measured["n_units"],
                "pearson_r": _clean(measured.get("pearson_r")),
                "pearson_r_ci95": [_clean(v) for v in measured.get("pearson_r_ci95", [])],
                "spearman_rho": _clean(measured.get("spearman_rho")),
                "spearman_brown_full_length_reliability": _clean(
                    measured.get("spearman_brown_full_length_reliability")
                ),
                "probability_positive": _clean(measured.get("probability_positive")),
                "reliability": measured["reliability"],
                "reliability_low": measured["reliability_low"],
                "reliability_high": measured["reliability_high"],
                "status": status or measured["status"],
                "method": measured["method"],
            }
        )
    if extra:
        base.update(extra)
    return base


def team_long_from_game_table(features: pd.DataFrame, wanted: list[str]) -> pd.DataFrame:
    """Team-week long frame for ``wanted`` metrics of a game-level table.

    Reuses ``reliability_map.discover_family_pairs`` /
    ``reliability_map.build_long_frame`` (the 2026-08-26 precedent) rather
    than re-folding home/away columns by hand.
    """

    dtypes = {column: features[column].dtype for column in features.columns}
    families, _excluded = relmap.discover_family_pairs(list(features.columns), dtypes)
    missing = [column for column in wanted if column not in families]
    if missing:
        raise SystemExit(f"no home/away pair discovered for {missing}")
    return relmap.build_long_frame(features, {name: families[name] for name in wanted})


def flag_exposure_long(table: pd.DataFrame, flag: pd.Series, *, team_col: str) -> pd.DataFrame:
    """Team-week exposure frame from a builder's own TEAM-GAME table.

    The flag builders in ``nfl_ats.experiment_runner`` already return one row
    per (game, team side), so the flag is team-attributed at source and
    ``reliability_lib.game_flag_to_team_week`` -- which explodes a GAME-level
    flag onto both sides -- would attribute one side's flag to both teams.
    Same output shape and the same ``exposure`` semantics, correct attribution.
    """

    frame = table.loc[:, [team_col, "season", "week"]].rename(columns={team_col: "team_id"}).copy()
    frame["exposure"] = (
        flag.reindex(table.index).fillna(False).astype(bool).astype(float).to_numpy()
    )
    return frame


@contextlib.contextmanager
def pinned_nflcom_snapshot(module: Any) -> Iterator[None]:
    """Force the NFL.com screens' ``latest()`` onto :data:`NFLCOM_SNAPSHOT`.

    The screens pick their injury snapshot with
    ``sorted(root.glob("*/injuries.parquet"))[-1]``, and the two newest
    snapshots on disk hold ZERO rows (measured 2026-09-01), so an unpinned run
    returns all-zero Out-counts. Patching the selector -- rather than
    re-implementing ``load_report_flags`` -- keeps every line of the builders'
    own normalisation, status filtering and starter matching intact.
    """

    original = module.latest

    def pinned(root: Path, pattern: str) -> Path:
        if pattern == "*/injuries.parquet" and Path(root).name == "nflcom_injuries":
            return NFLCOM_SNAPSHOT / "injuries.parquet"
        return original(root, pattern)

    module.latest = pinned
    try:
        yield
    finally:
        module.latest = original


def target_entries(names: list[str]) -> dict[str, dict[str, Any]]:
    registry = load_registry(default_registry_path(REPO / "registry"))
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        signal = registry.signals[name]
        out[name] = {
            "seasons": (int(signal.seasons[0]), int(signal.seasons[1])),
            "reliability": signal.reliability,
            "effect": signal.effect,
            "classification": signal.classification,
        }
    return out


# ==========================================================================
# Group 1 -- injury_value_lost_* (8 cells)
# ==========================================================================
#
# The construct behind all eight is ``injury_value_lost_narrowed``'s D-A
# contrast: the ``player_value`` arm minus the ``player`` arm, i.e. the
# marginal feature block ``FEATURE_FAMILIES["player_values"]`` =
# ``diff_injury_skill_epa_value_lost`` + ``diff_injury_defense_disruption_
# value_lost`` (read: src/nfl_ats/constants.py FEATURE_FAMILIES, and
# src/nfl_ats/margin.py:113-118 where ``player_injury_value`` / ``player_value``
# add exactly that family). The parent team-week quantities are therefore
# ``injury_skill_epa_value_lost`` and ``injury_defense_disruption_value_lost``.
#
# DOES THE UNDERLYING QUANTITY CHANGE ACROSS THE VARIANTS? Two answers, and
# they must not be merged:
#
#  * ``tuesday_cutoff_official`` / ``tuesday_cutoff_pft_augmented`` and the two
#    ``tuesday_saturday_channel_*`` deltas: NO. Read
#    scripts/injury_tuesday_cutoff_experiment.py:24-40 -- the arm pre-filters
#    the INJURIES DATAFRAME to rows whose own ``date_modified`` is at or before
#    that game's Tuesday noon and then calls the unmodified
#    ``enrich_with_player_features``; "every other piece of accumulated state
#    ... is built from strictly prior COMPLETED games and is untouched by this
#    parameter". Same construct, read at a different report timestamp, so by
#    the registry's own ``attention_battery_*`` precedent they would inherit
#    the Saturday trait's number -- EXCEPT that the Tuesday filter leaves so
#    few visible rows that the arm's own column can be degenerate, which is
#    itself decision-relevant, so each arm is measured on its own table.
#  * ``prior_week_absence`` / ``prior_week_report`` and their
#    ``_saturday_channel`` deltas: YES. Read
#    scripts/injury_prior_week_variant_experiment.py:169-250 -- these replace
#    the injuries table with a SYNTHETIC one keyed at the current week but
#    built only from last week's zero-snap players (severity fixed at 1.0 for
#    the absence arm). Same functional form and column names, a DIFFERENT
#    input population. They do not inherit the Saturday number.
#
# The four ``*_channel`` cells are PAIRED DELTAS between two arms on the same
# games, so their parent is both arms' columns and the reported number is the
# minimum over all four, the same conservative rule the feature blocks use.

INJURY_VALUE_COLUMNS = ("injury_skill_epa_value_lost", "injury_defense_disruption_value_lost")

INJURY_CELL_ARMS: dict[str, tuple[tuple[str, ...], str]] = {
    "injury_value_lost_tuesday_cutoff_official": (
        ("tuesday_official",),
        "own-week Tuesday-noon official-report visibility arm",
    ),
    "injury_value_lost_tuesday_cutoff_pft_augmented": (
        ("tuesday_pft",),
        "own-week Tuesday-noon visibility augmented by PFT foreshadowing",
    ),
    "injury_value_lost_tuesday_saturday_channel_official_only": (
        ("saturday", "tuesday_official"),
        "paired Saturday-minus-Tuesday-official channel delta; min over both arms",
    ),
    "injury_value_lost_tuesday_saturday_channel_pft_augmented": (
        ("saturday", "tuesday_pft"),
        "paired Saturday-minus-Tuesday+PFT channel delta; min over both arms",
    ),
    "injury_value_lost_prior_week_report": (
        ("prior_week_report",),
        "synthetic prior-week official-report arm",
    ),
    "injury_value_lost_prior_week_absence": (
        ("prior_week_absence",),
        "synthetic prior-week actual-absence arm",
    ),
    "injury_value_lost_prior_week_report_saturday_channel": (
        ("saturday", "prior_week_report"),
        "paired Saturday-minus-prior-week-report channel delta; min over both arms",
    ),
    "injury_value_lost_prior_week_absence_saturday_channel": (
        ("saturday", "prior_week_absence"),
        "paired Saturday-minus-prior-week-absence channel delta; min over both arms",
    ),
}

INJURY_ARM_SPECS: tuple[tuple[str, str, float], ...] = (
    ("saturday", "injuries", 24.0),
    ("tuesday_official", "tuesday_official", 0.0),
    ("tuesday_pft", "tuesday_pft", 0.0),
    ("prior_week_report", "prior_week_report", 24.0),
    ("prior_week_absence", "prior_week_absence", 24.0),
)

INJURY_ARM_KEEP = [
    "game_id",
    "season",
    "week",
    "game_type",
    "home_team",
    "away_team",
    "home_injury_skill_epa_value_lost",
    "away_injury_skill_epa_value_lost",
    "home_injury_defense_disruption_value_lost",
    "away_injury_defense_disruption_value_lost",
]

INJURY_PROVENANCE = (
    "src/nfl_ats/constants.py FEATURE_FAMILIES['player_values'] is the marginal block the "
    "D-A contrast adds (margin.py:113-118); the five arm tables are rebuilt by importing "
    "injury_tuesday_cutoff_experiment / injury_prior_week_variant_experiment and calling "
    "the unmodified nfl_ats.players.enrich_with_player_features. Read 2026-09-01."
)


def build_injury_arm_tables(cache_dir: Path) -> dict[str, pd.DataFrame]:
    """Rebuild (or read from ``cache_dir``) the five injury-arm feature tables."""

    import injury_prior_week_variant_experiment as ipw
    import injury_tuesday_cutoff_experiment as itc

    cache_dir.mkdir(parents=True, exist_ok=True)
    wanted = {label: cache_dir / f"{label}.parquet" for label, _key, _hours in INJURY_ARM_SPECS}
    if all(path.is_file() for path in wanted.values()):
        return {label: pd.read_parquet(path) for label, path in wanted.items()}

    from nfl_ats.pbp import load_pbp_snapshot
    from nfl_ats.pbp import snapshot_from_root as pbp_snapshot_from_root
    from nfl_ats.players import (
        load_player_snapshot,
        load_player_value_snapshot,
        player_snapshot_from_root,
        player_value_snapshot_from_root,
    )

    games = pd.read_parquet(PBP_TABLE)
    injuries, rosters, snaps = load_player_snapshot(
        player_snapshot_from_root(REPO / "data/players/raw" / itc.PLAYER_SNAPSHOT_ID)
    )
    pbp = load_pbp_snapshot(pbp_snapshot_from_root(REPO / "data/pbp/raw" / itc.PBP_SNAPSHOT_ID))
    player_stats = load_player_value_snapshot(
        player_value_snapshot_from_root(
            REPO / "data/players/values/raw" / itc.PLAYER_VALUE_SNAPSHOT_ID
        )
    )
    pft = pd.read_parquet(REPO / f"data/raw/injury_news/{itc.PFT_SNAPSHOT_ID}/index.parquet")

    matched = itc.build_pft_match_table(
        injuries,
        itc.build_player_name_map(rosters),
        pft,
        itc.team_week_tuesday_noon(games),
        lookback_days=9.0,
    )
    official_visible = matched["date_modified"] <= matched["tuesday_noon_utc"]
    pft_visible = official_visible | matched["pft_match_lastmod"].notna()
    injury_cols = list(injuries.columns)

    schedule = ipw.team_schedule_prior_week(games)
    injuries_c, played, active_roster = ipw.build_played_and_active(injuries, rosters, snaps)
    prior_report, _ = ipw.build_prior_week_report(injuries_c, played, schedule)
    prior_absence, _ = ipw.build_prior_week_absence(played, active_roster, schedule)

    sources = {
        "injuries": injuries,
        "tuesday_official": matched.loc[official_visible, injury_cols].reset_index(drop=True),
        "tuesday_pft": matched.loc[pft_visible, injury_cols].reset_index(drop=True),
        "prior_week_report": prior_report,
        "prior_week_absence": prior_absence,
    }

    tables: dict[str, pd.DataFrame] = {}
    for label, key, hours in INJURY_ARM_SPECS:
        path = wanted[label]
        if not path.is_file():
            enriched = ipw.build_enriched(
                games=games,
                injuries=sources[key],
                rosters=rosters,
                snaps=snaps,
                pbp=pbp,
                player_stats=player_stats,
                decision_hours_before_kickoff=hours,
            )
            enriched.loc[:, INJURY_ARM_KEEP].to_parquet(path, index=False)
        tables[label] = pd.read_parquet(path)
    return tables


def measure_injury_group(
    entries: dict[str, dict[str, Any]], cache_dir: Path, *, n_boot: int
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    tables = build_injury_arm_tables(cache_dir)
    longs = {
        label: team_long_from_game_table(table, list(INJURY_VALUE_COLUMNS))
        for label, table in tables.items()
    }

    cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for entry_name, (arm_labels, note) in INJURY_CELL_ARMS.items():
        seasons = entries[entry_name]["seasons"]
        members: list[dict[str, Any]] = []
        degenerate: list[dict[str, Any]] = []
        for label in arm_labels:
            for column in INJURY_VALUE_COLUMNS:
                key = f"{label}:{column}"
                if key not in cache:
                    window = longs[label].loc[longs[label]["season"].between(*seasons)]
                    is_degenerate, stats = _near_constant(window[column])
                    measured = rlib.measure_reliability(
                        longs[label],
                        column,
                        method=rlib.METHOD_TRAIT,
                        seasons=seasons,
                        n_boot=n_boot,
                    )
                    measured["metric"] = key
                    measured["degenerate"] = is_degenerate
                    measured["column_stats"] = stats
                    cache[key] = measured
                entry_measured = cache[key]
                (degenerate if entry_measured["degenerate"] else members).append(entry_measured)

        common = {
            "arms": list(arm_labels),
            "arm_note": note,
            "excluded_near_constant": {m["metric"]: m["column_stats"] for m in degenerate},
            "near_constant_readings": {
                m["metric"]: {
                    "pearson_r": _clean(m.get("pearson_r")),
                    "reliability": m["reliability"],
                    "interval": [m["reliability_low"], m["reliability_high"]],
                }
                for m in degenerate
            },
            "half_season_replication": {
                "status": "not_applicable_no_flag",
                "note": (
                    "This cell is a model-vs-model paired accuracy delta, not a subset flag, "
                    "so there is no flagged-minus-complement cover-rate gap to replicate."
                ),
            },
        }
        if not members:
            rows.append(
                _row(
                    entry=entry_name,
                    parent=" + ".join(f"{a}:{c}" for a in arm_labels for c in INJURY_VALUE_COLUMNS),
                    unit="team-season",
                    seasons=seasons,
                    measured=None,
                    method_tag="TRAIT",
                    provenance=INJURY_PROVENANCE + " " + note,
                    status=STATUS_NEAR_CONSTANT,
                    extra=common,
                )
            )
            continue

        summary = _summarise(members)
        chosen = next(m for m in members if m["metric"] == summary["chosen"])
        rows.append(
            _row(
                entry=entry_name,
                parent=chosen["metric"],
                unit="team-season",
                seasons=seasons,
                measured=chosen,
                method_tag="TRAIT",
                provenance=INJURY_PROVENANCE + " " + note,
                extra={**common, "block_summary": summary},
            )
        )

    control_long = longs["saturday"]
    return rows, {key: _member_summary(m) for key, m in cache.items()}, control_long


def _member_summary(measured: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_units": measured["n_units"],
        "pearson_r": _clean(measured.get("pearson_r")),
        "pearson_r_ci95": [_clean(v) for v in measured.get("pearson_r_ci95", [])],
        "reliability": measured["reliability"],
        "interval": [measured["reliability_low"], measured["reliability_high"]],
        "probability_positive": _clean(measured.get("probability_positive")),
        "status": measured["status"],
        "degenerate": measured.get("degenerate"),
        "column_stats": measured.get("column_stats"),
    }


# ==========================================================================
# Group 2 -- player_family_base_vs_* (8) and participation (1)
# ==========================================================================
#
# Each cell is a feature-BLOCK ablation: profile X vs the ``base`` profile.
# The block a cell adds is exactly the FEATURE_FAMILIES difference between the
# two profiles' FEATURE_SETS entries (read: src/nfl_ats/constants.py lines
# 686-775, and src/nfl_ats/margin.py:96-121 which maps each profile name to
# its ("football_*", "full_*") feature-set pair). The member columns are
# ``diff_<metric>``; the parent team-week quantity is ``<metric>``.

PLAYER_FAMILY_BLOCKS: dict[str, tuple[str, ...]] = {
    "player_family_base_vs_qb": ("player_qb",),
    "player_family_base_vs_injuries": ("player_injuries",),
    "player_family_base_vs_continuity": ("player_continuity",),
    "player_family_base_vs_qb_injuries": ("player_qb", "player_injuries"),
    "player_family_base_vs_qb_continuity": ("player_qb", "player_continuity"),
    "player_family_base_vs_injuries_continuity": ("player_injuries", "player_continuity"),
    "player_family_base_vs_injury_value": ("player_injuries", "player_values"),
    "player_family_base_vs_value": (
        "player_qb",
        "player_injuries",
        "player_continuity",
        "player_values",
    ),
}

PARTICIPATION_BLOCKS: dict[str, tuple[str, ...]] = {
    "participation_offense_defense_rapm": ("player_participation_values",),
}


def block_member_metrics(families: tuple[str, ...]) -> list[str]:
    """The parent team-week metric names of a block, from the repo's own map."""

    from nfl_ats.constants import FEATURE_FAMILIES

    metrics: list[str] = []
    for family in families:
        for column in FEATURE_FAMILIES[family]:
            metric = column[len("diff_") :] if column.startswith("diff_") else column
            if metric not in metrics:
                metrics.append(metric)
    return metrics


def measure_block_group(
    entries: dict[str, dict[str, Any]],
    blocks: dict[str, tuple[str, ...]],
    table_path: Path,
    provenance: str,
    *,
    n_boot: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    features = pd.read_parquet(table_path)
    all_metrics = sorted(
        {m for families in blocks.values() for m in block_member_metrics(families)}
    )
    long = team_long_from_game_table(features, all_metrics)

    cache: dict[tuple[str, tuple[int, int]], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for entry_name, families in blocks.items():
        seasons = entries[entry_name]["seasons"]
        members: list[dict[str, Any]] = []
        degenerate: list[dict[str, Any]] = []
        for metric in block_member_metrics(families):
            key = (metric, seasons)
            if key not in cache:
                window = long.loc[long["season"].between(*seasons)]
                is_degenerate, stats = _near_constant(window[metric])
                measured = rlib.measure_reliability(
                    long, metric, method=rlib.METHOD_TRAIT, seasons=seasons, n_boot=n_boot
                )
                measured["metric"] = metric
                measured["degenerate"] = is_degenerate
                measured["column_stats"] = stats
                cache[key] = measured
            member = cache[key]
            (degenerate if member["degenerate"] else members).append(member)

        common = {
            "block_families": list(families),
            "block_members": block_member_metrics(families),
            "member_measurements": {m["metric"]: _member_summary(m) for m in members + degenerate},
            "half_season_replication": {
                "status": "not_applicable_no_flag",
                "note": (
                    "This cell is a model-vs-model paired accuracy delta over a feature block, "
                    "not a subset flag, so there is no cover-rate gap to replicate."
                ),
            },
        }
        if not members:
            rows.append(
                _row(
                    entry=entry_name,
                    parent=" + ".join(block_member_metrics(families)),
                    unit="team-season",
                    seasons=seasons,
                    measured=None,
                    method_tag="TRAIT",
                    provenance=provenance,
                    status=STATUS_NEAR_CONSTANT,
                    extra=common,
                )
            )
            continue
        summary = _summarise(members)
        chosen = next(m for m in members if m["metric"] == summary["chosen"])
        rows.append(
            _row(
                entry=entry_name,
                parent=chosen["metric"],
                unit="team-season",
                seasons=seasons,
                measured=chosen,
                method_tag="TRAIT",
                provenance=provenance,
                extra={**common, "block_summary": summary},
            )
        )
    return rows, long


# ==========================================================================
# Group 3 -- nflcom_* (7 cells)
# ==========================================================================
#
# Parent quantities are the NFL.com FINAL Friday/Saturday league injury page's
# own per-(season, week, team) counts, a distinct external source from the
# nflverse ``injury_*`` family measured in Group 1. Read:
#   scripts/nflcom_friday_designation_screen.py:239-250 (attach_flags's
#   groupby(["season","week","team"]).agg -> q_or_worse_any / out_count /
#   starter_q_or_worse / new_vs_tuesday, then left-merged onto the team-game
#   population and ZERO-FILLED), and :257-259 for the three thresholds;
#   scripts/nflcom_friday_refresh_feature.py:116-144 (build_out_counts ->
#   total_out / starter_out on the same key) and :222-226 for the thresholds.
# The counts are measured on the zero-filled team-game population, matching the
# builder's own analysis frame rather than the raw report rows.

NFLCOM_CELL_PARENTS: dict[str, tuple[str, str, str]] = {
    "nflcom_friday_out_count_ge2": (
        "out_count",
        "designation",
        "flag = out_count >= 2 (screen:258)",
    ),
    "nflcom_friday_q_or_worse_starter_caliber": (
        "starter_q_or_worse",
        "designation",
        "flag = starter_q_or_worse >= 1 (screen:257)",
    ),
    "nflcom_friday_new_saturday_designation": (
        "new_vs_tuesday",
        "designation",
        "flag = new_vs_tuesday >= 1 (screen:259)",
    ),
    "nflcom_refresh_out2_starters_on_chain": (
        "starter_out",
        "refresh",
        "overlay flag = starter_out >= 2 (refresh:222)",
    ),
    "nflcom_refresh_out2_starters_on_chain_gate_admitted": (
        "starter_out",
        "refresh",
        "same starter_out >= 2 overlay; the freshness gate changes only WHICH games are "
        "admitted (780 -> 719), not the underlying per-team-week quantity, so it shares the "
        "parent trait exactly",
    ),
    "nflcom_refresh_out1_starter_on_chain": (
        "starter_out",
        "refresh",
        "overlay flag = starter_out >= 1 (refresh:224)",
    ),
    "nflcom_refresh_net_out_diff_ge1_on_chain": (
        "total_out",
        "refresh",
        "overlay flag = picked total_out minus opponent total_out >= 1 (refresh:226); the "
        "differential is a game-level, pick-orientation-dependent derivative of the per-team-week "
        "total_out count, which is the measurable parent",
    ),
}

NFLCOM_FLAGS: dict[str, tuple[str, float]] = {
    "nflcom_friday_out_count_ge2": ("out_count", 2.0),
    "nflcom_friday_q_or_worse_starter_caliber": ("starter_q_or_worse", 1.0),
    "nflcom_friday_new_saturday_designation": ("new_vs_tuesday", 1.0),
    "nflcom_refresh_out2_starters_on_chain": ("starter_out", 2.0),
    "nflcom_refresh_out2_starters_on_chain_gate_admitted": ("starter_out", 2.0),
    "nflcom_refresh_out1_starter_on_chain": ("starter_out", 1.0),
    "nflcom_refresh_net_out_diff_ge1_on_chain": ("total_out", 1.0),
}


def build_nflcom_team_week() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Team-week frame carrying every NFL.com count, built by the screens."""

    import nflcom_friday_designation_screen as nfd
    import nflcom_friday_refresh_feature as nfr

    with pinned_nflcom_snapshot(nfd):
        schedules_path = nfd.latest(REPO / "data" / "raw", "*/schedules.parquet")
        long = nfd.load_population(schedules_path)
        qa, report_counts = nfd.load_report_flags(REPO / "data" / "raw" / "nflcom_injuries")
        snaps_path = nfd.latest(REPO / "data" / "players" / "raw", "*/snap_counts.parquet")
        starter_exact, starter_fuzzy = nfd.build_starter_keys(snaps_path)
        players_root = REPO / "data" / "players" / "raw"
        nflverse_exact, nflverse_fuzzy = nfd.build_tuesday_visible(players_root)
        work = nfd.attach_flags(
            long, qa, starter_exact, starter_fuzzy, nflverse_exact, nflverse_fuzzy
        )
        out_counts, refresh_counts = nfr.build_out_counts(
            REPO / "data" / "raw" / "nflcom_injuries", snaps_path
        )

    work = work.merge(out_counts, on=["season", "week", "team"], how="left")
    work[["total_out", "starter_out"]] = work[["total_out", "starter_out"]].fillna(0)
    work = work.rename(columns={"team": "team_id"})
    info = {
        "snapshot": str(NFLCOM_SNAPSHOT),
        "schedules": str(schedules_path),
        "snaps": str(snaps_path),
        "report_counts": report_counts,
        "refresh_counts": {k: int(v) for k, v in refresh_counts.items()},
        "team_games": len(work),
    }
    return work, info


def measure_nflcom_group(
    entries: dict[str, dict[str, Any]], *, n_boot: int
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    work, info = build_nflcom_team_week()
    cache: dict[tuple[str, tuple[int, int]], dict[str, Any]] = {}
    replications: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for entry_name, (metric, source, note) in NFLCOM_CELL_PARENTS.items():
        seasons = entries[entry_name]["seasons"]
        key = (metric, seasons)
        if key not in cache:
            window = work.loc[work["season"].between(*seasons)]
            is_degenerate, stats = _near_constant(window[metric])
            measured = rlib.measure_reliability(
                work, metric, method=rlib.METHOD_TRAIT, seasons=seasons, n_boot=n_boot
            )
            measured["metric"] = metric
            measured["degenerate"] = is_degenerate
            measured["column_stats"] = stats
            measured["within_unit"] = within_unit_variation(
                work, metric, unit_col="team_id", seasons=seasons
            )
            cache[key] = measured
        measured = cache[key]

        column, threshold = NFLCOM_FLAGS[entry_name]
        window = work.loc[work["season"].between(*seasons)].reset_index(drop=True)
        replication = rlib.half_season_replication(
            window, window[column] >= threshold, outcome_col="team_cover"
        )
        replications[entry_name] = replication

        extra = {
            "flag_definition": note,
            "source_script": source,
            "column_stats": measured["column_stats"],
            "within_unit_variation": measured["within_unit"],
            "half_season_replication": replication,
        }
        rows.append(
            _row(
                entry=entry_name,
                parent=metric,
                unit="team-season",
                seasons=seasons,
                measured=measured,
                method_tag="TRAIT",
                provenance=(
                    "scripts/nflcom_friday_designation_screen.py:239-250 (attach_flags) and "
                    "scripts/nflcom_friday_refresh_feature.py:116-144 (build_out_counts) build "
                    "the per-(season, week, team) counts this cell thresholds; snapshot pinned "
                    f"to {NFLCOM_SNAPSHOT.name} because the two newer snapshots hold zero rows. "
                    + note
                ),
                status=STATUS_NEAR_CONSTANT if measured["degenerate"] else None,
                extra=extra,
            )
        )
    return rows, {"inputs": info, "replications": replications}, work


# ==========================================================================
# Group 4 -- interim_hc_* (7) and player_arrests_* (5): EXPOSURE
# ==========================================================================
#
# Every one of these cells is a per-team-game FLAG with no continuous parent
# in its builder. Read: src/nfl_ats/experiment_runner.py -- ``_flag_interim_hc_
# active`` (2707), ``_flag_interim_hc_first_game`` (2738), ``_flag_interim_hc_
# home`` (2764), ``_flag_interim_hc_fired_year_one`` (2791) and
# ``_flag_recent_player_arrest`` (643), each of which hard-declares
# ``reliability=None`` with a ``reliability_note`` saying the construct is "a
# one-off situational event ... there is nothing to split-half" / "a per-game
# event exposure, not a persistent team trait". Each cell's own spec under
# registry/experiment_specs/ repeats that as ``reliability_check.method =
# "not_applicable"``.
#
# METHOD_EXPOSURE measures a strictly different thing from what those notes
# decline: not "is this a persistent trait" but "does the flag mark stable
# team-season structure, or pure event churn". A low value here is NOT an
# admissible ``no_split_half_reliability`` ground, and nothing here reclassifies
# any cell.

FLAG_CELLS: dict[str, dict[str, Any]] = {
    "interim_hc_active": {
        "builder": "interim_hc_active",
        "params": {},
        "flag_column": "under_interim",
    },
    "interim_hc_active_era_2009_2017": {
        "builder": "interim_hc_active",
        "params": {},
        "flag_column": "under_interim",
    },
    "interim_hc_active_era_2018_2025": {
        "builder": "interim_hc_active",
        "params": {},
        "flag_column": "under_interim",
    },
    "interim_hc_active_excl_suspension": {
        "builder": "interim_hc_active",
        "params": {"exclude_suspension_cases": True},
        "flag_column": "under_interim",
    },
    "interim_hc_first_game": {
        "builder": "interim_hc_first_game",
        "params": {},
        "flag_column": "first_game_under_interim",
    },
    "interim_hc_home_within_interim": {
        "builder": "interim_hc_home",
        "params": {},
        "flag_column": "is_home",
        # The hazard the sweep doc names, in its purest form: every team plays
        # almost exactly half its games at home, so the home-exposure rate has
        # essentially NO cross-team variance, and the odd/even split-half is
        # forced negative by the balance constraint (a home-heavy odd half
        # forces a road-heavy even half). Any |r| here is a property of the
        # schedule, not of the world. Reported with the number shown, never
        # recorded.
        "structural": (
            "structurally_balanced_by_schedule: is_home exposure is ~0.5 for every team-season "
            "by schedule construction, so the odd/even split-half is forced negative by the "
            "balance constraint rather than measuring anything about a team"
        ),
    },
    "interim_hc_fired_year_one": {
        "builder": "interim_hc_fired_year_one",
        "params": {},
        "flag_column": "fired_coach_was_year_one",
    },
    "player_arrests_recent_14d_fade_close": {
        "builder": "recent_player_arrest",
        "params": {"window_days": 14},
        "flag_column": None,
    },
    "player_arrests_recent_14d_fade_opener": {
        "builder": "recent_player_arrest",
        "params": {"window_days": 14},
        "flag_column": None,
    },
    "player_arrests_recent_14d_back_side_policy_opener": {
        "builder": "recent_player_arrest",
        "params": {"window_days": 14},
        "flag_column": None,
        "note": (
            "The policy cell flips the frozen production opener pick to the sole team carrying a "
            "broad USA Today incident 1-14 days before the Tuesday decision date; its exposure "
            "construct is the same recent_player_arrest flag, restricted at scoring time to games "
            "where production opposes the flagged team"
        ),
    },
    "player_arrests_violent_person_recent_14d_fade_close": {
        "builder": "recent_player_arrest",
        "params": {"window_days": 14, "category_contains_any": None},
        "flag_column": None,
    },
    "player_arrests_violent_person_recent_14d_fade_opener": {
        "builder": "recent_player_arrest",
        "params": {"window_days": 14, "category_contains_any": None},
        "flag_column": None,
    },
}


def _spec_params(name: str) -> dict[str, Any]:
    payload = json.loads(
        (REPO / "registry" / "experiment_specs" / f"{name}.json").read_text(encoding="utf-8")
    )
    return dict(payload["construct"].get("params", {}))


def measure_flag_group(
    entries: dict[str, dict[str, Any]], *, n_boot: int
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    from nfl_ats.experiment_runner import FLAG_BUILDERS

    features = pd.read_parquet(REPO / "data" / "processed" / "game_features.parquet")
    rows: list[dict[str, Any]] = []
    control_long: pd.DataFrame | None = None
    for entry_name, config in FLAG_CELLS.items():
        seasons = entries[entry_name]["seasons"]
        spec_path = REPO / "registry" / "experiment_specs" / f"{entry_name}.json"
        params = _spec_params(entry_name) if spec_path.is_file() else dict(config["params"])
        builder = FLAG_BUILDERS[config["builder"]]
        construct = builder.build(features, seasons, params, REPO)

        table = construct.table.reset_index(drop=True)
        flag = construct.flag.reset_index(drop=True)
        eligible = construct.eligible
        if eligible is not None:
            eligible = eligible.reset_index(drop=True)
            table = table.loc[eligible].reset_index(drop=True)
            flag = flag.loc[eligible].reset_index(drop=True)

        long = flag_exposure_long(table, flag, team_col="team")
        if control_long is None:
            control_long = long
        window = long.loc[long["season"].between(*seasons)]
        is_degenerate, stats = _near_constant(window["exposure"], binary_flag=True)
        measured = rlib.measure_reliability(
            long, "exposure", method=rlib.METHOD_EXPOSURE, seasons=seasons, n_boot=n_boot
        )
        measured["metric"] = f"{config['builder']}:exposure"
        within = within_unit_variation(long, "exposure", unit_col="team_id", seasons=seasons)
        degenerate_reason: str | None = None
        if is_degenerate:
            degenerate_reason = "empty_or_all_missing"
        elif within["share_zero_within_variance"] == 1.0:
            is_degenerate = True
            degenerate_reason = "season_constant_within_unit"
        elif config.get("structural"):
            is_degenerate = True
            degenerate_reason = str(config["structural"])

        scored = table.loc[table["season"].between(*seasons)].reset_index(drop=True)
        scored_flag = flag.loc[table["season"].between(*seasons)].reset_index(drop=True)
        replication = rlib.half_season_replication(scored, scored_flag, outcome_col="team_covered")

        rows.append(
            _row(
                entry=entry_name,
                parent=f"{config['builder']} flag exposure rate per team-season",
                unit="team-season",
                seasons=seasons,
                measured=measured,
                method_tag="EXPOSURE",
                provenance=(
                    f"nfl_ats.experiment_runner.FLAG_BUILDERS['{config['builder']}'] built with "
                    f"the cell's own spec params {params if params else '{}'}; the builder's own "
                    "reliability_note declares the construct not a persistent team trait, and "
                    "EXPOSURE measures a different quantity (does the flag mark stable "
                    "team-season structure) that is NOT an admissible closing ground. "
                    + str(config.get("note", ""))
                ).strip(),
                status=STATUS_NEAR_CONSTANT if is_degenerate else None,
                extra={
                    "flag_column": config["flag_column"],
                    "builder_reliability_note": construct.reliability_note,
                    "restricted_to_eligible": eligible is not None,
                    "near_constant_reason": degenerate_reason,
                    "degenerate_reading": _member_summary(measured) if is_degenerate else None,
                    "column_stats": stats,
                    "within_unit_variation": within,
                    "half_season_replication": replication,
                    "n_flag": int(scored_flag.sum()),
                    "n_population": len(scored),
                },
            )
        )
    assert control_long is not None
    return rows, control_long


# ==========================================================================
# Group 5 -- fluview_* (2 cells)
# ==========================================================================
#
# Parent: the CDC/Delphi FluView regional influenza-like-illness index ``ili``,
# read point-in-time-safe as of each game's own Tuesday cutoff. Read:
# scripts/fluview_battery_screen.py:159-196 (build_checkpoint_tables),
# :262-300 (attach_asof_ili -> home_ili/away_ili) and :303-318
# (build_state_week_panel). The registry's screen-stage siblings
# fluview_home_market_elevated / fluview_away_market_elevated already carry
# reliability 0.98144757 measured on the STATE-season panel over the full
# 2010-2025 archive (read: registry/weak_signals.json). The two cells here are
# opener-graded confirmations on much narrower windows, so both the team-season
# number (which is what METHOD_TRAIT's unit means) and the state-season number
# (comparable to the recorded 0.98) are measured on the cell's OWN seasons.

FLUVIEW_CELLS = (
    "fluview_home_market_elevated_opener_confirmation",
    "fluview_home_market_elevated_opener_confirmation_2022_2023",
)


def build_fluview_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    import fluview_battery_screen as fbs

    schedules = fbs._latest_schedules()
    fluview = pd.read_parquet(fbs.DEFAULT_FLUVIEW)
    df = fbs.load_schedules(schedules)
    checkpoints = fbs.build_checkpoint_tables(fluview.loc[fluview["region"] != "nat"])
    df = fbs.attach_asof_ili(df, checkpoints)
    panel = fbs.build_state_week_panel(df).rename(columns={"state": "team_id"})

    team_long = pd.concat(
        [
            df[["home_team", "season", "week", "home_ili"]].rename(
                columns={"home_team": "team_id", "home_ili": "ili"}
            ),
            df[["away_team", "season", "week", "away_ili"]].rename(
                columns={"away_team": "team_id", "away_ili": "ili"}
            ),
        ],
        ignore_index=True,
    )
    info = {
        "schedules": str(schedules),
        "fluview_snapshot": str(fbs.DEFAULT_FLUVIEW),
        "games": len(df),
        "state_week_rows": len(panel),
        "team_week_rows": len(team_long),
    }
    return team_long, panel, info


def measure_fluview_group(
    entries: dict[str, dict[str, Any]], *, n_boot: int
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    team_long, panel, info = build_fluview_frames()
    rows: list[dict[str, Any]] = []
    for entry_name in FLUVIEW_CELLS:
        seasons = entries[entry_name]["seasons"]
        window = team_long.loc[team_long["season"].between(*seasons)]
        is_degenerate, stats = _near_constant(window["ili"])
        measured = rlib.measure_reliability(
            team_long, "ili", method=rlib.METHOD_TRAIT, seasons=seasons, n_boot=n_boot
        )
        measured["metric"] = "ili"
        state_level = rlib.measure_reliability(
            panel, "ili", method=rlib.METHOD_TRAIT, seasons=seasons, n_boot=n_boot
        )
        rows.append(
            _row(
                entry=entry_name,
                parent="ili (as-of FluView influenza-like-illness index)",
                unit="team-season",
                seasons=seasons,
                measured=measured,
                method_tag="TRAIT",
                provenance=(
                    "scripts/fluview_battery_screen.py build_checkpoint_tables (159) + "
                    "attach_asof_ili (262) + build_state_week_panel (303), imported; the cell "
                    "thresholds this state's as-of ili at its frozen top decile. The screen-stage "
                    "siblings fluview_home/away_market_elevated already carry 0.98144757 measured "
                    "on the STATE-season panel over 2010-2025; measured here on this cell's own "
                    "seasons at the team-season unit METHOD_TRAIT names, with the state-season "
                    "number reported alongside for comparability."
                ),
                status=STATUS_NEAR_CONSTANT if is_degenerate else None,
                extra={
                    "column_stats": stats,
                    "state_season_comparison": _member_summary(state_level),
                    "registry_screen_stage_reliability": 0.9814475666016571,
                    "state_level_note": (
                        "ili is a STATE-week series mapped onto teams, so two teams in one state "
                        "share a value and the team-season number is not independent across "
                        "co-located teams."
                    ),
                    "half_season_replication": {
                        "status": "not_applicable_no_flag",
                        "note": (
                            "This cell is an opener-graded model-vs-model paired accuracy delta "
                            "(weak_stack vs weak_stack_fluview_home), not a subset flag."
                        ),
                    },
                },
            )
        )
    return rows, info, team_long


# ==========================================================================
# Group 6 -- season-constant parents: ffc_adp_* (6) and hc_year_one_fade (1)
# ==========================================================================
#
# ffc_adp: MEASURED 2026-09-01 -- artifacts/ffc_adp/20260822T004750Z/
# team_top8_feasibility.parquet has NO ``week`` column and is unique on
# (year, scoring, franchise_code), and the screen's own disclosure
# (scripts/ffc_adp_divergence_screen.py:422-426) says "ADP is a preseason
# covariate with NO in-season refresh". ``residual_z`` (145-175) is a
# deterministic within-season transform of it. So every ADP quantity is
# CONSTANT within a team-season, and its odd/even-week split-half is exactly
# +1.0 by construction -- an artifact of the degeneracy, not a trait.
#
# hc_year_one_fade: the ``year_one`` flag is built per (team, season) at
# scripts/hc_year_one_fade.py:120-135 from ``team_season_primary_coach``
# (107-117, the modal coach over the whole REG season) and broadcast to every
# week (:193). Same degeneracy.
#
# Both are reported with the degenerate number SHOWN, never recorded, plus a
# reported-only cross-SEASON diagnostic (does a team's value in odd seasons
# predict its value in even seasons) which is a legitimate question about the
# same quantity but is not one of the sweep's three methods and so is never
# written to the registry.

FFC_CELLS: dict[str, str] = {
    "ffc_adp_cellA_highadp_underdog_back_ppr_w14": "ppr",
    "ffc_adp_cellB_adpwins_residual_pos_back_ppr_w14": "ppr",
    "ffc_adp_cellC_highadp_underdog_back_ppr_w12": "ppr",
    "ffc_adp_cellD_adpwins_residual_pos_back_ppr_w12": "ppr",
    "ffc_adp_robust_std_cellA_highadp_underdog_back_w14": "standard",
    "ffc_adp_robust_std_cellB_adpwins_residual_pos_back_w14": "standard",
}

FFC_CELL_METRIC: dict[str, str] = {
    "ffc_adp_cellA_highadp_underdog_back_ppr_w14": "mean_adp_top8",
    "ffc_adp_cellB_adpwins_residual_pos_back_ppr_w14": "residual_z",
    "ffc_adp_cellC_highadp_underdog_back_ppr_w12": "mean_adp_top8",
    "ffc_adp_cellD_adpwins_residual_pos_back_ppr_w12": "residual_z",
    "ffc_adp_robust_std_cellA_highadp_underdog_back_w14": "mean_adp_top8",
    "ffc_adp_robust_std_cellB_adpwins_residual_pos_back_w14": "residual_z",
}


def cross_season_stability(frame: pd.DataFrame, metric: str, *, unit_col: str) -> dict[str, Any]:
    """Reported-only: odd-season vs even-season correlation of a unit's value.

    For a quantity that is constant inside a season, the week split says
    nothing and this says everything: is a team's ADP (or year-one status)
    stable across seasons? It is NOT one of the sweep's three methods, so it
    is never recorded -- it is here so the degenerate cells are not left with
    no information at all.
    """

    work = frame.loc[:, [unit_col, "season", metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    work["parity"] = np.where(work["season"] % 2 == 0, "even", "odd")
    means = work.groupby([unit_col, "parity"])[metric].mean().unstack("parity").dropna()
    if len(means) < 3 or means["odd"].std() == 0 or means["even"].std() == 0:
        return {"n_units": len(means), "pearson_r": None, "note": "too few units or constant"}
    return {
        "n_units": len(means),
        "pearson_r": float(np.corrcoef(means["odd"], means["even"])[0, 1]),
        "note": (
            "Odd-season vs even-season correlation of the unit's own mean. Reported only -- not "
            "one of the sweep's three methods and never written to the registry."
        ),
    }


def measure_ffc_group(
    entries: dict[str, dict[str, Any]], *, n_boot: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import ffc_adp_divergence_screen as ffc

    schedule = ffc.load_schedule(ffc.latest_schedules())
    prior_wins = ffc.prior_season_wins(schedule)
    adp_root = REPO / "artifacts" / "ffc_adp" / ffc.DEFAULT_ADP_SNAPSHOT
    featured = {
        scoring: ffc.add_residual_z(ffc.load_adp(adp_root, scoring), prior_wins)
        for scoring in ("ppr", "standard")
    }
    # The screen's own population clip: weeks 1-4 (:313, WEEK_BLOCK_MAX).
    games = schedule.loc[schedule["week"] <= ffc.WEEK_BLOCK_MAX]
    weeks = pd.concat(
        [
            games[["season", "week", "home_team"]].rename(columns={"home_team": "team_id"}),
            games[["season", "week", "away_team"]].rename(columns={"away_team": "team_id"}),
        ],
        ignore_index=True,
    )

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for entry_name, scoring in FFC_CELLS.items():
        seasons = entries[entry_name]["seasons"]
        metric = FFC_CELL_METRIC[entry_name]
        adp = featured[scoring]
        long = weeks.merge(
            adp.rename(columns={"franchise_code": "team_id"})[["season", "team_id", metric]],
            on=["season", "team_id"],
            how="left",
        )
        within = within_unit_variation(long, metric, unit_col="team_id", seasons=seasons)
        measured = rlib.measure_reliability(
            long, metric, method=rlib.METHOD_TRAIT, seasons=seasons, n_boot=n_boot
        )
        measured["metric"] = metric
        cross = cross_season_stability(
            adp.rename(columns={"franchise_code": "team_id"}), metric, unit_col="team_id"
        )
        diagnostics[entry_name] = {
            "degenerate_week_split_reading": _member_summary(measured),
            "within_unit_variation": within,
            "cross_season_stability_reported_only": cross,
        }
        rows.append(
            _row(
                entry=entry_name,
                parent=f"{metric} ({scoring} scoring, preseason FFC ADP aggregate)",
                unit="team-season",
                seasons=seasons,
                measured=None,
                method_tag="TRAIT",
                provenance=(
                    "scripts/ffc_adp_divergence_screen.py load_adp (126) / add_quality_ranks "
                    "(137) / add_residual_z (145) imported; MEASURED 2026-09-01 that "
                    "artifacts/ffc_adp/20260822T004750Z/team_top8_feasibility.parquet has no "
                    "week column and is unique on (year, scoring, franchise_code), so the "
                    "quantity is constant within a team-season and an odd/even-WEEK split-half "
                    "is +1.0 by construction."
                ),
                status=STATUS_NEAR_CONSTANT,
                extra={
                    "near_constant_reason": "season_constant_within_unit",
                    "degenerate_week_split_reading": _member_summary(measured),
                    "within_unit_variation": within,
                    "cross_season_stability_reported_only": cross,
                    "half_season_replication": {
                        "status": "not_applicable_flag_needs_market_side",
                        "note": (
                            "The cell's flag is (top-tercile ADP roster) AND (priced as underdog) "
                            "on exactly one side of the game; the market half of that conjunction "
                            "is not part of the ADP parent quantity being measured here."
                        ),
                    },
                },
            )
        )
    return rows, diagnostics


def measure_hc_year_one(
    entries: dict[str, dict[str, Any]], *, n_boot: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import hc_year_one_fade as hcy

    schedules = pd.read_parquet(hcy.DEFAULT_SCHEDULES)
    features = pd.read_parquet(hcy.DEFAULT_FEATURES)
    long = hcy.build_team_game_table(schedules, features)
    primary = hcy.team_season_primary_coach(long)
    flagged = hcy.flag_year_one(primary)

    seasons = entries["hc_year_one_fade"]["seasons"]
    reg = long.loc[long["game_type"].eq("REG")].reset_index(drop=True)
    merged = reg.merge(flagged[["team", "season", "year_one"]], on=["team", "season"], how="inner")
    merged = merged.rename(columns={"team": "team_id"})
    merged["exposure"] = merged["year_one"].fillna(False).astype(bool).astype(float)
    within = within_unit_variation(merged, "exposure", unit_col="team_id", seasons=seasons)
    measured = rlib.measure_reliability(
        merged, "exposure", method=rlib.METHOD_EXPOSURE, seasons=seasons, n_boot=n_boot
    )
    measured["metric"] = "year_one exposure"
    cross = cross_season_stability(merged, "exposure", unit_col="team_id")
    week_scoped = merged.loc[merged["week"] <= 8].reset_index(drop=True)
    replication = rlib.half_season_replication(
        week_scoped.loc[week_scoped["season"].between(*seasons)].reset_index(drop=True),
        week_scoped.loc[week_scoped["season"].between(*seasons)]
        .reset_index(drop=True)["year_one"]
        .fillna(False)
        .astype(bool),
        outcome_col="team_covered",
    )
    row = _row(
        entry="hc_year_one_fade",
        parent="year_one (first-year head coach, per team-season)",
        unit="team-season",
        seasons=seasons,
        measured=None,
        method_tag="EXPOSURE",
        provenance=(
            "scripts/hc_year_one_fade.py build_team_game_table (63) / "
            "team_season_primary_coach (107) / flag_year_one (120) imported; the flag is built "
            "per (team, season) from the modal REG-season coach and broadcast to every week "
            "(:193), so it is constant within a team-season and an odd/even-WEEK split-half is "
            "+1.0 by construction."
        ),
        status=STATUS_NEAR_CONSTANT,
        extra={
            "near_constant_reason": "season_constant_within_unit",
            "degenerate_week_split_reading": _member_summary(measured),
            "within_unit_variation": within,
            "cross_season_stability_reported_only": cross,
            "half_season_replication": replication,
        },
    )
    return [row], {
        "degenerate_week_split_reading": _member_summary(measured),
        "cross_season_stability_reported_only": cross,
        "half_season_replication": replication,
    }


# ==========================================================================
# Group 7 -- qb_news_backup_visible_by_deadline_screen (1 cell)
# ==========================================================================
#
# The cell's flag is ``_news_visible_flag`` (read:
# scripts/qb_backup_news_visibility.py:684-692): a backup-QB start whose
# earliest PFT/PFR news match lands in a pre-Sunday visibility bucket, joined
# onto the eligible population on ``game_id|team``. The eligible population
# (:550-552) is deterministic and is rebuilt here by importing the screen's own
# ``ground_truth_long_table``; the PFR half of the match, however, needs a live
# network fetch that the script itself budget-truncates
# (PFR_MAX_FETCHES / PFR_FETCH_WALLCLOCK_BUDGET_SECONDS, :436-444), so the
# BUCKET column is taken from the frozen artifact the registry cell was
# recorded from rather than re-fetched. No continuous parent exists: match_pft
# (:290-348) keeps only the earliest timestamp and never a headline count.


def measure_qb_news(
    entries: dict[str, dict[str, Any]], *, n_boot: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seasons = entries["qb_news_backup_visible_by_deadline_screen"]["seasons"]
    detail_path = QB_NEWS_ARTIFACT / "backup_rows_detail.parquet"
    if not detail_path.is_file():
        return [
            _row(
                entry="qb_news_backup_visible_by_deadline_screen",
                parent="_news_visible_flag",
                unit="team-season",
                seasons=seasons,
                measured=None,
                method_tag="EXPOSURE",
                provenance=f"frozen artifact absent: {detail_path}",
                status=STATUS_DATA_MISSING,
            )
        ], {"missing": str(detail_path)}

    import qb_backup_news_visibility as qbn

    long_df = qbn.ground_truth_long_table()
    scoped = long_df.loc[long_df["season"].between(*qbn.SEASONS)].reset_index(drop=True)
    eligible = scoped.loc[scoped["backup_qb_flag"].notna()].reset_index(drop=True)

    detail = pd.read_parquet(detail_path)
    pre_sunday = detail["bucket"].isin(
        ["by_tuesday_noon", "by_friday", "by_sunday_10am_or_kickoff"]
    )
    visible_ids = set(
        detail.loc[pre_sunday, "game_id"].astype(str)
        + "|"
        + detail.loc[pre_sunday, "team"].astype(str)
    )
    key = eligible["game_id"].astype(str) + "|" + eligible["team"].astype(str)
    flag = key.isin(visible_ids)

    long = flag_exposure_long(eligible, flag, team_col="team")
    window = long.loc[long["season"].between(*seasons)]
    is_degenerate, stats = _near_constant(window["exposure"], binary_flag=True)
    measured = rlib.measure_reliability(
        long, "exposure", method=rlib.METHOD_EXPOSURE, seasons=seasons, n_boot=n_boot
    )
    measured["metric"] = "_news_visible_flag exposure"
    scored = eligible.loc[eligible["season"].between(*seasons)].reset_index(drop=True)
    scored_flag = flag.loc[eligible["season"].between(*seasons)].reset_index(drop=True)
    replication = rlib.half_season_replication(scored, scored_flag, outcome_col="team_covered")
    row = _row(
        entry="qb_news_backup_visible_by_deadline_screen",
        parent="_news_visible_flag exposure rate per team-season",
        unit="team-season",
        seasons=seasons,
        measured=measured,
        method_tag="EXPOSURE",
        provenance=(
            "scripts/qb_backup_news_visibility.py ground_truth_long_table (181) rebuilds the "
            "eligible population; the visibility bucket comes from the frozen artifact "
            f"{QB_NEWS_ARTIFACT.name}/backup_rows_detail.parquet because the screen's PFR phase "
            "needs a live, budget-truncated network fetch (:436-444) and is not deterministically "
            "reproducible. Flag composed exactly as :684-692. No continuous parent exists: "
            "match_pft (:290-348) keeps only the earliest match timestamp, never a count."
        ),
        status=STATUS_NEAR_CONSTANT if is_degenerate else None,
        extra={
            "column_stats": stats,
            "n_flag": int(scored_flag.sum()),
            "n_population": len(scored),
            "half_season_replication": replication,
            "within_unit_variation": within_unit_variation(
                long, "exposure", unit_col="team_id", seasons=seasons
            ),
        },
    )
    return [row], {
        "eligible_team_games": len(eligible),
        "backup_rows_in_artifact": len(detail),
        "pre_sunday_visible": int(pre_sunday.sum()),
    }


# ==========================================================================
# Driver
# ==========================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=rlib.N_BOOT)
    parser.add_argument(
        "--injury-cache-dir",
        type=Path,
        default=None,
        help="reuse (or populate) rebuilt injury-arm feature tables here",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    started = time.time()
    names = sorted(
        set(INJURY_CELL_ARMS)
        | set(PLAYER_FAMILY_BLOCKS)
        | set(PARTICIPATION_BLOCKS)
        | set(NFLCOM_CELL_PARENTS)
        | set(FLAG_CELLS)
        | set(FLUVIEW_CELLS)
        | set(FFC_CELLS)
        | {"hc_year_one_fade", "qb_news_backup_visible_by_deadline_screen"}
    )
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        declared = sorted(e["name"] for e in manifest["groups"]["health_roster"]["entries"])
        if declared != names:
            missing = sorted(set(declared) - set(names))
            extra = sorted(set(names) - set(declared))
            raise SystemExit(f"manifest mismatch: missing={missing} extra={extra}")
    entries = target_entries(names)
    print(f"=== {len(entries)} health_roster registry cells in scope ===")

    cache_dir = args.injury_cache_dir or (
        REPO / "artifacts" / "reliability_sweep" / "health_roster" / "_injury_arms"
    )

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    control_frames: dict[str, tuple[pd.DataFrame, str]] = {}

    print("-- injury_value_lost_* (8)")
    injury_rows, injury_arms, injury_long = measure_injury_group(
        entries, cache_dir, n_boot=args.n_boot
    )
    rows.extend(injury_rows)
    diagnostics["injury_arm_measurements"] = injury_arms
    control_frames["injury_value_lost (team-week, saturday arm)"] = (injury_long, "team_id")

    print("-- player_family_base_vs_* (8)")
    family_rows, family_long = measure_block_group(
        entries,
        PLAYER_FAMILY_BLOCKS,
        PLAYER_VALUE_TABLE,
        (
            "src/nfl_ats/constants.py FEATURE_FAMILIES + FEATURE_SETS (686-775) and "
            "src/nfl_ats/margin.py:96-121 define each arm's feature set; the block a cell adds "
            "over 'base' is the FEATURE_FAMILIES difference, and each member's parent team-week "
            "column is the diff_ column with the prefix stripped. Table = "
            "game_features_player_value.parquet, the exact path in "
            "artifacts/player_experiments/20260813T122348Z/metadata.json."
        ),
        n_boot=args.n_boot,
    )
    rows.extend(family_rows)
    control_frames["player_family (team-week)"] = (family_long, "team_id")

    print("-- participation_offense_defense_rapm (1)")
    participation_rows, participation_long = measure_block_group(
        entries,
        PARTICIPATION_BLOCKS,
        PARTICIPATION_TABLE,
        (
            "artifacts/participation_experiments/20260813T132030Z/metadata.json records "
            "baseline_profile=player_value, candidate_profile=player_participation, so the "
            "marginal block is FEATURE_FAMILIES['player_participation_values'] on "
            "game_features_player_participation.parquet (that metadata's own feature_table path)."
        ),
        n_boot=args.n_boot,
    )
    rows.extend(participation_rows)
    control_frames["participation (team-week)"] = (participation_long, "team_id")

    print("-- nflcom_* (7)")
    nflcom_rows, nflcom_info, nflcom_long = measure_nflcom_group(entries, n_boot=args.n_boot)
    rows.extend(nflcom_rows)
    diagnostics["nflcom"] = nflcom_info
    control_frames["nflcom (team-week counts)"] = (nflcom_long, "team_id")

    print("-- interim_hc_* + player_arrests_* (12)")
    flag_rows, flag_long = measure_flag_group(entries, n_boot=args.n_boot)
    rows.extend(flag_rows)
    control_frames["flag exposure (team-week)"] = (flag_long, "team_id")

    print("-- fluview_* (2)")
    fluview_rows, fluview_info, fluview_long = measure_fluview_group(entries, n_boot=args.n_boot)
    rows.extend(fluview_rows)
    diagnostics["fluview"] = fluview_info
    control_frames["fluview (team-week ili)"] = (fluview_long, "team_id")

    print("-- ffc_adp_* (6)")
    ffc_rows, ffc_diag = measure_ffc_group(entries, n_boot=args.n_boot)
    rows.extend(ffc_rows)
    diagnostics["ffc_adp"] = ffc_diag

    print("-- hc_year_one_fade (1)")
    hc_rows, hc_diag = measure_hc_year_one(entries, n_boot=args.n_boot)
    rows.extend(hc_rows)
    diagnostics["hc_year_one_fade"] = hc_diag

    print("-- qb_news_backup_visible_by_deadline_screen (1)")
    qb_rows, qb_diag = measure_qb_news(entries, n_boot=args.n_boot)
    rows.extend(qb_rows)
    diagnostics["qb_news"] = qb_diag

    rows.sort(key=lambda r: r["entry"])
    if len(rows) != len(entries):
        raise SystemExit(f"produced {len(rows)} rows for {len(entries)} entries")

    # ---- battery-level replication correlations -------------------------
    batteries = {
        "nflcom_friday_designation": [
            "nflcom_friday_out_count_ge2",
            "nflcom_friday_q_or_worse_starter_caliber",
            "nflcom_friday_new_saturday_designation",
        ],
        "nflcom_friday_refresh": [
            "nflcom_refresh_out2_starters_on_chain",
            "nflcom_refresh_out2_starters_on_chain_gate_admitted",
            "nflcom_refresh_out1_starter_on_chain",
            "nflcom_refresh_net_out_diff_ge1_on_chain",
        ],
        "interim_hc": [name for name in FLAG_CELLS if name.startswith("interim_hc")],
        "player_arrests": [name for name in FLAG_CELLS if name.startswith("player_arrests")],
    }
    by_entry = {row["entry"]: row for row in rows}
    battery_replication: dict[str, Any] = {}
    for battery, members in batteries.items():
        cells = {
            name: by_entry[name]["half_season_replication"]
            for name in members
            if isinstance(by_entry[name].get("half_season_replication"), dict)
            and "odd_seasons" in by_entry[name]["half_season_replication"]
        }
        battery_replication[battery] = rlib.battery_replication_correlation(cells)

    # ---- positive control at each distinct unit structure ---------------
    controls: dict[str, Any] = {}
    windows = sorted({tuple(row["seasons"]) for row in rows})
    for label, (frame, unit_col) in control_frames.items():
        per_window: dict[str, Any] = {}
        for window in windows:
            restricted = frame.loc[
                pd.to_numeric(frame["season"], errors="coerce").between(window[0], window[1])
            ]
            if restricted.empty:
                continue
            per_window[f"{window[0]}-{window[1]}"] = rlib.positive_control(
                restricted, unit_col=unit_col, n_boot=1000
            )
        controls[label] = per_window

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = REPO / "artifacts" / "reliability_sweep" / "health_roster" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    configuration = {
        "command": "reliability-health-roster",
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": args.n_boot,
        "min_units": rlib.MIN_UNITS,
        "entries": names,
        "player_value_table": str(PLAYER_VALUE_TABLE),
        "participation_table": str(PARTICIPATION_TABLE),
        "nflcom_snapshot": str(NFLCOM_SNAPSHOT),
        "qb_news_artifact": str(QB_NEWS_ARTIFACT),
    }
    measured_rows = [r for r in rows if r["status"] == rlib.STATUS_MEASURED]
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": args.n_boot,
        "min_units": rlib.MIN_UNITS,
        "methods": {"trait": rlib.METHOD_TRAIT, "exposure": rlib.METHOD_EXPOSURE},
        "rolling_aggregate_caveat": (
            "Most continuous parents here (lineup continuity, injury unavailability, QB ratings, "
            "injury value lost) are season-to-date or span-smoothed ROLLING aggregates, so their "
            "odd/even-week split-half is high partly by construction of the smoothing. The number "
            "is still the feature as the model consumes it, and matches the registry's existing "
            "0.98-0.99 entries, but it must not be read as 'this construct is 98% repeatable "
            "news'."
        ),
        "closing_grounds_note": (
            "Measure-only. Nothing here closes or reclassifies any cell. EXPOSURE measurements "
            "are NOT an admissible no_split_half_reliability ground; only a TRAIT measurement is, "
            "and even then only via a separate, explicit closure step by the owner."
        ),
        "results": rows,
        "battery_replication_correlations": battery_replication,
        "positive_control": controls,
        "diagnostics": diagnostics,
        "provenance": artifact_provenance(configuration, PLAYER_VALUE_TABLE, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="reliability-health-roster",
        metrics={
            "n_entries": len(rows),
            "n_measured": len(measured_rows),
            "n_unmeasured": len(rows) - len(measured_rows),
        },
        notes=(
            "Measure-only split-half reliability for the 46 health_roster registry cells; "
            "nothing is closed or reclassified, per AGENTS.md's binding closing-grounds taxonomy."
        ),
    )
    results_path = output_dir / "results.json"
    print(f"\nwrote {results_path}")

    print(f"\n{len(measured_rows)} of {len(rows)} measured")
    for row in rows:
        shown = (
            f"{row['reliability']:+.4f} [{row['reliability_low']:+.4f}, "
            f"{row['reliability_high']:+.4f}]"
            if row["reliability"] is not None
            else "        n/a         "
        )
        print(
            f"  {row['entry']:<58} {row['method_tag']:<8} n={row['n_units']:>4} "
            f"{shown}  {row['status']}"
        )

    print("\n=== set-reliability commands (run each through the lock) ===")
    for row in measured_rows:
        print(
            "nfl-ats weak-signals set-reliability "
            f"--name {row['entry']} "
            f"--reliability {row['reliability']:.6f} "
            f"--reliability-low {row['reliability_low']:.6f} "
            f"--reliability-high {row['reliability_high']:.6f} "
            f'--method "{row["method"]}" '
            f'--source "{results_path}" '
            f'--reason "{row["parent_quantity"]}"'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
