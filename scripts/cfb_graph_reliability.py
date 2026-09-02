"""Split-half reliability of the three CFB graph ``team_stat`` traits (WP24).

WP8 recorded three CFB graph cells with a null ``reliability`` field, and
AGENTS.md makes that field decisive: zero split-half reliability is one of only
two admissible grounds for closing a line of work, and a signal recorded
without it cannot be adjudicated later. This script measures it. It touches no
outcome at all -- no spread, no margin, no accuracy -- so it runs BEFORE the
replacement screen's null, positive control and screen modes.

Predeclared in ``docs/graph_team_stat_cfb_replacement.md`` section 6.

**Method, mirrored from ``scripts/reliability_map.py`` and stated so the mirror
is checkable.** That script reshapes a game-level table carrying
``home_<x>``/``away_<x>`` pairs into TEAM-WEEK long form -- each game
contributes exactly two rows, one per side, carrying ``team_id``, ``season``,
``week`` and the metric value -- and then calls
``nfl_ats.cfb_qb_dependence.split_half_reliability`` (imported, never
reimplemented) on every metric. That function splits each team-season by
odd/even week, correlates the two halves' team-season MEANS, applies the
Spearman-Brown full-length correction ``2r/(1+r)``, and block-bootstraps over
team-seasons for a 95% CI and ``probability_positive``, requiring >=2
observations in each half. This script makes the identical call with
``n_boot=4000``.

Two families per cell, because a reliability number alone is not interpretable:

* ``graph_<cell>_katz`` -- the graph trait, from the home/away katz pair.
* ``raw_<cell>``        -- the untransformed statistic, as a REFERENCE on the
  same population.

WP8's public builder attaches only the katz DIFFERENTIAL, so the pair is taken
from the underlying ``add_graph_ratings_v2_features`` call using WP8's own
``cfb_graph_config``. A consistency assertion re-derives WP8's differential from
that pair and requires an exact match, so the two paths cannot silently
diverge; ``--skip-consistency-check`` exists only to halve the build cost on a
re-run and is never used for the recorded numbers.

Closing-grounds taxonomy (binding, AGENTS.md, restated verbatim so this file
stands on its own): an interval or CI that contains zero is NEVER grounds to
reject, fail, or close an experiment. At this evaluator's ~2-point resolution,
"contains zero" is the EXPECTED outcome for a real small signal. Only two
grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong
sign (whole interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that size.
Everything else is ``unresolved_below_power``: record it with ``nfl-ats
weak-signals record``, report ``probability_positive``, never the binary
"contains zero". A reliability point estimate near zero whose interval contains
zero is NOT the ``no_split_half_reliability`` ground; only a resolved-at-or-
below-zero reliability on the trait itself is. If a record command errors, the
verdict is wrong, not the validator.

This script never writes to ``registry/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nfl_ats.cfb_benchmark import CFB_CLEAN_CORE_SEASONS  # noqa: E402
from nfl_ats.cfb_qb_dependence import split_half_reliability  # noqa: E402
from nfl_ats.graph_ratings_v2 import (  # noqa: E402
    add_graph_ratings_v2_features,
    katz_feature_columns,
)
from nfl_ats.graph_team_stat_cfb_feature import (  # noqa: E402
    CFB_AWAY_ID_COLUMN,
    CFB_GRAPH_CELLS,
    CFB_GRAPH_FROZEN_STRUCTURE,
    CFB_HOME_ID_COLUMN,
    add_cfb_graph_team_stat_feature,
    cfb_graph_column,
    cfb_graph_config,
)
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from scripts.graph_team_stat_cfb_replication import (  # noqa: E402
    FEATURES_PATH,
    SEED,
    load_cfb_table,
)

N_BOOT = 4_000
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "graph_team_stat_cfb_replacement"
PREDECLARATION = "docs/graph_team_stat_cfb_replacement.md"


def graph_rating_pair(cell: str) -> tuple[str, str]:
    """The ``(home, away)`` katz RATING columns behind WP8's differential."""

    home_column, away_column, _diff = katz_feature_columns(cfb_graph_config(cell))
    return home_column, away_column


def add_graph_rating_pair(games: pd.DataFrame, cell: str) -> pd.DataFrame:
    """Attach the home/away katz rating pair for one cell.

    Identical construction to
    ``nfl_ats.graph_team_stat_cfb_feature.add_cfb_graph_team_stat_feature``
    (same frozen config, same adaptation-A1 id substitution, same join back by
    ``game_id``), differing only in WHICH of the builder's output columns are
    kept: WP8 keeps the differential, this keeps the two ratings it is formed
    from. The caller's frame is never mutated.
    """

    config = cfb_graph_config(cell)
    home_column, away_column = graph_rating_pair(cell)

    graph_input = games.copy()
    graph_input["home_team"] = (
        pd.to_numeric(graph_input[CFB_HOME_ID_COLUMN], errors="raise").astype("int64").astype(str)
    )
    graph_input["away_team"] = (
        pd.to_numeric(graph_input[CFB_AWAY_ID_COLUMN], errors="raise").astype("int64").astype(str)
    )
    rated = add_graph_ratings_v2_features(graph_input, config)

    result = games.copy()
    for column in (home_column, away_column):
        values = pd.Series(
            pd.to_numeric(rated[column], errors="coerce").to_numpy(),
            index=rated["game_id"].to_numpy(),
        )
        result[column] = games["game_id"].map(values).astype(float)
    return result


def long_frame(games: pd.DataFrame, families: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Team-week long form: one row per (game, side).

    ``reliability_map.build_long_frame``'s reshape, with two differences, both
    forced by the data rather than chosen: the team key is the ESPN id
    (``home_id``/``away_id``, WP8 adaptation A1 -- a rebranded program must stay
    one node) rather than a name string, and there is no ``game_type == "REG"``
    filter because ``nfl_ats.cfb_features`` already restricts this table to
    regular-season FBS-vs-FBS games, so the filter has nothing to remove.
    """

    pieces: list[pd.DataFrame] = []
    for side, id_column in (("home", CFB_HOME_ID_COLUMN), ("away", CFB_AWAY_ID_COLUMN)):
        piece = games[["game_id", "season", "week", id_column]].rename(
            columns={id_column: "team_id"}
        )
        for metric, (home_column, away_column) in families.items():
            source = home_column if side == "home" else away_column
            piece[metric] = pd.to_numeric(games[source], errors="coerce")
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def measure_cell(
    games: pd.DataFrame, cell: str, *, check_consistency: bool
) -> dict[str, dict[str, Any]]:
    """Both reliability families for one cell, on the clean core."""

    widened = add_graph_rating_pair(games, cell)
    home_katz, away_katz = graph_rating_pair(cell)
    diff_column = cfb_graph_column(cell)

    consistency: dict[str, Any] = {"checked": False}
    if check_consistency:
        wp8 = add_cfb_graph_team_stat_feature(games, cell)
        derived = widened[home_katz] - widened[away_katz]
        gap = (derived - wp8[diff_column]).abs()
        consistency = {
            "checked": True,
            "max_abs_gap_vs_wp8_differential": float(gap.max(skipna=True)),
            "n_compared": int(gap.notna().sum()),
        }
        if not (gap.fillna(0.0) <= 1e-12).all():
            raise SystemExit(
                f"graph rating pair for {cell} does not reproduce WP8's differential "
                f"(max gap {gap.max(skipna=True)}); the two builders have diverged"
            )

    core = widened.loc[widened["season"].isin(CFB_CLEAN_CORE_SEASONS)].copy()
    families = {
        f"graph_{cell}_katz": (home_katz, away_katz),
        f"raw_{cell}": (f"home_{cell}", f"away_{cell}"),
    }
    long = long_frame(core, families)

    measured: dict[str, dict[str, Any]] = {}
    for metric in families:
        result = split_half_reliability(long, metric, seed=SEED, n_boot=N_BOOT)
        result["home_column"], result["away_column"] = families[metric]
        result["n_rows_non_null"] = int(long[metric].notna().sum())
        measured[metric] = result
    measured["_consistency"] = consistency
    return measured


def _print_result(result: dict[str, Any]) -> None:
    low, high = result["pearson_r_ci95"]
    print(
        f"  {result['metric']:<44} SB {result['spearman_brown_full_length_reliability']:+.4f}  "
        f"r {result['pearson_r']:+.4f}  rho {result['spearman_rho']:+.4f}  "
        f"95% [{low:+.4f}, {high:+.4f}]  P+ {result['probability_positive']:.3f}  "
        f"team-seasons {result['n_team_seasons']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        choices=CFB_GRAPH_CELLS,
        action="append",
        help="restrict to one cell (repeatable); default is all three",
    )
    parser.add_argument(
        "--skip-consistency-check",
        action="store_true",
        help="skip re-deriving WP8's differential from the rating pair (halves build cost)",
    )
    args = parser.parse_args()
    cells = tuple(args.cell) if args.cell else CFB_GRAPH_CELLS

    started = time.time()
    print("=== CFB graph team_stat split-half reliability (WP24) ===")
    games = load_cfb_table()
    print(
        f"CFB table: {len(games)} rows, seasons "
        f"{int(games['season'].min())}-{int(games['season'].max())}; scoring the clean core "
        f"({len(games.loc[games['season'].isin(CFB_CLEAN_CORE_SEASONS)])} rows)"
    )

    results: dict[str, Any] = {}
    for cell in cells:
        cell_started = time.time()
        print(f"\n--- cell {cell} ---")
        measured = measure_cell(games, cell, check_consistency=not args.skip_consistency_check)
        consistency = measured.pop("_consistency")
        if consistency["checked"]:
            print(
                "  consistency vs WP8 differential: max abs gap "
                f"{consistency['max_abs_gap_vs_wp8_differential']:.3e} over "
                f"{consistency['n_compared']} rows"
            )
        for metric in sorted(measured):
            _print_result(measured[metric])
        results[cell] = {"reliability": measured, "consistency": consistency}
        print(f"  ({time.time() - cell_started:.1f}s)")

    configuration = {
        "command": "cfb-graph-reliability",
        "league": "cfb",
        "cells": list(cells),
        "population": "CFB_CLEAN_CORE_SEASONS (2012-2019 + 2021-2025)",
        "method": (
            "odd/even-week team-season split-half via "
            "nfl_ats.cfb_qb_dependence.split_half_reliability, the same call "
            "scripts/reliability_map.py makes; Spearman-Brown full-length corrected; "
            "block bootstrap over team-seasons"
        ),
        "team_key": "home_id/away_id (WP8 adaptation A1)",
        "frozen_structure": CFB_GRAPH_FROZEN_STRUCTURE,
        "seed": SEED,
        "n_boot": N_BOOT,
        "predeclaration": PREDECLARATION,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        **configuration,
        "results": results,
        "provenance": artifact_provenance(configuration, FEATURES_PATH, project_root=REPO_ROOT),
    }
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = ARTIFACT_ROOT / "reliability" / timestamp
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="cfb-graph-reliability",
        metrics={"n_cells": len(cells), "n_boot": N_BOOT},
        notes=(
            "Split-half reliability of the three CFB graph team_stat traits and their raw "
            "references; measure-only, no outcome touched. See "
            "docs/graph_team_stat_cfb_replacement.md section 6."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")
    print(f"done in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
