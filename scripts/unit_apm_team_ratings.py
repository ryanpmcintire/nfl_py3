"""PER-14 annual team-unit priors, using the unchanged PER-09 fit recipe.

Historical reconstruction: finalized_at is a conservative March 1 availability
boundary after the source season, not a claim of a contemporaneous archive.
Only locally available regular-season participation/PBP is used. Reliability
outputs from unit_apm_screen.py are never changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from unit_apm_screen import (  # noqa: E402
    UNIT_SIDE,
    UNITS,
    UnitLookup,
    build_unit_lookup,
    fit_unit_coefficients,
    modal_unit_by_player_season,
)

from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.io import run_id  # noqa: E402
from nfl_ats.participation import (  # noqa: E402
    _player_ids,
    build_participation_play_table,
    latest_participation_snapshot,
)
from nfl_ats.pbp import latest_pbp_snapshot  # noqa: E402
from nfl_ats.provenance import sha256_file, stamp_sidecar, write_stamped_artifact  # noqa: E402


def aggregate_team_unit(
    table: pd.DataFrame, unit: str, lookup: UnitLookup, coefficients: dict[str, float]
) -> pd.DataFrame:
    """Snap-weighted mean of player effects, excluding the nuisance team effect.

    A traded player's snaps belong to the team on each actual play. Players is
    the count of distinct mapped members; snaps counts player-plays, not plays.
    """
    counts: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    side = UNIT_SIDE[unit]
    for row in table.itertuples(index=False):
        team = str(row.posteam if side == "offense" else row.defteam)
        season = int(row.season)
        counts[season, team].update(
            p
            for p in _player_ids(getattr(row, f"{side}_players"))
            if lookup.get((p, season)) == unit
        )
    rows = []
    for (season, team), members in sorted(counts.items()):
        snaps = sum(members.values())
        if not snaps:
            continue
        rating = sum(coefficients[f"unit_player::{p}"] * n for p, n in members.items()) / snaps
        rows.append(
            {
                "season": season,
                "team": team,
                "unit": unit,
                "rating": rating,
                "players": len(members),
                "snaps": snaps,
                "members": ";".join(sorted(members)),
                "finalized_at": pd.Timestamp(f"{season + 1}-03-01", tz="UTC"),
            }
        )
    return pd.DataFrame(rows)


def correlation_summary(frame: pd.DataFrame, x: str, y: str) -> dict:
    """Pearson and 95% team-cluster bootstrap, retaining successive-year dependence."""
    frame = frame.dropna(subset=[x, y]).copy()
    a, b = frame[x].to_numpy(float), frame[y].to_numpy(float)
    if len(frame) < 3 or not a.std() or not b.std():
        return {"pairs": len(frame), "pearson": None, "ci95": None}
    r = float(np.corrcoef(a, b)[0, 1])
    # Sufficient statistics make 20,000 cluster draws inexpensive.
    stats = (
        pd.DataFrame(
            {
                "team": frame.team.to_numpy(),
                "n": 1,
                "a": a,
                "b": b,
                "aa": a * a,
                "bb": b * b,
                "ab": a * b,
            }
        )
        .groupby("team")
        .sum()
        .to_numpy(float)
    )
    rng = np.random.default_rng(20260902)
    sums = stats[rng.integers(0, len(stats), size=(20_000, len(stats)))].sum(axis=1)
    n, sa, sb, saa, sbb, sab = sums.T
    denominator = np.sqrt(np.maximum((saa - sa * sa / n) * (sbb - sb * sb / n), 0))
    draws = np.divide(
        sab - sa * sb / n, denominator, out=np.full(len(n), np.nan), where=denominator > 0
    )
    draws = draws[np.isfinite(draws)]
    return {
        "pairs": len(frame),
        "teams": len(stats),
        "pearson": r,
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "probability_positive": float((draws > 0).mean()),
        "spearman_brown": 2 * r / (1 + r),
        "bootstrap": "team clusters, 20000 draws, seed 20260902",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / "artifacts/unit_apm" / run_id()
    output.mkdir(parents=True, exist_ok=False)
    participation = latest_participation_snapshot(ROOT / "data/players/participation/raw")
    pbp_snapshot = latest_pbp_snapshot(ROOT / "data/pbp/raw")
    roster_path = ROOT / "data/players/raw/20260817T184901Z/weekly_rosters.parquet"
    units, unmapped = modal_unit_by_player_season(pd.read_parquet(roster_path))
    lookup = build_unit_lookup(units)
    seasons = sorted(
        set(range(2013, 2026)) & set(participation.seasons) & set(pbp_snapshot.seasons)
    )
    frames, sources = [], []
    for season in seasons:
        part_path, pbp_path = participation.season_path(season), pbp_snapshot.season_path(season)
        pbp = pd.read_parquet(pbp_path)
        pbp = pbp.loc[pbp.season_type.eq("REG")].copy()
        table = build_participation_play_table(pd.read_parquet(part_path), pbp)
        table = table.merge(
            pbp[["season", "game_id", "play_id", "week"]],
            on=["season", "game_id", "play_id"],
            validate="one_to_one",
        )
        if table.week.isna().any():
            raise DataContractError("Missing week for annual split fits")
        for unit in UNITS:
            coef, _ = fit_unit_coefficients(table, unit, lookup)
            annual = aggregate_team_unit(table, unit, lookup, coef)
            for label, parity in (("odd", 1), ("even", 0)):
                half = table.loc[table.week.astype(int).mod(2).eq(parity)]
                coef, _ = fit_unit_coefficients(half, unit, lookup)
                ratings = aggregate_team_unit(half, unit, lookup, coef)
                annual = annual.merge(
                    ratings[["season", "team", "unit", "rating"]].rename(
                        columns={"rating": f"rating_{label}"}
                    ),
                    on=["season", "team", "unit"],
                    how="left",
                    validate="one_to_one",
                )
            frames.append(annual)
        sources.append(
            {
                "season": season,
                "plays": len(table),
                "participation_sha256": sha256_file(part_path),
                "pbp_sha256": sha256_file(pbp_path),
            }
        )
        print(
            f"season={season} plays={len(table)} completed all four annual/odd/even fits",
            flush=True,
        )
    ratings = pd.concat(frames, ignore_index=True)
    if ratings.duplicated(["season", "team", "unit"]).any():
        raise DataContractError("Duplicate annual team-unit rating")
    path = output / "team_unit_ratings.parquet"
    ratings.to_parquet(path, index=False)
    reliability = {}
    for unit, frame in ratings.groupby("unit"):
        following = frame[["season", "team", "rating"]].copy()
        following["season"] -= 1
        pairs = frame.merge(following, on=["season", "team"], suffixes=("", "_next"))
        reliability[unit] = {
            "between_season": correlation_summary(pairs, "rating", "rating_next"),
            "odd_even": correlation_summary(frame, "rating_odd", "rating_even"),
        }
    payload = {
        "seasons": seasons,
        "missing_requested_seasons": sorted(set(range(2013, 2026)) - set(seasons)),
        "rows": len(ratings),
        "reliability": reliability,
        "sources": sources,
        "rosters_sha256": sha256_file(roster_path),
        "unmapped_positions": unmapped,
        "recipe": {"ridge_alpha": 1000, "team_feature_scale": 11, "epa_clip": 5.0},
        "rating": "player-effect snap-weighted mean; team nuisance coefficient excluded",
        "availability": (
            "historical reconstruction; March 1 after source season; not archived vintage"
        ),
        "continuity": (
            "members are realized annual membership; weekly roster has no publication timestamp"
        ),
        "ratings_sha256": sha256_file(path),
    }
    stamp_sidecar(path, payload, project_root=ROOT)
    write_stamped_artifact(payload, output / "annual_summary.json", project_root=ROOT)
    print(json.dumps(payload, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
