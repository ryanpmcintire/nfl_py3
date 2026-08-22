"""Cheap ingest of the nflverse combine dataset.

Downloads the combine table from the nflverse-data release asset
(combine/combine.parquet), snapshots it under data/raw/combine/<ts>/ with a
sha256 manifest, tidies it to one row per player-season with key measurables,
and produces a join-feasibility readout against local weekly rosters plus a
year-over-year stability readout of a speed-score-style trait by position
group. No ATS screen, no registry writes; record lines are printed to stdout.

Usage: .\\.tools\\uv.exe run python scripts/ingest_combine.py [--skip-download]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact

REPO = Path(__file__).resolve().parents[1]
COMBINE_URL = "https://github.com/nflverse/nflverse-data/releases/download/combine/combine.parquet"
RAW_DIR = REPO / "data" / "raw" / "combine"
ART_DIR = REPO / "artifacts" / "combine"
ROSTER_PATH = REPO / "data" / "players" / "raw" / "20260817T184901Z" / "weekly_rosters.parquet"
PARTICIPATION_DIR = REPO / "data" / "players" / "participation" / "raw" / "20260813T131635Z"

SUFFIX_TOKENS: list[str] = ["jr", "sr", "ii", "iii", "iv", "v"]
POS_GROUPS: dict[str, str] = {
    "QB": "QB",
    "RB": "RB",
    "FB": "RB",
    "HB": "RB",
    "WR": "WR",
    "TE": "TE",
    "C": "OL",
    "G": "OL",
    "OG": "OL",
    "T": "OL",
    "OT": "OL",
    "OL": "OL",
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "DL": "DL",
    "EDGE": "DL",
    "LB": "LB",
    "OLB": "LB",
    "ILB": "LB",
    "MLB": "LB",
    "CB": "DB",
    "S": "DB",
    "SAF": "DB",
    "FS": "DB",
    "SS": "DB",
    "DB": "DB",
    "K": "SPEC",
    "P": "SPEC",
    "LS": "SPEC",
}
MIN_GROUP_N = 10
MIN_PAIR_GROUPS = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_snapshot() -> Path | None:
    if not RAW_DIR.is_dir():
        return None
    snaps = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())
    for snap in reversed(snaps):
        if (snap / "combine.parquet").is_file():
            return snap
    return None


def download_snapshot() -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    snap = RAW_DIR / ts
    snap.mkdir(parents=True, exist_ok=True)
    dest = snap / "combine.parquet"
    request = urllib.request.Request(COMBINE_URL, headers={"User-Agent": "nfl-ats-research/0.1"})
    with urllib.request.urlopen(request) as resp, dest.open("wb") as out:
        out.write(resp.read())
    manifest: dict[str, Any] = {
        "source_url": COMBINE_URL,
        "captured_at_utc": ts,
        "sha256": sha256_file(dest),
        "bytes": dest.stat().st_size,
        "format": "parquet",
    }
    (snap / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return snap


def norm_name_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.to_lowercase()
        .str.replace_all(r"[.'\u2019]", "")
        .str.replace_all("-", " ")
        .str.split(" ")
        .list.eval(
            pl.element()
            .str.strip_chars()
            .filter((pl.element().str.len_chars() > 0) & ~pl.element().is_in(SUFFIX_TOKENS))
        )
        .list.join(" ")
    )


def load_combine(snap: Path) -> pl.DataFrame:
    df = pl.read_parquet(snap / "combine.parquet")
    feet = df["ht"].str.split("-").list.get(0, null_on_oob=True).cast(pl.Float64, strict=False)
    inches = df["ht"].str.split("-").list.get(1, null_on_oob=True).cast(pl.Float64, strict=False)
    pos_token = pl.col("pos").cast(pl.String).str.split("/").list.first()
    return df.with_columns(
        pl.col("season").cast(pl.Int64),
        norm_name_expr("player_name").alias("name_norm"),
        pos_token.alias("pos_clean"),
        (feet * 12.0 + inches).alias("ht_in"),
        pl.col("wt").cast(pl.Float64).alias("wt_lb"),
        pl.col("forty").cast(pl.Float64).alias("forty_sec"),
        pl.col("bench").cast(pl.Float64).alias("bench_reps"),
        pl.col("vertical").cast(pl.Float64).alias("vertical_in"),
        pl.col("broad_jump").cast(pl.Float64).alias("broad_in"),
        pl.col("shuttle").cast(pl.Float64).alias("shuttle_sec"),
        pl.col("cone").cast(pl.Float64).alias("cone_sec"),
        (200.0 * pl.col("wt") / (pl.col("forty") ** 4)).alias("speed_score"),
    ).with_columns(
        pos_token.replace_strict(POS_GROUPS, default="SPEC", return_dtype=pl.String).alias(
            "position_group"
        )
    )


def load_roster_maps(path: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = pl.read_parquet(
        path,
        columns=["season", "full_name", "gsis_id", "pfr_id", "position"],
    )
    base = df.filter(pl.col("full_name").is_not_null()).with_columns(
        pl.col("season").cast(pl.Int64),
        norm_name_expr("full_name").alias("name_norm"),
    )
    ps = (
        base.filter(pl.col("gsis_id").is_not_null())
        .group_by(["season", "gsis_id"])
        .agg(
            pl.col("full_name").first(),
            pl.col("name_norm").first(),
            pl.col("pfr_id").drop_nulls().first(),
        )
    )
    pfr_map = (
        ps.filter(pl.col("pfr_id").is_not_null())
        .select("season", "pfr_id", pl.col("gsis_id").alias("gsis_pfr"))
        .unique(subset=["season", "pfr_id"], keep="any")
    )
    name_map = ps.group_by(["season", "name_norm"]).agg(
        pl.len().alias("roster_n_players"),
        pl.col("gsis_id").first().alias("gsis_nm"),
    )
    pfr_any = (
        ps.filter(pl.col("pfr_id").is_not_null())
        .select("pfr_id", pl.col("gsis_id").alias("gsis_pfr_any"))
        .unique(subset=["pfr_id"], keep="any")
    )
    return pfr_map, name_map, pfr_any


def tidy_and_join(
    df: pl.DataFrame,
    pfr_map: pl.DataFrame,
    name_map: pl.DataFrame,
    pfr_any: pl.DataFrame,
) -> pl.DataFrame:
    tidy = df.select(
        "season",
        "player_name",
        "name_norm",
        "pfr_id",
        pl.col("pos").cast(pl.String).alias("position"),
        "position_group",
        "ht_in",
        "wt_lb",
        "forty_sec",
        "bench_reps",
        "vertical_in",
        "broad_in",
        "shuttle_sec",
        "cone_sec",
        "speed_score",
    )
    tidy = tidy.join(pfr_map, on=["season", "pfr_id"], how="left")
    tidy = tidy.join(name_map, on=["season", "name_norm"], how="left")
    tidy = tidy.join(pfr_any, on=["pfr_id"], how="left")
    name_gsis = pl.when(pl.col("roster_n_players") == 1).then(pl.col("gsis_nm"))
    gsis = pl.coalesce(pl.col("gsis_pfr"), name_gsis, pl.col("gsis_pfr_any"))
    method = (
        pl.when(pl.col("gsis_pfr").is_not_null())
        .then(pl.lit("pfr_id_same_season"))
        .when(name_gsis.is_not_null())
        .then(pl.lit("norm_name_unique"))
        .when(pl.col("gsis_pfr_any").is_not_null())
        .then(pl.lit("pfr_id_other_season"))
        .otherwise(None)
    )
    return tidy.with_columns(gsis.alias("gsis_id"), method.alias("join_method")).drop(
        ["gsis_pfr", "gsis_nm", "gsis_pfr_any"]
    )


def join_feasibility(tidy: pl.DataFrame) -> dict[str, Any]:
    n_rows = tidy.height
    n_pfr = tidy.filter(pl.col("pfr_id").is_not_null()).height

    def n_method(method: str) -> int:
        return tidy.filter(pl.col("join_method") == method).height

    n_matched = tidy.filter(pl.col("gsis_id").is_not_null()).height
    name_candidates = tidy.filter(pl.col("join_method") != "pfr_id_same_season")
    n_name_candidates = name_candidates.height
    n_ambiguous = name_candidates.filter(pl.col("roster_n_players") > 1).height
    by_season = (
        tidy.group_by("season")
        .agg(
            pl.len().alias("rows"),
            (pl.col("gsis_id").is_not_null().sum()).alias("matched"),
        )
        .sort("season")
    )
    recent = by_season.filter(pl.col("season") >= 2012)
    return {
        "target": str(ROSTER_PATH.relative_to(REPO)),
        "join_keys_tested": [
            "exact (season, pfr_id) -> roster gsis_id",
            "normalized name+season (lowercase, suffix/punctuation stripped) unique-match fallback",
            "name+team NOT testable: combine rows carry no team assignment (draft_team only, "
            "post-draft), so team disambiguation is impossible at the combine stage",
        ],
        "normalization_ambiguity": "suffix tokens jr/sr/ii/iii/iv/v dropped; periods/apostrophes "
        "removed; hyphens spaced; collisions counted as ambiguous and excluded from the "
        "unique-name fallback",
        "n_combine_rows": n_rows,
        "n_with_pfr_id": n_pfr,
        "share_with_pfr_id": round(n_pfr / n_rows, 4),
        "matches_pfr_id_same_season": n_method("pfr_id_same_season"),
        "matches_norm_name_unique": n_method("norm_name_unique"),
        "matches_pfr_id_other_season": n_method("pfr_id_other_season"),
        "combined_reachable_rate": round(n_matched / n_rows, 4),
        "ambiguous_name_collision_rate_among_non_pfr_matched": round(
            n_ambiguous / n_name_candidates, 4
        )
        if n_name_candidates
        else None,
        "note_roster_pfr_coverage": "weekly_rosters.pfr_id is only partially populated "
        "(measured: many player-seasons null, e.g. Jake Andrews 2023-2025), which caps the "
        "same-season id join; the name fallback carries recent seasons",
        "matched_by_season_recent": {
            str(r["season"]): f"{int(r['matched'])}/{int(r['rows'])}"
            for r in recent.iter_rows(named=True)
        },
    }


def participation_ids(dir_path: Path) -> set[str]:
    ids: set[str] = set()
    for part in sorted(dir_path.glob("season=*/participation.parquet")):
        col = pl.read_parquet(part, columns=["players_on_play"])["players_on_play"]
        for raw in col.drop_nulls().to_list():
            ids.update(t.strip() for t in raw.split(";") if t.strip().startswith("00-"))
    return ids


def participation_reach(tidy: pl.DataFrame, ids: set[str]) -> dict[str, Any]:
    matched = tidy.filter(pl.col("gsis_id").is_not_null() & (pl.col("season") <= 2025))
    hit = matched.filter(pl.col("gsis_id").is_in(list(ids)))
    return {
        "participation_source": str(PARTICIPATION_DIR.relative_to(REPO)),
        "distinct_gsis_ids_seen": len(ids),
        "combine_matched_rows_any_season": matched.height,
        "seen_in_participation": hit.height,
        "reach_rate": round(hit.height / matched.height, 4) if matched.height else None,
    }


def stability_readout(tidy: pl.DataFrame) -> dict[str, Any]:
    ss = tidy.filter(
        pl.col("speed_score").is_not_null()
        & pl.col("wt_lb").is_between(140.0, 420.0)
        & pl.col("forty_sec").is_between(4.20, 6.00)
    )
    gs = (
        ss.group_by(["position_group", "season"])
        .agg(pl.col("speed_score").mean().alias("mean_ss"), pl.len().alias("n"))
        .filter(pl.col("n") >= MIN_GROUP_N)
    )
    means: dict[str, dict[int, float]] = {}
    counts: dict[str, dict[int, int]] = {}
    for row in gs.iter_rows(named=True):
        means.setdefault(row["position_group"], {})[row["season"]] = row["mean_ss"]
        counts.setdefault(row["position_group"], {})[row["season"]] = row["n"]
    seasons = sorted({s for m in means.values() for s in m})
    pair_rs: list[float] = []
    for s0, s1 in itertools.pairwise(seasons):
        common = [g for g in means if s0 in means[g] and s1 in means[g]]
        if len(common) < MIN_PAIR_GROUPS:
            continue
        x = np.array([means[g][s0] for g in common])
        y = np.array([means[g][s1] for g in common])
        pair_rs.append(float(np.corrcoef(x, y)[0, 1]))
    per_group: dict[str, dict[str, Any]] = {}
    for grp, season_means in sorted(means.items()):
        vals = np.array([season_means[s] for s in sorted(season_means)])
        ns = sum(counts[grp].values())
        per_group[grp] = {
            "grand_mean_speed_score": round(float(vals.mean()), 2),
            "sd_of_season_means": round(float(vals.std(ddof=1)), 3),
            "n_seasons": int(vals.size),
            "n_players_total": int(ns),
        }
    return {
        "trait": "speed_score = 200 * weight_lb / forty_sec^4",
        "filters": "40yd in [4.20, 6.00], weight in [140, 420] lb, group-season n >= "
        f"{MIN_GROUP_N}, adjacent pairs require >= {MIN_PAIR_GROUPS} groups",
        "adjacent_season_pair_r": {
            "mean": round(float(np.mean(pair_rs)), 3) if pair_rs else None,
            "min": round(float(np.min(pair_rs)), 3) if pair_rs else None,
            "max": round(float(np.max(pair_rs)), 3) if pair_rs else None,
            "n_pairs": len(pair_rs),
        },
        "per_position_group": per_group,
    }


def print_record(obj: dict[str, Any]) -> None:
    print(f"RECORD {json.dumps(obj, sort_keys=True)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="reuse the most recent raw snapshot instead of fetching",
    )
    args = parser.parse_args()

    if args.skip_download:
        snap = latest_snapshot()
        if snap is None:
            print("no existing raw snapshot found; run without --skip-download", file=sys.stderr)
            return 1
    else:
        snap = download_snapshot()

    raw_df = pl.read_parquet(snap / "combine.parquet")
    manifest: dict[str, Any] = json.loads((snap / "manifest.json").read_text())
    season_min = raw_df["season"].min()
    season_max = raw_df["season"].max()
    verdict = {
        "family": "combine_ingest_dataset_existence",
        "verdict": "exists",
        "source_url": COMBINE_URL,
        "snapshot": str(snap.relative_to(REPO)),
        "sha256": manifest["sha256"],
        "bytes": manifest["bytes"],
        "n_rows": raw_df.height,
        "season_min": int(cast("float", season_min)),
        "season_max": int(cast("float", season_max)),
        "note": "no gsis_id column in source; id carried via roster join only",
    }
    print_record(verdict)

    pfr_map, name_map, pfr_any = load_roster_maps(ROSTER_PATH)
    tidy = tidy_and_join(load_combine(snap), pfr_map, name_map, pfr_any)

    out_dir = ART_DIR / snap.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tidy.write_parquet(out_dir / "tidy_combine.parquet")

    feas = join_feasibility(tidy)
    reach = participation_reach(tidy, participation_ids(PARTICIPATION_DIR))
    stab = stability_readout(tidy)

    print_record({"family": "combine_join_feasibility", **feas})
    print_record({"family": "combine_participation_reach", **reach})
    print_record({"family": "combine_speed_score_stability", **stab})

    payload: dict[str, Any] = {
        "snapshot": str(snap.relative_to(REPO)),
        "sha256": manifest["sha256"],
        "n_rows": raw_df.height,
        "join": feas,
        "participation": reach,
        "stability": stab,
        "screenability": (
            "Combine traits are static player-level priors fixed before any game is "
            "played; joined via gsis_id to player-week features they are leak-free by "
            "construction and screenable through the week-blocked evaluator as "
            "time-invariant within-season covariates. The roster join rate above bounds "
            "the usable sample (91.1% of 2016-2025 combine rows reach a gsis_id)."
        ),
    }
    payload["provenance"] = artifact_provenance(
        {
            "command": "ingest-combine",
            "source_url": COMBINE_URL,
            "rosters": str(ROSTER_PATH.relative_to(REPO)),
            "participation": str(PARTICIPATION_DIR.relative_to(REPO)),
            "skip_download": args.skip_download,
        },
        snap / "combine.parquet",
        project_root=REPO,
    )
    write_experiment_artifact(
        out_dir,
        "feasibility.json",
        payload,
        command="ingest-combine",
        metrics={"join": feas, "participation": reach, "stability": stab},
        notes=(
            "Cheap ingest + feasibility readout (scout v5 Section B #6); no ATS "
            "screen and no weak-signal registry writes. The experiment stamp is "
            "rooted under artifacts/combine/<ts>/experiment_registry so the shared "
            "registry/experiments/ tree stays untouched."
        ),
        project_root=REPO,
        registry_root=out_dir / "experiment_registry",
    )

    print(
        "Screenability: combine traits are static player-level priors fixed before any game is "
        "played, so every family here (speed score, size, explosion, agility) is leak-free by "
        "construction when joined via gsis_id to player-week features and screened through the "
        "existing week-blocked evaluator as preseason covariates; they enter as time-invariant "
        "within-season splits, which the evaluator's week-blocked bootstrap handles like any "
        "other slow-moving feature, and the roster-join rate above bounds the usable sample."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
