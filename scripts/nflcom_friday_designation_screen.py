"""Predeclared Friday-designation screen on NFL.com final injury reports.

Implements exactly the frozen predeclaration in
``docs/nflcom_friday_designation_screen.md`` (cells, starter proxy, and
negative direction were written down before this script first scored
anything). Population: REG 2022-2024 team-games from the newest schedules
snapshot. Flags come ONLY from the immutable snapshot of the league's final
weekly injury pages (data/raw/nflcom_injuries/<ts>/), each page being that
week's Friday/Saturday report that fully predates kickoff, plus prior-week
snap shares for the disclosed starter proxy.

Measure-only: never writes registry/weak_signals.json; recording happens via
separate explicit ``nfl-ats weak-signals record`` calls against this script's
results.json. Every cell is predeclared to record ``unresolved_below_power``
regardless of interval shape (mined family, uncorrected multiplicity); per
AGENTS.md an interval crossing zero is not a rejection.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260821
SEASON_START = 2022
SEASON_END = 2024
QA_OR_WORSE = frozenset({"questionable", "doubtful", "out"})
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
STARTER_SNAP_SHARE = 0.50


def normalize_name(name: object) -> str:
    if not isinstance(name, str):
        return ""
    lowered = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    lowered = lowered.casefold().replace("'", "").replace(".", " ")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return " ".join(tok for tok in lowered.split() if tok not in SUFFIXES)


def initial_last_key(name: str) -> tuple[str, str]:
    tokens = normalize_name(name).split()
    if not tokens:
        return ("", "")
    return (tokens[0][0], tokens[-1] if len(tokens) > 1 else "")


def latest(root: Path, pattern: str) -> Path:
    candidates = sorted(root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no match for {pattern} under {root}")
    return candidates[-1]


def load_population(schedules_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(
        schedules_path,
        columns=[
            "game_id",
            "season",
            "week",
            "game_type",
            "gameday",
            "home_team",
            "away_team",
            "result",
            "spread_line",
        ],
    )
    df = raw.loc[raw["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].copy()
    n_reg_in_window = len(df)
    df = add_ats_outcomes(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)
    df["week_block"] = df["season"] * 100 + df["week"]

    home_side = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "season": df["season"],
            "week": df["week"],
            "week_block": df["week_block"],
            "team": df["home_team"],
            "gameday": pd.to_datetime(df["gameday"]),
            "team_cover": df["home_cover"],
        }
    )
    away_side = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "season": df["season"],
            "week": df["week"],
            "week_block": df["week_block"],
            "team": df["away_team"],
            "gameday": pd.to_datetime(df["gameday"]),
            "team_cover": 1.0 - df["home_cover"],
        }
    )
    long = pd.concat([home_side, away_side], ignore_index=True)
    long.attrs["n_games"] = len(df)
    long.attrs["n_pushes_or_missing_dropped"] = n_reg_in_window - len(df)
    return long


def load_report_flags(injuries_root: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    parsed = pd.read_parquet(latest(injuries_root, "*/injuries.parquet")).copy()
    counts = {"rows_total": len(parsed)}
    parsed["norm_name"] = parsed["player"].map(normalize_name)
    parsed["status_norm"] = parsed["game_status"].astype("string").str.strip().str.casefold()
    qa = parsed.loc[parsed["status_norm"].isin(QA_OR_WORSE)].copy()
    counts["qa_or_worse_rows"] = len(qa)
    counts["out_rows"] = int((qa["status_norm"] == "out").sum())
    return qa, counts


def build_starter_keys(
    snaps_path: Path,
) -> tuple[set[tuple[int, int, str, str]], set[tuple[int, int, str, str]]]:
    snaps = pd.read_parquet(
        snaps_path,
        columns=["season", "game_type", "week", "team", "player", "offense_pct", "defense_pct"],
    )
    snaps = snaps.loc[snaps["game_type"] == "REG"].copy()
    snaps["norm_name"] = snaps["player"].map(normalize_name)
    snaps["share"] = snaps[["offense_pct", "defense_pct"]].max(axis=1)
    starters = snaps.loc[snaps["share"] >= STARTER_SNAP_SHARE].copy()
    exact: set[tuple[int, int, str, str]] = set()
    fuzzy: set[tuple[int, int, str, str]] = set()
    for season, week, team, name in zip(
        starters["season"], starters["week"], starters["team"], starters["norm_name"], strict=True
    ):
        key_next = (int(season), int(week) + 1, str(team))
        exact.add((*key_next, str(name)))
        init_last = initial_last_key(str(name))
        if init_last != ("", ""):
            fuzzy.add((*key_next, *init_last))
    return exact, fuzzy


def build_tuesday_visible(
    players_root: Path,
) -> tuple[dict[tuple[Any, ...], str], dict[tuple[Any, ...], str]]:
    injuries = pd.read_parquet(latest(players_root, "*/injuries.parquet")).copy()
    rosters = pd.read_parquet(
        latest(players_root, "*/weekly_rosters.parquet"),
        columns=["gsis_id", "full_name"],
    )
    name_map = (
        rosters.loc[rosters["gsis_id"].notna() & rosters["full_name"].notna()]
        .groupby("gsis_id")["full_name"]
        .agg(lambda s: s.mode().iat[0])
    )
    injuries = injuries.loc[injuries["game_type"] == "REG"].copy()
    injuries["full_name"] = injuries["gsis_id"].map(name_map)
    injuries["norm_name"] = injuries["full_name"].map(normalize_name)
    injuries["init_last"] = injuries["full_name"].map(initial_last_key)
    designated = injuries.loc[injuries["report_status"].notna()].copy()
    designated["date_modified"] = pd.to_datetime(designated["date_modified"], utc=True)

    def earliest_by(key_cols: list[str]) -> dict[tuple[Any, ...], str]:
        grouped = (
            designated.loc[designated[key_cols].notna().all(axis=1)]
            .groupby(key_cols)["date_modified"]
            .min()
        )
        return {key: str(value) for key, value in grouped.items()}

    exact = earliest_by(["season", "week", "team", "norm_name"])
    fuzzy = earliest_by(["season", "week", "team", "init_last"])
    return exact, fuzzy


def attach_flags(
    long: pd.DataFrame,
    qa: pd.DataFrame,
    starter_exact: set[tuple[int, int, str, str]],
    starter_fuzzy: set[tuple[int, int, str, str]],
    nflverse_earliest_exact: dict[tuple[Any, ...], str],
    nflverse_earliest_fuzzy: dict[tuple[Any, ...], str],
) -> pd.DataFrame:
    qa = qa.copy()
    is_starter: list[bool] = []
    for season, week, team, name in zip(
        qa["season"], qa["week"], qa["team"], qa["norm_name"], strict=True
    ):
        key3 = (int(season), int(week), str(team))
        init_last = initial_last_key(str(name))
        is_starter.append(
            (*key3, str(name)) in starter_exact
            or (init_last != ("", "") and (*key3, *init_last) in starter_fuzzy)
        )
    qa["is_starter_caliber"] = is_starter

    gameday_by_key = {
        (int(r.season), int(r.week), str(r.team)): r.gameday.normalize()
        for r in long.drop_duplicates(["season", "week", "team"]).itertuples(index=False)
    }

    def earliest_visible(row: pd.Series) -> str | None:
        key3 = (int(row["season"]), int(row["week"]), str(row["team"]))
        exact = nflverse_earliest_exact.get((*key3, row["norm_name"]))
        init_last = initial_last_key(str(row["norm_name"]))
        fuzzy = nflverse_earliest_fuzzy.get((*key3, *init_last)) if init_last != ("", "") else None
        if exact is None:
            return fuzzy
        if fuzzy is None:
            return exact
        return min(exact, fuzzy)

    def is_new(row: pd.Series) -> bool:
        gameday = gameday_by_key.get((int(row["season"]), int(row["week"]), str(row["team"])))
        if gameday is None:
            return False
        cutoff = gameday - pd.Timedelta(days=((gameday.weekday() - 1) % 7))
        earliest_str = earliest_visible(row)
        if earliest_str is None:
            return True
        return pd.Timestamp(earliest_str).tz_convert("UTC").date() > cutoff.date()

    qa["is_new_vs_tuesday"] = qa.apply(is_new, axis=1)

    grouped = qa.groupby(["season", "week", "team"])
    agg = grouped.agg(
        q_or_worse_any=("status_norm", "size"),
        out_count=("status_norm", lambda s: int((s == "out").sum())),
        starter_q_or_worse=("is_starter_caliber", "sum"),
        new_vs_tuesday=("is_new_vs_tuesday", "sum"),
    ).reset_index()

    work = long.merge(agg, on=["season", "week", "team"], how="left")
    work[["q_or_worse_any", "out_count", "starter_q_or_worse", "new_vs_tuesday"]] = work[
        ["q_or_worse_any", "out_count", "starter_q_or_worse", "new_vs_tuesday"]
    ].fillna(0)
    return work


def build_flag_masks(work: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    week_one = work["week"] == 1
    masks = {
        "q_or_worse_starter_caliber": work["starter_q_or_worse"] >= 1,
        "out_count_ge2": work["out_count"] >= 2,
        "new_saturday_designation": work["new_vs_tuesday"] >= 1,
    }
    missing = {
        "q_or_worse_starter_caliber": week_one & (work["starter_q_or_worse"] == 0),
        "out_count_ge2": pd.Series(False, index=work.index),
        "new_saturday_designation": pd.Series(False, index=work.index),
    }
    return masks, missing


def block_bootstrap_two_group(
    df: pd.DataFrame, *, flag_col: str, value_col: str, block_col: str, samples: int, seed: int
) -> np.ndarray:
    blocks, block_index = np.unique(df[block_col].to_numpy(), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    block_count = len(blocks)
    values = df[value_col].to_numpy(dtype=np.float64)
    flag = df[flag_col].to_numpy(dtype=bool)
    sums: dict[bool, np.ndarray] = {}
    counts: dict[bool, np.ndarray] = {}
    for group in (True, False):
        mask = flag == group
        sums[group] = np.bincount(
            block_index[mask], weights=values[mask], minlength=block_count
        ).astype(np.float64)
        counts[group] = np.bincount(block_index[mask], minlength=block_count).astype(np.float64)
    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)
    subset_count = drawn @ counts[True]
    complement_count = drawn @ counts[False]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_subset = (drawn @ sums[True]) / subset_count
        mean_complement = (drawn @ sums[False]) / complement_count
    gap = (mean_subset - mean_complement) * 100.0
    valid = (subset_count > 0) & (complement_count > 0)
    return gap[valid]


def summarize(
    df: pd.DataFrame, *, flag: pd.Series, block_col: str, samples: int, seed: int
) -> dict[str, Any]:
    n_total = len(df)
    n_flag = int(flag.sum())
    if n_flag == 0 or n_flag == n_total:
        return {"n_total": n_total, "n_flag": n_flag, "insufficient_data": True}
    work = df.copy()
    work["_flag"] = flag.to_numpy()
    subset_cover = float(work.loc[work["_flag"], "team_cover"].mean())
    complement_cover = float(work.loc[~work["_flag"], "team_cover"].mean())
    raw_gap_pts = (subset_cover - complement_cover) * 100.0
    fraction_of_slate = n_flag / n_total
    draws = block_bootstrap_two_group(
        work,
        flag_col="_flag",
        value_col="team_cover",
        block_col=block_col,
        samples=samples,
        seed=seed,
    )
    scaled_draws = draws * fraction_of_slate
    lower, upper = (
        np.quantile(scaled_draws, [0.025, 0.975]) if len(scaled_draws) else (np.nan, np.nan)
    )
    return {
        "n_total": n_total,
        "n_flag": n_flag,
        "n_blocks": int(work[block_col].nunique()),
        "subset_cover": subset_cover,
        "complement_cover": complement_cover,
        "raw_gap_pts": raw_gap_pts,
        "fraction_of_slate": fraction_of_slate,
        "full_slate_effect_pts": raw_gap_pts * fraction_of_slate,
        "ci95_scaled": [float(lower), float(upper)],
        "probability_positive": float(np.mean(draws > 0)) if len(draws) else np.nan,
        "bootstrap_samples": samples,
        "dropped_draws": int(samples - len(draws)),
        "insufficient_data": False,
    }


CELL_MECHANISMS = {
    "q_or_worse_starter_caliber": (
        ">=1 Questionable/Doubtful/Out designation on a >=50% prior-week snap-share "
        "player; frozen direction negative"
    ),
    "out_count_ge2": (">=2 Out designations, any players; frozen direction negative"),
    "new_saturday_designation": (
        ">=1 Q-or-worse designation absent from the Tuesday-visible nflverse state; "
        "frozen direction negative"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (
        REPO / "artifacts" / "nflcom_friday_designation_screen" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    schedules_path = args.schedules or latest(REPO / "data" / "raw", "*/schedules.parquet")
    print(f"=== schedules: {schedules_path} ===")
    long = load_population(schedules_path)
    n_games = long.attrs["n_games"]
    n_dropped = long.attrs["n_pushes_or_missing_dropped"]
    print(f"team-games: {len(long)} (games {n_games}, dropped {n_dropped})")

    qa, report_counts = load_report_flags(REPO / "data" / "raw" / "nflcom_injuries")
    print(f"=== nfl.com QA-or-worse rows: {report_counts['qa_or_worse_rows']} ===")

    snaps_path = latest(REPO / "data" / "players" / "raw", "*/snap_counts.parquet")
    starter_exact, starter_fuzzy = build_starter_keys(snaps_path)
    players_root = REPO / "data" / "players" / "raw"
    nflverse_exact, nflverse_fuzzy = build_tuesday_visible(players_root)

    work = attach_flags(long, qa, starter_exact, starter_fuzzy, nflverse_exact, nflverse_fuzzy)
    masks, missing = build_flag_masks(work)

    results = []
    for name, flag in masks.items():
        print(f"\n=== {name} ===")
        week_blocked = summarize(
            work, flag=flag, block_col="week_block", samples=args.samples, seed=args.seed
        )
        season_blocked = summarize(
            work, flag=flag, block_col="season", samples=args.samples, seed=args.seed
        )
        cell = {
            "name": f"nflcom_friday_{name}",
            "mechanism": CELL_MECHANISMS[name],
            "direction": "negative",
            "n_flag": int(flag.sum()),
            "n_missing_required_data": int(missing[name].sum()),
            "week_blocked_primary": week_blocked,
            "season_blocked_secondary": season_blocked,
        }
        results.append(cell)
        wb = week_blocked
        if wb.get("insufficient_data"):
            print("  insufficient data")
            continue
        print(
            f"  n_flag={cell['n_flag']} subset={wb['subset_cover']:.4f} "
            f"complement={wb['complement_cover']:.4f} raw_gap={wb['raw_gap_pts']:+.3f}pts"
        )
        print(
            f"  full_slate={wb['full_slate_effect_pts']:+.4f}pts "
            f"95% [{wb['ci95_scaled'][0]:+.4f}, {wb['ci95_scaled'][1]:+.4f}] "
            f"P+={wb['probability_positive']:.4f} blocks={wb['n_blocks']}"
        )
        sb = season_blocked
        if not sb.get("insufficient_data"):
            print(
                f"  [season secondary] {sb['full_slate_effect_pts']:+.4f}pts "
                f"[{sb['ci95_scaled'][0]:+.4f}, {sb['ci95_scaled'][1]:+.4f}] "
                f"P+={sb['probability_positive']:.4f}"
            )

    scored = [c for c in results if not c["week_blocked_primary"].get("insufficient_data")]
    ranked = sorted(
        scored, key=lambda c: abs(c["week_blocked_primary"]["full_slate_effect_pts"]), reverse=True
    )
    era_splits: dict[str, Any] = {}
    if ranked:
        strongest = ranked[0]["name"].removeprefix("nflcom_friday_")
        flag = masks[strongest]
        print(f"\n=== era split of strongest cell: {strongest} ===")
        for season in sorted(work["season"].unique()):
            sub = work.loc[work["season"] == season]
            sub_flag = flag.loc[sub.index]
            summary = summarize(
                sub, flag=sub_flag, block_col="week_block", samples=args.samples, seed=args.seed
            )
            era_splits[str(season)] = summary
            if not summary.get("insufficient_data"):
                print(
                    f"  {season}: n_flag={summary['n_flag']} "
                    f"gap={summary['raw_gap_pts']:+.3f}pts "
                    f"[{summary['ci95_scaled'][0]:+.3f}, {summary['ci95_scaled'][1]:+.3f}] "
                    f"P+={summary['probability_positive']:.4f}"
                )

    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "population": {
            "n_games": n_games,
            "n_team_games": len(work),
            "n_pushes_or_missing_dropped": n_dropped,
        },
        "report_counts": report_counts,
        "configuration": {
            "schedules": str(schedules_path),
            "predeclaration": ("docs/nflcom_friday_designation_screen.md (frozen before scoring)"),
        },
        "leakage_statement": (
            "flags derive only from each week's FINAL league injury page fetched-as-of "
            "that week (published Fri/Sat, predating kickoff); starter proxy uses "
            "prior-week snap shares only; cell (c) Tuesday cutoff uses nflverse "
            "date_modified metadata to reconstruct historical visibility"
        ),
        "recording_discipline": (
            "every cell predeclared unresolved_below_power unless a terminal AGENTS.md "
            "classification is admissible; registry writes only via explicit "
            "weak-signals record calls returned to the owner"
        ),
        "results": results,
        "era_splits": {"strongest_cell": next(iter(era_splits), None), "by_season": era_splits},
        "provenance": artifact_provenance(
            {
                "command": "nflcom-friday-designation-screen",
                "schedules": str(schedules_path),
                "bootstrap_samples": args.samples,
                "bootstrap_seed": args.seed,
                "predeclaration": (
                    "docs/nflcom_friday_designation_screen.md (frozen before scoring)"
                ),
            },
            schedules_path,
            project_root=REPO,
        ),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="nflcom-friday-designation-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation screen (3 predeclared cells + descriptive era "
            "split); mined family, every cell predeclared unresolved_below_power unless "
            "refuted_mechanism or positive-control bound applies."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
