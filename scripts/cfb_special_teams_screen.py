"""Cross-league replication (CFB) of two NFL PBP-06 special-teams battery
cells, both **read** this session from ``registry/weak_signals.json``,
both ``unresolved_below_power``, NFL REG 2009-2025, n=8,634:

- ``special_teams_return_top_quartile``: +0.4986 full-slate pts, week-blocked
  95% [-0.0742, +1.0797], P+ 0.9547 -- today's strongest new NFL lead.
- ``special_teams_punt_net_top_quartile``: -0.3890 full-slate pts,
  week-blocked 95% [-0.9674, +0.2038], P+ 0.0946 -- its INVERSE-LEAN sibling
  in the same battery (predeclared sign +1, measured result leans negative,
  despite `punt_net_yards` having the STRONGEST YoY reliability, +0.313, of
  the battery's four kept dimensions).

**Predeclaration**: ``docs/cfb_special_teams_replication.md``, written and
frozen BEFORE this script was run against any CFB cover outcome. Do not add,
remove, or redefine a cell here without updating that document first.

**Binding taxonomy (owned verbatim, per AGENTS.md/CLAUDE.md)**: an interval
or CI that contains zero is NEVER grounds to reject, fail, or close an
experiment -- at this evaluator's ~2-point resolution "contains zero" is the
EXPECTED outcome for a real small signal. Only two closing grounds: (1)
refuted mechanism -- RESOLVED wrong sign (whole interval on the wrong side
of zero) or zero split-half reliability; (2) bounded by a positive control
proven able to detect an effect that size. Everything else is
``unresolved_below_power``: record with ``probability_positive``, never
"contains zero". If a record command errors, the verdict is wrong, not the
validator. Every cell below is recorded to the registry regardless of sign
or interval shape -- this script does not decide anything, it only measures.

**Data source**: local CFB PBP snapshots only
(``data/cfb/pbp/raw/*/season=*/plays.parquet``, the XLG-02 55-column
canonical contract table). No fresh CFBD PBP fetch. 2004 is excluded (0%
kick-distance parse rate, 144 punt rows for the whole season -- effectively
absent, not merely thin; measured in ``docs/cfb_special_teams_replication.md``).
2005-2025 (21 seasons) are used to build team-season traits.

Return yardage uses ``statYardage`` DIRECTLY (a clean numeric column, not
text-parsed) -- measured more reliable than a text-regex parse of the return
number (see the predeclaration doc's feasibility section). ``text`` is used
only for play classification (touchback/fair catch/downed/out-of-bounds/
blocked/has-return, all robust substring checks) and to parse kick distance
(which has no numeric column and must come from text).

Writes results to the session scratchpad (never to ``registry/`` -- recording
happens via separate ``nfl-ats weak-signals record --league cfb`` invocations
under the repository's registry write-lock protocol).
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

sys.path.append(str(REPO / "scripts"))

from _common import block_bootstrap_two_group  # noqa: E402

from nfl_ats.cfb_benchmark import CFB_CLEAN_CORE_SEASONS  # noqa: E402
from nfl_ats.experiment_runner import scale_subset_effect  # noqa: E402

OUT_DIR = Path(
    r"C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3"
    r"\8e5a2321-bbd8-4adc-8835-f028fc2cc066\scratchpad\cfb_special_teams"
)
RESULTS_PATH = OUT_DIR / "cfb_special_teams_results.json"
TEAM_SEASON_PATH = OUT_DIR / "cfb_special_teams_team_season.parquet"

PBP_ROOT = REPO / "data" / "cfb" / "pbp" / "raw"
FEATURES_PATH = REPO / "data" / "processed" / "cfb_game_features.parquet"

SEASON_START = 2005  # 2004 excluded: see predeclaration doc feasibility section
SEASON_END = 2025
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819
QUARTILE_TOP = 0.75
CFB_REGULAR_SEASON_TYPE = 2

PUNT_TYPES = {
    "Punt",
    "Blocked Punt",
    "Punt Return Touchdown",
    "Punt Team Fumble Recovery",
    "Blocked Punt Touchdown",
    "Punt (Safety)",
}
KICKOFF_TYPES = {
    "Kickoff",
    "Kickoff Return (Offense)",
    "Kickoff Return Touchdown",
    "Kickoff Team Fumble Recovery",
}

DIST_RE = re.compile(r"(?:punt|kickoff)(?:\s+for)?\s+(\d+)\s*(?:yards|yds)", re.IGNORECASE)

RAW_DIMENSIONS = ("punt_net_yards", "punt_return_yards", "kickoff_return_yards")


def _parse_kick_distance(text: object) -> float:
    if not isinstance(text, str):
        return np.nan
    m = DIST_RE.search(text)
    return float(m.group(1)) if m else np.nan


def _classify(text: object) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Returns (touchback, fair_catch, downed, out_of_bounds, blocked, has_return)."""

    t = text.lower() if isinstance(text, str) else ""
    return (
        "touchback" in t,
        "fair catch" in t,
        "downed" in t,
        ("out-of-bounds" in t) or ("out of bounds" in t),
        "blocked" in t,
        "return" in t,
    )


def _season_snapshot_paths() -> dict[int, Path]:
    """Newest plays.parquet per season across every PBP snapshot directory."""

    found: dict[int, tuple[str, Path]] = {}
    for season_dir in PBP_ROOT.glob("*/season=*"):
        try:
            season = int(season_dir.name.split("=", 1)[1])
        except ValueError:
            continue
        plays_path = season_dir / "plays.parquet"
        if not plays_path.exists():
            continue
        snapshot_ts = season_dir.parent.name
        prior = found.get(season)
        if prior is None or snapshot_ts > prior[0]:
            found[season] = (snapshot_ts, plays_path)
    return {season: path for season, (_, path) in found.items()}


def _build_season_fragments(season: int, path: Path) -> dict[str, pd.DataFrame]:
    cols = [
        "season",
        "week",
        "game_id",
        "seasonType",
        "pos_team_id",
        "def_pos_team_id",
        "type.text",
        "text",
        "statYardage",
        "start.yardsToEndzone",
    ]
    raw = pd.read_parquet(path, columns=cols)
    raw = raw.loc[raw["seasonType"] == CFB_REGULAR_SEASON_TYPE].copy()

    punt = raw.loc[raw["type.text"].isin(PUNT_TYPES)].copy()
    ko = raw.loc[raw["type.text"].isin(KICKOFF_TYPES)].copy()

    def _classified(frame: pd.DataFrame) -> pd.DataFrame:
        flags = frame["text"].map(_classify).apply(pd.Series)
        flags.columns = [
            "touchback",
            "fair_catch",
            "downed",
            "out_of_bounds",
            "blocked",
            "has_return",
        ]
        out = pd.concat([frame.reset_index(drop=True), flags.reset_index(drop=True)], axis=1)
        out["is_real_return"] = (
            out["has_return"]
            & ~out["touchback"]
            & ~out["fair_catch"]
            & ~out["downed"]
            & ~out["out_of_bounds"]
            & ~out["blocked"]
        )
        return out

    punt = _classified(punt)
    ko = _classified(ko)

    # --- punt_net_yards, grouped by the KICKING team (pos_team_id) ---
    punt["kick_distance"] = punt["text"].map(_parse_kick_distance)
    punt["return_yards_for_net"] = np.where(punt["is_real_return"], punt["statYardage"], 0.0)
    normal_net = punt["kick_distance"] - punt["return_yards_for_net"]
    touchback_net = punt["start.yardsToEndzone"] - 20.0
    punt["net_yards"] = np.where(punt["touchback"], touchback_net, normal_net)
    punt_net_season = (
        punt.dropna(subset=["net_yards"])
        .groupby(["season", "pos_team_id"], sort=False)
        .agg(n_punts=("net_yards", "size"), punt_net_yards=("net_yards", "mean"))
        .reset_index()
        .rename(columns={"pos_team_id": "team_id"})
    )

    # --- punt_return_yards, grouped by the RETURNING team (def_pos_team_id) ---
    punt_real = punt.loc[punt["is_real_return"]].copy()
    punt_return_season = (
        punt_real.groupby(["season", "def_pos_team_id"], sort=False)
        .agg(n_punt_returns=("statYardage", "size"), punt_return_yards=("statYardage", "mean"))
        .reset_index()
        .rename(columns={"def_pos_team_id": "team_id"})
    )

    # --- kickoff_return_yards, grouped by the RETURNING team (def_pos_team_id) ---
    ko_real = ko.loc[ko["is_real_return"]].copy()
    kickoff_return_season = (
        ko_real.groupby(["season", "def_pos_team_id"], sort=False)
        .agg(
            n_kickoff_returns=("statYardage", "size"),
            kickoff_return_yards=("statYardage", "mean"),
        )
        .reset_index()
        .rename(columns={"def_pos_team_id": "team_id"})
    )

    return {
        "punt_net_season": punt_net_season,
        "punt_return_season": punt_return_season,
        "kickoff_return_season": kickoff_return_season,
        "n_raw_rows": len(raw),
        "n_punt_rows": len(punt),
        "n_ko_rows": len(ko),
    }


def build_team_season() -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = _season_snapshot_paths()
    seasons = [s for s in range(SEASON_START, SEASON_END + 1) if s in paths]
    missing_seasons = [s for s in range(SEASON_START, SEASON_END + 1) if s not in paths]

    punt_net_parts, punt_return_parts, kickoff_return_parts = [], [], []
    total_raw_rows = 0
    per_season_audit: dict[str, Any] = {}
    for season in seasons:
        frags = _build_season_fragments(season, paths[season])
        punt_net_parts.append(frags["punt_net_season"])
        punt_return_parts.append(frags["punt_return_season"])
        kickoff_return_parts.append(frags["kickoff_return_season"])
        total_raw_rows += frags["n_raw_rows"]
        per_season_audit[str(season)] = {
            "n_raw_rows": frags["n_raw_rows"],
            "n_punt_rows": frags["n_punt_rows"],
            "n_ko_rows": frags["n_ko_rows"],
            "snapshot": str(paths[season]),
        }
        print(
            f"  season {season}: raw={frags['n_raw_rows']} "
            f"punt={frags['n_punt_rows']} ko={frags['n_ko_rows']}"
        )

    punt_net = pd.concat(punt_net_parts, ignore_index=True)
    punt_return = pd.concat(punt_return_parts, ignore_index=True)
    kickoff_return = pd.concat(kickoff_return_parts, ignore_index=True)

    team_season = punt_net.merge(punt_return, on=["season", "team_id"], how="outer")
    team_season = team_season.merge(kickoff_return, on=["season", "team_id"], how="outer")
    team_season = team_season.sort_values(["season", "team_id"]).reset_index(drop=True)

    # League-center each dimension within its own season (era-drift removal,
    # identical convention to special_teams_features.py::add_league_centered).
    for dim in RAW_DIMENSIONS:
        league_mean = team_season.groupby("season")[dim].transform("mean")
        team_season[f"{dim}_centered"] = team_season[dim] - league_mean

    audit = {
        "seasons_used": seasons,
        "seasons_missing": missing_seasons,
        "total_raw_rows_processed": total_raw_rows,
        "team_season_rows": len(team_season),
        "per_season": per_season_audit,
    }
    return team_season, audit


def add_composites(team_season: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    result = team_season.copy()
    sds: dict[str, float] = {}
    for dim in RAW_DIMENSIONS:
        centered = f"{dim}_centered"
        sd = float(result[centered].std(ddof=1))
        sds[dim] = sd
        result[f"{dim}_z"] = result[centered] / sd if sd > 0 else np.nan
    result["return_composite_z"] = result[["punt_return_yards_z", "kickoff_return_yards_z"]].mean(
        axis=1
    )
    return result, sds


def _prior(table: pd.DataFrame, columns: list[str], rename_team: str) -> pd.DataFrame:
    shifted = table[["team_id", "season", *columns]].copy()
    shifted["season"] = shifted["season"] + 1
    rename = {c: f"prior_{c}" for c in columns}
    rename["team_id"] = rename_team
    return shifted.rename(columns=rename)


def load_population() -> pd.DataFrame:
    features = pd.read_parquet(FEATURES_PATH)
    features["season"] = pd.to_numeric(features["season"], errors="raise").astype(int)
    features["week"] = pd.to_numeric(features["week"], errors="raise").astype(int)
    features["game_id"] = pd.to_numeric(features["game_id"], errors="raise").astype("int64")
    features["home_id"] = pd.to_numeric(features["home_id"], errors="raise").astype("int64")
    features["away_id"] = pd.to_numeric(features["away_id"], errors="raise").astype("int64")

    cc = features.loc[features["season"].isin(CFB_CLEAN_CORE_SEASONS)].copy()
    n_clean_core_all = len(cc)
    cc = cc.loc[cc["home_cover"].notna()].reset_index(drop=True)
    n_total = len(cc)
    cc.attrs["n_clean_core_all"] = n_clean_core_all
    cc.attrs["n_pushes_dropped"] = n_clean_core_all - n_total
    cc.attrs["n_total"] = n_total
    cc.attrs["n_neutral_site_kept"] = int(
        pd.to_numeric(cc["neutral_site"], errors="coerce").fillna(0).astype(int).eq(1).sum()
    )
    cc["week_block"] = cc["season"] * 100 + cc["week"]
    return cc


def build_long_table(games: pd.DataFrame, team_season: pd.DataFrame) -> pd.DataFrame:
    trait_cols = ["punt_net_yards_z", "return_composite_z"]
    sides = []
    for is_home in (True, False):
        team_col = "home_id" if is_home else "away_id"
        side = pd.DataFrame(
            {
                "game_id": games["game_id"],
                "season": games["season"],
                "week": games["week"],
                "week_block": games["week_block"],
                "team_id": games[team_col],
                "team_covered": (games["home_cover"] if is_home else 1.0 - games["home_cover"]),
            }
        )
        prior = _prior(team_season, trait_cols, "team_id")
        side = side.merge(prior, on=["team_id", "season"], how="left")
        sides.append(side)
    return pd.concat(sides, ignore_index=True).reset_index(drop=True)


def summarize(
    df: pd.DataFrame,
    *,
    flag: pd.Series,
    value_col: str,
    block_col: str,
    sign: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    n_total = len(df)
    n_flag = int(flag.sum())
    n_complement = n_total - n_flag
    if n_flag == 0 or n_complement == 0:
        return {
            "n_total": n_total,
            "n_flag": n_flag,
            "n_complement": n_complement,
            "insufficient_data": True,
        }

    work = df.copy()
    work["_flag"] = flag.to_numpy()
    subset_mean = float(work.loc[work["_flag"], value_col].mean())
    complement_mean = float(work.loc[~work["_flag"], value_col].mean())
    raw_gap_fraction = subset_mean - complement_mean
    fraction_of_slate = n_flag / n_total
    full_slate_effect_pts = scale_subset_effect(
        raw_gap_fraction, sign=sign, fraction_of_slate=fraction_of_slate
    )

    draws = block_bootstrap_two_group(
        work, flag_col="_flag", value_col=value_col, block_col=block_col, samples=samples, seed=seed
    )
    signed_draws = sign * draws
    scaled_draws = signed_draws * fraction_of_slate
    lower, upper = (
        np.quantile(scaled_draws, [0.025, 0.975]) if len(scaled_draws) else (np.nan, np.nan)
    )

    return {
        "n_total": n_total,
        "n_flag": n_flag,
        "n_complement": n_complement,
        "n_blocks": int(work[block_col].nunique()),
        "subset_mean": subset_mean,
        "complement_mean": complement_mean,
        "raw_gap_pts": sign * raw_gap_fraction * 100.0,
        "fraction_of_slate": fraction_of_slate,
        "full_slate_effect_pts": full_slate_effect_pts,
        "ci95_scaled": [float(lower), float(upper)],
        "probability_positive": float(np.mean(scaled_draws > 0)) if len(scaled_draws) else np.nan,
        "bootstrap_samples": samples,
        "insufficient_data": False,
    }


def score_cell(
    df: pd.DataFrame,
    name: str,
    *,
    flag: pd.Series,
    missing_mask: pd.Series,
    sign: int,
    description: str,
    reliability_note: str,
    nfl_comparison: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    flag = flag.fillna(False).astype(bool)
    missing_mask = missing_mask.fillna(False).astype(bool)
    week_blocked = summarize(
        df,
        flag=flag,
        value_col="team_covered",
        block_col="week_block",
        sign=sign,
        samples=samples,
        seed=seed,
    )
    season_blocked = summarize(
        df,
        flag=flag,
        value_col="team_covered",
        block_col="season",
        sign=sign,
        samples=samples,
        seed=seed,
    )
    return {
        "name": name,
        "description": description,
        "sign_dir": sign,
        "reliability_note": reliability_note,
        "nfl_comparison": nfl_comparison,
        "n_flag": int(flag.sum()),
        "n_missing_required_data": int(missing_mask.sum()),
        "week_blocked": week_blocked,
        "season_blocked_secondary": season_blocked,
    }


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== building CFB special-teams team-season traits, {SEASON_START}-{SEASON_END} ===")
    print("(2004 excluded: see docs/cfb_special_teams_replication.md feasibility section)")
    team_season_raw, build_audit = build_team_season()
    team_season_raw.to_parquet(TEAM_SEASON_PATH, index=False)
    print(f"\nwrote {TEAM_SEASON_PATH} ({len(team_season_raw)} rows)")

    team_season, sds = add_composites(team_season_raw)

    thresholds: dict[str, float] = {
        "punt_net_yards_z": float(team_season["punt_net_yards_z"].quantile(QUARTILE_TOP)),
        "return_composite_z": float(team_season["return_composite_z"].quantile(QUARTILE_TOP)),
    }
    print(
        f"\n=== quartile thresholds (top=0.75, {len(team_season)}-row "
        f"{SEASON_START}-{SEASON_END} panel) ==="
    )
    for k, v in thresholds.items():
        print(f"  {k}: top={v:.4f}")

    print(f"\n=== loading CFB clean-core population ({FEATURES_PATH}) ===")
    games = load_population()
    n_total = games.attrs["n_total"]
    print(
        f"clean-core rows before push-drop: {games.attrs['n_clean_core_all']}; "
        f"pushes/missing dropped: {games.attrs['n_pushes_dropped']}; "
        f"scored population n_total={n_total}; "
        f"neutral-site kept: {games.attrs['n_neutral_site_kept']}"
    )

    long_df = build_long_table(games, team_season)

    cell_specs = [
        (
            "cfb_special_teams_return_top_quartile",
            "prior_return_composite_z",
            "return_composite_z",
            1,
            "Top-quartile CFB teams by prior-season return_composite (mean z of punt-return "
            "and kickoff-return yards) vs the field. Replicates NFL special_teams_return_top_"
            "quartile (predeclared sign +1). Predicted POSITIVE on team_covered "
            "(docs/cfb_special_teams_replication.md).",
            "CFB composite reliability not separately measured; NFL componentwise YoY r: "
            "punt_return +0.109 [+0.019,+0.196], kickoff_return +0.158 [+0.073,+0.243].",
            "NFL special_teams_return_top_quartile: +0.4986 pts, week-blocked 95% "
            "[-0.0742,+1.0797], P+ 0.9547, n=8634 (registry/weak_signals.json, read this session).",
        ),
        (
            "cfb_special_teams_punt_net_top_quartile",
            "prior_punt_net_yards_z",
            "punt_net_yards_z",
            1,
            "Top-quartile CFB teams by prior-season punt_net_yards vs the field. Replicates "
            "NFL special_teams_punt_net_top_quartile (predeclared sign +1, matching the NFL "
            "cell's predeclared -- not measured -- direction). Predicted POSITIVE on "
            "team_covered (docs/cfb_special_teams_replication.md).",
            "NFL YoY Pearson r +0.313, 95% CI [+0.233,+0.391] -- the strongest of the battery's "
            "four kept dimensions, despite the NFL cover-rate result leaning negative.",
            "NFL special_teams_punt_net_top_quartile: -0.3890 pts, week-blocked 95% "
            "[-0.9674,+0.2038], P+ 0.0946, n=8634 (registry/weak_signals.json, read this session) "
            "-- the battery's inverse-lean sibling to the return cell.",
        ),
    ]

    cells: list[dict[str, Any]] = []
    for name, col, threshold_key, sign, description, reliability_note, nfl_comparison in cell_specs:
        cutoff = thresholds[threshold_key]
        flag = long_df[col] >= cutoff
        missing = long_df[col].isna()
        cells.append(
            score_cell(
                long_df,
                name,
                flag=flag,
                missing_mask=missing,
                sign=sign,
                description=description,
                reliability_note=reliability_note,
                nfl_comparison=nfl_comparison,
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
            )
        )

    print("\n=== results ===")
    for cell in cells:
        wb = cell["week_blocked"]
        sb = cell["season_blocked_secondary"]
        print(f"\n--- {cell['name']} ---")
        if wb.get("insufficient_data"):
            print("  insufficient data")
            continue
        print(
            f"  n_flag={cell['n_flag']} n_total={wb['n_total']} "
            f"n_missing_required_data={cell['n_missing_required_data']}"
        )
        print(
            f"  full_slate_effect={wb['full_slate_effect_pts']:+.4f}pts week-blocked 95% "
            f"[{wb['ci95_scaled'][0]:+.4f}, {wb['ci95_scaled'][1]:+.4f}] "
            f"P+={wb['probability_positive']:.4f}"
        )
        if not sb.get("insufficient_data"):
            print(
                f"  [season-blocked secondary] 95% [{sb['ci95_scaled'][0]:+.4f}, "
                f"{sb['ci95_scaled'][1]:+.4f}] P+={sb['probability_positive']:.4f}"
            )

    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "predeclaration": (
            "docs/cfb_special_teams_replication.md (frozen before this script scored anything)"
        ),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "season_start_features": SEASON_START,
        "season_end_features": SEASON_END,
        "build_audit": build_audit,
        "n_clean_core_all": games.attrs["n_clean_core_all"],
        "n_pushes_dropped": games.attrs["n_pushes_dropped"],
        "n_total_population": n_total,
        "n_neutral_site_kept": games.attrs["n_neutral_site_kept"],
        "thresholds": thresholds,
        "pooled_sd": sds,
        "results": cells,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
