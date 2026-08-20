"""Movement attribution: what does the market know that the model doesn't?

Predeclaration: `docs/movement_attribution.md` (frozen before any
flip-value below was computed). Owner question: the already-recorded
`observed_movement_*` family shows flipping to the market's side when
Tuesday-to-close movement disagrees with the model pick is worth +5.26
accuracy points (n=494, unfiltered) / +9.66 points (n=290,
`|open_move| >= 1.0`) -- see `docs/opener_error_analysis.md`'s
reconciliation section. This script attributes each adverse move to an
observable cause (injury news, weather deterioration, public-betting
alignment) using archives already on disk, then measures where the
flip-value concentrates by attribution class.

Every cell here is a **correlated decomposition** of the already-recorded
`observed_movement_*` family (same archive, same population, a
subpopulation cut) -- reported and recorded as such, never pooled
additively with the parent entries. Per AGENTS.md, every cell is reported
regardless of sign or whether its interval contains zero.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/movement_attribution.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import pick_correct, week_blocked_bootstrap
from nfl_ats.io import atomic_csv, atomic_parquet, run_id
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact

REPO = Path(__file__).resolve().parents[1]

ANCHOR_ARTIFACT = (
    REPO / "artifacts/observed_movement_channel/20260820T093426Z/per_game_tue_close.parquet"
)
GAME_FEATURES = REPO / "data/processed/game_features.parquet"
FORECAST_ARCHIVE = REPO / "data/raw/forecast_archive/full_2020_2025/forecasts.parquet"
INJURY_SNAPSHOT = REPO / "data/players/raw/20260817T184901Z/injuries.parquet"
PFT_INDEX = REPO / "data/raw/injury_news/20260819T191639Z/index.parquet"
PUBLIC_BETTING_INDEX = REPO / "data/raw/public_betting/20260820T111148Z/actionnetwork/index.parquet"
DEFAULT_OUTPUT_ROOT = REPO / "artifacts/movement_attribution"
DEFAULT_REGISTRY_ROOT = REPO / "registry"

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260820

SEVERITY = {"Out": 4.0, "Doubtful": 3.0, "Questionable": 2.0, "Probable": 1.0}
SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}
INJURY_NET_THRESHOLD = 2.0
PFT_NET_THRESHOLD = 1
WEATHER_WIND_DELTA_MPH = 10.0
WEATHER_TEMP_DELTA_F = 15.0
OFFICIAL_INJURY_MAX_SEASON = 2024  # injuries.parquet coverage ends here (measured)

# Extends src/nfl_ats/constants.py's TEAM_ABBREVIATION_ALIASES the same way
# scripts/ingest_public_betting.py does -- reproduced locally rather than
# imported, matching that script's own stated convention.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX", "WSH": "WAS"}

TEAM_NICKNAMES: dict[str, tuple[str, ...]] = {
    "ARI": ("cardinals",),
    "ATL": ("falcons",),
    "BAL": ("ravens",),
    "BUF": ("bills",),
    "CAR": ("panthers",),
    "CHI": ("bears",),
    "CIN": ("bengals",),
    "CLE": ("browns",),
    "DAL": ("cowboys",),
    "DEN": ("broncos",),
    "DET": ("lions",),
    "GB": ("packers",),
    "HOU": ("texans",),
    "IND": ("colts",),
    "JAX": ("jaguars",),
    "KC": ("chiefs",),
    "LA": ("rams",),
    "LAC": ("chargers",),
    "LV": ("raiders",),
    "MIA": ("dolphins",),
    "MIN": ("vikings",),
    "NE": ("patriots",),
    "NO": ("saints",),
    "NYG": ("giants",),
    "NYJ": ("jets",),
    "PHI": ("eagles",),
    "PIT": ("steelers",),
    "SEA": ("seahawks",),
    "SF": ("49ers", "niners"),
    "TB": ("buccaneers", "bucs"),
    "TEN": ("titans",),
    "WAS": ("commanders", "washington", "football team", "redskins"),
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def own_week_tuesday_noon_utc(kickoff_utc: pd.Series) -> pd.Series:
    """Own-week Tuesday noon ET, in UTC. Duplicated (not imported) from
    ``scripts/injury_tuesday_cutoff_experiment.py``'s
    ``team_week_tuesday_noon``/``scripts/ingest_public_betting.py``'s
    identical inline formula, per this repo's convention of not importing
    across ``scripts/*.py`` files."""

    kickoff_et = kickoff_utc.dt.tz_convert("US/Eastern")
    days_since_tuesday = (kickoff_et.dt.weekday - 1) % 7
    tuesday_date_et = kickoff_et.dt.normalize() - pd.to_timedelta(days_since_tuesday, unit="D")
    tuesday_noon_et = tuesday_date_et + pd.Timedelta(hours=12)
    return tuesday_noon_et.dt.tz_convert("UTC")


def _paired_metric_fn(candidate_col: str, production_col: str):
    def _metric(rows: pd.DataFrame) -> dict[str, float]:
        cand = pd.to_numeric(rows[candidate_col], errors="coerce")
        prod = pd.to_numeric(rows[production_col], errors="coerce")
        both = rows.loc[cand.notna() & prod.notna()]
        cand_v = pd.to_numeric(both[candidate_col], errors="coerce")
        prod_v = pd.to_numeric(both[production_col], errors="coerce")
        if len(cand_v) == 0:
            return {
                "candidate_accuracy": float("nan"),
                "production_accuracy": float("nan"),
                "paired_delta": float("nan"),
            }
        return {
            "candidate_accuracy": float(cand_v.mean()),
            "production_accuracy": float(prod_v.mean()),
            "paired_delta": float((cand_v - prod_v).mean()),
        }

    return _metric


def score_cell(frame: pd.DataFrame, *, name: str) -> dict[str, Any]:
    """Paired flip-value: market pick (oracle_pick_home) vs production pick."""

    if frame.empty:
        return {
            "cell": name,
            "n_graded": 0,
            "candidate_point_accuracy": float("nan"),
            "production_point_accuracy": float("nan"),
            "paired_delta_point": float("nan"),
            "paired_delta_week_ci": [float("nan"), float("nan")],
            "paired_delta_week_probability_positive": float("nan"),
        }
    working = frame.copy()
    working["_candidate_correct"] = pick_correct(
        working["oracle_pick_home"].astype(bool), working["margin_vs_open"]
    )
    bootstrap = week_blocked_bootstrap(
        working,
        _paired_metric_fn("_candidate_correct", "correct_at_open_probability_rule"),
        block="week",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    row = bootstrap.loc[bootstrap["metric"].eq("paired_delta")].iloc[0]
    both = working.dropna(subset=["_candidate_correct", "correct_at_open_probability_rule"])
    return {
        "cell": name,
        "n_graded": len(both),
        "candidate_point_accuracy": float(both["_candidate_correct"].mean()),
        "production_point_accuracy": float(both["correct_at_open_probability_rule"].mean()),
        "paired_delta_point": float(row["estimate"]),
        "paired_delta_week_ci": [float(row["lower"]), float(row["upper"])],
        "paired_delta_week_probability_positive": float(row["probability_positive"]),
    }


# ---------------------------------------------------------------------------
# Population construction
# ---------------------------------------------------------------------------


def load_population() -> pd.DataFrame:
    anchor = pd.read_parquet(ANCHOR_ARTIFACT)
    features = pd.read_parquet(GAME_FEATURES)[
        ["game_id", "home_team", "away_team", "kickoff", "game_type"]
    ]
    merged = anchor.merge(features, on="game_id", how="left", validate="one_to_one")
    unmatched = int(merged["home_team"].isna().sum())
    if unmatched:
        raise SystemExit(f"{unmatched} anchor games failed to join game_features on game_id")

    pushed_excluded = merged.loc[merged["correct_at_open_probability_rule"].notna()].copy()
    disagrees = pushed_excluded["open_move"].ne(0.0) & pushed_excluded["oracle_pick_home"].astype(
        bool
    ).ne(pushed_excluded["pick_home_at_open_probability_rule"].astype(bool))
    pop = pushed_excluded.loc[disagrees].copy()

    pick_home = pop["pick_home_at_open_probability_rule"].astype(bool)
    pop["moved_against_team"] = np.where(pick_home, pop["home_team"], pop["away_team"])
    pop["favored_team"] = np.where(pick_home, pop["away_team"], pop["home_team"])
    return pop


# ---------------------------------------------------------------------------
# (a) INJURY attribution
# ---------------------------------------------------------------------------


def _team_week_cutoffs(pop: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team, game_id) with tuesday_noon_utc/kickoff_utc,
    for BOTH teams in every population game (moved_against and favored)."""

    rows = []
    for side_col in ("moved_against_team", "favored_team"):
        rows.append(
            pop[["season", "week", "game_id", "kickoff", side_col]].rename(
                columns={side_col: "team"}
            )
        )
    long = pd.concat(rows, ignore_index=True).drop_duplicates(["season", "week", "team", "game_id"])
    long["tuesday_noon_utc"] = own_week_tuesday_noon_utc(long["kickoff"])
    long = long.rename(columns={"kickoff": "kickoff_utc"})
    return long


def _player_severity_asof(
    injuries_skill: pd.DataFrame, cutoffs: pd.DataFrame, cutoff_col: str
) -> pd.DataFrame:
    merged = injuries_skill.merge(cutoffs, on=["season", "week", "team"], how="inner")
    eligible = merged.loc[merged["date_modified"] <= merged[cutoff_col]].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["season", "week", "team", "game_id", "gsis_id", "severity"])
    eligible["severity"] = eligible["report_status"].map(SEVERITY).fillna(0.0)
    eligible = eligible.sort_values("date_modified")
    latest = eligible.drop_duplicates(["season", "week", "team", "game_id", "gsis_id"], keep="last")
    return latest[["season", "week", "team", "game_id", "gsis_id", "severity"]]


def compute_official_injury_scores(pop: pd.DataFrame) -> pd.DataFrame:
    """Per-game net_injury_score (official report path, seasons <= 2024)."""

    injuries = pd.read_parquet(INJURY_SNAPSHOT)
    injuries = injuries.loc[
        injuries["game_type"].eq("REG") & injuries["position"].isin(SKILL_POSITIONS)
    ].dropna(subset=["gsis_id", "date_modified"])

    cutoffs = _team_week_cutoffs(pop)
    tue = _player_severity_asof(injuries, cutoffs, "tuesday_noon_utc").rename(
        columns={"severity": "severity_tue"}
    )
    fin = _player_severity_asof(injuries, cutoffs, "kickoff_utc").rename(
        columns={"severity": "severity_final"}
    )
    both = tue.merge(fin, on=["season", "week", "team", "game_id", "gsis_id"], how="outer")
    both[["severity_tue", "severity_final"]] = both[["severity_tue", "severity_final"]].fillna(0.0)
    both["player_delta"] = both["severity_final"] - both["severity_tue"]
    team_delta = (
        both.groupby(["game_id", "team"])["player_delta"]
        .sum()
        .reset_index(name="team_injury_delta")
    )

    out = pop[["game_id", "season", "moved_against_team", "favored_team"]].copy()
    out = out.merge(
        team_delta.rename(
            columns={"team": "moved_against_team", "team_injury_delta": "delta_against"}
        ),
        on=["game_id", "moved_against_team"],
        how="left",
    )
    out = out.merge(
        team_delta.rename(columns={"team": "favored_team", "team_injury_delta": "delta_favored"}),
        on=["game_id", "favored_team"],
        how="left",
    )
    out[["delta_against", "delta_favored"]] = out[["delta_against", "delta_favored"]].fillna(0.0)
    out["net_injury_score"] = out["delta_against"] - out["delta_favored"]
    return out[["game_id", "delta_against", "delta_favored", "net_injury_score"]]


def compute_pft_fallback_scores(pop: pd.DataFrame) -> pd.DataFrame:
    """Per-game net_pft_score (headline-count fallback, season 2025 only)."""

    pop_2025 = pop.loc[pop["season"].eq(2025)]
    if pop_2025.empty:
        return pd.DataFrame(
            columns=["game_id", "pft_hits_against", "pft_hits_favored", "net_pft_score"]
        )

    cutoffs = _team_week_cutoffs(pop_2025)
    pft = pd.read_parquet(PFT_INDEX)
    pft = pft.loc[pft["injury_relevant"]].copy()
    pft["lastmod"] = pd.to_datetime(pft["lastmod"], utc=True)
    window_start = cutoffs["tuesday_noon_utc"].min()
    window_end = cutoffs["kickoff_utc"].max()
    pft = pft.loc[pft["lastmod"].between(window_start, window_end)].copy()
    pft["headline_norm"] = pft["headline_guess"].fillna("")

    def count_hits(team: str, start, end) -> int:
        window = pft.loc[pft["lastmod"].gt(start) & pft["lastmod"].le(end)]
        if window.empty:
            return 0
        nicknames = TEAM_NICKNAMES.get(team, ())
        if not nicknames:
            return 0
        mask = window["headline_norm"].str.contains("|".join(nicknames), case=False, regex=True)
        return int(mask.sum())

    records = []
    cutoff_lookup = cutoffs.set_index(["game_id", "team"])
    for row in pop_2025.itertuples(index=False):
        try:
            against_row = cutoff_lookup.loc[(row.game_id, row.moved_against_team)]
            favored_row = cutoff_lookup.loc[(row.game_id, row.favored_team)]
        except KeyError:
            continue
        hits_against = count_hits(
            row.moved_against_team, against_row["tuesday_noon_utc"], against_row["kickoff_utc"]
        )
        hits_favored = count_hits(
            row.favored_team, favored_row["tuesday_noon_utc"], favored_row["kickoff_utc"]
        )
        records.append(
            {
                "game_id": row.game_id,
                "pft_hits_against": hits_against,
                "pft_hits_favored": hits_favored,
                "net_pft_score": hits_against - hits_favored,
            }
        )
    return pd.DataFrame.from_records(records)


def attach_injury_flag(pop: pd.DataFrame) -> pd.DataFrame:
    official = compute_official_injury_scores(pop)
    pft = compute_pft_fallback_scores(pop)
    pop = pop.merge(official, on="game_id", how="left")
    pop = pop.merge(pft, on="game_id", how="left")

    is_official_eligible = pop["season"].le(OFFICIAL_INJURY_MAX_SEASON)
    official_flag = is_official_eligible & pop["net_injury_score"].fillna(0.0).ge(
        INJURY_NET_THRESHOLD
    )
    pft_flag = pop["season"].gt(OFFICIAL_INJURY_MAX_SEASON) & pop["net_pft_score"].fillna(0.0).ge(
        PFT_NET_THRESHOLD
    )
    pop["injury_path"] = np.where(is_official_eligible, "official", "pft_fallback")
    pop["INJURY"] = official_flag | pft_flag
    return pop


# ---------------------------------------------------------------------------
# (b) WEATHER attribution
# ---------------------------------------------------------------------------


def attach_weather_flag(pop: pd.DataFrame) -> pd.DataFrame:
    forecast = pd.read_parquet(FORECAST_ARCHIVE)[
        [
            "game_id",
            "roof",
            "actual_temp_f",
            "actual_wind_mph",
            "forecast_temp_f",
            "forecast_wind_mph",
        ]
    ]
    pop = pop.merge(forecast, on="game_id", how="left")
    outdoor = pop["roof"].isin(["outdoors", "open"])
    usable = (
        pop["actual_temp_f"].notna()
        & pop["actual_wind_mph"].notna()
        & pop["forecast_temp_f"].notna()
        & pop["forecast_wind_mph"].notna()
    )
    pop["wind_delta"] = pop["actual_wind_mph"] - pop["forecast_wind_mph"]
    pop["temp_delta"] = pop["actual_temp_f"] - pop["forecast_temp_f"]
    deteriorated = pop["wind_delta"].ge(WEATHER_WIND_DELTA_MPH) | pop["temp_delta"].abs().ge(
        WEATHER_TEMP_DELTA_F
    )
    pop["weather_data_usable"] = outdoor & usable
    pop["WEATHER"] = outdoor & usable & deteriorated
    return pop


# ---------------------------------------------------------------------------
# (c) PUBLIC ALIGNMENT attribution
# ---------------------------------------------------------------------------


def attach_public_flag(pop: pd.DataFrame) -> pd.DataFrame:
    pb = pd.read_parquet(PUBLIC_BETTING_INDEX)
    pb = pb.dropna(subset=["home_team", "away_team", "capture_ts"]).copy()
    pb["capture_ts"] = pd.to_datetime(pb["capture_ts"], utc=True)
    pb["start_time_utc"] = pd.to_datetime(pb["start_time_utc"], utc=True, errors="coerce")

    join_cols = pop[["game_id", "home_team", "away_team", "kickoff"]].copy()
    merged = pb.merge(join_cols, on=["home_team", "away_team"], how="inner", suffixes=("", "_pop"))
    merged["kickoff_delta_hours"] = (
        merged["kickoff"] - merged["start_time_utc"]
    ).dt.total_seconds().abs() / 3600.0
    matched = merged.loc[
        merged["kickoff_delta_hours"].le(72.0) & merged["capture_ts"].lt(merged["kickoff"])
    ]
    matched = matched.dropna(subset=["spread_home_bet_pct", "spread_away_bet_pct"])
    if matched.empty:
        pop["PUBLIC"] = False
        pop["public_subtype"] = None
        pop["public_bet_pct_move"] = np.nan
        pop["public_bet_pct_against"] = np.nan
        return pop

    latest = matched.sort_values("capture_ts").drop_duplicates("game_id", keep="last")
    latest = latest.set_index("game_id")

    move_side = np.where(pop["open_move"].to_numpy() > 0.0, "home", "away")
    bet_pct_move = []
    bet_pct_against = []
    matched_flag = []
    for gid, side in zip(pop["game_id"], move_side, strict=True):
        if gid not in latest.index:
            bet_pct_move.append(np.nan)
            bet_pct_against.append(np.nan)
            matched_flag.append(False)
            continue
        r = latest.loc[gid]
        if side == "home":
            bet_pct_move.append(float(r["spread_home_bet_pct"]))
            bet_pct_against.append(float(r["spread_away_bet_pct"]))
        else:
            bet_pct_move.append(float(r["spread_away_bet_pct"]))
            bet_pct_against.append(float(r["spread_home_bet_pct"]))
        matched_flag.append(True)

    pop = pop.copy()
    pop["public_bet_pct_move"] = bet_pct_move
    pop["public_bet_pct_against"] = bet_pct_against
    pop["PUBLIC"] = matched_flag
    subtype = np.where(
        pop["PUBLIC"],
        np.where(
            pop["public_bet_pct_move"] > pop["public_bet_pct_against"],
            "book_shading_public",
            np.where(
                pop["public_bet_pct_move"] < pop["public_bet_pct_against"],
                "reverse_line_movement",
                "even",
            ),
        ),
        None,
    )
    pop["public_subtype"] = subtype
    return pop


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT)
    args = parser.parse_args()
    started = time.time()

    print(f"Loading population from {ANCHOR_ARTIFACT}")
    pop = load_population()
    print(f"POP_UNFILTERED: n={len(pop)} (anchor target: 494)")

    print("Attaching INJURY flag (official 2020-2024, PFT fallback 2025)...")
    pop = attach_injury_flag(pop)
    print("Attaching WEATHER flag...")
    pop = attach_weather_flag(pop)
    print("Attaching PUBLIC flag...")
    pop = attach_public_flag(pop)

    pop["ATTRIBUTED"] = pop["INJURY"] | pop["WEATHER"] | pop["PUBLIC"]
    pop["UNATTRIBUTED"] = ~pop["ATTRIBUTED"]

    pop_threshold = pop.loc[pop["open_move"].abs().ge(1.0)].copy()
    print(f"POP_THRESHOLD (|open_move|>=1.0): n={len(pop_threshold)} (anchor target: 290)")

    # ------------------------------------------------------------------
    # Anchor reproduction check
    # ------------------------------------------------------------------
    anchor_unfiltered = score_cell(pop, name="anchor_unfiltered")
    anchor_threshold = score_cell(pop_threshold, name="anchor_threshold_ge_1_0")
    print(
        f"\nAnchor reproduction: unfiltered n={anchor_unfiltered['n_graded']} "
        f"delta={anchor_unfiltered['paired_delta_point'] * 100:+.2f}pts "
        f"P+={anchor_unfiltered['paired_delta_week_probability_positive']:.3f} "
        f"(target: n=494, +5.26pts, P+0.880)"
    )
    print(
        f"Anchor reproduction: threshold n={anchor_threshold['n_graded']} "
        f"delta={anchor_threshold['paired_delta_point'] * 100:+.2f}pts "
        f"P+={anchor_threshold['paired_delta_week_probability_positive']:.3f} "
        f"(target: n=290, +9.66pts, P+0.935)"
    )

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------
    coverage = {
        pop_name: {
            "n": len(frame),
            "injury_flag_rate": float(frame["INJURY"].mean()) if len(frame) else float("nan"),
            "weather_flag_rate": float(frame["WEATHER"].mean()) if len(frame) else float("nan"),
            "public_flag_rate": float(frame["PUBLIC"].mean()) if len(frame) else float("nan"),
            "any_attributed_rate": float(frame["ATTRIBUTED"].mean())
            if len(frame)
            else float("nan"),
            "unattributed_rate": float(frame["UNATTRIBUTED"].mean())
            if len(frame)
            else float("nan"),
            "weather_data_usable_rate": float(frame["weather_data_usable"].mean())
            if len(frame)
            else float("nan"),
            "n_season_2025": int(frame["season"].eq(2025).sum()),
        }
        for pop_name, frame in (("POP_UNFILTERED", pop), ("POP_THRESHOLD", pop_threshold))
    }
    print(f"\nCoverage: {coverage}")

    # ------------------------------------------------------------------
    # Stratified cells
    # ------------------------------------------------------------------
    cells: list[dict[str, Any]] = [
        {"population": "POP_UNFILTERED", "attribution": "ANCHOR_ALL", **anchor_unfiltered},
        {"population": "POP_THRESHOLD", "attribution": "ANCHOR_ALL", **anchor_threshold},
    ]

    for pop_name, frame in (("POP_UNFILTERED", pop), ("POP_THRESHOLD", pop_threshold)):
        for attribution_name, mask in (
            ("INJURY", frame["INJURY"]),
            ("WEATHER", frame["WEATHER"]),
            ("PUBLIC", frame["PUBLIC"]),
            ("ATTRIBUTED_ANY", frame["ATTRIBUTED"]),
            ("UNATTRIBUTED", frame["UNATTRIBUTED"]),
        ):
            cell = score_cell(frame.loc[mask], name=f"{pop_name}_{attribution_name}")
            cells.append({"population": pop_name, "attribution": attribution_name, **cell})

        for subtype in ("book_shading_public", "reverse_line_movement", "even"):
            sub_mask = frame["public_subtype"].eq(subtype)
            cell = score_cell(frame.loc[sub_mask], name=f"{pop_name}_PUBLIC_{subtype}")
            cells.append({"population": pop_name, "attribution": f"PUBLIC_{subtype}", **cell})

    cells_frame = pd.json_normalize(cells)
    print("\n" + cells_frame.to_string(index=False))

    # ------------------------------------------------------------------
    # Proposed weak-signals record commands (never executed here)
    # ------------------------------------------------------------------
    proposed_records: list[dict[str, Any]] = []
    for c in cells:
        if c["attribution"] == "ANCHOR_ALL":
            continue  # already recorded under observed_movement_* names
        if not np.isfinite(c["paired_delta_point"]):
            continue
        name = f"movement_attribution_{c['population'].lower()}_{c['attribution'].lower()}"
        p_plus = c["paired_delta_week_probability_positive"]
        effect_pts = c["paired_delta_point"] * 100.0
        low_pts = c["paired_delta_week_ci"][0] * 100.0
        high_pts = c["paired_delta_week_ci"][1] * 100.0
        notes = (
            f"CORRELATED DECOMPOSITION of observed_movement_oracle_full_slate / "
            f"observed_movement_threshold_1_0 (docs/movement_attribution.md) -- same archive, "
            f"same population, attribution subcut, NOT an independent sample. Do not pool "
            f"additively with those entries. population={c['population']}, "
            f"attribution={c['attribution']}, n={c['n_graded']}."
        )
        proposed_records.append(
            {
                "cell": name,
                "probability_positive": p_plus,
                "proposed_command": (
                    "nfl-ats weak-signals record "
                    f"--name {name} "
                    f'--description "movement attribution, population {c["population"]}, '
                    f'class {c["attribution"]}" '
                    "--source docs/movement_attribution.md "
                    f"--effect {effect_pts:.6f} --effect-units accuracy_points "
                    "--classification unresolved_below_power --league nfl "
                    "--season-start 2020 --season-end 2025 "
                    f"--interval-low {low_pts:.6f} --interval-high {high_pts:.6f} "
                    f"--probability-positive {p_plus:.6f} --sample-games {c['n_graded']} "
                    f'--notes "{notes}"'
                ),
            }
        )

    # ------------------------------------------------------------------
    # Write artifacts
    # ------------------------------------------------------------------
    output_dir = args.output_root / run_id()
    export_cols = [
        "game_id",
        "season",
        "week",
        "moved_against_team",
        "favored_team",
        "open_move",
        "INJURY",
        "injury_path",
        "net_injury_score",
        "net_pft_score",
        "WEATHER",
        "roof",
        "wind_delta",
        "temp_delta",
        "weather_data_usable",
        "PUBLIC",
        "public_subtype",
        "public_bet_pct_move",
        "public_bet_pct_against",
        "ATTRIBUTED",
        "UNATTRIBUTED",
        "oracle_pick_home",
        "pick_home_at_open_probability_rule",
        "margin_vs_open",
        "correct_at_open_probability_rule",
    ]
    atomic_parquet(pop[export_cols], output_dir / "per_game_attribution.parquet")
    atomic_csv(cells_frame, output_dir / "cells_summary.csv")

    configuration = {
        "command": "movement-attribution",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "injury_net_threshold": INJURY_NET_THRESHOLD,
        "pft_net_threshold": PFT_NET_THRESHOLD,
        "weather_wind_delta_mph": WEATHER_WIND_DELTA_MPH,
        "weather_temp_delta_f": WEATHER_TEMP_DELTA_F,
        "official_injury_max_season": OFFICIAL_INJURY_MAX_SEASON,
        "predeclaration": "docs/movement_attribution.md",
        "anchor_artifact": str(ANCHOR_ARTIFACT),
    }
    metadata: dict[str, Any] = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **configuration,
        "anchor_reproduction": {
            "unfiltered": anchor_unfiltered,
            "threshold_ge_1_0": anchor_threshold,
            "target_unfiltered": {"n": 494, "delta_pts": 5.26, "probability_positive": 0.880},
            "target_threshold": {"n": 290, "delta_pts": 9.66, "probability_positive": 0.935},
        },
        "coverage": coverage,
        "cells": cells,
        "proposed_weak_signal_records": proposed_records,
        "provenance": artifact_provenance(configuration, GAME_FEATURES, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "metadata.json",
        metadata,
        command="movement-attribution",
        metrics=metadata,
        registry_root=args.registry_root,
        notes=(
            "Movement attribution: INJURY (official report 2020-2024 + PFT fallback 2025), "
            "WEATHER (Tuesday-noon forecast vs actual), PUBLIC (actionnetwork bet% alignment). "
            "All cells are correlated decompositions of observed_movement_* -- see notes on "
            "each proposed record."
        ),
    )
    print(f"\nelapsed: {time.time() - started:.1f}s")
    print(f"artifacts: {output_dir}")


if __name__ == "__main__":
    main()
