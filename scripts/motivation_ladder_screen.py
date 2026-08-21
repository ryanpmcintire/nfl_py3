"""Late-season motivation-state ladder screen (4 predeclared cells).

Predeclaration: docs/motivation_ladder_screen.md, frozen before this script
was run against any cover outcome. States (eliminated / clinched /
locked_seed / tank_zone / fighter / nothing_to_play_for) are reconstructed
week-by-week from completed games only -- every state attached to a game uses
strictly earlier gameday values, so the construction is point-in-time safe.
Elimination and clinching are approximations (no tiebreakers, conservative
worst-case rules); three historical spot-checks are asserted before scoring.
Binding taxonomy owned verbatim per AGENTS.md: an interval that crosses zero
is NEVER a closing ground; only refuted mechanism or bounded-by-positive-
control closes a line; everything else is unresolved_below_power, recorded
with probability_positive. This script only measures; it records nothing to
the weak-signal registry.

Method copied algorithm-identical from scripts/redzone_reversion_screen.py
(load_schedules / block_bootstrap_two_group / summarize / score_cell).

Writes artifacts/motivation_ladder_screen/<UTC stamp>/results.json and the
experiment-registry row under registry/experiments/motivation-ladder-screen/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.experiment_runner import scale_subset_effect  # noqa: E402
from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

SEASON_START = 2009
SEASON_END = 2025
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260821
WEEK_MIN = 13
WEEK_MAX = 18

DIVISIONS: dict[str, str] = {}
for _div, _teams in {
    "AFC_E": ("BUF", "MIA", "NE", "NYJ"),
    "AFC_N": ("BAL", "CIN", "CLE", "PIT"),
    "AFC_S": ("HOU", "IND", "JAX", "TEN"),
    "AFC_W": ("DEN", "KC", "LAC", "LV"),
    "NFC_E": ("DAL", "NYG", "PHI", "WAS"),
    "NFC_N": ("CHI", "DET", "GB", "MIN"),
    "NFC_S": ("ATL", "CAR", "NO", "TB"),
    "NFC_W": ("ARI", "LA", "SF", "SEA"),
}.items():
    for _team in _teams:
        DIVISIONS[_team] = _div

CONFERENCE = {t: d.split("_")[0] for t, d in DIVISIONS.items()}

SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "gameday",
    "game_type",
    "home_team",
    "away_team",
    "result",
    "spread_line",
]


def _latest_schedules() -> Path:
    candidates = sorted((REPO / "data/raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


def load_schedules(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    available = [c for c in SCHEDULE_COLUMNS if c in raw.columns]
    df = raw.loc[:, available].copy()
    df = df.loc[df["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)
    df["home_team"] = df["home_team"].replace(TEAM_ABBREVIATION_ALIASES)
    df["away_team"] = df["away_team"].replace(TEAM_ABBREVIATION_ALIASES)
    df = add_ats_outcomes(df)
    n_before_push_drop = len(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)
    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = n_before_push_drop - len(df)
    df["week_block"] = df["season"] * 100 + df["week"]
    return df


def _season_playoff_slots(season: int) -> int:
    return 7 if season >= 2020 else 6


def _ordering_key(wins: int, losses: int, team: str) -> tuple[int, int, str]:
    return (-wins, losses, team)


def compute_day_states(
    season: int,
    tallies: dict[str, tuple[int, int, int]],
    total_games: dict[str, int],
    played_through: dict[str, int],
    playing_today: set[str],
) -> dict[str, dict[str, Any]]:
    gr = {
        team: total_games[team] - played_through[team] - (1 if team in playing_today else 0)
        for team in DIVISIONS
    }
    states: dict[str, dict[str, Any]] = {}
    for conference in ("AFC", "NFC"):
        members = [t for t in DIVISIONS if CONFERENCE[t] == conference]
        ordered = sorted(members, key=lambda t: _ordering_key(*tallies[t][:2], t))
        rank_of = {t: i + 1 for i, t in enumerate(ordered)}
        k = _season_playoff_slots(season)
        cut_line = tallies[ordered[k - 1]][0]
        first_out_wins = tallies[ordered[k]][0] if len(ordered) > k else 0
        division_leader: dict[str, str] = {}
        for division in sorted({DIVISIONS[t] for t in members}):
            div_members = [t for t in members if DIVISIONS[t] == division]
            division_leader[division] = min(
                div_members, key=lambda t: _ordering_key(*tallies[t][:2], t)
            )
        for team in members:
            wins, losses, _ties = tallies[team]
            eliminated = wins + gr[team] < first_out_wins
            challengers = [
                o
                for o in members
                if o != team
                and not (division_leader[DIVISIONS[o]] == o and rank_of[o] < rank_of[team])
            ]
            challenger_max = max(tallies[o][0] + gr[o] for o in challengers)
            clinched = wins - gr[team] >= challenger_max
            if clinched:
                if rank_of[team] == 1:
                    locked = True
                else:
                    above = ordered[rank_of[team] - 2]
                    locked = wins + gr[team] < tallies[above][0]
            else:
                locked = False
            states[team] = {
                "wins": wins,
                "losses": losses,
                "games_remaining": gr[team],
                "conference_rank": rank_of[team],
                "cut_line": cut_line,
                "eliminated": bool(eliminated),
                "clinched": bool(clinched),
                "locked_seed": bool(locked),
            }

    league_ordered = sorted(DIVISIONS, key=lambda t: (tallies[t][0], -tallies[t][1], t))
    tank_zone = set(league_ordered[:2])
    for team in DIVISIONS:
        st = states[team]
        st["tank_zone"] = team in tank_zone
        st["fighter"] = (
            not st["eliminated"]
            and st["conference_rank"] > _season_playoff_slots(season)
            and st["cut_line"] - st["wins"] <= 1
        )
        st["nothing_to_play_for"] = st["eliminated"] and not st["tank_zone"]
    return states


def build_state_timeline(
    schedules: pd.DataFrame,
) -> dict[tuple[int, str], dict[str, dict[str, Any]]]:
    timeline: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    work = schedules.copy()
    work["gameday"] = work["gameday"].astype(str)
    for season, season_games in work.groupby("season"):
        season_games = season_games.sort_values("gameday")
        total_games: dict[str, int] = dict.fromkeys(DIVISIONS, 0)
        for team in DIVISIONS:
            mask = (season_games["home_team"] == team) | (season_games["away_team"] == team)
            total_games[team] = int(mask.sum())
        tallies: dict[str, tuple[int, int, int]] = dict.fromkeys(DIVISIONS, (0, 0, 0))
        played_through: dict[str, int] = dict.fromkeys(DIVISIONS, 0)
        for date, day_games in season_games.groupby("gameday", sort=True):
            timeline[(int(season), date)] = compute_day_states(
                int(season),
                tallies,
                total_games,
                played_through,
                set(day_games["home_team"]) | set(day_games["away_team"]),
            )
            for _, game in day_games.iterrows():
                home, away = game["home_team"], game["away_team"]
                result = game.get("result")
                if pd.isna(result):
                    continue
                result = float(result)
                if result > 0:
                    tallies[home] = (
                        tallies[home][0] + 1,
                        tallies[home][1],
                        tallies[home][2],
                    )
                    tallies[away] = (
                        tallies[away][0],
                        tallies[away][1] + 1,
                        tallies[away][2],
                    )
                elif result < 0:
                    tallies[away] = (
                        tallies[away][0] + 1,
                        tallies[away][1],
                        tallies[away][2],
                    )
                    tallies[home] = (
                        tallies[home][0],
                        tallies[home][1] + 1,
                        tallies[home][2],
                    )
                else:
                    tallies[home] = (
                        tallies[home][0],
                        tallies[home][1],
                        tallies[home][2] + 1,
                    )
                    tallies[away] = (
                        tallies[away][0],
                        tallies[away][1],
                        tallies[away][2] + 1,
                    )
                played_through[home] += 1
                played_through[away] += 1
    return timeline


def attach_states(
    long_df: pd.DataFrame, timeline: dict[tuple[int, str], dict[str, dict[str, Any]]]
) -> pd.DataFrame:
    long_df = long_df.copy()
    long_df["gameday"] = long_df["gameday"].astype(str)
    keys = list(
        zip(
            long_df["season"].astype(int),
            long_df["gameday"],
            long_df["team"],
            strict=True,
        )
    )
    opp_keys = list(
        zip(
            long_df["season"].astype(int),
            long_df["gameday"],
            long_df["opponent"],
            strict=True,
        )
    )
    for prefix, klist in (("st_", keys), ("opp_", opp_keys)):
        for field in (
            "eliminated",
            "clinched",
            "locked_seed",
            "tank_zone",
            "fighter",
            "nothing_to_play_for",
            "conference_rank",
            "cut_line",
            "wins",
            "games_remaining",
        ):
            values = []
            for season, date, team in klist:
                snap = timeline.get((season, date), {}).get(team)
                values.append(np.nan if snap is None else snap[field])
            long_df[f"{prefix}{field}"] = values
    return long_df


def build_long_table(schedules: pd.DataFrame) -> pd.DataFrame:
    sides = []
    for is_home in (True, False):
        team_col = "home_team" if is_home else "away_team"
        opp_col = "away_team" if is_home else "home_team"
        side = pd.DataFrame(
            {
                "game_id": schedules["game_id"],
                "season": schedules["season"],
                "week": schedules["week"],
                "gameday": schedules["gameday"],
                "week_block": schedules["week_block"],
                "team": schedules[team_col],
                "opponent": schedules[opp_col],
                "is_home": is_home,
                "team_covered": (
                    schedules["home_cover"] if is_home else 1.0 - schedules["home_cover"]
                ),
            }
        )
        sides.append(side)
    return pd.concat(sides, ignore_index=True).reset_index(drop=True)


def validate_known_examples(timeline: dict[tuple[int, str], dict[str, dict[str, Any]]]) -> None:
    def state(season: int, date: str, team: str) -> dict[str, Any]:
        return timeline[(season, date)][team]

    jax_dates = [d for (s, d) in timeline if s == 2020 and d >= "2020-12-13"]
    assert jax_dates, "no week-14+ December 2020 dates found"
    for date in sorted(jax_dates):
        st = state(2020, date, "JAX")
        assert st["eliminated"] and st["tank_zone"], f"JAX 2020 {date}: {st}"

    kc_dates = [d for (s, d) in timeline if s == 2020 and d >= "2021-01-01"]
    assert kc_dates, "no January 2021 dates found"
    kc_st = state(2020, sorted(kc_dates)[0], "KC")
    assert kc_st["clinched"] and kc_st["locked_seed"], f"KC week-17 2020: {kc_st}"

    car_dates = [d for (s, d) in timeline if s == 2023 and "2023-12-17" <= d <= "2023-12-31"]
    assert car_dates, "no late-December 2023 dates found"
    for date in sorted(car_dates):
        st = state(2023, date, "CAR")
        assert st["eliminated"], f"CAR 2023 {date}: {st}"


def block_bootstrap_two_group(
    df: pd.DataFrame,
    *,
    flag_col: str,
    value_col: str,
    block_col: str,
    samples: int,
    seed: int,
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
    dropped = samples - len(draws)
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
        "dropped_draws": int(dropped),
        "insufficient_data": False,
    }


def score_cell(
    df: pd.DataFrame,
    name: str,
    *,
    flag: pd.Series,
    value_col: str,
    sign: int,
    description: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    flag = flag.fillna(False).astype(bool)
    week_blocked = summarize(
        df,
        flag=flag,
        value_col=value_col,
        block_col="week_block",
        sign=sign,
        samples=samples,
        seed=seed,
    )
    season_blocked = summarize(
        df,
        flag=flag,
        value_col=value_col,
        block_col="season",
        sign=sign,
        samples=samples,
        seed=seed,
    )
    return {
        "name": name,
        "description": description,
        "sign_dir": sign,
        "n_flag": int(flag.sum()),
        "week_blocked": week_blocked,
        "season_blocked_secondary": season_blocked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    started = time.time()
    schedules_path = args.schedules or _latest_schedules()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "motivation_ladder_screen" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {schedules_path} ===")
    schedules = load_schedules(schedules_path)
    print(
        f"REG {SEASON_START}-{SEASON_END} close-graded games: {len(schedules)} "
        f"(pushes/missing dropped: {schedules.attrs['pushes_or_missing']})"
    )

    print("=== reconstructing standings timeline (completed games only) ===")
    timeline = build_state_timeline(schedules)
    print(f"state snapshots: {len(timeline)} (season, date) pairs")

    print("=== validating known historical examples ===")
    validate_known_examples(timeline)
    print(
        "validation passed: JAX-2020 Dec eliminated+tank; KC-2020 wk17 "
        "clinched+locked; CAR-2023 mid-Dec eliminated"
    )

    long_df = build_long_table(schedules)
    long_df = attach_states(long_df, timeline)
    population = long_df.loc[
        long_df["week"].between(WEEK_MIN, WEEK_MAX) & long_df["st_eliminated"].notna()
    ].reset_index(drop=True)
    print(f"population team-games weeks {WEEK_MIN}-{WEEK_MAX}: {len(population)}")

    cells: list[dict[str, Any]] = []

    cells.append(
        score_cell(
            population,
            "elim_visitor_alive_host",
            flag=(
                (~population["is_home"])
                & population["st_eliminated"].astype(bool)
                & (~population["opp_eliminated"].astype(bool))
            ),
            value_col="team_covered",
            sign=-1,
            description=(
                "ELIMINATED visitor vs alive host, weeks 13-18. Predicted NEGATIVE on "
                "the visitor's team_covered (docs/motivation_ladder_screen.md M1)."
            ),
            samples=args.samples,
            seed=args.seed,
        )
    )

    cells.append(
        score_cell(
            population,
            "locked_seed_wk16_18",
            flag=(
                population["st_locked_seed"].fillna(False).astype(bool)
                & population["week"].between(16, 18)
            ),
            value_col="team_covered",
            sign=-1,
            description=(
                "CLINCHED-and-cannot-improve-seed team, weeks 16-18 only inside the flag. "
                "Predicted NEGATIVE on team_covered (rest incentive; "
                "docs/motivation_ladder_screen.md M2)."
            ),
            samples=args.samples,
            seed=args.seed,
        )
    )

    cells.append(
        score_cell(
            population,
            "fighter_vs_nothing",
            flag=(
                population["st_fighter"].fillna(False).astype(bool)
                & population["opp_nothing_to_play_for"].fillna(False).astype(bool)
            ),
            value_col="team_covered",
            sign=1,
            description=(
                "Fighting-for-last-playoff-spot team vs an opponent with nothing to play "
                "for (eliminated, NOT tank zone). Predicted POSITIVE on the fighter's "
                "team_covered (docs/motivation_ladder_screen.md M3)."
            ),
            samples=args.samples,
            seed=args.seed,
        )
    )

    cells.append(
        score_cell(
            population,
            "tank_zone_wk14_18",
            flag=(
                population["st_tank_zone"].fillna(False).astype(bool)
                & population["week"].between(14, 18)
            ),
            value_col="team_covered",
            sign=-1,
            description=(
                "#1-overall-pick tank zone (bottom two league-wide records), weeks 14-18 "
                "only inside the flag. Predicted NEGATIVE on team_covered "
                "(docs/motivation_ladder_screen.md M4)."
            ),
            samples=args.samples,
            seed=args.seed,
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
            f"  n_flag={cell['n_flag']} n_total={wb['n_total']} frac={wb['fraction_of_slate']:.4f}"
        )
        print(
            f"  full_slate_effect={wb['full_slate_effect_pts']:+.4f}pts week-blocked 95% "
            f"[{wb['ci95_scaled'][0]:+.4f}, {wb['ci95_scaled'][1]:+.4f}] "
            f"P+={wb['probability_positive']:.4f}"
        )
        if not sb.get("insufficient_data"):
            print(
                f"  [season-blocked secondary] effect={sb['full_slate_effect_pts']:+.4f}pts "
                f"95% [{sb['ci95_scaled'][0]:+.4f}, {sb['ci95_scaled'][1]:+.4f}] "
                f"P+={sb['probability_positive']:.4f}"
            )

    configuration = {
        "command": "motivation-ladder-screen",
        "schedules": str(schedules_path),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "population_weeks": [WEEK_MIN, WEEK_MAX],
        "playoff_slots": {"2009_2019": 6, "2020_plus": 7},
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "n_reg_games": len(schedules),
        "n_population_team_games": len(population),
        "n_state_snapshots": len(timeline),
        "predeclaration": (
            "docs/motivation_ladder_screen.md (frozen before this script scored anything)"
        ),
        "point_in_time_note": (
            "states use strictly earlier gameday values only; elimination/clinching are "
            "approximations without tiebreakers, disclosed in the predeclaration"
        ),
        "results": cells,
        "provenance": artifact_provenance(configuration, schedules_path, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="motivation-ladder-screen",
        metrics=payload,
        notes=(
            "Late-season motivation-state ladder battery (4 predeclared cells); every cell "
            "recorded regardless of sign, per AGENTS.md binding taxonomy."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
