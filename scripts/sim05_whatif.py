from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import sim04_engine as eng

ARTIFACT_ROOT = REPO / "artifacts" / "sim05_whatif"
SEASONS = tuple(range(2009, 2026))
QB_SEASONS = tuple(range(2015, 2026))
LINES = (-3.0, -7.0, 3.0)
KEY_NUMBERS = (3, 7, 10, 14, 17)
N_GAMES = 20000
BASE_SEED = 20260925
MIN_PLAYS_PER_TEAM_SEASON_ARM = 100
N_BOOT = 2000


def cover_push_probs(margins: np.ndarray, line: float) -> dict:
    adj = margins + line
    cover = float(np.mean(adj > 0))
    push = float(np.mean(adj == 0))
    loss = float(np.mean(adj < 0))
    return {"cover": cover, "push": push, "loss": loss}


def se_prop(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / n))


def summarize_arm(margins: np.ndarray) -> dict:
    n = len(margins)
    out = {
        "n_games": n,
        "mean_margin": float(np.mean(margins)),
        "mean_margin_se": float(np.std(margins, ddof=1) / np.sqrt(n)),
        "home_win_prob": float(np.mean(margins > 0)),
        "home_win_prob_se": se_prop(float(np.mean(margins > 0)), n),
        "key_number_mass": {},
        "lines": {},
    }
    for k in KEY_NUMBERS:
        p = float(np.mean(np.abs(margins) == k))
        out["key_number_mass"][str(k)] = {"mass": p, "se": se_prop(p, n)}
    for line in LINES:
        probs = cover_push_probs(margins, line)
        out["lines"][str(line)] = {
            "cover_prob": probs["cover"],
            "cover_se": se_prop(probs["cover"], n),
            "push_prob": probs["push"],
            "push_se": se_prop(probs["push"], n),
        }
    return out


def make_fourth_down_policy(tables: dict, pool_rng: np.random.Generator):
    arrays = tables["arrays"]
    pool_mask = (
        (arrays["down_i"] == 4)
        & (arrays["dist_raw"] <= 3)
        & (arrays["fp_raw"] <= 50)
        & np.isin(arrays["play_type_code"], (0, 1))
    )
    pool_idx = np.flatnonzero(pool_mask)

    def policy(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        if qtr >= 5:
            return drawn
        half_remaining = clock_val if clock_val <= 1800.0 else clock_val - 1800.0
        if not (
            down == 4
            and distance <= 3.0
            and yardline <= 50.0
            and half_remaining > 120.0
            and drawn["play_type_code"] in (2, 3)
        ):
            return drawn
        pick = int(pool_idx[pool_rng.integers(len(pool_idx))])
        return {
            "points_off": arrays["points_off"][pick],
            "points_def": arrays["points_def"][pick],
            "clock_elapsed": arrays["clock_elapsed"][pick],
            "flip": bool(arrays["possession_flip"][pick]),
            "next_down": arrays["next_down"][pick],
            "next_distance": arrays["next_distance"][pick],
            "next_yardline": arrays["next_yardline"][pick],
            "yards_gained": arrays["yards_gained"][pick],
            "dist_gained": arrays["dist_gained"][pick],
            "auto_first": bool(arrays["auto_first"][pick]),
            "repeat_down": bool(arrays["repeat_down"][pick]),
            "off_to_used": arrays["off_to_used"][pick],
            "def_to_used": arrays["def_to_used"][pick],
            "play_type_code": arrays["play_type_code"][pick],
        }

    return policy


def run_pair(tables, home_ratings_base, away_ratings, n_games, seed, policy_whatif):
    rng_base = np.random.default_rng(seed)
    frame_base = eng.simulate(
        n_games, rng_base, tables, ot_seconds=600.0, home_ratings=home_ratings_base, away_ratings=away_ratings
    )
    rng_wi = np.random.default_rng(seed)
    frame_wi = eng.simulate(
        n_games,
        rng_wi,
        tables,
        ot_seconds=600.0,
        policy=policy_whatif,
        home_ratings=home_ratings_base,
        away_ratings=away_ratings,
    )
    return frame_base["margin"].to_numpy(), frame_wi["margin"].to_numpy()


def run_ratings_pair(tables, home_ratings_a, home_ratings_b, away_ratings, n_games, seed):
    rng_a = np.random.default_rng(seed)
    frame_a = eng.simulate(
        n_games, rng_a, tables, ot_seconds=600.0, home_ratings=home_ratings_a, away_ratings=away_ratings
    )
    rng_b = np.random.default_rng(seed)
    frame_b = eng.simulate(
        n_games, rng_b, tables, ot_seconds=600.0, home_ratings=home_ratings_b, away_ratings=away_ratings
    )
    return frame_a["margin"].to_numpy(), frame_b["margin"].to_numpy()


def diff_summary(base_margins, wi_margins, keys):
    n = len(base_margins)
    out = {}
    mean_diff = float(np.mean(wi_margins) - np.mean(base_margins))
    se_diff = float(np.std(wi_margins - base_margins, ddof=1) / np.sqrt(n))
    out["mean_margin_shift"] = mean_diff
    out["mean_margin_shift_se"] = se_diff
    out["home_win_prob_shift"] = float(np.mean(wi_margins > 0) - np.mean(base_margins > 0))
    out["key_number_mass_shift"] = {}
    for k in keys:
        pb = float(np.mean(np.abs(base_margins) == k))
        pw = float(np.mean(np.abs(wi_margins) == k))
        out["key_number_mass_shift"][str(k)] = {
            "baseline": pb,
            "whatif": pw,
            "shift": pw - pb,
            "shift_se": float(np.sqrt(se_prop(pb, n) ** 2 + se_prop(pw, n) ** 2)),
        }
    out["line_shift"] = {}
    for line in LINES:
        pb = cover_push_probs(base_margins, line)
        pw = cover_push_probs(wi_margins, line)
        out["line_shift"][str(line)] = {
            "cover_baseline": pb["cover"],
            "cover_whatif": pw["cover"],
            "cover_shift": pw["cover"] - pb["cover"],
            "push_baseline": pb["push"],
            "push_whatif": pw["push"],
            "push_shift": pw["push"] - pb["push"],
        }
    return out


def load_qb_pbp(seasons):
    frames = []
    for season in seasons:
        path = eng.PBP_SNAPSHOT_DIR / f"season={season}" / "plays.parquet"
        cols = [
            "game_id",
            "season",
            "week",
            "season_type",
            "posteam",
            "play_type",
            "epa",
            "passer_player_id",
        ]
        frame = pd.read_parquet(path, columns=cols)
        frame = frame[frame["season_type"] == "REG"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def derive_qb_out_drop(seasons) -> dict:
    pbp = load_qb_pbp(seasons)
    off_plays = pbp[pbp["play_type"].isin(("run", "pass")) & pbp["epa"].notna()].copy()

    pass_att = pbp[(pbp["play_type"] == "pass") & pbp["passer_player_id"].notna()].copy()

    season_att = pass_att.groupby(["posteam", "season", "passer_player_id"], as_index=False)["epa"].count()
    season_att = season_att.rename(columns={"epa": "attempts"})
    season_primary = (
        season_att.sort_values("attempts", ascending=False)
        .groupby(["posteam", "season"], as_index=False)
        .first()[["posteam", "season", "passer_player_id"]]
        .rename(columns={"passer_player_id": "season_primary_passer"})
    )

    game_att = pass_att.groupby(["game_id", "posteam", "passer_player_id"], as_index=False)["epa"].count()
    game_att = game_att.rename(columns={"epa": "attempts"})
    game_leader = (
        game_att.sort_values("attempts", ascending=False)
        .groupby(["game_id", "posteam"], as_index=False)
        .first()[["game_id", "posteam", "passer_player_id"]]
        .rename(columns={"passer_player_id": "game_leading_passer"})
    )

    game_meta = off_plays[["game_id", "posteam", "season"]].drop_duplicates()
    game_leader = game_leader.merge(game_meta, on=["game_id", "posteam"], how="left")
    game_leader = game_leader.merge(season_primary, on=["posteam", "season"], how="left")
    game_leader["backup_led"] = (
        game_leader["game_leading_passer"].notna()
        & game_leader["season_primary_passer"].notna()
        & (game_leader["game_leading_passer"] != game_leader["season_primary_passer"])
    )

    game_epa = off_plays.groupby(["game_id", "posteam"], as_index=False).agg(
        mean_epa=("epa", "mean"), n_plays=("epa", "count")
    )
    game_epa = game_epa.merge(game_leader[["game_id", "posteam", "season", "backup_led"]], on=["game_id", "posteam"], how="inner")

    rows = []
    for (team, season), grp in game_epa.groupby(["posteam", "season"]):
        backup = grp[grp["backup_led"]]
        primary = grp[~grp["backup_led"]]
        n_backup_plays = int(backup["n_plays"].sum())
        n_primary_plays = int(primary["n_plays"].sum())
        if n_backup_plays == 0 or n_primary_plays == 0:
            continue
        backup_epa = float((backup["mean_epa"] * backup["n_plays"]).sum() / n_backup_plays)
        primary_epa = float((primary["mean_epa"] * primary["n_plays"]).sum() / n_primary_plays)
        rows.append(
            {
                "team": team,
                "season": int(season),
                "n_backup_plays": n_backup_plays,
                "n_primary_plays": n_primary_plays,
                "backup_epa_per_play": backup_epa,
                "primary_epa_per_play": primary_epa,
                "diff": backup_epa - primary_epa,
                "weight": min(n_backup_plays, n_primary_plays),
            }
        )
    ts = pd.DataFrame(rows)
    ts = ts[(ts["n_backup_plays"] >= MIN_PLAYS_PER_TEAM_SEASON_ARM) & (ts["n_primary_plays"] >= MIN_PLAYS_PER_TEAM_SEASON_ARM)]

    weighted_mean_diff = float((ts["diff"] * ts["weight"]).sum() / ts["weight"].sum())

    rng = np.random.default_rng(BASE_SEED)
    idx = np.arange(len(ts))
    boot = []
    diffs = ts["diff"].to_numpy()
    weights = ts["weight"].to_numpy()
    for _ in range(N_BOOT):
        sample = rng.choice(idx, size=len(idx), replace=True)
        d = diffs[sample]
        w = weights[sample]
        boot.append(float((d * w).sum() / w.sum()))
    boot = np.array(boot)

    return {
        "n_team_seasons": int(len(ts)),
        "n_backup_plays_total": int(ts["n_backup_plays"].sum()),
        "n_primary_plays_total": int(ts["n_primary_plays"].sum()),
        "mean_epa_drop_per_play": weighted_mean_diff,
        "mean_epa_drop_ci_p05": float(np.percentile(boot, 5)),
        "mean_epa_drop_ci_p95": float(np.percentile(boot, 95)),
        "note": "diff = backup-led games offensive EPA/play minus same team-season's primary-passer games offensive EPA/play, weighted by min(n_backup_plays, n_primary_plays) per team-season, bootstrapped over team-seasons",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=N_GAMES)
    args = parser.parse_args()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = eng.build_tables(SEASONS, condition_on_team=True)
    league_off = tables["league_off_mean"]
    league_def = tables["league_def_mean"]
    avg_ratings = {"off": league_off, "def": league_def}

    pool_rng = np.random.default_rng(BASE_SEED + 1)
    fourth_down_policy = make_fourth_down_policy(tables, pool_rng)
    base1, wi1 = run_pair(tables, avg_ratings, avg_ratings, args.n_games, BASE_SEED + 100, fourth_down_policy)
    whatif1 = {
        "name": "fourth_down_aggression",
        "description": "On 4th-and-3-or-less at or past midfield, outside the final two minutes of a half, a real punt or field-goal-attempt draw is replaced with a real go-for-it outcome drawn from historical plays in that same situation.",
        "n_go_for_it_pool_rows": int(
            (
                (tables["arrays"]["down_i"] == 4)
                & (tables["arrays"]["dist_raw"] <= 3)
                & (tables["arrays"]["fp_raw"] <= 50)
                & np.isin(tables["arrays"]["play_type_code"], (0, 1))
            ).sum()
        ),
        "baseline": summarize_arm(base1),
        "whatif": summarize_arm(wi1),
        "shift": diff_summary(base1, wi1, KEY_NUMBERS),
        "plain_english": None,
    }
    shift_mean = whatif1["shift"]["mean_margin_shift"]
    shift_se = whatif1["shift"]["mean_margin_shift_se"]
    hw_shift = whatif1["shift"]["home_win_prob_shift"]
    whatif1["plain_english"] = (
        f"Simulated: if teams always went for it on short 4th downs past midfield instead of punting or "
        f"kicking, the home team's average margin would move by {shift_mean:+.2f} points "
        f"(give or take about {1.96 * shift_se:.2f}), and home win chance would shift by {hw_shift:+.1%}. "
        f"This is a research simulation, not a pick change."
    )

    qb_drop = derive_qb_out_drop(QB_SEASONS)
    drop = qb_drop["mean_epa_drop_per_play"]
    home_ratings_qb_out = {"off": league_off + drop, "def": league_def}
    base2b, wi2 = run_ratings_pair(tables, avg_ratings, home_ratings_qb_out, avg_ratings, args.n_games, BASE_SEED + 200)
    whatif2 = {
        "name": "starting_qb_out",
        "description": "Home offense EPA/play rating is lowered by the measured backup-passer drop, holding the away team at league-average; compared against both teams at league average.",
        "qb_drop_derivation": qb_drop,
        "baseline": summarize_arm(base2b),
        "whatif": summarize_arm(wi2),
        "shift": diff_summary(base2b, wi2, KEY_NUMBERS),
        "plain_english": None,
    }
    shift_mean2 = whatif2["shift"]["mean_margin_shift"]
    shift_se2 = whatif2["shift"]["mean_margin_shift_se"]
    hw_shift2 = whatif2["shift"]["home_win_prob_shift"]
    whatif2["plain_english"] = (
        f"Simulated: if the home team's normal starting quarterback sits out and the backup plays like the "
        f"typical backup does (about {drop:+.3f} EPA per play worse on offense), the home team's average "
        f"margin would move by {shift_mean2:+.2f} points (give or take about {1.96 * shift_se2:.2f}), and home "
        f"win chance would shift by {hw_shift2:+.1%}. This is a research simulation, not a pick change."
    )

    report = {
        "generated_at": timestamp,
        "engine": "scripts/sim04_engine.py",
        "whatif_script": "scripts/sim05_whatif.py",
        "seasons_for_tables": list(SEASONS),
        "n_games_per_arm": args.n_games,
        "shared_seed_scheme": "each baseline/whatif pair uses np.random.default_rng(seed) with the identical seed, so per-play draws stay aligned (common random numbers) until a policy intervention changes the trajectory; the 4th-down replacement draw uses a separate rng stream so it never desyncs the shared draw sequence.",
        "league_average_ratings": {"off_epa_per_play": league_off, "def_epa_per_play": league_def},
        "labels": "every number below is simulated/inferred; it is a research and dashboard artifact and never re-picks a card game",
        "fourth_down_aggression_whatif": whatif1,
        "starting_qb_out_whatif": whatif2,
    }

    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps({"out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
