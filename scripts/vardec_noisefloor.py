"""Variance-decomposition noise floor (SIM-02 lite).

Question: what is the minimum achievable outcome-margin variance given perfect
team-strength knowledge? Method: an empirical resampling simulator holds each
team's offensive and defensive per-play distributions FIXED at season-level
estimates (no within-season drift), simulates full REG games by sampling plays
WITHOUT replacement from the possessing team's observed play pool (drive
structure approximated: alternating possessions, chain logic, league field-goal
model, punts/turnovers), then calibrates a single yardage-scale parameter until
the simulated margin sd matches the real margin sd within 0.5 points. ABLATION:
every team-unit pool is replaced by its own mean-EPA-equivalent single play
(mean yards_gained, mean turnover rate), which removes play-level execution
noise while keeping each unit's central tendency; the resulting sd across the
real schedule is the pure scheduling/matchup floor.

Execution-noise share = 1 - (floor_sd / calibrated_sd)**2.

Limitations (declared): defense enters only through a half-weight additive
per-play yardage adjustment from yards allowed per play; field goals use a
league-level distance model rather than team kickers; no overtime; possession
starts fixed at the 25; two-point attempts not modelled. The ablated world
keeps turnover SKILL as a per-play probability but applies it in expectation
(geometric survival), so no Bernoulli execution noise survives into the floor.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

SEASON_START = 2021
SEASON_END = 2025
HFA_POINTS = 2.0
DRIVES_PER_TEAM = 11
MAX_DRIVE_PLAYS = 20
FG_RANGE_YARDLINE = 37.0
DEF_WEIGHT = 0.5
START_YARDLINE = 25.0
TD_POINTS = 7.0
FG_POINTS = 3.0
KNEEL_LEAD_MIN = 9.0
KNEEL_DRIVES_LEFT = 2
PUNT_NET = 38.0
PEN_RATE_PER_PLAY = 0.0872
REAL_SERIES_CONVERSION = 0.6616
OFF_PLAY_TYPES = ("pass", "run")

REPS_SEARCH = 16
REPS_BISECT = 40
REPS_FINAL = 96
REPS_FLOOR = 192
SEED = 20260822


def latest_snapshot(root: Path) -> Path:
    candidates = sorted(p for p in root.glob("*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no snapshot under {root}")
    return candidates[-1]


def _pool_frame(g: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "n_plays": len(g),
            "mean_yards": float(g["yards_gained"].mean()),
            "td_rate": float(g["touchdown"].fillna(0).astype(float).mean()),
            "to_rate": float(
                (g["interception"].fillna(0).astype(bool) | g["fumble_lost"].fillna(0).astype(bool))
                .astype(float)
                .mean()
            ),
            "mean_epa": float(g["epa"].mean()),
        }
    )


def _fg_table(plays: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fg = plays.loc[plays["play_type"] == "field_goal"].copy()
    fg["kick_dist"] = fg["yardline_100"] + 17.0
    fg["made"] = (fg["posteam_score_post"].fillna(0) - fg["posteam_score"].fillna(0)) == 3
    edges = [17, 25, 30, 35, 40, 45, 50, 55, 62]
    table: list[float] = []
    for lo, hi in itertools.pairwise(edges):
        cell = fg[(fg["kick_dist"] >= lo) & (fg["kick_dist"] < hi)]
        table.append(float(cell["made"].mean()) if len(cell) >= 50 else np.nan)
    valid = [v for v in table if np.isfinite(v)]
    fallback = float(np.mean(valid)) if valid else 0.85
    table = [v if np.isfinite(v) else fallback for v in table]
    return np.asarray(edges[:-1], dtype=np.float64), np.asarray(table, dtype=np.float64)


def load_inputs(
    season_start: int, season_end: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    snap = latest_snapshot(REPO / "data/pbp/raw")
    frames = []
    for season in range(season_start, season_end + 1):
        path = snap / f"season={season}" / "plays.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing pbp season file {path}")
        frames.append(pd.read_parquet(path))
    plays = pd.concat(frames, ignore_index=True)
    keep = plays.loc[
        (plays["season_type"] == "REG")
        & plays["play_type"].isin(OFF_PLAY_TYPES)
        & ~plays["qb_kneel"].fillna(0).astype(bool)
        & ~plays["qb_spike"].fillna(0).astype(bool)
        & ~plays["aborted_play"].fillna(0).astype(bool)
        & plays["posteam"].notna()
        & plays["yards_gained"].notna()
    ].copy()
    scored = plays.loc[plays["season_type"] == "REG", "game_id"]
    n_reg_games = int(scored.nunique())
    nonoff_mask = (plays["touchdown"].fillna(0) > 0) & (
        plays["play_type"].isin(["kickoff", "punt"])
        | (plays["interception"].fillna(0) > 0)
        | (plays["fumble_lost"].fillna(0) > 0)
    )
    nonoff_rate = float(nonoff_mask.sum()) / max(n_reg_games, 1)

    off_snaps = int((plays["season_type"] == "REG").sum() and keep.shape[0])
    reg_plays = plays.loc[plays["season_type"] == "REG"]
    pen_rows = reg_plays.loc[
        (reg_plays["penalty"].fillna(0) > 0)
        & (reg_plays["play_type"].isin(["no_play", "pass", "run"]))
        & (reg_plays["penalty_yards"].fillna(0) > 0),
        "penalty_yards",
    ].to_numpy(np.float64)
    pen_rate = len(pen_rows) / max(off_snaps, 1)

    pools = (
        keep.groupby(["season", "posteam"], sort=True)
        .apply(_pool_frame, include_groups=False)
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    deff = (
        keep.groupby(["season", "defteam"])
        .agg(ypp_allowed=("yards_gained", "mean"), n_def_plays=("yards_gained", "size"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    edges, rates = _fg_table(plays)

    sched_path = sorted((REPO / "data/raw").glob("*/schedules.parquet"))[-1]
    sched = pd.read_parquet(sched_path)
    games = sched.loc[
        (sched["game_type"] == "REG")
        & sched["result"].notna()
        & sched["season"].between(season_start, season_end),
        [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "result",
            "home_score",
            "away_score",
        ],
    ].reset_index(drop=True)

    drive_reg = plays.loc[(plays["season_type"] == "REG") & plays["fixed_drive"].notna()]
    sched_h = games.set_index("game_id")["home_team"]
    drive_counts = drive_reg.groupby(["game_id", "posteam"])["fixed_drive"].nunique().unstack()
    pairs: list[tuple[int, int]] = []
    for gid, row in drive_counts.iterrows():
        if gid not in sched_h.index:
            continue
        ht = sched_h[gid]
        others = [t for t in row.dropna().index if t != ht]
        if len(others) != 1:
            continue
        pairs.append((int(row[ht]), int(row[others[0]])))
    drive_pairs = np.asarray(pairs, dtype=np.float64)

    extras = {
        "lg_ypp": float(keep["yards_gained"].mean()),
        "fg_edges": edges,
        "fg_rates": rates,
        "nonoff_td_rate": nonoff_rate,
        "pen_pool": pen_rows,
        "pen_rate": pen_rate,
        "drive_pairs": drive_pairs,
        "pbp_snapshot": str(snap),
        "schedules_path": str(sched_path),
    }
    return pools, deff, extras, games


class Engine:
    def __init__(
        self,
        pools: pd.DataFrame,
        deff: pd.DataFrame,
        extras: dict[str, Any],
        games: pd.DataFrame,
        play_values: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        self.pools = pools.reset_index(drop=True)
        self.pool_id: dict[tuple[int, str], int] = {
            (int(s), str(t)): i
            for i, (s, t) in enumerate(zip(self.pools["season"], self.pools["team"], strict=True))
        }
        self.mean_yards = self.pools["mean_yards"].to_numpy(np.float64)
        self.td_rate = self.pools["td_rate"].to_numpy(np.float64)
        self.to_rate = self.pools["to_rate"].to_numpy(np.float64)
        self.mean_epa = self.pools["mean_epa"].to_numpy(np.float64)
        self.n_plays = self.pools["n_plays"].to_numpy(np.int64)
        self.lg_ypp = float(extras["lg_ypp"])
        self.fg_edges = np.asarray(extras["fg_edges"], dtype=np.float64)
        self.fg_rates = np.asarray(extras["fg_rates"], dtype=np.float64)

        def_ypp = {
            (int(s), str(t)): float(y)
            for s, t, y in zip(deff["season"], deff["team"], deff["ypp_allowed"], strict=True)
        }
        self.def_adj = np.zeros(len(self.pools))
        for i, (s, t) in enumerate(zip(self.pools["season"], self.pools["team"], strict=True)):
            y = def_ypp.get((int(s), str(t)))
            self.def_adj[i] = DEF_WEIGHT * (self.lg_ypp - y) if y is not None else 0.0

        self.games = games.reset_index(drop=True)
        self.g = len(self.games)
        self.home_pid = np.array(
            [
                self.pool_id[(int(s), str(t))]
                for s, t in zip(self.games["season"], self.games["home_team"], strict=True)
            ],
            dtype=np.int64,
        )
        self.away_pid = np.array(
            [
                self.pool_id[(int(s), str(t))]
                for s, t in zip(self.games["season"], self.games["away_team"], strict=True)
            ],
            dtype=np.int64,
        )

        self.yards_all, self.td_all, self.to_all = play_values
        self.maxlen = int(self.n_plays.max())
        self.min_pool = int(self.n_plays.min())
        self.misc_rate = float(extras["nonoff_td_rate"])
        self.drive_pairs = np.asarray(extras["drive_pairs"], dtype=np.float64)
        self.pen_pool = np.asarray(extras["pen_pool"], dtype=np.float64)
        self.pen_rate = float(extras["pen_rate"])
        self.series_drag = 0.0
        self.yard_mean_all = np.repeat(self.mean_yards, self.n_plays)

    @classmethod
    def from_plays(
        cls,
        pools: pd.DataFrame,
        deff: pd.DataFrame,
        extras: dict[str, Any],
        games: pd.DataFrame,
        keep: pd.DataFrame,
    ) -> Engine:
        key_to_id = {
            (int(s), str(t)): i
            for i, (s, t) in enumerate(
                zip(
                    pools["season"],
                    pools["posteam"] if "posteam" in pools else pools["team"],
                    strict=True,
                )
            )
        }
        keys = [
            key_to_id[(int(s), str(p))]
            for s, p in zip(keep["season"], keep["posteam"], strict=True)
        ]
        order = np.argsort(np.asarray(keys), kind="stable")
        grouped = keep.iloc[order].reset_index(drop=True)
        group_ids_sorted = np.asarray(keys)[order]
        yards = grouped["yards_gained"].to_numpy(np.float64)
        td = grouped["touchdown"].fillna(0).to_numpy(np.float64)
        to = (
            (
                grouped["interception"].fillna(0).astype(bool)
                | grouped["fumble_lost"].fillna(0).astype(bool)
            )
            .astype(float)
            .to_numpy()
        )
        sizes = pools["n_plays"].to_numpy(np.int64)
        assert np.all(np.bincount(group_ids_sorted) == sizes), "pool grouping mismatch"
        return cls(pools, deff, extras, games, play_values=(yards, td, to))

    def fg_rate(self, kick_dist: np.ndarray) -> np.ndarray:
        idx = np.clip(np.searchsorted(self.fg_edges[1:], kick_dist), 0, len(self.fg_rates) - 1)
        return self.fg_rates[idx]

    def _fresh_deck(self, rng: np.random.Generator) -> np.ndarray:
        deck = np.zeros((len(self.pools), self.maxlen), dtype=np.int64)
        offset = 0
        pad = np.arange(self.maxlen)
        for i, size in enumerate(self.n_plays):
            deck[i, :size] = offset + rng.permutation(size)
            overflow = pad >= size
            deck[i, overflow] = offset + (pad[overflow] % size)
            offset += size
        return deck

    def simulate_stochastic(self, disp: float, reps: int, seed: int) -> np.ndarray:
        out = np.empty(reps * self.g, dtype=np.float64)
        for r in range(reps):
            rng = np.random.default_rng(seed + r)
            deck = self._fresh_deck(rng)
            out[r * self.g : (r + 1) * self.g] = self._sim_one(disp, deck, rng)
        return out

    def _sim_one(self, disp: float, deck: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        g = self.g
        pair_idx = rng.integers(0, len(self.drive_pairs), size=g)
        quota = np.rint(self.drive_pairs[pair_idx].sum(axis=1) / 2.0).astype(np.int32)
        score_h = np.zeros(g)
        score_a = np.zeros(g)
        poss_home = np.ones(g, dtype=bool)
        yl = np.full(g, 100.0 - START_YARDLINE)
        down = np.ones(g, dtype=np.int8)
        dist = np.full(g, 10.0 + self.series_drag)
        drive_plays = np.zeros(g, dtype=np.int32)
        drives_h = np.zeros(g, dtype=np.int32)
        drives_a = np.zeros(g, dtype=np.int32)
        cur_h = np.zeros(g, dtype=np.int64)
        cur_a = np.zeros(g, dtype=np.int64)
        active = np.ones(g, dtype=bool)
        n_misc = rng.poisson(self.misc_rate, g).astype(np.float64)
        misc_home = rng.random(g) < 0.5
        misc_h = np.where(misc_home, n_misc, 0.0) * TD_POINTS
        misc_a = (n_misc - np.where(misc_home, n_misc, 0.0)) * TD_POINTS
        self._outcomes = {
            "TD": 0,
            "FGmake": 0,
            "FGmiss": 0,
            "TO": 0,
            "PUNT": 0,
            "KNEEL": 0,
            "PEN_PLAYS": 0,
        }
        _oc = self._outcomes
        self._drive_log: list[tuple[float, str]] = []
        _log = self._drive_log
        self._drive_detail: list[tuple[float, int, float, str]] = []
        _det = self._drive_detail
        start_yl = np.full(g, 100.0 - START_YARDLINE)
        cum_yd = np.zeros(g)

        while active.any():
            idx = np.nonzero(active)[0]
            home_off = poss_home[idx]
            side_lead = np.where(home_off, score_h[idx] - score_a[idx], score_a[idx] - score_h[idx])
            side_left = quota[idx] - np.where(home_off, drives_h[idx], drives_a[idx])
            kneel_end = (side_lead >= KNEEL_LEAD_MIN) & (side_left <= KNEEL_DRIVES_LEFT)

            pid = np.where(home_off, self.home_pid[idx], self.away_pid[idx])
            cur = np.where(home_off, cur_h[idx], cur_a[idx])
            card = deck[pid, cur]
            new_cur = cur + 1
            cur_h[idx] = np.where(home_off, new_cur, cur_h[idx])
            cur_a[idx] = np.where(home_off, cur_a[idx], new_cur)

            defender_adj = np.where(
                home_off,
                self.def_adj[self.away_pid[idx]],
                self.def_adj[self.home_pid[idx]],
            )
            y = (
                self.yard_mean_all[card]
                + defender_adj
                + disp * (self.yards_all[card] - self.yard_mean_all[card])
            )
            td_flag = self.td_all[card] > 0.5
            to_flag = self.to_all[card] > 0.5
            if disp < 1.0:
                keep_flag = rng.random(len(idx)) < disp
                td_flag &= keep_flag
                to_flag &= keep_flag

            pen = (~kneel_end) & (rng.random(len(idx)) < self.pen_rate)
            pen_py = np.zeros(len(idx))
            if pen.any():
                pen_py[pen] = self.pen_pool[
                    rng.integers(0, len(self.pen_pool), size=int(pen.sum()))
                ]
                y = np.where(pen, 0.0, y)
                td_flag = td_flag & ~pen
                to_flag = to_flag & ~pen

            yl_new = yl[idx] - y
            scored_td = (~kneel_end) & (td_flag | (yl_new <= 0.0))
            pts = np.where(scored_td, TD_POINTS, 0.0)
            gained_fd = (~kneel_end) & (~scored_td) & (~to_flag) & (y >= dist[idx])
            fourth_fail = (
                (~kneel_end) & (~pen) & (~scored_td) & (~to_flag) & (~gained_fd) & (down[idx] == 4)
            )
            fg_try = fourth_fail & (yl_new <= FG_RANGE_YARDLINE)
            fg_made = np.zeros(len(idx), dtype=bool)
            if fg_try.any():
                u = rng.random(len(idx))
                fg_made = fg_try & (u < self.fg_rate(np.clip(yl_new + 17.0, 18.0, 61.0)))
                pts = np.where(fg_made, FG_POINTS, pts)
            drive_end = (
                kneel_end
                | scored_td
                | to_flag
                | fourth_fail
                | (drive_plays[idx] + 1 >= MAX_DRIVE_PLAYS)
            )
            pts = np.where(kneel_end, 0.0, pts)

            score_h[idx] += pts * drive_end * home_off
            score_a[idx] += pts * drive_end * (~home_off)

            cont_idx = idx[~drive_end]
            if len(cont_idx):
                y_c = y[~drive_end]
                fd_c = gained_fd[~drive_end]
                pen_c = pen[~drive_end]
                yl[cont_idx] = yl_new[~drive_end]
                base_dist = np.maximum(dist[cont_idx] - y_c, 1.0)
                dist[cont_idx] = np.where(
                    fd_c, 10.0, np.minimum(base_dist + pen_c * pen_py[~drive_end], 45.0)
                )
                down[cont_idx] = np.where(
                    fd_c, 1, down[cont_idx] + (~pen_c).astype(np.int8)
                ).astype(np.int8)
                drive_plays[cont_idx] += 1
                cum_yd[cont_idx] += np.maximum(y[~drive_end], 0.0)

            end_idx = idx[drive_end]
            if len(end_idx):
                drives_h[end_idx] += poss_home[end_idx].astype(np.int32)
                drives_a[end_idx] += (~poss_home[end_idx]).astype(np.int32)
                std_e = scored_td[drive_end]
                ff_e = fourth_fail[drive_end]
                fgt_e = fg_try[drive_end]
                fgm_e = fg_made[drive_end]
                to_e = to_flag[drive_end]
                kneel_e = kneel_end[drive_end]
                spot_e = yl_new[drive_end]
                recv = np.full(len(end_idx), 100.0 - START_YARDLINE)
                conv = ff_e & ~fgt_e
                if conv.any():
                    recv[conv] = np.clip(100.0 - spot_e[conv], 1.0, 95.0)
                kicked_away = ff_e & ~fgm_e
                if kicked_away.any():
                    raw = 100.0 - spot_e[kicked_away] - PUNT_NET
                    recv[kicked_away] = np.where(raw <= 0.0, 20.0, np.clip(raw, 1.0, 95.0))
                yl[end_idx] = recv
                for j in range(len(end_idx)):
                    if std_e[j]:
                        o = "TD"
                    elif fgm_e[j]:
                        o = "FGmake"
                    elif ff_e[j]:
                        o = "FGmiss" if fgt_e[j] else "PUNT"
                    elif to_e[j]:
                        o = "TO"
                    elif kneel_e[j]:
                        o = "KNEEL"
                    else:
                        o = "CAP"
                    _log.append((float(start_yl[end_idx[j]]), o))
                    _det.append(
                        (
                            float(start_yl[end_idx[j]]),
                            int(drive_plays[end_idx[j]] + 1),
                            float(cum_yd[end_idx[j]] + max(y[drive_end][j], 0.0)),
                            o,
                        )
                    )
                start_yl[end_idx] = recv
                cum_yd[end_idx] = 0.0
                _oc["TD"] += int(std_e.sum())
                _oc["FGmake"] += int(fgm_e.sum())
                _oc["FGmiss"] += int((ff_e & fgt_e & ~fgm_e).sum())
                _oc["PUNT"] += int((ff_e & ~fgt_e).sum())
                _oc["TO"] += int((to_e & ~std_e).sum())
                _oc["KNEEL"] += int(kneel_e.sum())
                _oc["PEN_PLAYS"] += int(pen.sum())
                poss_home[end_idx] = ~poss_home[end_idx]
                down[end_idx] = 1
                dist[end_idx] = 10.0 + self.series_drag
                drive_plays[end_idx] = 0
                active[end_idx] = ~(
                    (drives_h[end_idx] >= quota[end_idx]) & (drives_a[end_idx] >= quota[end_idx])
                )

        self.last_score_h = score_h + HFA_POINTS + misc_h
        self.last_score_a = score_a + misc_a
        self.last_outcomes = self._outcomes
        return score_h + HFA_POINTS + misc_h - score_a - misc_a

    def ablated_floor(self, reps: int, seed: int, disp: float = 0.0) -> dict[str, Any]:
        """Pure scheduling/matchup floor: run the same engine with every play
        replaced by its pool mean (disp=0, the mean-EPA-equivalent ablation)
        and take the between-matchup sd of per-game mean margins, corrected
        for finite-rep Monte Carlo noise."""
        raw = self.simulate_stochastic(disp, reps, seed).reshape(reps, self.g)
        game_means = raw.mean(axis=0)
        within_var = float(raw.var(axis=0, ddof=1).mean())
        obs_var = float(np.var(game_means, ddof=1))
        floor_var = max(obs_var - within_var / reps, 0.0)
        return {
            "margins": game_means,
            "floor_sd": float(np.sqrt(floor_var)),
            "raw_between_sd": float(np.std(game_means, ddof=1)),
            "mc_noise_sd": float(np.sqrt(within_var / reps)),
            "reps": reps,
        }


def sd(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1))


def calibrate_series_drag(engine: Engine, target: float) -> tuple[float, list[dict[str, Any]]]:
    """Bisect the per-series yardage drag so chain series conversion matches
    the measured real rate. Uses a fast standalone chain Monte Carlo."""
    trail: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED + 77)
    n = 200000

    def conv_at(drag: float) -> float:
        cum = np.zeros(n)
        downs = np.ones(n, dtype=np.int8)
        converted = np.zeros(n, dtype=bool)
        active = np.ones(n, dtype=bool)
        req = 10.0 + drag
        for _ in range(24):
            ai = np.nonzero(active)[0]
            if not len(ai):
                break
            y = engine.yards_all[rng.integers(0, len(engine.yards_all), size=len(ai))]
            pen = rng.random(len(ai)) < engine.pen_rate
            gain = np.where(pen, 0.0, y)
            cum[ai] += gain
            conv = (~pen) & (cum[ai] >= req)
            converted[ai[conv]] = True
            active[ai[conv]] = False
            ci = ai[~conv]
            downs[ci] += 1
            active[ci] &= downs[ci] <= 4
        return float(converted.mean())

    lo, hi = 0.0, 6.0
    v_lo = conv_at(lo)
    trail.append({"drag": lo, "series_conversion": round(v_lo, 4)})
    if v_lo <= target:
        return lo, trail
    for _i in range(10):
        mid = 0.5 * (lo + hi)
        v = conv_at(mid)
        trail.append({"drag": round(mid, 4), "series_conversion": round(v, 4)})
        if abs(v - target) < 0.002:
            return round(mid, 4), trail
        if v > target:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 4), trail


def calibrate(engine: Engine, target_sd: float) -> tuple[float, list[dict[str, Any]]]:
    """Bisect the execution-dispersion dial disp in [0,1]: 0 = mean-equivalent
    play pools (the ablation limit), 1 = full empirical per-play dispersion."""
    trail: list[dict[str, Any]] = []

    def probe(disp: float, reps: int, seed: int) -> float:
        m = engine.simulate_stochastic(disp, reps, seed)
        val = sd(m)
        trail.append({"disp": round(disp, 4), "reps": reps, "sim_margin_sd": val})
        print(f"  disp={disp:.4f} reps={reps} sim margin sd={val:.3f}")
        return val

    v_lo = probe(0.0, REPS_SEARCH, SEED + 1000)
    v_hi = probe(1.0, REPS_SEARCH, SEED + 1001)
    if not (v_lo <= target_sd <= v_hi):
        raise RuntimeError(
            f"target sd {target_sd:.3f} not bracketed by disp extremes ({v_lo:.3f}, {v_hi:.3f})"
        )
    lo, hi = 0.0, 1.0
    disp = 0.5
    for i in range(8):
        v = probe(disp, REPS_BISECT, SEED + 3000 + i)
        if abs(v - target_sd) < 0.10:
            break
        if v < target_sd:
            lo = disp
        else:
            hi = disp
        disp = round(0.5 * (lo + hi), 4)
    return disp, trail


def distribution_stats(margins: np.ndarray) -> dict[str, Any]:
    am = np.abs(margins)
    q = np.quantile(am, [0.1, 0.25, 0.5, 0.75, 0.9])

    def mass(vals: set[int]) -> float:
        return float(np.mean(np.isin(margins.astype(int), list(vals))))

    return {
        "n": len(margins),
        "mean": float(np.mean(margins)),
        "sd": sd(margins),
        "abs_quantiles_10_25_50_75_90": [round(float(v), 2) for v in q],
        "share_margin_in_2_3_6_7": mass({2, 3, -2, -3, 6, -6, 7, -7}),
        "share_abs_eq_3": float(np.mean(np.round(am) == 3)),
        "share_abs_eq_7": float(np.mean(np.round(am) == 7)),
        "share_abs_le_3": float(np.mean(np.round(am) <= 3)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = args.output or (REPO / "artifacts" / "vardec_floor" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== loading pbp pools and schedules ===")
    pools, deff, extras, games = load_inputs(SEASON_START, SEASON_END)
    snap = Path(extras["pbp_snapshot"])
    keep_frames = []
    for season in range(SEASON_START, SEASON_END + 1):
        pf = pd.read_parquet(snap / f"season={season}" / "plays.parquet")
        keep_frames.append(pf)
    plays_all = pd.concat(keep_frames, ignore_index=True)
    keep = plays_all.loc[
        (plays_all["season_type"] == "REG")
        & plays_all["play_type"].isin(OFF_PLAY_TYPES)
        & ~plays_all["qb_kneel"].fillna(0).astype(bool)
        & ~plays_all["qb_spike"].fillna(0).astype(bool)
        & ~plays_all["aborted_play"].fillna(0).astype(bool)
        & plays_all["posteam"].notna()
        & plays_all["yards_gained"].notna()
    ]
    engine = Engine.from_plays(pools, deff, extras, games, keep)

    n_pools = len(pools)
    total_plays = int(pools["n_plays"].sum())
    real_margins = games["result"].to_numpy(dtype=np.float64)
    target_sd = sd(real_margins)
    real_stats = distribution_stats(real_margins)
    print(
        f"REG {SEASON_START}-{SEASON_END}: {len(games)} games, {n_pools} team-season "
        f"offense pools, {total_plays} pool plays, min pool {engine.min_pool}"
    )
    print(f"real margin sd = {target_sd:.3f}, mean = {float(np.mean(real_margins)):.3f}")

    assert engine.min_pool > 4 * DRIVES_PER_TEAM * 8, "pool too small for game length"

    print("=== calibrating series drag to real series-conversion rate ===")
    drag, drag_trail = calibrate_series_drag(engine, REAL_SERIES_CONVERSION)
    engine.series_drag = drag
    print(f"selected series drag = {drag} (target conversion {REAL_SERIES_CONVERSION})")

    print("=== calibrating execution dispersion disp (0=mean pools, 1=full empirical) ===")
    disp, trail = calibrate(engine, target_sd)
    print(f"selected disp = {disp}")

    print(f"=== final run: {REPS_FINAL} reps at disp={disp} ===")
    sim_margins = engine.simulate_stochastic(disp, REPS_FINAL, args.seed)
    calibrated_sd = sd(sim_margins)
    sim_stats = distribution_stats(sim_margins)
    print(f"calibrated sim margin sd = {calibrated_sd:.3f} (target {target_sd:.3f})")

    print("=== ablation: mean-EPA-equivalent pools (disp=0), between-matchup floor ===")
    floor = engine.ablated_floor(REPS_FLOOR, args.seed + 900000)
    floor_margins = floor["margins"]
    floor_sd = float(floor["floor_sd"])
    floor_stats = distribution_stats(floor_margins)
    exec_share = 1.0 - (floor_sd / calibrated_sd) ** 2
    print(
        f"ablated floor sd = {floor_sd:.3f} (raw between {floor['raw_between_sd']:.3f}, "
        f"mc noise {floor['mc_noise_sd']:.3f}); execution-noise share = {exec_share:.4f}"
    )

    totals_real = float((games["home_score"] + games["away_score"]).mean())
    payload = {
        "schema": 1,
        "generated_at_utc": timestamp,
        "population": {
            "seasons": [SEASON_START, SEASON_END],
            "game_type": "REG",
            "n_games": len(games),
            "n_team_season_pools": n_pools,
            "n_pool_plays": total_plays,
            "min_pool_size": engine.min_pool,
        },
        "method": {
            "engine": (
                "empirical resampling: without-replacement play decks, chain logic, "
                "league FG model, resampled real drive-count pairs, Poisson "
                "non-offensive-TD events at measured rate"
            ),
            "yardage_scale_fixed": 1.0,
            "drives_per_team_mean": DRIVES_PER_TEAM,
            "hfa_points": HFA_POINTS,
            "defense_weight": DEF_WEIGHT,
            "max_drive_plays": MAX_DRIVE_PLAYS,
            "calibration_parameter": (
                "execution-dispersion dial disp interpolating each play between "
                "its pool mean (0) and its observed value (1); bisected to match "
                "real margin sd"
            ),
            "series_drag": drag,
            "reps_search": REPS_SEARCH,
            "reps_bisect": REPS_BISECT,
            "reps_final": REPS_FINAL,
            "reps_floor": REPS_FLOOR,
            "seed": args.seed,
        },
        "real": {
            "margin_mean": float(np.mean(real_margins)),
            **real_stats,
            "mean_total_points": totals_real,
        },
        "series_drag_calibration": {
            "target_conversion": REAL_SERIES_CONVERSION,
            "trail": drag_trail,
        },
        "calibration_trail": trail,
        "disp_selected": disp,
        "calibrated": sim_stats,
        "calibration_gap_points": calibrated_sd - target_sd,
        "ablated_floor": floor_stats,
        "ablated_floor_mc": {
            "floor_sd_corrected": floor_sd,
            "raw_between_sd": floor["raw_between_sd"],
            "mc_noise_sd": floor["mc_noise_sd"],
            "reps": REPS_FLOOR,
        },
        "execution_noise_share": exec_share,
        "execution_noise_sd_points": float(np.sqrt(max(calibrated_sd**2 - floor_sd**2, 0.0))),
        "validation": {
            "sd_within_half_point": bool(abs(calibrated_sd - target_sd) <= 0.5),
            "note": "key-number shares and quantiles compare sim vs real margins",
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    configuration = {
        "seasons": f"{SEASON_START}-{SEASON_END}",
        "pbp_snapshot": extras["pbp_snapshot"],
        "schedules_path": extras["schedules_path"],
        "disp": disp,
        "seed": args.seed,
    }
    payload["provenance"] = artifact_provenance(
        configuration, snap / f"season={SEASON_START}" / "plays.parquet", project_root=REPO
    )
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="vardec-noisefloor",
        metrics={
            "real_margin_sd": target_sd,
            "calibrated_sd": calibrated_sd,
            "floor_sd": floor_sd,
            "execution_noise_share": exec_share,
            "disp": disp,
        },
        notes=(
            "SIM-02 lite variance floor: resampling simulator calibrated to real "
            "margin sd, then pools replaced by mean-EPA-equivalent plays; floor sd "
            "is pure scheduling/matchup variance. No wagering implication claimed."
        ),
        source="scripts/vardec_noisefloor.py",
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
