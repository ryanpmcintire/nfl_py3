from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.clv import HISTORICAL_CAPTURE_KIND, load_decision_quotes
from nfl_ats.io import run_id
from nfl_ats.market_data import spread_consensus
from nfl_ats.officials_archive import load_officials
from nfl_ats.officials_flag_features import ROOKIE_PRIOR_EXPERIENCE_MAX
from nfl_ats.overlay_composition import (
    DEFAULT_FEATURES,
    DEFAULT_INCIDENTS,
    blocked_bootstrap_matrix,
    load_inputs,
)
from nfl_ats.pick_refresh import (
    HANDLE_FOLLOW_POLICY,
    LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY,
    MOVEMENT_POLICY_MOVEMENT,
    PRODUCTION_COMPOSITION_POLICY_IDS,
    ROOKIE_CREW_POLICY,
    ROOKIE_CREW_SEASON_FLOOR,
)
from nfl_ats.reporting import artifact_directories
from nfl_ats.sharp_book_movement_features import leader_follow_threshold
from nfl_ats.unserved_tilt_marginals import served_card_flip_set
from nfl_ats.weak_stack_v3_features import latest_schedules_snapshot

SAMPLES = 20_000
SEED = 20260821
LATE_WEEK_THRESHOLD_SERVED = 1.0
LATE_WEEK_THRESHOLD_CODE_TODAY = 0.5
CONSENSUS_THRESHOLD = 1.0
HANDLE_MONEY_THRESHOLD = 70.0
NEWS_CONTRADICT = -2.0
PFT_CONTRADICT = -1.0
PREDECLARATION_DOC = "docs/served_refresh_card.md"
FAMILY = "served_refresh_card"
ROOKIE_PER_GAME_RELATIVE = "opener_evaluation/rookie_crew_reconciled/per_game.parquet"

WINDOWS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("2023_2025", (2023, 2024, 2025)),
    ("2023", (2023,)),
    ("2024", (2024,)),
    ("2025", (2025,)),
)

WINDOW_SEASONS: dict[str, tuple[int, int]] = {
    "2023_2025": (2023, 2025),
    "2023": (2023, 2023),
    "2024": (2024, 2024),
    "2025": (2025, 2025),
    "2020_2025": (2020, 2025),
    "2020_2022": (2020, 2022),
}

RETIRED_POLICY_IDS: dict[str, str] = {
    "composition": PRODUCTION_COMPOSITION_POLICY_IDS[-1],
    "late_week_follow": LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY,
    "consensus_movement": MOVEMENT_POLICY_MOVEMENT,
    "rookie_crew": ROOKIE_CREW_POLICY,
    "handle_follow": HANDLE_FOLLOW_POLICY,
}

OFF_ARM_0_5_POLICY_IDS: dict[str, str] = {
    **RETIRED_POLICY_IDS,
    "late_week_follow": "late_week_leader_median_follow_0_5_off_incumbent",
}

PLAIN: dict[str, str] = {
    "step1_tuesday_card": (
        "This is what the Tuesday adjustments are worth: nine situational rules that flip the "
        "model's pick when a coach, a bye, the weather or a revenge angle says to."
    ),
    "step2_late_week_follow": (
        "When the three leading sportsbooks move a line at least a full point after Wednesday, "
        "the pick switches to the side they moved toward, unless an injury filed since Tuesday "
        "points the other way."
    ),
    "step3_consensus": (
        "When the whole market has moved the line at least a point since Tuesday, the pick "
        "switches to the side the market moved toward."
    ),
    "step4_rookie_crew": (
        "When a first- or second-season officiating crew is assigned, the game is re-read with a "
        "model that knows about the new crew."
    ),
    "step5_handle_follow": (
        "On the weekend, when at least 70 percent of the money on a game is on the other side, "
        "the pick switches to the money's side."
    ),
    "total_refresh_chain": (
        "This is every through-the-week rule together, measured against simply keeping Tuesday's "
        "card and never touching it again."
    ),
    "total_whole_card": (
        "This is the whole card -- Tuesday's adjustments plus every through-the-week rule -- "
        "against the bare model's own pick."
    ),
    "reference_follow_at_0_5": (
        "The same follow-the-books rule at the half-point trigger the code uses today, instead of "
        "a full point."
    ),
    "reference_follow_no_veto": (
        "The same follow-the-books rule at a full point, with no injury check to stop it."
    ),
    "control_pc_chain": (
        "A yardstick, not a rule: what perfect hindsight would be worth on exactly the games the "
        "through-the-week rules move."
    ),
    "control_pc_reach": (
        "A yardstick, not a rule: what perfect hindsight would be worth on every game these rules "
        "can reach."
    ),
    "chain_without_consensus_vs_tuesday": (
        "Every through-the-week rule except the market-wide line-move rule, against keeping "
        "Tuesday's card."
    ),
    "chain_without_consensus_vs_served_chain": (
        "What dropping the market-wide line-move rule from the through-the-week chain is worth."
    ),
    "headline_chain_where_inputs_exist_vs_tuesday": (
        "The headline pair: six seasons of picks with the through-the-week rules applied wherever "
        "the hour-by-hour lines, injury news and money splits exist, against the same picks left "
        "alone after Tuesday."
    ),
    "headline_tuesday_vs_raw": (
        "What Tuesday's nine situational adjustments are worth across six seasons against the "
        "bare model."
    ),
    "best_pick_small_spread_vs_incumbent": (
        "The starred Best Pick of the week, restricted to games with a spread of 6.5 or less, "
        "against the nominator that ignores the spread size."
    ),
    "best_pick_control_vs_incumbent": (
        "A yardstick, not a rule: what a perfect weekly Best Pick would be worth."
    ),
}

CATEGORY: dict[str, str] = {
    "step1_tuesday_card": "modeling",
    "step2_late_week_follow": "market",
    "step3_consensus": "market",
    "step4_rookie_crew": "offfield",
    "step5_handle_follow": "market",
    "total_refresh_chain": "market",
    "total_whole_card": "modeling",
    "reference_follow_at_0_5": "market",
    "reference_follow_no_veto": "market",
    "control_pc_chain": "control",
    "control_pc_reach": "control",
    "chain_without_consensus_vs_tuesday": "market",
    "chain_without_consensus_vs_served_chain": "market",
    "headline_chain_where_inputs_exist_vs_tuesday": "market",
    "headline_tuesday_vs_raw": "modeling",
    "best_pick_small_spread_vs_incumbent": "modeling",
    "best_pick_control_vs_incumbent": "control",
}

NOTES = (
    "Correlated decomposition: shares the 816-game 2023-2025 intraday odds archive and the "
    "leader-median follow with sharp_weighted_follow_*, follow_news_gate_* and "
    "follow_threshold_live_card_*; the nine-member card with served_card_harness_*; the "
    "rookie-crew refit with rookie_crew_reconciliation_*; and the split population with "
    "handle_follow_on_card_*. Never pool additively with those or with each other. Mined "
    "battery, no multiplicity correction claimed. The served path loads live captures and the "
    "market store holds none before 2026-08-17, so 2023-2025 is replayed on the historical "
    "intraday backfill; the handle step reaches only 133 of 816 games, and 2025's official "
    "injury rows carry no readable timestamp so the veto falls back to the headline reader."
)
EVIDENCE = (
    "Interval crosses zero, which AGENTS.md forbids as a closing ground. No trait here has zero "
    "split-half reliability, and the positive controls in this same battery resolve at +9.8 to "
    "+43.9 accuracy points on this exact population while every chain step sits between -1.6 and "
    "+1.5, so neither wrong_sign_resolved nor no_split_half_reliability nor positive_control_bound "
    "applies."
)


def _latest_directory(artifacts_root: Path, family: str, required_file: str) -> Path:
    directories = artifact_directories(artifacts_root / family, required_file)
    if not directories:
        raise ValueError(f"no {family!r} artifact under {artifacts_root} holds {required_file!r}")
    return directories[0]


def served_refresh_policy_ids() -> dict[str, str]:
    return {
        "composition": PRODUCTION_COMPOSITION_POLICY_IDS[-1],
        "late_week_follow": LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY,
        "rookie_crew": ROOKIE_CREW_POLICY,
        "handle_follow": HANDLE_FOLLOW_POLICY,
    }


def rookie_flagged_game_ids(repo_root: Path, known: set[str]) -> set[str]:
    officials = load_officials(repo_root, include_archive=True)
    crews = officials.loc[
        officials["position"].eq("Referee") & officials["season_type"].eq("REG"),
        ["game_id", "official_name"],
    ]
    schedules = pd.read_parquet(latest_schedules_snapshot(repo_root)).loc[
        :, ["game_id", "old_game_id", "season", "week"]
    ]
    history = (
        crews.merge(
            schedules.rename(columns={"season": "crew_season"}),
            left_on="game_id",
            right_on="old_game_id",
            how="inner",
            suffixes=("_legacy", ""),
        )
        .loc[:, ["game_id", "official_name", "crew_season"]]
        .rename(columns={"crew_season": "season"})
        .astype({"season": int})
    )
    tenure = (
        history.loc[:, ["official_name", "season"]]
        .drop_duplicates()
        .sort_values(["official_name", "season"])
        .reset_index(drop=True)
    )
    tenure["prior_seasons_experience"] = tenure.groupby("official_name").cumcount()
    history = history.merge(tenure, on=["official_name", "season"], how="left")
    rookie = history.loc[
        history["prior_seasons_experience"].le(ROOKIE_PRIOR_EXPERIENCE_MAX)
        & history["season"].ge(ROOKIE_CREW_SEASON_FLOOR)
    ]
    return set(rookie["game_id"].astype(str)) & known


def deadline_consensus(market_root: Path, game_ids: set[str], deadlines: pd.Series) -> pd.Series:
    quotes = load_decision_quotes(
        market_root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("intraday_hourly",),
        seasons=(2023, 2024, 2025),
    )
    quotes = quotes.loc[quotes["nflverse_game_id"].astype(str).isin(game_ids)].copy()
    quotes["observed_at_utc"] = pd.to_datetime(quotes["observed_at_utc"], utc=True)
    cutoff = quotes["nflverse_game_id"].astype(str).map(deadlines)
    quotes = quotes.loc[quotes["observed_at_utc"].le(cutoff)]
    consensus = spread_consensus(quotes)
    consensus = consensus.assign(
        nflverse_game_id=consensus["nflverse_game_id"].astype(str),
        consensus_home_spread=pd.to_numeric(consensus["consensus_home_spread"], errors="coerce"),
    )
    collapsed = consensus.groupby("nflverse_game_id")["consensus_home_spread"].median()
    return pd.Series(collapsed.to_numpy(), index=collapsed.index.to_numpy())


def build_frame(
    *, archive_per_game: Path, data_root: Path, repo_root: Path, artifacts_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    gate_directory = _latest_directory(artifacts_root, "follow_news_gate", "frame.parquet")
    handle_directory = _latest_directory(
        artifacts_root, "handle_follow_on_card", "population.parquet"
    )
    rookie_directory = _latest_directory(
        artifacts_root,
        "rookie_crew_reconciliation",
        ROOKIE_PER_GAME_RELATIVE,
    )
    per_game, schedules, _pf, _snapshot_name, _p = load_inputs(archive_per_game, data_root)
    flips, members = served_card_flip_set(
        per_game,
        data_root=data_root,
        repo_root=repo_root,
        features=repo_root / DEFAULT_FEATURES,
        incidents=repo_root / DEFAULT_INCIDENTS,
        schedules=schedules,
        card="served",
    )
    ids = per_game["game_id"].astype(str)
    archive = pd.DataFrame(
        {
            "game_id": ids.to_numpy(),
            "season": pd.to_numeric(per_game["season"], errors="coerce").to_numpy(),
            "week": pd.to_numeric(per_game["week"], errors="coerce").to_numpy(),
            "tue_open_home_spread": pd.to_numeric(
                per_game["tue_open_home_spread"], errors="coerce"
            ).to_numpy(),
            "margin_vs_open": pd.to_numeric(per_game["margin_vs_open"], errors="coerce").to_numpy(),
            "raw_pick_home": pd.to_numeric(
                per_game["home_cover_probability_at_open"], errors="coerce"
            )
            .ge(0.5)
            .to_numpy(),
            "composition_flip": ids.isin(flips).to_numpy(),
        }
    )
    archive["card_pick_home"] = archive["raw_pick_home"] ^ archive["composition_flip"]

    rookie = pd.read_parquet(rookie_directory / ROOKIE_PER_GAME_RELATIVE)
    if not rookie["game_id"].astype(str).equals(ids):
        raise ValueError("rookie refit archive is not row-aligned with the served archive")
    archive["rookie_refit_pick_home"] = (
        pd.to_numeric(rookie["home_cover_probability_at_open"], errors="coerce").ge(0.5).to_numpy()
        ^ archive["composition_flip"].to_numpy()
    )
    archive["rookie_flag"] = archive["game_id"].isin(
        rookie_flagged_game_ids(repo_root, set(archive["game_id"]))
    )

    handle = pd.read_parquet(handle_directory / "population.parquet")
    money_home = pd.to_numeric(handle["spread_home_money_pct"], errors="coerce")
    money_away = pd.to_numeric(handle["spread_away_money_pct"], errors="coerce")
    heavy_home = money_home.ge(money_away)
    handle_frame = pd.DataFrame(
        {
            "game_id": handle["game_id"].astype(str).to_numpy(),
            "handle_heavy_home": heavy_home.to_numpy(),
            "handle_heavy_pct": np.where(heavy_home, money_home, money_away),
        }
    ).dropna(subset=["handle_heavy_pct"])
    archive = archive.merge(handle_frame.drop_duplicates("game_id"), on="game_id", how="left")

    gate = pd.read_parquet(gate_directory / "frame.parquet")
    gate["game_id"] = gate["game_id"].astype(str)
    gate = gate.loc[
        :,
        [
            "game_id",
            "leader_net",
            "leader_books",
            "equal_net",
            "eligible_books",
            "news_toward_market",
            "pft_toward_market",
            "news_readable",
            "kickoff",
            "deadline",
            "tuesday_pick_home",
            "composition_flip",
        ],
    ].rename(
        columns={
            "composition_flip": "gate_composition_flip",
            "tuesday_pick_home": "gate_card_pick_home",
        }
    )
    frame = archive.merge(gate, on="game_id", how="inner")
    if len(frame) != 816:
        raise ValueError(f"expected 816 archive games in window, got {len(frame)}")
    mismatch_flip = int((frame["composition_flip"] != frame["gate_composition_flip"]).sum())
    mismatch_pick = int((frame["card_pick_home"] != frame["gate_card_pick_home"]).sum())

    deadlines = pd.Series(
        pd.to_datetime(frame["deadline"], utc=True).to_numpy(),
        index=frame["game_id"].to_numpy(),
    )
    consensus = deadline_consensus(data_root / "market" / "raw", set(frame["game_id"]), deadlines)
    frame["deadline_consensus_home_spread"] = frame["game_id"].map(consensus)
    frame["consensus_delta"] = (
        frame["deadline_consensus_home_spread"] - frame["tue_open_home_spread"]
    )

    season_readable = frame.groupby("season")["news_readable"].any()
    frame["season_news_readable"] = frame["season"].map(season_readable)
    frame["news_contradicts"] = np.where(
        frame["season_news_readable"],
        frame["news_toward_market"].le(NEWS_CONTRADICT),
        frame["pft_toward_market"].le(PFT_CONTRADICT),
    )

    books = frame["leader_books"].gt(0)
    frame["leader_fires_10"] = frame["leader_net"].abs().ge(LATE_WEEK_THRESHOLD_SERVED) & books
    frame["leader_fires_05"] = frame["leader_net"].abs().ge(LATE_WEEK_THRESHOLD_CODE_TODAY) & books
    frame["leader_market_home"] = frame["leader_net"].gt(0.0)
    frame["consensus_fires"] = frame["consensus_delta"].abs().ge(CONSENSUS_THRESHOLD)
    frame["consensus_market_home"] = frame["consensus_delta"].gt(0.0)
    frame["rookie_differs"] = frame["rookie_flag"] & (
        frame["rookie_refit_pick_home"] != frame["card_pick_home"]
    )
    frame["handle_available"] = frame["handle_heavy_pct"].notna()
    frame["handle_fires_vs_card"] = (
        frame["handle_available"]
        & frame["handle_heavy_pct"].ge(HANDLE_MONEY_THRESHOLD)
        & (frame["handle_heavy_home"] != frame["card_pick_home"])
    )

    coverage = {
        "archive_games": len(frame),
        "scored_games": int(frame["margin_vs_open"].ne(0.0).sum()),
        "week_blocks": int(frame.groupby(["season", "week"]).ngroups),
        "season_blocks": int(frame["season"].nunique()),
        "composition_flips": int(frame["composition_flip"].sum()),
        "composition_flip_mismatch_vs_gate_lane": mismatch_flip,
        "card_pick_mismatch_vs_gate_lane": mismatch_pick,
        "leader_fires_05": int(frame["leader_fires_05"].sum()),
        "leader_fires_10": int(frame["leader_fires_10"].sum()),
        "leader_flips_05_vs_card": int(
            (
                frame["leader_fires_05"] & (frame["leader_market_home"] != frame["card_pick_home"])
            ).sum()
        ),
        "leader_flips_10_vs_card": int(
            (
                frame["leader_fires_10"] & (frame["leader_market_home"] != frame["card_pick_home"])
            ).sum()
        ),
        "news_readable_seasons": sorted(
            int(str(season)) for season, ok in season_readable.items() if bool(ok)
        ),
        "news_contradicted_fires_10": int(
            (frame["leader_fires_10"] & frame["news_contradicts"]).sum()
        ),
        "games_with_deadline_consensus": int(frame["deadline_consensus_home_spread"].notna().sum()),
        "consensus_fires": int(frame["consensus_fires"].sum()),
        "rookie_flagged_games": int(frame["rookie_flag"].sum()),
        "rookie_differs_from_card": int(frame["rookie_differs"].sum()),
        "handle_games": int(frame["handle_available"].sum()),
        "handle_fires_vs_card": int(frame["handle_fires_vs_card"].sum()),
        "member_flip_counts": {name: len(members[name]) for name in members},
        "inputs": {
            "archive_opener_evaluation": str(archive_per_game.parent),
            "follow_news_gate": str(gate_directory),
            "handle_follow_on_card": str(handle_directory),
            "rookie_crew_reconciliation": str(rookie_directory),
        },
    }
    return frame, archive, coverage


def _paired(delta: np.ndarray, blocks: pd.DataFrame, block: str) -> dict[str, float]:
    stats = blocked_bootstrap_matrix(
        delta[:, np.newaxis], blocks.reset_index(drop=True), block=block, samples=SAMPLES, seed=SEED
    )
    return {
        "estimate_accuracy_points": float(stats["estimate"][0] * 100.0),
        "lower_accuracy_points": float(stats["lower"][0] * 100.0),
        "upper_accuracy_points": float(stats["upper"][0] * 100.0),
        "probability_positive": float(stats["probability_positive"][0]),
        "standard_error_accuracy_points": float(stats["standard_error"][0] * 100.0),
        "blocks": int(stats["block_count"]),
    }


def _cell(label: str, cand: pd.Series, base: pd.Series, meta: pd.DataFrame) -> dict[str, Any]:
    live = pd.concat({"candidate": cand, "baseline": base}, axis=1).join(meta, how="inner")
    live = live.loc[live["candidate"].notna() & live["baseline"].notna()]
    delta = (live["candidate"] - live["baseline"]).to_numpy(dtype=float)
    return {
        "label": label,
        "n": len(live),
        "baseline_accuracy": float(live["baseline"].mean()),
        "candidate_accuracy": float(live["candidate"].mean()),
        "picks_changed": int((live["candidate"] != live["baseline"]).sum()),
        "delta_accuracy_points": float(delta.mean() * 100.0),
        "week_blocked": _paired(delta, live[["season", "week"]], "week"),
        "season_blocked": _paired(delta, live[["season", "week"]], "season"),
    }


def _correctness(picks: pd.Series, margin: pd.Series) -> pd.Series:
    covered = margin.gt(0.0)
    return picks.eq(covered).astype(float).where(margin.ne(0.0) & margin.notna())


def _record_stat(cand: pd.Series, base: pd.Series, mask: np.ndarray) -> dict[str, int | float]:
    scoped = pd.DataFrame({"candidate": cand, "baseline": base}).loc[mask]
    live = scoped.loc[scoped["candidate"].notna() & scoped["baseline"].notna()]
    return {
        "games": len(live),
        "candidate_wins": int(live["candidate"].sum()),
        "baseline_wins": int(live["baseline"].sum()),
        "candidate_rate": float(live["candidate"].mean()) if len(live) else float("nan"),
    }


def _side(frame: pd.DataFrame, step: str, card: np.ndarray) -> np.ndarray:
    if step == "step2_late_week_follow":
        return frame["leader_market_home"].to_numpy(bool)
    if step == "step3_consensus":
        return frame["consensus_market_home"].to_numpy(bool)
    if step == "step4_rookie_crew":
        return frame["rookie_refit_pick_home"].to_numpy(bool)
    return frame["handle_heavy_home"].fillna(False).to_numpy(bool)


def build_chain(frame: pd.DataFrame) -> pd.DataFrame:
    card = frame["card_pick_home"].to_numpy(bool)
    fires = frame["leader_fires_10"].to_numpy(bool)
    veto = fires & frame["news_contradicts"].to_numpy(bool)
    follow = fires & ~frame["news_contradicts"].to_numpy(bool)
    market = frame["leader_market_home"].to_numpy(bool)

    c2 = np.where(follow, market, card)
    late_week_engaged = fires

    consensus = frame["consensus_fires"].to_numpy(bool) & ~late_week_engaged
    c3 = np.where(consensus, frame["consensus_market_home"].to_numpy(bool), c2)

    rookie = frame["rookie_differs"].to_numpy(bool) & ~late_week_engaged & ~consensus
    c4 = np.where(rookie, frame["rookie_refit_pick_home"].to_numpy(bool), c3)

    model_only = ~late_week_engaged & ~consensus & ~rookie
    heavy_pct = pd.to_numeric(frame["handle_heavy_pct"], errors="coerce").to_numpy(float)
    heavy_home = frame["handle_heavy_home"].fillna(False).to_numpy(bool)
    handle = model_only & np.isfinite(heavy_pct) & (heavy_pct >= 70.0) & (heavy_home != c4)
    c5 = np.where(handle, heavy_home, c4)

    fires_05 = frame["leader_fires_05"].to_numpy(bool)
    follow_05 = fires_05 & ~frame["news_contradicts"].to_numpy(bool)
    c2_05 = np.where(follow_05, market, card)
    c2_noveto = np.where(fires, market, card)

    out = frame.loc[:, ["game_id", "season", "week", "margin_vs_open"]].copy()
    out["c0_pick_home"] = frame["raw_pick_home"].to_numpy(bool)
    out["c1_pick_home"] = card
    out["c2_pick_home"] = c2
    out["c3_pick_home"] = c3
    out["c4_pick_home"] = c4
    out["c5_pick_home"] = c5
    out["c2_05_pick_home"] = c2_05
    out["c2_noveto_pick_home"] = c2_noveto
    out["step_late_week"] = follow
    out["step_late_week_veto"] = veto
    out["step_consensus"] = consensus
    out["step_rookie"] = rookie
    out["step_handle"] = handle
    return out


def measure_chain(frame: pd.DataFrame, chain: pd.DataFrame) -> dict[str, Any]:
    ids = chain["game_id"].astype(str)
    margin = pd.Series(
        pd.to_numeric(chain["margin_vs_open"], errors="coerce").to_numpy(), index=ids.to_numpy()
    )
    meta_all = pd.DataFrame(
        {"season": chain["season"].to_numpy(), "week": chain["week"].to_numpy()},
        index=ids.to_numpy(),
    )
    arms = {
        name: _correctness(
            pd.Series(chain[f"{name}_pick_home"].to_numpy(), index=ids.to_numpy()), margin
        )
        for name in ("c0", "c1", "c2", "c3", "c4", "c5", "c2_05", "c2_noveto")
    }
    covered = margin.gt(0.0)
    oracle = pd.Series(1.0, index=ids.to_numpy()).where(margin.ne(0.0) & margin.notna())
    changed_by_chain = pd.Series(
        (chain["c5_pick_home"] != chain["c1_pick_home"]).to_numpy(), index=ids.to_numpy()
    )
    pc_chain_pick = pd.Series(
        np.where(changed_by_chain.to_numpy(), covered.to_numpy(), chain["c1_pick_home"].to_numpy()),
        index=ids.to_numpy(),
    )
    arms["pc_chain"] = _correctness(pc_chain_pick, margin)
    arms["pc_reach"] = oracle

    accuracies = {
        name: {
            "overall": float(series.mean()),
            **{
                str(season): float(series.loc[meta_all["season"].eq(season).to_numpy()].mean())
                for season in (2023, 2024, 2025)
            },
        }
        for name, series in arms.items()
    }

    step_cells: list[dict[str, Any]] = []
    steps = (
        ("step1_tuesday_card", "c1", "c0"),
        ("step2_late_week_follow", "c2", "c1"),
        ("step3_consensus", "c3", "c2"),
        ("step4_rookie_crew", "c4", "c3"),
        ("step5_handle_follow", "c5", "c4"),
        ("total_refresh_chain", "c5", "c1"),
        ("total_whole_card", "c5", "c0"),
        ("reference_follow_at_0_5", "c2_05", "c1"),
        ("reference_follow_no_veto", "c2_noveto", "c1"),
        ("control_pc_chain", "pc_chain", "c1"),
        ("control_pc_reach", "pc_reach", "c1"),
    )
    for window, seasons in WINDOWS:
        mask = meta_all["season"].isin(list(seasons)).to_numpy()
        meta = meta_all.loc[mask]
        for name, cand, base in steps:
            row = _cell(f"{name}_{window}", arms[cand].loc[mask], arms[base].loc[mask], meta)
            row.update(step=name, window=window, candidate=cand, baseline=base)
            step_cells.append(row)

    weeks = int(meta_all.groupby(["season", "week"]).ngroups)
    per_week: dict[str, Any] = {
        "week_blocks": weeks,
        "picks_changed": {
            "step1_tuesday_card": int(frame["composition_flip"].sum()),
            "step2_late_week_follow": int((chain["c2_pick_home"] != chain["c1_pick_home"]).sum()),
            "step3_consensus": int((chain["c3_pick_home"] != chain["c2_pick_home"]).sum()),
            "step4_rookie_crew": int((chain["c4_pick_home"] != chain["c3_pick_home"]).sum()),
            "step5_handle_follow": int((chain["c5_pick_home"] != chain["c4_pick_home"]).sum()),
            "total_refresh_chain": int((chain["c5_pick_home"] != chain["c1_pick_home"]).sum()),
            "total_whole_card": int((chain["c5_pick_home"] != chain["c0_pick_home"]).sum()),
        },
    }
    per_week["picks_changed_per_week"] = {
        key: value / weeks for key, value in per_week["picks_changed"].items()
    }
    per_week["policy_counts"] = {
        "late_week_follow_applied": int(chain["step_late_week"].sum()),
        "late_week_vetoed": int(chain["step_late_week_veto"].sum()),
        "consensus_applied": int(chain["step_consensus"].sum()),
        "rookie_crew_applied": int(chain["step_rookie"].sum()),
        "handle_applied": int(chain["step_handle"].sum()),
        "model_only": int(
            (
                ~chain["step_late_week"]
                & ~chain["step_late_week_veto"]
                & ~chain["step_consensus"]
                & ~chain["step_rookie"]
                & ~chain["step_handle"]
            ).sum()
        ),
    }

    card = frame["card_pick_home"].to_numpy(bool)
    standalone = {
        "step2_late_week_follow": frame["leader_fires_10"].to_numpy(bool),
        "step3_consensus": frame["consensus_fires"].to_numpy(bool),
        "step4_rookie_crew": frame["rookie_differs"].to_numpy(bool),
        "step5_handle_follow": frame["handle_fires_vs_card"].to_numpy(bool),
    }
    standalone_changes = {
        "step2_late_week_follow": frame["leader_fires_10"].to_numpy(bool)
        & ~frame["news_contradicts"].to_numpy(bool)
        & (frame["leader_market_home"].to_numpy(bool) != card),
        "step3_consensus": frame["consensus_fires"].to_numpy(bool)
        & (frame["consensus_market_home"].to_numpy(bool) != card),
        "step4_rookie_crew": frame["rookie_differs"].to_numpy(bool),
        "step5_handle_follow": frame["handle_fires_vs_card"].to_numpy(bool),
    }
    order = list(standalone)
    collision: dict[str, Any] = {"standalone_fires": {}, "standalone_changes": {}, "pairs": []}
    for name in order:
        collision["standalone_fires"][name] = int(standalone[name].sum())
        collision["standalone_changes"][name] = int(standalone_changes[name].sum())
    engaged = {
        "step2_late_week_follow": frame["leader_fires_10"].to_numpy(bool),
        "step3_consensus": chain["step_consensus"].to_numpy(bool),
        "step4_rookie_crew": chain["step_rookie"].to_numpy(bool),
        "step5_handle_follow": chain["step_handle"].to_numpy(bool),
    }
    for i, earlier in enumerate(order):
        for later in order[i + 1 :]:
            both = int((standalone[earlier] & standalone[later]).sum())
            both_change = int((standalone_changes[earlier] & standalone_changes[later]).sum())
            disagree = int(
                (
                    standalone[earlier]
                    & standalone[later]
                    & (_side(frame, earlier, card) != _side(frame, later, card))
                ).sum()
            )
            suppressed = int((engaged[earlier] & standalone[later]).sum())
            suppressed_change = int((engaged[earlier] & standalone_changes[later]).sum())
            collision["pairs"].append(
                {
                    "earlier": earlier,
                    "later": later,
                    "both_would_fire": both,
                    "both_would_change_the_card_pick": both_change,
                    "and_would_disagree_with_each_other": disagree,
                    "later_suppressed_by_earlier": suppressed,
                    "later_suppressed_when_it_would_have_changed_the_pick": suppressed_change,
                }
            )

    delta_series = pd.to_numeric(frame["consensus_delta"], errors="coerce")
    diagnostics = {
        "consensus_delta_mean": float(delta_series.mean()),
        "consensus_delta_median": float(delta_series.median()),
        "consensus_delta_abs_mean": float(delta_series.abs().mean()),
        "consensus_delta_zero_share": float(delta_series.abs().lt(0.25).mean()),
        "leader_and_consensus_same_direction": int(
            (
                frame["leader_fires_10"].to_numpy(bool)
                & frame["consensus_fires"].to_numpy(bool)
                & (
                    frame["leader_market_home"].to_numpy(bool)
                    == frame["consensus_market_home"].to_numpy(bool)
                )
            ).sum()
        ),
        "veto_suppressed_consensus_changes": int(
            (
                chain["step_late_week_veto"].to_numpy(bool) & standalone_changes["step3_consensus"]
            ).sum()
        ),
        "consensus_record_on_its_own_changes": _record_stat(
            arms["c3"],
            arms["c2"],
            chain["step_consensus"].to_numpy(bool)
            & (chain["c3_pick_home"].to_numpy(bool) != chain["c2_pick_home"].to_numpy(bool)),
        ),
        "follow_record_on_its_own_changes": _record_stat(
            arms["c2"],
            arms["c1"],
            (chain["c2_pick_home"].to_numpy(bool) != chain["c1_pick_home"].to_numpy(bool)),
        ),
        "handle_record_on_its_own_changes": _record_stat(
            arms["c5"],
            arms["c4"],
            (chain["c5_pick_home"].to_numpy(bool) != chain["c4_pick_home"].to_numpy(bool)),
        ),
    }
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "diagnostics": diagnostics,
        "samples": SAMPLES,
        "seed": SEED,
        "accuracies": accuracies,
        "cells": step_cells,
        "reach": per_week,
        "collision": collision,
    }


def build_extras(
    *, frame: pd.DataFrame, chain: pd.DataFrame, archive: pd.DataFrame, best_pick_path: Path
) -> dict[str, Any]:
    ids = chain["game_id"].astype(str).to_numpy()
    margin = pd.Series(
        pd.to_numeric(chain["margin_vs_open"], errors="coerce").to_numpy(), index=ids
    )
    meta = pd.DataFrame(
        {"season": chain["season"].to_numpy(), "week": chain["week"].to_numpy()}, index=ids
    )

    c2 = chain["c2_pick_home"].to_numpy(bool)
    late_week_engaged = frame["leader_fires_10"].to_numpy(bool)
    rookie_no_consensus = frame["rookie_differs"].to_numpy(bool) & ~late_week_engaged
    c4x = np.where(rookie_no_consensus, frame["rookie_refit_pick_home"].to_numpy(bool), c2)
    model_only_x = ~late_week_engaged & ~rookie_no_consensus
    heavy_pct = pd.to_numeric(frame["handle_heavy_pct"], errors="coerce").to_numpy(float)
    heavy_home = frame["handle_heavy_home"].fillna(False).to_numpy(bool)
    handle_x = model_only_x & np.isfinite(heavy_pct) & (heavy_pct >= 70.0) & (heavy_home != c4x)
    c5x = np.where(handle_x, heavy_home, c4x)

    arms = {
        name: _correctness(pd.Series(chain[f"{name}_pick_home"].to_numpy(), index=ids), margin)
        for name in ("c0", "c1", "c2", "c3", "c4", "c5")
    }
    arms["c5x"] = _correctness(pd.Series(c5x, index=ids), margin)

    rows: list[dict[str, Any]] = []
    for window, seasons in (
        ("2023_2025", (2023, 2024, 2025)),
        ("2023", (2023,)),
        ("2024", (2024,)),
        ("2025", (2025,)),
    ):
        mask = meta["season"].isin(list(seasons)).to_numpy()
        for label, cand, base in (
            ("chain_without_consensus_vs_tuesday", "c5x", "c1"),
            ("chain_without_consensus_vs_served_chain", "c5x", "c5"),
        ):
            row = _cell(
                f"{label}_{window}", arms[cand].loc[mask], arms[base].loc[mask], meta.loc[mask]
            )
            row.update(step=label, window=window)
            rows.append(row)

    archive_ids = archive["game_id"].astype(str).to_numpy()
    archive_margin = pd.Series(
        pd.to_numeric(archive["margin_vs_open"], errors="coerce").to_numpy(), index=archive_ids
    )
    archive_meta = pd.DataFrame(
        {"season": archive["season"].to_numpy(), "week": archive["week"].to_numpy()},
        index=archive_ids,
    )
    card_all = pd.Series(archive["card_pick_home"].to_numpy(bool), index=archive_ids)
    raw_all = pd.Series(archive["raw_pick_home"].to_numpy(bool), index=archive_ids)
    rookie_all = pd.Series(
        (
            archive["rookie_flag"].to_numpy(bool)
            & (
                archive["rookie_refit_pick_home"].to_numpy(bool)
                != archive["card_pick_home"].to_numpy(bool)
            )
        ),
        index=archive_ids,
    )
    chain_side = pd.Series(chain["c5_pick_home"].to_numpy(bool), index=ids)
    applied = card_all.copy()
    applied.loc[rookie_all.loc[rookie_all].index] = pd.Series(
        archive["rookie_refit_pick_home"].to_numpy(bool), index=archive_ids
    ).loc[rookie_all.loc[rookie_all].index]
    applied.loc[chain_side.index] = chain_side

    arms_all = {
        "c0_all": _correctness(raw_all, archive_margin),
        "c1_all": _correctness(card_all, archive_margin),
        "chain_where_inputs_exist": _correctness(applied, archive_margin),
    }
    wide_windows: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("2020_2025", (2020, 2021, 2022, 2023, 2024, 2025)),
        ("2020_2022", (2020, 2021, 2022)),
    )
    for window, wide_seasons in wide_windows:
        mask = archive_meta["season"].isin(list(wide_seasons)).to_numpy()
        for label, cand, base in (
            ("headline_chain_where_inputs_exist_vs_tuesday", "chain_where_inputs_exist", "c1_all"),
            ("headline_tuesday_vs_raw", "c1_all", "c0_all"),
        ):
            row = _cell(
                f"{label}_{window}",
                arms_all[cand].loc[mask],
                arms_all[base].loc[mask],
                archive_meta.loc[mask],
            )
            row.update(step=label, window=window)
            rows.append(row)

    headline = {
        "tuesday_card_2020_2025": float(arms_all["c1_all"].mean()),
        "chain_where_inputs_exist_2020_2025": float(arms_all["chain_where_inputs_exist"].mean()),
        "raw_model_2020_2025": float(arms_all["c0_all"].mean()),
        "scored_games_2020_2025": int(arms_all["c1_all"].notna().sum()),
        "tuesday_card_2020_2022": float(
            arms_all["c1_all"]
            .loc[archive_meta["season"].isin([2020, 2021, 2022]).to_numpy()]
            .mean()
        ),
        "chain_where_inputs_exist_2020_2022": float(
            arms_all["chain_where_inputs_exist"]
            .loc[archive_meta["season"].isin([2020, 2021, 2022]).to_numpy()]
            .mean()
        ),
        "tuesday_card_2023_2025": float(arms["c1"].mean()),
        "served_chain_2023_2025": float(arms["c5"].mean()),
        "chain_without_consensus_2023_2025": float(arms["c5x"].mean()),
    }

    best = pd.read_parquet(best_pick_path)
    best_rows: list[dict[str, Any]] = []
    best_pick_windows: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("2020_2025", (2020, 2021, 2022, 2023, 2024, 2025)),
        ("2023_2025", (2023, 2024, 2025)),
    )
    for window, best_seasons in best_pick_windows:
        scoped = best.loc[best["season"].isin(list(best_seasons))].copy()
        pairs = scoped.dropna(subset=["b0_v2_served_correct", "b2_small_spread_only_correct"])
        delta = pairs["b2_small_spread_only_correct"].to_numpy(float) - pairs[
            "b0_v2_served_correct"
        ].to_numpy(float)
        blocks = pairs[["season", "week"]].reset_index(drop=True)
        best_rows.append(
            {
                "label": f"best_pick_small_spread_vs_incumbent_{window}",
                "window": window,
                "weeks_paired": len(pairs),
                "incumbent_hits": int(pairs["b0_v2_served_correct"].sum()),
                "small_spread_hits": int(pairs["b2_small_spread_only_correct"].sum()),
                "incumbent_accuracy": float(pairs["b0_v2_served_correct"].mean()),
                "small_spread_accuracy": float(pairs["b2_small_spread_only_correct"].mean()),
                "weeks_nominee_differs": int(
                    (
                        pairs["b2_small_spread_only_game_id"].astype(str)
                        != pairs["b0_v2_served_game_id"].astype(str)
                    ).sum()
                ),
                "delta_accuracy_points": float(delta.mean() * 100.0),
                "week_blocked": _paired(delta, blocks, "week"),
                "season_blocked": _paired(delta, blocks, "season"),
            }
        )
        control = scoped.dropna(
            subset=["control_perfect_foresight_correct", "b0_v2_served_correct"]
        )
        cdelta = control["control_perfect_foresight_correct"].to_numpy(float) - control[
            "b0_v2_served_correct"
        ].to_numpy(float)
        best_rows.append(
            {
                "label": f"best_pick_control_vs_incumbent_{window}",
                "window": window,
                "weeks_paired": len(control),
                "incumbent_hits": int(control["b0_v2_served_correct"].sum()),
                "small_spread_hits": int(control["control_perfect_foresight_correct"].sum()),
                "incumbent_accuracy": float(control["b0_v2_served_correct"].mean()),
                "small_spread_accuracy": float(control["control_perfect_foresight_correct"].mean()),
                "weeks_nominee_differs": int(
                    (
                        control["control_perfect_foresight_game_id"].astype(str)
                        != control["b0_v2_served_game_id"].astype(str)
                    ).sum()
                ),
                "delta_accuracy_points": float(cdelta.mean() * 100.0),
                "week_blocked": _paired(
                    cdelta, control[["season", "week"]].reset_index(drop=True), "week"
                ),
                "season_blocked": _paired(
                    cdelta, control[["season", "week"]].reset_index(drop=True), "season"
                ),
            }
        )

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "headline": headline,
        "cells": rows,
        "best_pick": best_rows,
    }


def _chain_sides(
    frame: pd.DataFrame, *, threshold: float, veto: bool, use_consensus: bool
) -> np.ndarray:
    card = frame["card_pick_home"].to_numpy(bool)
    per_game = np.array(
        [
            leader_follow_threshold(v) if threshold >= 1.0 else threshold
            for v in frame["tue_open_home_spread"]
        ],
        dtype=float,
    )
    fires = (frame["leader_net"].abs().to_numpy(float) >= per_game) & frame["leader_books"].gt(
        0
    ).to_numpy(bool)
    contradicts = frame["news_contradicts"].to_numpy(bool)
    follow = fires & ~contradicts if veto else fires
    market = frame["leader_market_home"].to_numpy(bool)
    side = np.where(follow, market, card)
    if use_consensus:
        consensus = frame["consensus_fires"].to_numpy(bool) & ~fires
        side = np.where(consensus, frame["consensus_market_home"].to_numpy(bool), side)
    else:
        consensus = np.zeros(len(frame), dtype=bool)
    rookie = frame["rookie_differs"].to_numpy(bool) & ~fires & ~consensus
    side = np.where(rookie, frame["rookie_refit_pick_home"].to_numpy(bool), side)
    model_only = ~fires & ~consensus & ~rookie
    heavy_pct = pd.to_numeric(frame["handle_heavy_pct"], errors="coerce").to_numpy(float)
    heavy_home = frame["handle_heavy_home"].fillna(False).to_numpy(bool)
    handle = model_only & np.isfinite(heavy_pct) & (heavy_pct >= 70.0) & (heavy_home != side)
    return np.where(handle, heavy_home, side)


def build_headline(
    *,
    frame: pd.DataFrame,
    archive: pd.DataFrame,
    output_directory: Path,
    active_model_id: str,
    opener_evaluation_used: Path,
    opener_evaluation_identity_check: str,
) -> dict[str, Any]:
    ids = archive["game_id"].astype(str).to_numpy()
    margin = pd.Series(
        pd.to_numeric(archive["margin_vs_open"], errors="coerce").to_numpy(), index=ids
    )
    meta = pd.DataFrame(
        {"season": archive["season"].to_numpy(), "week": archive["week"].to_numpy()}, index=ids
    )
    card = pd.Series(archive["card_pick_home"].to_numpy(bool), index=ids)
    rookie_flip = archive["rookie_flag"].to_numpy(bool) & (
        archive["rookie_refit_pick_home"].to_numpy(bool) != archive["card_pick_home"].to_numpy(bool)
    )
    card_correct = _correctness(card, margin)

    window_ids = frame["game_id"].astype(str).to_numpy()
    variants: list[dict[str, Any]] = []
    for label, threshold, veto, use_consensus, policy_ids in (
        ("threshold_0_5_no_veto_off_arm", 0.5, False, True, OFF_ARM_0_5_POLICY_IDS),
        ("served_with_consensus_off_arm", 1.0, True, True, RETIRED_POLICY_IDS),
        ("served", 1.0, True, False, served_refresh_policy_ids()),
    ):
        window_side = pd.Series(
            _chain_sides(frame, threshold=threshold, veto=veto, use_consensus=use_consensus),
            index=window_ids,
        )
        applied = card.copy()
        applied.loc[rookie_flip] = pd.Series(
            archive["rookie_refit_pick_home"].to_numpy(bool), index=ids
        ).loc[rookie_flip]
        applied.loc[window_side.index] = window_side
        chain_correct = _correctness(applied, margin)
        live = pd.concat({"candidate": chain_correct, "baseline": card_correct}, axis=1).join(
            meta, how="inner"
        )
        live = live.loc[live["candidate"].notna() & live["baseline"].notna()]
        delta = (live["candidate"] - live["baseline"]).to_numpy(float)
        in_window = live.index.isin(set(window_ids))
        variants.append(
            {
                "variant": label,
                "policy_ids": policy_ids,
                "scored_games": len(live),
                "week_blocks": int(live.groupby(["season", "week"]).ngroups),
                "seasons": [int(value) for value in sorted(live["season"].unique())],
                "tuesday_card_accuracy": float(live["baseline"].mean()),
                "refresh_chain_accuracy": float(live["candidate"].mean()),
                "picks_changed": int((live["candidate"] != live["baseline"]).sum()),
                "late_week_input_seasons": [2023, 2024, 2025],
                "scored_games_with_late_week_inputs": int(in_window.sum()),
                "tuesday_card_accuracy_with_inputs": float(live.loc[in_window, "baseline"].mean()),
                "refresh_chain_accuracy_with_inputs": float(
                    live.loc[in_window, "candidate"].mean()
                ),
                "delta_accuracy_points": float(delta.mean() * 100.0),
                "week_blocked": _paired(delta, live[["season", "week"]], "week"),
                "season_blocked": _paired(delta, live[["season", "week"]], "season"),
            }
        )

    return {
        "lane": FAMILY,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "predeclaration": PREDECLARATION_DOC,
        "artifact": str(output_directory),
        "model_id": active_model_id,
        "opener_evaluation_used": str(opener_evaluation_used),
        "opener_evaluation_identity_check": opener_evaluation_identity_check,
        "grade": "opener",
        "samples": SAMPLES,
        "seed": SEED,
        "variants": variants,
    }


def _argv_for(
    *,
    uv: str,
    repo_root: Path,
    name: str,
    step: str,
    window: str,
    effect: float,
    low: float,
    high: float,
    probability: float,
    games: int,
    blocks: int,
    description: str,
) -> list[str]:
    start, end = WINDOW_SEASONS[window]
    return [
        uv,
        "run",
        "--no-sync",
        "--project",
        str(repo_root),
        "nfl-ats",
        "weak-signals",
        "record",
        "--name",
        name,
        "--description",
        description,
        "--source",
        f"artifacts/{FAMILY}/<stamp>/results.json ({PREDECLARATION_DOC})",
        "--effect",
        f"{effect:.6f}",
        "--effect-units",
        "accuracy_points",
        "--classification",
        "unresolved_below_power",
        "--league",
        "nfl",
        "--season-start",
        str(start),
        "--season-end",
        str(end),
        "--interval-low",
        f"{low:.6f}",
        "--interval-high",
        f"{high:.6f}",
        "--probability-positive",
        f"{probability:.6f}",
        "--sample-games",
        str(games),
        "--sample-blocks",
        str(blocks),
        "--family",
        FAMILY,
        "--classification-evidence",
        EVIDENCE,
        "--plain-summary",
        PLAIN[step],
        "--category",
        CATEGORY[step],
        "--notes",
        NOTES,
        "--replace",
    ]


def record_cells(
    *,
    output_directory: Path,
    results: dict[str, Any],
    extras: dict[str, Any],
    repo_root: Path,
    registry_dir: Path,
) -> dict[str, Any]:
    uv = str(repo_root / ".tools" / "uv.exe")
    if not Path(uv).is_file():
        uv = sys.executable
    commands: list[dict[str, Any]] = []
    for row in [*results["cells"], *extras["cells"]]:
        step = str(row["step"])
        window = str(row["window"])
        week = row["week_blocked"]
        season = row["season_blocked"]
        description = (
            f"Served refresh chain, {window.replace('_', '-')}: {step.replace('_', ' ')} on the "
            f"played card, replayed in the precedence src/nfl_ats/pick_refresh.py serves, graded "
            f"at the frozen Tuesday opener on {row['n']} non-push games. Baseline "
            f"{row['baseline_accuracy'] * 100:.3f}%, candidate "
            f"{row['candidate_accuracy'] * 100:.3f}%, {row['picks_changed']} picks changed. "
            f"Week-blocked bootstrap, 20,000 draws, seed 20260821; season-blocked read "
            f"{season['lower_accuracy_points']:+.3f} to {season['upper_accuracy_points']:+.3f}, "
            f"P+ {season['probability_positive']:.4f}."
        )
        commands.append(
            {
                "cell": f"{step}_{window}",
                "argv": _argv_for(
                    uv=uv,
                    repo_root=repo_root,
                    name=f"{FAMILY}_{step}_{window}",
                    step=step,
                    window=window,
                    effect=float(row["delta_accuracy_points"]),
                    low=float(week["lower_accuracy_points"]),
                    high=float(week["upper_accuracy_points"]),
                    probability=float(week["probability_positive"]),
                    games=int(row["n"]),
                    blocks=int(week["blocks"]),
                    description=description,
                ),
            }
        )
    for row in extras["best_pick"]:
        label = str(row["label"])
        window = str(row["window"])
        step = label[: -(len(window) + 1)]
        week = row["week_blocked"]
        season = row["season_blocked"]
        description = (
            f"Served refresh chain, {window.replace('_', '-')}: the weekly Best Pick nomination, "
            f"{step.replace('_', ' ')}, on {row['weeks_paired']} paired weeks. Incumbent "
            f"{row['incumbent_accuracy'] * 100:.3f}% ({row['incumbent_hits']} hits), arm "
            f"{row['small_spread_accuracy'] * 100:.3f}% ({row['small_spread_hits']} hits), "
            f"{row['weeks_nominee_differs']} weeks the nominee differs. Week-blocked bootstrap, "
            f"20,000 draws, seed 20260821; season-blocked read "
            f"{season['lower_accuracy_points']:+.3f} to {season['upper_accuracy_points']:+.3f}, "
            f"P+ {season['probability_positive']:.4f}."
        )
        commands.append(
            {
                "cell": label,
                "argv": _argv_for(
                    uv=uv,
                    repo_root=repo_root,
                    name=f"{FAMILY}_{label}",
                    step=step,
                    window=window,
                    effect=float(row["delta_accuracy_points"]),
                    low=float(week["lower_accuracy_points"]),
                    high=float(week["upper_accuracy_points"]),
                    probability=float(week["probability_positive"]),
                    games=int(row["weeks_paired"]),
                    blocks=int(week["blocks"]),
                    description=description,
                ),
            }
        )
    for entry in commands:
        entry["argv"] = [arg.replace("<stamp>", output_directory.name) for arg in entry["argv"]]
    (output_directory / "record_commands.json").write_text(
        json.dumps(commands, indent=1), encoding="utf-8"
    )
    env = {**os.environ, "NFL_ATS_REGISTRY_DIR": str(registry_dir)}
    failures: list[str] = []
    for entry in commands:
        result = subprocess.run(
            entry["argv"], cwd=str(repo_root), env=env, capture_output=True, text=True
        )
        if result.returncode != 0:
            failures.append(f"{entry['cell']}: {result.stderr.strip()[:400]}")
    return {"recorded": len(commands) - len(failures), "total": len(commands), "failures": failures}


def measure(
    *,
    artifacts_root: Path,
    data_root: Path,
    repo_root: Path,
    active: dict[str, Any] | None = None,
    output_directory: Path | None = None,
    record: bool = False,
    registry_dir: Path | None = None,
) -> dict[str, Any]:
    from nfl_ats.public_board import find_matching_opener_evaluation

    if active is None:
        active = load_active_ats_model(artifacts_root)
    if not active:
        raise ValueError(f"no active ATS model manifest under {artifacts_root}")
    match = find_matching_opener_evaluation(artifacts_root, active)
    if match is None:
        raise ValueError(
            "no opener-evaluation matches the active model; run opener-evaluation first"
        )
    _opener_metadata, opener_directory = match
    archive_per_game = opener_directory / "per_game.parquet"

    output_directory = output_directory or (artifacts_root / FAMILY / run_id())
    output_directory.mkdir(parents=True, exist_ok=True)

    frame, archive, coverage = build_frame(
        archive_per_game=archive_per_game,
        data_root=data_root,
        repo_root=repo_root,
        artifacts_root=artifacts_root,
    )
    frame.to_parquet(output_directory / "frame.parquet", index=False)
    archive.to_parquet(output_directory / "archive_frame.parquet", index=False)
    (output_directory / "coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )

    chain = build_chain(frame)
    chain.to_parquet(output_directory / "chain.parquet", index=False)
    results = measure_chain(frame, chain)
    (output_directory / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(results["cells"]).to_csv(output_directory / "cells.csv", index=False)

    best_pick_directory = _latest_directory(
        artifacts_root, "best_pick_bucket_confidence", "weekly_served.parquet"
    )
    extras = build_extras(
        frame=frame,
        chain=chain,
        archive=archive,
        best_pick_path=best_pick_directory / "weekly_served.parquet",
    )
    (output_directory / "extras.json").write_text(json.dumps(extras, indent=2), encoding="utf-8")

    identity_check = (
        f"Scored directly against {opener_directory} "
        f"(active model {active.get('model_id')}, feature-table digest "
        f"{active.get('feature_table_sha256')}); no substitution."
    )
    opener_used_relative = os.path.relpath(opener_directory, repo_root).replace("\\", "/")
    headline = build_headline(
        frame=frame,
        archive=archive,
        output_directory=output_directory,
        active_model_id=str(active.get("model_id")),
        opener_evaluation_used=Path(opener_used_relative),
        opener_evaluation_identity_check=identity_check,
    )
    (output_directory / "headline.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )

    predeclaration_source = repo_root / PREDECLARATION_DOC
    if predeclaration_source.is_file():
        (output_directory / "predeclaration_frozen.md").write_bytes(
            predeclaration_source.read_bytes()
        )
    script_source = repo_root / "scripts" / "served_refresh_card_measure.py"
    if script_source.is_file():
        (output_directory / "served_refresh_card_measure.py").write_bytes(
            script_source.read_bytes()
        )
    module_source = Path(__file__)
    if module_source.is_file():
        (output_directory / "served_refresh_card.py").write_bytes(module_source.read_bytes())

    record_summary: dict[str, Any] | None = None
    if record:
        record_summary = record_cells(
            output_directory=output_directory,
            results=results,
            extras=extras,
            repo_root=repo_root,
            registry_dir=registry_dir or (repo_root / "registry"),
        )

    served_variant = next((v for v in headline["variants"] if v["variant"] == "served"), None)
    return {
        "status": "refreshed",
        "directory": str(output_directory),
        "model_id": active.get("model_id"),
        "opener_evaluation_used": headline["opener_evaluation_used"],
        "coverage": coverage,
        "served_variant": served_variant,
        "record": record_summary,
    }


def reuse_or_measure(
    *,
    artifacts_root: Path,
    data_root: Path,
    repo_root: Path,
    record: bool = False,
    registry_dir: Path | None = None,
) -> dict[str, Any]:
    from nfl_ats.public_board import load_refresh_chain_measurement

    active = load_active_ats_model(artifacts_root)
    if not active:
        return {"status": "not_applicable", "reason": "no active ATS model manifest"}
    try:
        existing = load_refresh_chain_measurement(artifacts_root, active)
    except ValueError as error:
        return measure(
            artifacts_root=artifacts_root,
            data_root=data_root,
            repo_root=repo_root,
            active=active,
            record=record,
            registry_dir=registry_dir,
        ) | {"reused_check_error": str(error)}
    if existing is None:
        return {
            "status": "not_applicable",
            "reason": "no served_refresh_card measurement exists yet; not created automatically",
        }
    return {
        "status": "reused",
        "directory": str(existing.directory),
        "variant": existing.variant,
        "tuesday_card_accuracy": existing.tuesday_card_accuracy,
        "refresh_chain_accuracy": existing.refresh_chain_accuracy,
        "picks_changed": existing.picks_changed,
    }
