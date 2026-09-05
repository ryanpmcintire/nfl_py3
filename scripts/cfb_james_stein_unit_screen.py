"""MOD-06 CFB James-Stein unit-value screen (standalone, src/ untouched).

Executes the predeclaration drafted in
``C:\\Users\\Ryan\\AppData\\Local\\Temp\\claude\\F--Repos-nfl-py3\\c8c5fbdd-027f-438d-b992-979e83a91c2e``
``\\scratchpad\\mod06_scope\\predeclaration_draft.md`` and its companion
``implementation_plan.md``, as adjudicated by the reviewing orchestrator
(8 decisions, summarized in the task prompt this script was written for):

1. **Option A** (severity-free, construct-preserving isolation): this script
   builds a NEW, non-injury-gated CFB team-strength feature,
   ``{home,away,diff}_skill_unit_value``, rather than trying to reach the
   literal (injury-gated, CFB-infeasible) ``_injury_value_features`` path.
2. **QB carve-out**: passer credits never enter the skill-unit numerator at
   all (only ``rusher_player_id``/``receiver_player_id`` credits do), and any
   player identified as a passer (>= ``QB_MIN_PASS_ATTEMPTS`` career pass
   attempts anywhere in the loaded 2013-2025 corpus -- a full-corpus
   attribute lookup, flagged as a mild non-walk-forward step for player
   IDENTITY only, never for any outcome-bearing quantity) has ALL of their
   rush/reception credits excluded too, mirroring ``players.py``'s hard
   ``skill_epa = 0.0`` override for position == QB.
3. **Underived constants**: ``VALUE_PRIOR_ACTIONS`` (200, mirrors
   ``players.py``'s ``value_prior_snaps`` default) and ``VALUE_SPAN`` (16,
   mirrors ``value_span``) are reused verbatim, never tuned against this
   screen's outcome. ``JS_EPSILON`` (the positive-part variance floor) is a
   new CFB-specific constant with no NFL analogue; chosen after inspecting
   this run's own ``tau_between`` magnitude (see the report), not guessed
   blind, but still flagged as underived in every output.
4. **Defensive-credit wiring**: VERIFIED this session (grep) that
   ``CFB_PARTICIPANT_ROLES``/``CFB_PARTICIPANT_SNAPSHOT_COLUMNS``
   (``cfb.py:471-505``) are declared but never imported or read by
   ``cfb_features.py``'s canonical build -- the only other reference is
   ``cli.py``'s ``cfb-ingest --source participants`` (a raw-snapshot
   downloader, not a feature builder). Defensive credits are NOT wired into
   the canonical CFB feature pipeline. Per the adjudicated decision, this
   screen is skill/offense-side ONLY; no new participant-table plumbing is
   added.
5. **Split-half reliability helper**: implemented inline in this script
   (``split_half_reliability`` below), not added to ``src/``, per the
   adjudicated decision. A shared ``nfl_ats.reliability`` helper is still
   wanted later (this is the third copy of this recipe in the repo, after
   ``docs/injury_value_lost.md`` and ``cfb_role_features.py``/
   ``cfb_value_weighted_continuity_screen.py``).
6. Bootstrap sample count defaults to 20,000 from the start (Decision 8),
   matching ``experiments.paired_feature_comparisons``'s current default.

Design decisions made while executing beyond the 8 adjudicated items
(flagged here, not silently picked):

a. **Position grouping collapses to one non-QB "skill" group.** CFB PBP
   carries no player-position tag for credited actors, so the
   offensive-line/front/secondary subdivision in ``players.py::_position_group``
   cannot be reproduced. After the QB exclusion (item 2 above), every
   remaining credited rusher/receiver is treated as one pooled "skill"
   group -- which is exactly the adjudicated resolution of Decision 2
   (RB/WR/TE/FB collapsed into one non-QB prior pool) applied literally,
   just without the ability to verify each individual is truly RB/WR/TE/FB.
b. **Denominator is credited actions (carries + receptions), not snaps**
   (predeclaration Decision 7, flagged as a genuine construct difference,
   not smoothed over). ``career_actions`` is a running total of credited
   plays, exactly mirroring ``career_offense_snaps``'s running-total
   convention in ``players.py``.
c. **Role-share weight substitute.** With no snap-share data, each active
   player's team-aggregate weight is their own current EWMA action-volume
   share among the team's active roster (``actions_state_i / sum_j
   actions_state_j``) -- a data-grounded proxy for "how big a role", not a
   literal snap share. Flagged as an inferred design choice; the
   predeclaration did not resolve this precisely for CFB.
d. **Active roster mass = appeared >= 1 time this season, missed_streak <
   STREAK_CAP (4, reused from ``cfb_role_features.FROZEN_STREAK_CAP``).**
   No minimum-experience gate is applied to team-aggregate INCLUSION
   (unlike the plain role-continuity family's qualification threshold) --
   thin-sample players are deliberately included so the shrinkage
   mechanism itself (toward zero vs. toward the group prior) is what
   differentiates the old and new constructs on exactly those players.
   This is the point of the hypothesis under test, not an oversight.
e. **Weekly-refreshed prior/variance snapshot**, not per-game. ``prior_g``
   and ``tau_between``/``tau_within`` are recomputed once per
   ``(season, week)`` block from state strictly before that week, then
   reused for every game in the block. This is still leak-free (no game's
   own outcome or same-week action ever informs its own evaluation) but is
   a coarser cadence than "at each walk-forward point" in the literal
   predeclaration text -- flagged as an implementation simplification,
   matching this repo's own weekly walk-forward cadence elsewhere
   (``cfb_walk_forward_benchmark`` refits weekly, not per-game).
f. **``tau_hat_g^2_within`` is estimated from per-GAME rate dispersion**
   (Welford variance of each experienced player's own per-game rate around
   their trailing mean), while ``sigma_i^2 = tau_within / career_actions_i``
   divides by a PLAY count per the predeclaration's literal formula. These
   are two different sampling units (games vs. plays); the predeclaration
   itself leaves the exact estimator open (Decision 4: "not derived,
   needs to be chosen... informed by the real magnitude of tau_g^2 on CFB
   data, not guessed in advance"). This is the choice made here, flagged
   as an approximation, not a rigorous per-play derivation.
g. **No FROZEN_MIN_TEAM_ACTIONS-style floor** is applied to team-games for
   state-update purposes (unlike the plain/value-weighted continuity
   families' >=10-combined-actions gate). Virtually every FBS game clears
   10 combined rush+reception credits, so this is expected to be
   immaterial; flagged rather than silently matched to the frozen family's
   gate.
h. **Player-state history is built only for FROZEN_ROLE_SEASONS (2013-2025)**
   -- the same audited credited-player-id coverage window the rest of this
   repo's CFB role machinery already restricts to. The clean-core
   population's 2012 games therefore get the feature at neutral (0.0)
   imputation (no built history exists that far back), consistent with
   how the frozen role-continuity family already handles its own
   pre-coverage seasons.

CFB-only; rotation registry rule 8 makes CFB screens free (no NFL rotation
window touched, no ``registry/*.json`` write). Per AGENTS.md, an interval
containing zero is never grounds to close this line -- this script only
measures and reports ``probability_positive``; classification is a
human/orchestrator decision made from the numbers below, not asserted here.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import nfl_ats.cli as cli  # noqa: E402
from nfl_ats.cfb_benchmark import (  # noqa: E402
    _PREDICTION_PASSTHROUGH,
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    CFB_CLEAN_CORE_SEASONS,
    _score_week,
    cfb_evaluation_window,
    fit_cfb_residual_model,
    summarize_cfb_benchmark,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS, load_cfb_seasons  # noqa: E402
from nfl_ats.cfb_role_features import ROLE_BENCHMARK_BASELINE_ARM  # noqa: E402
from nfl_ats.cfb_roles import (  # noqa: E402
    CFB_ROLE_PBP_LOAD_COLUMNS,
    FROZEN_ROLE_SEASONS,
    _regular_season_mask,
)
from nfl_ats.data import DataContractError, require_columns  # noqa: E402
from nfl_ats.estimation_variance import mde80, picks_differ_fraction  # noqa: E402
from nfl_ats.experiments import paired_feature_comparisons  # noqa: E402
from nfl_ats.io import atomic_csv, atomic_json, atomic_parquet, run_id  # noqa: E402
from nfl_ats.margin import fit_market_baseline  # noqa: E402
from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "cfb_james_stein_unit"
CFB_FEATURES_PATH = REPO / "data" / "processed" / "cfb_game_features.parquet"

# ---------------------------------------------------------------------------
# Constants -- mirrored NFL starting points (flagged, decision 3) and new
# CFB-specific choices (flagged, decision-adjacent, see module docstring).
# ---------------------------------------------------------------------------
VALUE_PRIOR_ACTIONS = 200.0  # players.py value_prior_snaps default, reused verbatim
VALUE_SPAN = 16  # players.py value_span default, reused verbatim
STREAK_CAP = 4  # cfb_role_features.FROZEN_STREAK_CAP, reused for active-roster departure
QB_MIN_PASS_ATTEMPTS = 5  # new: full-corpus passer identification threshold, flagged
EPA_CLIP = 5.0  # mirrors VALUE_EPA_CLIP in cfb_value_weighted_continuity_screen.py
MIN_EXPERIENCED_POOL = 20  # new: below this, prior_g/tau fall back to 0.0/epsilon
JS_EPSILON = 1e-4  # positive-part floor on tau_between; flagged underived (decision 4)

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260818  # frozen before any accuracy number was seen

BASELINE_ARM = ROLE_BENCHMARK_BASELINE_ARM  # "cfb_benchmark_v1", the frozen XLG-03 arm
CANDIDATE_ARM = "cfb_benchmark_v1_js_unit_value"
CANDIDATE_COLUMNS = (
    "home_skill_unit_value",
    "away_skill_unit_value",
    "diff_skill_unit_value",
)
JS_ROLE_COLUMNS = (*CFB_MODEL_FEATURE_COLUMNS, *CANDIDATE_COLUMNS)


# ---------------------------------------------------------------------------
# Step 1: load pbp and build the credited skill-production table
# ---------------------------------------------------------------------------


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pbp with EPA columns, canonical CFB games) for FROZEN_ROLE_SEASONS."""

    seasons = list(range(FROZEN_ROLE_SEASONS[0], FROZEN_ROLE_SEASONS[1] + 1))
    pbp = load_cfb_seasons(
        cli._data_root() / "cfb",
        "pbp",
        seasons,
        columns=[*CFB_ROLE_PBP_LOAD_COLUMNS, "pos_team_id", "EPA", "EPA_rush", "EPA_pass"],
    )
    canonical_games = pd.read_parquet(CFB_FEATURES_PATH)
    return pbp, canonical_games


def identify_qb_ids(pbp: pd.DataFrame) -> tuple[set[str], dict[str, Any]]:
    """Full-corpus passer identification (decision 2, flagged non-walk-forward step).

    A player_id is treated as a QB (excluded from the skill-unit
    aggregation entirely, mirroring ``players.py``'s hard
    ``skill_epa = 0.0`` for position == QB) if they registered at least
    ``QB_MIN_PASS_ATTEMPTS`` pass attempts anywhere in the loaded
    2013-2025 corpus. This is a categorical player-identity attribute
    (analogous to a roster position tag), not an outcome-bearing quantity,
    but it is derived from the full corpus rather than walk-forward, and is
    flagged as such rather than silently applied.
    """

    pass_flag = pd.to_numeric(pbp["pass"], errors="coerce").fillna(0).astype(bool)
    attempts = (
        pbp.loc[pass_flag & pbp["passer_player_id"].notna(), "passer_player_id"]
        .astype(str)
        .value_counts()
    )
    qb_ids = set(attempts.loc[attempts.ge(QB_MIN_PASS_ATTEMPTS)].index)
    diagnostics = {
        "distinct_passers_seen": int(attempts.size),
        "qb_ids_identified": len(qb_ids),
        "min_pass_attempts_threshold": QB_MIN_PASS_ATTEMPTS,
    }
    return qb_ids, diagnostics


def build_credited_production(
    pbp: pd.DataFrame, canonical_games: pd.DataFrame, qb_ids: set[str]
) -> pd.DataFrame:
    """Per (game, team, player) credited skill EPA and action count.

    Rush credit goes to ``rusher_player_id`` (EPA_rush, fallback EPA);
    reception credit goes to ``receiver_player_id`` on a
    ``type.text == "Pass Reception"`` row (EPA_pass, fallback EPA) -- the
    same reception definition ``cfb_roles.cfb_role_actions`` uses. Passer
    credit is never read at all: QB skill_epa is excluded by construction,
    the same way ``players.py`` hard-zeroes it, plus any identified QB's
    rush credits are also dropped (item 2 in the module docstring). Each
    individual play's EPA is clipped to +/- ``EPA_CLIP`` before summing to
    the game level, mirroring the precedent value-weighted continuity
    screen's per-play clip.
    """

    require_columns(canonical_games, ("game_id", "gameday"), "cfb canonical games")
    games = canonical_games.loc[:, ["game_id", "gameday"]].copy()
    games["game_id"] = games["game_id"].astype(str)
    games = games.drop_duplicates("game_id")

    working = pbp.copy()
    working["game_id"] = working["game_id"].astype(str)
    working["season"] = pd.to_numeric(working["season"], errors="coerce")
    working["week"] = pd.to_numeric(working["week"], errors="coerce")
    working = working.merge(games, on="game_id", how="inner", validate="many_to_one")
    working = working.loc[
        working["season"].notna()
        & working["week"].notna()
        & working["season"].between(FROZEN_ROLE_SEASONS[0], FROZEN_ROLE_SEASONS[1])
    ].copy()
    working = working.loc[_regular_season_mask(working["seasonType"])].copy()
    working["season"] = working["season"].astype("int64")
    working["week"] = working["week"].astype("int64")

    rush_flag = pd.to_numeric(working["rush"], errors="coerce").fillna(0).astype(bool)
    reception_flag = working["type.text"].astype("string").eq("Pass Reception")
    definitions = {
        "rush": (rush_flag, working["rusher_player_id"], working["EPA_rush"]),
        "reception": (reception_flag, working["receiver_player_id"], working["EPA_pass"]),
    }
    frames: list[pd.DataFrame] = []
    for _, (eligible, credit_column, epa_column) in definitions.items():
        credited = eligible & credit_column.notna()
        rows = working.loc[credited, ["game_id", "gameday", "season", "week", "pos_team"]].copy()
        rows["player_id"] = credit_column.loc[credited].astype(str)
        epa_specific = pd.to_numeric(epa_column.loc[credited], errors="coerce")
        epa_fallback = pd.to_numeric(working.loc[credited, "EPA"], errors="coerce")
        rows["epa"] = epa_specific.where(epa_specific.notna(), epa_fallback)
        frames.append(rows)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.loc[~combined["player_id"].isin(qb_ids)].copy()
    combined["epa"] = combined["epa"].clip(lower=-EPA_CLIP, upper=EPA_CLIP)
    combined = combined.rename(columns={"pos_team": "team"})

    player_game = (
        combined.groupby(["game_id", "gameday", "season", "week", "team", "player_id"], sort=False)
        .agg(epa_sum=("epa", "sum"), actions=("epa", "size"))
        .reset_index()
    )
    return player_game


# ---------------------------------------------------------------------------
# Step 2: walk-forward state simulation -- OLD (shrink-to-zero) and NEW
# (James-Stein toward a position prior) constructs computed together, since
# they share the same active-roster iteration and only differ in the
# per-player shrinkage formula.
# ---------------------------------------------------------------------------


@dataclass
class _UnitTrail:
    value_state: float = math.nan  # EWM of per-game credited skill EPA (clipped)
    actions_state: float = math.nan  # EWM of per-game credited action count
    career_actions: float = 0.0  # running total, mirrors career_offense_snaps
    appearances: int = 0
    missed_streak: int = 0
    last_season: int = -1
    welford_mean: float = math.nan
    welford_m2: float = 0.0
    welford_n: int = 0


@dataclass
class _SimDiagnostics:
    team_games_evaluated: int = 0
    empty_active_mass_rows: int = 0
    empty_experienced_pool_weeks: int = 0
    total_week_snapshots: int = 0
    b_values: list[float] = field(default_factory=list)
    b_floored_zero: int = 0
    b_floored_one: int = 0
    tau_between_values: list[float] = field(default_factory=list)
    tau_within_values: list[float] = field(default_factory=list)
    prior_g_values: list[float] = field(default_factory=list)
    experienced_pool_sizes: list[int] = field(default_factory=list)


def _welford_update(trail: _UnitTrail, value: float) -> None:
    trail.welford_n += 1
    delta = value - (0.0 if not math.isfinite(trail.welford_mean) else trail.welford_mean)
    if not math.isfinite(trail.welford_mean):
        trail.welford_mean = value
    else:
        trail.welford_mean += delta / trail.welford_n
    delta2 = value - trail.welford_mean
    trail.welford_m2 += delta * delta2


def _snapshot_prior(
    trails_by_team: dict[str, dict[str, _UnitTrail]], season: int, diag: _SimDiagnostics
) -> tuple[float, float, float]:
    """Recompute (prior_g, tau_between, tau_within) from strictly-prior state."""

    experienced = [
        trail
        for team_trails in trails_by_team.values()
        for trail in team_trails.values()
        if trail.career_actions >= VALUE_PRIOR_ACTIONS
        and trail.last_season == season
        and trail.missed_streak < STREAK_CAP
        and math.isfinite(trail.actions_state)
        and trail.actions_state > 0.0
    ]
    diag.total_week_snapshots += 1
    diag.experienced_pool_sizes.append(len(experienced))
    if len(experienced) < MIN_EXPERIENCED_POOL:
        diag.empty_experienced_pool_weeks += 1
        return 0.0, JS_EPSILON, 0.0

    raws = np.array(
        [100.0 * trail.value_state / trail.actions_state for trail in experienced], dtype=float
    )
    within_vars = [
        trail.welford_m2 / (trail.welford_n - 1) for trail in experienced if trail.welford_n >= 2
    ]
    tau_within = float(np.mean(within_vars)) if within_vars else 0.0
    prior_g = float(np.mean(raws))
    total_var = float(np.var(raws, ddof=1)) if len(raws) >= 2 else 0.0
    mean_sigma2 = float(
        np.mean([tau_within / max(trail.career_actions, 1.0) for trail in experienced])
    )
    tau_between = max(total_var - mean_sigma2, JS_EPSILON)
    diag.prior_g_values.append(prior_g)
    diag.tau_between_values.append(tau_between)
    diag.tau_within_values.append(tau_within)
    return prior_g, tau_between, tau_within


def simulate_unit_value(
    player_game: pd.DataFrame,
) -> tuple[pd.DataFrame, _SimDiagnostics]:
    """Global chronological walk-forward: team-game skill-unit value, old vs. new.

    Processes ``(game_id, team)`` groups in strict (gameday, game_id, team)
    order. The league-wide experienced-player prior/variance snapshot
    refreshes once per ``(season, week)`` block from state strictly before
    that block (item e in the module docstring); each team's own active
    roster is evaluated from that team's state strictly before its own game,
    then updated with this game's credited actions.
    """

    working = player_game.copy()
    working["gameday"] = pd.to_datetime(working["gameday"], errors="raise")
    game_keys = (
        working[["game_id", "team", "gameday", "season", "week"]]
        .drop_duplicates()
        .sort_values(["gameday", "game_id", "team"])
        .reset_index(drop=True)
    )
    grouped_rows = {
        key: sub[["player_id", "epa_sum", "actions"]].reset_index(drop=True)
        for key, sub in working.groupby(["game_id", "team"], sort=False)
    }

    trails_by_team: dict[str, dict[str, _UnitTrail]] = {}
    diag = _SimDiagnostics()
    prior_g, tau_between, tau_within = 0.0, JS_EPSILON, 0.0
    current_week_key: tuple[int, int] | None = None

    rows: list[dict[str, Any]] = []
    for record in game_keys.itertuples(index=False):
        game_id, team, gameday, season, week = (
            str(record.game_id),
            str(record.team),
            record.gameday,
            int(record.season),
            int(record.week),
        )
        week_key = (season, week)
        if week_key != current_week_key:
            prior_g, tau_between, tau_within = _snapshot_prior(trails_by_team, season, diag)
            current_week_key = week_key

        team_players = trails_by_team.setdefault(team, {})
        active = {
            player_id: trail
            for player_id, trail in team_players.items()
            if trail.appearances >= 1
            and trail.last_season == season
            and trail.missed_streak < STREAK_CAP
            and math.isfinite(trail.actions_state)
            and trail.actions_state > 0.0
        }
        diag.team_games_evaluated += 1
        if not active:
            diag.empty_active_mass_rows += 1
            value_old, value_js, n_active = 0.0, 0.0, 0
        else:
            weights = {pid: trail.actions_state for pid, trail in active.items()}
            total_weight = sum(weights.values())
            value_old = 0.0
            value_js = 0.0
            for player_id, trail in active.items():
                raw_i = 100.0 * trail.value_state / trail.actions_state
                reliability_old = trail.career_actions / (
                    trail.career_actions + VALUE_PRIOR_ACTIONS
                )
                shrunk_old = raw_i * reliability_old
                sigma_i2 = tau_within / max(trail.career_actions, 1.0)
                b_i = (
                    tau_between / (tau_between + sigma_i2) if (tau_between + sigma_i2) > 0 else 0.0
                )
                b_i = min(max(b_i, 0.0), 1.0)
                if b_i <= 0.0:
                    diag.b_floored_zero += 1
                if b_i >= 1.0:
                    diag.b_floored_one += 1
                diag.b_values.append(b_i)
                shrunk_js = prior_g + b_i * (raw_i - prior_g)
                w = weights[player_id] / total_weight
                value_old += w * shrunk_old
                value_js += w * shrunk_js
            n_active = len(active)

        rows.append(
            {
                "game_id": game_id,
                "team": team,
                "season": season,
                "week": week,
                "gameday": gameday,
                "value_old": value_old,
                "value_js": value_js,
                "active_players": n_active,
                "prior_g": prior_g,
                "tau_between": tau_between,
                "tau_within": tau_within,
            }
        )

        this_game = grouped_rows.get((game_id, team))
        appeared_ids = set(this_game["player_id"].astype(str)) if this_game is not None else set()
        for player_id in team_players:
            if player_id not in appeared_ids:
                team_players[player_id].missed_streak += 1

        if this_game is not None:
            alpha = 2.0 / (VALUE_SPAN + 1.0)
            for row in this_game.itertuples(index=False):
                player_id = str(row.player_id)
                trail = team_players.setdefault(player_id, _UnitTrail())
                epa_sum = float(row.epa_sum)
                actions = float(row.actions)
                trail.value_state = (
                    epa_sum
                    if not math.isfinite(trail.value_state)
                    else alpha * epa_sum + (1.0 - alpha) * trail.value_state
                )
                trail.actions_state = (
                    actions
                    if not math.isfinite(trail.actions_state)
                    else alpha * actions + (1.0 - alpha) * trail.actions_state
                )
                per_game_rate = 100.0 * epa_sum / actions if actions > 0 else 0.0
                _welford_update(trail, per_game_rate)
                trail.career_actions += actions
                trail.appearances += 1
                trail.missed_streak = 0
                trail.last_season = season

    result = pd.DataFrame(rows)
    return result, diag


# ---------------------------------------------------------------------------
# Step 3: split-half reliability (BEFORE the accuracy screen; AGENTS.md)
# ---------------------------------------------------------------------------


def split_half_reliability(
    team_value: pd.DataFrame, value_column: str, *, seed: int, n_boot: int = 4000
) -> dict[str, Any]:
    """Odd/even-week split-half reliability of one team-game value series.

    Same recipe as ``docs/injury_value_lost.md`` Sec 3.1 / ``cfb_role_features.md``
    Sec 6 / ``cfb_value_weighted_continuity_screen.py``'s function of the
    same name: split each team-season's values by odd/even week, correlate
    the two halves' team-season means, Spearman-Brown correct to
    full-length reliability, bootstrap a CI and ``probability_positive``.
    Implemented inline per adjudicated decision 5 (no src/ change this pass).
    """

    from scipy.stats import spearmanr

    subset = team_value.copy()
    subset["half"] = np.where(subset["week"] % 2 == 0, "even", "odd")
    means = subset.groupby(["team", "season", "half"])[value_column].mean().unstack("half")
    counts = subset.groupby(["team", "season", "half"]).size().unstack("half")
    valid = (counts.get("odd", 0) >= 2) & (counts.get("even", 0) >= 2)
    means = means.dropna()
    valid = valid.reindex(means.index).fillna(False)
    means = means.loc[valid]

    odd_vals = means["odd"].to_numpy(dtype=float)
    even_vals = means["even"].to_numpy(dtype=float)
    n = len(means)
    if n < 5:
        return {"value_column": value_column, "n_team_seasons": n, "note": "too few team-seasons"}
    r = float(np.corrcoef(odd_vals, even_vals)[0, 1])
    rho = float(spearmanr(odd_vals, even_vals).correlation)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sel = rng.integers(0, n, size=n)
        boots[i] = np.corrcoef(odd_vals[sel], even_vals[sel])[0, 1]
    sb = (2.0 * r) / (1.0 + r) if r > -1.0 else float("nan")
    return {
        "value_column": value_column,
        "n_team_seasons": n,
        "pearson_r": r,
        "pearson_r_ci95": [
            float(np.nanquantile(boots, 0.025)),
            float(np.nanquantile(boots, 0.975)),
        ],
        "spearman_rho": rho,
        "spearman_brown_full_length_reliability": sb,
        "probability_positive": float(np.mean(boots > 0.0)),
    }


# ---------------------------------------------------------------------------
# Step 4: attach the NEW (James-Stein) construct to canonical games and run
# the accuracy screen against the frozen XLG-03 benchmark
# ---------------------------------------------------------------------------


def attach_unit_value(
    canonical_games: pd.DataFrame, team_value: pd.DataFrame, team_ids: pd.DataFrame
) -> pd.DataFrame:
    """Join home/away/diff ``skill_unit_value`` (JS construct) onto canonical games.

    Mirrors ``cfb_role_features.attach_role_continuity``'s team-id join
    exactly (team NAMES never match across pbp and canonical sources; only
    ESPN team ids do), with an additive neutral fallback of 0.0 (this is a
    value SUM, not a ratio, so 0.0 -- "no known skill value" -- is the
    correct neutral, unlike continuity's ratio-neutral of 1.0).
    """

    require_columns(canonical_games, ("game_id", "home_id", "away_id"), "cfb canonical games")
    require_columns(team_ids, ("game_id", "team", "team_id"), "team id map")

    result = canonical_games.copy()
    result["game_id"] = result["game_id"].astype(str)

    mapping = team_ids.copy()
    mapping["game_id"] = mapping["game_id"].astype(str)
    mapping["team"] = mapping["team"].astype(str)
    mapping["team_id"] = pd.to_numeric(mapping["team_id"], errors="coerce").astype("Int64")
    mapping = mapping.dropna(subset=["team_id"]).drop_duplicates(["game_id", "team"])
    mapping["team_id"] = mapping["team_id"].astype(str)

    working = team_value.copy()
    working["game_id"] = working["game_id"].astype(str)
    working["team"] = working["team"].astype(str)
    working = working.merge(
        mapping[["game_id", "team", "team_id"]], on=["game_id", "team"], how="left"
    )
    unmapped = working["team_id"].isna()
    if unmapped.any():
        examples = ", ".join(
            working.loc[unmapped, "team"].astype(str).drop_duplicates().head(5).tolist()
        )
        raise DataContractError(f"Unit-value teams missing a team id: {examples}")

    lookup = working.set_index(["game_id", "team_id"])["value_js"]

    def _normalized_id(values: pd.Series) -> pd.Series:
        return pd.to_numeric(values, errors="coerce").astype("Int64").astype(str)

    for side in ("home", "away"):
        keys = pd.MultiIndex.from_arrays([result["game_id"], _normalized_id(result[f"{side}_id"])])
        values = pd.Series(lookup.reindex(keys).to_numpy(dtype=float), index=result.index)
        result[f"{side}_skill_unit_value"] = values.fillna(0.0)
    result["diff_skill_unit_value"] = (
        result["home_skill_unit_value"] - result["away_skill_unit_value"]
    )
    return result


def js_unit_value_benchmark(
    features: pd.DataFrame,
    *,
    start_season: int = CFB_BENCHMARK_START_SEASON,
    end_season: int = CFB_BENCHMARK_END_SEASON,
    min_train_games: int = CFB_BENCHMARK_MIN_TRAIN_GAMES,
    ridge_alpha: float = CFB_BENCHMARK_RIDGE_ALPHA,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Walk-forward: market, frozen market_residual, and the JS-unit-value candidate.

    Structurally identical to ``cfb_role_features.cfb_role_benchmark`` /
    ``cfb_value_weighted_continuity_screen.value_role_benchmark`` (same
    weeks, same training-window/Ridge recipe, same paired-comparison
    machinery); the only difference in the candidate arm is the three
    ``CANDIDATE_COLUMNS`` appended to the frozen XLG-03 contract.
    """

    required = {*_PREDICTION_PASSTHROUGH, *CFB_MODEL_FEATURE_COLUMNS, *CANDIDATE_COLUMNS}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(
            f"CFB JS unit-value features are missing columns: {', '.join(missing)}"
        )
    candidate_values = features.loc[:, list(CANDIDATE_COLUMNS)]
    if bool(candidate_values.nunique(dropna=False).le(1).all()):
        raise DataContractError(
            "Every skill-unit-value column is constant; the feature join produced no "
            "information, so a benchmark comparison would be vacuous"
        )

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
    ].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    test = completed.loc[completed["season"].between(start_season, end_season)]
    if test.empty:
        raise ValueError(f"No completed CFB games found from {start_season} to {end_season}")

    batches: list[pd.DataFrame] = []
    for (_, _), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        models = {
            "market": fit_market_baseline(training),
            "market_residual": fit_cfb_residual_model(training, ridge_alpha=ridge_alpha),
            "market_residual_js_unit_value": fit_cfb_residual_model(
                training, ridge_alpha=ridge_alpha, feature_columns=JS_ROLE_COLUMNS
            ),
        }
        batches.extend(_score_week(weekly_games, models))
    if not batches:
        raise ValueError("No CFB week had enough prior training games")

    predictions = pd.concat(batches, ignore_index=True)
    predictions["evaluation_window"] = predictions["season"].map(
        lambda season: cfb_evaluation_window(int(season))
    )
    predictions = predictions.sort_values(["gameday", "game_id", "method"]).reset_index(drop=True)
    summary, season_summary = summarize_cfb_benchmark(predictions)

    clean = predictions.loc[
        predictions["evaluation_window"].eq("clean_core")
        & predictions["method"].isin(("market_residual", "market_residual_js_unit_value"))
    ].copy()
    if clean.empty:
        raise ValueError("No clean-core predictions available for the paired comparison")
    clean["feature_set"] = clean["method"].map(
        {"market_residual": BASELINE_ARM, "market_residual_js_unit_value": CANDIDATE_ARM}
    )
    paired = pd.concat(
        [
            paired_feature_comparisons(
                clean,
                baseline_feature_set=BASELINE_ARM,
                samples=bootstrap_samples,
                block="week",
                seed=bootstrap_seed,
                on_degenerate="raise",
            ),
            paired_feature_comparisons(
                clean,
                baseline_feature_set=BASELINE_ARM,
                samples=bootstrap_samples,
                block="season",
                seed=bootstrap_seed,
                on_degenerate="raise",
            ),
        ],
        ignore_index=True,
    )
    baseline_probs = clean.loc[clean["feature_set"].eq(BASELINE_ARM)].sort_values("game_id")[
        "home_cover_probability"
    ]
    candidate_probs = clean.loc[clean["feature_set"].eq(CANDIDATE_ARM)].sort_values("game_id")[
        "home_cover_probability"
    ]
    f_disagree = picks_differ_fraction(baseline_probs.to_numpy(), candidate_probs.to_numpy())
    n_games = clean["game_id"].nunique()
    return {
        "predictions": predictions,
        "summary": summary,
        "season_summary": season_summary,
        "paired": paired,
        "clean_paired_games": n_games,
        "picks_differ_fraction": f_disagree,
        "mde80_accuracy_points": mde80(f_disagree, n_games),
    }


def _describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    started = time.perf_counter()
    timings: dict[str, float] = {}
    output = OUT_ROOT / run_id()
    print(f"Artifact directory: {output}", flush=True)

    print("=== Step 1: load pbp and build credited skill-production table ===", flush=True)
    t0 = time.perf_counter()
    pbp, canonical_games = load_inputs()
    qb_ids, qb_diag = identify_qb_ids(pbp)
    player_game = build_credited_production(pbp, canonical_games, qb_ids)
    team_ids = (
        pbp.loc[
            pbp["pos_team"].notna() & pbp["pos_team_id"].notna(),
            ["game_id", "pos_team", "pos_team_id"],
        ]
        .rename(columns={"pos_team": "team", "pos_team_id": "team_id"})
        .drop_duplicates(["game_id", "team"])
    )
    timings["step1_load_and_credits_seconds"] = time.perf_counter() - t0
    print(
        f"pbp rows={len(pbp)} player_game rows={len(player_game)} qb_diag={json.dumps(qb_diag)}",
        flush=True,
    )

    print("\n=== Step 2: walk-forward state simulation (old vs. James-Stein) ===", flush=True)
    t0 = time.perf_counter()
    team_value, sim_diag = simulate_unit_value(player_game)
    timings["step2_simulation_seconds"] = time.perf_counter() - t0
    sim_report = {
        "team_games_evaluated": sim_diag.team_games_evaluated,
        "empty_active_mass_rows": sim_diag.empty_active_mass_rows,
        "empty_active_mass_fraction": (
            sim_diag.empty_active_mass_rows / sim_diag.team_games_evaluated
            if sim_diag.team_games_evaluated
            else math.nan
        ),
        "total_week_snapshots": sim_diag.total_week_snapshots,
        "empty_experienced_pool_weeks": sim_diag.empty_experienced_pool_weeks,
        "empty_experienced_pool_fraction": (
            sim_diag.empty_experienced_pool_weeks / sim_diag.total_week_snapshots
            if sim_diag.total_week_snapshots
            else math.nan
        ),
        "experienced_pool_size": _describe(sim_diag.experienced_pool_sizes),
        "b_i_summary": _describe(sim_diag.b_values),
        "b_floored_zero_fraction": (
            sim_diag.b_floored_zero / len(sim_diag.b_values) if sim_diag.b_values else math.nan
        ),
        "b_floored_one_fraction": (
            sim_diag.b_floored_one / len(sim_diag.b_values) if sim_diag.b_values else math.nan
        ),
        "tau_between_summary": _describe(sim_diag.tau_between_values),
        "tau_within_summary": _describe(sim_diag.tau_within_values),
        "prior_g_summary": _describe(sim_diag.prior_g_values),
        "value_old_summary": _describe(team_value["value_old"].tolist()),
        "value_js_summary": _describe(team_value["value_js"].tolist()),
    }
    print(json.dumps(sim_report, indent=2, default=str), flush=True)

    print(
        "\n=== Step 3: split-half reliability, OLD vs. NEW construct "
        "(BEFORE the accuracy screen) ===",
        flush=True,
    )
    t0 = time.perf_counter()
    reliability_results = {
        "old_shrink_to_zero": split_half_reliability(team_value, "value_old", seed=1),
        "new_james_stein": split_half_reliability(team_value, "value_js", seed=2),
    }
    timings["step3_split_half_reliability_seconds"] = time.perf_counter() - t0
    print(json.dumps(reliability_results, indent=2, default=str), flush=True)

    print("\n=== Step 4: attach the JS construct and run the accuracy screen ===", flush=True)
    t0 = time.perf_counter()
    features = attach_unit_value(canonical_games, team_value, team_ids)
    benchmark = js_unit_value_benchmark(features)
    timings["step4_accuracy_screen_seconds"] = time.perf_counter() - t0
    paired = benchmark["paired"]
    accuracy_week = paired.loc[
        paired["metric"].eq("accuracy_improvement") & paired["block"].eq("week")
    ].iloc[0]
    accuracy_season = paired.loc[
        paired["metric"].eq("accuracy_improvement") & paired["block"].eq("season")
    ].iloc[0]
    brier_week = paired.loc[
        paired["metric"].eq("brier_improvement") & paired["block"].eq("week")
    ].iloc[0]
    logloss_week = paired.loc[
        paired["metric"].eq("log_loss_improvement") & paired["block"].eq("week")
    ].iloc[0]
    screen_summary = {
        "clean_paired_games": int(benchmark["clean_paired_games"]),
        "picks_differ_fraction": benchmark["picks_differ_fraction"],
        "mde80_accuracy_points": benchmark["mde80_accuracy_points"],
        "accuracy_improvement_week": accuracy_week.to_dict(),
        "accuracy_improvement_season": accuracy_season.to_dict(),
        "brier_improvement_week": brier_week.to_dict(),
        "log_loss_improvement_week": logloss_week.to_dict(),
    }
    print(json.dumps(screen_summary, indent=2, default=str), flush=True)

    print("\n=== Step 5: write artifacts ===", flush=True)
    atomic_parquet(team_value, output / "team_value.parquet")
    stamp_sidecar(output / "team_value.parquet")  # ENG-38
    atomic_parquet(benchmark["predictions"], output / "predictions.parquet")
    stamp_sidecar(output / "predictions.parquet")  # ENG-38
    atomic_csv(benchmark["summary"], output / "summary.csv")
    stamp_sidecar(output / "summary.csv")  # ENG-38
    atomic_csv(benchmark["season_summary"], output / "season_summary.csv")
    stamp_sidecar(output / "season_summary.csv")  # ENG-38
    atomic_csv(paired, output / "paired_comparisons.csv")
    stamp_sidecar(output / "paired_comparisons.csv")  # ENG-38
    atomic_json(reliability_results, output / "split_half_reliability.json")
    atomic_json(sim_report, output / "simulation_diagnostics.json")
    atomic_json(qb_diag, output / "qb_identification_diagnostics.json")

    timings["total_seconds"] = time.perf_counter() - started
    metadata = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/cfb_james_stein_unit_screen.py",
        "predeclaration_source": (
            "scratchpad mod06_scope/predeclaration_draft.md + implementation_plan.md, "
            "adjudicated by the reviewing orchestrator (8 decisions)"
        ),
        "candidate_arm": CANDIDATE_ARM,
        "baseline_arm": BASELINE_ARM,
        "constants": {
            "value_prior_actions": VALUE_PRIOR_ACTIONS,
            "value_span": VALUE_SPAN,
            "streak_cap": STREAK_CAP,
            "qb_min_pass_attempts": QB_MIN_PASS_ATTEMPTS,
            "epa_clip": EPA_CLIP,
            "min_experienced_pool": MIN_EXPERIENCED_POOL,
            "js_epsilon": JS_EPSILON,
        },
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "role_seasons": list(FROZEN_ROLE_SEASONS),
        "clean_core_seasons": list(CFB_CLEAN_CORE_SEASONS),
        "qb_identification_diagnostics": qb_diag,
        "simulation_diagnostics": sim_report,
        "split_half_reliability": reliability_results,
        "screen_summary": screen_summary,
        "timing": timings,
    }
    write_stamped_artifact(metadata, output / "metadata.json")  # ENG-38
    print(f"\nWrote artifacts to {output}", flush=True)
    print(f"Total runtime: {timings['total_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
