"""Injury-signal refresh tilt (POL-11 follow-on): front-run the market's own
injury-driven moves, instead of waiting for the observed-movement policy to
confirm them after the fact.

**Binding closing-grounds taxonomy (AGENTS.md), restated verbatim per this
project's rule for any module that scores or adjudicates an experiment:** an
interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is
the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is ``unresolved_below_power``: record it with ``nfl-ats weak-signals
record``, report ``probability_positive``, never the binary "contains zero."

Evidence chain motivating this challenger (read this session from
``docs/movement_attribution.md`` and ``docs/injury_news_sourcing.md``
section 5.1, both already-recorded registry results, not re-measured here):

* Flipping to the market's side on an adverse Tuesday-to-close move is worth
  **+5.26 accuracy points** across the whole disagreement population
  (n=494, week-blocked 95% [-2.86, +14.12], ``probability_positive`` 0.880),
  and that value **concentrates in moves attributed to injury news**:
  **+17.07 points** at ``|open_move| >= 1.0`` (n=123, interval
  [+0.79, +31.67], ``probability_positive`` 0.976) -- ``docs/movement_attribution.md``'s
  ``pop_threshold_injury`` cell, itself a **correlated decomposition** of the
  already-recorded ``observed_movement_*`` family, not an independent
  sample.
* Independently, the Tuesday-to-Saturday injury-news channel itself (a
  *different* construct -- ``injury_value_lost``'s ``value_lost_diff``, not
  this module's ``net_injury_score``) reads **+1.32 to +1.54 accuracy
  points** between what is knowable at Tuesday-publish time and a
  Saturday-ish decision cutoff (``docs/injury_news_sourcing.md`` section
  5.1, ``probability_positive`` 0.90/0.92).

Both readings say the same thing from different angles: post-Tuesday injury
news is real, ingestible, pregame-safe information the market prices before
this project's Tuesday-locked card does. ``docs/movement_attribution.md``'s
own "Front-running sketch" section predeclares the natural next question --
"does acting on the injury signal itself, at a refresh pass, beat waiting for
the market to move" -- and explicitly defers testing it (see that section's
item 5: "This document does not test the lag... the natural next study").
This module is that follow-on, wired as a dual-tracked challenger exactly
the way ``model_only_refresh_incumbent`` tracks its own counterfactual arm
inside ``nfl_ats.pick_refresh`` -- **nothing here changes the production
observed-movement >=1.0 policy**, which stays exactly as
``nfl_ats.pick_refresh.plan_refresh`` already wires it.

Two honest caveats, stated up front rather than discovered later:

1. **Timing mismatch.** The backtested +17.07-point figure is graded
   Tuesday-to-CLOSE (the whole week's eventual move). This challenger's live
   arm acts at whatever instant a ``refresh-picks`` pass actually runs
   (Thursday afternoon, Saturday, Sunday morning...), which may be well
   before the line has finished moving. The two are not the same
   measurement, and this challenger's own prospective evidence is what
   settles whether front-running captures comparable value or less.
2. **Correlated decomposition, not independent confirmation.** The
   +17.07-point figure and the whole ``movement_attribution_*`` family it
   belongs to are subcuts of the ALREADY-RECORDED ``observed_movement_*``
   entries (same archive, same population) -- evidence for a mechanism, not
   a second independent replication of it. See
   ``docs/movement_attribution.md``'s own commensurability note.

The construction below is a LIVE port of ``docs/movement_attribution.md``'s
(a) INJURY class, reused verbatim wherever the backtest's "final" cutoff
(that game's own kickoff) becomes this module's live decision instant
(``now``, the refresh pass's own clock):

* Severity scale (predeclared there, reused here unchanged): ``Out=4,
  Doubtful=3, Questionable=2, Probable=1, not on report=0``.
* Skill positions only (``QB``/``RB``/``WR``/``TE``) -- the same disclosed
  proxy for "market-relevant player" (offensive-line/defensive-front
  injuries are invisible to this construction, a likely undercount, not an
  overcount).
* ``player_delta = current_severity - tuesday_severity`` (own-week Tuesday
  noon ET, computed the same way as
  ``scripts/injury_tuesday_cutoff_experiment.py``'s
  ``team_week_tuesday_noon``), ``team_injury_delta = sum(player_delta)``
  over that team's skill-position players, ``net_injury_score =
  team_injury_delta(picked_team) - team_injury_delta(opponent_team)``.
  ``net_injury_score >= 2`` fires (the identical predeclared bar).
* **Official-report path** whenever the target season has ANY official
  injury-report coverage in the locally ingested snapshot (checked
  dynamically, not hardcoded to a season boundary, since the whole point of
  a live challenger is to pick up 2026's own official reports once they are
  ingested); **PFT-headline fallback** otherwise (``net_pft_score >= 1``,
  the identical predeclared bar), reading whatever local
  ``data/raw/injury_news/<snapshot>/index.parquet`` bulk-scrape happens to
  exist -- a manually re-run, private research archive
  (``scripts/ingest_injury_news.py``), not something ``weekly-run``
  refreshes automatically. **FAIL-OPEN everywhere**: no official coverage
  and no PFT snapshot -> zero signal, the hold pick plays, never a raised
  error and never a blocked recording.

One live-only caveat that has no backtest analog: the official-report path
is only as fresh as the LOCAL ``injuries.parquet`` snapshot. A refresh pass
run before that week's player-ingest has been re-run will see only
Tuesday-dated rows (so ``current_severity == tuesday_severity`` for every
player, net score 0, quietly no-op) even if real Wednesday/Thursday/Friday
filings already exist upstream -- an operational freshness gap, not a
mechanism failure, and not distinguishable from "no post-Tuesday news
happened" without re-running player-ingest first. Disclosed here and in
``artifacts/prospective/challengers.json``.

Trigger, precisely: at each ``nfl-ats refresh-picks`` pass, for every game
not yet at its own deadline (``RefreshedGame.eligible``, the SAME
kickoff-or-Sunday-4pm-ET rule the production refresh already uses), compute
the injury signal for the model's own current-week pick (``model_only_pick_side``
-- the "hold" arm, untouched by any overlay or the movement policy) against
its opponent. When the signal fires, the "flip" arm (``injury_tilt_pick_side``)
takes the other side; otherwise it equals the hold arm. BOTH arms are
recorded on every eligible row of the append-only injury-signal ledger,
alongside that SAME pass's observed-movement-policy diagnostics
(``movement_policy``/``movement_delta``/``movement_pick_side``) and a
``disagreement_type`` classification -- the exact population that adjudicates
front-running value: cases where this signal wants to flip a game the
market-movement policy has NOT (yet) confirmed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.pick_refresh import (
    MOVEMENT_POLICY_MOVEMENT,
    RefreshResult,
    original_card,
)

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "injury_signal_refresh_tilt"

#: Predeclared in docs/movement_attribution.md, reused here unchanged.
SEVERITY: dict[str, float] = {"Out": 4.0, "Doubtful": 3.0, "Questionable": 2.0, "Probable": 1.0}
SKILL_POSITIONS: frozenset[str] = frozenset({"QB", "RB", "WR", "TE"})
INJURY_NET_THRESHOLD = 2.0
PFT_NET_THRESHOLD = 1.0

SOURCE_OFFICIAL = "official"
SOURCE_PFT_FALLBACK = "pft_fallback"
SOURCE_NONE = "none"

#: injury fires, movement below threshold (or no fresh line) -- the pure
#: front-running population this challenger exists to measure.
DISAGREEMENT_INJURY_ONLY = "injury_only"
#: movement fired (>=1.0 pt), injury signal did not -- the market moved for
#: some other visible-or-invisible reason.
DISAGREEMENT_MOVEMENT_ONLY = "movement_only"
#: both mechanisms want to move the pick, and to the SAME side.
DISAGREEMENT_BOTH_AGREE = "both_agree"
#: both mechanisms fired, but toward OPPOSITE sides -- a genuine conflict.
DISAGREEMENT_BOTH_DISAGREE = "both_disagree"
#: neither mechanism fired this pass.
DISAGREEMENT_NEITHER = "neither"

# Duplicated verbatim from scripts/movement_attribution.py's TEAM_NICKNAMES
# (itself extending src/nfl_ats/constants.py's TEAM_ABBREVIATION_ALIASES the
# same way scripts/ingest_public_betting.py does), per this repo's
# convention of not importing across scripts/*.py files -- and src/nfl_ats
# modules never import from scripts/ at all.
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
# Shared helpers (duplicated from scripts/movement_attribution.py, per this
# repo's cross-scripts/src convention -- see module docstring)
# ---------------------------------------------------------------------------


def own_week_tuesday_noon_utc(kickoff_utc: pd.Series) -> pd.Series:
    """Own-week Tuesday noon ET, in UTC. Duplicated (not imported) from
    ``scripts/movement_attribution.py``'s identical helper (itself
    duplicated from ``scripts/injury_tuesday_cutoff_experiment.py``'s
    ``team_week_tuesday_noon``), per this repo's convention of not importing
    across ``scripts/*.py`` files or from ``scripts/`` into ``src/nfl_ats``."""

    kickoff_et = kickoff_utc.dt.tz_convert("US/Eastern")
    days_since_tuesday = (kickoff_et.dt.weekday - 1) % 7
    tuesday_date_et = kickoff_et.dt.normalize() - pd.to_timedelta(days_since_tuesday, unit="D")
    tuesday_noon_et = tuesday_date_et + pd.Timedelta(hours=12)
    return cast(pd.Series, tuesday_noon_et.dt.tz_convert("UTC"))


def _canonical_team(code: str) -> str:
    return TEAM_ABBREVIATION_ALIASES.get(str(code), str(code))


# ---------------------------------------------------------------------------
# Fail-open source loaders
# ---------------------------------------------------------------------------


def _latest_official_injuries_fail_open(data_root: Path) -> pd.DataFrame | None:
    """The newest locally ingested official injury-report snapshot, or
    ``None`` on ANY failure (no snapshot fetched yet, a malformed source, a
    missing sibling file ``load_player_snapshot`` also requires) -- never
    raises. Mirrors ``interim_hc_first_game_tilt_overlay``'s fail-open
    contract for its own local-snapshot join."""

    from nfl_ats.players import latest_player_snapshot, load_player_snapshot

    try:
        snapshot = latest_player_snapshot(data_root / "players" / "raw")
        injuries, _rosters, _snaps = load_player_snapshot(snapshot)
    except Exception as exc:  # deliberate fail-open, see docstring
        warnings.warn(
            f"{CHALLENGER_ID}: no official injury-report snapshot available, falling back "
            f"to PFT news / no signal ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return injuries


def _latest_pft_index_fail_open(data_root: Path) -> pd.DataFrame | None:
    """The newest local ``scripts/ingest_injury_news.py`` bulk-scrape
    snapshot's ``index.parquet``, restricted to ``injury_relevant`` rows, or
    ``None`` on ANY failure -- never raises. This archive is a manually
    re-run private research pull, not something ``weekly-run`` refreshes
    automatically; a stale or absent snapshot is the expected common case,
    not an error."""

    root = data_root / "raw" / "injury_news"
    try:
        candidates = sorted(path for path in root.glob("*") if (path / "index.parquet").is_file())
        if not candidates:
            raise FileNotFoundError(f"No injury-news snapshot under {root}")
        pft = pd.read_parquet(candidates[-1] / "index.parquet")
    except Exception as exc:  # deliberate fail-open, see docstring
        warnings.warn(
            f"{CHALLENGER_ID}: no PFT injury-news snapshot available, proceeding with no "
            f"fallback signal ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    required = {"lastmod", "injury_relevant", "headline_guess"}
    missing = required.difference(pft.columns)
    if missing:
        warnings.warn(
            f"{CHALLENGER_ID}: PFT injury-news snapshot is missing columns "
            f"{sorted(missing)}, proceeding with no fallback signal",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    pft = pft.loc[pft["injury_relevant"]].copy()
    pft["lastmod"] = pd.to_datetime(pft["lastmod"], utc=True, errors="coerce")
    pft = pft.dropna(subset=["lastmod"])
    pft["headline_norm"] = pft["headline_guess"].fillna("")
    return pft


# ---------------------------------------------------------------------------
# The signal itself: docs/movement_attribution.md's INJURY class, live
# ---------------------------------------------------------------------------


def _severity_asof(rows: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    eligible = rows.loc[rows["date_modified"] <= cutoff]
    if eligible.empty:
        return pd.DataFrame(columns=["gsis_id", "severity"])
    eligible = eligible.assign(severity=eligible["report_status"].map(SEVERITY).fillna(0.0))
    eligible = eligible.sort_values("date_modified")
    return eligible.drop_duplicates("gsis_id", keep="last")[["gsis_id", "severity"]]


def _official_team_delta(
    injuries: pd.DataFrame,
    *,
    season: int,
    week: int,
    team: str,
    tuesday_noon_utc: pd.Timestamp,
    now: pd.Timestamp,
) -> float:
    """Sum of (current - Tuesday-noon) skill-position injury severity for one
    team/week -- ``now`` stands in for ``docs/movement_attribution.md``'s
    ``kickoff_utc``/"final" cutoff, since this is a LIVE decision input, not
    a backward-looking attribution. A team/week with no matching rows scores
    0.0 (healthy), not missing -- identical convention to the backtest's
    ``fillna(0.0)`` on ``team_injury_delta``."""

    scoped = injuries.loc[
        injuries["season"].eq(season)
        & injuries["week"].eq(week)
        & injuries["team"].eq(team)
        & injuries["position"].isin(SKILL_POSITIONS)
    ]
    if scoped.empty:
        return 0.0
    tue = _severity_asof(scoped, tuesday_noon_utc).rename(columns={"severity": "severity_tue"})
    cur = _severity_asof(scoped, now).rename(columns={"severity": "severity_cur"})
    both = tue.merge(cur, on="gsis_id", how="outer")
    both[["severity_tue", "severity_cur"]] = both[["severity_tue", "severity_cur"]].fillna(0.0)
    return float((both["severity_cur"] - both["severity_tue"]).sum())


def _pft_team_hits(pft: pd.DataFrame, team: str, start: pd.Timestamp, end: pd.Timestamp) -> int:
    if end <= start:
        return 0
    window = pft.loc[pft["lastmod"].gt(start) & pft["lastmod"].le(end)]
    if window.empty:
        return 0
    nicknames = TEAM_NICKNAMES.get(team, ())
    if not nicknames:
        return 0
    mask = window["headline_norm"].str.contains("|".join(nicknames), case=False, regex=True)
    return int(mask.sum())


@dataclass(frozen=True)
class InjurySignalReading:
    """One game's injury-signal read, whether or not it fires."""

    game_id: str
    picked_team: str
    opponent_team: str
    source: str  # SOURCE_OFFICIAL / SOURCE_PFT_FALLBACK / SOURCE_NONE
    net_score: float
    threshold: float
    fires: bool


def injury_signal_for_game(
    *,
    game_id: str,
    season: int,
    week: int,
    kickoff: pd.Timestamp,
    picked_team: str,
    opponent_team: str,
    now: pd.Timestamp,
    injuries: pd.DataFrame | None,
    pft: pd.DataFrame | None,
) -> InjurySignalReading:
    """The live asymmetric injury signal for one game's currently-picked
    side vs its opponent. FAIL-OPEN: no official coverage for this season
    AND no PFT snapshot -> zero signal, source ``"none"``, never fires."""

    picked = _canonical_team(picked_team)
    opponent = _canonical_team(opponent_team)
    tuesday_noon = own_week_tuesday_noon_utc(pd.Series([kickoff])).iloc[0]

    if injuries is not None and injuries["season"].eq(season).any():
        delta_picked = _official_team_delta(
            injuries,
            season=season,
            week=week,
            team=picked,
            tuesday_noon_utc=tuesday_noon,
            now=now,
        )
        delta_opponent = _official_team_delta(
            injuries,
            season=season,
            week=week,
            team=opponent,
            tuesday_noon_utc=tuesday_noon,
            now=now,
        )
        net = delta_picked - delta_opponent
        return InjurySignalReading(
            game_id=game_id,
            picked_team=picked,
            opponent_team=opponent,
            source=SOURCE_OFFICIAL,
            net_score=net,
            threshold=INJURY_NET_THRESHOLD,
            fires=net >= INJURY_NET_THRESHOLD,
        )

    if pft is not None:
        hits_picked = _pft_team_hits(pft, picked, tuesday_noon, now)
        hits_opponent = _pft_team_hits(pft, opponent, tuesday_noon, now)
        net = float(hits_picked - hits_opponent)
        return InjurySignalReading(
            game_id=game_id,
            picked_team=picked,
            opponent_team=opponent,
            source=SOURCE_PFT_FALLBACK,
            net_score=net,
            threshold=PFT_NET_THRESHOLD,
            fires=net >= PFT_NET_THRESHOLD,
        )

    return InjurySignalReading(
        game_id=game_id,
        picked_team=picked,
        opponent_team=opponent,
        source=SOURCE_NONE,
        net_score=0.0,
        threshold=float("nan"),
        fires=False,
    )


def classify_disagreement(
    *,
    injury_fires: bool,
    injury_tilt_pick_side: str,
    movement_policy: str,
    movement_pick_side: str,
) -> str:
    """Where the injury signal and the observed-movement policy agree,
    disagree, or fire alone on the SAME game at the SAME refresh pass --
    the exact population that adjudicates front-running value (see module
    docstring)."""

    movement_fires = movement_policy == MOVEMENT_POLICY_MOVEMENT
    if injury_fires and not movement_fires:
        return DISAGREEMENT_INJURY_ONLY
    if injury_fires and movement_fires:
        return (
            DISAGREEMENT_BOTH_AGREE
            if injury_tilt_pick_side == movement_pick_side
            else DISAGREEMENT_BOTH_DISAGREE
        )
    if not injury_fires and movement_fires:
        return DISAGREEMENT_MOVEMENT_ONLY
    return DISAGREEMENT_NEITHER


# ---------------------------------------------------------------------------
# The append-only injury-signal ledger
# ---------------------------------------------------------------------------

INJURY_SIGNAL_LEDGER_COLUMNS: tuple[str, ...] = (
    "revision_recorded_at_utc",
    "refresh_run_id",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "decision_home_spread",
    "hold_pick_side",
    "injury_tilt_pick_side",
    "injury_signal_fires",
    "injury_signal_source",
    "injury_signal_net_score",
    "injury_signal_threshold",
    "movement_policy",
    "movement_delta",
    "movement_pick_side",
    "played_pick_side",
    "disagreement_type",
    "model_id",
    "feature_table_sha256",
)


def injury_signal_ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "injury_signal_refresh_decisions.parquet"


def load_injury_signal_decisions(artifacts_root: Path) -> pd.DataFrame:
    """The append-only injury-signal ledger (empty frame when none exists)."""

    path = injury_signal_ledger_path(artifacts_root)
    if not path.is_file():
        return pd.DataFrame(columns=list(INJURY_SIGNAL_LEDGER_COLUMNS))
    ledger = pd.read_parquet(path)
    missing = sorted(set(INJURY_SIGNAL_LEDGER_COLUMNS).difference(ledger.columns))
    if missing:
        raise DataContractError(
            f"Injury-signal refresh ledger is missing columns: {', '.join(missing)}"
        )
    return ledger[list(INJURY_SIGNAL_LEDGER_COLUMNS)]


def build_injury_signal_rows(plan: RefreshResult, *, data_root: Path) -> pd.DataFrame:
    """Pure computation: one row per ELIGIBLE game in ``plan`` (not yet at
    its own deadline), regardless of whether the PLAYED pick changed this
    pass. Unlike ``pick_revisions.parquet``'s ``changed``-only gate, this
    challenger needs the full disagreement population
    (``docs/movement_attribution.md``'s front-running sketch): games where
    the injury signal fires but the production pick has not (yet) moved are
    exactly the rows this challenger exists to measure, and they are
    invisible under a ``changed``-only gate. Never writes anything -- see
    :func:`record_injury_signal_refresh_tilt` for the append-only write."""

    eligible_games = [game for game in plan.games if game.eligible]
    if not eligible_games:
        return pd.DataFrame(columns=list(INJURY_SIGNAL_LEDGER_COLUMNS))

    injuries = _latest_official_injuries_fail_open(data_root)
    pft = _latest_pft_index_fail_open(data_root)

    rows: list[dict[str, Any]] = []
    for game in eligible_games:
        hold_side = game.model_only_pick_side
        picked_team = game.home_team if hold_side == "HOME" else game.away_team
        opponent_team = game.away_team if hold_side == "HOME" else game.home_team

        reading = injury_signal_for_game(
            game_id=game.game_id,
            season=plan.season,
            week=plan.week,
            kickoff=game.kickoff,
            picked_team=picked_team,
            opponent_team=opponent_team,
            now=plan.computed_at_utc,
            injuries=injuries,
            pft=pft,
        )
        flip_side = "AWAY" if hold_side == "HOME" else "HOME"
        injury_tilt_side = flip_side if reading.fires else hold_side

        disagreement = classify_disagreement(
            injury_fires=reading.fires,
            injury_tilt_pick_side=injury_tilt_side,
            movement_policy=game.movement_policy,
            movement_pick_side=game.movement_pick_side,
        )

        rows.append(
            {
                "revision_recorded_at_utc": plan.computed_at_utc,
                "refresh_run_id": plan.refresh_run_id,
                "season": plan.season,
                "week": plan.week,
                "game_id": game.game_id,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "kickoff": game.kickoff,
                "decision_home_spread": game.decision_home_spread,
                "hold_pick_side": hold_side,
                "injury_tilt_pick_side": injury_tilt_side,
                "injury_signal_fires": bool(reading.fires),
                "injury_signal_source": reading.source,
                "injury_signal_net_score": reading.net_score,
                "injury_signal_threshold": reading.threshold,
                "movement_policy": game.movement_policy,
                "movement_delta": game.movement_delta,
                "movement_pick_side": game.movement_pick_side,
                "played_pick_side": game.new_pick_side,
                "disagreement_type": disagreement,
                "model_id": plan.model_id,
                "feature_table_sha256": plan.feature_table_sha256,
            }
        )
    return pd.DataFrame(rows, columns=list(INJURY_SIGNAL_LEDGER_COLUMNS))


def record_injury_signal_refresh_tilt(
    artifacts_root: Path,
    data_root: Path,
    plan: RefreshResult,
    *,
    record_decisions: bool = False,
) -> dict[str, Any]:
    """Append this pass's injury-signal reading -- BOTH arms, plus the
    disagreement classification against the SAME pass's observed-movement
    policy -- for every eligible game to the append-only injury-signal
    ledger.

    Mirrors ``pick_refresh.record_plan``'s opt-in ``record_decisions``
    contract and its ``refuse_if_outside_recording_lock_window`` guard
    (checked against the ORIGINAL card's kickoffs, exactly like
    ``record_plan``) -- but, unlike ``record_plan``, records EVERY eligible
    game each pass, not only games where the PLAYED pick changed, because
    the disagreement population this challenger exists to measure lives
    disproportionately in games the production pick never touches.

    Model identity is NOT re-checked here: ``plan_refresh`` already raises
    if the active model has changed since the week's original card was
    recorded, before a ``RefreshResult`` can even exist -- the same
    reasoning ``model_only_refresh_incumbent``'s own registration states for
    why it needs no separate fingerprint guard.

    Repeated passes across a week legitimately append MULTIPLE rows per
    game -- deliberately not deduped, because how the signal evolves hour to
    hour is itself part of what this challenger measures
    (``docs/movement_attribution.md``'s front-running sketch, item 5, "does
    not test the lag"). A later settlement pass should read the LATEST row
    per game before kickoff, mirroring ``pick_refresh.final_pick_per_game``;
    no such settlement command exists yet (see
    ``artifacts/prospective/challengers.json``'s ``known_gap`` for this
    challenger, matching ``model_only_refresh_incumbent``'s identical gap).
    """

    if not record_decisions:
        return {
            "recorded": 0,
            "skipped": True,
            "reason": (
                "pass --record-decisions to append this pass's injury-signal reading to "
                "the injury-signal refresh ledger"
            ),
        }

    original = original_card(artifacts_root, season=plan.season, week=plan.week)
    refuse_if_outside_recording_lock_window(
        original["kickoff"], plan.computed_at_utc, ledger="injury-signal-refresh"
    )

    rows = build_injury_signal_rows(plan, data_root=data_root)
    existing = load_injury_signal_decisions(artifacts_root)
    if rows.empty:
        return {
            "recorded": 0,
            "ledger_rows": len(existing),
            "reason": "no eligible games in this refresh pass",
        }

    combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows
    atomic_parquet(
        combined[list(INJURY_SIGNAL_LEDGER_COLUMNS)], injury_signal_ledger_path(artifacts_root)
    )

    fired = rows.loc[rows["injury_signal_fires"], "game_id"].tolist()
    disagreements = rows.loc[
        rows["disagreement_type"].isin((DISAGREEMENT_INJURY_ONLY, DISAGREEMENT_BOTH_DISAGREE)),
        "game_id",
    ].tolist()
    return {
        "recorded": len(rows),
        "ledger_rows": len(combined),
        "games_considered": len(rows),
        "injury_signal_fired_game_ids": fired,
        "disagreement_game_ids": disagreements,
        "source_counts": rows["injury_signal_source"].value_counts().to_dict(),
    }
