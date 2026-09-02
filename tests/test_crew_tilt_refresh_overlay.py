"""Refresh-path wiring of the late-week officiating-crew tilt challenger.

Frozen rule text: ``docs/referee_assignments_capture.md``, section "Late-week
crew-tilt challenger predeclaration (2026-09-01, WP47)", written before the
module existed.

**Binding closing-grounds taxonomy (AGENTS.md), restated verbatim:** an
interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. Only two grounds ever close a line of work: (1) refuted
mechanism -- a RESOLVED wrong sign (whole interval on the wrong side of zero)
or zero split-half reliability; (2) bounded by a positive control proven able
to detect an effect that size. Everything else is
``unresolved_below_power``.

Pins:
- the tilt magnitudes ARE the two registry cells' own measured raw per-game
  gaps -- checked against BOTH the experiments' artifacts and
  ``registry/weak_signals.json``'s own ``classification_evidence`` text, so an
  underived constant cannot survive here;
- the forward crew-trait adapter reproduces the screen's own builders exactly
  on the real snapshots (the measurement the module docstring claims);
- a crew snapshot before a game's pick deadline applies, one at or after it
  never does (anti-backdating, including a post-kickoff snapshot);
- flags fire only in the two predeclared populations, and nothing else moves;
- played-pick INVARIANCE: an unflagged game's would-be pick IS the incumbent's;
- the registered challenger's ``config_fingerprint`` is stable and matches its
  own recorded model block.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import nfl_ats.crew_tilt_refresh_overlay as crew_tilt_refresh_overlay
from nfl_ats.crew_tilt_refresh_overlay import (
    CHALLENGER_ID,
    CREW_TILT_REFRESH_COLUMNS,
    HEAVY_UNDERDOG_THRESHOLD,
    HIGH_FLAG_UNDERDOG_ARTIFACT,
    HIGH_FLAG_UNDERDOG_RAW_GAP_POINTS,
    HIGH_FLAG_UNDERDOG_SIGN,
    HIGH_FLAG_UNDERDOG_SIGNAL,
    HIGH_FLAG_UNDERDOG_TILT,
    HOLDING_RUN_HEAVY_ARTIFACT,
    HOLDING_RUN_HEAVY_RAW_GAP_POINTS,
    HOLDING_RUN_HEAVY_SIGN,
    HOLDING_RUN_HEAVY_SIGNAL,
    HOLDING_RUN_HEAVY_TILT,
    CrewTraitLookup,
    _build_lagged_trait,
    build_crew_tilt_refresh_rows,
    build_crew_trait_lookup,
    crew_tilt_flags,
    latest_crew_snapshot,
    record_crew_tilt_refresh_overlay,
    tilted_probability,
)
from nfl_ats.experiment_runner import (
    _HOLDING_PENALTY_TYPE,
    _build_referee_trait_data,
    _build_referee_type_trait_data,
)
from nfl_ats.four_overlay_composition import POLICY_FINGERPRINT, POLICY_ID
from nfl_ats.pick_refresh import (
    MOVEMENT_POLICY_MODEL_ONLY,
    MOVEMENT_POLICY_MOVEMENT,
    RefreshedGame,
    RefreshResult,
    pick_deadline,
    sunday_pick_lock,
)
from nfl_ats.prospective_scoring import config_fingerprint

_REPO_ROOT = Path(__file__).resolve().parents[1]

SEASON = 2026
WEEK = 2
#: Sunday 1:00 PM ET kickoff; its pick deadline is that day's 16:00 ET lock.
KICKOFF = pd.Timestamp("2026-09-20T17:00:00+00:00")
#: Monday-night kickoff -- deadline is the SUNDAY 16:00 ET lock, which a
#: Wednesday capture still precedes (the SNF/MNF playability claim).
MNF_KICKOFF = pd.Timestamp("2026-09-22T00:15:00+00:00")
WEDNESDAY_CAPTURE = "2026-09-16T19:00:00Z"  # Wed 15:00 ET
SATURDAY_PASS = pd.Timestamp("2026-09-19T15:00:00+00:00")
TUESDAY_RECORD = pd.Timestamp("2026-09-15T16:00:00+00:00")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _lookup() -> CrewTraitLookup:
    """A tiny two-trait lookup with controllable quartiles.

    Eight referees over two seasons, means 1..8, so the lagged qcut(4) puts
    referees 7 and 8 in the TOP quartile and 1 and 2 in the bottom.
    """

    rows = []
    for index, name in enumerate(["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"], start=1):
        rows.append({"official_name": name, "season": 2024, "mean_total": float(index)})
        rows.append({"official_name": name, "season": 2025, "mean_total": float(index)})
    frame = pd.DataFrame(rows)
    trait = _build_lagged_trait(frame)
    return CrewTraitLookup(
        holding=trait,
        flag_rate=trait,
        officials_snapshot_id="fixture-officials",
        penalty_type_snapshot_id="fixture-penalty-types",
    )


def _game(
    *,
    game_id: str,
    new_pick_side: str = "HOME",
    probability: float = 0.52,
    decision_home_spread: float = -2.5,
    kickoff: pd.Timestamp = KICKOFF,
    eligible: bool = True,
    movement: bool = False,
) -> RefreshedGame:
    lock = sunday_pick_lock(pd.Series([KICKOFF]))
    return RefreshedGame(
        game_id=game_id,
        home_team="HME",
        away_team="AWY",
        kickoff=kickoff,
        deadline=pick_deadline(kickoff, lock),
        decision_home_spread=decision_home_spread,
        original_recorded_at_utc=TUESDAY_RECORD,
        previous_pick_side=new_pick_side,
        previous_home_cover_probability=None,
        new_pick_side=new_pick_side,
        new_home_cover_probability=probability,
        decision_policy_id=POLICY_ID,
        decision_policy_fingerprint=POLICY_FINGERPRINT,
        coach_fade_flip=False,
        division_revenge_flip=False,
        player_arrests_flip=False,
        spread_gap_zone_flip=False,
        composed_overlay_flip=False,
        player_arrests_snapshot_id="snapshot-tuesday",
        player_arrests_safe_index_sha256="safe-index-sha",
        movement_policy=(MOVEMENT_POLICY_MOVEMENT if movement else MOVEMENT_POLICY_MODEL_ONLY),
        movement_delta=1.5 if movement else None,
        movement_pick_side=new_pick_side if movement else "",
        model_only_pick_side="HOME" if probability >= 0.5 else "AWAY",
        eligible=eligible,
        ineligible_reason="" if eligible else "kickoff_passed",
        changed=False,
    )


def _plan(games: tuple[RefreshedGame, ...]) -> RefreshResult:
    return RefreshResult(
        season=SEASON,
        week=WEEK,
        refresh_run_id="20260919T150000Z",
        computed_at_utc=SATURDAY_PASS,
        model_id="model-1",
        feature_table_path="unused",
        feature_table_sha256="feature-sha",
        games=games,
        unrefreshable_game_ids=(),
        missing_from_features_game_ids=(),
    )


def _write_crew_snapshot(
    data_root: Path,
    *,
    snapshot_id: str,
    captured_at_utc: str,
    assignments: dict[str, str],
    season: int = SEASON,
    week: int = WEEK,
) -> Path:
    snapshot = data_root / "players" / "referee_assignments" / snapshot_id
    snapshot.mkdir(parents=True)
    frame = pd.DataFrame(
        [
            {
                "captured_at_utc": captured_at_utc,
                "season": season,
                "week": week,
                "game_id": game_id,
                "home_team": "HME",
                "away_team": "AWY",
                "referee": referee,
                "referee_source_name": referee,
                "crew_number": None,
                "game_day_label": "Sunday",
                "source_url": "https://example.invalid/",
            }
            for game_id, referee in assignments.items()
        ]
    )
    frame.to_parquet(snapshot / "assignments.parquet", index=False)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "referee_assignments_snapshot/1",
                "snapshot_id": snapshot_id,
                "captured_at_utc": captured_at_utc,
                "season": season,
                "week": week,
                "row_count": len(frame),
                "empty_reason": None if len(frame) else "not_yet_published",
                "ok": True,
            }
        ),
        encoding="utf-8",
    )
    return snapshot


@pytest.fixture
def patched_traits(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Wire the fixture lookup and a controllable pass-rate quartile map."""

    quartiles: dict[str, int] = {}
    monkeypatch.setattr(
        "nfl_ats.crew_tilt_refresh_overlay.build_crew_trait_lookup",
        lambda _repo_root: _lookup(),
    )
    monkeypatch.setattr(
        "nfl_ats.crew_tilt_refresh_overlay.home_pass_rate_quartiles",
        lambda _repo_root, game_ids: {k: v for k, v in quartiles.items() if k in game_ids},
    )
    return quartiles


# ---------------------------------------------------------------------------
# 1. The tilt magnitudes are the cells' OWN measured values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("artifact", "signal", "expected_gap", "sign", "tilt"),
    [
        (
            HOLDING_RUN_HEAVY_ARTIFACT,
            HOLDING_RUN_HEAVY_SIGNAL,
            HOLDING_RUN_HEAVY_RAW_GAP_POINTS,
            HOLDING_RUN_HEAVY_SIGN,
            HOLDING_RUN_HEAVY_TILT,
        ),
        (
            HIGH_FLAG_UNDERDOG_ARTIFACT,
            HIGH_FLAG_UNDERDOG_SIGNAL,
            HIGH_FLAG_UNDERDOG_RAW_GAP_POINTS,
            HIGH_FLAG_UNDERDOG_SIGN,
            HIGH_FLAG_UNDERDOG_TILT,
        ),
    ],
)
def test_tilt_magnitudes_are_the_registry_cells_own_measured_gaps(
    artifact: str, signal: str, expected_gap: float, sign: int, tilt: float
) -> None:
    """An underived constant here would be a defect; both come from the cell.

    Checked twice over: against the experiment's own artifact
    (``result.raw_gap_pct``) and against the registry entry's own
    ``classification_evidence`` text, which carries both the gap and the sign.
    """

    metadata = json.loads((_REPO_ROOT / artifact).read_text(encoding="utf-8"))
    assert metadata["name"] == signal
    assert metadata["result"]["raw_gap_pct"] == pytest.approx(expected_gap, abs=1e-12)

    registry = json.loads(
        (_REPO_ROOT / "registry" / "weak_signals.json").read_text(encoding="utf-8")
    )
    evidence = str(registry["signals"][signal]["classification_evidence"])
    gap_match = re.search(r"raw_gap=([+-][\d.]+) pts", evidence)
    sign_match = re.search(r"sign=([+-]\d)", evidence)
    assert gap_match is not None and sign_match is not None
    assert float(gap_match.group(1)) == pytest.approx(expected_gap, abs=1e-4)
    assert int(sign_match.group(1)) == sign

    # The signed home-cover gap is sign * raw_gap_pct (experiment_runner's own
    # raw_gap_pct = sign * (subset_cover - complement_cover) * 100).
    assert tilt == pytest.approx(sign * expected_gap / 100.0, abs=1e-15)


def test_heavy_underdog_threshold_is_the_screens_own_default() -> None:
    assert HEAVY_UNDERDOG_THRESHOLD == 7.0


# ---------------------------------------------------------------------------
# 2. The forward adapter reproduces the screen's own builders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trait", ["holding", "flag_rate"])
def test_forward_lookup_reproduces_the_screen_builder_quartiles(trait: str) -> None:
    """MEASURED equality on the real snapshots, not assumed.

    The builders discard both ``prev_total`` and the qcut bin edges, so the
    forward hop for a not-yet-played game needs its own copy of the lag/qcut.
    This pins that copy against the builders on every historical (referee,
    season) pair -- a pinned second path, never a second definition.
    """

    officials_root = _REPO_ROOT / "data" / "raw" / "officials"
    if not list(officials_root.glob("*/officials.parquet")):
        pytest.skip("no local officials snapshot (generated data, absent in a fresh clone)")
    if not list(officials_root.glob("*/game_penalty_types.parquet")):
        pytest.skip("no local penalty-type snapshot")

    lookup = build_crew_trait_lookup(_REPO_ROOT)
    if trait == "holding":
        builder = _build_referee_type_trait_data(_REPO_ROOT, _HOLDING_PENALTY_TYPE).game_trait
        column = "lag_type_quartile"
        resolve = lookup.holding_quartile
    else:
        builder = _build_referee_trait_data(_REPO_ROOT).game_trait
        column = "lag_penalty_rate_quartile"
        resolve = lookup.flag_rate_quartile

    pairs = (
        builder.dropna(subset=[column])
        .loc[:, ["official_name", "season", column]]
        .drop_duplicates()
    )
    assert len(pairs) > 50, "fixture-free pin needs a real, non-trivial population"
    mismatches = [
        (str(row.official_name), int(row.season))
        for row in pairs.itertuples(index=False)
        if resolve(str(row.official_name), int(row.season)) != int(getattr(row, column))
    ]
    assert mismatches == []


def test_forward_lookup_buckets_a_season_the_builders_never_saw() -> None:
    """The hop the builders structurally cannot make: next season, no games."""

    lookup = _lookup()
    # 2026 is absent from the fixture's lagged table; it must fall back to the
    # referee's 2025 mean bucketed against the FROZEN cutpoints.
    assert lookup.holding_quartile("r8", 2026) == 4
    assert lookup.holding_quartile("r1", 2026) == 1
    assert lookup.holding_quartile("nobody", 2026) is None


# ---------------------------------------------------------------------------
# 3. Flag correctness and the additive tilt
# ---------------------------------------------------------------------------


def test_holding_cell_fires_only_for_a_run_heavy_home_and_a_top_holding_crew() -> None:
    lookup = _lookup()
    fired = crew_tilt_flags(
        referee="r8",
        season=2026,
        home_pass_rate_quartile=1,
        decision_home_spread=-2.5,
        lookup=lookup,
    )
    assert fired.holding_tilt_flag
    assert not fired.high_flag_underdog_flag
    assert fired.tilt_points == pytest.approx(HOLDING_RUN_HEAVY_TILT)

    # Top-quartile crew but the home team is not run-heavy.
    assert not crew_tilt_flags(
        referee="r8",
        season=2026,
        home_pass_rate_quartile=4,
        decision_home_spread=-2.5,
        lookup=lookup,
    ).holding_tilt_flag
    # Run-heavy home team but a bottom-quartile holding crew.
    assert not crew_tilt_flags(
        referee="r1",
        season=2026,
        home_pass_rate_quartile=1,
        decision_home_spread=-2.5,
        lookup=lookup,
    ).holding_tilt_flag


def test_underdog_cell_uses_the_frozen_tuesday_line_and_the_screens_threshold() -> None:
    lookup = _lookup()
    # Home getting exactly 7 points at the frozen Tuesday line fires.
    fired = crew_tilt_flags(
        referee="r8",
        season=2026,
        home_pass_rate_quartile=3,
        decision_home_spread=-7.0,
        lookup=lookup,
    )
    assert fired.high_flag_underdog_flag
    assert fired.tilt_points == pytest.approx(HIGH_FLAG_UNDERDOG_TILT)

    # 6.5 points is not a heavy underdog; a home FAVORITE is the opposite side
    # of the convention and must never fire.
    for spread in (-6.5, 7.0):
        assert not crew_tilt_flags(
            referee="r8",
            season=2026,
            home_pass_rate_quartile=3,
            decision_home_spread=spread,
            lookup=lookup,
        ).high_flag_underdog_flag
    # No line at all: no flag, never a crash.
    assert not crew_tilt_flags(
        referee="r8",
        season=2026,
        home_pass_rate_quartile=3,
        decision_home_spread=None,
        lookup=lookup,
    ).high_flag_underdog_flag


def test_both_cells_compose_additively_and_clip() -> None:
    flags = crew_tilt_flags(
        referee="r8",
        season=2026,
        home_pass_rate_quartile=1,
        decision_home_spread=-9.0,
        lookup=_lookup(),
    )
    assert flags.holding_tilt_flag and flags.high_flag_underdog_flag
    assert flags.tilt_points == pytest.approx(HOLDING_RUN_HEAVY_TILT + HIGH_FLAG_UNDERDOG_TILT)
    assert tilted_probability(0.98, flags.tilt_points) == 1.0
    assert tilted_probability(0.01, HOLDING_RUN_HEAVY_TILT) == 0.0


def test_no_effect_outside_the_flagged_populations() -> None:
    flags = crew_tilt_flags(
        referee="r4",
        season=2026,
        home_pass_rate_quartile=2,
        decision_home_spread=-1.0,
        lookup=_lookup(),
    )
    assert flags.tilt_points == 0.0
    assert tilted_probability(0.5123, 0.0) == pytest.approx(0.5123)


# ---------------------------------------------------------------------------
# 4. Deadline window and anti-backdating
# ---------------------------------------------------------------------------


def test_snapshot_before_the_deadline_applies_and_can_flip(
    tmp_path: Path, patched_traits: dict[str, int]
) -> None:
    data_root = tmp_path / "data"
    _write_crew_snapshot(
        data_root,
        snapshot_id="20260916T190000Z",
        captured_at_utc=WEDNESDAY_CAPTURE,
        assignments={"g_flag": "r8", "g_clean": "r4"},
    )
    patched_traits["g_flag"] = 1  # run-heavy home
    patched_traits["g_clean"] = 3

    plan = _plan(
        (
            _game(game_id="g_flag", new_pick_side="HOME", probability=0.52),
            _game(game_id="g_clean", new_pick_side="HOME", probability=0.52),
        )
    )
    rows, diagnostics = build_crew_tilt_refresh_rows(plan, data_root=data_root, repo_root=tmp_path)
    assert not diagnostics["skipped"]
    assert list(rows.columns) == list(CREW_TILT_REFRESH_COLUMNS)
    by_game = rows.set_index("game_id")

    flagged = by_game.loc["g_flag"]
    assert bool(flagged["holding_tilt_flag"])
    # 0.52 - 0.0599... = 0.4601 -> crosses 0.5 -> the pick flips.
    assert flagged["tilted_home_cover_probability"] == pytest.approx(0.52 + HOLDING_RUN_HEAVY_TILT)
    assert bool(flagged["crew_tilt_flip"])
    assert flagged["crew_would_be_pick_side"] == "AWAY"

    clean = by_game.loc["g_clean"]
    assert not bool(clean["holding_tilt_flag"])
    assert clean["tilt_points"] == 0.0
    assert not bool(clean["crew_tilt_flip"])
    assert clean["crew_would_be_pick_side"] == clean["played_pick_side"] == "HOME"
    assert diagnostics["would_flip_game_ids"] == ["g_flag"]


def test_a_wednesday_capture_is_in_window_for_a_monday_night_game(
    tmp_path: Path, patched_traits: dict[str, int]
) -> None:
    """SNF/MNF lock EARLY at Sunday 16:00 ET -- still after Wednesday."""

    data_root = tmp_path / "data"
    _write_crew_snapshot(
        data_root,
        snapshot_id="20260916T190000Z",
        captured_at_utc=WEDNESDAY_CAPTURE,
        assignments={"g_mnf": "r8"},
    )
    patched_traits["g_mnf"] = 1
    plan = _plan((_game(game_id="g_mnf", kickoff=MNF_KICKOFF, probability=0.52),))
    rows, diagnostics = build_crew_tilt_refresh_rows(plan, data_root=data_root, repo_root=tmp_path)
    assert len(rows) == 1
    assert diagnostics["snapshot_after_deadline_skipped_game_ids"] == []
    game = plan.games[0]
    assert game.deadline < game.kickoff  # the Sunday lock, not its own kickoff
    assert bool(rows.iloc[0]["holding_tilt_flag"])


def test_a_snapshot_at_or_after_the_deadline_never_applies(
    tmp_path: Path, patched_traits: dict[str, int]
) -> None:
    """Anti-backdating: a post-kickoff capture can never flip anything."""

    data_root = tmp_path / "data"
    _write_crew_snapshot(
        data_root,
        snapshot_id="20260921T000000Z",
        captured_at_utc="2026-09-21T00:00:00Z",  # after Sunday's kickoff
        assignments={"g_flag": "r8"},
    )
    patched_traits["g_flag"] = 1
    plan = _plan((_game(game_id="g_flag", probability=0.52),))
    rows, diagnostics = build_crew_tilt_refresh_rows(plan, data_root=data_root, repo_root=tmp_path)
    assert rows.empty
    assert diagnostics["skipped"] is True
    assert diagnostics["reason"] == "crew_snapshot_at_or_after_pick_deadline"


def test_no_snapshot_for_the_week_is_a_documented_no_op(
    tmp_path: Path, patched_traits: dict[str, int]
) -> None:
    data_root = tmp_path / "data"
    _write_crew_snapshot(
        data_root,
        snapshot_id="20260916T190000Z",
        captured_at_utc=WEDNESDAY_CAPTURE,
        assignments={"g_flag": "r8"},
        week=WEEK + 1,
    )
    plan = _plan((_game(game_id="g_flag", probability=0.52),))
    rows, diagnostics = build_crew_tilt_refresh_rows(plan, data_root=data_root, repo_root=tmp_path)
    assert rows.empty
    assert diagnostics["reason"] == "no_crew_snapshot_for_week"


def test_a_game_absent_from_the_snapshot_keeps_the_tuesday_pick(
    tmp_path: Path, patched_traits: dict[str, int]
) -> None:
    data_root = tmp_path / "data"
    _write_crew_snapshot(
        data_root,
        snapshot_id="20260916T190000Z",
        captured_at_utc=WEDNESDAY_CAPTURE,
        assignments={"g_other": "r8"},
    )
    patched_traits["g_missing"] = 1
    plan = _plan((_game(game_id="g_missing", new_pick_side="AWAY", probability=0.48),))
    rows, _diagnostics = build_crew_tilt_refresh_rows(plan, data_root=data_root, repo_root=tmp_path)
    row = rows.iloc[0]
    assert row["overlay_status"] == "game_absent_from_crew_snapshot"
    assert row["tilt_points"] == 0.0
    assert row["crew_would_be_pick_side"] == "AWAY"
    assert not bool(row["crew_tilt_flip"])


def test_latest_crew_snapshot_prefers_the_newest_capture(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_crew_snapshot(
        data_root,
        snapshot_id="20260916T190000Z",
        captured_at_utc=WEDNESDAY_CAPTURE,
        assignments={"g": "r1"},
    )
    _write_crew_snapshot(
        data_root,
        snapshot_id="20260918T190000Z",
        captured_at_utc="2026-09-18T19:00:00Z",
        assignments={"g": "r8"},
    )
    snapshot = latest_crew_snapshot(data_root, season=SEASON, week=WEEK)
    assert snapshot is not None
    assert snapshot.snapshot_id == "20260918T190000Z"
    assert snapshot.referee_by_game_id == {"g": "r8"}


# ---------------------------------------------------------------------------
# 5. Played-pick invariance and the opt-in recorder
# ---------------------------------------------------------------------------


def test_the_movement_policy_pick_is_never_disturbed_by_a_zero_tilt(
    tmp_path: Path, patched_traits: dict[str, int]
) -> None:
    """A game the movement policy moved OFF the model's side stays moved.

    The tilt flips the PLAYED side only when it crosses 0.5; with no flag it
    cannot, so the challenger's pick is the incumbent's even where the played
    pick and the model-only pick disagree.
    """

    data_root = tmp_path / "data"
    _write_crew_snapshot(
        data_root,
        snapshot_id="20260916T190000Z",
        captured_at_utc=WEDNESDAY_CAPTURE,
        assignments={"g_move": "r4"},
    )
    patched_traits["g_move"] = 3
    game = _game(game_id="g_move", new_pick_side="AWAY", probability=0.52, movement=True)
    assert game.model_only_pick_side == "HOME" != game.new_pick_side
    rows, _diagnostics = build_crew_tilt_refresh_rows(
        _plan((game,)), data_root=data_root, repo_root=tmp_path
    )
    row = rows.iloc[0]
    assert row["crew_would_be_pick_side"] == "AWAY"
    assert not bool(row["crew_tilt_flip"])


def test_recording_is_opt_in(tmp_path: Path) -> None:
    result = record_crew_tilt_refresh_overlay(
        tmp_path / "artifacts",
        tmp_path / "data",
        _plan((_game(game_id="g"),)),
        repo_root=tmp_path,
        record_decisions=False,
    )
    assert result == {
        "challenger_id": CHALLENGER_ID,
        "recorded": 0,
        "skipped": True,
        "reason": (
            "pass --record-decisions to append this pass's would-be picks to the "
            "crew-tilt refresh ledger"
        ),
    }
    assert not (tmp_path / "artifacts").exists()


def test_recording_outside_the_lock_window_writes_no_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refresh hook cannot backdate crew rows after the recording window."""

    artifacts_root = tmp_path / "artifacts"
    plan = replace(
        _plan((_game(game_id="g"),)), computed_at_utc=pd.Timestamp("2026-09-01T15:00:00Z")
    )
    monkeypatch.setattr(
        crew_tilt_refresh_overlay,
        "original_card",
        lambda *_args, **_kwargs: pd.DataFrame({"kickoff": [KICKOFF]}),
    )

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_crew_tilt_refresh_overlay(
            artifacts_root,
            tmp_path / "data",
            plan,
            repo_root=tmp_path,
            record_decisions=True,
        )

    assert not (artifacts_root / "prospective" / "crew_tilt_refresh_decisions.parquet").exists()


# ---------------------------------------------------------------------------
# 6. Registration: fingerprint stability
# ---------------------------------------------------------------------------


def _registered_entry() -> dict[str, Any]:
    payload = json.loads(
        (_REPO_ROOT / "artifacts" / "prospective" / "challengers.json").read_text(encoding="utf-8")
    )
    entries = [entry for entry in payload["challengers"] if entry["challenger_id"] == CHALLENGER_ID]
    assert len(entries) == 1, f"{CHALLENGER_ID} must be registered exactly once"
    return dict(entries[0])


def test_registered_challenger_fingerprint_is_stable() -> None:
    entry = _registered_entry()
    assert entry["status"] == "ACTIVE_PROSPECTIVE"
    fingerprint = config_fingerprint(entry["model"])
    assert entry["config_fingerprint"] == fingerprint
    # Stable across repeated computation and key ordering.
    reordered = dict(reversed(list(entry["model"].items())))
    assert config_fingerprint(reordered) == fingerprint


def test_registration_names_the_refresh_recording_path() -> None:
    """The crew arm is late-refresh only, never a Tuesday publish recorder."""

    entry = _registered_entry()
    assert "publish-predictions --record-decisions" not in entry["weekly_recording_command"]
    assert "refresh-picks --record-decisions" in entry["weekly_recording_command"]


def test_registration_declares_the_predeclaration_and_both_parent_cells() -> None:
    entry = _registered_entry()
    sources = entry["evidence"]["registry_source"]
    assert f"registry/weak_signals.json:{HOLDING_RUN_HEAVY_SIGNAL}" in sources
    assert f"registry/weak_signals.json:{HIGH_FLAG_UNDERDOG_SIGNAL}" in sources
    assert "docs/referee_assignments_capture.md" in entry["evidence"]["write_up"]
    assert "unresolved_below_power" in entry["evidence"]["classification"]
