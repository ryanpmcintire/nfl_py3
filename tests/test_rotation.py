from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nfl_ats import cli, rotation
from nfl_ats.constants import (
    DEFAULT_MIN_CALIBRATION_GAMES,
    DEFAULT_MIN_TRAIN_GAMES,
    EARLY_SEASON_GAME_COUNT,
    MIN_FITTABLE_TRAIN_GAMES,
)
from nfl_ats.rotation import (
    GRADE_POOLS,
    STRATIFIED_GRADE,
    STRATIFIED_LEG_COUNT,
    Registry,
    RegistryError,
    assign_stratified_window,
    assign_window,
    confirmation_split,
    confirmation_split_legs,
    declare_family,
    earliest_eligible_start_season,
    eligible_blocks,
    eligible_stratified_seasons,
    load_registry,
    record_look,
    registry_from_payload,
    registry_payload,
    registry_status,
    save_registry,
)
from nfl_ats.weak_signals import EFFECT_UNITS

# The LIVE ledger is append-only research data: every recorded look changes its
# contents and its capacity counts, so asserting behaviour against it would make
# these tests fail the moment the registry does its job. Behaviour is pinned to a
# frozen copy of the seeded state; the live file gets its own contract test below.
SEEDED_REGISTRY = Path(__file__).resolve().parent / "data" / "seeded_rotation_registry.json"
LIVE_REGISTRY = Path(__file__).resolve().parents[1] / "registry" / "rotation_registry.json"


def _seeded() -> Registry:
    return load_registry(SEEDED_REGISTRY)


def test_live_ledger_loads_and_validates() -> None:
    """The shipped ledger must always satisfy the schema, whatever it now holds.

    Deliberately asserts no counts: spending a window is the registry working.
    """

    registry = load_registry(LIVE_REGISTRY)
    assert registry.families
    for name, family in registry.families.items():
        assert family.grade in GRADE_POOLS, name
        assigned = [window for window in family.windows if window.state == "assigned"]
        assert len(assigned) <= 1, name
        for window in family.windows:
            if window.state == "spent":
                assert window.artifact, name
                assert window.verdict, name
            if window.verdict == "closed_negative":
                # AGENTS.md binding taxonomy: the tracked ledger may never
                # hold a closure that names no admissible ground -- "the
                # interval contains zero" closed lines for two days before
                # this was enforced. Legacy tolerance exists only for frozen
                # test fixtures, never for the live registry.
                assert window.closing_ground, (
                    f"{name}: closed_negative without an admissible closing_ground; "
                    "an interval containing zero is NOT one (AGENTS.md, binding)"
                )
                assert window.probability_positive is not None, name
            if window.effect_units is not None:
                # Redundant with load-time enforcement in _validate_effect_fields,
                # pinned here the same way the closed_negative checks above are:
                # a visible contract on the TRACKED registry, not just the loader.
                assert window.effect_units in EFFECT_UNITS, (
                    f"{name}: effect_units {window.effect_units!r} is not a known unit"
                )
                assert window.effect is not None, f"{name}: effect_units without effect"


def _payload(**families: dict[str, Any]) -> dict[str, Any]:
    return {"version": 1, "notes": ["test ledger"], "families": families}


def _family(**overrides: Any) -> dict[str, Any]:
    family = {
        "declared_at": "2026-08-17",
        "description": "test family",
        "grade": "nflverse_spread",
        "status": "open",
        "inherits": [],
        "acknowledges_mined_2018_2025": False,
        "windows": [],
    }
    family.update(overrides)
    return family


def _window(**overrides: Any) -> dict[str, Any]:
    window = {
        "seasons": [2009, 2011],
        "state": "assigned",
        "assigned_at": "2026-08-17",
        "spent_at": None,
        "artifact": None,
        "verdict": None,
        "probability_positive": None,
        "effect": None,
        "effect_units": None,
        "interval": None,
        "standard_error": None,
        "sample_blocks": None,
        "notes": "",
    }
    window.update(overrides)
    return window


def _synthetic_seasons(
    *, seasons: Iterable[int], weeks: int = 16, games_per_week: int = 16
) -> pd.DataFrame:
    """A regular schedule of whole 256-game seasons, for warm-up arithmetic.

    Deliberately synthetic: the eligibility floor must be provable without a
    local feature table, which a fresh clone does not have.
    """

    rows: list[dict[str, Any]] = []
    for season in seasons:
        for week in range(1, weeks + 1):
            for slot in range(games_per_week):
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_{slot:02d}",
                        "season": season,
                        "week": week,
                        "game_type": "REG",
                        # Week ordering only needs to be monotone within a season.
                        "gameday": f"{season}-09-01T00:00:00",
                        "result": 3.0,
                    }
                )
    frame = pd.DataFrame(rows)
    # Space the weeks out so chronological sorting is unambiguous.
    frame["gameday"] = pd.to_datetime(frame["season"].astype(str) + "-09-01") + pd.to_timedelta(
        (frame["week"] - 1) * 7, unit="D"
    )
    return frame


def _features() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season in range(2009, 2015):
        for index, month_day in enumerate(("09-10", "10-10", "12-20")):
            rows.append(
                {
                    "game_id": f"{season}_{index}_REG",
                    "season": season,
                    "week": index + 1,
                    "game_type": "REG",
                    "gameday": f"{season}-{month_day}",
                    "result": 3.0,
                }
            )
        rows.append(
            {
                "game_id": f"{season}_wc",
                "season": season,
                "week": 19,
                "game_type": "WC",
                "gameday": f"{season + 1}-01-10",
                "result": 7.0,
            }
        )
    # One scheduled-but-unplayed regular-season game: never training data.
    rows.append(
        {
            "game_id": "2011_unplayed",
            "season": 2011,
            "week": 4,
            "game_type": "REG",
            "gameday": "2011-11-01",
            "result": None,
        }
    )
    return pd.DataFrame(rows)


def test_seeded_ledger_round_trips(tmp_path: Path) -> None:
    registry = _seeded()
    assert sorted(registry.families) == [
        "cfb_role_continuity",
        "pbp_drive_bundle",
        "player_qb_continuity",
    ]
    assert registry.families["pbp_drive_bundle"].windows[0].seasons == (2013, 2017)

    destination = tmp_path / "rotation_registry.json"
    save_registry(registry, destination)
    reloaded = load_registry(destination)
    assert reloaded == registry
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert written["season_usage"] == {"2013": 1, "2014": 2, "2015": 2, "2016": 2, "2017": 2}


def test_unknown_fields_raise() -> None:
    with pytest.raises(RegistryError, match="unknown top-level fields"):
        registry_from_payload({**_payload(), "budget": 3})
    with pytest.raises(RegistryError, match="unknown fields"):
        registry_from_payload(_payload(alpha=_family(owner="ryan")))
    with pytest.raises(RegistryError, match="unknown window fields"):
        registry_from_payload(_payload(alpha=_family(windows=[_window(seed=1)])))


def test_second_assigned_window_in_ledger_raises() -> None:
    payload = _payload(
        alpha=_family(windows=[_window(seasons=[2009, 2011]), _window(seasons=[2012, 2014])])
    )
    with pytest.raises(RegistryError, match="more than one assigned window"):
        registry_from_payload(payload)


def test_spent_window_without_artifact_raises() -> None:
    payload = _payload(alpha=_family(windows=[_window(state="spent", verdict="closed_negative")]))
    with pytest.raises(RegistryError, match="without an artifact and verdict"):
        registry_from_payload(payload)


def test_window_outside_grade_pool_raises() -> None:
    payload = _payload(
        alpha=_family(
            grade="opener",
            acknowledges_mined_2018_2025=True,
            windows=[_window(seasons=[2018, 2019])],
        )
    )
    with pytest.raises(RegistryError, match="outside the opener pool"):
        registry_from_payload(payload)


def test_mined_window_without_acknowledgment_raises() -> None:
    payload = _payload(alpha=_family(windows=[_window(seasons=[2019, 2021])]))
    with pytest.raises(RegistryError, match="acknowledges_mined_2018_2025"):
        registry_from_payload(payload)


def test_overlapping_windows_in_inherits_chain_raise() -> None:
    payload = _payload(
        parent=_family(
            windows=[
                _window(
                    seasons=[2009, 2011],
                    state="spent",
                    artifact="docs/x.md",
                    verdict="closed_negative",
                )
            ]
        ),
        child=_family(inherits=["parent"], windows=[_window(seasons=[2010, 2012])]),
    )
    with pytest.raises(RegistryError, match="re-looks at seasons already seen"):
        registry_from_payload(payload)


def test_two_parents_may_legitimately_have_spent_the_same_seasons() -> None:
    # Rule 4 retires windows per-family, so two independent families drawing the
    # same block is explicitly allowed -- and both can then be honest parents of
    # one candidate. Rejecting that made a truthful `inherits` undeclarable: a
    # family downstream of both mod07_weak_signal_stack and
    # best_pick_ranker_opener could name neither, since each spent [2020, 2021].
    payload = _payload(
        first=_family(
            acknowledges_mined_2018_2025=True,
            windows=[
                _window(
                    seasons=[2020, 2021],
                    state="spent",
                    artifact="docs/a.md",
                    verdict="unresolved",
                )
            ],
        ),
        second=_family(
            acknowledges_mined_2018_2025=True,
            windows=[
                _window(
                    seasons=[2020, 2021],
                    state="spent",
                    artifact="docs/b.md",
                    verdict="confirmed",
                )
            ],
        ),
        child=_family(inherits=["first", "second"], acknowledges_mined_2018_2025=True),
    )
    registry = registry_from_payload(payload)
    # And the child must still avoid the union of what its ancestors saw.
    assert all(
        not (block[0] <= 2021 and block[1] >= 2020)
        for block in eligible_blocks(registry, "child", size=2)
    )


def test_assignment_is_earliest_eligible_and_retires_per_family() -> None:
    registry = _seeded()
    for name in ("first_opener", "second_opener"):
        registry = declare_family(
            registry,
            name,
            description="opener candidate",
            grade="opener",
            acknowledges_mined_2018_2025=True,
        )
        registry = assign_window(registry, name)
    # Windows retire per family, so both independent hypotheses draw the same
    # earliest block; only the global season-usage table records the overlap.
    assert registry.families["first_opener"].windows[0].seasons == (2020, 2021)
    assert registry.families["second_opener"].windows[0].seasons == (2020, 2021)


def test_assignment_skips_inherited_spent_seasons() -> None:
    registry = declare_family(
        _seeded(),
        "qb_continuity_variant",
        description="variant of the QB continuity line",
        grade="nflverse_spread",
        inherits=("player_qb_continuity",),
        acknowledges_mined_2018_2025=True,
    )
    registry = assign_window(registry, "qb_continuity_variant")
    # 2009-2010 starts sit below the warm-up floor, and 2012-2017 starts all
    # intersect the inherited 2014-2017 spend -- but [2011, 2013] clears both,
    # so an inheriting family no longer has to reach into the mined 2018-2025
    # era for its first window.
    assert registry.families["qb_continuity_variant"].windows[0].seasons == (2011, 2013)
    registry = record_look(
        registry,
        "qb_continuity_variant",
        artifact="docs/variant.md",
        verdict="unresolved",
        probability_positive=0.42,
    )
    registry = assign_window(registry, "qb_continuity_variant")
    # Its own [2011, 2013] spend now joins the inherited [2014, 2017], so the
    # next clean block is the first one past both.
    assert registry.families["qb_continuity_variant"].windows[1].seasons == (2018, 2020)
    assert registry.families["qb_continuity_variant"].status == "open"


def test_assignment_respects_the_warmup_floor() -> None:
    # Rule 9: the first two feature-table seasons (2009-2010) are warm-up
    # history — enough games to fit a margin model at all, then 200 calibration
    # prediction rows — so the earliest assignable block starts 2011 even
    # though the nflverse_spread pool opens in 2009. Both figures are derived:
    # calibration from calibrate_cover_prediction_stream, and the training term
    # from fit_margin_model's own preconditions. The floor read 2013 while the
    # calibration constant was an inherited 400, and 2012 while rule 9 borrowed
    # the underived 500-game reporting default.
    registry = declare_family(
        _seeded(), "fresh", description="untainted candidate", grade="nflverse_spread"
    )
    registry = assign_window(registry, "fresh")
    assert registry.families["fresh"].windows[0].seasons == (2011, 2013)


def test_warmup_floor_is_computed_from_the_thresholds_it_depends_on() -> None:
    # The floor is arithmetic on the thresholds, not a season somebody typed.
    # Rule 9 asks whether a window's first week can be SCORED, so the training
    # term is the feasibility minimum, NOT the conservative reporting default.
    # Pinning that distinction is the point of this test: an irreversible
    # decision must not silently inherit an underived number.
    assert rotation.WARMUP_TRAINING_GAMES == MIN_FITTABLE_TRAIN_GAMES
    assert rotation.WARMUP_TRAINING_GAMES != DEFAULT_MIN_TRAIN_GAMES
    assert rotation.WARMUP_CALIBRATION_ROWS == DEFAULT_MIN_CALIBRATION_GAMES
    assert (
        rotation.MIN_ELIGIBLE_START_SEASON
        == rotation.FEATURE_TABLE_START_SEASON + rotation.WARMUP_PRIOR_SEASONS
    )
    # Today's values, recorded so a change to either threshold shows up here as
    # a visible move in what seasons a fresh family may draw.
    assert rotation.MIN_ELIGIBLE_START_SEASON == 2011


def test_warmup_closed_form_is_never_optimistic_about_a_real_schedule() -> None:
    # The training and calibration requirements are CHAINED: prediction rows
    # only accrue once the training floor is met, and only whole weeks are
    # scorable. Naively summing them under-counts, and at the feasibility floor
    # that error is large enough to matter -- it would claim 2010, a season
    # whose week 1 has too few prior prediction rows to calibrate. The closed
    # form must therefore never sit BELOW the exact walk.
    schedule = _synthetic_seasons(seasons=range(2009, 2016))
    exact = earliest_eligible_start_season(schedule)
    assert exact == rotation.MIN_ELIGIBLE_START_SEASON
    naive = rotation.FEATURE_TABLE_START_SEASON + math.ceil(
        (rotation.WARMUP_TRAINING_GAMES + rotation.WARMUP_CALIBRATION_ROWS)
        / EARLY_SEASON_GAME_COUNT
    )
    assert naive < exact, "the naive sum should still be demonstrably optimistic here"


def test_warmup_floor_tracks_a_raised_training_requirement() -> None:
    # The exact walk is a function of its inputs, not a hardcoded season.
    schedule = _synthetic_seasons(seasons=range(2009, 2016))
    assert earliest_eligible_start_season(schedule, min_train_games=500) == 2012
    assert earliest_eligible_start_season(schedule, min_train_games=1000) == 2014


def test_capacity_partition_starts_at_the_warmup_floor() -> None:
    pools = registry_status(_seeded())["grade_pools"]
    for grade in ("close", "nflverse_spread"):
        assert pools[grade]["total_windows"] == 5
        assert pools[grade]["unspent_blocks"] == [[2020, 2022], [2023, 2025]]
    # The opener pool starts well past the floor and is untouched by rule 9.
    assert pools["opener"]["total_windows"] == 3


def test_confirmation_split_refuses_a_window_with_no_history() -> None:
    # The floor guards assignment only; a hand-written ledger can still hold a
    # pre-2013 window, and the split is the fail-closed backstop.
    registry = registry_from_payload(
        _payload(alpha=_family(windows=[_window(seasons=[2009, 2011])]))
    )
    with pytest.raises(RegistryError, match="no completed games before it"):
        confirmation_split(_features(), registry, "alpha")


def test_assign_refuses_a_second_unspent_window() -> None:
    registry = declare_family(_seeded(), "alpha", description="candidate", grade="nflverse_spread")
    registry = assign_window(registry, "alpha")
    with pytest.raises(RegistryError, match="already holds an unspent window"):
        assign_window(registry, "alpha")


def test_confirmation_split_is_forward_chained_and_regular_season_only() -> None:
    registry = registry_from_payload(
        _payload(alpha=_family(windows=[_window(seasons=[2012, 2013])]))
    )
    training, window = confirmation_split(_features(), registry, "alpha")

    assert sorted(window["season"].unique()) == [2012, 2013]
    assert set(window["game_type"]) == {"REG"}
    assert set(training["game_type"]) == {"REG"}
    assert training["gameday"].max() < window["gameday"].min()
    assert sorted(training["season"].unique()) == [2009, 2010, 2011]
    assert training["result"].notna().all()
    assert "2011_unplayed" not in set(training["game_id"])


def test_confirmation_split_raise_conditions() -> None:
    registry = _seeded()
    with pytest.raises(RegistryError, match="no assigned window"):
        confirmation_split(_features(), registry, "cfb_role_continuity")

    live = registry_from_payload(_payload(alpha=_family(windows=[_window(seasons=[2012, 2013])])))
    with pytest.raises(RegistryError, match="missing columns"):
        confirmation_split(_features().drop(columns=["result"]), live, "alpha")
    with pytest.raises(RegistryError, match="Unknown family"):
        confirmation_split(_features(), live, "beta")

    missing = registry_from_payload(
        _payload(
            alpha=_family(
                acknowledges_mined_2018_2025=True, windows=[_window(seasons=[2020, 2022])]
            )
        )
    )
    with pytest.raises(RegistryError, match="missing window seasons"):
        confirmation_split(_features(), missing, "alpha")


def test_record_look_spends_the_window_and_blocks_a_re_split() -> None:
    registry = registry_from_payload(
        _payload(alpha=_family(windows=[_window(seasons=[2012, 2013])]))
    )
    recorded = record_look(
        registry,
        "alpha",
        artifact="docs/alpha.md",
        verdict="closed_negative",
        probability_positive=0.08,
        closing_ground="wrong_sign_resolved",
        notes="-0.3 pts on 512 games",
    )
    window = recorded.families["alpha"].windows[0]
    assert window.state == "spent"
    assert window.artifact == "docs/alpha.md"
    assert window.probability_positive == pytest.approx(0.08)
    assert recorded.families["alpha"].status == "closed_negative"
    assert registry_status(recorded)["season_usage"] == {"2012": 1, "2013": 1}

    with pytest.raises(RegistryError, match="already spent"):
        confirmation_split(_features(), recorded, "alpha")
    with pytest.raises(RegistryError, match="no assigned window to record"):
        record_look(
            recorded,
            "alpha",
            artifact="docs/alpha.md",
            verdict="confirmed",
            probability_positive=0.95,
        )


def test_record_look_replace_corrects_only_latest_spent_window_and_preserves_provenance() -> None:
    registry = registry_from_payload(
        _payload(alpha=_family(windows=[_window(seasons=[2012, 2013])]))
    )
    recorded = record_look(
        registry,
        "alpha",
        artifact="docs/alpha.md",
        verdict="closed_negative",
        probability_positive=0.08,
        closing_ground="wrong_sign_resolved",
        notes="incorrect closure",
    )
    original = recorded.families["alpha"].windows[-1]
    corrected = record_look(
        recorded,
        "alpha",
        artifact="docs/alpha.md",
        verdict="unresolved",
        probability_positive=0.42,
        notes="correction: closure ground was inadmissible",
        replace_existing=True,
    )
    replacement = corrected.families["alpha"].windows[-1]
    assert replacement.seasons == original.seasons
    assert replacement.window_kind == original.window_kind
    assert replacement.assigned_at == original.assigned_at
    assert replacement.spent_at == original.spent_at
    assert replacement.artifact == original.artifact
    assert replacement.verdict == "unresolved"
    assert replacement.closing_ground is None
    assert corrected.families["alpha"].status == "open"


def test_record_look_replace_rejects_mismatched_or_nonlatest_windows() -> None:
    registry = registry_from_payload(
        _payload(alpha=_family(windows=[_window(seasons=[2012, 2013])]))
    )
    recorded = record_look(
        registry,
        "alpha",
        artifact="docs/alpha.md",
        verdict="unresolved",
        probability_positive=0.42,
    )
    with pytest.raises(RegistryError, match="exact latest-window artifact"):
        record_look(
            recorded,
            "alpha",
            artifact="docs/other.md",
            verdict="unresolved",
            probability_positive=0.42,
            replace_existing=True,
        )

    active = assign_window(recorded, "alpha")
    with pytest.raises(RegistryError, match="assigned window"):
        record_look(
            active,
            "alpha",
            artifact="docs/alpha.md",
            verdict="unresolved",
            probability_positive=0.42,
            replace_existing=True,
        )


def test_status_reports_remaining_opener_capacity() -> None:
    status = registry_status(_seeded())
    assert status["grade_pools"]["opener"]["unspent_windows"] == 3
    assert "opener pool: 3 windows unspent" in status["summary"]
    assert [family["name"] for family in status["families"]] == [
        "cfb_role_continuity",
        "pbp_drive_bundle",
        "player_qb_continuity",
    ]


def test_cli_rotation_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "rotation_registry.json").write_text(
        SEEDED_REGISTRY.read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_dir))

    assert cli.main(["rotation", "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert "opener pool: 3 windows unspent" in status["summary"]

    assert (
        cli.main(
            [
                "rotation",
                "declare",
                "--name",
                "stack",
                "--description",
                "weak-signal stack",
                "--grade",
                "opener",
                "--inherits",
                "player_qb_continuity",
                "--acknowledge-mined",
            ]
        )
        == 0
    )
    declared = json.loads(capsys.readouterr().out)
    assert declared["family"]["inherits"] == ["player_qb_continuity"]

    assert cli.main(["rotation", "assign", "--name", "stack"]) == 0
    assigned = json.loads(capsys.readouterr().out)
    assert assigned["family"]["windows"][0]["seasons"] == [2020, 2021]

    assert (
        cli.main(
            [
                "rotation",
                "record",
                "--name",
                "stack",
                "--artifact",
                "docs/mod07_stack.md",
                "--verdict",
                "unresolved",
                "--probability-positive",
                "0.61",
            ]
        )
        == 0
    )
    recorded = json.loads(capsys.readouterr().out)
    assert recorded["family"]["windows"][0]["state"] == "spent"
    assert recorded["grade_pools"]["opener"]["unspent_windows"] == 2

    assert (
        cli.main(
            [
                "rotation",
                "record",
                "--name",
                "stack",
                "--artifact",
                "docs/mod07_stack.md",
                "--verdict",
                "unresolved",
                "--probability-positive",
                "0.42",
                "--notes",
                "correction",
                "--replace",
            ]
        )
        == 0
    )
    corrected = json.loads(capsys.readouterr().out)
    assert corrected["family"]["status"] == "open"
    assert corrected["family"]["windows"][0]["probability_positive"] == pytest.approx(0.42)

    assert cli.main(["rotation", "assign", "--name", "stack"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["family"]["windows"][1]["seasons"] == [2022, 2023]


def test_closed_negative_requires_an_admissible_closing_ground() -> None:
    """AGENTS.md, binding: an interval containing zero never closes a family.

    The registry refuses a closed_negative verdict that names no admissible
    ground, quoting the rule -- so a session that never loaded the prose rule
    hits it anyway, at the exact moment it matters.
    """

    registry = declare_family(_seeded(), "alpha", description="candidate", grade="nflverse_spread")
    registry = assign_window(registry, "alpha")
    with pytest.raises(RegistryError, match="admissible closing_ground"):
        record_look(
            registry,
            "alpha",
            artifact="docs/alpha.md",
            verdict="closed_negative",
            probability_positive=0.04,
        )
    with pytest.raises(RegistryError, match=r"requires probability_positive"):
        record_look(
            registry,
            "alpha",
            artifact="docs/alpha.md",
            verdict="closed_negative",
            probability_positive=None,
            closing_ground="wrong_sign_resolved",
        )
    closed = record_look(
        registry,
        "alpha",
        artifact="docs/alpha.md",
        verdict="closed_negative",
        probability_positive=0.04,
        closing_ground="wrong_sign_resolved",
        notes="whole interval on the wrong side of zero",
    )
    assert closed.families["alpha"].status == "closed_negative"
    assert closed.families["alpha"].windows[0].closing_ground == "wrong_sign_resolved"


def test_non_terminal_verdicts_cannot_carry_a_closing_ground() -> None:
    registry = declare_family(_seeded(), "alpha", description="candidate", grade="nflverse_spread")
    registry = assign_window(registry, "alpha")
    with pytest.raises(RegistryError, match="cannot carry"):
        record_look(
            registry,
            "alpha",
            artifact="docs/alpha.md",
            verdict="unresolved",
            probability_positive=0.42,
            closing_ground="positive_control_bound",
        )


def test_legacy_groundless_closures_load_but_new_grounds_are_checked() -> None:
    """Historical ledger entries are never re-judged on load (the warm-up-floor
    principle), so a prose-era closed_negative without a ground still parses.
    A ground that IS present must be admissible and belong to a closure, and
    the live-ledger contract test separately holds the tracked registry to the
    full taxonomy.
    """

    legacy = _window(
        seasons=[2012, 2014],
        state="spent",
        spent_at="2026-08-18",
        artifact="docs/alpha.md",
        verdict="closed_negative",
        probability_positive=0.04,
    )
    loaded = registry_from_payload(
        _payload(alpha=_family(status="closed_negative", windows=[legacy]))
    )
    assert loaded.families["alpha"].windows[0].closing_ground is None

    with pytest.raises(RegistryError, match="unknown closing_ground"):
        registry_from_payload(
            _payload(
                alpha=_family(
                    status="closed_negative",
                    windows=[_window(**{**legacy, "closing_ground": "vibes"})],
                )
            )
        )
    with pytest.raises(RegistryError, match="cannot carry"):
        registry_from_payload(
            _payload(
                alpha=_family(
                    windows=[
                        _window(
                            seasons=[2012, 2014],
                            state="spent",
                            spent_at="2026-08-18",
                            artifact="docs/alpha.md",
                            verdict="unresolved",
                            probability_positive=0.4,
                            closing_ground="wrong_sign_resolved",
                        )
                    ]
                )
            )
        )


def test_effect_fields_round_trip_through_load_and_save(tmp_path: Path) -> None:
    """The five effect fields (docs/estimation_variance.md's proposed backfill)
    parse from a payload, survive a save/load round trip, and serialize back
    out unchanged -- the same contract the pre-existing fields already hold.
    """

    window = _window(
        seasons=[2012, 2014],
        state="spent",
        spent_at="2026-08-18",
        artifact="docs/alpha.md",
        verdict="unresolved",
        probability_positive=0.6,
        effect=1.23,
        effect_units="accuracy_points",
        interval=[-0.5, 2.9],
        standard_error=0.87,
        sample_blocks=35,
    )
    registry = registry_from_payload(_payload(alpha=_family(windows=[window])))
    loaded = registry.families["alpha"].windows[0]
    assert loaded.effect == pytest.approx(1.23)
    assert loaded.effect_units == "accuracy_points"
    assert loaded.interval == (-0.5, 2.9)
    assert loaded.standard_error == pytest.approx(0.87)
    assert loaded.sample_blocks == 35

    destination = tmp_path / "rotation_registry.json"
    save_registry(registry, destination)
    reloaded = load_registry(destination)
    assert reloaded == registry
    written = json.loads(destination.read_text(encoding="utf-8"))
    written_window = written["families"]["alpha"]["windows"][0]
    assert written_window["effect"] == pytest.approx(1.23)
    assert written_window["effect_units"] == "accuracy_points"
    assert written_window["interval"] == [-0.5, 2.9]
    assert written_window["standard_error"] == pytest.approx(0.87)
    assert written_window["sample_blocks"] == 35

    # A window that never carried the new fields still round-trips as all-None.
    bare = registry_from_payload(_payload(alpha=_family(windows=[_window()])))
    bare_window = registry_payload(bare)["families"]["alpha"]["windows"][0]
    for field in ("effect", "effect_units", "interval", "standard_error", "sample_blocks"):
        assert bare_window[field] is None, field


def test_effect_and_effect_units_must_travel_together() -> None:
    with pytest.raises(RegistryError, match="effect and effect_units must be given together"):
        registry_from_payload(_payload(alpha=_family(windows=[_window(effect=1.0)])))
    with pytest.raises(RegistryError, match="effect and effect_units must be given together"):
        registry_from_payload(
            _payload(alpha=_family(windows=[_window(effect_units="accuracy_points")]))
        )


def test_unknown_effect_units_raises() -> None:
    with pytest.raises(RegistryError, match="unknown effect_units"):
        registry_from_payload(
            _payload(alpha=_family(windows=[_window(effect=1.0, effect_units="furlongs")]))
        )


def test_interval_must_be_a_two_element_list() -> None:
    with pytest.raises(RegistryError, match="two-element"):
        registry_from_payload(_payload(alpha=_family(windows=[_window(interval=[1.0])])))
    with pytest.raises(RegistryError, match="two-element"):
        registry_from_payload(_payload(alpha=_family(windows=[_window(interval=[1.0, 2.0, 3.0])])))


def test_interval_low_must_not_exceed_high() -> None:
    with pytest.raises(RegistryError, match="low > high"):
        registry_from_payload(_payload(alpha=_family(windows=[_window(interval=[2.0, -1.0])])))


def test_standard_error_must_be_positive() -> None:
    with pytest.raises(RegistryError, match="standard_error must be positive"):
        registry_from_payload(_payload(alpha=_family(windows=[_window(standard_error=0.0)])))
    with pytest.raises(RegistryError, match="standard_error must be positive"):
        registry_from_payload(_payload(alpha=_family(windows=[_window(standard_error=-0.4)])))


def test_record_look_accepts_and_validates_effect_fields() -> None:
    registry = declare_family(_seeded(), "alpha", description="candidate", grade="nflverse_spread")
    registry = assign_window(registry, "alpha")
    recorded = record_look(
        registry,
        "alpha",
        artifact="docs/alpha.md",
        verdict="unresolved",
        probability_positive=0.7,
        effect=2.5,
        effect_units="ats_points",
        interval=(-1.0, 6.0),
        standard_error=1.75,
        sample_blocks=42,
        notes="week-blocked [-1.0, 6.0]",
    )
    window = recorded.families["alpha"].windows[0]
    assert window.effect == pytest.approx(2.5)
    assert window.effect_units == "ats_points"
    assert window.interval == (-1.0, 6.0)
    assert window.standard_error == pytest.approx(1.75)
    assert window.sample_blocks == 42

    with pytest.raises(RegistryError, match="unknown effect_units"):
        record_look(
            registry,
            "alpha",
            artifact="docs/alpha.md",
            verdict="unresolved",
            probability_positive=0.7,
            effect=2.5,
            effect_units="furlongs",
        )


def test_cli_rotation_record_accepts_effect_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "rotation_registry.json").write_text(
        SEEDED_REGISTRY.read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_dir))

    assert (
        cli.main(
            [
                "rotation",
                "declare",
                "--name",
                "stack",
                "--description",
                "weak-signal stack",
                "--grade",
                "nflverse_spread",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert cli.main(["rotation", "assign", "--name", "stack"]) == 0
    capsys.readouterr()

    assert (
        cli.main(
            [
                "rotation",
                "record",
                "--name",
                "stack",
                "--artifact",
                "docs/mod07_stack.md",
                "--verdict",
                "unresolved",
                "--probability-positive",
                "0.61",
                "--effect",
                "1.97",
                "--effect-units",
                "accuracy_points",
                "--interval-low",
                "-1.10",
                "--interval-high",
                "5.00",
                "--standard-error",
                "1.75",
                "--sample-blocks",
                "35",
            ]
        )
        == 0
    )
    recorded = json.loads(capsys.readouterr().out)
    window = recorded["family"]["windows"][0]
    assert window["effect"] == pytest.approx(1.97)
    assert window["effect_units"] == "accuracy_points"
    assert window["interval"] == [-1.10, 5.00]
    assert window["standard_error"] == pytest.approx(1.75)
    assert window["sample_blocks"] == 35


# --------------------------------------------------------------------------
# Era-stratified confirmation windows
# (docs/era_stratified_windows_proposal.md, owner-approved 2026-08-19)
# --------------------------------------------------------------------------


def _leg(season: int, effect: float, probability_positive: float = 0.6) -> dict[str, Any]:
    return {
        "season": season,
        "effect": effect,
        "probability_positive": probability_positive,
        "sample_blocks": 12,
    }


def test_stratified_scoped_to_close_grade_only() -> None:
    # The proposal's own scope-limit text names only close-graded families.
    # Opener is explicitly excluded there (six-season archive, "buys little").
    # nflverse_spread shares close's numeric season pool but is never named in
    # that sentence, so it is excluded too -- a documented resolution, not a
    # proposal-stated fact.
    registry = declare_family(_seeded(), "opener_cand", description="x", grade="opener")
    with pytest.raises(RegistryError, match="close-graded only"):
        assign_stratified_window(registry, "opener_cand")
    with pytest.raises(RegistryError, match="close-graded only"):
        eligible_stratified_seasons(registry, "opener_cand")

    registry = declare_family(_seeded(), "nflverse_cand", description="x", grade="nflverse_spread")
    with pytest.raises(RegistryError, match="close-graded only"):
        assign_stratified_window(registry, "nflverse_cand")


def test_stratified_assignment_leg_pair_is_earliest_and_maximally_distant() -> None:
    # Deterministic leg-pair rule, implemented exactly as proposed: earliest
    # untouched season + the untouched season maximally distant from it. That
    # collapses to (min(eligible), max(eligible)) because distance from a
    # fixed minimum is maximized at the largest remaining eligible season --
    # verified here by telescoping the family through three successive draws
    # rather than asserting it algebraically.
    registry = declare_family(
        _seeded(),
        "close_cand",
        description="era-stratified candidate",
        grade="close",
        acknowledges_mined_2018_2025=True,
    )
    registry = assign_stratified_window(registry, "close_cand")
    window = registry.families["close_cand"].windows[0]
    assert window.window_kind == "stratified"
    assert window.seasons == (rotation.MIN_ELIGIBLE_START_SEASON, 2025)
    registry = record_look(
        registry,
        "close_cand",
        artifact="docs/close_cand.md",
        verdict="unresolved",
        probability_positive=0.6,
        leg_effects=[_leg(rotation.MIN_ELIGIBLE_START_SEASON, 1.1), _leg(2025, 0.4)],
    )

    registry = assign_stratified_window(registry, "close_cand")
    second = registry.families["close_cand"].windows[1]
    assert second.seasons == (rotation.MIN_ELIGIBLE_START_SEASON + 1, 2024)
    registry = record_look(
        registry,
        "close_cand",
        artifact="docs/close_cand.md",
        verdict="unresolved",
        probability_positive=0.55,
        leg_effects=[_leg(rotation.MIN_ELIGIBLE_START_SEASON + 1, 0.9), _leg(2024, 0.2)],
    )

    registry = assign_stratified_window(registry, "close_cand")
    third = registry.families["close_cand"].windows[2]
    assert third.seasons == (rotation.MIN_ELIGIBLE_START_SEASON + 2, 2023)


def test_stratified_assignment_is_deterministic_given_the_ledger() -> None:
    base = declare_family(
        _seeded(),
        "close_cand",
        description="x",
        grade="close",
        acknowledges_mined_2018_2025=True,
    )
    first = assign_stratified_window(base, "close_cand")
    second = assign_stratified_window(base, "close_cand")
    assert first.families["close_cand"].windows == second.families["close_cand"].windows


def test_stratified_assignment_refuses_fewer_than_two_eligible_seasons() -> None:
    # Every close-pool season except one is pre-touched by a hand-built
    # inherited window, leaving a single eligible season -- not enough for a
    # leg pair.
    almost_all = list(range(rotation.MIN_ELIGIBLE_START_SEASON, 2025))  # excludes 2025 only
    payload = _payload(
        parent=_family(
            grade="close",
            acknowledges_mined_2018_2025=True,
            windows=[
                _window(
                    seasons=[almost_all[0], almost_all[-1]],
                    state="spent",
                    artifact="docs/x.md",
                    verdict="unresolved",
                    probability_positive=0.5,
                )
            ],
        ),
        child=_family(
            grade="close",
            inherits=["parent"],
            acknowledges_mined_2018_2025=True,
        ),
    )
    registry = registry_from_payload(payload)
    assert eligible_stratified_seasons(registry, "child") == (2025,)
    with pytest.raises(RegistryError, match="Fewer than 2 eligible seasons"):
        assign_stratified_window(registry, "child")


def test_stratified_assign_refuses_a_second_unspent_window() -> None:
    registry = declare_family(
        _seeded(), "close_cand", description="x", grade="close", acknowledges_mined_2018_2025=True
    )
    registry = assign_stratified_window(registry, "close_cand")
    with pytest.raises(RegistryError, match="already holds an unspent window"):
        assign_stratified_window(registry, "close_cand")


def test_touched_seasons_use_real_legs_not_the_span_between_them() -> None:
    # The correctness fix this proposal forces: a stratified window's
    # [min, max] endpoints are its two legs, NOT a contiguous span. A parent
    # that spends legs (2011, 2025) has touched exactly those two seasons --
    # everything in between must still be freely eligible to an inheriting
    # family. Under the old range-overlap logic this would have wrongly
    # blocked out all of 2011-2025.
    registry = declare_family(
        _seeded(),
        "parent",
        description="x",
        grade="close",
        acknowledges_mined_2018_2025=True,
    )
    registry = assign_stratified_window(registry, "parent")
    assert registry.families["parent"].windows[0].seasons == (2011, 2025)
    registry = record_look(
        registry,
        "parent",
        artifact="docs/parent.md",
        verdict="unresolved",
        probability_positive=0.5,
        leg_effects=[_leg(2011, 1.0), _leg(2025, 0.3)],
    )

    registry = declare_family(
        registry,
        "child",
        description="x",
        grade="close",
        inherits=("parent",),
        acknowledges_mined_2018_2025=True,
    )
    # A contiguous 3-season block entirely inside (2011, 2025) but touching
    # neither actual leg must be eligible.
    blocks = eligible_blocks(registry, "child", size=3)
    assert (2012, 2014) in blocks
    assert (2011, 2013) not in blocks  # touches the 2011 leg
    assert (2023, 2025) not in blocks  # touches the 2025 leg

    registry = assign_window(registry, "child", size=3)
    assert registry.families["child"].windows[0].seasons == (2012, 2014)


def test_grade_pool_capacity_uses_real_legs_not_the_span_for_a_stratified_window() -> None:
    registry = declare_family(
        _seeded(),
        "parent",
        description="x",
        grade="close",
        acknowledges_mined_2018_2025=True,
    )
    registry = assign_stratified_window(registry, "parent")
    registry = record_look(
        registry,
        "parent",
        artifact="docs/parent.md",
        verdict="unresolved",
        probability_positive=0.5,
        leg_effects=[_leg(2011, 1.0), _leg(2025, 0.3)],
    )
    capacity = registry_status(registry)["grade_pools"]["close"]
    # The seeded ledger's pbp_drive_bundle/player_qb_continuity windows
    # already consume (2011,13)/(2014,16)/(2017,19) globally (pinned by
    # test_capacity_partition_starts_at_the_warmup_floor); parent's
    # stratified legs (2011, 2025) additionally touch (2023,25) via its 2025
    # leg, leaving only (2020,22) unspent. Under the pre-fix range-overlap
    # logic, treating (2011, 2025) as a contiguous SPAN would have wrongly
    # excluded (2020,22) too, since 2011 <= 2020 and 2022 <= 2025 -- leaving
    # zero blocks unspent instead of the one this window's real legs leave.
    assert capacity["unspent_blocks"] == [[2020, 2022]]


def test_window_kind_disambiguates_a_bare_two_element_seasons_list() -> None:
    # The registry's resolution to the schema ambiguity flagged in the
    # implementation task: a bare [a, b] is structurally identical whether it
    # means "contiguous range a..b" or "leg pair {a, b}". window_kind, not
    # list shape, decides which -- covered_seasons proves the two windows
    # below cover different seasons despite an identical seasons field.
    contiguous = (
        registry_from_payload(
            _payload(alpha=_family(grade="close", windows=[_window(seasons=[2013, 2015])]))
        )
        .families["alpha"]
        .windows[0]
    )
    stratified = (
        registry_from_payload(
            _payload(
                alpha=_family(
                    grade="close", windows=[_window(seasons=[2013, 2015], window_kind="stratified")]
                )
            )
        )
        .families["alpha"]
        .windows[0]
    )
    assert contiguous.covered_seasons == (2013, 2014, 2015)
    assert stratified.covered_seasons == (2013, 2015)


def test_stratified_window_wrong_leg_count_raises() -> None:
    with pytest.raises(RegistryError, match="exactly 2 distinct leg seasons"):
        registry_from_payload(
            _payload(
                alpha=_family(
                    grade="close",
                    windows=[_window(seasons=[2013, 2014, 2015], window_kind="stratified")],
                )
            )
        )
    with pytest.raises(RegistryError, match="exactly 2 distinct leg seasons"):
        registry_from_payload(
            _payload(
                alpha=_family(
                    grade="close",
                    windows=[_window(seasons=[2013, 2013], window_kind="stratified")],
                )
            )
        )


def test_stratified_window_wrong_grade_raises_at_load() -> None:
    # Belt-and-braces: assign_stratified_window already refuses a non-close
    # grade; a hand-edited ledger must not be able to smuggle one past
    # load-time validation either.
    with pytest.raises(RegistryError, match=f"{STRATIFIED_GRADE}-graded only"):
        registry_from_payload(
            _payload(
                alpha=_family(
                    grade="opener",
                    acknowledges_mined_2018_2025=True,
                    windows=[_window(seasons=[2020, 2021], window_kind="stratified")],
                )
            )
        )


def test_spent_stratified_window_requires_leg_effects_at_load() -> None:
    with pytest.raises(RegistryError, match="without per-leg magnitudes"):
        registry_from_payload(
            _payload(
                alpha=_family(
                    grade="close",
                    windows=[
                        _window(
                            seasons=[2013, 2022],
                            window_kind="stratified",
                            state="spent",
                            artifact="docs/x.md",
                            verdict="unresolved",
                            probability_positive=0.5,
                        )
                    ],
                )
            )
        )


def test_record_look_requires_leg_effects_for_a_stratified_window() -> None:
    registry = declare_family(
        _seeded(), "close_cand", description="x", grade="close", acknowledges_mined_2018_2025=True
    )
    registry = assign_stratified_window(registry, "close_cand")
    with pytest.raises(RegistryError, match="requires leg_effects"):
        record_look(
            registry,
            "close_cand",
            artifact="docs/close_cand.md",
            verdict="unresolved",
            probability_positive=0.5,
        )


def test_record_look_rejects_leg_effects_for_a_contiguous_window() -> None:
    registry = declare_family(_seeded(), "alpha", description="x", grade="nflverse_spread")
    registry = assign_window(registry, "alpha")
    with pytest.raises(RegistryError, match="only meaningful on a stratified window"):
        record_look(
            registry,
            "alpha",
            artifact="docs/alpha.md",
            verdict="unresolved",
            probability_positive=0.5,
            leg_effects=[_leg(2011, 1.0)],
        )


def test_leg_effects_must_match_the_windows_own_legs() -> None:
    registry = declare_family(
        _seeded(), "close_cand", description="x", grade="close", acknowledges_mined_2018_2025=True
    )
    registry = assign_stratified_window(registry, "close_cand")
    window = registry.families["close_cand"].windows[0]
    wrong_season = window.seasons[0] + 1
    with pytest.raises(RegistryError, match="exactly one result per leg"):
        record_look(
            registry,
            "close_cand",
            artifact="docs/close_cand.md",
            verdict="unresolved",
            probability_positive=0.5,
            leg_effects=[_leg(wrong_season, 1.0), _leg(window.seasons[1], 0.3)],
        )


def test_leg_effects_round_trip_through_load_and_save(tmp_path: Path) -> None:
    registry = declare_family(
        _seeded(), "close_cand", description="x", grade="close", acknowledges_mined_2018_2025=True
    )
    registry = assign_stratified_window(registry, "close_cand")
    seasons = registry.families["close_cand"].windows[0].seasons
    recorded = record_look(
        registry,
        "close_cand",
        artifact="docs/close_cand.md",
        verdict="unresolved",
        probability_positive=0.62,
        leg_effects=[_leg(seasons[0], 1.4, 0.7), _leg(seasons[1], -0.2, 0.45)],
    )
    window = recorded.families["close_cand"].windows[0]
    assert window.leg_effects is not None
    assert {leg.season: leg.effect for leg in window.leg_effects} == {
        seasons[0]: pytest.approx(1.4),
        seasons[1]: pytest.approx(-0.2),
    }

    destination = tmp_path / "rotation_registry.json"
    save_registry(recorded, destination)
    reloaded = load_registry(destination)
    assert reloaded == recorded
    written = json.loads(destination.read_text(encoding="utf-8"))
    written_window = written["families"]["close_cand"]["windows"][0]
    assert written_window["window_kind"] == "stratified"
    assert sorted(entry["season"] for entry in written_window["leg_effects"]) == sorted(seasons)


def test_confirmation_split_legs_trains_strictly_before_each_leg_independently() -> None:
    # docs/era_stratified_windows_proposal.md point 2: EACH leg gets its own
    # forward-chained cutoff. Leg 2013's training therefore naturally
    # includes leg 2011's season data (2011 < 2013) -- the proposal only
    # forbids a leg training on data at or after its OWN scoring season, not
    # mutual exclusion between legs.
    registry = registry_from_payload(
        _payload(
            alpha=_family(
                grade="close",
                windows=[_window(seasons=[2011, 2013], window_kind="stratified")],
            )
        )
    )
    legs = confirmation_split_legs(_features(), registry, "alpha")
    assert [leg.season for leg in legs] == [2011, 2013]

    leg_2011, leg_2013 = legs
    assert sorted(leg_2011.scoring["season"].unique()) == [2011]
    assert sorted(leg_2011.training["season"].unique()) == [2009, 2010]
    assert leg_2011.training["gameday"].max() < leg_2011.scoring["gameday"].min()
    assert leg_2011.training["result"].notna().all()

    assert sorted(leg_2013.scoring["season"].unique()) == [2013]
    # 2011 (the other leg's own season) is legitimately part of leg 2013's
    # training set: it is strictly before 2013, and nothing here forbids a
    # later leg from training on an earlier leg's data.
    assert sorted(leg_2013.training["season"].unique()) == [2009, 2010, 2011, 2012]
    assert leg_2013.training["gameday"].max() < leg_2013.scoring["gameday"].min()

    for leg in legs:
        assert set(leg.scoring["game_type"]) == {"REG"}
        assert set(leg.training["game_type"]) == {"REG"}


def test_confirmation_split_legs_raises_for_a_contiguous_window() -> None:
    registry = registry_from_payload(
        _payload(alpha=_family(grade="close", windows=[_window(seasons=[2012, 2013])]))
    )
    with pytest.raises(RegistryError, match="contiguous; use confirmation_split"):
        confirmation_split_legs(_features(), registry, "alpha")


def test_confirmation_split_raises_for_a_stratified_window() -> None:
    registry = registry_from_payload(
        _payload(
            alpha=_family(
                grade="close",
                windows=[_window(seasons=[2011, 2013], window_kind="stratified")],
            )
        )
    )
    with pytest.raises(RegistryError, match="use confirmation_split_legs"):
        confirmation_split(_features(), registry, "alpha")


def test_stratified_leg_count_constant_is_two() -> None:
    # Pinned: the proposal's worked example and "same data budget per look
    # (2 seasons)" framing both describe exactly two legs. Extending to 3+ is
    # explicitly out of scope for this implementation.
    assert STRATIFIED_LEG_COUNT == 2


def test_cli_rotation_stratified_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "rotation_registry.json").write_text(
        SEEDED_REGISTRY.read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_dir))

    assert (
        cli.main(
            [
                "rotation",
                "declare",
                "--name",
                "era_cand",
                "--description",
                "era-stratified candidate",
                "--grade",
                "close",
                "--acknowledge-mined",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert cli.main(["rotation", "assign", "--name", "era_cand", "--stratified"]) == 0
    assigned = json.loads(capsys.readouterr().out)
    window = assigned["family"]["windows"][0]
    assert window["window_kind"] == "stratified"
    leg_a, leg_b = window["seasons"]

    with pytest.raises(SystemExit) as error:
        cli.main(["rotation", "assign", "--name", "era_cand", "--stratified", "--size", "3"])
    assert error.value.code == 2

    leg_effects_json = json.dumps(
        [
            {"season": leg_a, "effect": 1.1, "probability_positive": 0.6, "sample_blocks": 10},
            {"season": leg_b, "effect": 0.2, "probability_positive": 0.52, "sample_blocks": 9},
        ]
    )
    assert (
        cli.main(
            [
                "rotation",
                "record",
                "--name",
                "era_cand",
                "--artifact",
                "docs/era_cand.md",
                "--verdict",
                "unresolved",
                "--probability-positive",
                "0.58",
                "--leg-effects",
                leg_effects_json,
            ]
        )
        == 0
    )
    recorded = json.loads(capsys.readouterr().out)
    recorded_window = recorded["family"]["windows"][0]
    assert recorded_window["state"] == "spent"
    assert sorted(entry["season"] for entry in recorded_window["leg_effects"]) == sorted(
        [leg_a, leg_b]
    )


def test_cli_rotation_stratified_rejects_opener_grade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "rotation_registry.json").write_text(
        SEEDED_REGISTRY.read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_dir))

    assert (
        cli.main(
            [
                "rotation",
                "declare",
                "--name",
                "opener_cand",
                "--description",
                "x",
                "--grade",
                "opener",
            ]
        )
        == 0
    )
    capsys.readouterr()

    with pytest.raises(SystemExit) as error:
        cli.main(["rotation", "assign", "--name", "opener_cand", "--stratified"])
    assert error.value.code == 2
