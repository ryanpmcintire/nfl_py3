from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nfl_ats import cli
from nfl_ats.rotation import (
    GRADE_POOLS,
    Registry,
    RegistryError,
    assign_window,
    confirmation_split,
    declare_family,
    load_registry,
    record_look,
    registry_from_payload,
    registry_status,
    save_registry,
)

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
        "notes": "",
    }
    window.update(overrides)
    return window


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
    with pytest.raises(RegistryError, match="overlapping windows in its inheritance chain"):
        registry_from_payload(payload)


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
    # 2009-2012 starts sit below the warm-up floor; 2013-2017 starts all
    # intersect the inherited 2014-2017 spend.
    assert registry.families["qb_continuity_variant"].windows[0].seasons == (2018, 2020)
    registry = record_look(
        registry,
        "qb_continuity_variant",
        artifact="docs/variant.md",
        verdict="unresolved",
        probability_positive=0.42,
    )
    registry = assign_window(registry, "qb_continuity_variant")
    assert registry.families["qb_continuity_variant"].windows[1].seasons == (2021, 2023)
    assert registry.families["qb_continuity_variant"].status == "open"


def test_assignment_respects_the_warmup_floor() -> None:
    # Rule 9: the first three feature-table seasons (2009-2011) are warm-up
    # history — 500 training games plus 200 calibration prediction rows must
    # precede a window's first week — so the earliest assignable block starts
    # 2012 even though the nflverse_spread pool opens in 2009. The calibration
    # figure is derived, not inherited (see calibrate_cover_prediction_stream);
    # it was 400 until 2026-08-17, which put this floor at 2013.
    registry = declare_family(
        _seeded(), "fresh", description="untainted candidate", grade="nflverse_spread"
    )
    registry = assign_window(registry, "fresh")
    assert registry.families["fresh"].windows[0].seasons == (2012, 2014)


def test_capacity_partition_starts_at_the_warmup_floor() -> None:
    pools = registry_status(_seeded())["grade_pools"]
    for grade in ("close", "nflverse_spread"):
        assert pools[grade]["total_windows"] == 4
        assert pools[grade]["unspent_blocks"] == [[2018, 2020], [2021, 2023]]
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

    assert cli.main(["rotation", "assign", "--name", "stack"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["family"]["windows"][1]["seasons"] == [2022, 2023]
