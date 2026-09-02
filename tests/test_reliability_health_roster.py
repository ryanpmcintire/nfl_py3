"""Guards for ``scripts/reliability_health_roster.py`` (ORCH-D health_roster).

Two things are proved here, and they are the two things that would silently
corrupt the 46 recorded reliabilities:

1. **The parent-trait mapping matches each builder's own definition.** The
   script maps every registry cell to a parent team-week quantity. If a
   builder renames a feature family or a screen renames a count column, the
   mapping would keep running and would quietly measure the wrong thing. Each
   test below imports the BUILDER's own constant and asserts equality, so a
   rename breaks the test instead of the number.

2. **The split arithmetic is right on a known answer.** A synthetic long frame
   with a hand-computable Pearson r is fed through ``measure_reliability`` and
   checked against the arithmetic, including the Spearman-Brown step-up, plus
   the case that matters most for the taxonomy: a frame with too few units
   must come back UNMEASURED with ``reliability is None``, never as 0.

Binding taxonomy (verbatim, AGENTS.md / CLAUDE.md): an interval or CI that
contains zero is NEVER grounds to reject, fail, or close an experiment. Only
two grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED
wrong sign or zero split-half reliability; (2) bounded by a positive control.
Everything else is ``unresolved_below_power``. Nothing here closes anything.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _load(name: str) -> ModuleType:
    if str(SCRIPTS) not in sys.path:
        sys.path.append(str(SCRIPTS))
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sweep() -> ModuleType:
    return _load("reliability_health_roster")


@pytest.fixture(scope="module")
def rlib() -> ModuleType:
    return _load("reliability_lib")


# ---------------------------------------------------------------------------
# (a) the parent-trait mapping matches each builder's own definition
# ---------------------------------------------------------------------------


def test_injury_value_lost_parents_are_the_player_values_family(sweep: ModuleType) -> None:
    """Cell 1 of 3: the injury value-lost parents ARE the marginal block.

    The eight ``injury_value_lost_*`` cells score a D-A contrast whose
    marginal feature block is ``FEATURE_FAMILIES["player_values"]``. If that
    family is ever renamed or extended, this sweep's two hard-coded parent
    columns stop being the block and the recorded number stops meaning what
    its ``--reason`` says.
    """

    from nfl_ats.constants import FEATURE_FAMILIES

    family = FEATURE_FAMILIES["player_values"]
    derived = tuple(column.removeprefix("diff_") for column in family)
    assert derived == sweep.INJURY_VALUE_COLUMNS
    for column in family:
        assert column.startswith("diff_"), column


def test_player_family_blocks_match_constants_feature_families(sweep: ModuleType) -> None:
    """Cell 2 of 3: every ablation cell's block members come from the repo's map.

    ``block_member_metrics`` must return exactly the ``diff_``-stripped
    members of the named ``FEATURE_FAMILIES`` entries -- not a hand-copied
    list that can drift from ``constants.py``.
    """

    from nfl_ats.constants import FEATURE_FAMILIES

    for _entry, families in sweep.PLAYER_FAMILY_BLOCKS.items():
        expected: list[str] = []
        for family in families:
            assert family in FEATURE_FAMILIES, family
            for column in FEATURE_FAMILIES[family]:
                metric = column.removeprefix("diff_")
                if metric not in expected:
                    expected.append(metric)
        assert sweep.block_member_metrics(families) == expected

    # And the composed arms really are the union their FEATURE_SETS name says.
    from nfl_ats.constants import FEATURE_SETS

    base = set(FEATURE_SETS["football"])
    for profile, families in (
        ("football_player_qb", ("player_qb",)),
        ("football_player_injuries", ("player_injuries",)),
        ("football_player_continuity", ("player_continuity",)),
        (
            "football_player_value",
            ("player_qb", "player_injuries", "player_continuity", "player_values"),
        ),
    ):
        marginal = set(FEATURE_SETS[profile]) - base
        expected_columns = {c for f in families for c in FEATURE_FAMILIES[f]}
        assert marginal == expected_columns, profile


def test_participation_block_matches_the_experiment_baseline(sweep: ModuleType) -> None:
    """The participation cell's marginal block is player_participation minus player_value."""

    from nfl_ats.constants import FEATURE_FAMILIES, FEATURE_SETS

    marginal = set(FEATURE_SETS["football_player_participation"]) - set(
        FEATURE_SETS["football_player_value"]
    )
    assert marginal == set(FEATURE_FAMILIES["player_participation_values"])
    assert sweep.PARTICIPATION_BLOCKS["participation_offense_defense_rapm"] == (
        "player_participation_values",
    )


def test_nflcom_parent_columns_are_the_screens_own_aggregate_names(sweep: ModuleType) -> None:
    """Cell 3 of 3: the NFL.com parents are the count columns the screens emit.

    ``attach_flags`` names its aggregates inline, so the guard is that the
    names this sweep records against are exactly the ones the screen's own
    source text produces -- a rename in either file breaks the test rather
    than silently re-pointing a recorded reliability.
    """

    designation_source = (SCRIPTS / "nflcom_friday_designation_screen.py").read_text(
        encoding="utf-8"
    )
    for column in ("q_or_worse_any=", "out_count=", "starter_q_or_worse=", "new_vs_tuesday="):
        assert column in designation_source, column
    refresh_source = (SCRIPTS / "nflcom_friday_refresh_feature.py").read_text(encoding="utf-8")
    for column in ("total_out=", "starter_out="):
        assert column in refresh_source, column

    emitted = {"q_or_worse_any", "out_count", "starter_q_or_worse", "new_vs_tuesday"} | {
        "total_out",
        "starter_out",
    }
    for entry, (parent, _source, _note) in sweep.NFLCOM_CELL_PARENTS.items():
        assert parent in emitted, f"{entry} -> {parent}"

    # The starter-out threshold the two out>=2 cells fade on is a real
    # production constant, not a number this sweep invented.
    from nfl_ats.prospective import NFLCOM_STARTER_OUT_THRESHOLD

    assert NFLCOM_STARTER_OUT_THRESHOLD == 2
    assert sweep.NFLCOM_FLAGS["nflcom_refresh_out2_starters_on_chain"] == (
        "starter_out",
        float(NFLCOM_STARTER_OUT_THRESHOLD),
    )


def test_flag_cells_name_real_registered_flag_builders(sweep: ModuleType) -> None:
    """Every EXPOSURE cell delegates to a builder that actually exists."""

    from nfl_ats.experiment_runner import FLAG_BUILDERS

    for entry, config in sweep.FLAG_CELLS.items():
        assert config["builder"] in FLAG_BUILDERS, f"{entry} -> {config['builder']}"
        spec = REPO / "registry" / "experiment_specs" / f"{entry}.json"
        if spec.is_file():
            import json

            payload = json.loads(spec.read_text(encoding="utf-8"))
            assert payload["construct"]["flag_builder"] == config["builder"], entry


# ---------------------------------------------------------------------------
# (b) split arithmetic on a known answer
# ---------------------------------------------------------------------------


def _known_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """24 units, 2 odd-week and 2 even-week observations each.

    The per-unit odd/even MEANS are chosen outright, so the Pearson r the
    estimator must return is just ``np.corrcoef`` of those two vectors -- no
    reimplementation of the estimator, an independent hand computation of the
    same number.
    """

    rng = np.random.default_rng(7)
    n_units = 24
    odd_means = np.linspace(-1.0, 1.0, n_units)
    even_means = 0.6 * odd_means + 0.4 * rng.normal(size=n_units)

    rows = []
    for index in range(n_units):
        # Two odd-week values averaging exactly odd_means[index], and likewise
        # for the even weeks -- the +/- offsets cancel in the mean.
        unit = f"T{index:02d}"
        for week, value in (
            (1, odd_means[index] - 0.5),
            (3, odd_means[index] + 0.5),
            (2, even_means[index] - 0.25),
            (4, even_means[index] + 0.25),
        ):
            rows.append({"team_id": unit, "season": 2020, "week": week, "x": value})
    return pd.DataFrame(rows), odd_means, even_means


def test_measure_reliability_reproduces_a_hand_computed_pearson_r(rlib: ModuleType) -> None:
    frame, odd_means, even_means = _known_frame()
    expected_r = float(np.corrcoef(odd_means, even_means)[0, 1])

    result = rlib.measure_reliability(
        frame, "x", method=rlib.METHOD_TRAIT, seasons=(2020, 2020), n_boot=200
    )

    assert result["status"] == rlib.STATUS_MEASURED
    assert result["n_units"] == 24
    assert result["pearson_r"] == pytest.approx(expected_r, abs=1e-12)


def test_spearman_brown_step_up_is_two_r_over_one_plus_r(rlib: ModuleType) -> None:
    frame, odd_means, even_means = _known_frame()
    expected_r = float(np.corrcoef(odd_means, even_means)[0, 1])
    expected_sb = (2.0 * expected_r) / (1.0 + expected_r)
    assert -1.0 <= expected_sb <= 1.0  # the reportable branch, not the fallback

    result = rlib.measure_reliability(
        frame, "x", method=rlib.METHOD_TRAIT, seasons=(2020, 2020), n_boot=200
    )

    assert result["reliability"] == pytest.approx(expected_sb, abs=1e-12)
    assert result["spearman_brown_full_length_reliability"] == pytest.approx(expected_sb, abs=1e-12)
    # The recorded interval must always bracket the recorded point estimate --
    # the set-reliability validator enforces this and the artifact must not
    # depend on that second line of defence.
    assert result["reliability_low"] <= result["reliability"] <= result["reliability_high"]


def test_too_few_units_is_unmeasured_never_reliability_zero(rlib: ModuleType) -> None:
    """The taxonomy's sharpest edge: an unmeasurable reliability is NOT a zero.

    Writing NaN through as a number would manufacture the appearance of the
    ``no_split_half_reliability`` closing ground out of nothing.
    """

    frame, _odd, _even = _known_frame()
    thin = frame.loc[frame["team_id"].isin({f"T{i:02d}" for i in range(5)})]

    result = rlib.measure_reliability(
        thin, "x", method=rlib.METHOD_TRAIT, seasons=(2020, 2020), n_boot=100
    )

    assert result["status"] != rlib.STATUS_MEASURED
    assert result["status"] == rlib.STATUS_INSUFFICIENT_UNITS
    assert result["reliability"] is None
    assert result["reliability_low"] is None
    assert result["reliability_high"] is None


def test_binary_flag_is_not_condemned_for_having_two_distinct_values(sweep: ModuleType) -> None:
    """The guard that misfired once already, pinned so it cannot misfire again.

    A 0/1 exposure indicator ALWAYS has exactly two distinct values; applying
    the continuous-parent distinct-value test to it would report every
    EXPOSURE measurement as ``not_informative_near_constant``. The test still
    has to bite on a genuinely degenerate CONTINUOUS column.
    """

    flag = pd.Series([0.0] * 90 + [1.0] * 10)
    degenerate, stats = sweep._near_constant(flag, binary_flag=True)
    assert degenerate is False
    assert stats["n_distinct"] == 2
    assert stats["binary_flag"] is True

    two_valued_continuous = pd.Series([0.0] * 90 + [3.5] * 10)
    degenerate, _stats = sweep._near_constant(two_valued_continuous, binary_flag=False)
    assert degenerate is True

    almost_all_zero = pd.Series([0.0] * 999 + [1.0, 2.0, 3.0])
    degenerate, stats = sweep._near_constant(almost_all_zero, binary_flag=False)
    assert degenerate is True
    assert stats["nonzero_share"] < sweep.NEAR_CONSTANT_NONZERO_SHARE


def test_within_unit_variation_detects_a_season_constant_quantity(sweep: ModuleType) -> None:
    """A season-constant parent has a split-half of +1.0 by construction.

    ``ffc_adp_*`` and ``hc_year_one_fade`` are exactly this shape, and the
    sweep must detect it from the design rather than read the +1.0 as a trait.
    """

    constant = pd.DataFrame(
        {
            "team_id": ["A"] * 4 + ["B"] * 4,
            "season": [2020] * 8,
            "week": [1, 2, 3, 4] * 2,
            "x": [5.0] * 4 + [9.0] * 4,
        }
    )
    diagnosis = sweep.within_unit_variation(constant, "x", unit_col="team_id", seasons=(2020, 2020))
    assert diagnosis["share_zero_within_variance"] == 1.0
    assert diagnosis["mean_within_std"] == 0.0

    varying, _odd, _even = _known_frame()
    diagnosis = sweep.within_unit_variation(varying, "x", unit_col="team_id", seasons=(2020, 2020))
    assert diagnosis["share_zero_within_variance"] == 0.0
    assert diagnosis["mean_within_std"] > 0.0


def test_block_summary_rule_is_the_minimum_member(sweep: ModuleType) -> None:
    """The stated summary rule for a feature BLOCK, pinned."""

    members = [
        {"metric": "a", "status": "measured", "reliability": 0.9},
        {"metric": "b", "status": "measured", "reliability": 0.2},
        {"metric": "c", "status": "measured", "reliability": 0.5},
    ]
    summary = sweep._summarise(members)
    assert summary["summary_rule"] == "min_member_reliability"
    assert summary["chosen"] == "b"
    assert summary["usable_members"] == 3

    unusable = [{"metric": "a", "status": "insufficient_split_units", "reliability": None}]
    assert sweep._summarise(unusable)["chosen"] is None


def test_every_manifest_entry_has_exactly_one_mapping(sweep: ModuleType) -> None:
    """All 46 cells are covered once, by exactly one group -- none silently dropped."""

    groups = [
        set(sweep.INJURY_CELL_ARMS),
        set(sweep.PLAYER_FAMILY_BLOCKS),
        set(sweep.PARTICIPATION_BLOCKS),
        set(sweep.NFLCOM_CELL_PARENTS),
        set(sweep.FLAG_CELLS),
        set(sweep.FLUVIEW_CELLS),
        set(sweep.FFC_CELLS),
        {"hc_year_one_fade", "qb_news_backup_visible_by_deadline_screen"},
    ]
    union: set[str] = set()
    for group in groups:
        assert not (union & group), sorted(union & group)
        union |= group
    assert len(union) == 46

    from nfl_ats.weak_signals import default_registry_path, load_registry

    registry = load_registry(default_registry_path(REPO / "registry"))
    missing = sorted(name for name in union if name not in registry.signals)
    assert not missing, missing


def test_exposure_method_string_says_it_is_not_a_closing_ground(rlib: ModuleType) -> None:
    """A recorded EXPOSURE number must carry its own disclaimer into the registry."""

    assert "NOT an admissible no_split_half_reliability ground" in rlib.METHOD_EXPOSURE
    assert "team-season" in rlib.METHOD_TRAIT
    assert math.isclose(rlib.RELIABILITY_SEED, 20260901)
