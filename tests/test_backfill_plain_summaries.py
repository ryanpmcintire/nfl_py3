"""Tests for ``scripts/backfill_plain_summaries.py``'s TEMPLATE-driven
backfill mode (``--prefix``/``--family``/``--template-set``/``--apply``):
the ``cfb_home_side_location_v1`` and ``nfl_home_side_cell_v1`` cell
grammars, the banned-token safety net, and the preview/apply/
never-overwrite contract.

Closing-grounds taxonomy (verbatim, AGENTS.md/CLAUDE.md, binding, pasted
here because this file constructs and reads ``unresolved_below_power``
weak-signal rows): An interval or CI that contains zero is NEVER grounds to
reject, fail, or close an experiment. At this evaluator's ~2-point
resolution, "contains zero" is the EXPECTED outcome for a real small
signal. Only two grounds ever close a line of work: (1) refuted mechanism
-- a RESOLVED wrong sign (whole interval on the wrong side of zero) or zero
split-half reliability; (2) bounded by a positive control proven able to
detect an effect that size. Everything else is `unresolved_below_power`:
record it with `nfl-ats weak-signals record`, report `probability_positive`,
never the binary "contains zero". The registry code hard-rejects
inadmissible closures; if a record command errors, the verdict is wrong,
not the validator.

Nothing in this file adjudicates an experiment -- every fixture signal
below is built (and stays) `unresolved_below_power`, exactly like the real
MOD-18 rows this script backfills. What is under test is narrower: does the
generated PLAIN-ENGLISH TEXT correctly restate an already-recorded effect's
sign, size and population, and never leak the registry's own vocabulary
into it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.backfill_plain_summaries as backfill
from nfl_ats.weak_signals import (
    WEAK_SIGNAL_REGISTRY_VERSION,
    Registry,
    load_registry,
    save_registry,
    signal_from_payload,
)


@pytest.fixture(autouse=True)
def _never_touch_the_real_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Belt-and-suspenders isolation, on top of ``run_template_backfill``'s
    own ``_forced_registry_dir`` guard: every test in this file gets
    ``NFL_ATS_REGISTRY_DIR`` pointed at a scratch directory before it runs,
    so nothing here can EVER resolve to the real ``registry/weak_signals.json``
    -- even a test that (like this suite briefly did, 2026-09-08) forgets to
    pass an explicit isolated path somewhere. A test that wants a different
    scratch location may still override this with its own
    ``monkeypatch.setenv`` call."""

    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "isolated_registry"))


def _cfb_body(
    *,
    arm: str,
    era: str,
    spread: str,
    home: str,
    effect: float,
    effect_units: str,
    seasons: tuple[int, int],
    plain_summary: str | None = None,
) -> dict[str, Any]:
    """One ``mod18_home_side_location_cfb_v1_*`` row, shaped exactly like the
    real ones (``artifacts/research/laneO/cells.json``, family
    ``mod18_home_side_location_cfb_v1``, description ``"CFB {arm} {era}
    {spread} {home}"``)."""

    body: dict[str, Any] = {
        "recorded_at": "2026-09-08",
        "description": f"CFB {arm} {era} {spread} {home}",
        "source": "artifacts/research/laneO/cells.json",
        "effect": effect,
        "effect_units": effect_units,
        "classification": "unresolved_below_power",
        "league": "cfb",
        "seasons": list(seasons),
        "family": "mod18_home_side_location_cfb_v1",
        "category": "modeling",
        "notes": "Independent CFB replication of frozen NFL shape.",
    }
    if plain_summary is not None:
        body["plain_summary"] = plain_summary
    return body


def _nfl_body(
    *,
    family: str,
    cell_suffix: str,
    effect: float,
    effect_units: str,
    seasons: tuple[int, int],
    plain_summary: str | None = None,
) -> dict[str, Any]:
    """One row of the NFL cell grammar shared by
    ``mod18_home_side_location_v1_{s3,s4,s5}_*`` and
    ``mod18_conditional_margin_v1_{m1,mp1}_*``. The real rows' own
    ``description`` is boilerplate ("Home-side location arm {suffix},
    positive favours the candidate") -- reproduced here so the template
    function is exercised exactly as it runs on the name, never on a richer
    description this grammar does not actually have."""

    body: dict[str, Any] = {
        "recorded_at": "2026-09-08",
        "description": f"Home-side location arm {cell_suffix}, positive favours the candidate",
        "source": "artifacts/research/laneX/cells.json",
        "effect": effect,
        "effect_units": effect_units,
        "classification": "unresolved_below_power",
        "league": "nfl",
        "seasons": list(seasons),
        "family": family,
        "category": "modeling",
        "notes": "measured",
    }
    if plain_summary is not None:
        body["plain_summary"] = plain_summary
    return body


def _write_registry(path: Path, signals: dict[str, dict[str, Any]]) -> Registry:
    registry = Registry(
        version=WEAK_SIGNAL_REGISTRY_VERSION,
        notes=(),
        signals={name: signal_from_payload(name, body) for name, body in signals.items()},
    )
    save_registry(registry, path)
    return registry


def test_banned_tokens_in_flags_every_known_violation() -> None:
    assert backfill.banned_tokens_in("A short, clean sentence a fan can read.") == []
    assert backfill.banned_tokens_in("This is a wagering recommendation.") != []
    assert backfill.banned_tokens_in("probability_positive is high here") != []
    assert backfill.banned_tokens_in("logged at 20260908T110201Z") != []
    assert backfill.banned_tokens_in("timestamped 2026-09-08T11:02 exactly") != []
    assert backfill.banned_tokens_in("read as P+ 0.93 here") != []
    assert backfill.banned_tokens_in("used a week-blocked bootstrap") != []
    assert backfill.banned_tokens_in("policy mod18_home_side_location_v1 applies") != []
    assert backfill.banned_tokens_in("model a4c757efd2525da6 is active") != []


def test_cfb_diagnosis_matches_the_worked_example() -> None:
    """Reproduces AGENTS's/the lane brief's own worked example almost
    verbatim: a big-spread, home-underdog diagnosis cell where the home
    team beat the model's number by about 3 points."""

    body = _cfb_body(
        arm="diagnosis",
        era="era_2006_2011",
        spread="10.5+",
        home="home_underdog",
        effect=3.0,
        effect_units="ats_points",
        seasons=(2006, 2011),
    )
    signal = signal_from_payload("x", body)
    summary = backfill.describe_cfb_home_side_location_cell(signal)
    assert summary == (
        "College football check of the home-team push: on 10.5+ point spreads with "
        "a home underdog, 2006-2011, the home team beat the model's number by about "
        "3 points."
    )
    assert backfill.banned_tokens_in(summary) == []


def test_cfb_diagnosis_negative_effect_reads_missed() -> None:
    body = _cfb_body(
        arm="diagnosis",
        era="all",
        spread="7.5-10",
        home="home_favourite",
        effect=-1.7,
        effect_units="ats_points",
        seasons=(2007, 2025),
    )
    summary = backfill.describe_cfb_home_side_location_cell(signal_from_payload("x", body))
    assert "missed the model's number by about 1.7 points" in summary
    assert "a home favourite" in summary
    assert backfill.banned_tokens_in(summary) == []


def test_cfb_diagnosis_near_zero_effect_avoids_by_about_zero_points() -> None:
    body = _cfb_body(
        arm="diagnosis",
        era="all",
        spread="all",
        home="all",
        effect=0.01,
        effect_units="ats_points",
        seasons=(2007, 2025),
    )
    summary = backfill.describe_cfb_home_side_location_cell(signal_from_payload("x", body))
    assert "landed almost exactly on the model's number" in summary
    assert "0.0" not in summary
    assert "across every spread size" in summary


def test_cfb_diagnosis_clean_core_era_notes_a_cleaner_sample() -> None:
    body = _cfb_body(
        arm="diagnosis",
        era="clean_core",
        spread="0-3",
        home="all",
        effect=0.9,
        effect_units="ats_points",
        seasons=(2012, 2025),
    )
    summary = backfill.describe_cfb_home_side_location_cell(signal_from_payload("x", body))
    assert "2012-2025 (a cleaner sample of games)" in summary
    assert backfill.banned_tokens_in(summary) == []


def test_cfb_diagnosis_season_token_uses_single_year() -> None:
    body = _cfb_body(
        arm="diagnosis",
        era="season_2014",
        spread="7",
        home="home_underdog",
        effect=1.1,
        effect_units="ats_points",
        seasons=(2014, 2014),
    )
    summary = backfill.describe_cfb_home_side_location_cell(signal_from_payload("x", body))
    assert ", 2014," in summary
    assert "on 7 point spreads" in summary


@pytest.mark.parametrize("arm", ["s2", "s3"])
@pytest.mark.parametrize(
    "effect_units,effect",
    [("accuracy_points", 0.4), ("brier_improvement", 0.00072), ("log_loss_improvement", -0.0013)],
)
def test_cfb_serve_arms_cover_every_effect_unit(arm: str, effect_units: str, effect: float) -> None:
    body = _cfb_body(
        arm=arm,
        era="era_2021_2025",
        spread="all",
        home="all",
        effect=effect,
        effect_units=effect_units,
        seasons=(2021, 2025),
    )
    summary = backfill.describe_cfb_home_side_location_cell(signal_from_payload("x", body))
    assert summary.startswith("College football check of the home-team push:")
    assert "2021-2025" in summary
    if arm == "s3":
        assert "only on spreads of seven points or more" in summary
    else:
        assert "across every spread size" in summary
    assert backfill.banned_tokens_in(summary) == []


def test_cfb_serve_arm_names_its_own_sub_bucket_when_not_all() -> None:
    body = _cfb_body(
        arm="s3",
        era="all",
        spread="7.5-10",
        home="all",
        effect=-0.9,
        effect_units="accuracy_points",
        seasons=(2020, 2025),
    )
    summary = backfill.describe_cfb_home_side_location_cell(signal_from_payload("x", body))
    assert "looking only at 7.5-10 point spreads" in summary
    assert "worse than not applying it" in summary


def test_cfb_unrecognized_arm_raises() -> None:
    body = _cfb_body(
        arm="s9",
        era="all",
        spread="all",
        home="all",
        effect=0.1,
        effect_units="accuracy_points",
        seasons=(2020, 2025),
    )
    with pytest.raises(ValueError, match="unrecognised CFB arm"):
        backfill.describe_cfb_home_side_location_cell(signal_from_payload("x", body))


def test_cfb_malformed_description_raises() -> None:
    body = _cfb_body(
        arm="diagnosis",
        era="all",
        spread="all",
        home="all",
        effect=0.1,
        effect_units="ats_points",
        seasons=(2020, 2025),
    )
    body["description"] = "not the expected shape at all"
    with pytest.raises(ValueError, match="does not match"):
        backfill.describe_cfb_home_side_location_cell(signal_from_payload("x", body))


def test_nfl_s3_vs_s2_shape() -> None:
    name = "mod18_home_side_location_v1_s3_2020_vs_s2"
    body = _nfl_body(
        family="mod18_home_side_location_v1",
        cell_suffix="s3_2020_vs_s2",
        effect=0.4545,
        effect_units="accuracy_points",
        seasons=(2020, 2020),
    )
    summary = backfill.describe_nfl_home_side_cell(signal_from_payload(name, body))
    assert summary.startswith("Comparing the home-team push applied only on spreads of seven")
    assert "2020 NFL season" in summary
    assert "better than applying it on every spread" in summary
    assert backfill.banned_tokens_in(summary) == []


@pytest.mark.parametrize(
    "group,variant,note_fragment",
    [
        ("s4", "s4", None),
        ("s4", "s4b", "50-game sample"),
        ("s5", "s5a", "50-game sample"),
        ("s5", "s5b", "25-game sample"),
        ("s5", "s5c", "200-game sample"),
    ],
)
@pytest.mark.parametrize("home", ["home_favourite", "home_underdog", "pickem", None])
@pytest.mark.parametrize(
    "metric,effect_units,effect",
    [
        ("standalone", "accuracy_points", 2.1),
        ("card", "accuracy_points", -1.2),
        ("brier", "brier_improvement", 0.00043),
        ("log_loss", "log_loss_improvement", -0.0009),
    ],
)
def test_nfl_home_side_location_cell_bucket_shapes(
    group: str,
    variant: str,
    note_fragment: str | None,
    home: str | None,
    metric: str,
    effect_units: str,
    effect: float,
) -> None:
    bucket = "10p5plus"
    home_part = f"_{home}" if home else "_all"
    name = (
        f"mod18_home_side_location_v1_{group}_{variant}_cell_bucket_"
        f"{bucket}{home_part}_{metric}_2020_2025"
    )
    cell_suffix = name[len(f"mod18_home_side_location_v1_{group}_") :]
    body = _nfl_body(
        family="mod18_home_side_location_v1",
        cell_suffix=cell_suffix,
        effect=effect,
        effect_units=effect_units,
        seasons=(2020, 2025),
    )
    summary = backfill.describe_nfl_home_side_cell(signal_from_payload(name, body))
    assert summary.startswith("Check of the home-team push in the NFL")
    assert "10.5+ point spreads" in summary
    assert "2020-2025" in summary
    if note_fragment:
        assert note_fragment in summary
    if home in ("home_favourite", "home_underdog"):
        assert "the home team" in summary
    elif home == "pickem":
        assert "pick'em" in summary
    assert backfill.banned_tokens_in(summary) == []


@pytest.mark.parametrize(
    "shape_suffix,expected_fragment",
    [
        ("overall_standalone_2020_2025", "across the whole sample"),
        ("season_2023_card_2023_2023", "in the 2023 season"),
    ],
)
def test_nfl_overall_and_season_shapes(shape_suffix: str, expected_fragment: str) -> None:
    name = f"mod18_home_side_location_v1_s5_s5a_{shape_suffix}"
    cell_suffix = f"s5a_{shape_suffix}"
    body = _nfl_body(
        family="mod18_home_side_location_v1",
        cell_suffix=cell_suffix,
        effect=0.6,
        effect_units="accuracy_points",
        seasons=(2020, 2025),
    )
    summary = backfill.describe_nfl_home_side_cell(signal_from_payload(name, body))
    assert expected_fragment in summary
    assert "50-game sample" in summary
    assert backfill.banned_tokens_in(summary) == []


def test_nfl_conditional_margin_push_at_3_shape() -> None:
    name = "mod18_conditional_margin_v1_m1_m1_push_at_3_brier_2020_2025"
    body = _nfl_body(
        family="mod18_conditional_margin_v1",
        cell_suffix="m1_push_at_3_brier_2020_2025",
        effect=0.0002,
        effect_units="brier_improvement",
        seasons=(2020, 2025),
    )
    summary = backfill.describe_nfl_home_side_cell(signal_from_payload(name, body))
    assert "3-point push" in summary
    assert "reading the home team's win chance from nearby past final scores" in summary
    assert backfill.banned_tokens_in(summary) == []


def test_nfl_conditional_margin_m1b_variant_names_key_number_check() -> None:
    name = "mod18_conditional_margin_v1_m1_m1b_cell_bucket_7p5-10_all_standalone_2020_2025"
    body = _nfl_body(
        family="mod18_conditional_margin_v1",
        cell_suffix="m1b_cell_bucket_7p5-10_all_standalone_2020_2025",
        effect=-1.0,
        effect_units="accuracy_points",
        seasons=(2020, 2025),
    )
    summary = backfill.describe_nfl_home_side_cell(signal_from_payload(name, body))
    assert "which side of the nearest key number" in summary
    assert "7.5-10 point spreads" in summary


def test_nfl_mp1_overall_conditional_log_loss_shape() -> None:
    name = "mod18_conditional_margin_v1_mp1_mp1_overall_conditional_log_loss_2020_2025"
    body = _nfl_body(
        family="mod18_conditional_margin_v1",
        cell_suffix="mp1_overall_conditional_log_loss_2020_2025",
        effect=-0.0002,
        effect_units="log_loss_improvement",
        seasons=(2020, 2025),
    )
    summary = backfill.describe_nfl_home_side_cell(signal_from_payload(name, body))
    assert "weighted toward the model's own pick" in summary
    assert "further from what actually happened" in summary
    assert backfill.banned_tokens_in(summary) == []


def test_nfl_unrecognized_family_raises() -> None:
    name = "mod18_some_other_family_v1_s4_s4_overall_brier_2020_2025"
    body = _nfl_body(
        family="mod18_some_other_family_v1",
        cell_suffix="s4_overall_brier_2020_2025",
        effect=0.1,
        effect_units="brier_improvement",
        seasons=(2020, 2025),
    )
    with pytest.raises(ValueError, match="not one of"):
        backfill.describe_nfl_home_side_cell(signal_from_payload(name, body))


def test_nfl_unrecognized_group_raises() -> None:
    name = "mod18_home_side_location_v1_s9_s9_overall_brier_2020_2025"
    body = _nfl_body(
        family="mod18_home_side_location_v1",
        cell_suffix="s9_overall_brier_2020_2025",
        effect=0.1,
        effect_units="brier_improvement",
        seasons=(2020, 2025),
    )
    with pytest.raises(ValueError, match="unrecognised sub-arm group"):
        backfill.describe_nfl_home_side_cell(signal_from_payload(name, body))


def test_nfl_unrecognized_cell_shape_raises() -> None:
    name = "mod18_home_side_location_v1_s4_s4_totally_unknown_shape_2020_2025"
    body = _nfl_body(
        family="mod18_home_side_location_v1",
        cell_suffix="s4_totally_unknown_shape_2020_2025",
        effect=0.1,
        effect_units="brier_improvement",
        seasons=(2020, 2025),
    )
    with pytest.raises(ValueError, match="unrecognised NFL cell shape"):
        backfill.describe_nfl_home_side_cell(signal_from_payload(name, body))


def test_select_candidates_scopes_by_prefix_family_and_excludes_filled_rows(
    tmp_path: Path,
) -> None:
    signals = {
        "mod18_home_side_location_cfb_v1_diagnosis_all_all_all_ats_points": _cfb_body(
            arm="diagnosis",
            era="all",
            spread="all",
            home="all",
            effect=0.2,
            effect_units="ats_points",
            seasons=(2020, 2025),
        ),
        "mod18_home_side_location_cfb_v1_s2_all_all_all_accuracy_points": _cfb_body(
            arm="s2",
            era="all",
            spread="all",
            home="all",
            effect=0.2,
            effect_units="accuracy_points",
            seasons=(2020, 2025),
            plain_summary="Already has one.",
        ),
        "unrelated_other_family_row": _nfl_body(
            family="mod18_home_side_location_v1",
            cell_suffix="s4_s4_overall_brier_2020_2025",
            effect=0.1,
            effect_units="brier_improvement",
            seasons=(2020, 2025),
        ),
    }
    registry_path = tmp_path / "weak_signals.json"
    registry = _write_registry(registry_path, signals)

    by_prefix = backfill._select_candidates(
        registry, prefix="mod18_home_side_location_cfb_v1_", family=None
    )
    assert [s.name for s in by_prefix] == [
        "mod18_home_side_location_cfb_v1_diagnosis_all_all_all_ats_points"
    ]

    by_family = backfill._select_candidates(
        registry, prefix=None, family="mod18_home_side_location_v1"
    )
    assert [s.name for s in by_family] == ["unrelated_other_family_row"]

    by_both = backfill._select_candidates(
        registry,
        prefix="mod18_home_side_location_cfb_v1_",
        family="mod18_home_side_location_cfb_v1",
    )
    assert [s.name for s in by_both] == [
        "mod18_home_side_location_cfb_v1_diagnosis_all_all_all_ats_points"
    ]

    none_match = backfill._select_candidates(registry, prefix="does_not_exist_", family=None)
    assert none_match == []


def _cfb_fixture_registry(tmp_path: Path) -> Path:
    signals = {
        "mod18_home_side_location_cfb_v1_diagnosis_all_all_all_ats_points": _cfb_body(
            arm="diagnosis",
            era="all",
            spread="all",
            home="all",
            effect=0.6,
            effect_units="ats_points",
            seasons=(2007, 2025),
        ),
        "mod18_home_side_location_cfb_v1_s2_all_all_all_accuracy_points": _cfb_body(
            arm="s2",
            era="all",
            spread="all",
            home="all",
            effect=-0.3,
            effect_units="accuracy_points",
            seasons=(2007, 2025),
        ),
        "mod18_home_side_location_cfb_v1_s3_all_all_all_accuracy_points": _cfb_body(
            arm="s3",
            era="all",
            spread="all",
            home="all",
            effect=0.9,
            effect_units="accuracy_points",
            seasons=(2007, 2025),
            plain_summary="A hand-written summary that must never be touched.",
        ),
        "mod18_home_side_location_cfb_v1_diagnosis_all_all_broken_ats_points": {
            **_cfb_body(
                arm="diagnosis",
                era="all",
                spread="all",
                home="all",
                effect=0.6,
                effect_units="ats_points",
                seasons=(2007, 2025),
            ),
            "description": "not a shape the template recognises",
        },
        "unrelated_row_outside_prefix": _nfl_body(
            family="mod18_home_side_location_v1",
            cell_suffix="s4_s4_overall_brier_2020_2025",
            effect=0.1,
            effect_units="brier_improvement",
            seasons=(2020, 2025),
        ),
    }
    registry_path = tmp_path / "weak_signals.json"
    _write_registry(registry_path, signals)
    return registry_path


def test_dry_run_previews_without_writing(tmp_path: Path) -> None:
    registry_path = _cfb_fixture_registry(tmp_path)
    before_bytes = registry_path.read_bytes()

    report = backfill.run_template_backfill(
        registry_path,
        prefix="mod18_home_side_location_cfb_v1_",
        family=None,
        template_set="cfb_home_side_location_v1",
        apply=False,
    )

    assert registry_path.read_bytes() == before_bytes
    assert report["applied"] is False
    assert report["candidates"] == 3
    assert report["generated_count"] == 2
    assert report["skipped_count"] == 1
    assert (
        "mod18_home_side_location_cfb_v1_diagnosis_all_all_broken_ats_points" in report["skipped"]
    )
    assert (
        "mod18_home_side_location_cfb_v1_s3_all_all_all_accuracy_points" not in report["generated"]
    )


def test_apply_writes_only_plain_summary_on_matching_rows_lacking_one(tmp_path: Path) -> None:
    registry_path = _cfb_fixture_registry(tmp_path)
    before = load_registry(registry_path)

    report = backfill.run_template_backfill(
        registry_path,
        prefix="mod18_home_side_location_cfb_v1_",
        family=None,
        template_set="cfb_home_side_location_v1",
        apply=True,
    )

    assert report["applied"] is True
    assert report["recorded_count"] == 2
    assert report["non_plain_summary_changes"] == []

    after = load_registry(registry_path)
    for name in before.signals:
        before_signal = before.signals[name]
        after_signal = after.signals[name]
        if name in report["recorded"]:
            assert after_signal.plain_summary
            assert after_signal.plain_summary != before_signal.plain_summary
            assert after_signal.__dict__ == {
                **before_signal.__dict__,
                "plain_summary": after_signal.plain_summary,
            }
        else:
            assert after_signal == before_signal


def test_apply_never_overwrites_an_existing_plain_summary(tmp_path: Path) -> None:
    registry_path = _cfb_fixture_registry(tmp_path)
    hand_written = "A hand-written summary that must never be touched."

    backfill.run_template_backfill(
        registry_path,
        prefix="mod18_home_side_location_cfb_v1_",
        family=None,
        template_set="cfb_home_side_location_v1",
        apply=True,
    )

    after = load_registry(registry_path)
    assert (
        after.signals[
            "mod18_home_side_location_cfb_v1_s3_all_all_all_accuracy_points"
        ].plain_summary
        == hand_written
    )


def test_apply_leaves_unparseable_rows_still_missing_a_summary(tmp_path: Path) -> None:
    registry_path = _cfb_fixture_registry(tmp_path)

    backfill.run_template_backfill(
        registry_path,
        prefix="mod18_home_side_location_cfb_v1_",
        family=None,
        template_set="cfb_home_side_location_v1",
        apply=True,
    )

    after = load_registry(registry_path)
    broken = after.signals["mod18_home_side_location_cfb_v1_diagnosis_all_all_broken_ats_points"]
    assert broken.plain_summary is None


def test_apply_never_touches_rows_outside_the_selector(tmp_path: Path) -> None:
    registry_path = _cfb_fixture_registry(tmp_path)
    before_unrelated = load_registry(registry_path).signals["unrelated_row_outside_prefix"]

    backfill.run_template_backfill(
        registry_path,
        prefix="mod18_home_side_location_cfb_v1_",
        family=None,
        template_set="cfb_home_side_location_v1",
        apply=True,
    )

    after_unrelated = load_registry(registry_path).signals["unrelated_row_outside_prefix"]
    assert after_unrelated == before_unrelated


def test_run_template_backfill_with_no_candidates_is_a_no_op(tmp_path: Path) -> None:
    registry_path = _cfb_fixture_registry(tmp_path)
    before_bytes = registry_path.read_bytes()

    report = backfill.run_template_backfill(
        registry_path,
        prefix="name_prefix_matching_nothing_",
        family=None,
        template_set="cfb_home_side_location_v1",
        apply=True,
    )

    assert report["candidates"] == 0
    assert report["applied"] is False
    assert registry_path.read_bytes() == before_bytes


def test_nfl_template_set_applies_through_run_template_backfill(tmp_path: Path) -> None:
    signals = {
        "mod18_conditional_margin_v1_m1_m1_overall_standalone_2020_2025": _nfl_body(
            family="mod18_conditional_margin_v1",
            cell_suffix="m1_overall_standalone_2020_2025",
            effect=0.3,
            effect_units="accuracy_points",
            seasons=(2020, 2025),
        ),
    }
    registry_path = tmp_path / "weak_signals.json"
    _write_registry(registry_path, signals)

    report = backfill.run_template_backfill(
        registry_path,
        prefix=None,
        family="mod18_conditional_margin_v1",
        template_set="nfl_home_side_cell_v1",
        apply=True,
    )

    assert report["recorded_count"] == 1
    after = load_registry(registry_path)
    summary = after.signals[
        "mod18_conditional_margin_v1_m1_m1_overall_standalone_2020_2025"
    ].plain_summary
    assert summary is not None
    assert "reading the home team's win chance from nearby past final scores" in summary
    assert backfill.banned_tokens_in(summary) == []


def test_main_template_mode_previews_by_default_and_requires_apply_to_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry_path = _cfb_fixture_registry(tmp_path)
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_path.parent))
    before_bytes = registry_path.read_bytes()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill_plain_summaries.py",
            "--prefix",
            "mod18_home_side_location_cfb_v1_",
            "--template-set",
            "cfb_home_side_location_v1",
        ],
    )
    backfill.main()
    out = capsys.readouterr().out
    assert registry_path.read_bytes() == before_bytes
    assert "->" in out
    assert '"applied": false' in out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill_plain_summaries.py",
            "--prefix",
            "mod18_home_side_location_cfb_v1_",
            "--template-set",
            "cfb_home_side_location_v1",
            "--apply",
        ],
    )
    backfill.main()
    assert registry_path.read_bytes() != before_bytes
    after = load_registry(registry_path)
    assert after.signals[
        "mod18_home_side_location_cfb_v1_diagnosis_all_all_all_ats_points"
    ].plain_summary


def test_main_rejects_template_set_without_prefix_or_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["backfill_plain_summaries.py", "--template-set", "cfb_home_side_location_v1"],
    )
    with pytest.raises(SystemExit, match="requires --prefix"):
        backfill.main()


def test_main_rejects_prefix_without_template_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    monkeypatch.setattr(
        sys, "argv", ["backfill_plain_summaries.py", "--prefix", "mod18_home_side_location_cfb_v1_"]
    )
    with pytest.raises(SystemExit, match="require --template-set"):
        backfill.main()


def test_main_missing_plain_summary_mode_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unaffected by this lane's additions: still reports the live findings
    page's plain_summary backlog and exits without touching the registry."""

    registry_path = tmp_path / "registry" / "weak_signals.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    _write_registry(registry_path, {})
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_path.parent))
    monkeypatch.setattr(sys, "argv", ["backfill_plain_summaries.py", "--missing-plain-summary"])
    backfill.main()
    payload = json.loads(capsys.readouterr().out)
    assert "watching_leads_missing_plain_summary" in payload
