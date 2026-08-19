from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.experiment_runner import (
    FLAG_BUILDERS,
    HONEST_REFIT_WIDENING_UPPER_BOUND,
    ExperimentRunnerError,
    ExperimentSpecError,
    _base_team_game_table,
    _block_bootstrap_subset_gap,
    _flag_home_underdog,
    _flag_large_favorite,
    _RegistryLock,
    classify_subset_bias_result,
    experiment_spec_from_payload,
    experiment_spec_to_payload,
    load_experiment_spec,
    run_experiment,
    run_experiment_cli,
    run_subset_bias_experiment,
    scale_subset_effect,
    widening_factor_to_recross_zero,
)
from nfl_ats.weak_signals import WeakSignalError, load_registry

REPO_ROOT = Path(__file__).resolve().parents[1]


def _spec_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "example_subset_bias",
        "hypothesis": "Some pregame-safe subset covers more than its complement.",
        "experiment_type": "subset_bias",
        "population": {"league": "nfl", "seasons": [2009, 2025], "grade": "close"},
        "construct": {"flag_builder": "home_underdog", "params": {}},
        "endpoints": {"primary": "accuracy", "secondary": []},
        "blocking": {"primary": "week", "secondary": "season"},
        "samples": 2000,
        "seed": 20260818,
        "reliability_check": {"method": "not_applicable", "reason": "situational, not a trait"},
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_valid_spec_parses_and_round_trips() -> None:
    spec = experiment_spec_from_payload(_spec_payload())
    assert spec.name == "example_subset_bias"
    assert spec.seasons == (2009, 2025)
    assert spec.samples == 2000
    assert spec.seed == 20260818
    assert spec.block_primary == "week"
    assert spec.block_secondary == "season"
    assert experiment_spec_from_payload(experiment_spec_to_payload(spec)) == spec


def test_spec_rejects_unknown_top_level_fields() -> None:
    with pytest.raises(ExperimentSpecError, match="unknown top-level fields"):
        experiment_spec_from_payload(_spec_payload(bogus=1))


def test_spec_requires_seed_with_no_default() -> None:
    payload = _spec_payload()
    del payload["seed"]
    with pytest.raises(ExperimentSpecError, match="missing required field 'seed'"):
        experiment_spec_from_payload(payload)


def test_spec_rejects_non_integer_seed() -> None:
    with pytest.raises(ExperimentSpecError, match="seed must be an integer"):
        experiment_spec_from_payload(_spec_payload(seed=20260818.5))


def test_spec_rejects_unknown_experiment_type() -> None:
    with pytest.raises(ExperimentSpecError, match="Unknown experiment_type"):
        experiment_spec_from_payload(_spec_payload(experiment_type="mystery"))


def test_spec_rejects_unknown_league() -> None:
    payload = _spec_payload()
    payload["population"] = {**payload["population"], "league": "xfl"}  # type: ignore[dict-item]
    with pytest.raises(ExperimentSpecError, match="population\\.league"):
        experiment_spec_from_payload(payload)


def test_spec_rejects_seasons_out_of_order() -> None:
    payload = _spec_payload()
    payload["population"] = {**payload["population"], "seasons": [2020, 2010]}  # type: ignore[dict-item]
    with pytest.raises(ExperimentSpecError, match="out of order"):
        experiment_spec_from_payload(payload)


def test_spec_requires_flag_builder() -> None:
    payload = _spec_payload()
    payload["construct"] = {"params": {}}
    with pytest.raises(ExperimentSpecError, match="flag_builder is required"):
        experiment_spec_from_payload(payload)


def test_spec_endpoints_primary_must_be_accuracy() -> None:
    payload = _spec_payload()
    payload["endpoints"] = {"primary": "brier", "secondary": []}
    with pytest.raises(ExperimentSpecError, match="endpoints\\.primary must be 'accuracy'"):
        experiment_spec_from_payload(payload)


def test_spec_endpoints_secondary_rejects_unknown_metric() -> None:
    payload = _spec_payload()
    payload["endpoints"] = {"primary": "accuracy", "secondary": ["mae"]}
    with pytest.raises(ExperimentSpecError, match="unknown metric"):
        experiment_spec_from_payload(payload)


def test_spec_blocking_secondary_must_differ_from_primary() -> None:
    payload = _spec_payload()
    payload["blocking"] = {"primary": "week", "secondary": "week"}
    with pytest.raises(ExperimentSpecError, match="must differ from"):
        experiment_spec_from_payload(payload)


def test_spec_reliability_not_applicable_requires_a_reason() -> None:
    payload = _spec_payload()
    payload["reliability_check"] = {"method": "not_applicable", "reason": "   "}
    with pytest.raises(ExperimentSpecError, match="reason is required"):
        experiment_spec_from_payload(payload)


def test_spec_reliability_split_half_does_not_require_a_reason() -> None:
    payload = _spec_payload()
    payload["reliability_check"] = {"method": "split_half"}
    spec = experiment_spec_from_payload(payload)
    assert spec.reliability_method == "split_half"


def test_spec_rejects_too_few_bootstrap_samples() -> None:
    with pytest.raises(ExperimentSpecError, match="samples must be at least 10"):
        experiment_spec_from_payload(_spec_payload(samples=5))


def test_load_experiment_spec_rejects_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "spec.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ExperimentSpecError, match="not valid JSON"):
        load_experiment_spec(bad)


def test_load_experiment_spec_reads_a_valid_file(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_payload()), encoding="utf-8")
    spec = load_experiment_spec(spec_path)
    assert spec.name == "example_subset_bias"


# ---------------------------------------------------------------------------
# scale_subset_effect: the one 100x-fraction-vs-points scaling function
# ---------------------------------------------------------------------------


def test_scale_subset_effect_matches_hand_arithmetic() -> None:
    # 2 raw points of gap, firing on 25% of the slate -> 0.5 scaled points.
    assert scale_subset_effect(0.02, sign=1, fraction_of_slate=0.25) == pytest.approx(0.5)
    assert scale_subset_effect(0.02, sign=-1, fraction_of_slate=0.25) == pytest.approx(-0.5)


def test_scale_subset_effect_rejects_bad_sign_and_fraction() -> None:
    with pytest.raises(ValueError, match="sign must be"):
        scale_subset_effect(0.02, sign=0, fraction_of_slate=0.5)
    with pytest.raises(ValueError, match="fraction_of_slate"):
        scale_subset_effect(0.02, sign=1, fraction_of_slate=1.5)


# ---------------------------------------------------------------------------
# Mechanical classification: the runner's one auto-computed terminal verdict
# ---------------------------------------------------------------------------


def test_widening_factor_matches_the_registered_mod06_precedent() -> None:
    # registry/weak_signals.json mod06_js_shrinkage_position_prior_cfb: "re-crossing
    # zero requires only a 1.082x widening (centre -0.53, upper -0.04)". The exact
    # (unrounded) inputs give 1.089, which is what the module docstring cites.
    factor = widening_factor_to_recross_zero(-0.526, -0.043)
    assert factor == pytest.approx(1.089, abs=0.005)
    assert factor < HONEST_REFIT_WIDENING_UPPER_BOUND


def test_classification_stays_unresolved_inside_the_honest_band() -> None:
    # Same mod06-shaped inputs: interval is entirely negative, but the widening
    # needed to re-cross zero (~1.089x) sits INSIDE the honest 1.099x band, so
    # AGENTS.md says this must not become a terminal refutation.
    result = classify_subset_bias_result(estimate=-0.526, lower=-1.019, upper=-0.043)
    assert result.classification == "unresolved_below_power"
    assert result.closing_ground is None
    assert result.widening_factor == pytest.approx(1.089, abs=0.005)


def test_classification_refutes_only_past_the_honest_band() -> None:
    result = classify_subset_bias_result(estimate=-1.0, lower=-1.3, upper=-0.2)
    assert result.widening_factor is not None
    assert result.widening_factor > HONEST_REFIT_WIDENING_UPPER_BOUND
    assert result.classification == "refuted_mechanism"
    assert result.closing_ground == "wrong_sign_resolved"


def test_classification_never_closes_on_an_interval_crossing_zero() -> None:
    # AGENTS.md binding rule: crossing zero is never grounds for rejection.
    result = classify_subset_bias_result(estimate=0.4, lower=-0.6, upper=1.4)
    assert result.classification == "unresolved_below_power"
    assert result.closing_ground is None
    assert result.widening_factor is None


def test_classification_never_produces_bounded_by_control() -> None:
    # Sweep a range of shapes; the runner must never emit a closing ground
    # outside its one documented mechanical path.
    for estimate, lower, upper in (
        (-0.1, -5.0, 5.0),
        (-2.0, -3.0, -1.0),
        (2.0, -1.0, 5.0),
        (-0.01, -0.02, -0.001),
    ):
        result = classify_subset_bias_result(estimate=estimate, lower=lower, upper=upper)
        assert result.classification in ("unresolved_below_power", "refuted_mechanism")
        assert result.closing_ground in (None, "wrong_sign_resolved")


def test_widening_factor_requires_a_negative_upper_and_a_lower_estimate() -> None:
    with pytest.raises(ValueError, match="upper must be strictly below zero"):
        widening_factor_to_recross_zero(-1.0, 0.5)
    with pytest.raises(ValueError, match="estimate must be strictly below upper"):
        widening_factor_to_recross_zero(-0.01, -0.02)


# ---------------------------------------------------------------------------
# The shared team-game long table and the named flag builders
# ---------------------------------------------------------------------------


def _synthetic_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "season": [2020, 2020, 2020, 2020],
            "week": [1, 1, 2, 1],
            "home_team": ["NE", "OAK", "NE", "NE"],
            "away_team": ["BUF", "KC", "BUF", "BUF"],
            "home_cover": [1.0, 0.0, np.nan, 1.0],  # g3 is a push
            "spread_line": [-3.0, 4.0, -3.0, 6.5],
            "game_type": ["REG", "REG", "REG", "POST"],  # g4 is postseason
        }
    )


def test_base_team_game_table_drops_pushes_and_postseason_and_canonicalizes() -> None:
    table = _base_team_game_table(_synthetic_features())
    # g3 (push) and g4 (POST) must be gone; g1/g2 each contribute a home+away row.
    assert len(table) == 4
    assert set(table["game_id"]) == {"g1", "g2"}
    # OAK canonicalizes to LV (nfl_ats.constants.TEAM_ABBREVIATION_ALIASES).
    assert "OAK" not in set(table["team"])
    assert "LV" in set(table["team"])
    home_rows = table.loc[table["is_home"]]
    assert set(home_rows["team"]) == {"NE", "LV"}
    g1_home = table.loc[(table["game_id"] == "g1") & table["is_home"]].iloc[0]
    assert g1_home["team_covered"] == pytest.approx(1.0)
    assert g1_home["team_spread"] == pytest.approx(-3.0)
    g1_away = table.loc[(table["game_id"] == "g1") & ~table["is_home"]].iloc[0]
    assert g1_away["team_covered"] == pytest.approx(0.0)
    assert g1_away["team_spread"] == pytest.approx(3.0)
    assert (table["week_block"] == 202001).all()


def test_base_team_game_table_requires_needed_columns() -> None:
    with pytest.raises(ExperimentRunnerError, match="missing columns"):
        _base_team_game_table(pd.DataFrame({"game_id": ["g1"]}))


def test_flag_home_underdog_matches_hand_computation() -> None:
    features = _synthetic_features()
    construct = _flag_home_underdog(features, (2009, 2025), {}, REPO_ROOT)
    table = construct.table.reset_index(drop=True)
    flag = construct.flag.reset_index(drop=True)
    # Only g1's home row (is_home=True, spread_line=-3.0<0) is flagged.
    expected = (table["game_id"] == "g1") & table["is_home"]
    assert (flag == expected).all()
    assert construct.sign == 1
    assert construct.eligible is None
    assert construct.reliability is None


def test_flag_large_favorite_respects_threshold_param() -> None:
    features = _synthetic_features()
    construct = _flag_large_favorite(features, (2009, 2025), {"threshold": 2.0}, REPO_ROOT)
    table = construct.table.reset_index(drop=True)
    flag = construct.flag.reset_index(drop=True)
    expected = table["team_spread"] > 2.0
    assert (flag == expected).all()
    assert int(flag.sum()) == 2
    assert construct.sign == -1


def test_flag_builders_registry_has_the_documented_names() -> None:
    assert set(FLAG_BUILDERS) == {"penalty_rate_quartile", "home_underdog", "large_favorite"}
    for builder in FLAG_BUILDERS.values():
        assert builder.leagues == ("nfl",)


# ---------------------------------------------------------------------------
# The generic joint block bootstrap
# ---------------------------------------------------------------------------


def test_block_bootstrap_subset_gap_recovers_a_known_gap() -> None:
    # Two blocks; flag always covers, complement never does -> every resample
    # (however the blocks are drawn) must read exactly a 100-point gap.
    df = pd.DataFrame(
        {
            "week_block": [1, 1, 2, 2],
            "team_covered": [1.0, 0.0, 1.0, 0.0],
        }
    )
    flag = pd.Series([True, False, True, False])
    draws = _block_bootstrap_subset_gap(
        df, flag=flag, value_col="team_covered", block_col="week_block", samples=200, seed=1
    )
    assert len(draws) == 200
    assert np.allclose(draws, 100.0)


# ---------------------------------------------------------------------------
# run_subset_bias_experiment: validation-time errors
# ---------------------------------------------------------------------------


def test_run_subset_bias_experiment_rejects_unknown_builder(tmp_path: Path) -> None:
    spec = experiment_spec_from_payload(
        _spec_payload(construct={"flag_builder": "not_a_real_builder", "params": {}})
    )
    with pytest.raises(ExperimentRunnerError, match="Unknown construct\\.flag_builder"):
        run_subset_bias_experiment(spec, repo_root=tmp_path)


def test_run_subset_bias_experiment_rejects_opener_grade(tmp_path: Path) -> None:
    payload = _spec_payload()
    payload["population"] = {**payload["population"], "grade": "opener"}  # type: ignore[dict-item]
    spec = experiment_spec_from_payload(payload)
    with pytest.raises(ExperimentRunnerError, match="only supports population\\.grade='close'"):
        run_subset_bias_experiment(spec, repo_root=tmp_path)


def test_run_subset_bias_experiment_rejects_split_half_on_a_traitless_builder(
    tmp_path: Path,
) -> None:
    payload = _spec_payload(reliability_check={"method": "split_half"})
    spec = experiment_spec_from_payload(payload)
    with pytest.raises(ExperimentRunnerError, match="no persistent per-entity trait"):
        run_subset_bias_experiment(spec, repo_root=tmp_path)


def test_run_subset_bias_experiment_rejects_a_missing_feature_table(tmp_path: Path) -> None:
    spec = experiment_spec_from_payload(_spec_payload())
    with pytest.raises(ExperimentRunnerError, match="Feature table not found"):
        run_subset_bias_experiment(
            spec, repo_root=tmp_path, features_path=tmp_path / "absent.parquet"
        )


def test_run_experiment_feature_arm_is_a_clear_stub(tmp_path: Path) -> None:
    payload = _spec_payload(
        name="feature_arm_stub",
        experiment_type="feature_arm",
        construct={"flag_builder": "unused", "params": {}},
    )
    spec = experiment_spec_from_payload(payload)
    with pytest.raises(ExperimentRunnerError, match="ridge_alpha_promotion_eval"):
        run_experiment(spec, repo_root=tmp_path)


# ---------------------------------------------------------------------------
# A deterministic end-to-end subset_bias run on a small synthetic feature table
# ---------------------------------------------------------------------------


def _deterministic_features(n_weeks: int = 10) -> pd.DataFrame:
    """One home-underdog game per week that covers 80% of the time; the rest
    of the slate (three games/week, both non-favoured directions) covers
    exactly 50% overall.

    Built so ``effect``/``fraction_of_slate`` are hand-checkable exactly (they
    are deterministic point arithmetic, independent of the bootstrap), while
    the bootstrap-derived interval is only sanity-checked (matching this
    project's own testing convention -- see ``tests/test_experiments.py``).
    Recall ``_flag_home_underdog``'s convention (matching
    ``scripts/nfl_bias_battery_screen.py``): ``spread_line < 0`` on the home
    side is what makes a team the home underdog, not ``> 0``.
    """

    rows = []
    game = 0
    for week in range(1, n_weeks + 1):
        for pair in range(4):
            game += 1
            home_dog = pair == 0  # exactly one home-underdog game per week
            spread_line = -3.0 if home_dog else 3.0
            # home dogs cover 8/10 weeks; the other three games/week split
            # evenly between the two sides so the complement nets to 50%.
            if home_dog:
                home_cover = 1.0 if week % 5 != 0 else 0.0
            else:
                home_cover = 1.0 if pair % 2 == 0 else 0.0
            rows.append(
                {
                    "game_id": f"g{game}",
                    "season": 2020,
                    "week": week,
                    "home_team": f"T{pair}A",
                    "away_team": f"T{pair}B",
                    "home_cover": home_cover,
                    "spread_line": spread_line,
                    "game_type": "REG",
                }
            )
    return pd.DataFrame(rows)


def test_run_subset_bias_experiment_end_to_end_on_synthetic_data(tmp_path: Path) -> None:
    features = _deterministic_features()
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path)

    spec = experiment_spec_from_payload(_spec_payload(samples=2000))
    result = run_subset_bias_experiment(spec, repo_root=tmp_path, features_path=features_path)

    # Deterministic pieces: hand-computable exactly.
    home_dog_rows = features.loc[features["spread_line"] < 0]
    expected_subset_cover = float(home_dog_rows["home_cover"].mean())
    assert expected_subset_cover == pytest.approx(0.8)
    assert result.n_total == 80  # 10 weeks * 4 games * 2 sides
    assert result.n_flag == 10  # one flagged (home) row per week
    assert result.fraction_of_slate == pytest.approx(result.n_flag / result.n_total)
    assert result.effect == pytest.approx(result.raw_gap_pct * result.fraction_of_slate, abs=1e-9)
    # The flagged side genuinely covers more, and the sign convention (+1 for
    # home_underdog) must carry that through to raw_gap_pct unflipped.
    assert result.raw_gap_pct > 0.0
    assert result.sign == 1

    # Bootstrap-derived pieces: sanity only.
    assert 0.0 <= result.primary.probability_positive <= 1.0
    assert result.primary.lower <= result.primary.estimate <= result.primary.upper
    assert result.secondary is not None
    assert result.classification.classification in ("unresolved_below_power", "refuted_mechanism")


# ---------------------------------------------------------------------------
# Validation anchor: bit-for-bit reproduction of the penalty_discipline entry
# ---------------------------------------------------------------------------

_PBP_ROOT = REPO_ROOT / "data" / "pbp" / "raw"
_FEATURES_PATH = REPO_ROOT / "data" / "processed" / "game_features.parquet"
_LOCAL_DATA_AVAILABLE = _PBP_ROOT.is_dir() and _FEATURES_PATH.is_file()


@pytest.mark.skipif(
    not _LOCAL_DATA_AVAILABLE, reason="local PBP/feature data not present in this checkout"
)
def test_penalty_discipline_reproduces_the_recorded_registry_entry() -> None:
    """Full fidelity (samples=20000, seed=20260818) against
    ``registry/weak_signals.json``'s ``penalty_discipline`` entry (effect
    +0.3288, week-blocked interval [-1.0389, +1.6849], P+ 0.6828, reliability
    0.261, sample_games 4085, sample_blocks 277).

    This reproduction is measured, not merely toleranced: the runner's
    ``penalty_rate_quartile`` builder and generic bootstrap are a faithful
    port of ``scripts/penalty_discipline_interval.py``'s own construct and
    joint block bootstrap (same seed, same block-id derivation order, same
    single ``rng.multinomial`` call shape), so re-running that script
    (``.tools/uv.exe run --no-sync python scripts/penalty_discipline_interval.py``,
    ~3 seconds) reproduces these exact floats to more digits than are
    asserted here. Full samples, not reduced -- this runs in a few seconds,
    so there was no need to trade fidelity for test runtime.
    """

    spec = experiment_spec_from_payload(
        {
            "name": "penalty_discipline_reproduction",
            "hypothesis": (
                "Teams with the lowest lagged penalty rate cover more against the "
                "spread than teams with the highest."
            ),
            "experiment_type": "subset_bias",
            "population": {"league": "nfl", "seasons": [2009, 2025], "grade": "close"},
            "construct": {"flag_builder": "penalty_rate_quartile", "params": {}},
            "endpoints": {"primary": "accuracy", "secondary": []},
            "blocking": {"primary": "week", "secondary": "season"},
            "samples": 20000,
            "seed": 20260818,
            "reliability_check": {
                "method": "split_half",
                "reason": "penalty_rate_quartile carries a persistent team-season trait",
            },
        }
    )
    result = run_subset_bias_experiment(spec, repo_root=REPO_ROOT)

    assert result.effect == pytest.approx(0.3288, abs=5e-4)
    assert result.n_flag + result.n_complement == 4085
    assert result.primary.block_count == 277
    assert result.primary.lower == pytest.approx(-1.0389, abs=1e-3)
    assert result.primary.upper == pytest.approx(1.6849, abs=1e-3)
    assert result.primary.probability_positive == pytest.approx(0.6828, abs=1e-3)
    assert result.primary.standard_error == pytest.approx(0.6938, abs=1e-3)
    assert result.reliability == pytest.approx(0.261, abs=2e-3)
    assert result.reliability_pairs == 512
    # This entry is recorded unresolved_below_power in the registry: the
    # interval crosses zero, so the mechanical classifier must agree.
    assert result.classification.classification == "unresolved_below_power"
    assert result.classification.closing_ground is None


# ---------------------------------------------------------------------------
# The registry lock
# ---------------------------------------------------------------------------


def test_registry_lock_is_exclusive_and_cleans_up(tmp_path: Path) -> None:
    registry_path = tmp_path / "weak_signals.json"
    with (
        _RegistryLock(registry_path, timeout=0.2, poll=0.02),
        pytest.raises(ExperimentRunnerError, match="Could not acquire"),
        _RegistryLock(registry_path, timeout=0.2, poll=0.02),
    ):
        pass
    # Lock file must be removed once the holder exits.
    assert not registry_path.with_suffix(".json.lock").exists()


# ---------------------------------------------------------------------------
# run_experiment_cli: dry-run, real-run, single-writer, replace
# ---------------------------------------------------------------------------


def test_run_experiment_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    features = _deterministic_features()
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_payload(samples=500)), encoding="utf-8")
    registry_path = tmp_path / "weak_signals.json"
    artifacts_root = tmp_path / "artifacts"

    outcome = run_experiment_cli(
        spec_path,
        repo_root=tmp_path,
        dry_run=True,
        features_path=features_path,
        registry_path=registry_path,
        artifacts_root=artifacts_root,
    )
    assert outcome.dry_run is True
    assert outcome.artifact_directory is None
    assert outcome.registry_record is None
    assert outcome.preview["classification"] in ("unresolved_below_power", "refuted_mechanism")
    assert not registry_path.exists()
    assert not artifacts_root.exists()


def test_run_experiment_cli_writes_artifact_and_registry_then_enforces_single_write(
    tmp_path: Path,
) -> None:
    features = _deterministic_features()
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_payload(samples=500)), encoding="utf-8")
    registry_path = tmp_path / "weak_signals.json"
    artifacts_root = tmp_path / "artifacts"

    first = run_experiment_cli(
        spec_path,
        repo_root=tmp_path,
        dry_run=False,
        features_path=features_path,
        registry_path=registry_path,
        artifacts_root=artifacts_root,
        run_id_value="20260818T000000Z",
    )
    assert first.artifact_directory is not None
    artifact_dir = Path(first.artifact_directory)
    assert (artifact_dir / "metadata.json").is_file()
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["name"] == "example_subset_bias"
    assert "provenance" in metadata

    registry = load_registry(registry_path)
    assert "example_subset_bias" in registry.signals

    # A second, non-replacing run must refuse to silently overwrite.
    with pytest.raises(WeakSignalError, match="already recorded"):
        run_experiment_cli(
            spec_path,
            repo_root=tmp_path,
            dry_run=False,
            features_path=features_path,
            registry_path=registry_path,
            artifacts_root=artifacts_root,
            run_id_value="20260818T000001Z",
        )

    second = run_experiment_cli(
        spec_path,
        repo_root=tmp_path,
        dry_run=False,
        replace=True,
        features_path=features_path,
        registry_path=registry_path,
        artifacts_root=artifacts_root,
        run_id_value="20260818T000002Z",
    )
    assert second.registry_record is not None
    assert second.registry_record["recorded"] == "example_subset_bias"
