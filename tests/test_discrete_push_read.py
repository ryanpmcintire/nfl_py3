"""The served discrete push read (docs/discrete_push_read.md, MOD-18 lane S).

Pins the contract: the mass-preserving lattice is the source of the served
push and every alternative-line three-way split, the atoms keep their
absolute key-number mass, every read is walk-forward, and the pick-deciding
``home_cover_probability`` is bit-for-bit what the smooth read produced --
switched on or off.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats import prediction_safety
from nfl_ats.card_explanation import explain_pick
from nfl_ats.cli_commands import prediction as prediction_cli
from nfl_ats.margin import _three_way_probabilities
from nfl_ats.mass_preserving_lattice import (
    BAND_HALF_WIDTH,
    DISCRETE_PUSH_READ_POLICY,
    MIN_BAND_GAMES,
    THETA_BRACKET,
    THREE_WAY_COLUMNS,
    DiscretePushReader,
    ProductionDiscretePushRead,
    ServedPushRead,
    band_read,
    discrete_read,
    discrete_reads_for_frame,
    fit_production_discrete_push_reader,
    key_number_mass,
    prior_pool,
    prior_pool_for_week,
    residual_location,
    tilted_atoms,
    walk_forward_reads,
)
from nfl_ats.outcomes import score_outcome_week, score_outcome_week_line_sweep
from nfl_ats.prediction_safety import validate_three_way_split
from nfl_ats.spread_explorer import SpreadExplorerGameDistribution, spread_explorer_three_way

REPO = Path(__file__).resolve().parents[1]


def key_number_atoms() -> tuple[np.ndarray, np.ndarray]:
    """A base distribution with real key-number spikes at 3, 7, 10 and 14."""

    margins = np.arange(-21.0, 22.0)
    counts = np.ones_like(margins)
    for key, weight in ((3, 12.0), (7, 7.0), (10, 4.0), (14, 3.0)):
        counts[margins == key] += weight
        counts[margins == -key] += weight
    return margins, counts


def synthetic_pool(push_share: float = 0.10, games: int = 600, seed: int = 3) -> pd.DataFrame:
    """Prior games quoted near 3 whose finals land exactly on 3 a declared share
    of the time -- the push mass the served read must reproduce."""

    rng = np.random.default_rng(seed)
    lines = rng.choice([2.5, 3.0, 3.5], size=games)
    margins = np.rint(rng.normal(3.0, 13.0, size=games))
    on_three = rng.random(games) < push_share
    margins[on_three] = 3.0
    margins[~on_three & (margins == 3.0)] = 4.0
    rows = []
    for index in range(games):
        season = 2015 + index % 5
        rows.append(
            {
                "game_id": f"{season}-{index}",
                "season": season,
                "week": 1 + index % 17,
                "gameday": pd.Timestamp(f"{season}-09-10") + pd.Timedelta(days=index % 120),
                "spread_line": float(lines[index]),
                "line": float(lines[index]),
                "result": float(margins[index]),
            }
        )
    return pd.DataFrame(rows)


def reader_for_2020_week_1(pool: pd.DataFrame) -> DiscretePushReader:
    return DiscretePushReader.for_week(
        pool, season=2020, week=1, cutoff=pd.Timestamp("2020-09-10"), exclude_game_ids=()
    )


def integer_line_week(model_frame: pd.DataFrame) -> pd.DataFrame:
    """The shared fixture with 2020 week 1 quoted on 3 / 3.5 alternately.

    Whole numbers on purpose: these are the lines the key-number machinery
    exists FOR, and the archive it was measured on is quoted on them. The
    owner's pool never posts one, so any orchestration test built on this
    fixture pairs it with :func:`allow_whole_number_pool_lines`.
    """

    frame = model_frame.copy()
    target = frame["season"].eq(2020) & frame["week"].eq(1)
    positions = np.flatnonzero(target.to_numpy())
    frame.loc[target, "spread_line"] = [3.0 if i % 2 == 0 else 3.5 for i in range(len(positions))]
    return frame


def allow_whole_number_pool_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declare that this test's week is NOT the owner's pool.

    ``prediction_safety.validate_pool_lines`` fails the served card closed
    when a decision line is a whole number, because the pool posts only half
    points and a whole number therefore proves the line came from the
    schedule feed instead. A fixture that deliberately quotes 3.0 to
    exercise the key-number read has to say so out loud; nothing in
    production may take this path.
    """

    monkeypatch.setattr(prediction_safety, "POOL_QUOTES_HALF_POINT_LINES", False)


def test_atom_mass_at_the_key_numbers_is_kept_and_stays_local_maximal() -> None:
    margins, counts = key_number_atoms()
    base = counts / counts.sum()
    total = counts.sum()
    assert key_number_mass(margins, base) == pytest.approx(
        {3: 26 / total, 7: 16 / total, 10: 10 / total, 14: 8 / total}
    )
    mass, theta = tilted_atoms(margins, counts, line=3.0, target=6.0)
    assert theta > 0.0
    assert mass.shape == margins.shape and (mass > 0).all()
    for key in (3.0, 7.0, 10.0, 14.0):
        index = int(np.flatnonzero(margins == key)[0])
        assert mass[index] > mass[index - 1] and mass[index] > mass[index + 1]
    assert float((margins * mass).sum()) == pytest.approx(6.0)
    assert float(mass.sum()) == pytest.approx(1.0)


def test_tilt_solver_hits_the_target_and_clamps_outside_the_bracket() -> None:
    margins, counts = key_number_atoms()
    base = counts / counts.sum()
    base_mean = float((margins * base).sum())
    mass, theta = tilted_atoms(margins, counts, line=0.0, target=base_mean)
    assert theta == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(mass, base, atol=1e-12)
    lower, _ = tilted_atoms(margins, counts, line=0.0, target=-4.0)
    upper, _ = tilted_atoms(margins, counts, line=0.0, target=+4.0)
    assert upper[margins > 0].sum() > lower[margins > 0].sum()
    _, high = tilted_atoms(margins, counts, line=0.0, target=1e6)
    _, low = tilted_atoms(margins, counts, line=0.0, target=-1e6)
    assert (high, low) == (THETA_BRACKET, -THETA_BRACKET)
    single, theta = tilted_atoms(np.array([3.0]), np.array([5.0]), line=3.0, target=9.0)
    assert theta == 0.0 and single.tolist() == [1.0]


def test_band_read_pushes_only_on_an_integer_line_and_sums_to_one() -> None:
    lines = np.repeat(np.arange(-10.0, 11.0), 40)
    margins = np.tile(key_number_atoms()[0][:40], 21)
    integer = band_read(lines, margins, line=3.0, point=3.0)
    half = band_read(lines, margins, line=3.5, point=3.5)
    assert integer.push > 0.0 and half.push == 0.0
    for read in (integer, half):
        assert read.cover + read.push + read.loss == pytest.approx(1.0)
        assert read.home_cover_probability == pytest.approx(read.cover + 0.5 * read.push)
        assert read.three_way() == (read.cover, read.push, read.loss)
    assert integer.band == BAND_HALF_WIDTH
    sparse_lines = np.concatenate([np.full(30, 12.0), np.full(400, 0.0)])
    sparse_margins = np.concatenate([np.full(30, 14.0), np.zeros(400)])
    widened = band_read(sparse_lines, sparse_margins, line=12.0, point=12.0)
    assert widened.band > BAND_HALF_WIDTH and widened.band_games >= MIN_BAND_GAMES


def test_residual_location_follows_the_probability_method() -> None:
    residuals = np.array([-3.0, -1.0, 0.0, 10.0])
    assert residual_location(residuals, "gaussian_median") == pytest.approx(-0.5)
    assert residual_location(residuals, "gaussian") == pytest.approx(1.5)
    with pytest.raises(ValueError, match="residual sample"):
        residual_location(np.array([]), "gaussian_median")


def test_no_row_uses_its_own_week_a_later_week_or_a_sixth_season_back() -> None:
    pool = synthetic_pool()
    targets = pd.DataFrame(
        {
            "game_id": ["t1", "t2"],
            "season": [2020, 2020],
            "week": [1, 1],
            "gameday": [pd.Timestamp("2020-09-10")] * 2,
            "line": [3.0, -3.0],
            "point": [3.0, -3.0],
        }
    )
    expected = walk_forward_reads(pool, targets, BAND_HALF_WIDTH)
    poison = pd.DataFrame(
        {
            "game_id": ["same", "later", "ancient", "t1"],
            "season": [2020, 2020, 2014, 2019],
            "week": [1, 2, 1, 17],
            "gameday": [
                pd.Timestamp("2020-09-10"),
                pd.Timestamp("2020-09-20"),
                pd.Timestamp("2014-09-10"),
                pd.Timestamp("2019-12-29"),
            ],
            "spread_line": [3.0, 3.0, 3.0, 3.0],
            "line": [3.0, 3.0, 3.0, 3.0],
            "result": [40.0, 40.0, 40.0, 40.0],
        }
    )
    poisoned = walk_forward_reads(pd.concat([pool, poison], ignore_index=True), targets, 2.5)
    pd.testing.assert_frame_equal(expected, poisoned)
    eligible = prior_pool_for_week(
        pd.concat([pool, poison], ignore_index=True),
        season=2020,
        week=1,
        cutoff=pd.Timestamp("2020-09-10"),
        exclude_game_ids={"t1"},
    )
    assert not set(eligible["game_id"]).intersection({"same", "later", "ancient", "t1"})
    earlier = poison.iloc[[0]].assign(game_id="earlier", season=2020, week=0)
    earlier["gameday"] = pd.Timestamp("2020-09-01")
    moved = walk_forward_reads(pd.concat([pool, earlier], ignore_index=True), targets, 2.5)
    assert moved["prior_rows"].iloc[0] == expected["prior_rows"].iloc[0] + 1
    assert moved["atoms"].iloc[0] == expected["atoms"].iloc[0] + 1


def test_prior_pool_prefers_archived_opener_lines_and_drops_unplayed_rows() -> None:
    features = pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "post"],
            "season": [2020] * 4,
            "week": [1, 1, 1, 20],
            "gameday": [pd.Timestamp("2020-09-10")] * 4,
            "game_type": ["REG", "REG", "REG", "WC"],
            "spread_line": [1.0, 2.0, 3.0, 4.0],
            "result": [7.0, -3.0, np.nan, 5.0],
        }
    )
    pool = prior_pool(features, {"a": 4.5})
    assert list(pool["game_id"]) == ["a", "b"]
    assert list(pool["line"]) == [4.5, 2.0]


def test_one_pure_per_game_function_backs_every_served_answer() -> None:
    """The coordinator's key-line pick override, the card's push path and the
    sweep must all read the SAME lattice: ``discrete_read`` (walk-forward
    window applied inside), ``DiscretePushReader.read`` and ``band_read`` on
    the same prior rows give one identical read."""

    pool = synthetic_pool()
    cutoff = pd.Timestamp("2020-09-10")
    pure = discrete_read(pool, 3.0, 3.4, season=2020, week=1, cutoff=cutoff, game_id="t1")
    reader = DiscretePushReader.for_week(
        pool, season=2020, week=1, cutoff=cutoff, exclude_game_ids={"t1"}
    )
    eligible = prior_pool_for_week(
        pool, season=2020, week=1, cutoff=cutoff, exclude_game_ids={"t1"}
    )
    core = band_read(
        eligible["line"].to_numpy(dtype=float), eligible["result"].to_numpy(dtype=float), 3.0, 3.4
    )
    assert pure == reader.read(3.0, 3.4) == core
    assert pure.cover + pure.push + pure.loss == pytest.approx(1.0)
    assert 0.07 < pure.push < 0.13 and pure.atoms > 10
    poison = pd.DataFrame(
        {
            "game_id": ["same", "t1"],
            "season": [2020, 2019],
            "week": [1, 17],
            "gameday": [pd.Timestamp("2020-09-10"), pd.Timestamp("2019-12-29")],
            "spread_line": [3.0, 3.0],
            "line": [3.0, 3.0],
            "result": [40.0, 40.0],
        }
    )
    poisoned = discrete_read(
        pd.concat([pool, poison], ignore_index=True),
        3.0,
        3.4,
        season=2020,
        week=1,
        cutoff=cutoff,
        game_id="t1",
    )
    assert poisoned == pure
    wide = discrete_read(pool, 3.0, 3.4, season=2020, week=1, cutoff=cutoff, band=5.0, prior=50)
    assert wide.band == 5.0 and wide.band_games >= 50
    with pytest.raises(ValueError, match="No prior games"):
        discrete_read(pool, 3.0, 3.4, season=2010, week=1, cutoff=pd.Timestamp("2010-09-01"))


def test_frame_helper_aligns_to_the_index_and_matches_the_pure_function() -> None:
    pool = synthetic_pool()
    frame = pd.DataFrame(
        {
            "game_id": ["a", "b", "c"],
            "season": [2020, 2020, 2020],
            "week": [1, 1, 2],
            "gameday": [
                pd.Timestamp("2020-09-10"),
                pd.Timestamp("2020-09-13"),
                pd.Timestamp("2020-09-20"),
            ],
            "spread_line": [3.0, -3.5, 7.0],
            "point": [3.4, -2.0, 6.1],
        },
        index=[10, 20, 30],
    )
    reads = discrete_reads_for_frame(pool, frame)
    assert list(reads.index) == [10, 20, 30]
    for index, row in frame.iterrows():
        cutoff = pd.Timestamp(frame.loc[frame["week"].eq(row["week"]), "gameday"].min())
        expected = discrete_read(
            pool,
            float(row["spread_line"]),
            float(row["point"]),
            season=int(row["season"]),
            week=int(row["week"]),
            cutoff=cutoff,
            game_id=str(row["game_id"]),
        )
        assert reads.loc[index, "push"] == expected.push
        assert reads.loc[index, "cover"] == expected.cover
        assert reads.loc[index, "atoms"] == expected.atoms
    assert reads.loc[20, "push"] == 0.0 and reads.loc[10, "push"] > 0.0
    assert reads["prior_rows"].dtype.kind == "i"


def test_served_push_at_three_reads_the_lattice_and_never_moves_the_pick(
    model_frame: pd.DataFrame,
) -> None:
    features = integer_line_week(model_frame)
    reader = reader_for_2020_week_1(synthetic_pool(push_share=0.10))
    baseline = score_outcome_week(features, season=2020, week=1, min_train_games=80)
    log: dict[str, ServedPushRead] = {}
    served = score_outcome_week(
        features,
        season=2020,
        week=1,
        min_train_games=80,
        discrete_read=reader,
        discrete_read_log=log,
    )
    untouched = [column for column in baseline.columns if column not in THREE_WAY_COLUMNS]
    pd.testing.assert_frame_equal(baseline[untouched], served[untouched])
    assert (
        baseline["home_cover_probability"].ge(0.5) == served["home_cover_probability"].ge(0.5)
    ).all()
    others = baseline["method"].ne("market_residual")
    pd.testing.assert_frame_equal(
        baseline.loc[others, list(THREE_WAY_COLUMNS)], served.loc[others, list(THREE_WAY_COLUMNS)]
    )
    ats = served.loc[served["method"].eq("market_residual")]
    on_three = ats.loc[ats["spread_line"].eq(3.0)]
    on_hook = ats.loc[ats["spread_line"].eq(3.5)]
    assert not on_three.empty and not on_hook.empty
    assert on_three["push_probability"].between(0.07, 0.13).all()
    assert on_hook["push_probability"].eq(0.0).all()
    validate_three_way_split(ats)
    smooth_three = baseline.loc[
        baseline["method"].eq("market_residual") & baseline["spread_line"].eq(3.0),
        "push_probability",
    ]
    assert (on_three["push_probability"].to_numpy() > smooth_three.to_numpy()).all()
    assert set(log) == set(ats["game_id"].astype(str))
    for game_id, record in log.items():
        row = baseline.loc[
            baseline["method"].eq("market_residual") & baseline["game_id"].astype(str).eq(game_id)
        ].iloc[0]
        assert record.smooth == (
            row["home_cover_probability_excluding_push"],
            row["push_probability"],
            row["home_loss_probability"],
        )
        assert record.home_cover_probability == row["home_cover_probability"]
        assert (
            record.discrete.push
            == ats.set_index(ats["game_id"].astype(str)).loc[game_id, "push_probability"]
        )


def test_line_sweep_reads_push_off_the_lattice_at_every_alternative_line(
    model_frame: pd.DataFrame,
) -> None:
    features = integer_line_week(model_frame)
    reader = reader_for_2020_week_1(synthetic_pool())
    baseline = score_outcome_week_line_sweep(features, season=2020, week=1, min_train_games=80)
    served = score_outcome_week_line_sweep(
        features, season=2020, week=1, min_train_games=80, discrete_read=reader
    )
    two_way = ["home_cover_probability", "pick_probability", "confidence"]
    pd.testing.assert_frame_equal(
        baseline.drop(columns=list(THREE_WAY_COLUMNS)), served.drop(columns=list(THREE_WAY_COLUMNS))
    )
    ats = served.loc[served["method"].eq("market_residual")]
    total = ats[list(THREE_WAY_COLUMNS)].sum(axis=1)
    assert np.allclose(total, 1.0)
    integer_lines = np.isclose(ats["alternative_line"] % 1.0, 0.0)
    assert (ats.loc[integer_lines, "push_probability"] > 0.0).all()
    assert (ats.loc[~integer_lines, "push_probability"] == 0.0).all()
    assert (baseline[two_way].to_numpy() == served[two_way].to_numpy()).all()
    others = served["method"].ne("market_residual")
    pd.testing.assert_frame_equal(baseline.loc[others], served.loc[others])


def test_flag_off_serves_no_reader_and_a_fit_failure_degrades_with_the_error(
    monkeypatch: pytest.MonkeyPatch, model_frame: pd.DataFrame
) -> None:
    request = prediction_cli.MarginPredictRequest(
        features=Path("unused.parquet"),
        season=2020,
        week=1,
        regressor="ridge",
        min_edge=0.02,
        min_train_games=80,
        feature_profile="base",
        ridge_alpha=10.0,
        probability_method="gaussian_median",
        line_sweep=False,
    )
    monkeypatch.setattr(prediction_cli, "DISCRETE_PUSH_READ_SERVED", False)
    assert prediction_cli._served_discrete_push_read(model_frame, request) is None

    monkeypatch.setattr(prediction_cli, "DISCRETE_PUSH_READ_SERVED", True)

    def explode(*args: object, **kwargs: object) -> ProductionDiscretePushRead:
        raise ValueError("no prior games")

    monkeypatch.setattr(prediction_cli, "fit_production_discrete_push_reader", explode)
    degraded = prediction_cli._served_discrete_push_read(model_frame, request)
    assert degraded is not None and not degraded.served
    assert degraded.error == "no prior games"
    baseline = score_outcome_week(model_frame, season=2020, week=1, min_train_games=80)
    again = score_outcome_week(
        model_frame, season=2020, week=1, min_train_games=80, discrete_read=degraded.reader
    )
    pd.testing.assert_frame_equal(baseline, again)


def test_sidecar_and_metadata_carry_both_reads(model_frame: pd.DataFrame) -> None:
    features = integer_line_week(model_frame)
    reader = reader_for_2020_week_1(synthetic_pool())
    production = ProductionDiscretePushRead(
        policy=DISCRETE_PUSH_READ_POLICY,
        reader=reader,
        source_path=None,
        source_model_id=None,
        active_model_id="m",
        opener_lines_matched=0,
        warnings=("no opener evaluation archive found; feature-table lines used",),
    )
    log: dict[str, ServedPushRead] = {}
    predictions = score_outcome_week(
        features,
        season=2020,
        week=1,
        min_train_games=80,
        discrete_read=reader,
        discrete_read_log=log,
    )
    sidecar = prediction_cli._discrete_push_read_sidecar(production, log, predictions)
    json.dumps(sidecar)
    assert sidecar["served"] is True and sidecar["policy"] == DISCRETE_PUSH_READ_POLICY
    assert sidecar["challenger"] == "smooth_rounded_residuals"
    ats = predictions.loc[predictions["method"].eq("market_residual")]
    assert [g["game_id"] for g in sidecar["games"]] == sorted(ats["game_id"].astype(str))
    for game in sidecar["games"]:
        assert game["served"]["cover"] + game["served"]["push"] + game["served"][
            "loss"
        ] == pytest.approx(1.0)
        assert game["smooth"]["cover"] + game["smooth"]["push"] + game["smooth"][
            "loss"
        ] == pytest.approx(1.0)
        assert {"theta", "band", "band_games", "atoms", "key_mass_3", "point"} <= set(game)
    summary = prediction_cli._discrete_push_read_summary(production, log)
    assert summary["served"] is True and summary["games"] == len(ats)
    assert summary["whole_number_lines"] == int(ats["spread_line"].eq(3.0).sum())
    assert 0.07 < summary["mean_push_at_whole_number_lines"] < 0.13


def test_production_reader_prefers_archived_opener_lines_and_degrades_without_them(
    tmp_path: Path,
) -> None:
    pool = synthetic_pool()
    features = pool.drop(columns="line").assign(game_type="REG")
    target = pd.DataFrame(
        {
            "game_id": ["2020_01_A_B"],
            "season": [2020],
            "week": [1],
            "gameday": [pd.Timestamp("2020-09-10")],
            "spread_line": [3.0],
            "result": [np.nan],
            "game_type": ["REG"],
        }
    )
    features = pd.concat([features, target], ignore_index=True)
    bare = fit_production_discrete_push_reader(features, tmp_path, None, season=2020, week=1)
    assert bare.served and bare.opener_lines_matched == 0
    assert any("no opener evaluation archive" in warning for warning in bare.warnings)
    assert bare.reader is not None
    assert bare.reader.prior_rows == len(pool)

    evaluation = tmp_path / "opener_evaluation" / "20260908T000000Z"
    evaluation.mkdir(parents=True)
    archived = pool.head(50)[["game_id"]].assign(tue_open_home_spread=9.0)
    archived.to_parquet(evaluation / "per_game.parquet", index=False)
    (evaluation / "metadata.json").write_text(json.dumps({"active_model_id": "old"}))
    with_archive = fit_production_discrete_push_reader(
        features, tmp_path, {"model_id": "new"}, season=2020, week=1
    )
    assert with_archive.opener_lines_matched == 50
    assert with_archive.source_model_id == "old"
    assert with_archive.source_path == "opener_evaluation/20260908T000000Z"
    assert with_archive.reader is not None
    assert int(np.count_nonzero(with_archive.reader.lines == 9.0)) == 50
    assert with_archive.to_dict()["served"] is True


def test_margin_predict_writes_the_sidecar_with_both_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_frame: pd.DataFrame
) -> None:
    """End to end: the real ``margin-predict`` orchestration serves the
    discrete split on the card, keeps the pick, and writes the sidecar."""

    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(artifacts_root))
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    features_path = data_root / "processed" / "game_features.parquet"
    features_path.parent.mkdir(parents=True)
    integer_line_week(model_frame).to_parquet(features_path, index=False)
    allow_whole_number_pool_lines(monkeypatch)
    request = prediction_cli.MarginPredictRequest(
        features=features_path,
        season=2020,
        week=1,
        regressor="ridge",
        min_edge=0.02,
        min_train_games=80,
        feature_profile="base",
        ridge_alpha=10.0,
        probability_method="gaussian_median",
        line_sweep=True,
    )
    result = prediction_cli.orchestrate_margin_predict(request)
    sidecar_path = result.output / "discrete_push_read.json"
    assert sidecar_path.is_file()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["served"] is True and sidecar["error"] is None
    summary = result.metadata["discrete_push_read"]
    assert summary["served"] is True and summary["games"] == len(sidecar["games"])
    assert result.metadata["line_sweep"]["push_read"] == DISCRETE_PUSH_READ_POLICY
    card = pd.read_csv(result.output / "recommendations.csv")
    by_game = {game["game_id"]: game for game in sidecar["games"]}
    features = pd.read_parquet(features_path)
    pool = prior_pool(features)
    cutoff = pd.Timestamp(
        pd.to_datetime(
            features.loc[features["season"].eq(2020) & features["week"].eq(1), "gameday"]
        ).min()
    )
    for _, row in card.iterrows():
        game = by_game[str(row["game_id"])]
        assert row["push_probability"] == pytest.approx(game["served"]["push"])
        assert row["home_cover_probability"] == pytest.approx(game["home_cover_probability"])
        expected = discrete_read(
            pool,
            float(row["spread_line"]),
            float(game["point"]),
            season=2020,
            week=1,
            cutoff=cutoff,
            game_id=str(row["game_id"]),
        )
        assert game["served"] == {
            "cover": expected.cover,
            "push": expected.push,
            "loss": expected.loss,
        }
        assert game["atoms"] == expected.atoms and game["theta"] == expected.theta
        if not float(row["spread_line"]).is_integer():
            assert game["served"]["push"] == 0.0 == game["smooth"]["push"]
    sweep = pd.read_parquet(result.output / "line_sweep.parquet")
    ats = sweep.loc[sweep["method"].eq("market_residual")]
    assert np.allclose(ats[list(THREE_WAY_COLUMNS)].sum(axis=1), 1.0)
    assert (ats.loc[~np.isclose(ats["alternative_line"] % 1.0, 0.0), "push_probability"] == 0).all()


def test_spread_explorer_three_way_uses_the_reader_when_given() -> None:
    rng = np.random.default_rng(20260908)
    distribution = SpreadExplorerGameDistribution(
        game_id="g",
        season=2020,
        week=1,
        home_team="HME",
        away_team="AWY",
        center=2.7,
        residuals=rng.normal(0.0, 13.0, 2000),
        card_line=3.0,
        card_home_cover_probability=0.49,
        card_probability_method="gaussian_median",
    )
    smooth = spread_explorer_three_way(distribution, 3.0)
    assert smooth == _three_way_probabilities(distribution.center + distribution.residuals, 3.0)
    reader = reader_for_2020_week_1(synthetic_pool(push_share=0.10))
    served = spread_explorer_three_way(distribution, 3.0, discrete_read=reader)
    assert sum(served) == pytest.approx(1.0)
    assert 0.07 < served[1] < 0.13
    expected = reader.three_way(
        3.0, distribution.center + residual_location(distribution.residuals, "gaussian_median")
    )
    assert served == expected
    assert spread_explorer_three_way(distribution, 3.5, discrete_read=reader)[1] == 0.0


def test_explanation_names_the_push_chance_in_pool_player_words() -> None:
    base = {
        "game_id": "2026_01_NYJ_TEN",
        "home_team": "TEN",
        "away_team": "NYJ",
        "home_cover_probability": 0.49,
        "gameday": "2026-09-13",
    }
    on_key = explain_pick({**base, "spread_line": 3.0, "push_probability": 0.091}).text
    assert "right on 3, a number games land on a lot" in on_key
    assert "about 9 in 100 games like this finish exactly there, a push" in on_key
    whole = explain_pick({**base, "spread_line": 5.0, "push_probability": 0.042}).text
    assert "on a whole number: about 4 in 100" in whole
    hook = explain_pick({**base, "spread_line": 3.5, "push_probability": 0.0}).text
    assert "push" not in hook
    absent = explain_pick({**base, "spread_line": 3.0}).text
    assert "push" not in absent
    tiny = explain_pick({**base, "spread_line": 1.0, "push_probability": 0.002}).text
    assert "push" not in tiny


def _lane_k_root() -> Path:
    override = os.environ.get("NFL_ATS_ARTIFACTS_DIR")
    root = Path(override) if override else REPO / "artifacts"
    return root / "research" / "laneK"


@pytest.mark.skipif(
    not (_lane_k_root() / "replay.parquet").is_file()
    or not (_lane_k_root() / "pool.parquet").is_file(),
    reason="lane K's research artifacts are local-only and absent here",
)
def test_lane_k_research_numbers_replay_bit_for_bit_through_the_module() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    from mass_preserving_lattice_opener_eval import ARMS, mass_preserving

    root = _lane_k_root()
    pool = pd.read_parquet(root / "pool.parquet")
    frame = pd.read_parquet(root / "replay.parquet")
    targets = frame[["game_id", "season", "week"]].copy()
    targets["gameday"] = pool.set_index("game_id")["gameday"].reindex(frame["game_id"]).to_numpy()
    targets["line"] = frame["tue_open_home_spread"].to_numpy()
    targets["point"] = frame["center_S3"].to_numpy()
    assert not targets["gameday"].isna().any()
    for arm, half_width in ARMS.items():
        if f"p_{arm}" not in frame.columns:
            pytest.skip(f"replay.parquet predates the {arm} mapping")
        mapped = mass_preserving(pool, targets, half_width).set_index("game_id")
        mapped = mapped.reindex(frame["game_id"])
        for column, recorded in (
            ("home_cover_probability", f"p_{arm}"),
            ("push", f"push_{arm}"),
            ("theta", f"theta_{arm}"),
            ("band", f"band_{arm}"),
            ("band_games", f"band_games_{arm}"),
        ):
            assert np.array_equal(
                mapped[column].to_numpy(dtype=float), frame[recorded].to_numpy(dtype=float)
            ), f"{arm} {column} does not replay bit-for-bit"


def _synthetic_reader() -> DiscretePushReader:
    rng = np.random.default_rng(20260908)
    n = 3000
    lines = rng.choice([-7.0, -3.0, -2.5, 0.0, 2.5, 3.0, 6.5, 7.0, 10.0], size=n)
    margins = np.round(lines + rng.normal(0.0, 13.0, size=n))
    on_three = rng.random(n) < 0.12
    margins[on_three] = 3.0
    return DiscretePushReader(lines=lines.astype(float), margins=margins.astype(float))


def test_alternative_lines_keep_the_games_own_distribution() -> None:
    reader = _synthetic_reader()
    point = 3.6
    quoted = 3.0
    covers = [
        reader.three_way(alt, point, conditioning_line=quoted)[0]
        for alt in np.arange(-6.0, 12.5, 0.5)
    ]
    assert (np.diff(np.asarray(covers)) <= 1e-12).all()
    assert reader.three_way(quoted, point, conditioning_line=quoted) == reader.three_way(
        quoted, point
    )
    conditioned = reader.read(9.0, point, conditioning_line=quoted)
    unconditioned = reader.read(9.0, point)
    assert conditioned.band_games == reader.read(quoted, point).band_games
    assert conditioned.theta == reader.read(quoted, point).theta
    assert conditioned.band_games != unconditioned.band_games or conditioned.theta != (
        unconditioned.theta
    )


def test_discrete_sweep_conditions_on_the_quoted_line() -> None:
    from nfl_ats.mass_preserving_lattice import serve_discrete_sweep

    reader = _synthetic_reader()
    sweep = pd.DataFrame(
        {
            "game_id": ["g"] * 5,
            "alternative_line": [1.0, 2.0, 3.0, 4.0, 5.0],
            "home_cover_probability_excluding_push": 0.0,
            "push_probability": 0.0,
            "home_loss_probability": 0.0,
        }
    )
    served = serve_discrete_sweep(
        sweep, reader, points_by_game={"g": 3.6}, quoted_lines_by_game={"g": 3.0}
    )
    covers = served["home_cover_probability_excluding_push"].to_numpy()
    assert (np.diff(covers) <= 1e-12).all()
    expected = [
        reader.three_way(line, 3.6, conditioning_line=3.0) for line in sweep.alternative_line
    ]
    assert np.allclose(covers, [e[0] for e in expected])


def test_scoring_failure_falls_back_to_the_smooth_read() -> None:
    from nfl_ats.cli_commands.prediction import _with_discrete_fallback
    from nfl_ats.mass_preserving_lattice import ProductionDiscretePushRead

    reader = _synthetic_reader()
    discrete = ProductionDiscretePushRead(
        policy="test",
        reader=reader,
        source_path=None,
        source_model_id=None,
        active_model_id=None,
        opener_lines_matched=0,
        warnings=(),
    )
    log: dict[str, object] = {"stale": object()}
    calls: list[object] = []

    def scorer(active: DiscretePushReader | None) -> pd.DataFrame:
        calls.append(active)
        if active is not None:
            raise ValueError("No prior games within 20.0 points of line 100.0")
        return pd.DataFrame({"ok": [1]})

    frame, served = _with_discrete_fallback(scorer, discrete, log)
    assert frame["ok"].tolist() == [1]
    assert calls == [reader, None]
    assert served is not None and served.reader is None and served.served is False
    assert "smooth read was served" in (served.error or "")
    assert log == {}

    smooth_only = ProductionDiscretePushRead(
        policy="test",
        reader=None,
        source_path=None,
        source_model_id=None,
        active_model_id=None,
        opener_lines_matched=0,
        warnings=(),
    )

    def broken(active: DiscretePushReader | None) -> pd.DataFrame:
        raise RuntimeError("model failure")

    with pytest.raises(RuntimeError):
        _with_discrete_fallback(broken, smooth_only, None)
