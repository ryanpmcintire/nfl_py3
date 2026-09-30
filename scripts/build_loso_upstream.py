from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from build_loso_base import diagnostics, frame_digest, metric_rows, require
from threadpoolctl import threadpool_limits

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.home_side_location import (
    HOME_SIDE_OFFSET_SERVED,
    archive_prior_stream,
    fit_home_side_offsets,
    prior_rows_before,
)
from nfl_ats.margin import fit_margin_model, margin_feature_columns
from nfl_ats.mass_preserving_lattice import (
    DiscretePushReader,
    prior_pool,
    prior_pool_for_week,
    serve_discrete_three_way,
)
from nfl_ats.pbp import analysis_plays, snapshot_from_root
from nfl_ats.pbp08_matchup_flags import MIN_QUANTILE_POOL, _team_game_windows
from nfl_ats.pick_probability import (
    BASE_PROBABILITY_POLICY,
    COUNTED_FLAG_COLUMNS,
    FLAG_SUM_COLUMN,
    MOVE_AVAILABLE_COLUMN,
    MOVE_COLUMN,
    PROBABILITY_EPSILON,
    signed_composition_flags,
)
from nfl_ats.pick_probability_fit import (
    FIT_FEATURES,
    FIT_ITERATIONS,
    FIT_RIDGE,
    FORECAST_TEMP_ARCHIVE,
    MARKET_MOVE_COLUMN,
    _design,
    _fit_logit,
    _natural_coefficients,
    _predict,
    _standardisers,
)

SEASONS = tuple(range(2020, 2026))


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class Replay:
    def __init__(self, source: Path, features: Path, schedules: Path, pbp: Path):
        self.hashes: dict[str, str] = {}
        self.raw_cache: dict[tuple[int, ...], tuple[pd.DataFrame, np.ndarray, dict]] = {}
        self.input_cache: dict[tuple[tuple[int, ...], int], pd.DataFrame] = {}
        self.fold_audits: dict[str, dict] = {}
        self.active = self.json(Path("artifacts/active_ats_model.json"))
        self.source_metadata = self.json(source / "metadata.json")
        self.source = self.parquet(source / "per_game.parquet")
        self.feature_hash = self.track(features)
        require(self.feature_hash == self.active["feature_table_sha256"], "Feature hash mismatch")
        require(self.active["regressor"] == "ridge", "Active regressor is not ridge")
        require(self.active["method"] == "market_residual", "Active target changed")
        require(self.active["calibration_method"] == "none", "Margin calibration changed")
        require(
            self.source_metadata["active_model_id"] == self.active["model_id"],
            "Source model mismatch",
        )
        require(self.source_metadata["features"] == list(FIT_FEATURES), "Four terms changed")
        require(self.source_metadata["ridge"] == FIT_RIDGE, "Calibration ridge changed")
        require(
            self.source_metadata["counted_flag_columns"] == list(COUNTED_FLAG_COLUMNS),
            "Composition changed",
        )
        self.features = pd.read_parquet(features)
        self.features = self.features.loc[
            self.features.game_type.eq("REG") & self.features.season.le(max(SEASONS))
        ].copy()
        self.features["gameday"] = pd.to_datetime(self.features.gameday)
        require(not self.features.game_id.duplicated().any(), "Duplicate feature rows")
        self.opener_source = Path("artifacts") / self.source_metadata["opener_evaluation"]
        opener_meta = self.json(self.opener_source / "metadata.json")
        require(opener_meta["active_model_id"] == self.active["model_id"], "Opener model mismatch")
        opener = self.parquet(self.opener_source / "per_game.parquet")
        opener = opener.loc[opener.season.isin(SEASONS)].copy()
        require(not opener.game_id.duplicated().any(), "Duplicate opener rows")
        require(opener.base_probability_policy.eq(BASE_PROBABILITY_POLICY).all(), "Opener policy")
        keep = ["game_id", "season", "week", "tue_open_home_spread", "margin_vs_open"]
        self.games = self.features.merge(
            opener[keep], on=["game_id", "season", "week"], how="inner", validate="one_to_one"
        )
        require(len(self.games) == len(opener), "Missing opener feature rows")
        self.games = self.games.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
        require(
            np.allclose(
                self.games.result - self.games.tue_open_home_spread,
                self.games.margin_vs_open,
                rtol=0,
                atol=1e-12,
            ),
            "Opener grade changed",
        )
        self.games["spread_line"] = self.games.tue_open_home_spread
        self.games["home_covered"] = self.games.margin_vs_open.gt(0).astype(float)
        self.games.loc[self.games.margin_vs_open.eq(0), "home_covered"] = np.nan
        self.games["served_evaluation_home_probability"] = self.games.game_id.map(
            self.source.set_index("game_id").out_of_season_home_probability
        )
        require(
            set(self.games.loc[self.games.home_covered.notna(), "game_id"])
            == set(self.source.loc[self.source.season.isin(SEASONS), "game_id"]),
            "Decisive population differs from frozen calibration",
        )
        self.pool = prior_pool(self.features, opener.set_index("game_id").tue_open_home_spread)
        schedule = self.parquet(schedules)
        schedule = schedule.loc[schedule.game_type.eq("REG") & schedule.season.le(max(SEASONS))]
        arrest_paths = sorted(
            Path("data/raw/player_arrests").glob("*/incidents_point_in_time.parquet")
        )
        require(bool(arrest_paths), "Missing point-in-time arrest source")
        incidents = self.parquet(arrest_paths[-1])
        forecasts = self.parquet(Path("data") / FORECAST_TEMP_ARCHIVE)
        flags = signed_composition_flags(
            self.games, schedule, incidents=incidents, forecasts_tuesday_noon=forecasts
        )
        self.games = self.games.merge(flags, on="game_id", validate="one_to_one")
        frozen = self.source.set_index("game_id")
        for column in COUNTED_FLAG_COLUMNS:
            if column != "flag_protection":
                actual = self.games.set_index("game_id")[column].reindex(frozen.index)
                require(actual.eq(frozen[column]).all(), f"Fixed input changed: {column}")
        market_path = Path("artifacts") / self.source_metadata["market_move_artifact"]
        market = self.parquet(market_path)
        market_meta = self.json(market_path.parent / "summary.json")
        require(
            self.hashes[market_path.as_posix()] == market_meta["market_move_sha256"],
            "Market source hash mismatch",
        )
        require(not market.game_id.duplicated().any(), "Duplicate movement rows")
        move = self.games.game_id.map(market.set_index("game_id")[MARKET_MOVE_COLUMN])
        self.games[MOVE_COLUMN] = move.fillna(0.0)
        self.games[MOVE_AVAILABLE_COLUMN] = move.notna().astype(float)
        for column in (MOVE_COLUMN, MOVE_AVAILABLE_COLUMN):
            actual = self.games.set_index("game_id")[column].reindex(frozen.index)
            require(
                np.allclose(actual, frozen[column], rtol=0, atol=1e-12),
                f"Movement input changed: {column}",
            )
        self.pressure = self.pressure_inputs(pbp, schedule)
        original = self.protection(self.games, ())
        actual = pd.Series(original["flag"], index=self.games.game_id).reindex(frozen.index)
        require(
            actual.eq(frozen.flag_protection).all(), "Pressure source differs from served flags"
        )
        for path in (
            Path(__file__),
            Path("scripts/build_loso_base.py"),
            Path("src/nfl_ats/margin.py"),
            Path("src/nfl_ats/mass_preserving_lattice.py"),
            Path("src/nfl_ats/home_side_location.py"),
            Path("src/nfl_ats/pbp.py"),
            Path("src/nfl_ats/pbp08_matchup_flags.py"),
            Path("src/nfl_ats/pick_probability.py"),
            Path("src/nfl_ats/pick_probability_fit.py"),
        ):
            self.track(path)
        self.protocol = (
            Path("docs/lanes/loso-base-artifact.md")
            .read_text(encoding="utf-8-sig")
            .split("## Protocol\n", 1)[1]
            .split("\n## Tried", 1)[0]
            .strip()
        )
        print(
            f"inventory passed: {len(self.games)} opener rows; frozen flags/moves reproduced",
            flush=True,
        )

    def track(self, path: Path) -> str:
        key = path.resolve().relative_to(Path.cwd()).as_posix()
        value = sha256(path)
        require(key not in self.hashes or self.hashes[key] == value, f"Input changed: {key}")
        self.hashes[key] = value
        return value

    def json(self, path: Path) -> dict:
        self.track(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def parquet(self, path: Path) -> pd.DataFrame:
        self.track(path)
        return pd.read_parquet(path)

    def pressure_inputs(self, root: Path, schedule: pd.DataFrame) -> pd.DataFrame:
        self.track(root / "manifest.json")
        snapshot = snapshot_from_root(root)
        batches = []
        for season in snapshot.seasons:
            if season > max(SEASONS):
                continue
            plays = self.parquet(snapshot.season_path(season))
            plays = analysis_plays(plays.loc[plays.season_type.eq("REG")])
            plays = plays.loc[plays.competitive_play].copy()
            for column in ("posteam", "defteam"):
                plays[column] = plays[column].replace(TEAM_ABBREVIATION_ALIASES)
            for column in ("qb_dropback", "sack", "qb_hit"):
                plays[column] = pd.to_numeric(plays[column], errors="coerce")
            plays = plays.loc[plays.qb_dropback.fillna(0).eq(1)].copy()
            plays["pressure"] = (plays.sack.fillna(0).eq(1) | plays.qb_hit.fillna(0).eq(1)).astype(
                float
            )
            allowed = (
                plays.groupby(["game_id", "posteam"])
                .pressure.mean()
                .rename("press_allow_g")
                .reset_index()
                .rename(columns={"posteam": "team"})
            )
            generated = (
                plays.groupby(["game_id", "defteam"])
                .pressure.mean()
                .rename("press_gen_g")
                .reset_index()
                .rename(columns={"defteam": "team"})
            )
            batches.append(allowed.merge(generated, on=["game_id", "team"], how="outer"))
        traits = pd.concat(batches, ignore_index=True)
        schedule = schedule.copy()
        schedule["gameday"] = pd.to_datetime(schedule.gameday)
        for column in ("home_team", "away_team"):
            schedule[column] = schedule[column].replace(TEAM_ABBREVIATION_ALIASES)
        return _team_game_windows(schedule, traits)

    def protection(self, games: pd.DataFrame, excluded: tuple[int, ...]) -> dict:
        long = self.pressure
        selected = long.loc[long.game_id.isin(games.game_id)].copy()
        pool = long.loc[~long.season.isin(excluded)]
        if excluded:
            require(not pool.season.isin(excluded).any(), "Protection training overlap")
        selected["allow_top"] = False
        selected["gen_top"] = False
        selected["threshold_training_rows"] = 0
        for block, rows in selected.groupby("week_block", sort=True):
            prior = pool.loc[pool.week_block.lt(block)]
            selected.loc[rows.index, "threshold_training_rows"] = len(prior)
            for source, target in (("press_allow_w", "allow_top"), ("press_gen_w", "gen_top")):
                values = prior[source].dropna()
                if len(values) >= MIN_QUANTILE_POOL:
                    q25, q75 = values.quantile([0.25, 0.75]).to_numpy()
                    selected.loc[rows.index, target] = rows[source].gt(q25) & rows[source].ge(q75)
        home = selected.loc[selected.is_home].set_index("game_id").reindex(games.game_id)
        away = selected.loc[~selected.is_home].set_index("game_id").reindex(games.game_id)
        home_bad = home.allow_top & away.gen_top
        away_bad = away.allow_top & home.gen_top
        return {
            "flag": np.where(home_bad & ~away_bad, -1, np.where(away_bad & ~home_bad, 1, 0)),
            "rows": home.threshold_training_rows.to_numpy(),
        }

    def raw(self, excluded: tuple[int, ...]) -> tuple[pd.DataFrame, np.ndarray, dict]:
        excluded = tuple(sorted(set(excluded)))
        if excluded in self.raw_cache:
            return self.raw_cache[excluded]
        training = self.features.loc[
            ~self.features.season.isin(excluded)
            & self.features.result.notna()
            & self.features.ats_margin.notna()
        ].copy()
        target = self.games.loc[self.games.season.isin(excluded)].copy().reset_index(drop=True)
        require(not training.season.isin(target.season.unique()).any(), "Margin training overlap")
        require(not training.game_id.isin(target.game_id).any(), "Margin game overlap")
        model = fit_margin_model(
            training,
            target="market_residual",
            model_name="ridge",
            feature_profile=self.active["feature_profile"],
            ridge_alpha=float(self.active["ridge_alpha"]),
        )
        require(model.training_rows == len(training), "Margin training row count mismatch")
        forecast = model.predict(target, probability_method=self.active["probability_method"])
        forecast = forecast.reset_index(drop=True)
        for column in ("game_id", "season", "week", "tue_open_home_spread", "result"):
            forecast[column] = target[column].to_numpy()
        forecast["residual_at_open"] = forecast.predicted_market_residual
        audit = {
            "excluded_seasons": list(excluded),
            "training_rows": len(training),
            "training_seasons": sorted(int(x) for x in training.season.unique()),
            "training_game_ids_sha256": frame_digest(training[["game_id", "season"]]),
            "training_max_gameday": model.training_max_gameday,
            "residual_distribution_rows": model.distribution_rows,
            "same_season_training_rows": 0,
        }
        self.raw_cache[excluded] = forecast, model.residuals, audit
        print(
            f"margin fit {len(self.raw_cache)}: exclude {excluded}; train {len(training)}",
            flush=True,
        )
        return self.raw_cache[excluded]

    def inputs(self, excluded: tuple[int, ...], season: int) -> pd.DataFrame:
        excluded = tuple(sorted(set(excluded)))
        key = excluded, season
        require(season in excluded, "Input prediction season must be held out")
        if key in self.input_cache:
            return self.input_cache[key].copy()
        all_raw, residuals, audit = self.raw(excluded)
        raw = all_raw.loc[all_raw.season.eq(season)].set_index("game_id")
        games = self.games.loc[self.games.season.eq(season)].copy().reset_index(drop=True)
        stream_parts = []
        for other in SEASONS:
            if other not in excluded:
                other_raw, _, _ = self.raw(tuple(sorted((*excluded, other))))
                stream_parts.append(archive_prior_stream(other_raw.loc[other_raw.season.eq(other)]))
        stream = pd.concat(stream_parts, ignore_index=True)
        require(not stream.season.isin(excluded).any(), "Location training overlap")
        pool = self.pool.loc[~self.pool.season.isin(excluded)]
        require(not pool.season.isin(excluded).any(), "Discrete training overlap")
        for week, group in games.groupby("week", sort=True):
            forecast = raw.reindex(group.game_id).reset_index()
            prior = prior_rows_before(stream, season, int(week))
            offsets = (
                fit_home_side_offsets(prior).offset_for(group.spread_line).fillna(0).to_numpy()
                if HOME_SIDE_OFFSET_SERVED
                else np.zeros(len(group))
            )
            forecast["predicted_margin"] += offsets
            cutoff = self.features.loc[
                self.features.season.eq(season) & self.features.week.eq(week), "gameday"
            ].min()
            eligible = prior_pool_for_week(pool, season=season, week=int(week), cutoff=cutoff)
            require(not eligible.season.isin(excluded).any(), "Discrete reader season overlap")
            reader = DiscretePushReader.for_week(pool, season=season, week=int(week), cutoff=cutoff)
            predicted = serve_discrete_three_way(
                forecast,
                group,
                reader,
                residuals=residuals,
                probability_method=self.active["probability_method"],
            )
            for column in (
                "home_cover_probability",
                "home_cover_probability_excluding_push",
                "push_probability",
                "home_loss_probability",
                "predicted_margin",
            ):
                games.loc[group.index, column] = predicted[column].to_numpy()
            games.loc[group.index, "home_side_offset_at_open"] = offsets
            games.loc[group.index, "location_training_rows"] = len(prior)
            games.loc[group.index, "discrete_training_rows"] = len(eligible)
            games.loc[group.index, "discrete_training_seasons"] = ",".join(
                str(int(x)) for x in sorted(eligible.season.unique())
            )
        protection = self.protection(games, excluded)
        games["flag_protection"] = protection["flag"]
        games["protection_threshold_training_rows"] = protection["rows"]
        games[FLAG_SUM_COLUMN] = games[list(COUNTED_FLAG_COLUMNS)].sum(axis=1).astype(float)
        games["home_cover_probability_at_open"] = games.home_cover_probability
        games["model_probability"] = games.home_cover_probability.clip(
            PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
        )
        games["model_logit"] = np.log(games.model_probability / (1.0 - games.model_probability))
        games["upstream_training_seasons"] = ",".join(map(str, audit["training_seasons"]))
        games["upstream_excluded_seasons"] = ",".join(map(str, excluded))
        games["upstream_training_games"] = audit["training_rows"]
        games["upstream_same_season_training_rows"] = 0
        games["location_same_season_training_rows"] = 0
        games["discrete_same_season_training_rows"] = 0
        games["protection_same_season_training_rows"] = 0
        games["held_out_season"] = season
        require(np.isfinite(games[list(FIT_FEATURES)].to_numpy()).all(), "Nonfinite input")
        require(
            np.allclose(
                games.home_cover_probability_excluding_push
                + games.push_probability
                + games.home_loss_probability,
                1.0,
                rtol=0,
                atol=1e-10,
            ),
            "Discrete masses do not sum to one",
        )
        self.input_cache[key] = games.copy()
        return games

    def run(self, output: Path) -> dict:
        held_out, training_scores = [], []
        for season in SEASONS:
            test = self.inputs((season,), season)
            train = pd.concat(
                [
                    self.inputs(tuple(sorted((season, other))), other)
                    for other in SEASONS
                    if other != season
                ],
                ignore_index=True,
            )
            train = train.loc[train.home_covered.notna()].copy()
            require(not train.season.eq(season).any(), "Calibration training overlap")
            require(not train.game_id.isin(test.game_id).any(), "Calibration game overlap")
            means, stds = _standardisers(train)
            beta = _fit_logit(_design(train, means, stds), train.home_covered.to_numpy(), FIT_RIDGE)
            test["base_home_probability"] = _predict(test, beta, means, stds)
            test["four_term_probability"] = test.base_home_probability
            test["base_training_seasons"] = ",".join(str(x) for x in SEASONS if x != season)
            test["base_training_games"] = len(train)
            test["same_season_training_rows"] = 0
            train_probability = _predict(train, beta, means, stds)
            scores = metric_rows(train.home_covered.to_numpy(), train_probability)
            training_scores.append(scores)
            self.fold_audits[str(season)] = {
                "calibration_training_seasons": sorted(int(x) for x in train.season.unique()),
                "calibration_training_games": len(train),
                "same_season_training_rows": 0,
                "training_inputs_sha256": frame_digest(train[["game_id", "season", *FIT_FEATURES]]),
                "coefficients": _natural_coefficients(beta, means, stds),
                "standardization": {"means": means, "stds": stds},
                "training_metrics": dict(
                    zip(
                        ("accuracy", "log_loss", "brier"), scores.mean(axis=0).tolist(), strict=True
                    )
                ),
            }
            held_out.append(test)
            print(f"calibration fold {season} complete: {len(test)} predictions", flush=True)
        frame = pd.concat(held_out, ignore_index=True)
        frame["market_even_home_probability"] = 0.5
        frame["base_pick_home"] = frame.base_home_probability.ge(0.5)
        frame["base_pick_probability"] = np.maximum(
            frame.base_home_probability, 1.0 - frame.base_home_probability
        )
        frame["base_correct"] = (
            frame.base_pick_home.astype(float).eq(frame.home_covered).astype(float)
        )
        frame.loc[frame.home_covered.isna(), "base_correct"] = np.nan
        frame["feature_table_sha256"] = self.feature_hash
        frame["opener_source"] = self.opener_source.as_posix()
        frame["base_probability_policy"] = BASE_PROBABILITY_POLICY
        decisive = frame.loc[frame.home_covered.notna()].copy()
        result = diagnostics(decisive, np.concatenate(training_scores))
        for section in ("overall", "by_season", "reliability"):
            result[section]["loso_upstream"] = result[section].pop("loso_base")
        for season in SEASONS:
            fold = self.fold_audits[str(season)]
            held = result["by_season"]["loso_upstream"][str(season)]
            fold["held_out_metrics"] = held
            fold["held_out_minus_training"] = {
                name: held[name] - fold["training_metrics"][name]
                for name in ("accuracy", "log_loss", "brier")
            }
        coefficients = pd.DataFrame({k: v["coefficients"] for k, v in self.fold_audits.items()}).T
        columns = [
            "game_id",
            "season",
            "week",
            "game_type",
            "gameday",
            "home_team",
            "away_team",
            "tue_open_home_spread",
            "result",
            "margin_vs_open",
            "home_covered",
            "model_probability",
            *FIT_FEATURES,
            *COUNTED_FLAG_COLUMNS,
            "home_cover_probability_at_open",
            "home_cover_probability_excluding_push",
            "push_probability",
            "home_loss_probability",
            "predicted_margin",
            "home_side_offset_at_open",
            "base_home_probability",
            "four_term_probability",
            "base_pick_home",
            "base_pick_probability",
            "base_correct",
            "served_evaluation_home_probability",
            "market_even_home_probability",
            "held_out_season",
            "base_training_games",
            "base_training_seasons",
            "same_season_training_rows",
            "upstream_training_seasons",
            "upstream_training_games",
            "upstream_excluded_seasons",
            "upstream_same_season_training_rows",
            "location_same_season_training_rows",
            "location_training_rows",
            "discrete_same_season_training_rows",
            "discrete_training_rows",
            "discrete_training_seasons",
            "protection_same_season_training_rows",
            "protection_threshold_training_rows",
            "feature_table_sha256",
            "opener_source",
            "base_probability_policy",
        ]
        predictions = frame[list(dict.fromkeys(columns))].sort_values(["season", "week", "game_id"])
        require(not predictions.duplicated(["game_id", "season"]).any(), "Duplicate predictions")
        for key, value in self.hashes.items():
            require(sha256(Path(key)) == value, f"Source changed during replay: {key}")
        output.mkdir(parents=True, exist_ok=False)
        predictions.to_parquet(output / "predictions.parquet", index=False)
        metadata = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "command": "build_loso_upstream",
            "active_model": self.active,
            "feature_table_sha256": self.feature_hash,
            "opener_source": self.opener_source.as_posix(),
            "protocol": self.protocol,
            "source_hashes": self.hashes,
            "predictions_sha256": sha256(output / "predictions.parquet"),
            "rows": len(predictions),
            "decisive_games": len(decisive),
            "pushes": int(frame.home_covered.isna().sum()),
            "missing_predictions": 0,
            "features": list(FIT_FEATURES),
            "margin_features": list(
                margin_feature_columns("market_residual", self.active["feature_profile"])
            ),
            "calibration_ridge": FIT_RIDGE,
            "calibration_iterations": FIT_ITERATIONS,
            "market_move_feature_version": self.source_metadata["market_move_feature_version"],
            "base_probability_policy": BASE_PROBABILITY_POLICY,
            "unique_margin_fits": len(self.raw_cache),
            "numerical_threads": 2,
            "margin_folds": {",".join(map(str, k)): v[2] for k, v in self.raw_cache.items()},
            "calibration_folds": self.fold_audits,
            "coefficient_stability": {
                c: {
                    "min": float(coefficients[c].min()),
                    "max": float(coefficients[c].max()),
                    "std": float(coefficients[c].std(ddof=0)),
                }
                for c in coefficients
            },
            "diagnostics": result,
            "descriptive_looks": 53,
            "bootstrap": {"unit": "season", "draws": 10000, "seed": 20260929, "coverage": 0.95},
            "limitations": [
                "Retrospective LOSO includes later seasons and uses previously selected features.",
                "Frozen features are not refit; rolling inputs may use earlier same-season games.",
                "Sunday market movement does not establish Tuesday or Thursday availability.",
                "Opener probabilities only; alternative lines need another discrete read.",
                "Four-term probability excludes pushes; push mass comes from the upstream model.",
                "Calibration-training diagnostics reuse their fitted calibration coefficients.",
            ],
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        reloaded = pd.read_parquet(output / "predictions.parquet")
        require(len(reloaded) == len(predictions), "Artifact readback lost rows")
        return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build nested season-held-out margin and four-term inputs"
    )
    parser.add_argument(
        "--source-artifact", type=Path, default=Path("artifacts/pick_probability/20260929T192747Z")
    )
    parser.add_argument(
        "--features", type=Path, default=Path("data/processed/game_features_weak_stack.parquet")
    )
    parser.add_argument(
        "--schedules", type=Path, default=Path("data/raw/20260929T191306Z/schedules.parquet")
    )
    parser.add_argument("--pbp", type=Path, default=Path("data/pbp/raw/20260929T191306Z"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("artifacts/loso_upstream") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    require(
        output.resolve().is_relative_to(Path("artifacts/loso_upstream").resolve()),
        "Output must stay under artifacts/loso_upstream",
    )
    require(not output.exists(), "Output already exists")
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    with threadpool_limits(limits=2):
        replay = Replay(args.source_artifact, args.features, args.schedules, args.pbp)
        metadata = replay.run(output)
    print(
        json.dumps(
            {
                "artifact": output.as_posix(),
                "rows": metadata["rows"],
                "decisive_games": metadata["decisive_games"],
                "pushes": metadata["pushes"],
                "unique_margin_fits": metadata["unique_margin_fits"],
                "metrics": metadata["diagnostics"]["overall"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
