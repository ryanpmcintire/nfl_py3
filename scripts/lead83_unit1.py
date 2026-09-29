from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

FIT_ROOT = Path("artifacts/pick_probability/20260929T192747Z")
MARKET_ROOT = Path("data/market/raw")
SCHEDULE = Path("data/raw/20260908T162105Z/schedules.parquet")
OUTPUT = Path("tests/scratch/codex/lead83_unit1")
REPORT = Path("docs/lead83_unit1.md")
ZONE = "America/New_York"
LABELS = ("tue_open", "sun_early_close")
SEASONS = tuple(range(2020, 2026))
EXPECTED_GAMES = 1311
EXPECTED_FILES = 131


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def instant(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def load_population() -> tuple[pd.DataFrame, dict]:
    metadata = json.loads((FIT_ROOT / "metadata.json").read_text(encoding="utf-8-sig"))
    opener_root = Path("artifacts") / metadata["opener_evaluation"]
    opener_metadata = json.loads((opener_root / "metadata.json").read_text(encoding="utf-8-sig"))
    policy = "discrete_conditional_non_push_v1"
    if (
        metadata["base_probability_policy"] != policy
        or opener_metadata["base_probability_policy"] != policy
        or metadata["active_model_id"] != opener_metadata["active_model_id"]
    ):
        raise ValueError("Frozen opener and four-term artifacts disagree")
    fitted = read_frame(FIT_ROOT / "per_game.parquet", ["game_id", "season", "week"])
    games = read_frame(
        opener_root / "per_game.parquet",
        ["game_id", "season", "week", "tue_open_home_spread"],
    )
    if fitted.game_id.duplicated().any() or games.game_id.duplicated().any():
        raise ValueError("Duplicate game in a frozen source")
    if not set(fitted.game_id).issubset(games.game_id):
        raise ValueError("Four-term population is outside the opener population")
    if len(games) - len(fitted) != int(metadata["pushes_dropped"]):
        raise ValueError("Non-fit opener rows cannot all be identified as pushes")
    games["declared_fit_population"] = games.game_id.isin(fitted.game_id)
    schedule = read_frame(
        SCHEDULE,
        ["game_id", "season", "week", "game_type", "gameday", "gametime", "home_team", "away_team"],
    )
    schedule = schedule.loc[schedule.season.isin(SEASONS) & schedule.game_type.eq("REG")].copy()
    schedule["gameday"] = pd.to_datetime(schedule.gameday)
    local = pd.to_datetime(
        schedule.gameday.dt.strftime("%Y-%m-%d") + " " + schedule.gametime.astype(str)
    )
    schedule["kickoff"] = local.dt.tz_localize(ZONE).dt.tz_convert("UTC")
    anchor = schedule.groupby(["season", "week"]).gameday.transform("min")
    sunday = anchor + pd.to_timedelta((6 - anchor.dt.dayofweek) % 7, unit="D")
    schedule["tuesday_noon"] = (
        (sunday + pd.Timedelta(days=-5, hours=12)).dt.tz_localize(ZONE).dt.tz_convert("UTC")
    )
    schedule["sunday_cutoff"] = (
        (sunday + pd.Timedelta(hours=12, minutes=30)).dt.tz_localize(ZONE).dt.tz_convert("UTC")
    )
    games = games.merge(schedule, on=["game_id", "season", "week"], validate="one_to_one")
    if len(games) != int(opener_metadata["games"]):
        raise ValueError("Frozen opener rows are missing from the schedule")
    lineage = {
        "fit_source": str(FIT_ROOT / "per_game.parquet"),
        "fit_sha256": digest(FIT_ROOT / "per_game.parquet"),
        "opener_source": str(opener_root / "per_game.parquet"),
        "opener_sha256": digest(opener_root / "per_game.parquet"),
        "schedule_source": str(SCHEDULE),
        "schedule_sha256": digest(SCHEDULE),
        "fit_games": len(fitted),
        "opener_games_with_pushes": len(games),
        "pushes_preserved_before_quote_matching": int(metadata["pushes_dropped"]),
        "upstream_cutoff_columns": [
            name
            for name in pq.read_schema(opener_root / "per_game.parquet").names
            if "train" in name or "cutoff" in name
        ],
    }
    return games, lineage


def recover_books(games: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict]:
    columns = [
        "nflverse_game_id",
        "bookmaker_key",
        "market",
        "home_spread_line",
        "line",
        "outcome_side",
        "observed_at_utc",
        "bookmaker_last_update_utc",
        "market_last_update_utc",
        "commence_time_utc",
        "home_team",
        "away_team",
    ]
    frames = []
    sources = []
    counts = {label: {"files": 0, "rows": 0} for label in LABELS}
    invalid_clock_rows = 0
    for path in sorted(MARKET_ROOT.glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        request = manifest.get("request", {})
        label = request.get("decision_label")
        season = int(request.get("season", 0))
        if (
            label not in LABELS
            or season not in SEASONS
            or manifest.get("provider") != "the-odds-api"
            or manifest.get("capture_kind") != "historical_backfill"
            or request.get("sport") != "americanfootball_nfl"
        ):
            continue
        quotes_path = path.parent / "quotes.parquet"
        expected = manifest.get("files", {}).get("quotes.parquet", {}).get("sha256")
        if not expected or digest(quotes_path) != expected:
            raise ValueError(f"Quote integrity failure: {path}")
        stamp = instant(manifest.get("snapshot_timestamp_utc"))
        requested = instant(manifest.get("requested_at_utc"))
        if pd.isna(stamp) or pd.isna(requested) or stamp > requested:
            raise ValueError(f"Invalid snapshot clock: {path}")
        local = requested.tz_convert(ZONE)
        weekday, hour, minute = (1, 12, 0) if label == LABELS[0] else (6, 12, 30)
        if local.dayofweek != weekday or (local.hour, local.minute) > (hour, minute):
            raise ValueError(f"Capture exceeds its declared clock: {path}")
        q = read_frame(quotes_path, columns)
        counts[label]["files"] += 1
        counts[label]["rows"] += len(q)
        sources.append(
            {
                "manifest": str(path),
                "manifest_sha256": digest(path),
                "quotes_sha256": expected,
                "label": label,
                "season": season,
                "week": int(request["week"]),
                "rows": len(q),
            }
        )
        q = q.loc[q.market.eq("spreads")].rename(columns={"nflverse_game_id": "game_id"})
        target = games.loc[games.season.eq(season) & games.week.eq(int(request["week"]))]
        q = q.merge(target, on="game_id", suffixes=("", "_schedule"), validate="many_to_one")
        if q.empty:
            continue
        if not (q.home_team.eq(q.home_team_schedule) & q.away_team.eq(q.away_team_schedule)).all():
            raise ValueError(f"Quote and schedule teams disagree: {path}")
        for name in [
            "observed_at_utc",
            "bookmaker_last_update_utc",
            "market_last_update_utc",
            "commence_time_utc",
        ]:
            q[name] = pd.to_datetime(q[name], utc=True, errors="coerce", format="mixed")
        observed = q.observed_at_utc
        local_day = observed.dt.tz_convert(ZONE).dt.date
        anchor = q.tuesday_noon if label == LABELS[0] else q.sunday_cutoff
        valid = (
            observed.eq(stamp)
            & local_day.eq(anchor.dt.tz_convert(ZONE).dt.date)
            & observed.le(anchor)
            & observed.lt(q.kickoff)
            & observed.lt(q.commence_time_utc)
            & q.bookmaker_last_update_utc.le(observed)
            & q.market_last_update_utc.le(observed)
            & q.bookmaker_key.notna()
            & q.bookmaker_key.ne("")
            & np.isfinite(q.home_spread_line)
            & np.isfinite(q.line)
        )
        invalid_clock_rows += int((~valid).sum())
        q = q.loc[valid].copy()
        signed = np.where(q.outcome_side.eq("HOME"), -q.line, q.line)
        if not q.outcome_side.isin(["HOME", "AWAY"]).all() or not np.allclose(
            signed, q.home_spread_line
        ):
            raise ValueError(f"Book handicap sign conflict: {path}")
        keys = ["game_id", "bookmaker_key"]
        if q.duplicated([*keys, "outcome_side"]).any():
            raise ValueError(f"Duplicate book outcome: {path}")
        grouped = q.groupby(keys)
        if grouped.home_spread_line.nunique().gt(1).any():
            raise ValueError(f"Book sides disagree: {path}")
        complete = grouped.outcome_side.transform("nunique").eq(2)
        q = q.loc[complete].drop_duplicates(keys)
        q["capture_label"] = label
        frames.append(
            q[
                [
                    *keys,
                    "home_spread_line",
                    "observed_at_utc",
                    "bookmaker_last_update_utc",
                    "capture_label",
                ]
            ]
        )
    if any(item["files"] != EXPECTED_FILES for item in counts.values()):
        raise ValueError(f"Declared archive file counts changed: {counts}")
    quotes = pd.concat(frames, ignore_index=True)
    keys = ["game_id", "bookmaker_key"]
    if quotes.duplicated([*keys, "capture_label"]).any():
        raise ValueError("Multiple snapshots for the same declared book/capture")
    early = quotes.loc[quotes.capture_label.eq(LABELS[0])].drop(columns="capture_label")
    late = quotes.loc[quotes.capture_label.eq(LABELS[1])].drop(columns="capture_label")
    paired = early.merge(late, on=keys, suffixes=("_tuesday", "_sunday"), validate="one_to_one")
    if not paired.observed_at_utc_tuesday.lt(paired.observed_at_utc_sunday).all():
        raise ValueError("Matched-book clocks are not increasing")
    paired["move_toward_home"] = paired.home_spread_line_sunday - paired.home_spread_line_tuesday
    return paired, sources, {"captures": counts, "invalid_clock_rows": invalid_clock_rows}


def variance_features(paired: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game_id, group in paired.groupby("game_id", sort=True):
        moves = group.move_toward_home.to_numpy(dtype=float)
        if len(moves) < 2:
            continue
        jackknife = np.array([np.median(np.delete(moves, i)) for i in range(len(moves))])
        variance = (len(moves) - 1) / len(moves) * np.sum((jackknife - jackknife.mean()) ** 2)
        rows.append(
            {
                "game_id": game_id,
                "matched_books": len(moves),
                "median_move_toward_home": float(np.median(moves)),
                "jackknife_variance": float(variance),
            }
        )
    features = pd.DataFrame(rows).merge(games, on="game_id", validate="one_to_one")
    return features


def fold_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    panels = []
    summaries = []
    for outer in (2023, 2024, 2025):
        train = features.loc[features.season.le(outer - 3)]
        signal_variance = float(train.median_move_toward_home.var(ddof=1))
        noise_variance = float(train.jackknife_variance.mean())
        tau_squared = max(0.0, signal_variance - noise_variance)
        if len(train) < 2 or not np.isfinite(tau_squared):
            raise ValueError("Insufficient training games for a finite variance estimate")
        panel = features.loc[features.season.le(outer)].copy()
        denominator = tau_squared + panel.jackknife_variance.to_numpy()
        weight = np.divide(
            tau_squared, denominator, out=np.zeros(len(panel)), where=denominator > 0
        )
        panel["outer_season"] = outer
        panel["role"] = np.select(
            [panel.season.le(outer - 3), panel.season.eq(outer - 2), panel.season.eq(outer - 1)],
            ["fit", "tune", "calibrate"],
            default="outer",
        )
        panel["tau_squared"] = tau_squared
        panel["shrink_weight"] = weight
        panel["shrunk_move_toward_home"] = weight * panel.median_move_toward_home
        if (
            not np.isfinite(panel.shrunk_move_toward_home).all()
            or not panel.shrink_weight.between(0, 1).all()
        ):
            raise ValueError("Invalid empirical-Bayes move")
        panels.append(panel)
        summaries.append(
            {
                "outer": outer,
                "fit_through": outer - 3,
                "variance_train_games_with_pushes": len(train),
                "train_games": int(train.declared_fit_population.sum()),
                "move_variance": signal_variance,
                "mean_noise_variance": noise_variance,
                "tau_squared": tau_squared,
                "tune_games": int((panel.declared_fit_population & panel.role.eq("tune")).sum()),
                "calibration_games": int(
                    (panel.declared_fit_population & panel.role.eq("calibrate")).sum()
                ),
                "outer_games": int((panel.declared_fit_population & panel.role.eq("outer")).sum()),
            }
        )
    return pd.concat(panels, ignore_index=True), summaries


def write_report(features: pd.DataFrame, summary: dict) -> None:
    lineage = summary["lineage"]
    inventory = summary["inventory"]
    declared = features.loc[features.declared_fit_population]
    lines = [
        "# LEAD-83 unit 1: empirical-Bayes move features",
        "",
        "**Measured:** the real command completed the roadmap's variance-feature unit. "
        "No game-outcome columns were loaded; no probability fit or score was computed.",
        "",
        "## Population and clocks",
        "",
        f"**Measured:** {len(declared):,} non-push games across 2020-2025, "
        f"plus {summary['extra_push_games_preserved']} source-complete pushes retained separately. "
        f"The four-term source has {lineage['fit_games']:,} rows; its opener parent has "
        f"{lineage['opener_games_with_pushes']:,}, including "
        f"{lineage['pushes_preserved_before_quote_matching']} pushes. "
        f"Declared population match: {summary['population_matches_declaration']}.",
        "",
        "| Season | Non-push games | Extra pushes | Median common books | "
        "Zero jackknife variance |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for season, group in features.groupby("season"):
        selected = group.loc[group.declared_fit_population]
        lines.append(
            f"| {season} | {len(selected)} | {int((~group.declared_fit_population).sum())} | "
            f"{selected.matched_books.median():.1f} | "
            f"{int(selected.jackknife_variance.eq(0).sum())} |"
        )
    lines += [
        "",
        "**Measured:** target season/week, game/team identity, archive hashes, "
        "Tuesday/Sunday calendar dates, paired book outcomes, observed/book/market timestamps, "
        "and both archived and schedule kickoff bounds are checked. Tuesday captures precede noon; "
        "they establish available anchors, not noon-capture evidence. The later snapshot is before "
        "Sunday 12:30. Final closing quotes and game-outcome columns are never loaded.",
        "",
    ]
    for label, counts in inventory["captures"].items():
        lines.append(
            f"- **Measured:** {label}: {counts['files']} files, {counts['rows']:,} quote rows."
        )
    lines += [
        f"- **Measured:** {inventory['invalid_clock_rows']} target-population spread rows "
        "excluded by clock/finite-value checks.",
        "",
        "## Frozen construction and training-only variance",
        "",
        "Each book contributes its Sunday minus Tuesday home spread; "
        "positive means movement toward "
        "home. The game measurement is the median. With n books and leave-one-book-out medians "
        "m_i, v=(n-1)/n * sum((m_i-mean(m_i))^2). Training tau^2=max(0, sample variance of game "
        "medians - mean(v)); shrink weight=tau^2/(tau^2+v). "
        "If both terms are zero, weight is zero. "
        "The original move remains available beside the shrunk move. "
        "The variance estimate retains pushes; conditional cover fitting excludes them. "
        "No ranking, bands, or outcomes "
        "choose books or shrinkage. Books are measurement replicates; "
        "games remain the sample units.",
        "",
        "| Outer | Fit through | Fit / tune / calibrate / outer games | "
        "Variance fit including pushes | Move variance | Mean v | tau^2 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for fold in summary["folds"]:
        lines.append(
            f"| {fold['outer']} | {fold['fit_through']} | "
            f"{fold['train_games']} / {fold['tune_games']} / {fold['calibration_games']} / "
            f"{fold['outer_games']} | {fold['variance_train_games_with_pushes']} | "
            f"{fold['move_variance']:.8f} | "
            f"{fold['mean_noise_variance']:.8f} | {fold['tau_squared']:.8f} |"
        )
    lines += [
        "",
        "## Replay handoff",
        "",
        "**Measured:** 0 outcome looks executed; 501 remain predeclared for the cached replay. "
        "Decisive-game record, IS/OOS scores and gap, probability coefficients, season/week-block "
        "intervals, reliability, and probability_positive are not estimated in this feature unit. "
        "The variance estimates above are nuisance feature parameters, "
        "not cover-probability coefficients.",
        "",
        "**Measured:** the parent opener artifact exposes "
        f"{len(lineage['upstream_cutoff_columns'])} "
        "per-row training/cutoff columns. The frozen cache does not certify Protocol C cutoffs. "
        "Protocol C fit or score. Cached replay must preserve fit through Y-3, tuning on Y-2, "
        "calibration on Y-1, and outer Y; do not substitute unrestricted LOSO "
        "or missing-move zeros.",
        "",
        "**Read:** src/nfl_ats/clv.py:2177-2189 fits the opener parent weekly, using all "
        "completed games before the target week. **Inferred:** those upstream fits do not "
        "respect a fixed training cutoff through outer season Y-3. A fold-specific upstream "
        "probability cache is required before this protocol can be scored. The next unit "
        "must reconstruct those folds from cached inputs without a full-history rebuild.",
        "",
        "**Inferred:** this feature unit says nothing about predictive gain. No closure, "
        "promotion, or serving "
        "decision is made. Zero crossing cannot close a signal, and one fitted calibrated discrete "
        "probability must select the side.",
        "",
        "**Measured:** feature rows, matched-book rows, fold roles, input hashes, "
        "and inventory are "
        f"saved under {OUTPUT.as_posix()}/. No prediction-row dump is stored in docs.",
        "",
        "Command: .tools/uv.exe run --no-sync --no-cache python scripts/lead83_unit1.py. "
        "No-cache avoids the inaccessible shared uv cache; no dependencies are installed.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    with threadpool_limits(limits=2):
        games, lineage = load_population()
        paired, sources, inventory = recover_books(games)
        features = variance_features(paired, games)
        population_match = int(features.declared_fit_population.sum()) == EXPECTED_GAMES
        if population_match:
            folds, summaries = fold_features(features)
        else:
            folds, summaries = pd.DataFrame(), []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("matched_books", paired),
        ("features", features),
        ("fold_features", folds),
    ]:
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False), OUTPUT / f"{name}.parquet"
        )
    summary = {
        "status": "variance_feature_unit_complete_replay_pending"
        if population_match
        else "population_mismatch_no_fit",
        "declared_looks": 501,
        "outcome_looks_executed": 0,
        "expected_games": EXPECTED_GAMES,
        "source_complete_nonpush_games": int(features.declared_fit_population.sum()),
        "population_matches_declaration": population_match,
        "extra_push_games_preserved": int((~features.declared_fit_population).sum()),
        "matched_book_pairs": len(paired),
        "folds": summaries,
        "inventory": inventory,
        "lineage": lineage,
        "source_manifests": sources,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(features, summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in [
                    "status",
                    "source_complete_nonpush_games",
                    "extra_push_games_preserved",
                    "outcome_looks_executed",
                ]
            }
        )
    )
    for fold in summaries:
        print(json.dumps(fold))


if __name__ == "__main__":
    run()
