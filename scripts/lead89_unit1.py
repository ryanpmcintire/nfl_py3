from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

BASE = Path("artifacts/pick_probability/20260929T192747Z")
OPENER = Path("artifacts/opener_evaluation/20260929T192743Z")
UPSTREAM = Path("artifacts/margins/20260929T192312Z/predictions.parquet")
SCHEDULE = Path("data/raw/20260908T162105Z/schedules.parquet")
OUTPUT = Path("tests/scratch/codex/lead89_unit1")
REPORT = Path("docs/lead89_unit1.md")
LANE = Path("docs/lanes/lead89.md")
ZONE = "America/New_York"
LABELS = {"tue_open": "tuesday", "thu_pre_tnf": "thursday", "sun_early_close": "sunday"}
STAGES = tuple(LABELS.values())
SEASONS = tuple(range(2020, 2026))


def read_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_population() -> tuple[pd.DataFrame, dict]:
    columns = ["game_id", "season", "week", "tue_open_home_spread"]
    openers = read_frame(OPENER / "per_game.parquet", columns)
    base = read_frame(BASE / "per_game.parquet", columns)
    if openers.game_id.duplicated().any() or base.game_id.duplicated().any():
        raise ValueError("Frozen game identifiers are not unique")
    if not base.game_id.isin(openers.game_id).all():
        raise ValueError("Four-term games are missing from their opener artifact")
    schedule = read_frame(
        SCHEDULE,
        ["game_id", "season", "week", "game_type", "gameday", "gametime", "home_team", "away_team"],
    )
    schedule = schedule.loc[schedule.season.isin(SEASONS) & schedule.game_type.eq("REG")].copy()
    local = pd.to_datetime(schedule.gameday.astype(str) + " " + schedule.gametime.astype(str))
    schedule["kickoff"] = local.dt.tz_localize(ZONE).dt.tz_convert("UTC")
    first = local.dt.normalize().groupby([schedule.season, schedule.week]).transform("min")
    sunday = first + pd.to_timedelta((6 - first.dt.dayofweek) % 7, unit="D")
    for name, offset in {
        "monday": pd.Timedelta(days=-6),
        "tuesday": pd.Timedelta(days=-5, hours=12),
        "thursday_date": pd.Timedelta(days=-3),
        "sunday": pd.Timedelta(hours=12, minutes=30),
    }.items():
        schedule[name] = (sunday + offset).dt.tz_localize(ZONE).dt.tz_convert("UTC")
    games = openers.merge(
        schedule, on=["game_id", "season", "week"], how="left", validate="one_to_one"
    )
    if games.kickoff.isna().any() or not np.isfinite(games.tue_open_home_spread).all():
        raise ValueError("A frozen opener has no regular-season clock or finite line")
    games["in_four_term_artifact"] = games.game_id.isin(base.game_id)
    base_meta = json.loads((BASE / "metadata.json").read_text(encoding="utf-8-sig"))
    opener_meta = json.loads((OPENER / "metadata.json").read_text(encoding="utf-8-sig"))
    if base_meta["base_probability_policy"] != "discrete_conditional_non_push_v1":
        raise ValueError("The frozen base is not the declared discrete probability")
    if Path("artifacts", base_meta["opener_evaluation"]) != OPENER:
        raise ValueError("The base points to a different opener artifact")
    upstream = read_frame(
        UPSTREAM, ["game_id", "season", "method", "model_name", "train_max_gameday"]
    )
    config = opener_meta["active_model_config"]
    upstream = upstream.loc[
        upstream.game_id.isin(games.game_id)
        & upstream.method.eq(config["target"])
        & upstream.model_name.eq(config["regressor"])
    ].copy()
    if upstream.game_id.duplicated().any():
        raise ValueError("Upstream margin predictions are not unique")
    upstream["train_max_gameday"] = pd.to_datetime(upstream.train_max_gameday)
    starts = pd.to_datetime(schedule.groupby("season").gameday.min())
    upstream["season_start"] = upstream.season.map(starts)
    upstream["cutoff_violation"] = (
        upstream.train_max_gameday.isna() | upstream.train_max_gameday.ge(upstream.season_start)
    )
    audit = {
        "opener_games": len(games),
        "four_term_games": len(base),
        "opener_only_games_retained": int((~games.in_four_term_artifact).sum()),
        "weeks": len(games[["season", "week"]].drop_duplicates()),
        "upstream_matched_games": len(upstream),
        "upstream_missing_games": int((~games.game_id.isin(upstream.game_id)).sum()),
        "upstream_cutoff_violations": int(upstream.cutoff_violation.sum()),
        "upstream_train_max_by_season": {
            str(int(season)): str(group.train_max_gameday.max().date())
            for season, group in upstream.groupby("season")
        },
        "base_terms": base_meta["features"],
        "base_move_version": base_meta["market_move_feature_version"],
        "historical_selection_limitation": base_meta["validation_limitations"],
        "inputs": {
            path.as_posix(): digest(path)
            for path in [
                BASE / "per_game.parquet",
                BASE / "metadata.json",
                OPENER / "per_game.parquet",
                OPENER / "metadata.json",
                UPSTREAM,
                SCHEDULE,
                LANE,
            ]
        },
    }
    upstream.to_parquet(OUTPUT / "upstream_cutoffs.parquet", index=False)
    return games, audit


def load_quotes(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    columns = [
        "nflverse_game_id",
        "home_team",
        "away_team",
        "bookmaker_key",
        "market",
        "home_spread_line",
        "line",
        "outcome_side",
        "observed_at_utc",
        "bookmaker_last_update_utc",
        "market_last_update_utc",
        "commence_time_utc",
    ]
    counts = Counter()
    sources, frames = [], []
    for path in sorted(Path("data/market/raw").glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        request = manifest.get("request", {})
        if not (
            manifest.get("provider") == "the-odds-api"
            and manifest.get("capture_kind") == "historical_backfill"
            and request.get("sport") == "americanfootball_nfl"
            and request.get("season") in SEASONS
            and request.get("decision_label") in LABELS
            and "spreads" in str(request.get("markets", "")).split(",")
        ):
            continue
        quote_path = path.parent / "quotes.parquet"
        if not quote_path.is_file():
            counts["missing_quote_files"] += 1
            continue
        expected = manifest.get("files", {}).get("quotes.parquet", {}).get("sha256")
        if not expected or digest(quote_path) != expected:
            raise ValueError(f"Quote digest mismatch: {quote_path}")
        stamp = pd.to_datetime(manifest["snapshot_timestamp_utc"], utc=True)
        stage = LABELS[request["decision_label"]]
        frame = read_frame(quote_path, columns)
        counts["source_files"] += 1
        counts["source_rows"] += len(frame)
        frame = frame.loc[frame.market.eq("spreads")].copy()
        counts["spread_rows"] += len(frame)
        frame = frame.merge(
            games,
            left_on="nflverse_game_id",
            right_on="game_id",
            how="inner",
            validate="many_to_one",
            suffixes=("", "_schedule"),
        )
        target = frame.season.eq(request["season"]) & frame.week.eq(request["week"])
        counts["off_target_week_rows"] += int((~target).sum())
        frame = frame.loc[target].copy()
        if not (
            frame.home_team.eq(frame.home_team_schedule)
            & frame.away_team.eq(frame.away_team_schedule)
        ).all():
            raise ValueError(f"Quote and schedule teams disagree: {path}")
        for name in [
            "observed_at_utc",
            "bookmaker_last_update_utc",
            "market_last_update_utc",
            "commence_time_utc",
        ]:
            frame[name] = pd.to_datetime(frame[name], utc=True, errors="coerce", format="mixed")
        signed = np.where(frame.outcome_side.eq("HOME"), -frame.line, frame.line)
        if not frame.outcome_side.isin(("HOME", "AWAY")).all() or not np.allclose(
            signed, frame.home_spread_line, equal_nan=True, rtol=0, atol=1e-9
        ):
            raise ValueError(f"Quote handicap signs disagree: {path}")
        if stage == "thursday":
            frame["decision_at"] = stamp
            valid_stage = frame.thursday_date.dt.tz_convert(ZONE).dt.date.eq(
                stamp.tz_convert(ZONE).date()
            )
        else:
            frame["decision_at"] = frame[stage]
            valid_stage = pd.Series(True, index=frame.index)
            if stage == "sunday":
                valid_stage &= frame.observed_at_utc.dt.tz_convert(ZONE).dt.date.eq(
                    frame.sunday.dt.tz_convert(ZONE).dt.date
                )
        valid = (
            valid_stage
            & np.isfinite(frame.home_spread_line)
            & frame.bookmaker_key.notna()
            & frame.observed_at_utc.eq(stamp)
            & frame.bookmaker_last_update_utc.le(frame.observed_at_utc)
            & (
                frame.market_last_update_utc.isna()
                | frame.market_last_update_utc.le(frame.observed_at_utc)
            )
            & frame.observed_at_utc.ge(frame.monday)
            & frame.observed_at_utc.le(frame.decision_at)
            & frame.decision_at.lt(frame.kickoff)
            & frame.decision_at.lt(frame.commence_time_utc)
        )
        counts["inadmissible_clock_or_line_rows"] += int((~valid).sum())
        frame = frame.loc[valid].copy()
        frame["stage"] = stage
        frame["source_manifest"] = path.as_posix()
        frames.append(frame)
        sources.append(
            {
                "season": request["season"],
                "week": request["week"],
                "stage": stage,
                "observed_at_utc": stamp,
                "manifest": path.as_posix(),
                "manifest_sha256": digest(path),
                "quotes_sha256": expected,
                "admissible_rows": len(frame),
            }
        )
    if not frames:
        return pd.DataFrame(), pd.DataFrame(sources), dict(counts)
    quotes = pd.concat(frames, ignore_index=True)
    keys = ["season", "week", "game_id", "stage", "bookmaker_key", "observed_at_utc"]
    if quotes.groupby(keys).home_spread_line.nunique().gt(1).any():
        raise ValueError("A book has conflicting paired handicap quotes")
    quotes = quotes.sort_values("bookmaker_last_update_utc").drop_duplicates(keys, keep="last")
    quotes = quotes.sort_values("observed_at_utc").drop_duplicates(keys[:-1], keep="last")
    openers = quotes.loc[
        quotes.stage.eq("tuesday"),
        ["game_id", "bookmaker_key", "home_spread_line", "observed_at_utc"],
    ].rename(
        columns={
            "home_spread_line": "book_opener_line",
            "observed_at_utc": "book_opener_observed_at",
        }
    )
    quotes = quotes.merge(
        openers, on=["game_id", "bookmaker_key"], how="left", validate="many_to_one"
    )
    quotes["stage_move_toward_home"] = quotes.home_spread_line - quotes.book_opener_line
    quotes["move_available"] = quotes.book_opener_line.notna()
    paired = quotes.loc[quotes.move_available]
    if not paired.book_opener_observed_at.le(paired.observed_at_utc).all():
        raise ValueError("A movement uses an opener observed after its stage quote")
    keep = [
        "season",
        "week",
        "game_id",
        "stage",
        "bookmaker_key",
        "home_spread_line",
        "tue_open_home_spread",
        "book_opener_line",
        "stage_move_toward_home",
        "move_available",
        "book_opener_observed_at",
        "observed_at_utc",
        "bookmaker_last_update_utc",
        "market_last_update_utc",
        "decision_at",
        "commence_time_utc",
        "kickoff",
        "source_manifest",
        "in_four_term_artifact",
    ]
    return quotes[keep], pd.DataFrame(sources), dict(counts)


def replay(games: pd.DataFrame, quotes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    witnesses, weekly = [], []
    for (season, week), game_group in games.groupby(["season", "week"]):
        week_quotes = quotes.loc[quotes.season.eq(season) & quotes.week.eq(week)]
        stage_times = {"tuesday": game_group.tuesday.iloc[0], "sunday": game_group.sunday.iloc[0]}
        thursday = week_quotes.loc[
            week_quotes.stage.eq("thursday"), "decision_at"
        ].drop_duplicates()
        if len(thursday) > 1:
            raise ValueError("A target week has multiple Thursday decision captures")
        stage_times["thursday"] = thursday.iloc[0] if len(thursday) else pd.NaT
        entry = {"season": int(season), "week": int(week)}
        stage_games = {}
        for stage in STAGES:
            panel = week_quotes.loc[week_quotes.stage.eq(stage)].copy()
            stage_games[stage] = panel
            entry[f"{stage}_games"] = panel.game_id.nunique()
            entry[f"{stage}_paired_games"] = panel.loc[panel.move_available, "game_id"].nunique()
        entry["three_stage_source_complete"] = all(entry[f"{stage}_games"] > 0 for stage in STAGES)
        entry["paired_week_complete"] = entry["three_stage_source_complete"] and all(
            entry[f"{stage}_paired_games"] > 0 for stage in STAGES
        )
        if not entry["three_stage_source_complete"]:
            weekly.append(entry)
            continue
        if not stage_times["tuesday"] < stage_times["thursday"] < stage_times["sunday"]:
            raise ValueError("Decision stages are not chronological")
        for witness in ("retain_earliest", "retain_option"):
            nominee, lock_at = None, None
            previous_time = None
            for stage in STAGES:
                now = stage_times[stage]
                panel = stage_games[stage]
                choices = panel.groupby("game_id", as_index=False).agg(
                    lock_at=("commence_time_utc", "min"), kickoff=("kickoff", "min")
                )
                choices["lock_at"] = choices[["lock_at", "kickoff"]].min(axis=1)
                choices = choices.sort_values(["lock_at", "game_id"])
                locked = nominee is not None and now >= lock_at
                previous = nominee
                if not locked and (nominee is None or witness == "retain_option"):
                    choice = choices.iloc[0 if witness == "retain_earliest" else -1]
                    nominee, lock_at = choice.game_id, choice.lock_at
                if nominee is None or (locked and nominee != previous):
                    raise ValueError("Nominee is missing or changed after locking")
                if not locked and not now < lock_at:
                    raise ValueError("A new nominee has already kicked off")
                if previous_time is not None and not previous_time < now:
                    raise ValueError("Replay time moved backwards")
                witnesses.append(
                    {
                        "season": int(season),
                        "week": int(week),
                        "witness": witness,
                        "stage": stage,
                        "decision_at": now,
                        "game_id": nominee,
                        "lock_at": lock_at,
                        "locked": locked,
                        "action": "retain_locked" if locked else "nominate_or_retain_unlocked",
                    }
                )
                previous_time = now
            entry[f"{witness}_locked_by_sunday"] = bool(stage_times["sunday"] >= lock_at)
        weekly.append(entry)
    return pd.DataFrame(witnesses), pd.DataFrame(weekly)


def write_report(summary: dict, weeks: pd.DataFrame) -> None:
    audit, counts = summary["population"], summary["sources"]
    rows = []
    if not weeks.empty:
        for season, group in weeks.groupby("season"):
            values = [
                int(season),
                len(group),
                int(group.three_stage_source_complete.sum()),
                int(group.paired_week_complete.sum()),
                int(group.tuesday_games.sum()),
                int(group.thursday_games.sum()),
                int(group.sunday_games.sum()),
            ]
            rows.append("| " + " | ".join(map(str, values)) + " |")
    coverage = "\n".join(rows)
    cutoffs = "\n".join(
        f"| {season} | {cutoff} |"
        for season, cutoff in audit["upstream_train_max_by_season"].items()
    )
    text = f"""# LEAD-89 unit 1 - legal-state and clock replay

**Measured:** `.tools/uv.exe run --no-sync python scripts/lead89_unit1.py`.
Metadata/quote replay only; no outcome columns read, fits or scores.

## Scope and declaration

**Read:** ROADMAP.md:872 separates this legal-state/clock unit from transition fitting
and policy scoring. The unchanged declaration was saved in `docs/lanes/lead89.md`
before execution. Historical openers are the frozen pool-line proxy.

**Read:** outer 2023/2024/2025; fit through Y-3, tune Y-2, calibrate Y-1.
One four-term calibrated discrete-margin probability selects every side.
Weekly Best Pick reward is primary: F=3, B=7, K=6; 717 planned looks.
This unit consumes zero outcome looks.

## Source and clock results

**Measured:** {audit["opener_games"]} opener games / {audit["weeks"]} weeks;
{audit["four_term_games"]} four-term rows;
{audit["opener_only_games_retained"]} additional opener rows retained without outcome filtering.

**Measured:** {counts.get("source_files", 0)} three-stage quote files /
{counts.get("source_rows", 0)} source rows; {counts.get("spread_rows", 0)} spread rows.
Excluded {counts.get("off_target_week_rows", 0)} off-target linked rows and
{counts.get("inadmissible_clock_or_line_rows", 0)} inadmissible clock/line rows.
Missing files: {counts.get("missing_quote_files", 0)}.

**Measured:** {summary["complete_weeks"]} weeks have legal candidates at all three stages;
{summary["paired_weeks"]} also have same-book opener/stage pairs at each stage.
{summary["book_stage_rows"]} book/game/stage rows saved.
These are census counts; confidence intervals do not apply.

| Season | Weeks | Three-stage complete | Paired complete | Tue games | Thu games | Sun games |
| --- | --- | --- | --- | --- | --- | --- |
{coverage}

**Measured:** checks enforce quote digests, target week, teams, handicap signs,
observation/book update and available market update clocks, and quote/schedule kickoffs.
Tuesday uses captures available by noon, Thursday its actual pre-kickoff capture,
and Sunday 12:30. Same-book movement is the stage quote minus the Tuesday anchor;
absent pairs remain unavailable.

**Read:** Tuesday captures are earlier anchors, not evidence of noon captures.
The final schedule is retrospective; quote kickoffs are also enforced.
This does not establish when every schedule revision became known.

## Legal nomination witnesses

**Measured:** {summary["witness_rows"]} structural stage states;
{summary["early_locked_weeks"]} earliest-nominee paths lock before Sunday;
{summary["wait_unlocked_weeks"]} latest-kickoff paths remain unlocked at Sunday.
One nominee is retained after initial nomination; locked nominees never change.
No side or reward is assigned and no future quote chooses a witness.

**Inferred:** these paths establish legality only; they do not price waiting or add
fitted policy variants. Early locked games need no Sunday quote. Replacement occurs
only before the nominee's known kickoff.

## Upstream cutoff and next-unit gates

**Measured:** {audit["upstream_matched_games"]} upstream matches;
{audit["upstream_missing_games"]} missing;
{audit["upstream_cutoff_violations"]} training-cutoff violations against season start.

| Prediction season | Latest upstream training game |
| --- | --- |
{cutoffs}

**Read:** `src/nfl_ats/clv.py:2184` fits opener models on completed games before
each target week's first date, including earlier games in the prediction season.
At lines 2207-2218 it recomputes forecasts with the opener as the model input line.
The companion ledger is not verified direct lineage for those recomputed points.

**Inferred, decision gate:** season-held-out training is not established by these
sources. Unit 2 scoring is blocked until the orchestrator supplies compatible
forecasts, verifies exact opener lineage and checks stage-specific feature timing.
This is not evidence of future-game leakage or a refuted waiting mechanism.
No full-history rebuild was attempted in this shared-resource unit.

**Read:** base terms: `{", ".join(audit["base_terms"])}`.
Movement version: `{audit["base_move_version"]}`.
Stored Sunday probabilities cannot support Tuesday/Thursday decisions.
Unit 2 must reconstruct stage-matched terms, fit/calibrate in the declared blocks,
retain push mass and verify all other term timestamps.

**Read:** frozen metadata limitation: {audit["historical_selection_limitation"]}

**Inferred:** one earlier-season joint weekly transition/resampling law must retain
cross-game dependence and compare early expected reward with the expected best
still-playable reward. Training-cutoff checks alone do not certify stage availability
or untouched model selection.

## Requested scoring fields

Decisive-game record, IS/OOS and gap, fold coefficients/stability,
accuracy/Brier/log loss/RPS, reliability bands, weekly reward, nominee Brier,
season/week-block 95% intervals and `probability_positive`:
**not computed in unit 1**. No score or verdict exists to record.
The mechanism is unadjudicated; AGENTS.md:65-85 forbids closing on an interval
spanning zero.

## Saved output and continuation

`tests/scratch/codex/lead89_unit1/` holds quote panels, source hashes, upstream cutoffs,
weekly coverage, structural state traces and summary JSON. Next is the predeclared
transition-fit/policy-replay unit, with 717 planned looks. The orchestrator records
any eventual research results serially.
"""
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    if "717 looks" not in LANE.read_text(encoding="utf-8-sig"):
        raise ValueError("Save the predeclared lane before replay")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "protocol_before_run.md").write_text(
        LANE.read_text(encoding="utf-8-sig"), encoding="utf-8"
    )
    with threadpool_limits(limits=1):
        games, audit = load_population()
        quotes, sources, counts = load_quotes(games)
        witnesses, weeks = (
            replay(games, quotes) if not quotes.empty else (pd.DataFrame(), pd.DataFrame())
        )
        for name, frame in [
            ("stage_quotes", quotes),
            ("sources", sources),
            ("weekly_coverage", weeks),
            ("legal_states", witnesses),
        ]:
            frame.to_parquet(OUTPUT / f"{name}.parquet", index=False)
        summary = {
            "unit": "legal_state_clock_replay",
            "planned_looks": 717,
            "outcome_looks": 0,
            "population": audit,
            "sources": counts,
            "book_stage_rows": len(quotes),
            "complete_weeks": int(weeks.three_stage_source_complete.sum())
            if not weeks.empty
            else 0,
            "paired_weeks": int(weeks.paired_week_complete.sum()) if not weeks.empty else 0,
            "witness_rows": len(witnesses),
            "early_locked_weeks": int(
                weeks.get("retain_earliest_locked_by_sunday", pd.Series(dtype=bool)).eq(True).sum()
            ),
            "wait_unlocked_weeks": int(
                weeks.get("retain_option_locked_by_sunday", pd.Series(dtype=bool)).eq(False).sum()
            ),
            "scoring_status": "not_started_unit2_required",
        }
        (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(summary, weeks)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in ("population", "sources")}
        )
    )


if __name__ == "__main__":
    main()
