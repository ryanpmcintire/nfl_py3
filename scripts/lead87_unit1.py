from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from nfl_ats.sharp_book_movement_features import LEADER_BOOKS

SCHEDULE = Path("data/raw/20260908T162105Z/schedules.parquet")
BASE = Path("artifacts/pick_probability/20260929T192747Z")
UPSTREAM = Path("artifacts/margins/20260929T192312Z")
OUTPUT = Path("tests/scratch/codex/lead87_unit1")
REPORT = Path("docs/lead87_unit1.md")
SEASONS = tuple(range(2020, 2026))
LABELS = ("tue_open", "sun_early_close")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_frame(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def games_and_bounds() -> pd.DataFrame:
    columns = [
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "gametime",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]
    games = read_frame(SCHEDULE, columns)
    games = games.loc[games.season.isin(SEASONS)].copy()
    if games.game_id.duplicated().any():
        raise ValueError("Schedule game identifiers are not unique")
    local = pd.to_datetime(games.gameday.astype(str) + " " + games.gametime.astype(str))
    games["kickoff"] = local.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    dates = pd.to_datetime(games.gameday)
    first = dates.groupby([games.season, games.week]).transform("min")
    sunday = first + pd.to_timedelta((6 - first.dt.dayofweek) % 7, unit="D")
    for name, offset in (
        ("monday", pd.Timedelta(days=-6)),
        ("freeze", pd.Timedelta(days=-5, hours=12)),
        ("sunday", pd.Timedelta(0)),
        ("deadline", pd.Timedelta(hours=12, minutes=45)),
    ):
        games[name] = (sunday + offset).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    games["deadline"] = games[["kickoff", "deadline"]].min(axis=1)
    games["actual_margin"] = games.home_score - games.away_score
    if games[["kickoff", "actual_margin"]].isna().any().any():
        raise ValueError("A historical game lacks a kickoff or final score")
    return games


def source_quotes(games: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict]:
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
        "commence_time_utc",
    ]
    sources, frames = [], []
    counts = Counter()
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
        frame = read_frame(quote_path, columns)
        counts["source_rows"] += len(frame)
        frame = frame.loc[frame.market.eq("spreads")].copy()
        counts["spread_rows"] += len(frame)
        for name in ("observed_at_utc", "bookmaker_last_update_utc", "commence_time_utc"):
            frame[name] = pd.to_datetime(frame[name], utc=True, errors="coerce", format="mixed")
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
            raise ValueError(f"Quote teams disagree with schedule: {path}")
        finite = np.isfinite(pd.to_numeric(frame.home_spread_line, errors="coerce"))
        signed = np.where(frame.outcome_side.eq("HOME"), -frame.line, frame.line)
        if not frame.outcome_side.isin(("HOME", "AWAY")).all() or not np.allclose(
            signed,
            frame.home_spread_line,
            atol=1e-9,
            rtol=0,
            equal_nan=True,
        ):
            raise ValueError(f"Quote handicap convention mismatch: {path}")
        valid = (
            finite
            & frame.observed_at_utc.eq(stamp)
            & frame.bookmaker_last_update_utc.le(frame.observed_at_utc)
            & frame.observed_at_utc.lt(frame.kickoff)
            & frame.observed_at_utc.lt(frame.commence_time_utc)
            & frame.observed_at_utc.ge(frame.monday)
        )
        if request["decision_label"] == "tue_open":
            valid &= frame.observed_at_utc.le(frame.freeze)
        else:
            valid &= frame.observed_at_utc.ge(frame.sunday) & frame.observed_at_utc.lt(
                frame.deadline
            )
        counts["inadmissible_clock_or_line_rows"] += int((~valid).sum())
        frame = frame.loc[valid].copy()
        frame["decision_label"] = request["decision_label"]
        frame["source_manifest"] = path.as_posix()
        frames.append(frame)
        sources.append(
            {
                "manifest": path.as_posix(),
                "manifest_sha256": digest(path),
                "quotes_sha256": expected,
                "season": request["season"],
                "week": request["week"],
                "decision_label": request["decision_label"],
                "admissible_rows": len(frame),
            }
        )
    if not frames:
        return pd.DataFrame(), sources, dict(counts)
    quotes = pd.concat(frames, ignore_index=True)
    keys = ["game_id", "bookmaker_key", "decision_label", "observed_at_utc"]
    if quotes.groupby(keys).home_spread_line.nunique().gt(1).any():
        raise ValueError("A book has conflicting spread lines at a capture")
    quotes = quotes.sort_values([*keys, "bookmaker_last_update_utc"]).drop_duplicates(
        keys, keep="last"
    )
    quotes = quotes.drop_duplicates(keys[:-1], keep="last")
    counts["admissible_unique_book_captures"] = len(quotes)
    return quotes, sources, dict(counts)


def join_pairs(quotes: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["game_id", "bookmaker_key"]
    evidence = [
        "home_spread_line",
        "observed_at_utc",
        "bookmaker_last_update_utc",
        "source_manifest",
    ]
    early = quotes.loc[quotes.decision_label.eq(LABELS[0]), keys + evidence]
    late = quotes.loc[quotes.decision_label.eq(LABELS[1]), keys + evidence]
    pairs = early.merge(late, on=keys, validate="one_to_one", suffixes=("_open", "_late"))
    pairs["move_toward_home"] = pairs.home_spread_line_late - pairs.home_spread_line_open
    pairs["leader"] = pairs.bookmaker_key.isin(LEADER_BOOKS)
    if not pairs.observed_at_utc_open.lt(pairs.observed_at_utc_late).all():
        raise ValueError("Late capture does not follow opener")
    aggregated = pairs.groupby("game_id").agg(
        paired_books=("bookmaker_key", "size"),
        all_book_move=("move_toward_home", "median"),
        latest_observed_at=("observed_at_utc_late", "max"),
    )
    leaders = (
        pairs.loc[pairs.leader]
        .groupby("game_id")
        .agg(
            leader_books=("bookmaker_key", "size"),
            market_move_toward_home=("move_toward_home", "median"),
        )
    )
    openers = early.groupby("game_id").agg(
        archived_opener=("home_spread_line", "median"),
        opener_books=("bookmaker_key", "size"),
    )
    joined = games.merge(openers, on="game_id", how="left", validate="one_to_one")
    joined = joined.merge(aggregated, on="game_id", how="left", validate="one_to_one")
    joined = joined.merge(leaders, on="game_id", how="left", validate="one_to_one")
    joined["paired_books"] = joined.paired_books.fillna(0).astype(int)
    joined["leader_books"] = joined.leader_books.fillna(0).astype(int)
    return pairs, joined


def regular_base(joined: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8-sig"))
    if metadata["base_probability_policy"] != "discrete_conditional_non_push_v1":
        raise ValueError("Frozen regular-season base is not the declared discrete probability")
    columns = [
        "game_id",
        "season",
        "week",
        "game_type",
        "tue_open_home_spread",
        "model_logit",
        "model_probability",
        "composition_flag_sum",
        "market_move_available",
        "market_move_toward_home",
        "home_covered",
        "margin_vs_open",
    ]
    base = read_frame(BASE / "per_game.parquet", columns)
    if base.game_id.duplicated().any() or not base.game_type.eq("REG").all():
        raise ValueError("Frozen base is not unique regular-season games")
    if not base.season.isin(SEASONS).all():
        raise ValueError("Frozen base is outside the declared seasons")
    recover = joined[
        [
            "game_id",
            "actual_margin",
            "archived_opener",
            "paired_books",
            "leader_books",
            "market_move_toward_home",
            "latest_observed_at",
            "kickoff",
        ]
    ].rename(columns={"market_move_toward_home": "reconstructed_move_toward_home"})
    base = base.merge(recover, on="game_id", how="left", validate="one_to_one")
    if not np.allclose(base.actual_margin - base.tue_open_home_spread, base.margin_vs_open):
        raise ValueError("Frozen opener labels do not reproduce schedule margins")
    if not base.home_covered.eq(base.margin_vs_open.gt(0)).all() or base.margin_vs_open.eq(0).any():
        raise ValueError("Frozen conditional cover labels disagree")
    base["reconstructed_move_available"] = base.leader_books.gt(0)
    opener_root = Path("artifacts") / metadata["opener_evaluation"]
    opener_metadata = json.loads((opener_root / "metadata.json").read_text(encoding="utf-8-sig"))
    config = opener_metadata["active_model_config"]
    upstream = read_frame(
        UPSTREAM / "predictions.parquet",
        [
            "game_id",
            "season",
            "gameday",
            "train_max_gameday",
            "method",
            "model_name",
        ],
    )
    upstream = upstream.loc[
        upstream.game_id.isin(base.game_id)
        & upstream.method.eq(config["target"])
        & upstream.model_name.eq(config["regressor"])
    ].copy()
    if upstream.game_id.duplicated().any():
        raise ValueError("Upstream margin predictions are not unique")
    upstream["train_max_gameday"] = pd.to_datetime(upstream.train_max_gameday)
    season_start = joined.loc[joined.game_type.eq("REG")].groupby("season").gameday.min()
    upstream["season_start"] = pd.to_datetime(upstream.season.map(season_start))
    within_season = int(upstream.train_max_gameday.ge(upstream.season_start).sum())
    violations = int(
        (
            upstream.train_max_gameday.isna()
            | upstream.train_max_gameday.ge(pd.to_datetime(upstream.gameday))
        ).sum()
    )
    if violations:
        raise ValueError("Upstream training reaches a prediction game")
    audit = {
        "base_games": len(base),
        "base_sha256": digest(BASE / "per_game.parquet"),
        "base_model_id": metadata["active_model_id"],
        "base_policy": metadata["base_probability_policy"],
        "upstream_matched_games": len(upstream),
        "upstream_cutoff_violations": violations,
        "upstream_within_season_training_games": within_season,
        "upstream_train_max_by_season": {
            str(season): str(group.train_max_gameday.max().date())
            for season, group in upstream.groupby("season")
        },
        ("upstream_note"): (
            "Companion margin cutoffs audited; opener-specific lineage still needs replay "
            "verification."
        ),
    }
    upstream.to_parquet(OUTPUT / "upstream_cutoffs.parquet", index=False)
    return base, audit


def write_report(summary: dict) -> None:
    rows = [
        "# LEAD-87 unit 1: playoff clock and label join",
        "",
        "**Measured:** `.tools/uv.exe run --no-sync python scripts/lead87_unit1.py` completed.",
        "**Read:** the immutable declaration is in `docs/lead87_protocol.md`; it was copied",
        "into `docs/lanes/lead87.md` before source inventory or outcome access.",
        "",
        "## Scope and decisive record",
        "",
        "This executes the row's first unit: playoff clock/label join. No probability was fitted",
        (
            "or scored, and no side was selected. Decisive record, IS/OOS results and gap, "
            "coefficients,"
        ),
        (
            "95% effect intervals, and probability_positive are **unmeasured**, pending "
            "joint-likelihood replay."
        ),
        (
            "**Read:** the complete replay declares 497 looks; this join consumes zero fit/score "
            "looks."
        ),
        "",
        "## Joined source inventory",
        "",
        (
            "**Measured:** counts below are complete local inventories, not estimates; sampling "
            "intervals do not apply."
        ),
        "",
        (
            "| Season | POST scheduled | POST paired | POST leader paired | POST pushes | POST "
            "leader nonpush | REG frozen | REG paired | REG leader paired |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["seasons"]:
        rows.append(
            "| "
            + " | ".join(
                str(row[key])
                for key in (
                    "season",
                    "post_scheduled",
                    "post_paired",
                    "post_leader_paired",
                    "post_pushes",
                    "post_leader_nonpush",
                    "reg_frozen",
                    "reg_paired",
                    "reg_leader_paired",
                )
            )
            + " |"
        )
    rows += [
        "",
        (
            f"**Measured:** {summary['source_files']} verified quote files; "
            f"{summary['book_pairs']} unique same-book pairs."
        ),
        (
            f"**Measured:** {summary['post_paired']} paired postseason games; "
            f"{summary['post_unpaired']} lack an admissible Tuesday/Sunday pair."
        ),
        (
            "The latter remain in the saved join with missing moves; missing values are never "
            "converted to zero."
        ),
        (
            "Pushes remain in the join and are excluded only by the separately saved "
            "conditional-fit eligibility flag."
        ),
        "",
        (
            "**Measured:** quote hashes, target season/week, team identity, signed handicap, "
            "bookmaker update"
        ),
        (
            "at or before observation, observation before both recorded kickoffs, Tuesday "
            "observation by noon,"
        ),
        (
            "and Sunday observation before min(kickoff, 12:45 ET) were checked. Late quotes are "
            "paired within"
        ),
        (
            "book before taking a median. Tuesday anchors precede noon; they are not "
            "noon-capture evidence."
        ),
        (
            "The archive label sun_early_close names a pre-kick capture; no final closing-line "
            "column is read."
        ),
        (
            "Postseason frozen line is the median of eligible Tuesday books. REG keeps its "
            "original frozen opener."
        ),
        (
            "Leader moves use the served recipe's declared bookmaker set. All-book moves are "
            "saved as source"
        ),
        (
            "evidence only, without creating another fitted arm. No postseason model or flag "
            "columns are fabricated."
        ),
        "",
        "## Chronological partitions",
        "",
        (
            "**Measured:** counts below are available postseason conditional-fit rows; later "
            "seasons never enter an earlier partition."
        ),
        "",
        (
            "| Outer season | Fit through | POST fit | Tune season | POST tune | Calibration "
            "season | POST calibration | REG outer source-complete |"
        ),
        "| --- | --- | ---: | --- | ---: | --- | ---: | ---: |",
    ]
    for row in summary["folds"]:
        rows.append(
            "| "
            + " | ".join(
                str(row[key])
                for key in (
                    "outer",
                    "fit_through",
                    "post_fit",
                    "tune",
                    "post_tune",
                    "calibrate",
                    "post_calibrate",
                    "reg_outer",
                )
            )
            + " |"
        )
    audit = summary["base_audit"]
    rows += [
        "",
        (
            f"**Measured:** {audit['upstream_matched_games']}/{audit['base_games']} frozen REG "
            f"games match companion margin-cache cutoff rows;"
        ),
        (
            f"{audit['upstream_cutoff_violations']} cutoffs reach their own game date. Last "
            f"training dates by prediction season:"
        ),
        "; ".join(
            f"{season}: {date}" for season, date in audit["upstream_train_max_by_season"].items()
        )
        + ".",
        (
            f"**Measured:** {audit['upstream_within_season_training_games']} companion "
            f"predictions train on earlier games in their own season."
        ),
        (
            "**Inferred:** pregame chronology is verified for the matched companion cache; "
            "strict season exclusion"
        ),
        (
            "and exact opener-probability lineage are not established. The replay must resolve "
            "this before"
        ),
        (
            "claiming the prescribed outer-season result and preserve distinct fit, tune, "
            "calibration, and outer periods."
        ),
        "",
        "## Next unit and handoff",
        "",
        (
            "Implement the declared single joint likelihood: retain every REG term, add a "
            "POST-only intercept,"
        ),
        (
            "and share only the move coefficient with the auxiliary market-only logit. Keep all "
            "headline"
        ),
        (
            "grades on the REG population and compare the four predeclared baselines. No "
            "research verdict"
        ),
        "or registry command is warranted by source availability alone; no signal is closed.",
        (
            "The full replay still owes decisive records, IS/OOS/gaps, per-fold coefficients, "
            "four endpoints,"
        ),
        (
            "season/week-block intervals, probability_positive, and five fixed-width reliability "
            "bands."
        ),
        (
            "**Read:** AGENTS.md requires one fitted calibrated probability to select a side; a "
            "zero-crossing"
        ),
        (
            "interval cannot close a signal. No serving or promotion decision is made by this "
            "inventory."
        ),
        "",
        (
            "**Measured:** detailed joins, book evidence, folds, source hashes and cutoff audit "
            "are saved only under"
        ),
        "`tests/scratch/codex/lead87_unit1/`; no prediction-row dumps are written to docs.",
    ]
    REPORT.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = Path("docs/lead87_protocol.md")
    if not protocol.is_file() or "497 looks" not in protocol.read_text(encoding="utf-8"):
        raise ValueError("The predeclared protocol must exist before inventory")
    games = games_and_bounds()
    quotes, sources, counters = source_quotes(games)
    if quotes.empty:
        raise ValueError("No local timestamp-eligible NFL Tuesday/Sunday quotes")
    pairs, joined = join_pairs(quotes, games)
    base, audit = regular_base(joined)
    post = joined.loc[~joined.game_type.eq("REG")].copy()
    post["margin_vs_open"] = post.actual_margin - post.archived_opener
    post["push"] = post.margin_vs_open.eq(0) & post.archived_opener.notna()
    post["home_covered"] = post.margin_vs_open.gt(0).where(post.archived_opener.notna())
    post["conditional_fit_eligible"] = (
        post.leader_books.gt(0) & post.archived_opener.notna() & ~post.push
    )
    post["market_move_available"] = post.leader_books.gt(0)
    seasons = []
    for season in SEASONS:
        p = post.loc[post.season.eq(season)]
        r = base.loc[base.season.eq(season)]
        seasons.append(
            {
                "season": season,
                "post_scheduled": len(p),
                "post_paired": int(p.paired_books.gt(0).sum()),
                "post_leader_paired": int(p.leader_books.gt(0).sum()),
                "post_pushes": int((p.paired_books.gt(0) & p.push).sum()),
                "post_leader_nonpush": int(p.conditional_fit_eligible.sum()),
                "reg_frozen": len(r),
                "reg_paired": int(r.paired_books.gt(0).sum()),
                "reg_leader_paired": int(r.leader_books.gt(0).sum()),
            }
        )
    eligible = post.loc[post.conditional_fit_eligible]
    folds = []
    for outer in (2023, 2024, 2025):
        folds.append(
            {
                "outer": outer,
                "fit_through": outer - 3,
                "post_fit": int(eligible.season.le(outer - 3).sum()),
                "tune": outer - 2,
                "post_tune": int(eligible.season.eq(outer - 2).sum()),
                "calibrate": outer - 1,
                "post_calibrate": int(eligible.season.eq(outer - 1).sum()),
                "reg_outer": int((base.season.eq(outer) & base.reconstructed_move_available).sum()),
            }
        )
    summary = {
        "unit": "clock_label_join",
        "protocol_sha256": digest(protocol),
        "schedule_sha256": digest(SCHEDULE),
        "source_files": len(sources),
        "book_pairs": len(pairs),
        "post_paired": int(post.paired_books.gt(0).sum()),
        "post_unpaired": int(post.paired_books.eq(0).sum()),
        "counts": counters,
        "seasons": seasons,
        "folds": folds,
        "base_audit": audit,
        "declared_replay_looks": 497,
        "computed_fit_score_looks": 0,
        "decisive_record": None,
        "in_sample": None,
        "out_of_sample": None,
        "gap": None,
        "coefficients": None,
        "interval": None,
        "probability_positive": None,
        "status": "join_complete_replay_pending",
        "record_commands": [],
    }
    pairs.to_parquet(OUTPUT / "book_pairs.parquet", index=False)
    post.to_parquet(OUTPUT / "postseason_join.parquet", index=False)
    base.to_parquet(OUTPUT / "regular_base_join.parquet", index=False)
    joined.to_parquet(OUTPUT / "all_games_join.parquet", index=False)
    pd.DataFrame(folds).to_csv(OUTPUT / "fold_inventory.csv", index=False)
    (OUTPUT / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "status",
                    "source_files",
                    "book_pairs",
                    "post_paired",
                    "post_unpaired",
                    "seasons",
                    "folds",
                    "computed_fit_score_looks",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
