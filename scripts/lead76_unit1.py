from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

HISTORY = tuple(range(1999, 2020))
FOLDS = tuple(range(2020, 2026))
ERAS = (
    (1999, 2011, "2-yard PAT; 15-minute sudden death"),
    (2012, 2014, "2-yard PAT; 15-minute modified sudden death"),
    (2015, 2016, "15-yard PAT; 15-minute modified sudden death"),
    (2017, 2024, "15-yard PAT; 10-minute modified sudden death"),
    (2025, 2025, "15-yard PAT; both teams receive possession, 10-minute limit"),
)


def schedule_inventory(root: Path) -> dict:
    games = {}
    sources = []
    errors = []
    required = {"game_id", "season", "game_type", "home_score", "away_score"}
    for path in sorted(root.glob("**/schedules.parquet")):
        try:
            manifest = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8-sig"))
            expected = manifest["files"]["schedules.parquet"]["sha256"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("schedule digest mismatch")
            columns = set(pq.read_schema(path).names)
            if not required.issubset(columns):
                raise ValueError("schedule schema is missing required fields")
            table = pq.read_table(path, columns=sorted(required), use_threads=False)
            score_present = pc.and_(
                pc.is_valid(table["home_score"]), pc.is_valid(table["away_score"])
            )
            metadata = table.select(["game_id", "season", "game_type"])
            metadata = metadata.append_column("score_present", score_present)
            rows = metadata.to_pylist()
            seasons = sorted({int(row["season"]) for row in rows})
            sources.append(
                {
                    "path": path.as_posix(),
                    "rows": len(rows),
                    "first": min(seasons),
                    "last": max(seasons),
                    "rule_fields": sorted(c for c in columns if "rule" in c or "pat_distance" in c),
                }
            )
            candidates = {}
            for row in rows:
                if row["game_type"] != "REG" or row["season"] not in HISTORY + FOLDS:
                    continue
                game_id = str(row["game_id"])
                previous = games.get(game_id)
                if previous and previous["season"] != row["season"]:
                    raise ValueError("game identity has inconsistent seasons")
                candidates[game_id] = row
            games.update(candidates)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{path.as_posix()}: {type(error).__name__}: {error}")
    scheduled = Counter(row["season"] for row in games.values())
    completed = Counter(row["season"] for row in games.values() if row["score_present"])
    return {"sources": sources, "errors": errors, "scheduled": scheduled, "completed": completed}


def opener_inventory(root: Path) -> dict:
    captures = Counter()
    errors = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if payload.get("source") != "splashsports.com":
                raise ValueError("capture does not name the canonical Splash source")
            captures[int(payload["season"])] += 1
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{path.as_posix()}: {type(error).__name__}: {error}")
    return {"exists": root.is_dir(), "captures": captures, "errors": errors}


def write_report(path: Path, schedules: dict, openers: dict, raw: Path, splash: Path) -> str:
    completed = schedules["completed"]
    scheduled = schedules["scheduled"]
    missing = [season for season in HISTORY if not completed[season]]
    target_captures = sum(openers["captures"][season] for season in FOLDS)
    errors = schedules["errors"] + openers["errors"]
    rule_fields = sorted({c for source in schedules["sources"] for c in source["rule_fields"]})
    if errors:
        status = "inventory_incomplete"
    elif not target_captures:
        status = "source_gated_missing_authentic_openers"
    else:
        status = "inventory_only_source_audit_required"
    lines = [
        "# LEAD-76 unit 1: rules and history inventory",
        "",
        f"**Measured:** `{status}`. Decisive-game record: unavailable; zero games scored.",
        f"**Measured:** {sum(completed[s] for s in HISTORY):,} unique REG score-present "
        f"games in 1999-2019;",
        f"{target_captures} authentic-pool capture files in the six scheduled outer seasons.",
        "Performance intervals and probability_positive are unavailable because no candidate "
        "was fitted or scored.",
        "**Inferred:** LEAD-76 remains open at its source gate; this inventory supplies no "
        "closure evidence.",
        "",
        "## Reproduction and scope",
        "",
        "`.tools/uv.exe run --no-sync python scripts/lead76_unit1.py`",
        "",
        "**Read:** the exact roadmap row and Protocol B were saved in `docs/lanes/lead76.md` "
        "before inventory.",
        "**Read:** `src/nfl_ats/splash_lines.py:343` identifies the canonical pool-capture "
        "directory.",
        "**Read:** Protocol B requires authentic frozen openers and pre-deadline "
        "issuance/ingestion;",
        "bookmaker Tuesday quotes, reconstructed labels and closing lines cannot fill this "
        "source gap.",
        f"**Measured:** scanned `{raw.as_posix()}/**/schedules.parquet` and "
        f"`{splash.as_posix()}/*.json`.",
        "Only schedule identifiers, REG labels, seasons, score-null masks and capture "
        "source/season were inventoried.",
        "No score values were converted into margins, wins, cover labels, fitted terms or "
        "performance metrics.",
        "Schedule snapshots are digest-checked and deduplicated by game ID, retaining the "
        "last snapshot path.",
        "A score-present count is availability only; historical score integrity and game "
        "completion need later validation.",
        "Recent retrieval of old final scores does not certify historical pregame prediction "
        "or quote availability.",
        "",
        "## Historical coverage",
        "",
        "| Season | Unique REG schedule rows | Both scores present |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(f"| {season} | {scheduled[season]} | {completed[season]} |" for season in HISTORY)
    lines += [
        "",
        f"**Measured:** missing declared score-history seasons: "
        f"{', '.join(map(str, missing)) or 'none'}.",
        "",
        "## Rules inventory",
        "",
        "**Inferred:** the following REG rule chronology is an inventory checklist, pending "
        "documentary verification.",
        "PAT refers to the snap spot for the one-point kick. No rule dates or categories "
        "were fitted to outcomes.",
        f"**Measured:** candidate rules fields in inspected schedule schemas: "
        f"{', '.join(rule_fields) or 'none'}.",
        "No documentary rules source was verified or downloaded in this unit.",
        "",
        "| Proposed era | Rule checklist | Score-present REG games |",
        "| --- | --- | ---: |",
    ]
    for start, end, label in ERAS:
        count = sum(completed[season] for season in range(start, end + 1))
        lines.append(f"| {start}-{end} | {label} | {count} |")
    lines += [
        "",
        "**Read:** the declaration requires an unseen rules regime to inherit the pooled prior.",
        "**Inferred:** after documentary verification, the 2025 rules category would require "
        "that fallback.",
        "The 1999-2019 score-only prior must remain distinct from later earlier-season "
        "fitting data.",
        "",
        "## Scheduled folds and unavailable results",
        "",
        "| Outer season | REG rows | Score-present rows | Pool capture files | Coefficients |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    lines.extend(
        f"| {season} | {scheduled[season]} | {completed[season]} | "
        f"{openers['captures'][season]} | Not fitted |"
        for season in FOLDS
    )
    lines += [
        "",
        "**Measured:** zero fits, zero scores, zero predictive looks; the unchanged full "
        "protocol reserves 361 looks",
        "from B=4, F=6, E=1: (4 + 36 + 8)(6 + 1) + 25.",
        "For pooled-prior, rule-conditioned-prior, probability-term and joint-fit arms, all "
        "six folds remain unavailable.",
        "Combined-model, model-only, market-only and Elo paired comparisons were not run.",
        "IS and OOS accuracy, Brier, log loss, margin MAE, whole-margin log score and their "
        "gaps are unavailable.",
        "Season-block 95% intervals, probability_positive, fold coefficients, season "
        "stability and five-band",
        "training-quantile reliability tables are unavailable. An unmeasured probability is "
        "not reported as 0.5.",
        "Coverage is a census of these local files; a statistical effect interval is not "
        "applicable.",
        "",
        "## Source accounting",
        "",
        "| Verified schedule file | Rows | Seasons |",
        "| --- | ---: | --- |",
    ]
    lines.extend(
        f"| `{source['path']}` | {source['rows']} | {source['first']}-{source['last']} |"
        for source in schedules["sources"]
    )
    lines += [
        "",
        "**Measured:** pool capture files by season: "
        + (", ".join(f"{s}: {n}" for s, n in sorted(openers["captures"].items())) or "none")
        + ".",
        f"**Measured:** {len(errors)} input errors. Canonical pool directory exists: "
        f"{openers['exists']}.",
    ]
    lines.extend(f"- {error}" for error in errors)
    lines += [
        "",
        "## Next unit",
        "",
        "Obtain and verify the missing score histories, documentary REG rules chronology, and",
        "2020-2025 authentic frozen-pool opener captures with deadline provenance before "
        "cached replay.",
        "Audit certified pregame predictions and earlier training/selection/calibration "
        "partitions once source coverage exists.",
        "Keep the declared folds, terms, endpoints and 361-look budget; do not replace "
        "missing sources with proxy grades.",
        "No effect or uncertainty was estimated; no statistical record command is prepared.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory LEAD-76 rules and historical sources without fitting or scoring."
    )
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--splash", type=Path, default=Path("data/splash"))
    parser.add_argument("--report", type=Path, default=Path("docs/lead76_unit1_inventory.md"))
    args = parser.parse_args()
    schedules = schedule_inventory(args.raw)
    openers = opener_inventory(args.splash)
    status = write_report(args.report, schedules, openers, args.raw, args.splash)
    print(f"LEAD-76: {status}; report={args.report.as_posix()}")
    print(
        f"history_games={sum(schedules['completed'][s] for s in HISTORY)}; "
        f"target_capture_files={sum(openers['captures'][s] for s in FOLDS)}; "
        f"input_errors={len(schedules['errors']) + len(openers['errors'])}; predictive_looks=0/361"
    )


if __name__ == "__main__":
    main()
