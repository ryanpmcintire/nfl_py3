from __future__ import annotations

import json
from pathlib import Path

import lead88_unit2 as unit2
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

ROOT = Path("tests/scratch/codex/lead88_unit3")
LANE = Path("docs/lanes/lead88.md")
REPORT = Path("docs/lead88_unit3.md")
BASELINE = unit2.ROOT / "baseline.parquet"
PRIOR_SUMMARY = unit2.ROOT / "summary.json"
SEASONS = unit2.SEASONS
DRAWS = 10000


def cell(total, margin):
    return int(np.floor((total + margin) / 2)), int(np.floor((total - margin) / 2))


def predict(frame, prior, coefficient):
    rows = []
    for row in frame.itertuples():
        delta = coefficient * row.total_move
        total = row.served_centre_total + delta
        crossed = cell(total, row.centre_margin) != cell(row.served_centre_total, row.centre_margin)
        scores = row.guess_home, row.guess_away
        declined = False
        if delta != 0 and crossed:
            proposed = unit2.choose_score(
                prior[(row.season, row.week)], row, total, allow_missing=True
            )
            declined = not np.isfinite(proposed).all()
            if not declined:
                scores = proposed
        rows.append((*scores, delta, crossed, declined))
    result = pd.DataFrame(
        rows, index=frame.index, columns=["home", "away", "delta", "crossed", "declined"]
    )
    unchanged = result.delta.eq(0) | ~result.crossed
    if not np.array_equal(
        result.loc[unchanged, ["home", "away"]].to_numpy(),
        frame.loc[unchanged, ["guess_home", "guess_away"]].to_numpy(),
    ):
        raise ValueError("Zero adjustment or unchanged cell changed the served score")
    return result


def replay(frame, prior, previous):
    folds = []
    for year in SEASONS:
        train, held = frame.loc[frame.season.ne(year)], frame.loc[frame.season.eq(year)]
        coefficient = unit2.fit_response(train)
        old = next(fold["b"] for fold in previous["folds"] if fold["season"] == year)
        if coefficient != old:
            raise ValueError("Response fit differs from Unit 2")
        scores = predict(held, prior, coefficient)
        for column in scores:
            frame.loc[held.index, "candidate_" + column] = scores[column]
        frame.loc[held.index, "candidate"] = scores.home + scores.away
        frame.loc[held.index, "response_coefficient"] = coefficient
        folds.append({"season": year, "train": len(train), "held": len(held), "b": coefficient})
        print(f"Completed held season {year}", flush=True)
    coefficient = unit2.fit_response(frame)
    if coefficient != previous["full_sample_b"]:
        raise ValueError("Full-data diagnostic fit differs from Unit 2")
    optimistic = predict(frame, prior, coefficient)
    frame["candidate_is"] = optimistic.home + optimistic.away
    for arm in ("served", "candidate", "candidate_is"):
        frame[arm + "_error"] = (frame[arm] - frame.actual_total).abs()
    zero = predict(frame, prior, 0.0)
    if not np.array_equal(zero[["home", "away"]], frame[["guess_home", "guess_away"]]):
        raise ValueError("Zero-coefficient identity failed")
    margin = frame.candidate_home - frame.candidate_away
    admissible = np.where(
        frame.pick_side.eq("HOME"), margin.gt(frame.spread_line), margin.lt(frame.spread_line)
    )
    if not admissible.all() or not np.isfinite(frame.candidate).all():
        raise ValueError("Candidate violates fixed-side or finite-score contract")
    return folds, coefficient


def exact_counts(frame):
    gain = frame.served_error - frame.candidate_error
    closer, worse, tied = (int(getattr(gain, op)(0).sum()) for op in ("gt", "lt", "eq"))
    decisive = closer + worse
    exact = binomtest(closer, decisive, p=0.5) if decisive else None
    share = binomtest(closer, len(frame)).proportion_ci(method="exact")
    conditional = exact.proportion_ci(method="exact") if exact else None
    return {
        "closer": closer,
        "worse": worse,
        "tied": tied,
        "decisive_share": closer / decisive if decisive else None,
        "decisive_interval": list(conditional) if conditional else None,
        "unconditional_share": closer / len(frame),
        "unconditional_interval": list(share),
        "exact_two_sided_p": float(exact.pvalue) if exact else 1.0,
    }


def panel(frame, rng):
    blocks = frame.groupby(["season", "week"], sort=True)
    counts = blocks.size()
    arms = ("served", "candidate", "candidate_is")
    sums = blocks[[arm + "_error" for arm in arms]].sum()
    weights = np.zeros((DRAWS, len(counts)), dtype=np.int16)
    for year in sorted(frame.season.unique()):
        selected = np.flatnonzero(counts.index.get_level_values("season") == year)
        weights[:, selected] = rng.multinomial(
            len(selected), np.full(len(selected), 1 / len(selected)), DRAWS
        )
    errors = (weights @ sums.to_numpy()) / (weights @ counts.to_numpy())[:, None]
    improvement = errors[:, 0] - errors[:, 1]
    in_sample = errors[:, 0] - errors[:, 2]
    effect = float((frame.served_error - frame.candidate_error).mean())
    is_effect = float((frame.served_error - frame.candidate_is_error).mean())
    return {
        "n": len(frame),
        "blocks": len(counts),
        "arms": {
            arm: {
                "mae": float(frame[arm + "_error"].mean()),
                "interval": unit2.interval(errors[:, i]),
            }
            for i, arm in enumerate(arms)
        },
        "effect": effect,
        "interval": unit2.interval(improvement),
        "standard_error": float(np.std(improvement, ddof=1)),
        "probability_positive": float(np.mean(improvement > 0) + 0.5 * np.mean(improvement == 0)),
        "is_effect": is_effect,
        "is_interval": unit2.interval(in_sample),
        "gap": effect - is_effect,
        "gap_interval": unit2.interval(improvement - in_sample),
        "counts": exact_counts(frame),
    }


def record_command(population, result):
    meaning = "the final game of the week" if population == "last" else "all scored games"
    parts = [
        ".tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record",
        f"--name lead88_unit3_{population}_mae --family lead88_total_response",
        f"--description 'Changing totals versus the current score guess on {meaning}; 2 looks'",
        "--source docs/lead88_unit3.md --league nfl --season-start 2020 --season-end 2025",
        f"--effect {result['effect']:.12g} --effect-units mae_improvement",
        f"--interval-low {result['interval'][0]:.12g} --interval-high {result['interval'][1]:.12g}",
        f"--standard-error {result['standard_error']:.12g}",
        f"--probability-positive {result['probability_positive']:.12g}",
        f"--sample-games {result['n']} --sample-blocks {result['blocks']}",
        "--classification unresolved_below_power",
        "--classification-evidence 'Two declared looks; no refuted mechanism or powered "
        "control; pending review'",
        f"--plain-summary 'For {meaning}, the score guess follows changes in sportsbook "
        "totals only far enough to change a whole score cell. No change in the total "
        "leaves the current guess alone. These results do not settle whether this helps; "
        "keep the current guess.'",
    ]
    return " ".join(parts)


def metric(result, key="effect", bounds="interval"):
    return unit2.fmt(result[key], result[bounds])


def write_report(payload, declaration):
    results, facts = payload["results"], payload["facts"]
    lines = [
        "# LEAD-88 unit 3: isolate the total-news adjustment",
        "",
        "**Measured:** one LOSO replay, two declared candidate-versus-served looks.",
        "Command: .tools/uv.exe run --no-sync python scripts/lead88_unit3.py with local "
        "UV_CACHE_DIR.",
        "The frozen declaration below preceded outcome access; full declaration and "
        "hashes are in scratch.",
        "",
        "## Closer, worse and tied",
        "",
        "**Measured:** exact two-sided null is 0.5 among games with unequal absolute errors.",
        "Ties stay in MAE and the unconditional closer share; exact intervals are "
        "Clopper-Pearson 95%.",
    ]
    count_rows, mae_rows, gap_rows = [], [], []
    for name in ("last", "all"):
        result = results[name]
        count = result["counts"]
        conditional = (
            metric(count, "decisive_share", "decisive_interval")
            if count["decisive_share"] is not None
            else "undefined"
        )
        count_rows.append(
            [
                name,
                count["closer"],
                count["worse"],
                count["tied"],
                conditional,
                metric(count, "unconditional_share", "unconditional_interval"),
                unit2.fmt(count["exact_two_sided_p"]),
            ]
        )
        mae_rows.append(
            [
                name,
                result["n"],
                result["blocks"],
                *[metric(result["arms"][arm], "mae") for arm in ("served", "candidate")],
                metric(result),
                unit2.fmt(result["probability_positive"]),
            ]
        )
        gap_rows.append(
            [
                name,
                metric(result["arms"]["candidate_is"], "mae"),
                metric(result, "is_effect", "is_interval"),
                unit2.fmt(result["effect"]),
                metric(result, "gap", "gap_interval"),
            ]
        )
    lines += [
        unit2.table(
            [
                "Population",
                "Closer",
                "Worse",
                "Tied",
                "Decisive share [95% CI]",
                "Unconditional closer share [95% CI]",
                "Exact p",
            ],
            count_rows,
        ),
        "",
        "## MAE and generalization",
        "",
        "**Measured:** positive improvement means a smaller total-score error than the "
        "served guess.",
        "Intervals use 10,000 paired season-stratified week-block draws, seed 88; "
        "probability_positive gives ties half weight.",
        unit2.table(
            [
                "Population",
                "Games",
                "Weeks",
                "Served MAE [95% CI]",
                "Candidate OOS MAE [95% CI]",
                "MAE improvement [95% CI]",
                "probability_positive",
            ],
            mae_rows,
        ),
        "",
        unit2.table(
            [
                "Population",
                "Optimistic IS MAE [95% CI]",
                "IS improvement [95% CI]",
                "OOS improvement",
                "OOS minus IS improvement [95% CI]",
            ],
            gap_rows,
        ),
        "",
        "**Inferred:** these measurements do not establish a serving change or settle "
        "the mechanism.",
        "Proposed classification is unresolved_below_power, pending serial orchestrator recording.",
        "AGENTS.md permits closure only for a refuted mechanism or a positive control "
        "with sufficient power; neither is established here.",
        "",
        "## Fold coefficients and season stability",
        "",
        "**Measured:** full-data optimistic coefficient "
        f"b={payload['full_sample_b']:.9f}; all six LOSO slopes exactly reproduce Unit 2.",
    ]
    rows = []
    for fold in payload["folds"]:
        year = fold["season"]
        last, all_games = (
            payload["seasons"][str(year)]["last"],
            payload["seasons"][str(year)]["all"],
        )
        rows.append(
            [
                year,
                fold["train"],
                fold["held"],
                unit2.fmt(fold["b"]),
                metric(last),
                unit2.fmt(last["probability_positive"]),
                metric(all_games),
                unit2.fmt(all_games["probability_positive"]),
            ]
        )
    lines += [
        unit2.table(
            [
                "Held season",
                "Train",
                "Held",
                "b",
                "Last improvement [95% CI]",
                "Last probability_positive",
                "All improvement [95% CI]",
                "All probability_positive",
            ],
            rows,
        ),
        "",
        "**Measured:** frozen base-probability coefficients below are reused, never "
        "refitted or changed by total news.",
    ]
    coefficients = payload["base_fold_coefficients"]
    terms = list(coefficients[str(SEASONS[0])])
    lines += [
        unit2.table(
            ["Held season", *terms],
            [
                [year, *[unit2.fmt(coefficients[str(year)][term]) for term in terms]]
                for year in SEASONS
            ],
        ),
        "",
        "## Mapping checks and coverage",
        "",
        f"**Measured:** {facts['paired_games']} paired rows retained; "
        f"{facts['scored_games']} scored, {facts['last_games']} last games; "
        f"{facts['declined_baselines']} production-declined baselines remain unscored.",
        f"All {facts['restored_rows']} restored four-term rows remain. Zero coefficient "
        "reproduced both served team scores on "
        f"{facts['zero_coefficient_identity_games']} games.",
        f"Of {facts['zero_move_games']} zero-move games, {facts['zero_move_changed']} "
        f"changed; {facts['zero_move_last_changed']} last games changed at zero move.",
        f"The centre crossed a cell on {facts['crossed_cells']} games; "
        f"{facts['same_cell_changed']} same-cell guesses changed; "
        f"{facts['adjusted_selector_declines']} adjusted selectors declined and retained "
        "the served score.",
        f"Overall {facts['changed_totals']} total guesses changed, including "
        f"{facts['changed_last_totals']} last games. Fixed-side and finite-score guards "
        "passed.",
        "**Read:** src/nfl_ats/score_lattice.py:182 uses floor coordinates for "
        "team-score cells; scripts/lead88_unit2.py:222 defines the original served "
        "continuous total.",
        "**Inferred:** the declared cell gate is a fixed conservative rule for this "
        "study, not a claim that the production selector has constant output everywhere "
        "inside that cell.",
        "",
        "## Provenance and limits",
        "",
        "**Measured:** reused Unit 2 baseline; no upstream model rebuild. Source hashes, "
        "clocks, population, probability-side identity and response coefficients were "
        "checked in this command.",
        "Baseline guess = production score lattice at Tuesday total + 0.1 * joint "
        "residual - 1, with opener margin and frozen four-term probability.",
        "The historical pool-line proxy is the opener. This study changes no ATS picks "
        "and requires no pre-2026 pool capture.",
        "**Inferred limitations:** upstream artifacts and feature vintages are "
        "retrospective. LOSO includes future seasons for earlier holdouts; baseline "
        "histories can include other held-season prior games.",
        "Bootstrap intervals condition on fitted coefficients and baseline; they exclude "
        "parameter refitting, prior-unit selection and prospective capture uncertainty.",
        "Two new looks belong to lead88_total_response; they do not erase Unit 2's 82 "
        "reported looks or its broader 713-look parent protocol.",
        "No market comparator was rescored; this bounded unit isolates the requested "
        "candidate-versus-served mapping correction.",
        "No serving change, registry write, publication, new tests, commit or push occurred.",
        "",
        "## Frozen protocol",
        "",
        declaration.split("## Protocol (frozen before outcomes)\n", 1)[1]
        .split("\n## Tried", 1)[0]
        .strip(),
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def finish_lane(payload):
    results = payload["results"]
    lines = [
        "# LEAD-88 unit 3 - total response with identity at zero",
        "",
        "## Goal",
        "Isolate total-news response from lattice reprojection; research only.",
        "",
        "## State",
        "**Measured:** one declared replay complete, 2 candidate-versus-served looks; "
        "1,333/1,343 games and 101 last games scored.",
        f"Zero-move changes: {payload['facts']['zero_move_changed']}; zero-coefficient "
        "identity: 1,333/1,333 team-score pairs.",
    ]
    for name in ("last", "all"):
        result = results[name]
        lines.append(
            f"{name}: MAE improvement {metric(result)}; "
            f"probability_positive={result['probability_positive']:.6f}."
        )
    lines += [
        "Proposed unresolved_below_power; no registry writes. Report: docs/lead88_unit3.md.",
        "",
        "## Protocol (frozen before outcomes)",
        "Original declaration saved before scoring in "
        "tests/scratch/codex/lead88_unit3/protocol.md and copied verbatim into the "
        "report.",
        f"Protocol SHA256: {payload['protocol_sha256']}.",
        "Same 2020-2025 population, opener proxy, four-term side, LAD LOSO response; "
        "adjust continuous centre only across a team-score lattice cell.",
        "Two looks: last/all MAE; week-block intervals, exact closer null, fold slopes "
        "and IS/OOS gap. No outcome-driven revisions.",
        "",
        "## Tried",
        "**Measured:** .tools/uv.exe run --no-sync python scripts/lead88_unit3.py; local "
        "UV_CACHE_DIR; one job, at most two compute threads.",
        "Replay guards passed; source hashes, clocks, zero identity, retained rows, same "
        "response slopes and fixed side checked.",
        "Ruff check and format verification pending final worker review.",
        "",
        "## Record commands",
        "Orchestrator only; serial bash commands, candidate versus served.",
        chr(96) * 3 + "bash",
        record_command("last", results["last"]),
        record_command("all", results["all"]),
        chr(96) * 3,
        "",
        "## Next",
        "Orchestrator reviews the report and runs the two record commands serially; any "
        "further study needs a new declaration.",
        "",
        "## Open",
        "Retrospective feature-vintage and power limits remain. Zero crossing closes "
        "nothing; one fitted probability selects the side.",
        "No serving change, publication, tests, commit or push. Scratch rows and "
        "summary: tests/scratch/codex/lead88_unit3/.",
        "",
    ]
    LANE.write_text("\n".join(lines), encoding="utf-8")


def run():
    declaration = LANE.read_text(encoding="utf-8-sig")
    if (
        "exactly 2 candidate-versus-served looks" not in declaration
        or "floor((T+M)/2)" not in declaration
    ):
        raise ValueError("Frozen Unit 3 protocol is absent")
    ROOT.mkdir(parents=True, exist_ok=True)
    marker = ROOT / "replay_started.json"
    if marker.exists():
        raise ValueError("Unit 3 replay already began; inspect saved artifacts")
    protocol = ROOT / "protocol.md"
    protocol.write_text(declaration, encoding="utf-8")
    previous = json.loads(PRIOR_SUMMARY.read_text(encoding="utf-8"))
    for path, expected in previous["source_hashes"].items():
        if unit2.digest(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    hashes = {
        str(path): unit2.digest(path)
        for path in (BASELINE, PRIOR_SUMMARY, Path(__file__), Path(unit2.__file__))
    }
    pairs = unit2.read_frame(BASELINE)
    if (
        len(pairs) != 1343
        or pairs.game_id.duplicated().any()
        or int(pairs.four_term_restored.sum()) != 33
    ):
        raise ValueError("Frozen paired population changed")
    if not (
        pairs.tuesday_last_observed.le(pairs.freeze).all()
        and pairs.deadline_last_observed.le(pairs.deadline).all()
        and pairs.deadline.lt(pairs.kickoff).all()
    ):
        raise ValueError("Frozen quote clocks failed")
    if not np.array_equal(
        np.where(pairs.home_probability.ge(0.5), "HOME", "AWAY"), pairs.pick_side
    ):
        raise ValueError("Frozen probability no longer selects the side")
    scored = pairs.loc[pairs.served.notna()].copy()
    if len(scored) != 1333 or int(scored.is_last_game.sum()) != 101:
        raise ValueError("Scored baseline population changed")
    if not np.array_equal(scored.guess_home + scored.guess_away, scored.served):
        raise ValueError("Served integer total does not match scores")
    prior = unit2.make_prior_finals(unit2.read_frame(unit2.SCHEDULE), pairs)
    marker.write_text(
        json.dumps({"protocol_sha256": unit2.digest(protocol)}) + "\n", encoding="utf-8"
    )
    folds, full_coefficient = replay(scored, prior, previous)
    rng = np.random.default_rng(88)
    results = {"all": panel(scored, rng), "last": panel(scored.loc[scored.is_last_game], rng)}
    seasons = {}
    for year in SEASONS:
        held = scored.loc[scored.season.eq(year)]
        seasons[str(year)] = {
            "all": panel(held, rng),
            "last": panel(held.loc[held.is_last_game], rng),
        }
    zero_move = scored.total_move.eq(0)
    changed = scored.candidate.ne(scored.served)
    facts = {
        "paired_games": len(pairs),
        "scored_games": len(scored),
        "last_games": int(scored.is_last_game.sum()),
        "declined_baselines": int(pairs.served.isna().sum()),
        "restored_rows": int(pairs.four_term_restored.sum()),
        "zero_coefficient_identity_games": len(scored),
        "zero_move_games": int(zero_move.sum()),
        "zero_move_changed": int((zero_move & changed).sum()),
        "zero_move_last_changed": int((zero_move & changed & scored.is_last_game).sum()),
        "crossed_cells": int(scored.candidate_crossed.sum()),
        "same_cell_changed": int((~scored.candidate_crossed.astype(bool) & changed).sum()),
        "adjusted_selector_declines": int(scored.candidate_declined.sum()),
        "changed_totals": int(changed.sum()),
        "changed_last_totals": int((changed & scored.is_last_game).sum()),
    }
    added = [column for column in scored if column not in pairs]
    pairs.merge(
        scored[["game_id", *added]], on="game_id", how="left", validate="one_to_one"
    ).to_parquet(ROOT / "predictions.parquet", index=False)
    payload = {
        "protocol_sha256": unit2.digest(protocol),
        "source_hashes": hashes,
        "upstream_source_hashes": previous["source_hashes"],
        "facts": facts,
        "folds": folds,
        "base_fold_coefficients": previous["facts"]["four_term_fold_coefficients"],
        "full_sample_b": full_coefficient,
        "results": results,
        "seasons": seasons,
        "looks": 2,
        "bootstrap_draws": DRAWS,
        "seed": 88,
    }
    (ROOT / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(payload, declaration)
    finish_lane(payload)
    print(json.dumps({"facts": facts, "results": results}), flush=True)


if __name__ == "__main__":
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    with threadpool_limits(limits=2):
        run()
