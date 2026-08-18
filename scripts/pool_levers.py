"""POL-04/05: what is the pool's FORMAT worth, separately from the line?

Everything here is arithmetic conditional on probabilities measured elsewhere.
It spends no rotation window and reserves no data: the simulator answers "given
this edge, how does the format convert it into a chance of finishing first?",
never "is the edge real?".

Run:

    .\\.tools\\uv.exe run --no-sync python scripts/pool_levers.py

Write-up: ``docs/pool_format_levers.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from nfl_ats.pool import (
    Entry,
    FieldModel,
    PoolFormat,
    build_entry,
    deviate,
    head_to_head_win_probability,
    simulate_pool_finish,
)

REPO = Path(__file__).resolve().parents[1]

# The 2026 pool: 272 regular-season games across 18 weeks with one Best Pick
# each, then 13 postseason games with none. The exact weekly split moves nothing
# (only the count of Best Pick weeks matters), so a representative one is used.
REGULAR_WEEKS = (16, 16) + (15,) * 16
POSTSEASON = (13,)
SEASON_GAMES = sum(REGULAR_WEEKS) + sum(POSTSEASON)

# Measured inputs, all from committed artifacts:
#   0.5250  forced-pick accuracy at the Tuesday opener, 1,503 paired games
#           (artifacts/opener_evaluation/, docs/opener_evaluation.md)
#   0.548   share of our picks that are on the favourite (measured below-fold)
BASELINE_ACCURACY = 0.5250
FIELD_SIZES = (5, 10, 25, 50, 100, 250, 1000)
SAMPLES = 40_000
SEED = 20260818


def season_format(best_pick_bonus: float) -> PoolFormat:
    return PoolFormat(
        weekly_games=REGULAR_WEEKS + POSTSEASON,
        best_pick_bonus=best_pick_bonus,
    )


def _nominations(fmt: PoolFormat, offsets: np.ndarray | None = None) -> np.ndarray:
    """Global game index nominated per week; the postseason block gets none."""

    slots = fmt.week_slices()
    picks = []
    for index, slot in enumerate(slots):
        if index >= len(REGULAR_WEEKS):
            picks.append(-1)
            continue
        shift = 0 if offsets is None else int(offsets[index])
        picks.append(slot.start + shift)
    return np.array(picks, dtype=int)


def experiment_edge_value(results: dict[str, Any]) -> None:
    """How much a 52.5% card is worth against a field of coin-flippers."""

    fmt = season_format(best_pick_bonus=0.0)
    rows = []
    for entrants in FIELD_SIZES:
        field = FieldModel(entrants=entrants, public_lean=0.65)
        places = max(1, round(0.15 * (entrants + 1)))
        for label, accuracy in (("coin_flip", 0.50), ("model", BASELINE_ACCURACY)):
            entry = build_entry(
                fmt,
                cover_probability=accuracy,
                public_agreement=0.548,
                best_pick_game=_nominations(fmt),
                seed=SEED,
            )
            outcome = simulate_pool_finish(
                entry, field, fmt, samples=SAMPLES, seed=SEED, prize_places=places
            )
            rows.append({"entrants": entrants, "entry": label, **outcome})
    results["edge_value"] = rows


def experiment_best_pick(results: dict[str, Any]) -> None:
    """What a working Best Pick ranker buys, at each candidate effect size.

    The card is held fixed: one game per week genuinely covers at ``high`` and
    the rest at ``low``, chosen so the whole card still averages 52.50%. The only
    thing that changes between strategies is WHICH game is nominated, which is
    the actual decision a ranker makes.
    """

    rows = []
    for delta_label, delta in (
        ("none", 0.0),
        ("tie_break_agnostic_+0.9pts", 0.0092),
        ("as_recorded_+8.7pts", 0.0868),
    ):
        for bonus in (1.0, 2.0):
            fmt = season_format(best_pick_bonus=bonus)
            weeks = len(REGULAR_WEEKS)
            high = BASELINE_ACCURACY + delta * (1.0 - weeks / SEASON_GAMES)
            low = (SEASON_GAMES * BASELINE_ACCURACY - weeks * high) / (SEASON_GAMES - weeks)
            probability = np.full(SEASON_GAMES, low)
            marked = np.array([slot.start for slot in fmt.week_slices()[:weeks]])
            probability[marked] = high
            for entrants in (10, 100, 1000):
                field = FieldModel(entrants=entrants, public_lean=0.65)
                for label, offsets in (
                    ("ranker_nominates_the_high_game", None),
                    ("arbitrary_nomination", np.ones(weeks)),
                ):
                    entry = build_entry(
                        fmt,
                        cover_probability=probability,
                        public_agreement=0.548,
                        best_pick_game=_nominations(fmt, offsets),
                        seed=SEED,
                    )
                    outcome = simulate_pool_finish(entry, field, fmt, samples=SAMPLES, seed=SEED)
                    rows.append(
                        {
                            "ranker_effect": delta_label,
                            "best_pick_bonus": bonus,
                            "entrants": entrants,
                            "strategy": label,
                            **outcome,
                        }
                    )
    results["best_pick"] = rows


def experiment_differentiation(results: dict[str, Any]) -> None:
    """Deliberately taking worse sides to separate from the field: worth it?

    Every flip costs ``2 * (p - 0.5)`` expected points and moves one pick off the
    public side, widening the spread of our margin over the field. The sweep is
    over the number of flips, at three field sizes and two levels of field
    cohesion.
    """

    fmt = season_format(best_pick_bonus=1.0)
    base = build_entry(
        fmt,
        cover_probability=BASELINE_ACCURACY,
        public_agreement=0.548,
        best_pick_game=_nominations(fmt),
        seed=SEED,
    )
    agreeing = np.flatnonzero(base.on_public_side)
    rows = []
    for lean in (0.65, 0.85):
        for entrants in (10, 100, 1000):
            field = FieldModel(entrants=entrants, public_lean=lean)
            for flips in (0, 10, 25, 50, 100, len(agreeing)):
                entry: Entry = base if flips == 0 else deviate(base, agreeing[:flips])
                outcome = simulate_pool_finish(entry, field, fmt, samples=SAMPLES, seed=SEED)
                rows.append({"public_lean": lean, "entrants": entrants, "flips": flips, **outcome})
    results["differentiation"] = rows


def experiment_free_differentiation(results: dict[str, Any]) -> None:
    """The same sweep when flips are FREE: coin-flip games we deliberately fade.

    This is the optimistic bound. It assumes we can identify games our model has
    no opinion on (p = 0.500), which the flat-confidence finding says we cannot
    do reliably -- so read it as a ceiling on the lever, not a plan.
    """

    fmt = season_format(best_pick_bonus=1.0)
    probability = np.full(SEASON_GAMES, BASELINE_ACCURACY)
    coin_flips = np.arange(0, SEASON_GAMES, 3)  # a third of the card carries no edge
    probability[coin_flips] = 0.5
    # Hold the card's overall accuracy at 52.50% by concentrating the edge.
    remaining = np.setdiff1d(np.arange(SEASON_GAMES), coin_flips)
    probability[remaining] = (
        SEASON_GAMES * BASELINE_ACCURACY - 0.5 * coin_flips.size
    ) / remaining.size
    base = build_entry(
        fmt,
        cover_probability=probability,
        public_agreement=0.548,
        best_pick_game=_nominations(fmt),
        seed=SEED,
    )
    free = np.array([index for index in coin_flips if base.on_public_side[index]])
    rows = []
    for entrants in (10, 100, 1000):
        field = FieldModel(entrants=entrants, public_lean=0.65)
        for flips in (0, 10, 25, free.size):
            entry: Entry = base if flips == 0 else deviate(base, free[:flips])
            outcome = simulate_pool_finish(entry, field, fmt, samples=SAMPLES, seed=SEED)
            rows.append({"entrants": entrants, "free_flips": flips, **outcome})
    results["free_differentiation"] = rows


def experiment_head_to_head(results: dict[str, Any]) -> None:
    """The closed form, which is where the differentiation intuition comes from."""

    rows = [
        {
            "disagreements": disagreements,
            "accuracy_on_disagreements": accuracy,
            "probability_beat_one_rival": head_to_head_win_probability(disagreements, accuracy),
        }
        for accuracy in (0.50, 0.525, 0.55)
        for disagreements in (10, 25, 50, 100, 143, 285)
    ]
    results["head_to_head"] = rows


def experiment_flip_cost(results: dict[str, Any]) -> None:
    """How cheap must a contrarian flip be before it pays?

    This is the decision-relevant version of the differentiation question. The
    card carries a block of near-coin-flip picks -- measured: 41% of games have
    a model residual under one point, which the 13.1-point scatter converts to
    an edge of at most 1.5 accuracy points -- and each flip inside that block
    costs ``cost`` accuracy points while moving one pick off the public side.
    The sweep finds the break-even cost.
    """

    fmt = season_format(best_pick_bonus=1.0)
    weak = np.arange(0, SEASON_GAMES, 5)  # a fifth of the card, evenly spread
    rows = []
    for cost in (0.0, 0.005, 0.010, 0.025, 0.050):
        probability = np.full(SEASON_GAMES, 0.5 + cost / 2.0)
        strong = np.setdiff1d(np.arange(SEASON_GAMES), weak)
        probability[strong] = (
            SEASON_GAMES * BASELINE_ACCURACY - probability[weak].sum()
        ) / strong.size
        base = build_entry(
            fmt,
            cover_probability=probability,
            public_agreement=0.548,
            best_pick_game=_nominations(fmt),
            seed=SEED,
        )
        available = np.array([index for index in weak if base.on_public_side[index]])
        for entrants in (10, 100, 1000):
            field = FieldModel(entrants=entrants, public_lean=0.65)
            # Splash pools commonly pay roughly the top 15%; carry both objectives.
            places = max(1, round(0.15 * (entrants + 1)))
            for flips in (0, 10, 20, available.size):
                entry: Entry = base if flips == 0 else deviate(base, available[:flips])
                outcome = simulate_pool_finish(
                    entry, field, fmt, samples=SAMPLES, seed=SEED, prize_places=places
                )
                rows.append(
                    {
                        "accuracy_cost_per_flip": cost,
                        "entrants": entrants,
                        "flips": flips,
                        "flips_available": int(available.size),
                        **outcome,
                    }
                )
    results["flip_cost"] = rows


def experiment_accuracy_scaling(results: dict[str, Any]) -> None:
    """What a point of ATS accuracy is worth in standings position.

    This is the benchmark every format lever has to be measured against: it is
    the same simulator, the same field, and the only thing that changes is the
    quantity the rest of the project is trying to move.
    """

    fmt = season_format(best_pick_bonus=0.0)
    rows = []
    for accuracy in (0.500, 0.5250, 0.5350, 0.5450, 0.5550):
        entry = build_entry(
            fmt,
            cover_probability=accuracy,
            public_agreement=0.548,
            best_pick_game=_nominations(fmt),
            seed=SEED,
        )
        for entrants in (25, 100, 1000):
            field = FieldModel(entrants=entrants, public_lean=0.65)
            places = max(1, round(0.15 * (entrants + 1)))
            outcome = simulate_pool_finish(
                entry, field, fmt, samples=SAMPLES, seed=SEED, prize_places=places
            )
            rows.append({"accuracy": accuracy, "entrants": entrants, **outcome})
    results["accuracy_scaling"] = rows


EXPERIMENTS = {
    "head_to_head": experiment_head_to_head,
    "accuracy_scaling": experiment_accuracy_scaling,
    "edge_value": experiment_edge_value,
    "best_pick": experiment_best_pick,
    "differentiation": experiment_differentiation,
    "free_differentiation": experiment_free_differentiation,
    "flip_cost": experiment_flip_cost,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/pool_levers/levers.json")
    parser.add_argument("--only", action="append", choices=sorted(EXPERIMENTS), default=None)
    args = parser.parse_args()

    results: dict[str, Any] = {
        "baseline_accuracy": BASELINE_ACCURACY,
        "season_games": SEASON_GAMES,
        "best_pick_weeks": len(REGULAR_WEEKS),
        "samples": SAMPLES,
        "seed": SEED,
    }
    if args.out.is_file():
        results = {**json.loads(args.out.read_text(encoding="utf-8")), **results}
    for name in args.only or list(EXPERIMENTS):
        EXPERIMENTS[name](results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({', '.join(args.only or list(EXPERIMENTS))})")


if __name__ == "__main__":
    main()
