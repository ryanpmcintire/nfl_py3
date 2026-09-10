from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


def build_ats_pool_card(predictions: pd.DataFrame) -> pd.DataFrame:

    required = {
        "game_id",
        "gameday",
        "away_team",
        "home_team",
        "spread_line",
        "home_cover_probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing pool columns: {', '.join(missing)}")
    card = predictions.copy()
    card["pool_side"] = card["home_cover_probability"].ge(0.5).map({True: "HOME", False: "AWAY"})
    card["pool_pick"] = card["home_team"].where(card["pool_side"].eq("HOME"), card["away_team"])
    card["pick_probability"] = card["home_cover_probability"].where(
        card["pool_side"].eq("HOME"), 1.0 - card["home_cover_probability"]
    )
    card["pick_line"] = card["spread_line"].where(
        card["pool_side"].eq("AWAY"), -card["spread_line"]
    )
    card["confidence"] = (card["pick_probability"] - 0.5).abs()
    card = card.sort_values(["confidence", "game_id"], ascending=[False, True]).reset_index(
        drop=True
    )
    card["confidence_rank"] = range(1, len(card) + 1)
    return card[
        [
            "confidence_rank",
            "gameday",
            "away_team",
            "home_team",
            "pool_pick",
            "pool_side",
            "pick_line",
            "pick_probability",
            "confidence",
            "game_id",
        ]
    ]


def pool_card_markdown(card: pd.DataFrame, season: int, week: int) -> str:
    display = card.drop(columns="game_id").copy()
    display["gameday"] = pd.to_datetime(display["gameday"]).dt.date.astype(str)
    display["pick_probability"] = display["pick_probability"].map(lambda value: f"{value:.1%}")
    display["confidence"] = display["confidence"].map(lambda value: f"{value:.1%}")
    return (
        f"# ATS pool card: {season} week {week}\n\n"
        "Every game receives a forced side; rank 1 is the model's highest-confidence pick. "
        "Confidence is not evidence of a profitable betting edge.\n\n"
        + display.to_markdown(index=False)
        + "\n"
    )


def build_straight_up_pool_card(
    predictions: pd.DataFrame, method: str = "market_residual"
) -> pd.DataFrame:

    required = {
        "game_id",
        "gameday",
        "away_team",
        "home_team",
        "method",
        "home_win_probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing straight-up pool columns: {', '.join(missing)}")
    card = predictions.loc[predictions["method"].eq(method)].copy()
    if card.empty:
        raise ValueError(f"No straight-up predictions found for method {method!r}")
    if card["game_id"].duplicated().any():
        raise ValueError(f"Method {method!r} contains duplicate game predictions")
    if card["home_win_probability"].isna().any():
        raise ValueError(f"Method {method!r} has missing winner probabilities")
    card["pool_side"] = card["home_win_probability"].ge(0.5).map({True: "HOME", False: "AWAY"})
    card["pool_pick"] = card["home_team"].where(card["pool_side"].eq("HOME"), card["away_team"])
    card["pick_probability"] = card["home_win_probability"].where(
        card["pool_side"].eq("HOME"), 1.0 - card["home_win_probability"]
    )
    card["confidence"] = card["pick_probability"] - 0.5
    card = card.sort_values(["confidence", "game_id"], ascending=[False, True]).reset_index(
        drop=True
    )
    card["confidence_rank"] = range(1, len(card) + 1)
    optional = [
        column
        for column in ("market_spread", "fair_spread", "predicted_market_residual")
        if column in card
    ]
    return card[
        [
            "confidence_rank",
            "gameday",
            "away_team",
            "home_team",
            "pool_pick",
            "pool_side",
            "pick_probability",
            "confidence",
            *optional,
            "method",
            "game_id",
        ]
    ]


def straight_up_pool_markdown(card: pd.DataFrame, season: int, week: int) -> str:
    display = card.drop(columns="game_id").copy()
    display["gameday"] = pd.to_datetime(display["gameday"]).dt.date.astype(str)
    display["pick_probability"] = display["pick_probability"].map(lambda value: f"{value:.1%}")
    display["confidence"] = display["confidence"].map(lambda value: f"{value:.1%}")
    method = str(card["method"].iloc[0])
    return (
        f"# Straight-up pool card: {season} week {week}\n\n"
        f"Method: `{method}`. Every game receives a forced winner; rank 1 is the "
        "highest-confidence pick. Optimize for the actual pool rules before using confidence "
        "as points.\n\n" + display.to_markdown(index=False) + "\n"
    )


@dataclass(frozen=True)
class PoolFormat:
    weekly_games: tuple[int, ...]
    best_pick_bonus: float = 1.0
    best_pick_penalty: float = 0.0

    def __post_init__(self) -> None:
        if not self.weekly_games or any(count <= 0 for count in self.weekly_games):
            raise ValueError("weekly_games must be a non-empty tuple of positive counts")

    @property
    def games(self) -> int:
        return int(sum(self.weekly_games))

    @property
    def weeks(self) -> int:
        return len(self.weekly_games)

    def week_slices(self) -> list[slice]:
        bounds = np.concatenate([[0], np.cumsum(self.weekly_games)])
        return [slice(int(bounds[index]), int(bounds[index + 1])) for index in range(self.weeks)]


@dataclass(frozen=True)
class FieldModel:
    entrants: int
    public_lean: float = 0.65

    def __post_init__(self) -> None:
        if self.entrants < 1:
            raise ValueError("entrants must be at least 1")
        if not 0.0 <= self.public_lean <= 1.0:
            raise ValueError("public_lean must be between 0 and 1")


@dataclass(frozen=True)
class Entry:
    cover_probability: np.ndarray
    on_public_side: np.ndarray
    best_pick_index: np.ndarray

    def __post_init__(self) -> None:
        probability = np.asarray(self.cover_probability, dtype=float)
        if probability.ndim != 1 or probability.size == 0:
            raise ValueError("cover_probability must be a non-empty 1-D array")
        if not np.all((probability >= 0.0) & (probability <= 1.0)):
            raise ValueError("cover_probability must lie in [0, 1]")
        if np.asarray(self.on_public_side).shape != probability.shape:
            raise ValueError("on_public_side must match cover_probability")


def build_entry(
    fmt: PoolFormat,
    *,
    cover_probability: float | np.ndarray,
    public_agreement: float | np.ndarray,
    best_pick_game: np.ndarray | None = None,
    seed: int = 20260818,
) -> Entry:

    games = fmt.games
    probability = np.broadcast_to(np.asarray(cover_probability, dtype=float), (games,)).copy()
    agreement = np.asarray(public_agreement, dtype=float)
    if agreement.ndim == 0:
        generator = np.random.default_rng(seed)
        public = generator.random(games) < float(agreement)
    else:
        public = np.broadcast_to(agreement, (games,)).astype(bool).copy()
    if best_pick_game is None:
        nominated = np.array([slot.start for slot in fmt.week_slices()], dtype=int)
    else:
        nominated = np.asarray(best_pick_game, dtype=int)
        if nominated.shape != (fmt.weeks,):
            raise ValueError("best_pick_game must have one entry per week")
    return Entry(cover_probability=probability, on_public_side=public, best_pick_index=nominated)


def deviate(entry: Entry, indices: np.ndarray) -> Entry:

    probability = entry.cover_probability.copy()
    public = entry.on_public_side.copy()
    selected = np.asarray(indices, dtype=int)
    probability[selected] = 1.0 - probability[selected]
    public[selected] = ~public[selected]
    return replace(entry, cover_probability=probability, on_public_side=public)


def _field_scores(
    public_won: np.ndarray,
    fmt: PoolFormat,
    field: FieldModel,
    generator: np.random.Generator,
) -> np.ndarray:

    samples = public_won.shape[0]
    lean = field.public_lean
    won = public_won.sum(axis=1)
    scores = generator.binomial(won[:, None], lean, size=(samples, field.entrants)).astype(float)
    scores += generator.binomial(
        (fmt.games - won)[:, None], 1.0 - lean, size=(samples, field.entrants)
    )
    if fmt.best_pick_bonus == 0.0 and fmt.best_pick_penalty == 0.0:
        return scores
    for slot in fmt.week_slices():
        rate = np.where(public_won[:, slot], lean, 1.0 - lean).mean(axis=1)
        hit = generator.random((samples, field.entrants)) < rate[:, None]
        scores += np.where(hit, fmt.best_pick_bonus, -fmt.best_pick_penalty)
    return scores


def simulate_pool_finish(
    entry: Entry,
    field: FieldModel,
    fmt: PoolFormat,
    *,
    samples: int = 20_000,
    seed: int = 20260818,
    chunk: int = 2_000,
    prize_places: int = 1,
) -> dict[str, float]:

    if entry.cover_probability.size != fmt.games:
        raise ValueError("entry does not match the format's game count")
    if samples < 100:
        raise ValueError("samples must be at least 100")
    if prize_places < 1:
        raise ValueError("prize_places must be at least 1")
    generator = np.random.default_rng(seed)

    outright = 0.0
    shared = 0.0
    tied = 0.0
    in_money = 0.0
    score_sum = 0.0
    score_square_sum = 0.0
    rank_sum = 0.0
    drawn = 0
    while drawn < samples:
        size = min(chunk, samples - drawn)
        covered = generator.random((size, fmt.games)) < entry.cover_probability
        ours = covered.sum(axis=1).astype(float)
        for index in entry.best_pick_index:
            if int(index) < 0:
                continue
            hit = covered[:, int(index)]
            ours += np.where(hit, fmt.best_pick_bonus, -fmt.best_pick_penalty)
        public_won = np.where(entry.on_public_side, covered, ~covered)
        rivals = _field_scores(public_won, fmt, field, generator)

        beaten = (rivals > ours[:, None]).sum(axis=1)
        level = (rivals == ours[:, None]).sum(axis=1)
        outright += float(np.count_nonzero(beaten + level == 0))
        tied += float(np.count_nonzero((beaten == 0) & (level > 0)))
        shared += float(np.where(beaten == 0, 1.0 / (1.0 + level), 0.0).sum())
        in_money += float(np.count_nonzero(beaten < prize_places))
        rank_sum += float((beaten + 1).sum())
        score_sum += float(ours.sum())
        score_square_sum += float((ours**2).sum())
        drawn += size

    mean_score = score_sum / samples
    variance = max(0.0, score_square_sum / samples - mean_score**2)
    return {
        "probability_first": shared / samples,
        "probability_outright": outright / samples,
        "probability_tied_first": tied / samples,
        "probability_in_the_money": in_money / samples,
        "prize_places": float(prize_places),
        "expected_score": mean_score,
        "score_sd": math.sqrt(variance),
        "expected_rank": rank_sum / samples,
        "entrants": float(field.entrants),
        "samples": float(samples),
    }


def head_to_head_win_probability(disagreements: int, accuracy: float) -> float:

    if disagreements < 0:
        raise ValueError("disagreements must be non-negative")
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("accuracy must be between 0 and 1")
    if disagreements == 0:
        return 0.0
    wins = np.arange(disagreements + 1)
    log_choose = np.array(
        [
            math.lgamma(disagreements + 1) - math.lgamma(w + 1) - math.lgamma(disagreements - w + 1)
            for w in wins
        ]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probability = (
            log_choose + wins * np.log(accuracy) + (disagreements - wins) * np.log1p(-accuracy)
        )
    probability = np.nan_to_num(np.exp(log_probability))
    return float(probability[wins > disagreements - wins].sum())


def strategy_comparison(
    strategies: dict[str, Entry],
    field: FieldModel,
    fmt: PoolFormat,
    *,
    samples: int = 20_000,
    seed: int = 20260818,
) -> pd.DataFrame:

    rows = [
        {"strategy": name, **simulate_pool_finish(entry, field, fmt, samples=samples, seed=seed)}
        for name, entry in strategies.items()
    ]
    frame = pd.DataFrame(rows)
    return frame.sort_values("probability_first", ascending=False).reset_index(drop=True)
