"""Phase 12 market microstructure lead features (LEAD-05, LEAD-03).

Two candidate columns for the PRODUCTION ``weak_stack`` chain, built entirely
from the local point-in-time odds archive under ``data/market/raw/`` via
``nfl_ats.clv``'s loaders and ``nfl_ats.odds``'s no-vig math. See
``docs/market_lead_battery.md`` for the predeclaration; this module only
computes the two feature columns, never touches the registry, and never calls
the paid Odds API.

**LEAD-05 -- opener-softness book ranking.** Ranks sportsbooks by how far
their own Tuesday-opener spread LINE sits from the eventual close, using the
per-book raw quote rows the pairing table (``nfl_ats.clv.build_pairing_table``)
collapses to a cross-book median. The book with the largest mean
opener-to-close error is the "softest" (least informative) book. The
candidate column, ``opener_softness_fade_signal``, fades the side implied
ONLY by that softest book's opener when it disagrees with the consensus
opener's favorite -- signed +1 (home) / -1 (away) / 0 (no disagreement), NaN
where the inputs are unavailable. The softest book is identified WALK-FORWARD
(``walk_forward_softest_book``): at each scored week's cutoff (that week's
earliest kickoff), only games that have ALREADY kicked off (and therefore
already have a resolved close) contribute to the ranking, so no game's own
close ever leaks into its own week's softest-book identification. This is the
declared alternative to "seasons strictly before the assigned window": the
opener-grade season pool is exactly 2020-2025 (``nfl_ats.rotation.GRADE_POOLS``),
so a new, uninherited family's assigned window is almost always the very
first eligible block (2020-2021) and there ARE NO whole seasons strictly
before it inside the archive. A trailing walk-forward ranking is the only
way to identify a softest book without looking into a game's own future.

**LEAD-03 -- moneyline-spread divergence.** The Tuesday-opener pairing table
already carries a consensus moneyline alongside the consensus spread
(``nfl_ats.clv.build_pairing_table``'s ``home_moneyline``/``away_moneyline``
columns, requested with ``with_h2h=True`` at the ``tue_open`` decision time --
``nfl_ats.odds_backfill.DECISION_TIMES``). Measured coverage (see
``tue_open_moneyline_coverage``): both moneyline sides are present for 100%
of the 1,537 games in the ``tue_open`` archive population 2020-2025, so no
fallback snapshot is needed -- the ROADMAP row's "true-opener + moneyline
archive" framing does not apply; ``true_open`` never requested the ``h2h``
market at all (measured: zero rows). The no-vig moneyline-implied home win
probability is compared against a walk-forward univariate logistic of home
win on the SAME snapshot's home spread (trained on strictly earlier weeks'
completed games only). The candidate column, ``ml_spread_divergence_signal``,
sides WITH the moneyline when the two probabilities diverge by at least
``DIVERGENCE_THRESHOLD_PP`` (0.03): signed +1 (home) / -1 (away) / 0 (no
divergence), NaN where the inputs are unavailable.

Both derive/attach functions follow the project's standard additive-merge
contract (``nfl_ats.redzone_reversion_production_feature``,
``nfl_ats.illness_production_feature``, ...): every pre-existing column comes
back bit-identical, only the one new column is added, and games with no
market-archive coverage (all games outside 2020-2025, plus any game missing
an input) come back NaN rather than 0 -- imputation is the model's own
training-fold job, not this module's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression

from nfl_ats.clv import build_pairing_table, close_reference_table, load_decision_quotes
from nfl_ats.constants import MIN_FITTABLE_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.odds import no_vig_probabilities
from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKET_ROOT = REPO_ROOT / "data/market/raw"

OPENER_SOFTNESS_FADE_COLUMN = "opener_softness_fade_signal"
ML_SPREAD_DIVERGENCE_COLUMN = "ml_spread_divergence_signal"

#: A book needs this many strictly-prior opener/close error observations
#: before a walk-forward ranking may name it the softest book.
MIN_BOOK_HISTORY_GAMES = 100

#: A book needs this many games in EACH season-parity half to enter the
#: descriptive split-half reliability read.
MIN_BOOK_GAMES_PER_HALF = 50

#: LEAD-03's predeclared divergence threshold, in home-win-probability points.
DIVERGENCE_THRESHOLD_PP = 0.03

CLOSE_LABEL_PRIORITY: tuple[str, ...] = ("sun_late_close", "sun_early_close")

_REQUIRED_SCHEDULE_COLUMNS = {"game_id", "season", "week", "gameday"}


def _validate_schedule(schedule: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")


def _true_week_correct(quotes: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Restrict quotes to each game's own scheduled (season, week).

    Duplicate of the same "early sighting" correction
    ``nfl_ats.clv.build_pairing_table``/``spread_price_consensus_table``
    apply internally (and ``scripts/odds_microstructure_battery.py`` applies
    a third time for its own book-level uses) -- a book can post a game's
    board more than a week ahead of the request that captures it, so the same
    game can be tagged under an earlier (season, week) than its true one.
    """

    if quotes.empty:
        return quotes
    true_week = (
        schedule[["game_id", "season", "week"]]
        .drop_duplicates("game_id")
        .rename(
            columns={"game_id": "nflverse_game_id", "season": "_true_season", "week": "_true_week"}
        )
    )
    merged = quotes.merge(true_week, on="nflverse_game_id", how="left")
    return merged.loc[
        merged["season"].eq(merged["_true_season"]) & merged["week"].eq(merged["_true_week"])
    ].drop(columns=["_true_season", "_true_week"])


def _no_vig_home_probability(home_odds: pd.Series, away_odds: pd.Series) -> np.ndarray:
    """Vectorized ``nfl_ats.odds.no_vig_probabilities`` for a column pair.

    Same NaN-guarded row-by-row pattern as
    ``scripts/odds_microstructure_battery.py``'s private ``_no_vig_pair``,
    reused rather than imported (that helper is private to a script) --
    calls the same audited, unit-tested primitive
    (``nfl_ats.odds.no_vig_probabilities``) for every finite, non-zero pair.
    """

    home_values = pd.to_numeric(home_odds, errors="coerce").to_numpy(dtype=float)
    away_values = pd.to_numeric(away_odds, errors="coerce").to_numpy(dtype=float)
    valid = (
        np.isfinite(home_values)
        & np.isfinite(away_values)
        & (home_values != 0.0)
        & (away_values != 0.0)
    )
    probability = np.full(len(home_values), np.nan, dtype=float)
    for position in np.flatnonzero(valid):
        home_probability, _ = no_vig_probabilities(home_values[position], away_values[position])
        probability[position] = home_probability
    return probability


# ---------------------------------------------------------------------------
# LEAD-05: opener-softness book ranking
# ---------------------------------------------------------------------------


def book_level_tue_open_spreads(root: Path, schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, book) with the book's own Tuesday-opener HOME spread.

    Reads the raw quote rows directly rather than through
    ``nfl_ats.clv.decision_market_consensus`` (which collapses every book to
    one cross-book median): ``home_spread_line`` is the ingest pipeline's own
    canonical home-perspective spread number, identical on a book's HOME and
    AWAY rows for the same game (measured), so filtering to ``outcome_side ==
    "HOME"`` and deduplicating to each book's latest pregame quote gives one
    row per (game, book) without re-deriving that normalization.
    """

    _validate_schedule(schedule)
    quotes = load_decision_quotes(root, capture_kind=HISTORICAL_CAPTURE_KIND, labels=("tue_open",))
    if quotes.empty:
        return pd.DataFrame(
            columns=["game_id", "season", "week", "gameday", "bookmaker_key", "book_home_spread"]
        )
    quotes = _true_week_correct(quotes, schedule)
    spreads = quotes.loc[
        quotes["market"].eq("spreads")
        & quotes["outcome_side"].eq("HOME")
        & quotes["nflverse_game_id"].notna()
    ].copy()
    spreads = spreads.loc[spreads["observed_at_utc"].lt(spreads["commence_time_utc"])]
    if spreads.empty:
        return pd.DataFrame(
            columns=["game_id", "season", "week", "gameday", "bookmaker_key", "book_home_spread"]
        )
    deduped = (
        spreads.sort_values("observed_at_utc")
        .groupby(["nflverse_game_id", "bookmaker_key"], as_index=False, dropna=False)
        .tail(1)
    )
    deduped = deduped.rename(
        columns={"nflverse_game_id": "game_id", "home_spread_line": "book_home_spread"}
    )[["game_id", "bookmaker_key", "book_home_spread"]]
    gameday = schedule[["game_id", "season", "week", "gameday"]].drop_duplicates("game_id")
    return deduped.merge(gameday, on="game_id", how="inner")[
        ["game_id", "season", "week", "gameday", "bookmaker_key", "book_home_spread"]
    ]


def close_reference(root: Path, schedule: pd.DataFrame) -> pd.DataFrame:
    """Per-game consensus close, reusing ``nfl_ats.clv`` exactly (no re-derivation)."""

    _validate_schedule(schedule)
    if "spread_line" not in schedule.columns:
        raise DataContractError("schedule is missing spread_line for the close fallback")
    pairing = build_pairing_table(
        root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=schedule,
    )
    return close_reference_table(pairing, schedule)


def book_opener_close_errors(root: Path, schedule: pd.DataFrame) -> pd.DataFrame:
    """Per (game, book) Tuesday-opener vs. close spread error.

    ``signed_error = book_home_spread - close_home_spread``; ``abs_error`` is
    its magnitude, the descriptive softness metric LEAD-05 ranks books on.
    """

    book_openers = book_level_tue_open_spreads(root, schedule)
    if book_openers.empty:
        return book_openers.assign(close_home_spread=pd.Series(dtype=float)).assign(
            signed_error=pd.Series(dtype=float), abs_error=pd.Series(dtype=float)
        )
    close = close_reference(root, schedule)[["game_id", "close_home_spread"]]
    merged = book_openers.merge(close, on="game_id", how="inner")
    merged["signed_error"] = pd.to_numeric(
        merged["book_home_spread"], errors="coerce"
    ) - pd.to_numeric(merged["close_home_spread"], errors="coerce")
    merged["abs_error"] = merged["signed_error"].abs()
    return merged


def book_softness_ranking(errors: pd.DataFrame, *, min_games: int = 1) -> pd.DataFrame:
    """Descriptive book ranking by mean |opener - close| error, ascending (sharpest first).

    No time restriction and no window -- LEAD-05 step 1, run once over the
    whole archive. ``min_games`` filters out books with too few observations
    to report meaningfully; the default (1) reports every book that appears
    at all.
    """

    working = errors.dropna(subset=["abs_error"])
    stats = (
        working.groupby("bookmaker_key")
        .agg(n_games=("abs_error", "size"), mean_abs_error=("abs_error", "mean"))
        .reset_index()
    )
    stats = stats.loc[stats["n_games"].ge(min_games)].sort_values("mean_abs_error", ascending=True)
    stats["rank"] = np.arange(1, len(stats) + 1)
    return stats.reset_index(drop=True)


def _half_ranking(errors: pd.DataFrame, seasons: tuple[int, ...], min_games: int) -> pd.Series:
    subset = errors.loc[errors["season"].astype(int).isin(seasons)]
    ranking = book_softness_ranking(subset, min_games=min_games)
    return ranking.set_index("bookmaker_key")["mean_abs_error"]


def split_half_rank_reliability(
    errors: pd.DataFrame,
    *,
    odd_seasons: tuple[int, ...],
    even_seasons: tuple[int, ...],
    min_games_per_half: int = MIN_BOOK_GAMES_PER_HALF,
    bootstrap_samples: int = 5_000,
    seed: int = 20260905,
) -> dict[str, object]:
    """Spearman rank reliability of the softness ranking across a season-parity split.

    Both halves' book rankings (mean |opener - close| error, restricted to
    books with at least ``min_games_per_half`` observations in EACH half) are
    correlated by Spearman rho over the books common to both. A season-blocked
    bootstrap resamples each half's seasons (with replacement, same count as
    the half) and recomputes both rankings and their correlation, giving a
    ``probability_positive`` (fraction of resamples with rho > 0) rather than
    a binary "contains zero" read, per the project's binding rule.
    """

    odd_ranking = _half_ranking(errors, odd_seasons, min_games_per_half)
    even_ranking = _half_ranking(errors, even_seasons, min_games_per_half)
    common = sorted(set(odd_ranking.index).intersection(even_ranking.index))
    if len(common) < 3:
        return {
            "n_books_common": len(common),
            "spearman_rho": float("nan"),
            "bootstrap_ci95": [float("nan"), float("nan")],
            "probability_positive": float("nan"),
            "bootstrap_samples": 0,
        }
    point_rho = float(
        scipy_stats.spearmanr(odd_ranking.loc[common], even_ranking.loc[common]).statistic
    )

    rng = np.random.default_rng(seed)
    rhos: list[float] = []
    for _ in range(bootstrap_samples):
        odd_draw = rng.choice(odd_seasons, size=len(odd_seasons), replace=True)
        even_draw = rng.choice(even_seasons, size=len(even_seasons), replace=True)
        odd_frame = pd.concat(
            [errors.loc[errors["season"].eq(int(season))] for season in odd_draw],
            ignore_index=True,
        )
        even_frame = pd.concat(
            [errors.loc[errors["season"].eq(int(season))] for season in even_draw],
            ignore_index=True,
        )
        odd_r = book_softness_ranking(odd_frame, min_games=1).set_index("bookmaker_key")[
            "mean_abs_error"
        ]
        even_r = book_softness_ranking(even_frame, min_games=1).set_index("bookmaker_key")[
            "mean_abs_error"
        ]
        resampled_common = sorted(set(odd_r.index).intersection(even_r.index).intersection(common))
        if len(resampled_common) < 3:
            continue
        rho = scipy_stats.spearmanr(
            odd_r.loc[resampled_common], even_r.loc[resampled_common]
        ).statistic
        if np.isfinite(rho):
            rhos.append(float(rho))
    values = np.asarray(rhos, dtype=float)
    if len(values) == 0:
        return {
            "n_books_common": len(common),
            "spearman_rho": point_rho,
            "bootstrap_ci95": [float("nan"), float("nan")],
            "probability_positive": float("nan"),
            "bootstrap_samples": 0,
        }
    return {
        "n_books_common": len(common),
        "spearman_rho": point_rho,
        "bootstrap_ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "probability_positive": float((values > 0.0).mean()),
        "bootstrap_samples": len(values),
    }


def walk_forward_softest_book(
    errors: pd.DataFrame, *, min_history_games: int = MIN_BOOK_HISTORY_GAMES
) -> pd.DataFrame:
    """Per (season, week) the softest book identified from STRICTLY EARLIER games.

    A week's cutoff is that week's earliest kickoff (``gameday``); only
    historical games with a strictly earlier ``gameday`` contribute -- every
    such game has already kicked off, so its own close is fully resolved and
    no future information reaches this week's identification. A book must
    have accumulated at least ``min_history_games`` such (game, book)
    observations before it is eligible to be named softest (ties broken by
    the lexicographically smallest book key, deterministic). Weeks before any
    book clears that bar get ``softest_book`` = ``None``.
    """

    working = errors.dropna(subset=["abs_error"]).copy()
    if working.empty:
        return pd.DataFrame(columns=["season", "week", "softest_book"])
    working["gameday"] = pd.to_datetime(working["gameday"], errors="raise")
    cutoffs = working.groupby(["season", "week"], as_index=False).agg(cutoff=("gameday", "min"))
    rows: list[tuple[int, int, str | None]] = []
    seasons = cutoffs["season"].to_numpy()
    weeks = cutoffs["week"].to_numpy()
    cutoff_values = cutoffs["cutoff"].to_numpy()
    for season_value, week_value, cutoff_value in zip(seasons, weeks, cutoff_values, strict=True):
        history = working.loc[working["gameday"].to_numpy() < cutoff_value]
        softest: str | None = None
        if not history.empty:
            stats = history.groupby("bookmaker_key")["abs_error"].agg(["mean", "count"])
            eligible = stats.loc[stats["count"].ge(min_history_games)]
            if not eligible.empty:
                top_mean = eligible["mean"].max()
                tied = sorted(eligible.index[eligible["mean"].eq(top_mean)])
                softest = tied[0]
        rows.append((int(season_value), int(week_value), softest))
    return pd.DataFrame(rows, columns=["season", "week", "softest_book"])


def _favored_side(home_spread: pd.Series) -> pd.Series:
    """``True`` home favored, ``False`` away favored, ``NaN`` pick'em/missing."""

    values = pd.to_numeric(home_spread, errors="coerce")
    return pd.Series(
        np.where(values.lt(0.0), True, np.where(values.gt(0.0), False, np.nan)),
        index=home_spread.index,
        dtype=object,
    )


def derive_opener_softness_fade_features(
    features: pd.DataFrame,
    *,
    market_root: Path | None = None,
    min_history_games: int = MIN_BOOK_HISTORY_GAMES,
) -> pd.DataFrame:
    """Return a ``(game_id, opener_softness_fade_signal)`` frame.

    +1.0 (home) / -1.0 (away) when the walk-forward-identified softest book's
    own Tuesday-opener favorite disagrees with the CONSENSUS Tuesday-opener
    favorite (fading the softest book, per LEAD-05's predeclared direction);
    0.0 when they agree; NaN when the softest book has not yet been
    identified, has no quote for this game, or either spread is exactly a
    pick'em (no favorite to disagree about).
    """

    root = market_root if market_root is not None else DEFAULT_MARKET_ROOT
    if "spread_line" not in features.columns:
        raise DataContractError("features is missing spread_line for the close fallback")
    schedule = (
        features[["game_id", "season", "week", "gameday", "spread_line"]]
        .drop_duplicates("game_id")
        .copy()
    )
    schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="raise")

    errors = book_opener_close_errors(root, schedule)
    if errors.empty:
        return pd.DataFrame(
            {"game_id": features["game_id"].astype(str), OPENER_SOFTNESS_FADE_COLUMN: np.nan}
        )

    softest_by_week = walk_forward_softest_book(errors, min_history_games=min_history_games)
    game_week = errors[["game_id", "season", "week"]].drop_duplicates("game_id")
    game_softest = game_week.merge(softest_by_week, on=["season", "week"], how="left")

    book_openers = errors[["game_id", "bookmaker_key", "book_home_spread"]]
    scoped = game_softest.merge(
        book_openers,
        left_on=["game_id", "softest_book"],
        right_on=["game_id", "bookmaker_key"],
        how="left",
    )

    pairing = build_pairing_table(
        root, capture_kind=HISTORICAL_CAPTURE_KIND, labels=("tue_open",), schedule=schedule
    )[["game_id", "home_spread"]].rename(columns={"home_spread": "consensus_home_spread"})
    scoped = scoped.merge(pairing, on="game_id", how="left")

    pick_home_consensus = _favored_side(scoped["consensus_home_spread"])
    pick_home_softest = _favored_side(scoped["book_home_spread"])
    both_known = pick_home_consensus.notna() & pick_home_softest.notna()
    disagreement = both_known & pick_home_consensus.ne(pick_home_softest)

    signal = pd.Series(np.nan, index=scoped.index, dtype=float)
    signal.loc[both_known] = 0.0
    signal.loc[disagreement & pick_home_consensus.eq(True)] = 1.0
    signal.loc[disagreement & pick_home_consensus.eq(False)] = -1.0

    result = pd.DataFrame({"game_id": scoped["game_id"], OPENER_SOFTNESS_FADE_COLUMN: signal})
    all_games = (
        features[["game_id"]].drop_duplicates().assign(game_id=lambda d: d["game_id"].astype(str))
    )
    result["game_id"] = result["game_id"].astype(str)
    return all_games.merge(result, on="game_id", how="left")


def attach_opener_softness_fade_features(
    features: pd.DataFrame,
    *,
    market_root: Path | None = None,
    min_history_games: int = MIN_BOOK_HISTORY_GAMES,
) -> pd.DataFrame:
    """Additively join ``opener_softness_fade_signal`` onto ``features`` by ``game_id``."""

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    if OPENER_SOFTNESS_FADE_COLUMN in features.columns:
        raise DataContractError(f"features already carries {OPENER_SOFTNESS_FADE_COLUMN}")
    derived = derive_opener_softness_fade_features(
        features, market_root=market_root, min_history_games=min_history_games
    )
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_market_lead"),
        validate="one_to_one",
    )
    merged = merged.drop(
        columns=[c for c in ("key_0", "game_id_market_lead") if c in merged.columns]
    )
    merged.index = features.index
    return merged


# ---------------------------------------------------------------------------
# LEAD-03: moneyline-spread divergence
# ---------------------------------------------------------------------------


def tue_open_moneyline_coverage(root: Path, schedule: pd.DataFrame) -> dict[str, object]:
    """Measured coverage of BOTH moneyline sides at the Tuesday opener.

    Reported once per ``docs/market_lead_battery.md``'s predeclaration
    (LEAD-03 requires this measurement before any outcome is scored).
    """

    _validate_schedule(schedule)
    pairing = build_pairing_table(
        root, capture_kind=HISTORICAL_CAPTURE_KIND, labels=("tue_open",), schedule=schedule
    )
    both = pairing["home_moneyline"].notna() & pairing["away_moneyline"].notna()
    by_season = (
        pairing.assign(has_both_moneyline=both)
        .groupby("season", as_index=False)
        .agg(coverage=("has_both_moneyline", "mean"))
    )
    coverage_by_season = {
        int(season_value): float(coverage_value)
        for season_value, coverage_value in zip(
            by_season["season"].to_numpy(), by_season["coverage"].to_numpy(), strict=True
        )
    }
    return {
        "n_games_tue_open": len(pairing),
        "n_games_with_both_moneyline": int(both.sum()),
        "coverage": float(both.mean()) if len(pairing) else float("nan"),
        "coverage_by_season": coverage_by_season,
    }


def _walk_forward_spread_implied_home_win_probability(
    frame: pd.DataFrame, *, min_train_games: int = MIN_FITTABLE_TRAIN_GAMES
) -> np.ndarray:
    """Per-week univariate logistic of home win on the SAME snapshot's home spread.

    Trained on strictly earlier weeks' completed, non-push games only (walk
    -forward, no leakage): each week's predictions come from a model fit
    before that week's earliest kickoff. Weeks without ``min_train_games``
    qualifying prior games get NaN.
    """

    working = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    working["gameday"] = pd.to_datetime(working["gameday"], errors="raise")
    result = np.full(len(working), np.nan, dtype=float)
    home_spread = pd.to_numeric(working["home_spread"], errors="coerce")
    result_margin = pd.to_numeric(working["result"], errors="coerce")
    trainable = result_margin.notna() & result_margin.ne(0.0) & home_spread.notna()

    for (_season, _week), group in working.groupby(["season", "week"], sort=True):
        cutoff = group["gameday"].min()
        train_mask = trainable & working["gameday"].lt(cutoff)
        if int(train_mask.sum()) < min_train_games:
            continue
        train_x = home_spread.loc[train_mask].to_numpy(dtype=float).reshape(-1, 1)
        train_y = (result_margin.loc[train_mask] > 0.0).to_numpy(dtype=float)
        if len(np.unique(train_y)) < 2:
            continue
        model = LogisticRegression()
        model.fit(train_x, train_y)
        # ``working`` was reset to a plain RangeIndex above, so ``group``'s
        # (inherited) index values are exactly this week's row POSITIONS in
        # ``result`` -- no separate position lookup needed.
        score_index = group.index[home_spread.loc[group.index].notna()]
        if score_index.empty:
            continue
        score_x = home_spread.loc[score_index].to_numpy(dtype=float).reshape(-1, 1)
        predicted = model.predict_proba(score_x)[:, list(model.classes_).index(1.0)]
        result[score_index.to_numpy()] = predicted
    return result


def _divergence_to_signal(divergence: np.ndarray | pd.Series, *, threshold: float) -> np.ndarray:
    """Map a signed probability divergence to LEAD-03's {-1.0, 0.0, +1.0, NaN} rule.

    NaN wherever ``divergence`` itself is NaN -- since it is computed as
    ``ml_probability - spread_probability``, NaN propagates automatically
    whenever either side was unavailable, so this one ``isfinite`` check
    covers both. 0.0 when both inputs are known but the divergence's
    magnitude is below ``threshold``; +1.0 (home) / -1.0 (away) when it is AT
    OR BEYOND ``threshold`` in that direction -- the predeclared
    ">= 3 percentage points" rule counts the boundary itself as a divergence.
    """

    values = np.asarray(divergence, dtype=float)
    known = np.isfinite(values)
    signal = np.full(values.shape, np.nan, dtype=float)
    signal[known] = 0.0
    signal[known & (values >= threshold)] = 1.0
    signal[known & (values <= -threshold)] = -1.0
    return signal


def derive_ml_spread_divergence_features(
    features: pd.DataFrame,
    *,
    market_root: Path | None = None,
    threshold: float = DIVERGENCE_THRESHOLD_PP,
    min_train_games: int = MIN_FITTABLE_TRAIN_GAMES,
) -> pd.DataFrame:
    """Return a ``(game_id, ml_spread_divergence_signal)`` frame.

    +1.0 (home) / -1.0 (away) when the no-vig Tuesday-opener moneyline home
    win probability diverges from the walk-forward spread-implied home win
    probability by at least ``threshold`` (0.03, LEAD-03's predeclared 3
    percentage points), siding WITH the moneyline; 0.0 when the divergence is
    smaller; NaN when either input is unavailable or the walk-forward
    logistic has no fittable training window yet.
    """

    root = market_root if market_root is not None else DEFAULT_MARKET_ROOT
    schedule = (
        features[["game_id", "season", "week", "gameday", "result"]]
        .drop_duplicates("game_id")
        .copy()
    )
    schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="raise")

    pairing = build_pairing_table(
        root, capture_kind=HISTORICAL_CAPTURE_KIND, labels=("tue_open",), schedule=schedule
    )[["game_id", "home_spread", "home_moneyline", "away_moneyline"]]
    merged = schedule.merge(pairing, on="game_id", how="left")

    ml_probability = _no_vig_home_probability(merged["home_moneyline"], merged["away_moneyline"])
    spread_probability = _walk_forward_spread_implied_home_win_probability(
        merged, min_train_games=min_train_games
    )
    signal = _divergence_to_signal(ml_probability - spread_probability, threshold=threshold)

    result = pd.DataFrame({"game_id": merged["game_id"], ML_SPREAD_DIVERGENCE_COLUMN: signal})
    all_games = (
        features[["game_id"]].drop_duplicates().assign(game_id=lambda d: d["game_id"].astype(str))
    )
    result["game_id"] = result["game_id"].astype(str)
    return all_games.merge(result, on="game_id", how="left")


def attach_ml_spread_divergence_features(
    features: pd.DataFrame,
    *,
    market_root: Path | None = None,
    threshold: float = DIVERGENCE_THRESHOLD_PP,
    min_train_games: int = MIN_FITTABLE_TRAIN_GAMES,
) -> pd.DataFrame:
    """Additively join ``ml_spread_divergence_signal`` onto ``features`` by ``game_id``."""

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    if ML_SPREAD_DIVERGENCE_COLUMN in features.columns:
        raise DataContractError(f"features already carries {ML_SPREAD_DIVERGENCE_COLUMN}")
    derived = derive_ml_spread_divergence_features(
        features, market_root=market_root, threshold=threshold, min_train_games=min_train_games
    )
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_market_lead"),
        validate="one_to_one",
    )
    merged = merged.drop(
        columns=[c for c in ("key_0", "game_id_market_lead") if c in merged.columns]
    )
    merged.index = features.index
    return merged


__all__ = [
    "CLOSE_LABEL_PRIORITY",
    "DEFAULT_MARKET_ROOT",
    "DIVERGENCE_THRESHOLD_PP",
    "MIN_BOOK_GAMES_PER_HALF",
    "MIN_BOOK_HISTORY_GAMES",
    "ML_SPREAD_DIVERGENCE_COLUMN",
    "OPENER_SOFTNESS_FADE_COLUMN",
    "attach_ml_spread_divergence_features",
    "attach_opener_softness_fade_features",
    "book_level_tue_open_spreads",
    "book_opener_close_errors",
    "book_softness_ranking",
    "close_reference",
    "derive_ml_spread_divergence_features",
    "derive_opener_softness_fade_features",
    "split_half_rank_reliability",
    "tue_open_moneyline_coverage",
    "walk_forward_softest_book",
]
