"""Plain-language glossary content (rewrite of the former UI-10 interpretation guide)."""

from __future__ import annotations

GLOSSARY: tuple[tuple[str, str], ...] = (
    (
        "Against the spread (ATS)",
        "A bet or pick judged against the point spread, not just who wins. If the Chiefs are "
        "favored by 3 and win by 10, they 'covered.' If they win by 1, they did not.",
    ),
    (
        "Confidence / model estimate",
        "The model's own probability that its picked side covers, for this specific game. It is "
        "not the model's overall track record -- a single game can say 61% even though the "
        "model is right about 52% of the time across all games.",
    ),
    (
        "Historical accuracy",
        "How often the model's picks have been right in the past, tested on games it never "
        "trained on. This is the model's report card, not a promise about next week.",
    ),
    (
        "Uncertainty range (interval)",
        "A range that reflects how much the historical accuracy could plausibly shift on a new "
        "batch of games, given how much randomness a modest sample carries. A narrower range is "
        "stronger evidence than a single point estimate.",
    ),
    (
        "Calibration",
        "Whether stated probabilities match reality: among games the model called '60% chance,' "
        "do about 60% of them actually go that way? Good calibration means the numbers are "
        "honest, even if the model is not highly accurate.",
    ),
    (
        "Opening line",
        "The point spread posted when a market first opens for a game (this project captures it "
        "Tuesday morning). The user's pool is scored against this line, not wherever it closes.",
    ),
    (
        "Line movement",
        "How much a spread has changed between two points in time, usually because of bets, "
        "injury news, or weather. Movement toward a team suggests the market now favors it more.",
    ),
    (
        "Book dispersion",
        "How much sportsbooks disagree on a line at a moment in time. Wide disagreement can mean "
        "a game is genuinely uncertain or that books haven't converged yet.",
    ),
    (
        "Push",
        "A tie against the spread (the final margin exactly matches the line). Pushes settle as "
        "a no-decision, not a win or a loss.",
    ),
    (
        "Coin-flip reference",
        "50%: what you would expect from picking sides at random. Any real skill has to clear "
        "this bar, and clear it consistently, not just once.",
    ),
)
