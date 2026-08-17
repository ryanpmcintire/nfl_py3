"""The content model behind "What we've learned" -- every finding, in plain words.

This module is the *text*; :mod:`nfl_ats.dashboard.app_pages.findings` is the
layout. Nothing here imports Streamlit and nothing here formats HTML, so the
wording can be reviewed, diffed, and argued about on its own.

The rules the wording follows, because the owner asked for them explicitly:

- No jargon survives contact with this page. "EPA" becomes "how many points a
  play was worth on average"; "Brier score" becomes "how often the picks
  actually landed"; "walk-forward" becomes "scored only on games it had never
  seen". A term is allowed only when the sentence that uses it also explains it
  in football terms.
- Every claim traces to a committed record (``source``). No number is invented,
  softened, or rounded in our favour.
- Negative results are stated proudly. They are the most trustworthy part of
  the record, and there are more of them than positives.
- Untested leads are labelled untested, every time, even when they are
  exciting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Verdict = Literal["helps", "no-edge", "unproven", "context"]
ChipKind = Literal["good", "warning", "muted", "plain"]


@dataclass(frozen=True)
class Finding:
    """One question a person might ask, and the honest answer to it."""

    question: str
    verdict: Verdict
    plain_answer: str
    detail: str
    source: str


@dataclass(frozen=True)
class VerdictGroup:
    """A section of the page: one verdict, its framing, and its chip."""

    verdict: Verdict
    kicker: str
    title: str
    blurb: str
    chip_label: str
    chip_kind: ChipKind
    legend: str


@dataclass(frozen=True)
class HeadlineTile:
    """One hero statistic."""

    kicker: str
    value: str
    context: str
    delta_text: str | None = None
    delta_good: bool | None = None


@dataclass(frozen=True)
class HonestyRule:
    """One rule we hold ourselves to when reporting a number."""

    title: str
    body: str


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

HERO_KICKER = "What we've learned"
HERO_TITLE = "Everything the research has settled, in plain English"
HERO_SUB = (
    "No jargon, and no hedging into meaninglessness. Every answer below is either something "
    "we measured on games the model had never seen, or something we tell you outright that "
    "we have not measured yet."
)

HERO_TILES: tuple[HeadlineTile, ...] = (
    HeadlineTile(
        kicker="Against the pool's line",
        value="52.5%",
        context="1,537 games, 2020-2025, every one scored by a model that never saw the result.",
        delta_text="2.5 points better than a coin flip",
        delta_good=True,
    ),
    HeadlineTile(
        kicker="Against the closing line",
        value="51.1%",
        context=(
            "Same model, same games, graded at Sunday's sharper number. The market spends the "
            "week drifting toward us, and a frozen Tuesday spread hands that drift back."
        ),
        delta_text="same picks, later line",
    ),
    HeadlineTile(
        kicker="A realistic ceiling",
        value="54-55%",
        context=(
            "What an excellent NFL model can hope for against a frozen line. Anything near 60% "
            "is a bug in the test, not a breakthrough."
        ),
    ),
)

HERO_PARAGRAPHS: tuple[str, ...] = (
    "Here is the state of the project in one paragraph. Our pool locks every pick on Tuesday "
    "at noon, against a spread that is posted early in the week and then frozen, and everyone "
    "in the pool picks every game. Graded exactly that way -- on 1,537 games from 2020 through "
    "2025 that the model was never trained on -- we took the right side 52.5% of the time. "
    "That works out to a 97-99% chance of being genuine skill rather than luck, and across a "
    "full 285-game season (272 regular season plus 13 playoff games) it is worth roughly seven "
    "more correct picks than flipping a coin.",
    "That edge is smaller than it sounds and bigger than it looks. Smaller, because 52.5% "
    "still loses a lot of Sundays and always will. Bigger, because the practical ceiling for "
    "this problem is around 55%, so we are already about halfway from a coin flip to the limit "
    "of what anyone does. Most of what follows is the things that did not work on the way "
    "here. They are not failures we are hiding; they are the reason the number above is "
    "believable.",
)

LEGEND_KICKER = "How to read the labels"


# ---------------------------------------------------------------------------
# Groups, in page order
# ---------------------------------------------------------------------------

GROUPS: tuple[VerdictGroup, ...] = (
    VerdictGroup(
        verdict="helps",
        kicker="What actually works",
        title="The findings that survived being tested properly",
        blurb=(
            "Each of these held up on games the model had never seen, with the test written "
            "down before it ran. There are not many of them, and that is the point."
        ),
        chip_label="confirmed signal",
        chip_kind="good",
        legend="Held up out of sample. It is in the model we run.",
    ),
    VerdictGroup(
        verdict="unproven",
        kicker="Promising, not proven",
        title="Leads we have not validated yet",
        blurb=(
            "Each of these points the right way, or has support in published research. None "
            "has cleared our bar. We list them here so nobody -- us included -- quietly "
            "promotes one to a finding by talking about it enough."
        ),
        chip_label="untested lead",
        chip_kind="warning",
        legend="Points the right way; not proven. Not in the model.",
    ),
    VerdictGroup(
        verdict="no-edge",
        kicker="Tested, and no",
        title="Good ideas that turned out not to help",
        blurb=(
            "The biggest section on the page, deliberately. Every one of these was built, "
            "measured, and closed. Knowing where the edge is not is what stops us spending "
            "another year looking there -- and most of them fail for the same interesting "
            "reason: the betting market already knew."
        ),
        chip_label="no edge found",
        chip_kind="muted",
        legend="Built it, measured it, it did not help. Recorded, not deleted.",
    ),
    VerdictGroup(
        verdict="context",
        kicker="How this actually works",
        title="Structural truths worth knowing before you read a percentage",
        blurb=(
            "Not results -- the shape of the problem. These are the facts that decide what any "
            "number on this dashboard is allowed to mean."
        ),
        chip_label="how it works",
        chip_kind="plain",
        legend="A fact about the problem rather than a result.",
    ),
)


# ---------------------------------------------------------------------------
# The findings
# ---------------------------------------------------------------------------

DETAIL_SUMMARY_LABEL = "How we know"
SOURCE_LABEL = "Source"

FINDINGS: tuple[Finding, ...] = (
    # -- helps ---------------------------------------------------------------
    Finding(
        question="Do our picks beat a coin flip against the line the pool actually uses?",
        verdict="helps",
        plain_answer=(
            "Yes, by about two and a half points. On 1,537 games from 2020 through 2025 -- "
            "every one of them scored by a model that had never seen the result -- we picked "
            "the side that covered 52.5% of the time, where a coin flip gets 50%. The chance "
            "that this is real skill rather than a hot streak works out to about 97-99%, and "
            "we finished above 50% in five of those six seasons."
        ),
        detail=(
            "The season we lost was 2020, the empty-stadium year: home-field advantage "
            "collapsed league-wide and a model trained on normal seasons kept over-crediting "
            "the home team. It is also the thinnest slice of our line archive (227 games "
            "instead of 272). We ran this measurement exactly once, with the model frozen and "
            "the scoring rules written down beforehand, so nothing was adjusted after the "
            "number came back."
        ),
        source="docs/opener_evaluation.md",
    ),
    Finding(
        question="Does it matter which line we are graded against?",
        verdict="helps",
        plain_answer=(
            "More than anything else we have found. The same picks on the same games score "
            "52.5% against Tuesday's opening line and only 51.1% against the line the market "
            "settles on by Sunday. The market spends the week drifting toward our number, and "
            "because the pool freezes its spread on Tuesday and never moves it, that drift "
            "gets handed straight back to us as accuracy."
        ),
        detail=(
            "The gap is 1.35 points, and it survives re-scoring the games in whole-week and "
            "whole-season chunks: about a 99.9% chance it is real. It also fixed a mistake we "
            "had been making for a year -- grading ourselves at the closing line, the way "
            "serious bettors do, was quietly understating the model against the only line we "
            "are actually judged on. For scale: someone who could perfectly foresee "
            "Wednesday-to-Sunday line movement, with zero football knowledge, would score "
            "55.1% against that frozen Tuesday number. We capture roughly half of that."
        ),
        source="docs/opener_evaluation.md",
    ),
    Finding(
        question="Is it better to correct the market's number than to predict the game ourselves?",
        verdict="helps",
        plain_answer=(
            "Much better, and we tried both. Predicting the final margin from football data "
            "alone and then comparing it to the spread is worse than starting from the spread "
            "and predicting only how wrong it is. The correcting version misses the real "
            "margin by about 9.9 points on average; the from-scratch version misses by 10.1. "
            "That gap sounds trivial and is not: a fifth of a point of line accuracy is most "
            "of the distance between a working model and a useless one."
        ),
        detail=(
            "Everything we ship is built this way -- the market line is the starting "
            "assumption and the model estimates only the correction to it. We confirmed the "
            "same thing on college football, where we have almost three times as many games: "
            "the correcting model picked 51.6% of 8,933 games against 49.6% for a control "
            "that just follows the market. That college benchmark exists precisely because it "
            "can resolve effects of around one accuracy point that our 2,000-game NFL sample "
            "cannot."
        ),
        source="docs/modeling.md, ROADMAP.md (XLG-03)",
    ),
    Finding(
        question="Does knowing who is hurt and who is starting at quarterback help?",
        verdict="helps",
        plain_answer=(
            "Modestly, and less than we first believed. The player layer -- quarterback form, "
            "the injury report weighted by how much each player actually plays, and how much "
            "the lineup has churned -- is in the model we run, and on matched games it is "
            "worth about a point of accuracy over the same model without it. But when we "
            "re-tested its most promising slice on seasons it had never been scored on, it "
            "added exactly nothing. So we keep it, and we do not brag about it."
        ),
        detail=(
            "The numbers: 51.1% for a market-and-team-form model, 52.1% with the full player "
            "layer, on the same 2,075 games; injury information on its own reached 51.3%. The "
            "'quarterback plus lineup continuity' variant looked like the best thing in the "
            "entire search at 52.3-52.6%, then scored +0.00 points on 997 untouched 2014-2017 "
            "games, splitting 88-88 on the games where it disagreed with the simpler model. "
            "Injury values are read from the report as it stood 24 hours before kickoff, "
            "never from what we learned afterwards, and they are split by unit -- offensive "
            "line, skill positions, front seven, secondary -- so each can be judged alone."
        ),
        source="docs/modeling.md, ROADMAP.md (PER-02/03/05)",
    ),
    # -- unproven ------------------------------------------------------------
    Finding(
        question="Can we learn who will actually play, instead of trusting the injury label?",
        verdict="unproven",
        plain_answer=(
            "This is our most promising unfinished idea. Rather than hand-assigning "
            "'questionable means a 35% chance of missing', we learned the real rates from "
            "sixteen years of injury reports and practice attendance. They are very "
            "different: a questionable player who practised fully misses about 20% of the "
            "time, one who did not practise at all misses about 60%. Predicting availability "
            "got clearly better. Whether that turns into better picks is still open."
        ),
        detail=(
            "Measured on 57,294 player-games from seasons the model was not fitted on, our "
            "error in predicting whether a player would take a snap fell by about 5%. Carried "
            "through into picks it moved accuracy from 52.14% to 52.24% -- two extra correct "
            "games out of 2,075, with a range around that change of -0.6 to +0.8 points. The "
            "best summary we have is roughly a 61% chance it helps at all, which is why it "
            "sits here as a weak positive to be combined with other weak positives rather "
            "than promoted on its own. Doubtful players who do not practise miss about 98% of "
            "the time, not the 85% our old hand-written table assumed."
        ),
        source="docs/modeling.md, docs/pool_edge_plan.md",
    ),
    Finding(
        question="Do bookmakers make predictable mistakes when they first post a line?",
        verdict="unproven",
        plain_answer=(
            "Published research says they might, and a pool that locks on Tuesday is exactly "
            "where that would pay. Four leads: teams coming off a playoff run look overrated "
            "in Week 1 (one study has them covering only 35.6% of the time), Week 2 lines stay "
            "anchored to Week 1's, last week's result gets over-weighted, and games nobody is "
            "watching move the most once money arrives. We have tested none of them."
        ),
        detail=(
            "We are being deliberately careful here: these come from the betting-market "
            "literature, not from our own data, and published market anomalies have a long "
            "history of evaporating when someone re-tests them. Each will be built as a "
            "feature, frozen, and scored once against a stretch of seasons it has never "
            "touched -- the same bar everything in the 'what works' section had to clear. "
            "Until then they belong in the 'interesting if true' column and nowhere else."
        ),
        source="docs/pool_edge_plan.md, ROADMAP.md",
    ),
    Finding(
        question="Can a pile of weak signals add up to one strong one?",
        verdict="unproven",
        plain_answer=(
            "That is the plan, and it has not been tested. We have several ideas that each "
            "look faintly positive and none of which can be proved alone: learned "
            "availability, playing-time-weighted injuries, the opening-line biases above, "
            "contract-year and off-field friction. Combining them into a single candidate and "
            "judging that once is legitimate -- averaging noisy signals cancels some of the "
            "noise. Doing it on seasons we have already mined would prove nothing at all."
        ),
        detail=(
            "The rule we have committed to: each new family of ideas draws a window of "
            "seasons from 2009-2025 that it has never touched, trains only on earlier games, "
            "and gets exactly one scored look. That registry does not exist yet, and building "
            "it is the work that has to happen before the combined candidate can be judged "
            "honestly. Pooling ten weak signals on the same seasons that suggested them "
            "proves only that we like our own ideas."
        ),
        source="docs/pool_edge_plan.md (MOD-07)",
    ),
    # -- no-edge -------------------------------------------------------------
    Finding(
        question="Does grading every single play tell us who is really better?",
        verdict="no-edge",
        plain_answer=(
            "It tells us plenty about football and nothing the line does not already know. We "
            "built the full play-by-play layer -- how many points each play was worth on "
            "average given down, distance and field position, how often teams stayed ahead of "
            "schedule, explosive plays, pressure allowed -- 48 columns of it. Added to the "
            "model it made the picks slightly worse. Adjusting it for opponent strength made "
            "them worse again."
        ),
        detail=(
            "This is the finding people find hardest to accept, so here is the strongest "
            "version. We re-tested the whole bundle on 1,247 games from 2013-2017 that it had "
            "never been scored on, with the test declared first. It came back at -0.08 points "
            "against the simpler model, and its margin error was resolvably worse. An earlier "
            "look at more recent seasons had shown +1.69 points -- that number is now on the "
            "record as an example of what happens when you compare enough versions on the "
            "same years. The layer stays in the codebase for future work on how games are "
            "shaped; it is not a source of edge."
        ),
        source="ROADMAP.md, docs/modeling.md",
    ),
    Finding(
        question="What about drive-level stats -- points per possession, field position?",
        verdict="no-edge",
        plain_answer=(
            "Same answer, one level up. We built a possession table -- points per drive, "
            "yards, plays, time of possession, scoring rate, turnover rate, and the same "
            "figures allowed to opponents -- and added it to a model that already had the "
            "play-level data. The picks got no better and the probability estimates got very "
            "slightly worse, with the range around that change straddling zero. Not a loss; "
            "just nothing there."
        ),
        detail=(
            "The null makes mechanical sense: drive statistics are mostly a re-summary of "
            "plays we had already counted, so the information was in the model twice. The "
            "layer is kept and reproducible because a future model of how games are shaped -- "
            "scores, pace, totals -- will want possessions as a building block. It is closed "
            "as an accuracy idea, and it will not be re-tuned on the same seasons."
        ),
        source="ROADMAP.md (PBP-03), docs/modeling.md",
    ),
    Finding(
        question="Does a team that keeps its lineup together beat the number?",
        verdict="no-edge",
        plain_answer=(
            "It looked that way, and then it did not. Lineup continuity -- the share of last "
            "week's starters who are back -- was the strongest player-side signal in our first "
            "search, apparently worth about a point of accuracy. We froze that exact recipe "
            "and scored it on four seasons it had never been tested on. On the 176 games where "
            "it disagreed with the simpler model it went 88-88. Exactly break-even."
        ),
        detail=(
            "That is the cleanest lesson this project has learned about itself. Run enough "
            "versions against the same eight seasons and one of them will look excellent "
            "whether or not anything is there. The fix was not a better continuity feature, "
            "it was a rule: every new family now earns its verdict on seasons it has never "
            "touched. The 2014-2017 window is marked spent for the player family and cannot "
            "be reused, which is a real cost we accepted in exchange for one trustworthy "
            "answer."
        ),
        source="ROADMAP.md, docs/modeling.md",
    ),
    Finding(
        question="Does it hurt a team when the players who normally carry the load are missing?",
        verdict="no-edge",
        plain_answer=(
            "Yes -- and the line already knows. We tested this on college football, where we "
            "have 8,933 clean games instead of 2,000, by tracking whether the players who "
            "normally take a team's snaps at quarterback and running back had actually played "
            "in the most recent game. Teams missing their usual load-carriers do play worse. "
            "Telling the model about it made the picks worse by about two thirds of a point."
        ),
        detail=(
            "This is a market-pricing result, not a football result: college spreads move on "
            "quarterback and lead-back news exactly like NFL ones, so by the time we see a "
            "line the disruption is inside it, and conditioning on it added noise instead of "
            "information. We had to answer a prerequisite first -- telling a temporary absence "
            "from a permanent departure -- and that study found only about 16-19% of college "
            "players holding a real role at the end of a season ever appear for that team "
            "again, which is why the feature counts only players who have already played this "
            "season. Because this failed, the plan to carry the same mechanism into the NFL "
            "is closed."
        ),
        source="docs/cfb_role_features.md",
    ),
    Finding(
        question="Can we rate players by what happens while they are on the field?",
        verdict="no-edge",
        plain_answer=(
            "We tried the basketball trick -- rate every player by how the team does with him "
            "out there, adjusted for who else is on the field -- using ten seasons of "
            "play-by-play participation data. It made the picks worse: 51.7% against 52.1% "
            "for the model without it, with the probability estimates degrading too. Football "
            "gives you sixty snaps a game with twenty-two players on the field, and the "
            "plus-minus maths that works in basketball starves on that."
        ),
        detail=(
            "The fit used competitive eleven-on-eleven plays only, three-season rolling "
            "windows, heavy shrinkage toward the average, and an extra discount for players "
            "with few snaps. It still had to rate between 1,758 and 2,872 players a season "
            "from as few as 24,000 plays. Narrower versions -- position groups, formations, "
            "specific unit matchups -- remain possible; this particular formulation is closed "
            "and will not be re-tuned on the same seasons."
        ),
        source="docs/data_feasibility.md, ROADMAP.md (PER-09)",
    ),
    Finding(
        question="Can we tell in advance which games will be blowouts and which will be close?",
        verdict="no-edge",
        plain_answer=(
            "No better than assuming every game has roughly the usual spread of outcomes. We "
            "built a model to predict how far a game would land from its expected margin -- "
            "using mismatch size, the total, how fast the teams play, and how early in the "
            "season it is -- and used it to widen or narrow each game's range of likely "
            "results. The predictions got measurably worse. Our plain one-size-fits-all range "
            "was already close to right."
        ),
        detail=(
            "We ran this on college football first, deliberately, because 8,933 games can "
            "resolve a difference this small and 2,000 cannot. The scale model did produce "
            "real variation -- the games it called widest came out roughly 20% wider than the "
            "ones it called narrowest -- and it still lost. That is a compliment to the "
            "simple version: the scatter "
            "of results around a good prediction is remarkably stable in football, and our "
            "50% and 80% ranges already land at almost exactly the rates they claim."
        ),
        source="docs/margin_variance.md",
    ),
    Finding(
        question="Does a web of who-beat-whom rank teams better than a plain rating?",
        verdict="no-edge",
        plain_answer=(
            "No. We built the season as a network -- beat a team that beat a good team and "
            "credit flows back to you, the same maths that once ranked web pages -- and "
            "compared it against an ordinary strength rating built from the identical games. "
            "Across eight test seasons the network version was chosen zero times. The plain "
            "rating was chosen three times, and the model with neither was chosen five."
        ),
        detail=(
            "The network version also made probability estimates and margin errors worse in "
            "most seasons, which is the tell that it was not merely unlucky. It is a "
            "genuinely elegant idea that turns out to measure roughly what an ordinary rating "
            "measures, with many more moving parts. Both are kept as research tools; neither "
            "is a default in anything we run."
        ),
        source="docs/modeling.md (MOD-15)",
    ),
    Finding(
        question="Do players in a contract year try harder?",
        verdict="no-edge",
        plain_answer=(
            "The published evidence says essentially no, at least not in a way that reaches "
            "the scoreboard. We went looking because it is exactly the sort of human factor a "
            "market of numbers might miss, and the research on an NFL contract-year effect is "
            "close to nothing. It has no place in the model and will not get one as a "
            "standalone idea."
        ),
        detail=(
            "It survives only as one weak input among several in a future combined candidate, "
            "where nothing has to be individually convincing to be worth carrying. This is "
            "what an honest 'no' looks like when someone else has already done the work: we "
            "did not spend one of our own limited untouched test windows re-discovering it."
        ),
        source="docs/pool_edge_plan.md",
    ),
    # -- context -------------------------------------------------------------
    Finding(
        question="How much of the answer is already in the betting line?",
        verdict="context",
        plain_answer=(
            "Almost all of it. The spread is the output of a market full of people who are "
            "very good at this and who move money the moment they disagree, and it is by far "
            "the strongest single input we have. Everything else in the model exists to nudge "
            "it, usually by a fraction of a point. If you take one idea from this page, take "
            "this one: we are not trying to predict football games. We are trying to find the "
            "few games a week where a very good number is slightly wrong."
        ),
        detail=(
            "In practice the model treats the line as its starting assumption and estimates "
            "only the correction, and those corrections are small -- half a point of "
            "disagreement is a big disagreement by our standards. Alongside the line it "
            "carries team form (a weighted average of recent scoring, turnovers, sacks and "
            "past cover margins that leans on the last few games), a team-strength rating, "
            "quarterback state, the injury report, rest, and basic schedule context."
        ),
        source="docs/modeling.md",
    ),
    Finding(
        question="What is the best we could possibly do?",
        verdict="context",
        plain_answer=(
            "About 54-55%, and 60% would mean we have a bug. Final scores scatter around even "
            "a perfect pregame prediction by about 13 points -- turnovers bounce, kickers "
            "miss, one-score games turn on one call. That noise sets the limit. The exchange "
            "rate is worth memorising: one point of line accuracy is worth about three points "
            "of pick accuracy, which is why tiny modelling gains matter and why huge results "
            "are impossible."
        ),
        detail=(
            "A perfect pregame oracle graded against a frozen Tuesday line would land around "
            "57%; against the sharper closing line, about 55-56%. Documented career bettors "
            "live in the mid-50s. So the ceiling is not physical, it is adversarial -- it "
            "measures how much better than the line-setter we can be -- and the pool's "
            "line-setter is handicapped on purpose by freezing its number midweek. That "
            "handicap is the entire opportunity. The corollary is a rule we actually enforce: "
            "any backtest showing 60% is a leak, and we have thrown out results for being too "
            "good."
        ),
        source="docs/pool_edge_plan.md",
    ),
    Finding(
        question="Are the picks we feel best about more likely to win?",
        verdict="context",
        plain_answer=(
            "No, and this one stings. Our single most confident pick of the week -- the game "
            "where the model disagrees most with the line -- has won 48.6% of the time across "
            "107 weeks. Sorting every pick into four confidence groups gives 53.2%, 47.3%, "
            "55.7%, 53.7%: no ladder, just noise. The model's answer to 'which side' carries "
            "the signal. Its answer to 'how sure are you' does not."
        ),
        detail=(
            "So until that changes, every pick is weighted equally, and the pool's one Best "
            "Pick per week costs nothing whichever game we assign it to -- they are all worth "
            "about the same 52.5%. Finding a confidence measure that genuinely ranks pick "
            "quality would be free points and it is high on the queue. The candidates: using "
            "the calibrated chance of covering rather than raw disagreement with the line, "
            "accounting for the key numbers 3 and 7 where NFL margins pile up, and calibrating "
            "separately by era. None of them are tested. This measurement was read-only -- no "
            "pick was selected on it -- which is why we can quote it at all."
        ),
        source="docs/opener_evaluation.md, docs/pool_edge_plan.md",
    ),
    Finding(
        question="What can't we see?",
        verdict="context",
        plain_answer=(
            "The things people who beat this market for a living actually use. We have no film "
            "-- we cannot see a guard's footwork or a blown coverage. We have no locker room, "
            "no beat-writer whispers, no coach quietly telling someone his starter is a decoy. "
            "We cannot see where the informed money is going, only where the line ended up. "
            "And we cannot see the weather forecast that existed when the line was posted, "
            "only the weather that happened."
        ),
        detail=(
            "We budget for this rather than pretend otherwise. Of the three to four accuracy "
            "points between us and a perfect pregame prediction, we concede about two to "
            "information we structurally cannot get. The reachable part -- maybe half a point "
            "to a point -- is better handling of thin evidence: what a backup quarterback is "
            "worth when we have seen twenty snaps of him, what a team is worth in Week 2, and "
            "borrowing college football as a starting guess where our NFL evidence runs out. "
            "That is the honest map of where the remaining room is."
        ),
        source="docs/pool_edge_plan.md",
    ),
    Finding(
        question="Do rest, travel and weather matter?",
        verdict="context",
        plain_answer=(
            "They matter to football and they ride along in the model as background, but none "
            "of them is an edge. Rest difference, divisional games, neutral sites, time of "
            "year, temperature and wind are all in there. They are small, and the market "
            "prices the obvious ones instantly -- everybody knows who is on a short week."
        ),
        detail=(
            "One honest limitation: the temperature and wind we hold are what the weather "
            "turned out to be, not the forecast that existed when the line was posted, so we "
            "treat those two columns as background rather than something to lean on. A proper "
            "decision-time weather feature needs an archived forecast source and has not been "
            "built. Travel distance, time-zone changes and body-clock effects are on the list "
            "and unbuilt as well."
        ),
        source="docs/data_feasibility.md, ROADMAP.md (ENV-01 to ENV-06)",
    ),
    Finding(
        question="Can we even make picks for the playoffs?",
        verdict="context",
        plain_answer=(
            "Now, yes. Until this month every table in the project stopped at Week 18 -- fine "
            "for research, useless for a pool that demands picks on all 13 playoff games. "
            "Playoff rows now exist, and they were built so that every regular-season number "
            "is byte-for-byte what it was before, because changing them would have quietly "
            "turned our measured model into a different, unmeasured one."
        ),
        detail=(
            "Playoff games see everything that came before them, including earlier rounds -- a "
            "Super Bowl row knows what both teams did in the divisional round -- while "
            "regular-season rows never see a playoff result, so a January game cannot leak "
            "backwards into the following September. Training still uses regular-season games "
            "only. One deliberate gap remains: we have not yet graded the model on historical "
            "playoff games. That is a fresh test with about 65 games available, and it gets "
            "declared before it is computed, not after."
        ),
        source="docs/postseason_support.md",
    ),
    Finding(
        question="Why not just keep testing ideas until something works?",
        verdict="context",
        plain_answer=(
            "Because we already did that, and we can measure what it cost. Roughly 130 to 150 "
            "versions of this model have been scored against the same eight seasons of "
            "results. Test that many things and the best of them looks good whether or not "
            "anything is there -- which is exactly why our prettiest numbers from those "
            "seasons (52.5-52.8%) are not evidence of anything, and why two of the three most "
            "promising leads from that era evaporated when re-tested on untouched seasons."
        ),
        detail=(
            "So we changed the rules. New ideas are scored on seasons that family has never "
            "used, with the method and the pass/fail line written down first; a used window is "
            "marked spent and never reused; and negative results are recorded permanently so "
            "nobody rediscovers them in two years and gets excited. The three untouched-window "
            "re-tests we have run came back -0.08 points, +0.00 points, and a failed "
            "prerequisite. That is what the discipline buys: not more wins, fewer fake ones."
        ),
        source="ROADMAP.md, docs/modeling.md",
    ),
    Finding(
        question="The pool scores one Best Pick a week. Can we tell which of our picks is best?",
        verdict="helps",
        plain_answer=(
            "Yes -- one signal out of three, and it is not the obvious one. The pick whose "
            "edge survives the widest range of line movement won 60% of the time as the "
            "week's Best Pick, against 51% for our picks overall. It cleared its pass mark "
            "twice: once on 2013-2015 graded against the standard line, then again on "
            "2020-2021 graded against the Tuesday line the pool actually uses. We use it "
            "to choose the Best Pick, and it changes nothing else -- the picks themselves "
            "are untouched."
        ),
        detail=(
            "The idea: re-score every pick at lines half a point either side of the real "
            "one, and measure how far the line can move before the pick stops being "
            "favoured. A pick that only works at exactly one number is fragile; one that "
            "survives four points in both directions is not. Read the 60% as a ranking that "
            "works, not as a rate to expect -- it rests on 35 weeks, the honest range around "
            "it runs from worse-than-nothing to enormous, and how well the signal orders the "
            "whole field is still unproven (the rank correlation misses significance). The "
            "reason to use it anyway is that the alternative -- picking one of our own picks "
            "arbitrarily -- has no evidence behind it at all, and this has two independent "
            "windows pointing the same way."
        ),
        source="docs/best_pick_ranker.md",
    ),
    Finding(
        question="Isn't our most confident pick the obvious Best Pick?",
        verdict="no-edge",
        plain_answer=(
            "No, and this is one of the more useful things we know. Taking the pick we are "
            "most confident in made the Best Pick materially WORSE: 41% against 49% for our "
            "picks overall. Ranking by how far our number sits from the market's did badly "
            "too, at 43%. Both were tested once on untouched seasons and both are closed."
        ),
        detail=(
            "This is the same result the track record shows from a different angle -- our "
            "confidence ordering carries no information, even after recalibrating the "
            "probabilities properly. The model's call on WHICH side to take is worth "
            "something; its opinion about HOW SURE it is, is not. That is why no pick on the "
            "picks page gets extra weight, and why the Best Pick is chosen by a completely "
            "different property of the pick rather than by confidence."
        ),
        source="docs/best_pick_ranker.md",
    ),
    Finding(
        question="Does stacking our leftover weak signals together beat the model we run?",
        verdict="unproven",
        plain_answer=(
            "It looked better and did not clear the bar. Adding learned injury availability, "
            "player-value weighting and three published early-season line biases scored 53.3% "
            "against the pool's line where our current model scored 51.3% on the same 456 "
            "games -- about two points better. Our pass mark was a 90% chance the improvement "
            "is real; it came out at 87%. So it is not promoted, and it is not being retuned "
            "and retried."
        ),
        detail=(
            "Both versions agreed on 407 of the 456 picks. The whole difference comes from "
            "the 49 they disagreed on, where the new version went 29-20 -- a net of nine "
            "picks, which is well inside what luck produces. The seasons involved are also "
            "ones we have already searched heavily, so the bar was rightly set high. The next "
            "honest test is to run it alongside the real model through 2026 and see what it "
            "does on games nobody has looked at yet. We since took the stack apart to see "
            "which ingredient was doing the work, and it was not the one we assumed -- see "
            "the next answer."
        ),
        source="docs/mod07_stack.md",
    ),
    Finding(
        question="Which part of that stack was actually doing the work?",
        verdict="context",
        plain_answer=(
            "Not the published line biases, which were the reason we built it. Taking them "
            "out changes almost nothing: injury availability and player-value weighting on "
            "their own score 53.1% against the model's 51.3%, and adding the three bias "
            "features on top moves that to 53.3%. So the biases are worth about a fifth of "
            "a point, with a coin-flip chance of being worth anything at all. Everything "
            "with a real lean came from knowing who is playing."
        ),
        detail=(
            "This matters because it changes what to build next. We had been treating the "
            "near-miss as evidence that published early-season line biases are worth "
            "chasing; it is not. On the 39 picks the bias features alone flipped, they went "
            "19-39 one way and 20-39 the other -- a one-pick difference. Separately, the "
            "biggest of those biases does not reproduce here at all: the published claim is "
            "that last year's playoff teams cover only about 36% as Week 1 favourites, and "
            "in our data they cover 52.5% over 120 such games. We are not building more of "
            "them. Availability is where the signal was, and that is the thread to pull."
        ),
        source="docs/mod07_stack.md, docs/rotation_registry.md",
    ),
    Finding(
        question="Plays outnumber games 166 to one. Shouldn't we learn from plays instead?",
        verdict="no-edge",
        plain_answer=(
            "No, and the reason is worth understanding because the number is seductive. "
            "There are about 780,000 plays behind 4,700 games. But we are predicting the "
            "game, and there are still only 4,700 of those to learn from. More plays make "
            "each team's average sharper; they do not create more things to learn from. "
            "Play data is also already most of what the model reads -- about 44 of its 79 "
            "inputs are built from plays."
        ),
        detail=(
            "We measured how much sharper. Knowing a team's per-play efficiency instead of "
            "just its points scored and allowed helps in weeks 1 to 3 and is worth nothing "
            "from week 4 onward, once a few games exist to average. Two specific ideas died "
            "here on measurement: forecasting the pace of a game (play counts barely vary, "
            "and games with more plays are LOWER-scoring blowouts, not higher -- teams kill "
            "the clock when ahead), and measuring how erratic a team is (a team's "
            "game-to-game spread turns out not to be a property of the team at all). One "
            "real gap survived: we measure defences about half as reliably as offences, and "
            "the standard fix for that has never been tested properly."
        ),
        source="docs/play_level_audit.md",
    ),
    Finding(
        question="How much history does the model need before its picks are trustworthy?",
        verdict="context",
        plain_answer=(
            "Less than we assumed, and there is no clean cut-off. We had a rule requiring "
            "500 finished games before the model would predict at all. Nobody had ever "
            "tested it. When we did, forced-pick accuracy turned out to be flat from 50 "
            "games all the way to 4,000 -- the rule was protecting against something that "
            "does not happen."
        ),
        detail=(
            "What DOES go wrong with little history is not being wrong, it is being loud: "
            "with 50 games the model moves the line by nine points on average, against under "
            "two points when fully trained. It is overconfident rather than misdirected, and "
            "the step that converts a prediction into a probability already fixes that. The "
            "practical consequence is about honesty rather than accuracy -- that unjustified "
            "500 was quietly deciding which seasons we are allowed to use to test future "
            "ideas, which is a permanent decision. It no longer does."
        ),
        source="docs/rotation_registry.md",
    ),
    Finding(
        question="Do football scores follow a bell curve?",
        verdict="context",
        plain_answer=(
            "Emphatically not. Almost 15% of games are decided by exactly three points -- "
            "nearly three times what a bell curve allows -- and seven is the next spike. "
            "The middle is stranger still: 346 games have ended +3 and 300 have ended -3, "
            "but only 13 in seventeen seasons have ended level, because overtime almost "
            "always breaks the tie. So the very centre really does have two humps with a "
            "hole between them."
        ),
        detail=(
            "Points arrive in threes and sevens, so some final margins are reachable many "
            "ways and others barely at all. Formally the test for a single peak is rejected "
            "outright. But here is the twist that matters: we do not predict the margin, we "
            "predict whether a team beats the spread -- and because the spread is a different "
            "number every week, those spikes get smeared out. Measured on the margin minus "
            "the spread, the lumpiness almost entirely vanishes and the shape comes back "
            "close to a bell curve. One thing survives: games pile up right ON the line about "
            "twice as often as a bell curve predicts, because the spread is deliberately set "
            "where the lump is. That is why a half-point matters enormously at a line of "
            "three and barely at all at a line of nine."
        ),
        source="docs/pool_edge_plan.md",
    ),
    Finding(
        question="Would a more careful statistical model squeeze more out of thin data?",
        verdict="no-edge",
        plain_answer=(
            "We built it to find out, and no. On 12,000 college games -- free to experiment "
            "on -- turning the caution dial across five orders of magnitude moves accuracy "
            "by less than a point, and turning it up, which is what the theory recommends, "
            "makes the thin-data games worse rather than better."
        ),
        detail=(
            "The reason is a structural fact worth remembering whenever someone proposes "
            "this kind of fix. Our pick is just which side of the line we land on. Being "
            "more cautious shrinks how far our number sits from the market's, but it cannot "
            "move it to the other side -- halving a number never changes its sign. So any "
            "method whose whole effect is to be more conservative cannot change a single "
            "pick, no matter how principled it is. It can improve how well-calibrated our "
            "stated confidence is, which is worth something, but not the thing we are "
            "actually scored on."
        ),
        source="docs/rotation_registry.md, ROADMAP.md",
    ),
)


# ---------------------------------------------------------------------------
# Closing section: the honesty rules
# ---------------------------------------------------------------------------

HONESTY_KICKER = "Before you trust a percentage"
HONESTY_TITLE = "How to read any number on this dashboard"
HONESTY_SUB = (
    "Four rules we hold ourselves to. They are the difference between a number that means "
    "something and a number that merely looks good."
)

HONESTY_RULES: tuple[HonestyRule, ...] = (
    HonestyRule(
        title="Every number comes from games the model had never seen",
        body=(
            "Nothing here is scored on games the model learned from. For each week, the model "
            "is rebuilt using only games that had already finished before that week's first "
            "kickoff, and then it picks once. It is slower, and the numbers come out lower "
            "than the alternative. That is the trade: a number you can believe instead of a "
            "number you can enjoy."
        ),
    ),
    HonestyRule(
        title="A single percentage is never the answer -- look for the range",
        body=(
            "52.5% is our best single guess; the honest statement is 'somewhere between about "
            "50.2% and 54.3% on new seasons'. Those ranges come from re-scoring the same games "
            "in whole-week and whole-season chunks, because games in the same week are not "
            "independent of each other. When a range crosses 50%, we say so rather than "
            "quoting the middle and moving on."
        ),
    ),
    HonestyRule(
        title="One look, decided in advance",
        body=(
            "Before a test runs we write down what is being measured, on which games, and "
            "what would count as passing. Then we run it once and record whatever comes out. "
            "No quiet re-runs with a different setting until it wins. When a result surprises "
            "us, the surprise goes into the record too, unedited -- the predeclaration is "
            "never rewritten after the fact."
        ),
    ),
    HonestyRule(
        title="Nothing that failed gets deleted",
        body=(
            "Negative results are the most trustworthy part of this record, because they are "
            "the part that hindsight cannot flatter. A quietly deleted failure just gets "
            "rediscovered in two years by someone who then over-fits it. If the 'tested, and "
            "no' section of this page ever gets shorter, something has gone wrong."
        ),
    ),
)

CLOSING_NOTE = (
    "And if a number on this dashboard ever reads 60%, treat it as a bug rather than a "
    "breakthrough. That is not modesty -- the sport's own randomness makes it arithmetically "
    "out of reach, and we have thrown out results before for being too good."
)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def findings_for(verdict: Verdict) -> tuple[Finding, ...]:
    """Every finding carrying ``verdict``, in declaration order."""

    return tuple(finding for finding in FINDINGS if finding.verdict == verdict)


def group_for(verdict: Verdict) -> VerdictGroup:
    """The section metadata for ``verdict``."""

    for group in GROUPS:
        if group.verdict == verdict:
            return group
    raise KeyError(verdict)
