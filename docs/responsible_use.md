# Responsible use

This is a research and paper-decision project (`AGENTS.md`, "Research
invariants": "This is a research and paper-decision project; do not add
automated wagering."). Every ledger this codebase writes -- the
paper-decision ledger (`nfl_ats.clv.record_paper_decisions`), the
pick-revision ledger (`nfl_ats.pick_refresh`), and every prospective or
challenger recorder -- stores a predicted side and a probability against a
market line; none of them places a real-money wager, and no dependency,
module, or script in this repository integrates with a sportsbook, betting
exchange, or wagering-placement API. Paper mode is therefore the only mode:
there is no code path from a model prediction to money changing hands, and
that absence is enforced in code, not just stated here -- `tests/test_no_wager_path.py`
fails the build if a wagering-client dependency or a wager-placement code
path is ever added. If a reader chooses to wager real money on these
predictions, that decision and its consequences are entirely their own; see
README.md's "Responsible use" section for the market-facing cautions this
project already publishes (track the exact line, price, book, and decision
timestamp; include pushes and vig; report uncertainty; never wager money you
cannot afford to lose).
