# Transaction realism for paper orders (BET-08)

This module makes the difference between a proposed paper position and an
actually observable fill explicit. It cannot contact a sportsbook or place a
wager.

## Execution contract

`execute_paper_orders` requires the requested stake, line, and price alongside
the quote actually available at one declared execution timestamp. It rejects a
row with a named reason when the book is unavailable, the quote is missing,
comes from the future, is stale, or kickoff has already closed the market
(**read**: `src/nfl_ats/transaction_realism.py`).

Known book limits cap the filled paper stake and preserve the unfilled
amount. An unknown limit remains visibly unknown rather than being guessed.
Every fill retains both requested and executed terms, plus side-oriented line
value, price-payout change, and break-even-probability change (**read**: the
same module).

## Settlement contract

`settle_paper_executions` grades only filled rows, uses the executed spread and
price rather than the requested values, preserves pushes as zero profit, and
leaves games without a final margin unsettled. Rejected orders cannot acquire a
fictional result or profit (**read**: `src/nfl_ats/transaction_realism.py`).

## Scope

The module is local paper-analysis plumbing. It does not choose a side, size a
portfolio, fetch a quote, submit an order, or claim that a historical return is
repeatable. Callers may feed it stake requests produced by `nfl_ats.portfolio`,
but execution and settlement remain a separate auditable stage.
