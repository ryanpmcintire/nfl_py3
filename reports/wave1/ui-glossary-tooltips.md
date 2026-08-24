# UI glossary tooltips — wave-1 vocabulary finding executed

Task: `ui-glossary-tooltips` · Branch: `swarm/ui-glossary-tooltips`

## Finding executed

Wave-1 UX report for `docs/index.html` (`reports/wave1/uix-index.md`, commit
`780f3c4`, **read** this session via `git show`) flagged that "P+",
"Ledger mini", "Evidence P+", and "Challenger watch" are unexplained at point
of use; a newcomer cannot decode "P+" on the picks page.

## Change

All in the generator (`src/nfl_ats/public_board.py`) — no generated page was
hand-edited:

- Added a module-level `_GLOSSARY` dict with plain-language expansions for the
  four terms, plus `glossary_abbr(term)` which renders
  `<abbr title="...">term</abbr>` — the same tooltip mechanism the week board
  already uses for its flip/best flags (**read**,
  `src/nfl_ats/public_board.py`, `best-flag`/`flip-flag` spans).
  Glosses state what P+ is ("confidence a measured effect is real rather than
  luck; 0.50 would be a coin flip. It is not an accuracy rate or a profit
  claim"), keeping provenance honesty (dimension 3) intact.
- Point-of-use applications:
  - "Ledger mini" panel heading → abbr tooltip.
  - "Evidence P+" column header in `_ledger_mini_table` → abbr tooltip.
  - "Challenger watch" panel heading → abbr tooltip.
  - Bare "P+" values in ledger-mini rows and challenger-watch lines → abbr on
    the "P+" token.
  - The two inline "P+ x.xx" policy-evidence sentences (arrest-flip note and
    track-record story) also carry the abbr.

Visible text is unchanged except that terms are now wrapped in `<abbr>`; the
tooltip copy deliberately avoids overclaiming adjectives ("strongest" tripped
the existing `"strongest" not in page` guard during testing and was reworded to
"best"/"highest").

## Tests

`tests/test_public_board.py` (**measured** this session):

- `test_glossary_covers_every_research_vocabulary_term` — all four flagged
  terms produce non-empty tooltips (>20 chars).
- `test_glossary_abbr_rejects_unknown_terms` — unknown term raises `KeyError`.
- `test_render_picks_page_glosses_research_vocabulary_at_point_of_use` —
  rendered picks page contains `>P+</abbr>`, `>Ledger mini</abbr>`,
  `>Evidence P+</abbr>`, `>Challenger watch</abbr>`, a glossed ledger cell, and
  passes `assert_public_safe`.
- `test_render_picks_page_glosses_survive_without_challengers` — default build
  (no challengers.json) still glosses all three panel/column labels.
- Updated three existing string assertions to the new markup:
  `test_ledger_mini_column_header_reads_evidence_p_plus`,
  `test_challenger_watch_shows_top_six_and_collapses_the_rest`,
  and the two `"P+ 0.86"` assertions in the track-record / arrest-flip tests.
  No safety guardrail was weakened — only exact-copy expectations moved to the
  tooltip-bearing form; every `assert_public_safe` call remains.

## Gates (all run this session)

```
ruff format --check .   -> 636 files already formatted
ruff check .            -> All checks passed!
mypy src                -> Success: no issues found in 105 source files
pytest                  -> 1859 passed, 5 skipped
```

(basetemp outside the repo per worker constitution.)

## Scope notes

- Only the four flagged terms were glossed; other pages' prose explanations of
  P+ (models page ledger explainer) already existed and are unchanged.
- Documentation-only? No — code + tests, so all four gates run.
