# Public-site / dashboard UX rubric

Adopted 2026-08-24 as the hill-climbing target for UI work. Every page is
scored 0–10 per dimension, weighted to a /100 total. Scores must cite evidence
(file+line, or rendered behaviour); an unexplained score is invalid.

| # | Dimension | Weight | What 10 looks like |
|---|-----------|-------:|--------------------|
| 1 | Answerability | 12 | A first-time visitor learns what next week's picks are and how good the model has been within 30 seconds of landing |
| 2 | Narrative & hierarchy | 14 | Track record tells the edge start-to-finish; detail is progressively disclosed; nothing important is below a wall of process text |
| 3 | Provenance & honesty | 14 | Every number carries its grade/opener-vs-close, interval where relevant, and never implies profit/stable edge; historical accuracy distinct from game probabilities |
| 4 | Navigation & IA | 10 | Five plain-language destinations; current page obvious; ≤2 clicks from any page to any other; tooltips carry ids, headings carry names |
| 5 | Accessibility | 10 | WCAG AA contrast, semantic landmarks, alt text on every informative image/table summary, keyboard operable, no colour-only encoding |
| 6 | Visual consistency | 10 | One theme token set (spacing/type/colour scales), zero one-off inline styles for things the theme already covers |
| 7 | Data-viz quality | 8 | Axes start honest, intervals drawn not implied, no chartjunk, tables sortable where >20 rows |
| 8 | Trust signals | 8 | Model card link, last-updated timestamp with timezone, data cutoff date visible near headline numbers |
| 9 | Robustness & perf | 8 | Loads fast on 3G-scale budget, sane no-JS/no-data empty states, no layout explosion at 360px width |
| 10 | Mobile | 6 | Full information parity at phone widths; tables scroll or collapse deliberately |

## Scoring protocol

- One scorer per page plus one cross-cutting scorer (navigation/a11y/mobile span pages).
- Score only what you can evidence; mark `inferred` anywhere you guess.
- Output: per-dimension score, evidence bullets, top-3 improvement actions with
  estimated point gain, all written to `reports/wave1/<page>.md`.
- Baseline = sum of wave-1 scores. Hill-climb rule: a UI change ships only if
  its report argues ≥+2 weighted points on the affected dimensions without
  regressing provenance/honesty (dimension 3 may never drop).
