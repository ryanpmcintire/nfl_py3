# UI/UX baseline — wave 1 scoring (2026-08-24)

Scores produced by swarm scorers against `docs/uiux_rubric.md`. Hill-climb
target: every wave must gain >= +2 weighted points on affected dimensions
without regressing dimension 3 (provenance & honesty).

| Page | Baseline | Scorer report |
|------|---------:|---------------|
| index.html | **63.8 / 100** | reports/wave1/uix-index.md |
| track_record.html | **63.8 / 100** | reports/wave1/uix-track-record.md |
| findings.html | **65.0 / 100** | reports/wave1/uix-findings.md |
| cross-cutting (nav/a11y/mobile) | **~64.0 / 100** | reports/wave1/uix-crosscutting.md |
| models.html | *qualitative only* — rescore after fix below | reports/wave1/uix-models.md |

**Repo-wide UX baseline: ~64 / 100.**

## Known defects already identified by scorers

1. **models.html P+ gap (dimension 3, highest priority):** three finding-ledger
   rows carry intervals without `probability_positive`:
   - Model + seven-rule stat stack: [-1.100, 5.000]
   - Best Pick by calibrated probability: [-3.920, 11.760]
   - Injury-news refresh flip: [0.790, 31.670]
   Fix: surface P+ next to each interval (data exists in registry), then rescore.
2. **index.html answerability:** historical accuracy reachable only via a small
   link; no headline accuracy figure on the landing page.
3. **Insider vocabulary at point of use** ("P+", "Ledger mini", "Evidence P+",
   "Challenger watch") unexplained on first mention.
4. Accessibility is the weakest dimension cluster (~5/10 across pages).

## Estimated hill-climb headroom

track_record scorer estimates ~+17.3 weighted points available from its top-3
improvements alone (63.8 -> ~81). Cross-page consolidation in
reports/wave1/uix-crosscutting.md ranks the top-5 backlog.
