# LEAD-80 source review

**Measured:** the inventory command completed once with exit 0. It inspected
149,356 file names: 95,805 under `data` and 53,551 under `artifacts`. Broad name
matching found 2,231 candidates; that count includes forecasts and unrelated
football articles, not 2,231 eligible observations. Five unreadable directories
are listed in `docs/lead80_inventory.md`; absence outside inspected sources is
not established. The generated candidate table was shortened after execution
to show the relevant source examples; inventory counts were retained.

**Measured:** registered `pool_observables` snapshots found = 0, including
2020-2025 snapshots = 0. **Read:** a separate source exists at `data/splash`:

| Local source | Season/week | What the metadata establishes |
|---|---|---|
| `data/splash/2026_week01_20260908_noon.json:4` | 2026/1 | Capture September 8, 12:45 ET; pick lock September 13, 16:00 ET |
| `data/splash/2026_week02_20260915_1302.json:3` | 2026/2 | Capture September 15, 13:02 ET; pick lock September 20, 16:00 ET |
| `data/splash/2026_week03_20260922_2019.json:3` | 2026/3 | Capture September 22, 20:19 ET; pick lock September 27, 16:00 ET |
| `data/splash/2026_week04_20260929_1514.json:3` | 2026/4 | Capture September 29, 15:14 ET; pick lock October 4, 16:00 ET |
| `data/splash/field/2026_week01_field_distribution.tsv:1` | 2026/1 | Browser-read slate summary captured September 23; 250 entries |
| `data/splash/field/2026_week02_field_distribution.tsv:1` | 2026/2 | Browser-read slate summary captured September 23; 250 entries |

**Read:** the week 1 JSON at line 18 and week 2 JSON at line 148 say
"Predict the total combined score." This clarifies the current input beyond
the generic `final_score_last_game` code default. It does not authenticate
closest-total loss, secondary tie handling, a separate guess deadline, or the
rules in 2020-2025. Week 1 line 22 describes half-point lines, so its displayed
slate cannot push; the code's generic 0.5 push default is not historical proof.

**Read:** both field-table headers at line 2 contain `away`, `home`,
`away_score`, `home_score`, `team`, `line`, `result`, `picks`, and `best_picks`.
They omit entrant identifiers and tiebreak guesses. No field outcome rows were
read or scored. The capture dates follow those weeks' pick locks; these could
only be evaluation observations, not pre-deadline field inputs.

**Inferred:** the inventory unit is complete; conditional replay remains
source-gated by ROADMAP.md:863 and Protocol B at
`docs/lanes/ideation-2026-09-29b.md:12`. No authenticated 2020-2025 entrant-level
cards/guesses or matching historical rules were identified. The 2026 sources
do not satisfy the fixed population, and aggregate counts cannot reconstruct
which entrants tied. The protocol was not expanded after inventory.

**Measured:** decisive games evaluated = 0; outcome looks executed = 0;
folds fitted = 0 of 6 scheduled. **Read:** declared look budget = 445.
IS/OOS metrics and gaps, fold coefficients, season stability, reliability bands,
95% intervals and `probability_positive` are not estimable. No effect or
negative verdict was manufactured, and no registry command is warranted.

Verification: `UV_NO_CACHE=1 UV_OFFLINE=1 .tools/uv.exe run --no-sync python scripts/lead80_unit1.py`
(bash-compatible spelling of the environment used). The default uv cache was
unreadable during formatting; disabling the cache allowed the locked local
environment to run without network access. No new tests or operational jobs.
**Measured:** `UV_NO_CACHE=1 UV_OFFLINE=1 .tools/uv.exe run --no-sync ruff check --output-format concise scripts/lead80_unit1.py`
passed after formatting; the scoped `git diff --check` also returned exit 0.
