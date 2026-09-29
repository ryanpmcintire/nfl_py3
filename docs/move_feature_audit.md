# Served market-move input audit

**Measured:** all 164 differences among 799 saved 2023–2025 inputs arise from later scheduled `sun_early_close` captures replacing hourly terminal quotes. There are 633 exact agreements. LEAD-82 minus saved ranges from −2.0 to +2.5 spread points; the maximum absolute difference is 2.5. These are exact counts and observed ranges in a finite inventory, so sampling confidence intervals and `probability_positive` are not applicable.

**Inferred:** the saved values correctly implement their historical hourly-source definition. LEAD-82 correctly measures a later, richer source grid, but its extractor is not a like-for-like reconstruction of the saved input. This does not establish which definition predicts better. No outcomes, probabilities, fitting, scoring, registry decisions, or publication were used in this audit.

**Measured:** a separate one-line snapshot-cutoff bug was found and fixed in `src/nfl_ats/sharp_book_movement_features.py:119`. It changes **0 of the 799 saved values** in the audited replay and explains **0 of the 164 discrepancies**. Saved artifacts were not rewritten.

## Declared scope and method

**Read:** the protocol was saved in `docs/lanes/move-feature-audit.md` before inspecting discrepancy rows. The historical opener remains the frozen pool-line proxy; the audit uses the existing served four-term fit only to read its saved move and availability columns. The historical 2020–2025 source population is retained when deriving week anchors; the comparison is the 799 quote-available 2023–2025 frozen-fit rows. This is one integrity comparison with zero outcome looks, no fitted parameters or folds, and no in-sample/out-of-sample performance claim.

**Measured:** the command was `.tools/uv.exe run --no-sync python -` with an inline, outcome-free reconstruction, two numerical threads, projected parquet columns, and PyArrow threading disabled. It called the existing `sunday_move`, `lead82_unit1.load_quotes`, and `sharp_book_movement_features` functions, not either script's `main`. It compared hourly versus all capture labels under the same schedule, gates, leader books, endpoints, and median. The per-game ledger below contains feature evidence only: no predictions, outcomes, or fitted scores.

## Which builder owns which number

| Path | Source and calculation | Clock |
| --- | --- | --- |
| **Read:** `src/nfl_ats/pick_probability_fit.py:122` | Reads and hash-validates `artifacts/sunday_market_probability/20260920_fixed/market_move.parquet`; `leader_median_net` becomes the saved move. | Historical artifact, not current-line refresh. |
| **Read:** `scripts/sunday_market_probability_eval.py:183`, `:22` | Historical producer selects `intraday_hourly`; within each leader book it sums signed quote changes beginning Wednesday, then takes the median across books. | Before `min(kickoff, Sunday 12:45 ET)`; Monday quotes supply the pre-Wednesday anchor. |
| **Read:** `scripts/lead82_unit1.py:97`, `:183` | Adds source-season, snapshot and kickoff gates, but passes **all eligible decision labels** to the same `sunday_move` calculation. | Same 12:45 ET terminal cutoff; schedule-derived kickoff. |
| **Read:** `src/nfl_ats/pick_probability.py:659`, `src/nfl_ats/card_view.py:472` | Live `market_move_toward_home` loads live decision quotes and uses `leader_median_net_move` from the versioned sharp-book builder. | Sunday version caps at kickoff and supplied `now`. It does not impose a Sunday 16:00 cap. |
| **Read:** `src/nfl_ats/pick_refresh.py:151`, `src/nfl_ats/market_data.py:378` | Current market line uses latest quotes from all available books and the **median of levels**; refresh additionally requires today's captures. | Default 30-minute cross-book coverage window, separate from move construction. |
| **Read:** `src/nfl_ats/board_content.py:3149` | Displayed market-now line uses the **mean of levels**, with six-hour cross-book coverage. | Drops game summaries older than 48 hours. This is not the saved move input. |

**Read:** leader books are Bovada, Caesars/`williamhill_us`, and MyBookie (`sharp_book_movement_features.py:24`). The builder also calculates broader weighted and equal-book means, but the served move selects the leader **median of changes**, not either mean and not a difference of consensus line levels (`:140–153`; `pick_probability.py:700`).

**Read:** `market_data.py:161` negates the provider's home handicap into `home_spread_line`. A positive standardized home line means home favored; later minus earlier greater than zero means movement toward home. Both reconstructions use this convention.

## Discrepancy adjudication

**Measured:** the saved fit agrees exactly with its source move artifact. Hourly quotes plus the final schedule reconstruct **799/799** saved inputs exactly. All eligible capture labels reconstruct **799/799** LEAD-82 inputs exactly, including all its extra provenance gates.

| Mutually exclusive primary cause | Games |
| --- | ---: |
| Later scheduled terminal quote, same book and starting line | 164 |
| Changed book membership | 0 |
| Changed starting line | 0 |
| Mean versus median | 0 |
| Sign inversion | 0 |
| Deadline 12:45 versus the packet's 16:00 ET | 0 |
| Provenance gates or other unexplained causes | 0 |

**Measured:** all 336 changed book moves within the discrepant games end at `sun_early_close` quotes observed at **Sunday 12:25 ET**. Their hourly predecessors end at Sunday **10:55 ET in 331 cases** and **08:55 ET in 5 cases**; times here are shown to the minute. No participating book or starting spread changes. Counts by book are Bovada 112, MyBookie 108, Caesars 116. The discrepant game counts are 61 in 2023, 49 in 2024, and 54 in 2025.

**Measured:** switching either capture grid from `min(kickoff, Sunday 12:45 ET)` to the requested `min(kickoff, Sunday 16:00 ET)` changes **0/799** values. The two deadlines are different definitions even though this cache has no intervening endpoint that changes the median. With all scheduled captures through kickoff instead, 72/799 differ from LEAD-82 and 213/799 differ from saved; therefore a through-kickoff replay must not be mislabeled a 16:00 replay. Hourly-only through-kickoff values still match all 799 saved values. The live Sunday's broader clock/source policy is explicit in the implementation, not an explanation to silently retrofit onto the historical artifact.

**Measured:** replacing the same hourly per-book median with a mean changes 444/799 inputs; the actual LEAD-82 extractor does not do this. The 48-hour and 30-minute rules are not on either saved/reconstructed move path and cannot cause these 164 differences.

**Read:** the historical producer deliberately retains incumbent values when its original archive-kickoff reconstruction fails parity (`sunday_market_probability_eval.py:203–220`). **Measured:** its archive-derived clocks differ on two additional games, `2025_04_SEA_ARI` (saved +0.5, original-clock replay 0.0) and `2025_05_SF_LA` (saved +1.5, original-clock replay +2.0). They are not among the 164; using the final schedule reproduces their saved values too.

**Measured example:** `2023_01_CIN_CLE` has the same −2.5 starting spread at all three books. Hourly net moves are +1.0, +1.0, +0.5 (median +1.0); scheduled terminal moves are +3.5, +2.0, +2.5 (median +2.5). The extra +1.5 comes entirely from later observations. For `2025_05_DAL_NYJ`, the median changes from +1.5 to +4.0, the largest positive discrepancy.

## Minimal source fix and its limits

**Read:** the Sunday snapshot gate used `~include_sunday` on a Python Boolean. **Measured:** `~True | Series([False, True])` returns `[True, True]`; the intended `(not True) | Series([False, True])` returns `[False, True]`. The scalar bitwise inversion disabled the requirement that a quote's snapshot precede the decision cutoff.

The sole code edit replaces `~include_sunday` with `(not include_sunday)` at `src/nfl_ats/sharp_book_movement_features.py:119`. It preserves the legacy branch while restoring the existing Sunday branch's snapshot guard. **Read:** AGENTS.md's pregame-information and prediction-safety rules require enforcing this guard at runtime.

**Measured:** a direct function probe used the real `2025_05_DAL_NYJ` quote lines and a 12:45 ET cutoff, delaying only the six duplicate-source scheduled snapshot timestamps to exactly that cutoff. Before the fix it incorrectly returned +4.0; after the fix it returns +1.5, matching explicit exclusion of those snapshots. This is a deliberately modified provenance probe, **not evidence that the real game's captures arrived late**.

**Measured:** on the untouched cache there are zero observed-before-cutoff/snapshot-at-or-after-cutoff candidates among these 799 games at the 16:00 ET deadline. The repaired production function equals the explicitly snapshot-filtered function on every audited game. The source fix changes zero historical values; the source-grid difference still affects 164. No fixture, test file, or test function was added.

**Inferred next action:** for a parity handoff, LEAD-82 should retain its useful later-capture column but separately reproduce the saved hourly feature with `decision_label == "intraday_hourly"`. A richer-source research replay must declare that input change explicitly. This audit does not select a new feature definition or authorize refitting, serving, registry writes, or publication.

## Verification

**Measured after the final source edit:**

- `.tools/uv.exe run --no-sync ruff check src` — exit 0, all checks passed.
- `.tools/uv.exe run --no-sync mypy src` — exit 0, no issues in 253 source files.
- `.tools/uv.exe run --no-sync pytest -q -n 2 -k "pick_refresh or market or move" --basetemp <fresh-temp>` — exit 0, 56 passed, two missing-injury-feed fallback warnings, 13.53 seconds. The Boolean-inversion warnings are gone.
- Direct production-function replay/provenance probe — exit 0, delayed-snapshot result +1.5 equals explicitly gated +1.5; historical changes 0/799.
- `git diff --check -- src/nfl_ats/sharp_book_movement_features.py docs/move_feature_audit.md docs/lanes/move-feature-audit.md` — exit 0.

Commands use a writable temporary `UV_CACHE_DIR`; pytest gets a new dedicated temporary directory because the shared `pytest-of-Ryan` directory denied access before collection (initial exit 3). Numerical thread counts are capped at two and `-n 2` overrides the repository's `-n auto`. Temporary log names are `nfl-move-feature-audit-{replay,ruff,mypy,pytest}.log` under the Windows temporary directory. No persistent test coverage was added.

## Per-game feature ledger

**Measured:** every row has the same primary classification: **later `sun_early_close` terminal quote**. B = Bovada, C = Caesars, M = MyBookie. The last column gives each changed book's hourly → later net move; omitted books have identical net moves. All numbers are spread points. This ledger contains no predictions or outcome rows.

| Game | Saved hourly | LEAD-82 later | Later − saved | Changed book moves |
| --- | ---: | ---: | ---: | --- |
| 2023_01_CIN_CLE | +1 | +2.5 | +1.5 | B +1 → +3.5; M +1 → +2; C +0.5 → +2.5 |
| 2023_01_JAX_IND | +0.5 | +1.5 | +1 | B +1.5 → +2; M +0.5 → +1.5; C +0.5 → +1.5 |
| 2023_01_LA_SEA | -0.5 | -1 | -0.5 | B -1 → -1.5; M -0.5 → -1; C +0 → -0.5 |
| 2023_01_TEN_NO | -0.5 | +0 | +0.5 | M -0.5 → +0 |
| 2023_02_CLE_PIT | +0 | +0.5 | +0.5 | B +0 → +0.5; M +0 → +0.5; C +0 → +0.5 |
| 2023_02_NYG_ARI | +1.5 | +1 | -0.5 | B +1 → +0.5; M +1.5 → +1; C +1.5 → +1 |
| 2023_03_CAR_SEA | -0.5 | -1 | -0.5 | B -0.5 → -1 |
| 2023_03_CHI_KC | -0.5 | +0 | +0.5 | M -0.5 → +0 |
| 2023_03_DAL_ARI | -1 | -0.5 | +0.5 | B -1 → +0; M -1 → -0.5; C -1 → -0.5 |
| 2023_03_NO_GB | -1 | -3 | -2 | B -1 → -4; M -1 → -3; C -1 → -3 |
| 2023_04_MIN_CAR | -1.5 | -1 | +0.5 | B -1.5 → -0.5; M -1.5 → -1; C -1.5 → -1 |
| 2023_04_PIT_HOU | +0 | +0.5 | +0.5 | B +0 → +0.5; M +0 → +0.5 |
| 2023_04_TB_NO | +0.5 | +1 | +0.5 | B +0.5 → +1; M +0.5 → +1 |
| 2023_04_WAS_PHI | +0.5 | +1 | +0.5 | B +0.5 → +1 |
| 2023_05_NO_NE | +0 | +0.5 | +0.5 | B +0 → +1; M +0 → +0.5; C +0 → +0.5 |
| 2023_06_NO_HOU | +0 | -1 | -1 | M +0 → -1; C +0 → -1 |
| 2023_06_NYG_BUF | +1 | +1.5 | +0.5 | M +1 → +1.5 |
| 2023_07_LAC_KC | +0 | +0.5 | +0.5 | M +0 → +0.5; C +0 → +0.5 |
| 2023_07_SF_MIN | +0.5 | +0 | -0.5 | B +0.5 → +0 |
| 2023_08_BAL_ARI | -1.5 | -1 | +0.5 | M -1.5 → -1; C -1.5 → -1 |
| 2023_08_LV_DET | +0 | -0.5 | -0.5 | B +0 → -0.5; M -0.5 → -1; C +0 → -0.5 |
| 2023_08_NO_IND | -3.5 | -3 | +0.5 | B -3.5 → -4; M -3.5 → -3 |
| 2023_09_CHI_NO | +2 | +1.5 | -0.5 | B +1.5 → +1; M +2 → +1.5; C +2 → +1.5 |
| 2023_09_LA_GB | +1 | +0.5 | -0.5 | B +1 → +0.5; M +1 → +0.5; C +1 → +0.5 |
| 2023_09_SEA_BAL | +1 | +0.5 | -0.5 | B +1 → +0.5; M +1 → +0.5; C +1 → +0.5 |
| 2023_10_CLE_BAL | +0.5 | +0 | -0.5 | B +0 → -0.5; M +0.5 → +0; C +0.5 → +0 |
| 2023_10_DET_LAC | -0.5 | +0 | +0.5 | B -0.5 → +0; C -0.5 → +0 |
| 2023_10_HOU_CIN | -1 | -1.5 | -0.5 | B -1 → -1.5; C -1 → -1.5 |
| 2023_10_NYJ_LV | +0.5 | +0 | -0.5 | B +0.5 → +0; M +0.5 → +0; C +0.5 → -0.5 |
| 2023_11_DAL_CAR | +0 | -0.5 | -0.5 | B +0 → -0.5; M +0 → -0.5; C +0 → -0.5 |
| 2023_11_NYG_WAS | -0.5 | -1.5 | -1 | B -0.5 → -1.5; M -1 → -1.5; C -0.5 → -1 |
| 2023_11_NYJ_BUF | +1 | +1.5 | +0.5 | B +1 → +1.5; M +1 → +1.5; C +1 → +1.5 |
| 2023_12_NO_ATL | -2.5 | -2 | +0.5 | B -3 → -2.5; M -2.5 → -2 |
| 2023_12_PIT_CIN | -1.5 | -1 | +0.5 | B -1.5 → -1; C -1.5 → -1 |
| 2023_13_KC_GB | +0.5 | +1.5 | +1 | B +0.5 → +1.5; M +0.5 → +1.5; C +0.5 → +1.5 |
| 2023_13_MIA_WAS | +0.5 | +1 | +0.5 | B +0.5 → +1; M +0.5 → +1 |
| 2023_14_BUF_KC | -1 | -0.5 | +0.5 | M -1 → -0.5; C -1 → -0.5 |
| 2023_14_IND_CIN | +3 | +3.5 | +0.5 | B +1.5 → +2; M +3 → +3.5; C +3.5 → +4 |
| 2023_14_JAX_CLE | -1.5 | -0.5 | +1 | B -1.5 → -0.5; M -1.5 → -0.5; C -1.5 → -0.5 |
| 2023_14_SEA_SF | +3 | +3.5 | +0.5 | B +3 → +3.5; C +3 → +3.5 |
| 2023_14_TEN_MIA | -0.5 | +0.5 | +1 | B +0.5 → +1; M -1 → +0; C -0.5 → +0.5 |
| 2023_15_KC_NE | +0.5 | +0 | -0.5 | B -0.5 → -1.5; M +0.5 → +0; C +0.5 → +0 |
| 2023_15_NYG_NO | -0.5 | +0 | +0.5 | M -0.5 → +0; C -0.5 → +0 |
| 2023_15_NYJ_MIA | +0 | -1.5 | -1.5 | B +0 → -1; M -1 → -2; C +0 → -1.5 |
| 2023_15_PHI_SEA | +1 | +1.5 | +0.5 | B +0 → +1; M +1 → +1.5; C +1 → +1.5 |
| 2023_15_SF_ARI | +1.5 | +2 | +0.5 | B +1.5 → +2; C +1.5 → +2 |
| 2023_15_WAS_LA | +0 | +0.5 | +0.5 | B +0 → +0.5 |
| 2023_16_GB_CAR | +1 | +1.5 | +0.5 | B +0.5 → +1; M +1 → +1.5; C +1 → +1.5 |
| 2023_16_IND_ATL | +2 | +1.5 | -0.5 | B +2 → +1.5; M +2 → +1.5; C +2 → +1.5 |
| 2023_16_JAX_TB | -1 | -1.5 | -0.5 | M -1 → -1.5 |
| 2023_16_NE_DEN | +1 | +0.5 | -0.5 | B +1 → +0.5; M +1 → +0.5 |
| 2023_17_CAR_JAX | -3 | -3.5 | -0.5 | B -2.5 → -3; C -3 → -3.5 |
| 2023_17_CIN_KC | -0.5 | +0 | +0.5 | M -0.5 → +0 |
| 2023_17_LAC_DEN | -2 | -2.5 | -0.5 | B -2 → -2.5 |
| 2023_17_NE_BUF | +2 | +2.5 | +0.5 | B +2 → +2.5; M +2 → +2.5; C +2.5 → +3 |
| 2023_17_PIT_SEA | +1 | +0.5 | -0.5 | B +1 → +0.5; M +1 → +0.5 |
| 2023_17_TEN_HOU | +0.5 | +1.5 | +1 | B +0 → +1.5; M +0.5 → +1.5; C +0.5 → +2 |
| 2023_18_CLE_CIN | +1.5 | +1 | -0.5 | B +1.5 → +1; M +1.5 → +1; C +1.5 → +1 |
| 2023_18_DAL_WAS | +0 | -0.5 | -0.5 | B +0 → -0.5; C +0 → -0.5 |
| 2023_18_NYJ_NE | -0.5 | +0 | +0.5 | M -0.5 → +0 |
| 2023_18_TB_CAR | +1.5 | +1 | -0.5 | B +1.5 → +1; M +1.5 → +1; C +1.5 → +1 |
| 2024_01_DAL_CLE | +0 | -0.5 | -0.5 | B -0.5 → +0; M +0 → -0.5; C +0 → -0.5 |
| 2024_01_WAS_TB | +0 | +0.5 | +0.5 | M +0 → +0.5 |
| 2024_02_ATL_PHI | -0.5 | -1 | -0.5 | B -0.5 → -1; M +0 → -1; C -0.5 → -1 |
| 2024_02_CHI_HOU | -0.5 | -1 | -0.5 | B -0.5 → -1; M -0.5 → -1 |
| 2024_02_IND_GB | +0.5 | +1 | +0.5 | B +0.5 → +1; C +0.5 → +1 |
| 2024_02_LAC_CAR | +1.5 | +2 | +0.5 | B +2 → +2.5; M +1 → +2; C +1.5 → +2 |
| 2024_02_LA_ARI | -1.5 | -2.5 | -1 | M -1.5 → -2.5 |
| 2024_02_NO_DAL | -0.5 | +0 | +0.5 | M -0.5 → +0; C -0.5 → +0 |
| 2024_02_TB_DET | +0.5 | +1 | +0.5 | B +0.5 → +1; C +0.5 → +1 |
| 2024_03_BAL_DAL | +0.5 | +0 | -0.5 | M +0.5 → +0; C +0 → -0.5 |
| 2024_04_JAX_HOU | -1.5 | -1 | +0.5 | M -1.5 → -1 |
| 2024_05_MIA_NE | -2.5 | -3 | -0.5 | B -2.5 → -3; C -2.5 → -3 |
| 2024_07_DET_MIN | -1.5 | -0.5 | +1 | B -1.5 → -0.5; M -1 → -0.5; C -1.5 → -0.5 |
| 2024_07_LAC_ARI | +0.5 | +1 | +0.5 | B +0.5 → +1.5; C +0.5 → +1 |
| 2024_07_MIA_IND | +0 | -0.5 | -0.5 | B +0 → -0.5 |
| 2024_08_ATL_TB | +0.5 | +1 | +0.5 | C +0.5 → +1 |
| 2024_08_TEN_DET | +0.5 | +1 | +0.5 | M +0.5 → +1; C +0 → +1 |
| 2024_09_CHI_ARI | +2 | +1.5 | -0.5 | B +0.5 → +0; C +2 → +1.5 |
| 2024_09_LAC_CLE | +0.5 | +0 | -0.5 | M +0.5 → -0.5; C +1 → +0 |
| 2024_09_LA_SEA | +0 | +0.5 | +0.5 | M +0 → +0.5 |
| 2024_10_ATL_NO | -0.5 | +0 | +0.5 | B -1 → -0.5; M -0.5 → +0 |
| 2024_10_DEN_KC | -0.5 | -1 | -0.5 | C -0.5 → -1 |
| 2024_10_NE_CHI | +0.5 | +0 | -0.5 | B +0.5 → +0 |
| 2024_11_ATL_DEN | -0.5 | +0 | +0.5 | B -0.5 → +0 |
| 2024_11_GB_CHI | +0.5 | +0 | -0.5 | B +1 → +0; M +0.5 → +0 |
| 2024_11_IND_NYJ | +0 | +0.5 | +0.5 | B +0.5 → +1; M +0 → +0.5 |
| 2024_11_KC_BUF | -0.5 | +0 | +0.5 | M -0.5 → +0; C -0.5 → +0 |
| 2024_12_ARI_SEA | +0 | +2 | +2 | B +0 → +2; C +0 → -1 |
| 2024_12_TEN_HOU | -1 | -0.5 | +0.5 | C -1 → -0.5 |
| 2024_13_HOU_JAX | +1.5 | +2 | +0.5 | M +1.5 → +2; C +1.5 → +2 |
| 2024_13_SEA_NYJ | +3 | +3.5 | +0.5 | B +3 → +3.5; C +3 → +3.5 |
| 2024_14_NO_NYG | +0 | -0.5 | -0.5 | M +0 → -0.5; C +0 → -0.5 |
| 2024_14_NYJ_MIA | +0 | +0.5 | +0.5 | M +0 → +0.5; C +0 → +0.5 |
| 2024_15_BUF_DET | +0.5 | +0 | -0.5 | C +0.5 → +0 |
| 2024_15_CIN_TEN | -0.5 | -1 | -0.5 | C -0.5 → -1 |
| 2024_15_IND_DEN | +0.5 | +1 | +0.5 | M +0.5 → +1; C +0.5 → +1 |
| 2024_15_KC_CLE | -0.5 | +0 | +0.5 | B -0.5 → +0; M -0.5 → +0; C -0.5 → +0 |
| 2024_15_MIA_HOU | +0 | -0.5 | -0.5 | C +0 → -0.5 |
| 2024_16_ARI_CAR | -1 | -2 | -1 | B -1.5 → -2; M -1 → -2; C -0.5 → -1 |
| 2024_16_CLE_CIN | +2 | +2.5 | +0.5 | B +2 → +2.5; M +2 → +2.5; C +2 → +2.5 |
| 2024_16_DET_CHI | +0 | -0.5 | -0.5 | B +0 → -0.5 |
| 2024_16_NYG_ATL | +0.5 | +1 | +0.5 | C +0.5 → +1 |
| 2024_16_TEN_IND | +0 | +0.5 | +0.5 | B +0 → +0.5; M +0 → +0.5; C +0 → +0.5 |
| 2024_17_LV_NO | -0.5 | +0 | +0.5 | C -0.5 → +0 |
| 2024_18_BUF_NE | -1 | -0.5 | +0.5 | B -1 → -0.5; M -0.5 → +0; C -1 → -0.5 |
| 2024_18_CHI_GB | +0.5 | +1 | +0.5 | M +0.5 → +1; C +0.5 → +1 |
| 2024_18_HOU_TEN | +2 | +2.5 | +0.5 | M +1 → +2; C +2 → +2.5 |
| 2024_18_NYG_PHI | -1 | +0 | +1 | B -1 → +0; M -1.5 → -0.5; C -0.5 → +0 |
| 2024_18_SEA_LA | -2 | -1.5 | +0.5 | B -2.5 → -2; C -2 → -1.5 |
| 2025_01_CIN_CLE | +1 | +0.5 | -0.5 | B +1 → +0 |
| 2025_01_MIN_CHI | +0 | +0.5 | +0.5 | C +0 → +0.5 |
| 2025_01_TB_ATL | +2.5 | +3 | +0.5 | M +1 → +3 |
| 2025_02_CHI_DET | +1 | +0.5 | -0.5 | B +0.5 → +0; M +1 → +0.5 |
| 2025_02_CLE_BAL | +1 | +1.5 | +0.5 | B +1 → +1.5; M +1 → +2 |
| 2025_02_LA_TEN | +0 | +0.5 | +0.5 | B +0 → +0.5; M +1 → +0.5 |
| 2025_02_NYG_DAL | -0.5 | -1 | -0.5 | B -0.5 → -1; C -0.5 → -1 |
| 2025_03_DAL_CHI | +0 | -0.5 | -0.5 | B +0 → -0.5 |
| 2025_03_DEN_LAC | +0.5 | +0 | -0.5 | C +0.5 → +0 |
| 2025_03_IND_TEN | -1.5 | -2.5 | -1 | B -1.5 → -2; M -1.5 → -2.5; C -2 → -2.5 |
| 2025_04_CHI_LV | +0.5 | +1 | +0.5 | B +0.5 → +1 |
| 2025_04_LAC_NYG | +0 | +0.5 | +0.5 | B +0.5 → +1; M +0 → +0.5; C +0 → +0.5 |
| 2025_04_TEN_HOU | +0 | +0.5 | +0.5 | B +0 → +0.5; C +0 → +0.5 |
| 2025_05_DAL_NYJ | +1.5 | +4 | +2.5 | B +1.5 → +4; M +1.5 → +3.5; C +1.5 → +4 |
| 2025_05_DET_CIN | +0 | +0.5 | +0.5 | M +0 → +0.5 |
| 2025_05_HOU_BAL | +0 | -1 | -1 | B +0 → -1; M +0 → -1; C +0 → -1 |
| 2025_05_NYG_NO | -0.5 | +0 | +0.5 | C -0.5 → +0 |
| 2025_06_CHI_WAS | +0 | +0.5 | +0.5 | B +0 → +0.5; M +0 → +0.5 |
| 2025_06_CLE_PIT | +0.5 | +1 | +0.5 | B +0.5 → +1; M +1 → +1.5; C +0.5 → +1 |
| 2025_07_CAR_NYJ | +2 | +2.5 | +0.5 | C +0.5 → +3 |
| 2025_07_NO_CHI | -1 | -1.5 | -0.5 | M -1 → -1.5 |
| 2025_08_NYG_PHI | +0.5 | +0 | -0.5 | M +0.5 → +0 |
| 2025_09_CHI_CIN | +0 | -0.5 | -0.5 | C +0 → -0.5 |
| 2025_10_BAL_MIN | -0.5 | -1 | -0.5 | B -0.5 → -1; M -0.5 → -1 |
| 2025_10_BUF_MIA | +1 | +1.5 | +0.5 | B +1.5 → +2; M +0.5 → +1.5; C +1 → +1.5 |
| 2025_10_CLE_NYJ | +0.5 | +1 | +0.5 | B +0.5 → +1; C +0.5 → +1 |
| 2025_10_DET_WAS | +1 | +1.5 | +0.5 | C +1 → +1.5 |
| 2025_10_JAX_HOU | +0 | -1 | -1 | M +0 → -2; C +0 → -0.5 |
| 2025_10_PIT_LAC | -0.5 | +0 | +0.5 | B +0 → +0.5; M -0.5 → +0; C -0.5 → +0 |
| 2025_11_GB_NYG | +0 | -0.5 | -0.5 | B +0 → -0.5 |
| 2025_11_TB_BUF | +0.5 | +1 | +0.5 | M +0.5 → +1; C +0.5 → +1 |
| 2025_12_IND_KC | +0 | +1 | +1 | B +0.5 → +1.5; M +0 → +0.5; C +0 → +1 |
| 2025_13_JAX_TEN | +1 | +0.5 | -0.5 | B +1 → +0.5; M +1 → +0.5; C +1 → +0.5 |
| 2025_13_LA_CAR | +0 | +0.5 | +0.5 | B +0 → +0.5; C +0 → +0.5 |
| 2025_14_IND_JAX | +0 | -1 | -1 | B +0.5 → -0.5; M +0 → -1; C +0 → -1 |
| 2025_14_LA_ARI | -1.5 | -1 | +0.5 | M -2 → -1; C -1.5 → -1 |
| 2025_14_NO_TB | -0.5 | -1 | -0.5 | M -0.5 → -1; C -0.5 → -1 |
| 2025_14_PHI_LAC | +0.5 | +1 | +0.5 | M +0.5 → +1; C +0.5 → +1 |
| 2025_14_WAS_MIN | -3 | -2.5 | +0.5 | B -3 → -1 |
| 2025_15_ARI_HOU | +1 | +0.5 | -0.5 | B +1 → +0.5; C +1 → +0.5 |
| 2025_15_MIN_DAL | +0 | -0.5 | -0.5 | B +0 → -0.5; M -0.5 → -1 |
| 2025_15_NYJ_JAX | +1.5 | +1 | -0.5 | B +1.5 → +1; C +1.5 → +1 |
| 2025_16_ATL_ARI | -0.5 | +0 | +0.5 | B -0.5 → +0 |
| 2025_16_LAC_DAL | -0.5 | -1 | -0.5 | B -0.5 → -1 |
| 2025_17_JAX_IND | +1.5 | +2.5 | +1 | B +2.5 → +3.5; M +1.5 → +2.5; C +1.5 → +2.5 |
| 2025_17_NE_NYJ | +0 | +0.5 | +0.5 | B +0 → +0.5; M +0 → +0.5; C +0 → +0.5 |
| 2025_17_NYG_LV | -4 | -4.5 | -0.5 | M -3.5 → -4.5; C -4 → -4.5 |
| 2025_17_PHI_BUF | +1 | +1.5 | +0.5 | B +1 → +1.5; C +1 → +1.5 |
| 2025_17_PIT_CLE | -1.5 | -1 | +0.5 | B -1.5 → -1; C -1.5 → -1 |
| 2025_17_SEA_CAR | +0 | +0.5 | +0.5 | C +0 → +0.5 |
| 2025_17_TB_MIA | +1 | +0.5 | -0.5 | B +1 → +0.5 |
| 2025_18_CLE_CIN | +1 | +2 | +1 | C +1 → +2 |
| 2025_18_IND_HOU | +0 | -0.5 | -0.5 | B +0.5 → +0; M +0 → -0.5; C -0.5 → -1 |
| 2025_18_NYJ_BUF | +3 | +5 | +2 | B +1 → +3; M +3 → +5; C +3 → +5 |

## Reproduction command

Run from the repository root with the locked environment and writable temporary UV cache. This bash-compatible command reads only input and feature columns, does not invoke a fit or score, and writes nothing:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .tools/uv.exe run --no-sync python - <<'PY'
import sys
sys.path.insert(0, "scripts")
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from sunday_market_probability_eval import sunday_move
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS, sharp_book_movement_features
threadpool_limits(2)
f = pq.read_table("tests/scratch/codex/lead82_unit1/features.parquet", columns=["game_id", "season", "week", "kickoff", "week_first_commence_utc", "reconstructed_late_move_toward_home", "late_available"], use_threads=False).to_pandas(use_threads=False)
s = pq.read_table("artifacts/pick_probability/20260929T192747Z/per_game.parquet", columns=["game_id", "market_move_toward_home"], use_threads=False).to_pandas(use_threads=False)
a = f.merge(s, on="game_id", validate="one_to_one")
a = a.loc[a.season.between(2023, 2025) & a.late_available].set_index("game_id")
g = f[["game_id", "kickoff", "week_first_commence_utc"]].rename(columns={"kickoff": "commence_time_utc"})
columns = ["nflverse_game_id", "bookmaker_key", "observed_at_utc", "bookmaker_last_update_utc", "home_spread_line", "commence_time_utc", "market", "snapshot_timestamp_utc", "archive_season", "decision_label"]
q = pq.read_table("artifacts/sharp_book_weighted_movement/spread_quotes.parquet", columns=columns, filters=[("bookmaker_key", "in", list(LEADER_BOOKS))], use_threads=False).to_pandas(use_threads=False)
h = q.loc[q.decision_label.eq("intraday_hourly") & q.archive_season.isin([2023, 2024, 2025])]
for name, source in [("hourly", h), ("all", q)]:
    a[name] = sunday_move(source, g).set_index("game_id").sunday_move
print("games", len(a), "hourly versus saved mismatches", int(a.hourly.ne(a.market_move_toward_home).sum()))
print("all versus LEAD82 mismatches", int(a["all"].ne(a.reconstructed_late_move_toward_home).sum()))
delta = a["all"] - a.hourly
print("discrepancies", int(delta.ne(0).sum()), "range", delta.min(), delta.max())
print("by season", a.loc[delta.ne(0)].groupby("season").size().to_dict())
one = g.loc[g.game_id.eq("2025_05_DAL_NYJ")].copy()
one["cutoff_utc"] = pd.Timestamp("2025-10-05T16:45:00Z")
r = q.loc[q.nflverse_game_id.eq("2025_05_DAL_NYJ")].copy()
mask = r.decision_label.eq("sun_early_close") & r.observed_at_utc.ge("2025-10-05T16:00:00Z") & r.observed_at_utc.lt("2025-10-05T17:00:00Z")
r.loc[mask, "snapshot_timestamp_utc"] = one.cutoff_utc.iloc[0]
actual = sharp_book_movement_features(r, one, include_sunday=True).leader_median_net_move.iloc[0]
expected = sharp_book_movement_features(r.loc[r.snapshot_timestamp_utc.lt(one.cutoff_utc.iloc[0])], one, include_sunday=True).leader_median_net_move.iloc[0]
print("modified-snapshot probe", actual, expected, "expected both 1.5")
PY
```
