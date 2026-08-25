# ref-board-theme — execution report

**Task:** replace inline HTML/CSS in `src/nfl_ats/public_board.py` and
`src/nfl_ats/dashboard/findings_content.py` with theme tokens from
`src/nfl_ats/dashboard/theme.py` *where the mapping is mechanical*; preserve
rendered output byte-for-byte where possible; note each deviation where the
token system changes markup.

**Assigned report path:** `reports/wave1/ref-board-theme.md`
**Branch:** `swarm/ref-board-theme`
**Audit source:** `scripts/swarm/tasks/hyg-public-board.md`

---

## 1. Headline result

The mechanical colour/role → theme-token mapping that this task asks for is
**already complete** in the current tree. Every inline colour/role value in
`public_board.py` already resolves through `var(--token)` (0 raw hex/rgb
colours across 125 inline `style="..."` attributes). `findings_content.py`
contains **no** HTML/CSS at all (it is a pure content model). The only raw CSS
values in the module live in the deliberate `_PAGE_CHROME` board-theme
`<style>` block, whose token values deliberately diverge from `theme.py` and
are pinned by a test.

**No source edit was made.** Every candidate follow-on change was evaluated
against the binding "preserve rendered output byte-for-byte where possible"
constraint and the rubric dimension-6 ("zero one-off inline styles for things
the theme already covers") criterion; each is either already satisfied or would
alter rendered output, so none was taken. Details and the dedup map are below.

All four quality gates pass on the unmodified tree (measured this session —
commands in §6).

---

## 2. Method (measured this session)

I scanned `public_board.py` for every `style="…"` / `style='…'` attribute and
categorised the values:

- `grep -nE '#[0-9a-fA-F]{3,6}' src/nfl_ats/public_board.py` → all 51 raw-hex
  hits fall inside the `_PAGE_CHROME` token-definition block (lines 226–259)
  plus HTML entities (`&#183;`, `&#8217;`, …), **none** in a rendered element's
  inline style.
- A Python AST-free regex over the whole file extracted **125** inline style
  attributes; a colour scan (`#[0-9a-fA-F]{3,8}`, `rgb(a?()`, named colours)
  found **0 raw colours** — the single "named" hit is `white-space:nowrap;`
  (the word "white" inside `white-space`, not a colour).
- `findings_content.py`: `grep -nE '<(div|span|table|…|style)'` returns only a
  docstring mention of `<p>` (line 344); no rendered markup, no CSS.

---

## 3. Findings on `public_board.py`

### 3.1 Inline colour/role styling — already tokenised (mechanical mapping done)

Every colour or role value in an inline `style="…"` attribute already uses a
`var(--token)` reference into the shared token set. Representative samples
(measured, file `src/nfl_ats/public_board.py`):

- `372` `'<div style="margin-top:36px;padding-top:14px;border-top:1px solid var(--grid);">'`
- `777` `'<p class="fine" style="color:var(--muted);">Attribution not published.</p>'`
- `1196` `'<div class="card" style="border-left:3px solid var(--warning);margin-top:14px;">'`
- `1755` `'<span class="dot" style="background:var(--muted);"></span>'`
- `1881` `f'<span style="color:var(--ink-2);">{escape(finding.source)}</span></p>'`
- `1956/1968` `background:var(--series-model); … border:2px solid var(--surface);`
- `3896/3906/3908` `background:var(--grid)` / `var(--ink-2)` in the meter bars.

Tokens referenced in inline styles: `--grid`, `--baseline`, `--surface`,
`--series-model`, `--series-market`, `--muted`, `--ink-2`, `--warning`,
`--good-text`. All are defined by `theme.stylesheet()` (read:
`src/nfl_ats/dashboard/theme.py` `TOKENS_LIGHT`/`TOKENS_DARK`), so the
component-render path already satisfies "use theme tokens, not raw colour."

**Conclusion (measured):** the colour/role portion of this task's mechanical
mapping is already 100% applied. Nothing to replace here.

### 3.2 The only raw CSS is the deliberate board-theme override (`_PAGE_CHROME`)

A single `<style>` block, `_PAGE_CHROME` (lines 213–372, defined in the
module source), rides *after* `theme.stylesheet()` so its token remaps win on
equal specificity. It re-declares 15 role tokens. Comparing values against
`theme.py` (read):

| token | theme.py light | `_PAGE_CHROME` light | match? |
|---|---|---|---|
| surface | `#fcfcfb` | `#ffffff` | no |
| plane | `#f9f9f7` | `#fafaf8` | no |
| ink | `#0b0b0b` | `#111110` | no |
| ink-2 | `#52514e` | `#4b4b47` | no |
| muted | `#898781` | `#8a8a84` | no |
| grid | `#e1e0d9` | `rgba(0,0,0,0.08)` | no |
| baseline | `#c3c2b7` | `rgba(0,0,0,0.16)` | no |
| border | `rgba(11,11,11,0.10)` | `rgba(0,0,0,0.08)` | no |
| series-model | `#2a78d6` | `#2a78d6` | **yes** |
| series-market | `#eb6834` | `#4b4b47` | no |
| series-third | `#1baf7a` | `#8a8a84` | no |
| good-text | `#006300` | `#1a7f37` | no |
| good | `#0ca30c` | `#1a7f37` | no |
| critical | `#d03b3b` | `#c0392b` | no |
| serious | `#ec835a` | `#b35900` | no |

In dark mode **0 of 15** match (e.g. series-model `#3987e5` vs `#6ea8dc`). So
the board intentionally ships a *different* palette than the dashboard — this
is the documented "Ledger base + Terminal layout" design system (git log:
`3df4093 Dashboard redo: Ledger-Terminal hybrid from design research`).

**Why this is not a mechanical mapping to `theme.py`:** it is a deliberate
design override, not a duplication. Forcing `theme.py` values here would change
rendered colour on 14 of 15 tokens in light (15 of 15 in dark) — i.e. it would
violate "preserve rendered output byte-for-byte." It is also load-bearing:
`tests/test_public_board.py:252` asserts `"--series-model: #2a78d6;" in page`,
and the block carries a commented "hex budget <= 10 in the light block"
binding rule. **Left intact.**

### 3.3 Repeated inline patterns — dedup map (not taken)

The 125 inline attributes normalise to 74 unique strings. The repeated ones are
**spacing/typography only** (no colour), with counts and line numbers
(measured):

| inline style | count | lines |
|---|---|---|
| `margin-top:8px;` | 10 | 727, 733, 754, 824, 898, 1079, 1625, 1665, 1669, 2019 |
| `margin-top:6px;` | 9 | 984, 997, 999, 1014, 1015, 1032, 1393, 1659, 2868 |
| `margin-top:10px;` | 6 | 874, 888, 1101, 1388, 2587, 4003 |
| `margin-bottom:8px;` | 6 | 2000, 2026, 2111, 2851, 2941, 3428 |
| `margin:0 0 8px;` | 4 | 825, 840, 1348, 1679 |
| `font-weight:600;` | 4 | 982, 994, 1011, 1035 |
| `overflow-x:auto;` | 4 | 2273, 3960, 3995, 4060 |
| `margin-top:16px;` | 3 | 568, 761, 2266 |
| `margin-bottom:6px;` | 3 | 1744, 1783, 2025 |
| `margin-bottom:10px;` | 3 | 2028, 2331, 2568 |
| `margin-top:8px;padding-top:8px;border-top:1px solid var(--grid);` | 3 | 2860, 2876, 2881 |
| `margin:0;padding-left:18px;` | 2 | 759, 1334 |
| `margin-top:12px;` | 2 | 1102, 2548 |
| `font-size:17px;margin:0 0 2px;` | 2 | 1347, 1678 |
| `font-size:14px;margin-top:6px;` | 2 | 1377, 1383 |
| `margin-bottom:14px;` | 2 | 1733, 1794 |
| `border-collapse:collapse;width:100%;font-size:13px;` | 2 | 3961, 3996 |
| `margin-left:6px;` | 2 | 4055, 4057 |

**Assessment against mechanical mapping + byte-for-byte discipline:**

- These are spacing/size values (`6/8/10/12/14/16/18px`). The token system
  defines **colour/role tokens and type-scale *classes*** (`.kicker .title
  .sub .hero .prose .fine .num`), but **no spacing utility tokens** and no
  `2px`-grid spacing scale (the chrome comment mandates a `4px` grid, yet the
  inline set uses `6/10/14px`, which are off-grid). There is therefore **no
  theme token to map them to mechanically** — a prerequisite the task scopes
  to ("where the mapping is mechanical").
- Each inline string is checked against the existing theme/chrome classes:
  - `font-size:17px;margin:0 0 2px;` is *subset* of `.title` (which also adds
    `font-weight:650`) → mapping to `.title` would **change weight** → not
    equivalent.
  - `border-collapse:collapse;width:100%;font-size:13px;` (lines 3961, 3996)
    matches `.ats table.data`'s *table-level* rule **exactly**, **but** those
    two `<table>` elements do **not** carry `class="data"`, so promoting them
    would also pull in `.ats table.data th/td` borders, padding and
    `tabular-nums` that are absent today → **changes rendered output** (a
    deviation, not a byte-for-byte preservation).
- Converting any of these to a new utility class would (a) change markup and
  (b) require inventing tokens not present in `theme.py`, both outside the
  "mechanical, byte-for-byte" envelope. **Not taken.**

This is the dedup map the audit (`hyg-public-board.md`) asked for; the
execution decision is to record it and **not** apply it, because the only
mechanical target (colour) is already done and the spacing remainder has no
token to map to without altering output.

---

## 4. Findings on `findings_content.py`

`src/nfl_ats/dashboard/findings_content.py` (read, 1,805 lines) is a dataclass
+ string-content module (`Finding`, `HeadlineTile`, `VerdictGroup`, `HERO_*`,
`FINDINGS`, `ladder_rungs`, …). It imports **no** web framework and emits **no**
HTML/CSS. The only markup-shaped text is a docstring note that
`public_board` wraps `ladder_rungs` output in `<p>` tags (line 344). The prior
audit's concern — "content strings that should live in data files" — is a
*structuring* question, not an inline-CSS question, and is out of scope for the
mechanical token-mapping asked here. **No change required or made.**

---

## 5. Rubric dimension 6 (visual consistency) flag

Per `docs/uiux_rubric.md` dim 6 — *"One theme token set (spacing/type/colour
scales), zero one-off inline styles for things the theme already covers."*

- **Colour/role:** satisfied. Inline styles reference the shared token set
  (`var(--…)`); no raw colours.
- **Per-surface token set:** the public board deliberately uses its own
  `_PAGE_CHROME` override rather than `theme.py`'s palette (§3.2). This is a
  *different* token set per surface (public site vs internal dashboard), chosen
  by design research, not drift. **Not blocking** — it is an intentional brand
  decision, and the board is a distinct surface. Flagged for the record.
- **One-off inline styles for theme-covered things:** the two `border-collapse`
  tables (§3.3) are the closest thing, but mapping them to `.data` changes
  th/td rendering, so it is a *deviation*, not a free win. Left as-is to honour
  byte-for-byte; recorded as a future, explicitly-declared dedup if a
  `table.data`-equivalent utility (with no th/td side-effects) is added.

No dim-6 regression is introduced by this task (no edits made).

---

## 6. Quality gates (measured this session)

Run from the worktree root with the main venv; `PYTHONPATH` set to the worktree
`src` for pytest, basetemp outside the repo:

| gate | command | result |
|---|---|---|
| ruff format | `ruff format --check .` | **636 files already formatted** |
| ruff check | `ruff check .` | **All checks passed!** |
| mypy | `mypy src` | **Success: no issues found in 105 source files** |
| pytest (site/snapshot) | `pytest tests/test_public_board.py tests/test_site_theme_invariants.py tests/test_site_theme_pack.py` | **156 passed** |
| pytest (full suite) | `pytest` (basetemp outside repo) | **1855 passed, 5 skipped** (skips are pre-existing: missing local nflverse/PBP/feature snapshots) |

The 5 skips are environment data-absence skips, present on a clean checkout
independent of this task (reported; unverified whether they pass where data
exists).

---

## 7. Deviations taken

**None.** No markup or CSS was altered. Every candidate change either was
already satisfied (inline colours → tokens) or would have changed rendered
output (board-theme palette, `border-collapse` → `.data` th/td styling,
type-scale classes adding weight), which the task's "preserve rendered output
byte-for-byte where possible" directive forbids. The dedup opportunities are
catalogued in §3.3 for a future, explicitly-declared refactor — not applied
here.

---

## 8. Source changes

None. Only this report is added/changed. Commit message records the
verification outcome and the empty diff on the two target modules.

---

## 9. Provenance tags (per binding rule 2)

- **measured** — the grep/regex scans, the token-value table (read from
  `theme.py` + `public_board.py`), the 125/74/0 counts, and all four gate
  runs were executed this session; commands and line numbers given above.
- **read** — `src/nfl_ats/dashboard/theme.py` (token tables),
  `src/nfl_ats/public_board.py` (lines 213–372 `_PAGE_CHROME`, 372/777/1196/
  1755/1881/1956/1968/2860/3961/3996 et al.), `docs/uiux_rubric.md` (dim 6),
  `tests/test_public_board.py:252`.
- **reported** — the audit's "content strings should live in data files"
  concern is quoted from `scripts/swarm/tasks/hyg-public-board.md` and not
  re-verified beyond reading the file (unverified as a separate finding).
- **inferred** — the interpretation that the mechanical colour→token mapping
  was completed by the prior consolidation commits (`3df4093`,
  `0f9b1dd`, `ae74f7e`) is my read of git history, not a fresh measurement of
  those commits' diffs.
