# why-this-pick-flipback

## Goal
The board's "Why this pick" text must name the side the card serves. Done when docs/index.html shows ATL@TB, IND@JAX and ARI@DAL explanations naming ATL, IND and ARI, and every other row still matches its pick.

## State (final, 2026-10-31)
- Cause (read): src/nfl_ats/board_content.py row loop compared the served team to the row's own team, not to the side the stored explanation argues. After Friday's flip back (refresh_last_call_fri_1330) the two matched, so the Wednesday-side text from explanations.json (written 12:18 Fri, before the 13:37 flip back) showed unprefixed.
- Fix: _load_pick_explanation_sides reads each explanation's model_probability.pick_side, passed as model_team. The prefix now reads "<TEAM> is the card's pick after the latest line and injury check. The model's own read: ..." (the old "when picks locked" wording was false before the lock).
- Verified: publish-board, all Week 8 rows open with the served team or the prefix; ruff format/check clean on the file; mypy src clean; pytest board/explanation/card subset 387 passed.

## Tried
- Regenerating the explanation for the served side inside refresh needs lineage and overlays; not done.

## Next
Optional: have refresh-picks write an explanation for the new side on every recorded flip so the prefix is rarely needed.

## Open
