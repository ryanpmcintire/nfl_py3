# Projected lineups on This Week

The This Week game deep dive includes an optional **Projected lineups & model
impact** panel. It is a static view designed for GitHub Pages: a refresh job
builds `artifacts/lineups/current/lineups.json`, and the renderer only reads
that artifact. The artifact is ignored by Git along with the other local data
and model outputs. The lineup file is REPLACED on every refresh, never
accumulated: the builder overwrites the stable path atomically and removes
superseded stamped `*/lineups.json` runs. History lives in the depth-chart
snapshot the payload cites (`depth_snapshot`), not in display copies.

The scheduled refresh runs Monday through Sunday at noon Eastern. It snapshots
the current depth chart, rebuilds the player features and weekly forecast from
that snapshot, then creates the final lineup artifact linked to the new
forecast. Run the same flow locally with:

```powershell
.\.tools\uv.exe run --no-sync python scripts\refresh_lineup_forecast.py
```

The panel shows the complete latest depth-chart snapshot — offense, defense,
and special teams in their own sections with per-section player counts —
along with source timestamps, injury-feed availability, and the active model's
scored QB family contribution when the waterfall contains it. Unit filter
buttons above the panel toggle each section's visibility for the current
reader only; every player row stays in the published HTML. The renderer
refuses to publish the panel beside a forecast whose projected QB is absent
from that lineup snapshot.

Play probability is only shown when it is present in the model artifact. A
missing injury feed or probability is displayed as unavailable rather than
estimated. Player rows outside the scored model family are labeled context-only
so readers can distinguish lineup context from a model input.

## UI-20 (2026-09-05): every listed player, not only the QB

Before this change, `play_probability` was only ever set for the one QB the
active model consumed (`model_role: "base_model"`); every other listed
starter had `model_role: "context_only"` and no probability at all -- the
owner's complaint that prompted this section.

Every player with a `gsis_id` now carries a `play_probability`, and a new
`probability_source` field says exactly how it was produced:

- **`base_model_qb`** -- unchanged, bit-identical to the pre-existing
  behaviour: the forecast's own `{side}_qb_start_probability` for the QB the
  active model actually scored. Never recomputed by this feature.
- **`availability_model`** -- every other player, from the SAME learned or
  fixed availability rule `nfl_ats.availability.resolve_unavailability`
  already uses for the QB, applied per-player instead of only to one:
  - if the player has a visible injury-report row for this week (observed
    strictly before the artifact's own `generated_at`, using the same
    leakage-safe `week_proxy` fallback ENG-39 built for the historical
    feature table), the rule is applied to their own report/practice status;
  - otherwise, the position's historical **no-designation base rate**
    (`nfl_ats.lineup_availability`) -- the empirical rate at which an
    eligible, active-roster player who carries no injury designation at all
    still does not play that week (a healthy scratch, a coach's decision,
    and so on), conditioned on whether that same player recently played at
    all (`recent_role`), because a bare position average would materially
    understate a starter's real probability (measured: it pooled a WR1 who
    plays nearly every healthy week with a WR5 who rarely dresses). This is
    a genuinely new quantity, not read from any production table, derived
    entirely from the local player snapshot -- no network fetch of its own.
- **`unavailable`** -- reserved for a depth-chart row this machinery
  genuinely cannot score: no `gsis_id`, or no rate at all could be produced.
  `play_probability` stays `None` and the panel still shows an em dash --
  never an invented number.

Every player also carries a short `probability_reason` naming exactly which
rule fired; the rendered panel exposes it as a tooltip on the percentage, and
prints a one-line legend once per lineup block. The lineup-aware assistant
(`nfl_ats.board_assistant_lineups.player_availability_answer`) answers an
"is `<player>` playing" question for any listed player now, not only the
model's QB, and names the source in its answer text.

**Known limitation, measured, not this feature's fault:** the active
model's own `{side}_qb_start_probability` is currently a dead constant
`1.0` for every 2026 game (verified against the live
`artifacts/lineups/current/lineups.json` and the weekly forecast) because
the production feature table has no 2025/2026 injury coverage at all --
the pre-existing ENG-39 defect (`docs/injury_timestamp_fallback.md`).
Fixing that is a separate, owner-scheduled production rebuild; this
feature's `availability_model` numbers for every other player are
unaffected and do show real variation, since they are computed fresh at
build time rather than read from that stale feature table.
